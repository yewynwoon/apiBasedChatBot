from flask import Flask, request, jsonify
import pickle
import logging
import os
from pathlib import Path
from typing import Optional
import openai
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QASystem:
    def __init__(self, index_path: str = "./index/index.pkl"):
        self.index_path = Path(index_path)
        
        openai_api_key_from_env = os.getenv("OPENAI_API_KEY")
        if openai_api_key_from_env:
            openai.api_key = openai_api_key_from_env
        else:
            raise ValueError("OpenAI API key is required")

        self.index = self._load_index()
        self.query_engine = self._setup_query_engine()

    def _load_index(self) -> VectorStoreIndex:
        try:
            with open(self.index_path, 'rb') as f:
                index = pickle.load(f)
            logger.info("Successfully loaded index")
            return index
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise

    def _setup_query_engine(self):
        try:
            llm = OpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.1
            )
        
            Settings.llm = llm
            Settings.context_window = 4096
            Settings.num_output = 512
            Settings.system_prompt = {"You are a helpful assistant specialized in Minecraft. "
                    "Provide clear, concise answers without using any special formatting or symbols. "
                "Do not include links, markdown, or bullet points. "
                "Present information in a natural, flowing way using plain text. "
                "Focus on essential information and avoid unnecessary structure or lists. "
                "Write as if you are having a natural conversation."}


            
            query_engine = self.index.as_query_engine(
                similarity_top_k=3,
                response_mode="compact"
            )
            return query_engine
            
        except Exception as e:
            logger.error(f"Failed to setup query engine: {str(e)}")
            raise

    def query(self, question: str) -> dict:
        try:
            # First, check local sources (document retrieval)
            response = self.query_engine.query(question)
            source_nodes = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source_nodes.append({
                        'text': node.node.text,
                        'score': float(node.score) if node.score else None,
                        'document': node.node.metadata.get('file_name', 'Unknown')
                    })
            
            # Enhanced conditions for resorting to web search:
            should_search_web = False
            
            # 1. Empty response check
            if not str(response).strip():
                should_search_web = True
                logger.info("Empty response, resorting to web search")
            
            # 2. Response quality checks
            elif source_nodes:
                # Calculate average relevance score
                valid_scores = [node['score'] for node in source_nodes if node['score'] is not None]
                avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
                
                # Check various quality indicators
                low_quality_conditions = [
                    avg_score < 0.3,  # Average relevance too low
                    all(node['score'] < 0.35 for node in source_nodes if node['score'] is not None),  # All scores too low
                    len(str(response).split()) < 20,  # Response too short
                    len(source_nodes) < 2  # Too few source nodes
                ]
                
                if any(low_quality_conditions):
                    should_search_web = True
                    logger.info(f"Low quality response (avg_score: {avg_score:.2f}), resorting to web search")
            
            if should_search_web:
                return self._search_web(question)
            else:
                return {
                    'answer': str(response),
                    'source_nodes': source_nodes,
                    'metadata': {
                        'total_nodes': len(source_nodes),
                        'response_type': type(response).__name__
                    }
                }
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise

    
    def _search_web(self, question: str) -> dict:
        try:
            tavily_tool_spec = TavilyToolSpec(
                api_key=os.getenv("TAVILY_API_KEY")
            )
                
            agent = OpenAIAgent.from_tools(
                tavily_tool_spec.to_tool_list(),
                llm=Settings.llm,
                system_prompt=Settings.system_prompt
            )
            
            # Get response from agent
            minecraft_question = f"Minecraft: {question}"
            response = agent.chat(minecraft_question)
                
            return {
                "answer": str(response)
            }
                
        except Exception as e:
            logger.error(f"Failed to search using Tavily: {str(e)}")
            return {"error": f"Failed to retrieve search results: {str(e)}"}
def create_app(config=None):
    app = Flask(__name__)
    
    try:
        qa_system = QASystem(
            index_path="./index/index.pkl"
        )
        logger.info("API has been started")
    except Exception as e:
        logger.error(f"Failed to start API service: {str(e)}")
        raise

    @app.route('/query', methods=['POST'])
    def query():
        try:
            data = request.get_json()
            
            if not data or 'question' not in data:
                return jsonify({'error': 'Missing question in request'}), 400
                
            question = data['question']
            response = qa_system.query(question)
            
            return jsonify(response)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)