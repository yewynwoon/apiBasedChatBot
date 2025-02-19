from flask import Flask, request, jsonify
import pickle
import logging
import os
from pathlib import Path
import openai
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import QueryEngineTool
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentQASystem:
    def __init__(self, index_path: str = "./index/index.pkl"):
        self.index_path = Path(index_path)
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
            
        openai.api_key = openai_api_key
        self.index = self._load_index()
        self.agent = self._setup_agent()

    def _load_index(self) -> VectorStoreIndex:
        try:
            with open(self.index_path, 'rb') as f:
                index = pickle.load(f)
            logger.info("Successfully loaded index")
            return index
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise

    def _setup_agent(self):
        try:
            llm = OpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.1
            )
            
            Settings.llm = llm
            Settings.context_window = 4096
            Settings.num_output = 512
            
            query_engine = self.index.as_query_engine(
                similarity_top_k=5,
                similarity_threshold=0.8,
                response_mode="refine"
            )
            
            query_engine_tool = QueryEngineTool.from_defaults(
                query_engine=query_engine,
                name="minecraft_knowledge",
                description="Use this tool to search through Minecraft documentation and knowledge"
            )
            
            tavily_tool_spec = TavilyToolSpec(
                api_key=os.getenv("TAVILY_API_KEY")
            )
            
            tools = [query_engine_tool] + tavily_tool_spec.to_tool_list()
            agent = OpenAIAgent.from_tools(
                tools,
                llm=Settings.llm,
                system_prompt=(
                    "You are a helpful assistant specialized in Minecraft. "
                    "When answering questions, prioritize checking the minecraft_knowledge tool. "
                    "Provide clear, concise answers without special formatting. "
                    "Write naturally as if having a conversation."
                    "Only use the web search tool as a last resort and if you can't find any relevant information from the index."
                )
            )
            
            return agent
            
        except Exception as e:
            logger.error(f"Failed to setup agent: {str(e)}")
            raise

    def query(self, question: str) -> dict:
        try:
            response = self.agent.chat(question)
            
            return {
                'answer': str(response),
                'metadata': {
                    'response_type': 'agent_response'
                }
            }
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise

def create_app(config=None):
    app = Flask(__name__)
    
    try:
        qa_system = AgentQASystem(
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