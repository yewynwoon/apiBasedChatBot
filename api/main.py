from flask import Flask, request, jsonify
import pickle
import logging
import os
from pathlib import Path
from typing import Optional
import openai
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from serpapi.google_search import GoogleSearch

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
            # Settings.system_prompt = (
            #     "You are a helpful AI assistant that answers questions based on the "
            #     "provided document context. Always provide clear, concise responses "
            #     "and cite specific parts of the document when possible."
            # )
            
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
            
            # 1. Check if the response is empty
            # 2. Check if the response quality is too low
            if not str(response).strip() or (source_nodes and all(node['score'] < 0.5 for node in source_nodes)):
                # If no relevant answer is found, search the web using SerpAPI
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
            search_params = {
                "q": question + " Minecraft",  # Append "Minecraft" to focus search
                "hl": "en",
                "google_domain": "google.com",
                "api_key": os.getenv("SERPAPI_API_KEY")
            }
            
            search = GoogleSearch(search_params)
            search_results = search.get_dict()
            
            top_result = search_results.get('organic_results', [])[0]
            logger.info(top_result)
            
            return {
                "answer": top_result['snippet'],
                "source_url": top_result['link'],
                "metadata": {"source_title": top_result['title']}
            }
        
        except Exception as e:
            logger.error(f"Failed to search the web: {str(e)}")
            return {"error": "Failed to retrieve web search results."}

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