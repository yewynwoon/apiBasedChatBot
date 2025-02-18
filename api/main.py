from flask import Flask, request, jsonify
import pickle
import logging
import os
from pathlib import Path
from typing import Optional
import openai
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QASystem:
    def __init__(
        self, 
        index_path: str = "./index/index.pkl",
        openai_api_key: Optional[str] = None
    ):
        self.index_path = Path(index_path)
        
        if openai_api_key:
            openai.api_key = openai_api_key
        elif "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OpenAI API key is required")
        
        self.index = self._load_index()
        self.query_engine = self._setup_query_engine()

    def _load_index(self) -> VectorStoreIndex:
        """Load the saved index from pickle file."""
        try:
            with open(self.index_path, 'rb') as f:
                index = pickle.load(f)
            logger.info("Successfully loaded index")
            return index
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise

    def _setup_query_engine(self):
        """Set up the query engine with custom configuration."""
        try:
            # Initialize LLM
            llm = OpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.1
            )
            
            # Configure settings
            Settings.llm = llm
            Settings.context_window = 4096
            Settings.num_output = 512
            Settings.system_prompt = (
                "You are a helpful AI assistant that answers questions based on the "
                "provided document context. Always provide clear, concise responses "
                "and cite specific parts of the document when possible."
            )
            
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=3,  # Number of relevant chunks to consider
                response_mode="compact"  # For concise responses
            )
            
            return query_engine
            
        except Exception as e:
            logger.error(f"Failed to setup query engine: {str(e)}")
            raise

    def query(self, question: str) -> dict:
        """
        Process a question and return the answer with metadata.
        
        Args:
            question: The question to answer
            
        Returns:
            dict: Contains answer, source nodes, and metadata
        """
        try:
            response = self.query_engine.query(question)
            
            # Extract source nodes information
            source_nodes = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source_nodes.append({
                        'text': node.node.text,
                        'score': float(node.score) if node.score else None,
                        'document': node.node.metadata.get('file_name', 'Unknown')
                    })
            
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

def create_app(config=None):
    app = Flask(__name__)
    
    try:
        qa_system = QASystem(
            index_path="./index/index.pkl",
            openai_api_key="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"  # Replace with your API key
        )
        logger.info("Chat Bot has been started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Chat Bot {str(e)}")
        raise


    @app.route('/query', methods=['POST'])
    def query():
        """Handle query requests."""
        try:
            data = request.get_json()
            
            if not data or 'question' not in data:
                return jsonify({
                    'error': 'Missing question in request'
                }), 400
                
            question = data['question']
            
            # Get response from QA system
            response = qa_system.query(question)
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return jsonify({
                'error': str(e)
            }), 500

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)