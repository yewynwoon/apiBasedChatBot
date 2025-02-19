import os
import pickle
import logging
from pathlib import Path
from typing import Optional

import openai
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentIngester:
    def __init__(
        self,
        data_dir: str = "./data",
        index_dir: str = "./index",
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        
        self.data_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)

        if openai_api_key:
            openai.api_key = os.environ["OPENAI_API_KEY"] = openai_api_key
        elif "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError("OpenAI API key is required for embeddings")

        self._initialize_settings()

    def _initialize_settings(self):
        try:
            embed_model = OpenAIEmbedding(
                model=self.embedding_model,
                dimensions=1536  # dimensions for text-embedding-3-small
            )
            Settings.embed_model = embed_model
            logger.info(f"Initialized OpenAI embedding model: {self.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise

    def ingest_documents(self) -> Optional[VectorStoreIndex]:
        try:
            if not any(self.data_dir.iterdir()):
                raise ValueError(f"No documents found in {self.data_dir}")

            logger.info(f"Loading documents from {self.data_dir}")
            documents = SimpleDirectoryReader(
                str(self.data_dir),
                recursive=True,
                filename_as_id=True
            ).load_data()
            
            logger.info(f"Loaded {len(documents)} documents")

            node_parser = SimpleNodeParser.from_defaults(
                chunk_size=1024,
                chunk_overlap=200
            )
            
            index = VectorStoreIndex.from_documents(
                documents,
                node_parser=node_parser,
                show_progress=True
            )

            return index

        except Exception as e:
            logger.error(f"Failed to ingest documents: {str(e)}")
            raise

    def save_index(self, index: VectorStoreIndex, filename: str = "index.pkl"):
        try:
            output_path = self.index_dir / filename
            with open(output_path, 'wb') as f:
                pickle.dump(index, f)
            logger.info(f"Index saved successfully to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise

def main():
    ingester = DocumentIngester(
        data_dir="./data",
        index_dir="./index",
        openai_api_key=os.getenv("OPENAI_API_KEY")  
    )

    try:
        index = ingester.ingest_documents()
        ingester.save_index(index)
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()