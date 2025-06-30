import os
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/rag-logs.log"),
        logging.StreamHandler()
    ]
)

def create_embeddings(chunks: list, vector_dir: str) -> Chroma:
    """
    Create embeddings for the document chunks and store them in a vector database.

    Args:
        chunks (list): List of document chunks.
        vector_dir (str): Directory to store the vector database.
    
    Returns:
        Chroma: The vector database with embeddings.
    """
    # Initialize Bedrock embeddings
    bedrock_embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
    )
    if not bedrock_embeddings:
        logging.error("Failed to initialize Bedrock embeddings.")
        raise RuntimeError("Bedrock embeddings initialization failed.")
    logging.info("Bedrock embeddings initialized successfully.")
    # Create a Chroma vector database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=bedrock_embeddings,
        persist_directory=vector_dir
    )
    if not vector_db:
        logging.error("Failed to instantiate vector database and load documents.")
        raise RuntimeError("Vector database creation failed.")
    logging.info("Vector database created and documents loaded successfully.")

    return vector_db