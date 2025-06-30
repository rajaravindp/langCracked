import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader 
from langchain.text_splitter import CharacterTextSplitter
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

def load_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list:
    """
    Load a document and split it into chunks.
    
    Args:
        file_path (str): Path to the document file.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.
        
    Returns:
        list: List of document chunks.
    """
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        logging.info(f"Loading PDF document from {file_path}")
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        logging.info(f"Loading text document from {file_path}")
    else:
        logging.error("Unsupported file type. Only .pdf and .txt files are supported.")
        raise ValueError("Unsupported file type. Only .pdf and .txt files are supported.")
    
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} documents from {file_path}")
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    if text_splitter is None:
        logging.error("Failed to initialize text splitter.")
        raise RuntimeError("Text splitter initialization failed.")
    logging.info("Chunking successful.")
    return text_splitter.split_documents(documents)