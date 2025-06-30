import os
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

def document_input() -> str:
    """
    This function is used to input documents for the RAG system.
    It loads the documents from the specified paths and returns them as a list of chunks.
    """
    inp = input("Enter the path to the document (PDF or TXT): ")
    if not inp:
        logging.error("No document path provided. Please provide a valid path.")
        raise ValueError("No document path provided. Please provide a valid path.")
    logging.info(f"Loading document from path ::: {inp}")
    return inp