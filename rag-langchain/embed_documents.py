import os
import logging
from load_documents import load_document
from create_embeddings import create_embeddings
from document_input import document_input

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/rag-logs.log"),
        logging.StreamHandler()
    ]
)

curr_dir = os.path.dirname(os.path.abspath(__file__))
vector_dir = os.path.join(curr_dir, "db", "Chroma")

def embed_documents():
    # Create necessary directories
    if not os.path.exists("log"):
        os.makedirs("log")
        print("Created log directory")
    
    if not os.path.exists(vector_dir):
        print("Vector directory does not exist, creating it...")
        os.makedirs(vector_dir, exist_ok=True)
    
    try:
        # Get document paths from user input
        document_file_path = document_input()
        
        # Load the documents
        print("Loading and chunking document...")
        chunks = load_document(document_file_path)
        print(f"Number of document chunks: {len(chunks)}")
        print(f"Sample chunk:\n{chunks[0].page_content[:200]}...\n")
        
        # Create embeddings and store them in the vector database
        print("Creating embeddings and storing in vector database...")
        create_embeddings(chunks, vector_dir)
        print(f"Successfully created vector database and persisted data at {vector_dir}")
        
        # Verify the database was created
        if os.path.exists(vector_dir) and os.listdir(vector_dir):
            print("✓ Vector database files created successfully")
            print(f"Files in vector directory: {os.listdir(vector_dir)}")
        else:
            print("⚠ Warning: Vector database directory appears to be empty")
            
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    embed_documents()
    print("Embedding process completed successfully.")
    print("You can now run the RAG application.")