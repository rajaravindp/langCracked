import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from transform_document import transform_document
from visualize_graph import visualize_graph

load_dotenv()

def main():
    """Main function to run the knowledge graph construction."""
    # Initialize the Neo4j graph connection
    # Ensure that the Neo4j credentials are set in the .env file
    graph = Neo4jGraph()

    # Load the document text from a file
    # Ensure that the document.txt file exists in the data directory
    document_path = os.path.join(os.getcwd(), "data", "document.txt")
    with open(document_path, "r", encoding="utf-8") as f:
        document_text = f.read()

    # Transform the document text into nodes and edges    
    graph_documents  = transform_document(document_text)
    print(f"Nodes:{graph_documents[0].nodes}")
    print(f"Relationships:{graph_documents[0].relationships}")
    
    # Uncomment the line below to visualize the graph
    # This will display the graph using NetworkX and Matplotlib
    # visualize_graph(graph_documents[0])

    # Add the graph documents to the Neo4j graph
    graph.add_graph_documents(graph_documents)


if __name__ == "__main__":
    main()
