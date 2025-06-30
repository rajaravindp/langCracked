import os
from dotenv import load_dotenv
import chromadb
from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from model_call import model_call

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "Chroma")

embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
    )
# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Check if the database is loaded correctly
print(f"Number of documents in DB: {db._collection.count()}")

def main():
    # Get user input for the query
    print("Welcome!")
    while True:
        query = input("Enter your question. Or 'qq' to quit ::: ")
        if query.lower() == "qq":
            print("Exiting the program.")
            break
        # Retrieve relevant documents based on the query
        retriever = db.as_retriever(
            search_type="similarity",
        )
        # retriever = db.as_retriever(
        #     search_type="similarity_score_threshold",
        #     search_kwargs={"k": 10, "score_threshold": 0.9}
        # )
        # retriever = db.as_retriever(
        #     search_type="mmr",
        #     search_kwargs={'k': 5, 'fetch_k': 50}
        # )
        relevant_docs = retriever.invoke(query)

        # Display the relevant results with metadata
        print("\n--- Relevant Documents ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

        # Call the model with the query and relevant documents
        relevant_documents = [doc.page_conetnt for doc in relevant_docs]
        response = model_call(query, relevant_documents)
        return response

if __name__ == "__main__":
    main()