from reranker import reranker
from prepare_documents_for_rereanking import prepare_documents
from get_documents_csvloader import get_documents

def main():
    while True:
        # query = input("Enter your query: ")
        query = "Give me some reviews about sauces."
        if query.lower() == 'qq' or query.lower() == 'exit':
            print("Exiting the program.")
            break
        docs = get_documents(document_path="data/Reviews.csv")
        document_sources = prepare_documents(docs=docs)
        response = reranker(text_query=query, document_sources=document_sources, num_results=5)
        return "RESPONSE ::: ",  response
    
        ### Pass response - reranked results to any LLM for better response generation
    
if __name__ == "__main__":
    main()