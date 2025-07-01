from langchain_community.document_loaders import CSVLoader

def get_documents(document_path: str) -> list:
    """
    Reads a text file and returns its content as a list of strings, where each string is a line from the file.

    Args:
        document_path (str): The path to the text file.

    Returns:
        list: A list of strings, each representing a line from the file.
    """
    loader = CSVLoader(file_path=document_path)
    documents = loader.load()
    return documents
