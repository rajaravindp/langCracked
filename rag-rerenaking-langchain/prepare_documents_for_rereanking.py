def prepare_documents(docs: list) -> list:
    """
    Prepares documents for re-ranking by extracting the 'text' field from each document.

    Args:
        docs (list): A list of documents, where each document is a dictionary containing a 'text' field.

    Returns:
        list: A list of strings, each representing the 'text' field from the input documents.
    """
    docx = list()
    for doc in docs:
        docx.append({
        "type": "INLINE",
        "inlineDocumentSource": {
            "type": "TEXT",
            "textDocument": {
                "text": doc.page_content,
            }
        }
    })
    return docx