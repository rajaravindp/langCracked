from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage

def model_call(query: str, relevant_documents: list) -> str:
    """Call the LLM with the query and relevant documents.""" 
    llm_inp = (
        f"""""
        Please answer my query based on the relevant documents.
        Here are the relevant documents : {relevant_documents}.
        Here is my query : {query}. 
        Provide an answer based on the documents and the query. If the answer is not in the documents, say to the effect of "I don't know".
        """
    )

    llm = ChatBedrockConverse(model="amazon.nova-lite-v1:0")
    messages = [
        SystemMessage(content="You are helping a user with their query based on the provided documents."),
        HumanMessage(content=llm_inp)
    ]
    res = llm.invoke(messages)
    return res.content