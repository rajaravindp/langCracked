import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

load_dotenv()

# pip install langchain-aws

def chat_with_bedrock(prompt: str) -> str:
    """Sends a chat message to the Bedrock model and returns the response."""
    bedrock = ChatBedrockConverse(model="amazon.nova-lite-v1:0", temperature=0.7, max_tokens=100)
    response = bedrock.invoke(prompt)
    return response.content

def messages():
    """Creates a list of messages for the chat prompt."""
    messages = [
        ("system", "You are a helpful translator. Translate the user sentence to Hindi."),
        ("human", "I live in Dallas."),
    ]

    return messages

def main():
    """Main function to run the chat with Bedrock."""
    prompt = messages()
    response = chat_with_bedrock(prompt)
    print(response)

if __name__ == "__main__":
    main()