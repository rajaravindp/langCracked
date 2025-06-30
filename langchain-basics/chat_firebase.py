import os
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

"""
1. Create a Firebase account
2. Create a new Firebase project and FireStore Database
3. Retrieve the Project ID
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate the Google Cloud CLI with your Google account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to the new Firebase project you created
5. pip install langchain-google-firestore
6. Enable the Firestore API in the Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
"""

PROJECT_ID = "langchain-basics-70e02"
SESSION_ID="raj-user_session"
COLLECTION_NAME = "raj_chat_history"

# Initialize Firestore client
print("Init Firestore client")
client = firestore.Client(project=PROJECT_ID)

# Init Firestore chat message history
print("Init Firestore chat message history")
message_history = FirestoreChatMessageHistory(
    client=client,
    session_id=SESSION_ID,
    collection=COLLECTION_NAME
)

print("Chat history initialized")
print(f"Current chat history ::: {message_history.messages}")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

print("Starting chat session with model. Type 'exit' to quit")

while True: 
    user_inp = input(f"User: ")
    if user_inp.lower() == "exit":
        print("Exiting chat session.")
        message_history.save()
        break
    message_history.add_user_message(user_inp)
    response = llm.invoke(message_history.messages)
    message_history.add_ai_message(response.content)
    print(f"AI: {response.content}")

# Print the final chat history
print("Final chat history:")
for message in message_history.messages:    
    print(f"{message['role']}: {message['content']}")