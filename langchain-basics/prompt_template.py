import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", max_tokens=50)

# Dry run
# print(llm.invoke("What is the capital of France?"))

messages = [
    ('system', "You are an excellent content writer. \
               You are writing a blog post about {topic}. \
               You are writing in a {tone} tone. "), 
    ('user', "Write a blog post about {topic}")
]

# print(messages)

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke(
    {'topic': 'CSS', 'tone': 'funny'}
    )
res = llm.invoke(prompt)
print(res.content)