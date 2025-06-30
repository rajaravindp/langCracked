import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

"""
Types of chains in LangChain:
1. Sequential chain
2. Parallel chain
3. Conditional chain
"""

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", max_tokens=50)

# Prompt Template - no separate Runnable chain required
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', "You are exceptional at dad jokes. Give me a dad joke about {topic}"), 
        ('user', "Write a dad joke about {topic} in a {tone} tone.")
    ]
)

# Combined chain using LCEL
chain = prompt_template | llm | StrOutputParser()
print(chain.invoke({"topic": "beers", "tone": "serious"}))