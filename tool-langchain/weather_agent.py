import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langchain_tavily import TavilySearch
from langchain.agents import AgentExecutor, create_react_agent, tool
# from langchain.tools import tool

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

query = "What is the weather in Dallas like right now? Answer in both F and C. And can you also tell me the current time there?"

prompt = hub.pull("hwchase17/react")

@tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get the current time in a specified format."""
    from datetime import datetime
    return datetime.now().strftime(format)

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location using Tavily."""
    tavily_search = TavilySearch()
    results = tavily_search.run(f"current weather in {location}")
    return results

@tool
def celcius_to_fahrenheitr(celsius: float) -> str:
    """Convert Celsius to Fahrenheit."""
    fahrenheit = (celsius * 9/5) + 32
    return f"{fahrenheit}°F"

@tool
def fahrenheit_to_celcius(fahrenheit: float) -> str:
    """Convert Fahrenheit to Celsius."""
    celsius = (fahrenheit - 32) * 5/9
    return f"{celsius}°C"

tools = [get_current_time, get_weather, celcius_to_fahrenheitr, fahrenheit_to_celcius]
agent = create_react_agent(
    llm=llm,
    prompt=prompt,
    tools=tools)
agent_eecutor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, 
    max_iterations=3
)

res = agent_eecutor.invoke({"input": query})
print(res)