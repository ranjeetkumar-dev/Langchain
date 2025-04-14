import requests
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchResults
from langchain import hub
from langchain.agents import tool
import datetime
from dotenv import load_dotenv

load_dotenv()

# OPENAI_API_KEY = "<your-openai-api-key>"


# Weather API Tool
@tool
def get_weather(city: str):
    """Fetches current weather for a specified city."""
    api_key = (
        "b29e81994068038d00f4f918337ec434"  # Replace with your OpenWeatherMap API key
    )
    base_url = "https://api.openweathermap.org/data/2.5/weather"

    params = {"q": city, "appid": api_key, "units": "metric"}

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        weather_description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        return f"Weather in {city}: {weather_description}, Temperature: {temperature}°C"
    else:
        return "Could not fetch weather data. Please check the city name."


# System Time Tool
# @tool
# def get_system_time(format: str = "%Y:%m:%d %H:%M:%S"):
#     """Returns the current date and time in the specified format."""
#     current_time = datetime.datetime.now()
#     return current_time.strftime(format)


# Search Tool
@tool
def search_engine(query):
    """Returns the search result using DuckDuckGo."""
    search = DuckDuckGoSearchResults()
    return search.invoke(query)


# LLM Default Answer Tool (Fallback for all other queries)
@tool
def general_qa(question: str):
    """Uses LLM to answer general questions when no tool is available."""
    model = ChatGroq(model_name="llama-3.3-70b-versatile")
    response = model.invoke(question)
    return response.content


# Language Model Setup
model = ChatGroq(model_name="llama-3.3-70b-versatile")

# Load Prompt Template
prompt_template = hub.pull("hwchase17/react")

# Tools List
tools = [get_weather, get_system_time, search_engine, general_qa]

# Create Agent
agent = create_react_agent(
    model,
    tools,
    prompt_template,
)

# Create Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

# Example Query
# query = "what is todays temperature in jabalpur today ?"
query = "tell me the current time"
# query = "who is current president of usa"
result = agent_executor.invoke({"input": query})
print(result["output"])
