# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------

import os
import asyncio
import operator
from typing import Annotated, Sequence, TypedDict, Literal, Optional
from env_config import load_env
import random

from llm_factory import get_chat_model, get_embeddings_model
from vectorstore_manager import get_travel_info_vectorstore
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.managed.is_last_step import RemainingSteps
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# -----------------------------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------------------------

ENV = load_env() #A
#A load environment variables from the project root .env file

_model_map = {
    "openai": (ENV.openai_model,  ENV.openai_embedding_model),
    "ollama": (ENV.ollama_model,  ENV.ollama_embedding_model),
    "gemini": (ENV.gemini_model,  ENV.gemini_embedding_model),
}
_active_model, _active_embed = _model_map.get(ENV.llm_provider, ("?", "?"))
print(f"⚙️  Provider : {ENV.llm_provider}")
print(f"🤖 Model    : {_active_model}")
print(f"📐 Embed    : {_active_embed}")

# -----------------------------------------------------------------------------
# 1. Prepare knowledge base at startup
# -----------------------------------------------------------------------------

UK_DESTINATIONS = [ #A
    "Cornwall",
    "North_Cornwall",
    "South_Cornwall",
    "West_Cornwall",
]

ti_vectorstore_client = get_travel_info_vectorstore(UK_DESTINATIONS) #J
ti_retriever = ti_vectorstore_client.as_retriever() #K

#A Destination list; you can add more destinations here
#J Instantiate the vectorstore client
#K Instantiate the vectorstore retriever


# ----------------------------------------------------------------------------
# 2. Define the only tool
# ----------------------------------------------------------------------------

@tool(description="""Search travel information 
about destinations in England.""") #A
def search_travel_info(query: str) -> str: #B
    """Search embedded WikiVoyage content for information about destinations in England."""
    print(f"🔍 [search_travel_info] query='{query}'")
    docs = ti_retriever.invoke(query) #C
    top = docs[:4] if isinstance(docs, list) else docs #C
    print(f"📄 [search_travel_info] → {len(top)} chunks returned")
    return "\n---\n".join(d.page_content for d in top) #D

#A Define the tool using the @tool decorator
#B Define the tool function, which takes a query, performs a semantic search and returns a string response from the vectorstore
#C Perform a semantic search on the vectorstore and return the top 4 results
#D Joins the top 4 results into a single string

@tool(description="Get the weather forecast, given a town name.")
def weather_forecast(town: str) -> dict:
    """Get a mock weather forecast for a given town. Returns a WeatherForecast object with weather and temperature."""
    print(f"🌤️  [weather_forecast] town='{town}'")
    forecast = WeatherForecastService.get_forecast(town)
    if forecast is None:
        return {"error": f"No weather data available for '{town}'."}
    print(f"📄 [weather_forecast] → {forecast}")
    return forecast

# ----------------------------------------------------------------------------
# 3. Configure LLM with tool awareness
# ----------------------------------------------------------------------------
TOOLS = [search_travel_info, weather_forecast] #A

llm_model = get_chat_model(temperature=0, #B
                       use_responses_api=True) #B


#A Define the tools list (in our case, only one tool)
#B Instantiate the LLM model with the configured provider and the responses API

# ----------------------------------------------------------------------------
# 4. Initialize the dependencies for the LangGraph graph
# ----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# AgentState: it only contains LLM messages
# -----------------------------------------------------------------------------
class AgentState(TypedDict): #A
    messages: Annotated[Sequence[BaseMessage], operator.add]
    remaining_steps: RemainingSteps #B

#A Define the agent state
#B this is a special type of state that contains the remaining steps of the agent

# ----------------------------------------------------------------------------
# Build the travel info assistant React Agent
# ----------------------------------------------------------------------------

travel_info_agent = create_react_agent(
    model=llm_model,
    tools=TOOLS,
    state_schema=AgentState,
    prompt="You are a helpful assistant that can search travel information and get the weather forecast. Only use the tools to find the information you need (including town names).",
)

# ----------------------------------------------------------------------------
# 5. Simple CLI interface
# ----------------------------------------------------------------------------

def chat_loop(): #A
    print("UK Travel Assistant (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip() #B
        if user_input.lower() in {"exit", "quit"}: #C
            break
        state = {"messages": [HumanMessage(content=user_input)]} #D
        result = travel_info_agent.invoke(state) #E
        response_msg = result["messages"][-1] #F
        print(f"Assistant: {response_msg.content}\n") #G

#A Define the chat loop
#B Get the user input
#C Check if the user input is "exit" or "quit" to exit the loop
#D Create the initial state with a HumanMessage containing the user input
#E Invoke the graph with the initial state
#F Get the last message from the result, which contains the final answer
#G Print the assistant's final answer, from the content of the last message


# -----------------------------------------------------------------------------
# WeatherForecastService (Mock)
# -----------------------------------------------------------------------------

class WeatherForecast(TypedDict):
    town: str
    weather: Literal["sunny", "foggy", "rainy", "windy"]
    temperature: int

class WeatherForecastService:

    _weather_options = ["sunny", "foggy", "rainy", "windy"]
    _temp_min = 18
    _temp_max = 31

    @classmethod
    def get_forecast(cls, town: str) -> Optional[WeatherForecast]: #A
        weather = random.choice(cls._weather_options)
        temperature = random.randint(cls._temp_min, cls._temp_max)
        return WeatherForecast(town=town, weather=weather, temperature=temperature)

#A Define the get_forecast method, which returns a WeatherForecast object

if __name__ == "__main__":
    chat_loop() 
