# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------

import os
import asyncio
import operator
from typing import Annotated, Sequence, TypedDict, Literal, Optional, List, Dict
from env_config import load_env
import random


from llm_factory import get_chat_model, get_embeddings_model
from vectorstore_manager import get_travel_info_vectorstore
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.managed.is_last_step import RemainingSteps
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit


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

@tool(description="Search travel information about destinations in England.") #A
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

llm_model = get_chat_model(#B
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
        result = accommodation_booking_agent.invoke(state) #E
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

# -----------------------------------------------------------------------------
# SQLDatabaseToolkit for Hotel Booking (SQLite)
# -----------------------------------------------------------------------------
hotel_db = SQLDatabase.from_uri("sqlite:///hotel_db/cornwall_hotels.db")
hotel_db_toolkit = SQLDatabaseToolkit(db=hotel_db, llm=llm_model)
hotel_db_toolkit_tools = hotel_db_toolkit.get_tools()

# -----------------------------------------------------------------------------
# BnBBookingService (Mock REST API client)
# -----------------------------------------------------------------------------

class BnBOffer(TypedDict): #A
    bnb_id: int
    bnb_name: str
    town: str
    available_rooms: int
    price_per_room: float

class BnBBookingService: #B
    @staticmethod
    def get_offers_near_town(town: str, num_rooms: int) \
        -> List[BnBOffer]: #C
        # Mocked REST API response: multiple BnBs per destination
        mock_bnb_offers = [ #D
            # Newquay
            {"bnb_id": 1, "bnb_name": "Seaside BnB", 
            "town": "Newquay", "available_rooms": 3, 
            "price_per_room": 80.0},
            {"bnb_id": 2, "bnb_name": "Surfside Guesthouse", 
            "town": "Newquay", "available_rooms": 2, 
            "price_per_room": 85.0},
            # Falmouth
            {"bnb_id": 3, "bnb_name": "Harbour View BnB", 
            "town": "Falmouth", "available_rooms": 4, 
            "price_per_room": 78.0},
            {"bnb_id": 4, "bnb_name": "Seafarer's Rest", 
            "town": "Falmouth", "available_rooms": 1, 
            "price_per_room": 90.0},
            # St Austell
            {"bnb_id": 5, "bnb_name": "Garden Gate BnB", 
            "town": "St Austell", "available_rooms": 2, "price_per_room": 82.0},
            {"bnb_id": 6, "bnb_name": "Coastal Cottage BnB", 
            "town": "St Austell", "available_rooms": 3, "price_per_room": 88.0},
            # Penzance
            {"bnb_id": 7, "bnb_name": "Penzance Pier BnB", 
            "town": "Penzance", "available_rooms": 2, "price_per_room": 95.0},
            {"bnb_id": 8, "bnb_name": "Cornish Charm BnB", 
            "town": "Penzance", "available_rooms": 3, "price_per_room": 87.0},
            # Camborne
            {"bnb_id": 9, "bnb_name": "Camborne Corner BnB", 
            "town": "Camborne", "available_rooms": 2, "price_per_room": 75.0},
            {"bnb_id": 10, "bnb_name": "Rose Cottage BnB", 
            "town": "Camborne", "available_rooms": 2, "price_per_room": 79.0},
            # Hayle
            {"bnb_id": 11, "bnb_name": "Hayle Haven BnB", 
            "town": "Hayle", "available_rooms": 3, "price_per_room": 83.0},
            {"bnb_id": 12, "bnb_name": "Dune View BnB", 
            "town": "Hayle", "available_rooms": 1, "price_per_room": 81.0},
            # Land's End
            {"bnb_id": 13, "bnb_name": "Land's End Lookout BnB", 
            "town": "Land's End", "available_rooms": 2, "price_per_room": 100.0},
            {"bnb_id": 14, "bnb_name": "Atlantic Edge BnB", 
            "town": "Land's End", "available_rooms": 2, "price_per_room": 105.0},
            # Bude
            {"bnb_id": 15, "bnb_name": "Bude Beach BnB", 
            "town": "Bude", "available_rooms": 2, "price_per_room": 77.0},
            {"bnb_id": 16, "bnb_name": "Cliffside BnB", 
            "town": "Bude", "available_rooms": 3, "price_per_room": 80.0},
            # Padstow
            {"bnb_id": 17, "bnb_name": "Padstow Harbour BnB", 
            "town": "Padstow", "available_rooms": 2, "price_per_room": 92.0},
            {"bnb_id": 18, "bnb_name": "Fisherman's Rest BnB", 
            "town": "Padstow", "available_rooms": 2, "price_per_room": 89.0},
            # St Ives
            {"bnb_id": 19, "bnb_name": "St Ives Bay BnB", "town": "St Ives", "available_rooms": 3, "price_per_room": 97.0},
            {"bnb_id": 20, "bnb_name": "Artists' Retreat BnB", "town": "St Ives", "available_rooms": 2, "price_per_room": 102.0},
            # Looe
            {"bnb_id": 21, "bnb_name": "Looe Riverside BnB", "town": "Looe", "available_rooms": 2, "price_per_room": 84.0},
            {"bnb_id": 22, "bnb_name": "Harbour Lights BnB", "town": "Looe", "available_rooms": 2, "price_per_room": 86.0},
            # Polperro
            {"bnb_id": 23, "bnb_name": "Polperro Cove BnB", "town": "Polperro", "available_rooms": 2, "price_per_room": 91.0},
            {"bnb_id": 24, "bnb_name": "Smuggler's Rest BnB", "town": "Polperro", "available_rooms": 2, "price_per_room": 93.0},
            # Mevagissey
            {"bnb_id": 25, "bnb_name": "Mevagissey Harbour BnB", "town": "Mevagissey", "available_rooms": 2, "price_per_room": 90.0},
            {"bnb_id": 26, "bnb_name": "Seafarer's BnB", "town": "Mevagissey", "available_rooms": 2, "price_per_room": 88.0},
            # Port Isaac
            {"bnb_id": 27, "bnb_name": "Port Isaac View BnB", 
            "town": "Port Isaac", "available_rooms": 2, 
            "price_per_room": 99.0},
            {"bnb_id": 28, "bnb_name": "Fisherman's Cottage BnB", 
            "town": "Port Isaac", "available_rooms": 2, 
            "price_per_room": 101.0},
            # Fowey
            {"bnb_id": 29, "bnb_name": "Fowey Quay BnB", 
            "town": "Fowey", "available_rooms": 2, 
            "price_per_room": 94.0},
            {"bnb_id": 30, "bnb_name": "Riverside Rest BnB", 
            "town": "Fowey", "available_rooms": 2, 
            "price_per_room": 96.0},
        ]
        offers = [offer for offer in 
            mock_bnb_offers 
            if offer["town"].lower() == town.lower() 
               and offer["available_rooms"] >= num_rooms]
        return offers
    
#A Define the return type of the BnB availability tool
#B Define the BnB availability tool
#C Call the BnB booking service to get the offers
#D Mocked BnB offers

# -----------------------------------------------------------------------------
# BnB Availability Tool
# -----------------------------------------------------------------------------

@tool(description="""Check BnB room availability and 
price for a destination in Cornwall.""") #A
def check_bnb_availability(destination: str, num_rooms: int) \
    -> List[Dict]: #B

    offers = BnBBookingService.get_offers_near_town(
        destination, num_rooms)
    if not offers:
        return [{"error": f"No available BnBs found in {destination} for {num_rooms} rooms."}]
    return offers


#A Define the BnB availability tool
#B Define the input and return type of the BnB availability tool

# -----------------------------------------------------------------------------
# Accommodation Booking Agent
# -----------------------------------------------------------------------------
BOOKING_TOOLS = hotel_db_toolkit_tools + \
   [check_bnb_availability] #A

accommodation_booking_agent = create_react_agent( #B
    model=llm_model,
    tools=BOOKING_TOOLS,
    state_schema=AgentState,
    prompt="""You are a helpful assistant that can check 
    hotel and BnB room availability and price for a
    destination in Cornwall. You can use the tools to 
    get the information you need. If the users does 
    not specify the accommodation type, you should 
    check both hotels and BnBs.""",
)

#A Define the booking tools, which are the tools from the hotel database toolkit and the BnB availability tool
#B Create the accommodation booking agent


if __name__ == "__main__":
    chat_loop() 
