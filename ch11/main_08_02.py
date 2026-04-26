# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------

import os
import uuid
import asyncio
import operator
from typing import Annotated, Sequence, TypedDict, Literal, Optional, List, Dict
from env_config import load_env
import random
from enum import Enum
from pydantic import BaseModel, Field


from llm_factory import get_chat_model
from vectorstore_manager import get_travel_info_vectorstore
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.managed.is_last_step import RemainingSteps
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


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

#A Define the tools list (in our case, only one tool)


llm_model = get_chat_model(#A
                       use_responses_api=True, #B                      
                       use_previous_response_id=True) #C


#A Instantiate the LLM model with the configured provider
#B Use the Responses API
#C Use the previous response ID to continue the conversation

# -----------------------------------------------------------------------------
# AgentState: it only contains LLM messages
# -----------------------------------------------------------------------------
class AgentState(TypedDict): #A
    messages: Annotated[Sequence[BaseMessage], operator.add]
    remaining_steps: RemainingSteps #B

#A Define the agent state
#B this is a special type of state that contains the remaining steps of the agent

# -----------------------------------------------------------------------------
# AgentType Enum and Structured Output Model
# -----------------------------------------------------------------------------
class AgentType(str, Enum):
    travel_info_node = "travel_info_node"
    accommodation_booking_node = "accommodation_booking_node"

class AgentTypeOutput(BaseModel): 
    agent: AgentType = Field(..., description="Which agent should handle the query?")

# Structured LLM for routing
llm_router = llm_model.with_structured_output(AgentTypeOutput)

# -----------------------------------------------------------------------------
# Router Agent System Prompt Constant
# -----------------------------------------------------------------------------
ROUTER_SYSTEM_PROMPT = (
    "You are a router. Given the following user message, decide if it is a travel information question (about destinations, attractions, or general travel info) "
    "or an accommodation booking question (about hotels, BnBs, room availability, or prices).\n"
    "If it is a travel information question, respond with 'travel_info_node'.\n"
    "If it is an accommodation booking question, respond with 'accommodation_booking_node'."
)

# -----------------------------------------------------------------------------
# Router Agent Node for LangGraph (with structured output)
# -----------------------------------------------------------------------------
def router_agent_node(state: AgentState) -> Command[AgentType]:
    """Router node: decides which agent should handle the user query."""
    messages = state["messages"] #A
    last_msg = messages[-1] if messages else None #B
    if isinstance(last_msg, HumanMessage): #C
        user_input = last_msg.content #D
        print(f"🧠 [router_agent] Routing query: '{user_input[:50]}...'")
        router_messages = [ #E
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_input)
        ]
        router_response = llm_router.invoke(router_messages) #F
        agent_name = router_response.agent.value #G
        print(f"🔀 [router_agent] → selected agent: {agent_name}")
        return Command(update=state, goto=agent_name) #H
    
    return Command(update=state, goto=AgentType.travel_info_node) #I

#A Get the messages from the state
#B Get the last message from the messages list
#C Check if the last message is a HumanMessage
#D Get the content of the last message
#E Create the router messages, including the system prompt and the user input
#F Invoke the router model, which returns the relevant agent name
#G Get the agent name from the router response
#H Return the command to update the state and go to the agent
#I If the last message is not a HumanMessage, return the command to update the state and go to the travel_info_agent (default agent)

# -----------------------------------------------------------------------------
# 4. Initialize the dependencies for the LangGraph graph
# -----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Build the travel info assistant React Agent
# ----------------------------------------------------------------------------

travel_info_agent = create_react_agent(
    model=llm_model,
    tools=TOOLS,
    state_schema=AgentState,
    prompt="""You are a helpful assistant that can search
    travel information and get the weather forecast.
    Only use the tools to find destination information,
    town names, and weather. Do not search for hotel,
    BnB, accommodation availability, room availability,
    or prices.""",
)


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

# Add logging to database tools
def patch_db_tool(tool):
    original_invoke = tool.invoke
    def patched_invoke(input, config=None, **kwargs):
        def get_val(inp, key):
            if isinstance(inp, dict): return inp.get(key, inp)
            if hasattr(inp, key): return getattr(inp, key)
            return inp

        if tool.name == "sql_db_query":
            print(f"🗄️  [database:query] {get_val(input, 'query')}")
        elif tool.name == "sql_db_schema":
            print(f"🗄️  [database:schema] {get_val(input, 'table_names')}")
        else:
            print(f"🗄️  [database:tool] {tool.name}")
        return original_invoke(input, config=config, **kwargs)
    object.__setattr__(tool, 'invoke', patched_invoke)
    return tool

hotel_db_toolkit_tools = [patch_db_tool(t) for t in hotel_db_toolkit_tools]

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
    def get_offers_near_town(town: str, num_rooms: int) -> List[BnBOffer]: #C
        # Mocked REST API response: multiple BnBs per destination
        mock_bnb_offers = [ #D
            # Newquay
            {"bnb_id": 1, "bnb_name": "Seaside BnB", "town": "Newquay", "available_rooms": 3, "price_per_room": 80.0},
            {"bnb_id": 2, "bnb_name": "Surfside Guesthouse", "town": "Newquay", "available_rooms": 2, "price_per_room": 85.0},
            # Falmouth
            {"bnb_id": 3, "bnb_name": "Harbour View BnB", "town": "Falmouth", "available_rooms": 4, "price_per_room": 78.0},
            {"bnb_id": 4, "bnb_name": "Seafarer's Rest", "town": "Falmouth", "available_rooms": 1, "price_per_room": 90.0},
            # St Austell
            {"bnb_id": 5, "bnb_name": "Garden Gate BnB", "town": "St Austell", "available_rooms": 2, "price_per_room": 82.0},
            {"bnb_id": 6, "bnb_name": "Coastal Cottage BnB", "town": "St Austell", "available_rooms": 3, "price_per_room": 88.0},
            # Penzance
            {"bnb_id": 7, "bnb_name": "Penzance Pier BnB", "town": "Penzance", "available_rooms": 2, "price_per_room": 95.0},
            {"bnb_id": 8, "bnb_name": "Cornish Charm BnB", "town": "Penzance", "available_rooms": 3, "price_per_room": 87.0},
            # Camborne
            {"bnb_id": 9, "bnb_name": "Camborne Corner BnB", "town": "Camborne", "available_rooms": 2, "price_per_room": 75.0},
            {"bnb_id": 10, "bnb_name": "Rose Cottage BnB", "town": "Camborne", "available_rooms": 2, "price_per_room": 79.0},
            # Hayle
            {"bnb_id": 11, "bnb_name": "Hayle Haven BnB", "town": "Hayle", "available_rooms": 3, "price_per_room": 83.0},
            {"bnb_id": 12, "bnb_name": "Dune View BnB", "town": "Hayle", "available_rooms": 1, "price_per_room": 81.0},
            # Land's End
            {"bnb_id": 13, "bnb_name": "Land's End Lookout BnB", "town": "Land's End", "available_rooms": 2, "price_per_room": 100.0},
            {"bnb_id": 14, "bnb_name": "Atlantic Edge BnB", "town": "Land's End", "available_rooms": 2, "price_per_room": 105.0},
            # Bude
            {"bnb_id": 15, "bnb_name": "Bude Beach BnB", "town": "Bude", "available_rooms": 2, "price_per_room": 77.0},
            {"bnb_id": 16, "bnb_name": "Cliffside BnB", "town": "Bude", "available_rooms": 3, "price_per_room": 80.0},
            # Padstow
            {"bnb_id": 17, "bnb_name": "Padstow Harbour BnB", "town": "Padstow", "available_rooms": 2, "price_per_room": 92.0},
            {"bnb_id": 18, "bnb_name": "Fisherman's Rest BnB", "town": "Padstow", "available_rooms": 2, "price_per_room": 89.0},
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
            {"bnb_id": 27, "bnb_name": "Port Isaac View BnB", "town": "Port Isaac", "available_rooms": 2, "price_per_room": 99.0},
            {"bnb_id": 28, "bnb_name": "Fisherman's Cottage BnB", "town": "Port Isaac", "available_rooms": 2, "price_per_room": 101.0},
            # Fowey
            {"bnb_id": 29, "bnb_name": "Fowey Quay BnB", "town": "Fowey", "available_rooms": 2, "price_per_room": 94.0},
            {"bnb_id": 30, "bnb_name": "Riverside Rest BnB", "town": "Fowey", "available_rooms": 2, "price_per_room": 96.0},
        ]
        offers = [offer for offer in mock_bnb_offers if offer["town"].lower() == town.lower() and offer["available_rooms"] >= num_rooms]
        return offers
    
#A Define the return type of the BnB availability tool
#B Define the BnB availability tool
#C Call the BnB booking service to get the offers
#D Mocked BnB offers

# -----------------------------------------------------------------------------
# BnB Availability Tool
# -----------------------------------------------------------------------------

@tool(description="Check BnB room availability and price for a destination in Cornwall.") #A
def check_bnb_availability(destination: str, num_rooms: int) -> List[Dict]: #B
    """Check BnB room availability and price for the requested destination and number of rooms."""
    print(f"🏠 [check_bnb_availability] destination='{destination}', rooms={num_rooms}")
    offers = BnBBookingService.get_offers_near_town(destination, num_rooms)
    if not offers:
        print(f"📄 [check_bnb_availability] → No offers found")
        return [{"error": f"No available BnBs found in {destination} for {num_rooms} rooms."}]
    print(f"📄 [check_bnb_availability] → {len(offers)} offers found")
    return offers


#A Define the BnB availability tool
#B Define the input and return type of the BnB availability tool

# -----------------------------------------------------------------------------
# Accommodation Booking Agent
# -----------------------------------------------------------------------------
BOOKING_TOOLS = hotel_db_toolkit_tools + [check_bnb_availability] #A

accommodation_booking_agent = create_react_agent( #B
    model=llm_model,
    tools=BOOKING_TOOLS,
    state_schema=AgentState,
    prompt="""You are a helpful assistant that can check hotel
    and BnB room availability and price for a destination in
    Cornwall. You can use the tools to get the information you
    need. If the users does not specify the accommodation type,
    you should check both hotels and BnBs. If a previous message
    has already recommended a town, use that town for availability
    and pricing instead of choosing a different destination.""",
)

#A Define the booking tools, which are the tools from the hotel database toolkit and the BnB availability tool
#B Create the accommodation booking agent

# -----------------------------------------------------------------------------
# Agent nodes for the graph (to add logging)
# -----------------------------------------------------------------------------

def travel_info_node(state: AgentState):
    print(f"🤖 [travel_info_agent] node entered")
    return travel_info_agent.invoke(state)

def accommodation_booking_node(state: AgentState):
    print(f"🏨 [accommodation_booking_agent] node entered")
    return accommodation_booking_agent.invoke(state)

# -----------------------------------------------------------------------------
# Build the LangGraph graph with router, travel_info_agent, and accommodation_booking_agent
# -----------------------------------------------------------------------------
graph = StateGraph(AgentState) #A
graph.add_node("router_agent", router_agent_node) #B
graph.add_node("travel_info_node", 
    travel_info_node) #C
graph.add_node("accommodation_booking_node", 
    accommodation_booking_node) #D

graph.add_edge("travel_info_node", END) #E
graph.add_edge("accommodation_booking_node", END) #F

graph.set_entry_point("router_agent") #G

checkpointer = InMemorySaver() #H
travel_assistant = graph.compile(checkpointer=checkpointer) #I

#A Define the graph
#B Add the router agent node
#C Add the travel info agent node
#D Add the accommodation booking agent node
#E Add the edge from the travel info agent to the end
#F Add the edge from the accommodation booking agent to the end
#G Set the entry point to the router agent
#H Instantiate the in-memory checkpointer
#I Compile the graph with the in-memory checkpointer

# ----------------------------------------------------------------------------
# 5. Simple CLI interface
# ----------------------------------------------------------------------------

def message_content_to_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip() if content else ""

def get_last_ai_response(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = message_content_to_text(message)
            if text:
                return text
    return ""

def log_memory_state_message(message: HumanMessage, thread_id) -> None:
    content = str(message.content)
    preview = content if len(content) <= 120 else f"{content[:120]}..."
    print(f"💾 [memory:{thread_id}] adding latest user message to state: {preview}")

def chat_loop(): 
    thread_id=uuid.uuid1() #A
    print(f'Thread ID: {thread_id}') 
    config={"configurable": 
       {"thread_id": thread_id}} #B

    user_input = input("You: ").strip() #C

    latest_message = HumanMessage(content=user_input)
    log_memory_state_message(latest_message, thread_id)
    question = {"messages": 
        [latest_message]} #D
    result = travel_assistant.invoke(
        question, config=config) #E
    response_text = get_last_ai_response(result["messages"]) #F
    print(
       f"Assistant: {response_text}\n") #G

    state_history = travel_assistant.get_state_history(
        config) #H
    state_history_list = list(state_history) #I
    print(f'State history: {state_history_list}') #I

    last_snapshot = list(state_history_list)[0] #J
    print(f'Last snapshot: {last_snapshot.config}')

    thread_id = last_snapshot.config[
        "configurable"]["thread_id"] #K
    last_checkpoint_id = last_snapshot.config[
        "configurable"]["checkpoint_id"] #L

    new_config = {"configurable": #M
              {"thread_id": thread_id, 
               "checkpoint_id": last_checkpoint_id}}
    
    retrieved_snapshot = travel_assistant.get_state(
        new_config) #N
    print(
       f'Retrieved snapshot: {retrieved_snapshot}') #O

    travel_assistant.invoke(None, 
    config=new_config) #P

    latest_message = HumanMessage(
        content="What is the weather in the same town?")
    log_memory_state_message(latest_message, thread_id)
    new_question = {"messages": [latest_message]}
    result = travel_assistant.invoke(new_question, 
        config=new_config) #Q
    response_text = get_last_ai_response(result["messages"]) #R

    print(
       f"Assistant: {response_text}\n") #S

#A Create a unique thread id
#B Create a config with the thread id
#C Create the initial state with a HumanMessage containing the user input
#D Set the state with the HumanMessage
#E Invoke the graph with the state and the config
#F Get the last message from the result, which contains the final answer
#G Print the assistant's final answer, from the content of the last message
#H Get the state history from the graph
#I Print the state history
#J Get the last snapshot from the state history
#K Get the thread id from the last snapshot
#L Get the checkpoint id from the last snapshot
#M Create a new config with the thread id and the checkpoint id
#N Get the snapshot from the graph with the new config referencing the last checkpoint
#O Print the retrieved snapshot
#P Rewind the graph to the last checkpoint
#Q Invoke the graph with the new question referencing content from the last checkpoint
#R Get the last message from the result, which contains the final answer
#S Print the assistant's final answer, from the content of the last message

if __name__ == "__main__":
    chat_loop() 
