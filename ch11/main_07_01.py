# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------

import asyncio
import operator
import os
from typing import Annotated, Sequence, TypedDict, Literal, Optional, List, Dict
from env_config import load_env


from llm_factory import get_chat_model, get_embeddings_model
from vectorstore_manager import get_travel_info_vectorstore
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.managed.is_last_step import RemainingSteps
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph_supervisor.supervisor import create_supervisor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage

from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
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

# A Destination list; you can add more destinations here
# J Instantiate the vectorstore client
# K Instantiate the vectorstore retriever


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

# ----------------------------------------------------------------------------
# 3. Configure LLM with tool awareness
# ----------------------------------------------------------------------------

MCP_TOOL_NAMES = set()

def log_mcp_tool_decisions(state: dict) -> dict:
    last_msg = state["messages"][-1] if state.get("messages") else None
    if not isinstance(last_msg, AIMessage):
        return {}

    for tool_call in last_msg.tool_calls:
        tool_name = tool_call.get("name")
        if tool_name in MCP_TOOL_NAMES:
            print(f"🌐 [mcp:{tool_name}] selected args={tool_call.get('args')}")
    return {}

async def get_accuweather_tools(): #A
    mcp_client = MultiServerMCPClient({ #B
        "accuweather": { #C
            "url": "http://127.0.0.1:8020/accu-mcp-server",
            "transport": "streamable_http"
        }
    })
    tools = await mcp_client.get_tools() #D
    MCP_TOOL_NAMES.update(tool.name for tool in tools) #E
    return tools #F

#A Define the function to get the AccuWeather tools as an async function
#B Instantiate the MultiServerMCPClient
#C Register the AccuWeather MCP server
#D Return the AccuWeather tools exposed by the MCP server
#E Track MCP tool names so LLM tool-call decisions can be logged
#F Return the MCP tools unchanged


async def chat_loop(agent): #A
    print("UK Travel Assistant (type 'exit' to quit)")
    while True: #B
        user_input = input("You: ").strip() #C
        if user_input.lower() in {"exit", "quit"}: #D
            break
        state = {"messages": [HumanMessage(
            content=user_input)]} #E
        result = await agent.ainvoke(state) #F
        response_text = get_last_ai_response(result["messages"]) #G
        print(
           f"Assistant: {response_text}\n") #H

#A Define the chat loop as an async function
#B Start the chat loop
#C Get the user input
#D Check if the user input is "exit" or "quit" to exit the loop
#E Create the initial state with a HumanMessage containing the user input
#F Invoke the agent with the initial state, asyncronously
#G Get the last message from the result, which contains the final answer
#H Print the assistant's final answer, from the content of the last message

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

class AgentState(TypedDict): #A
    messages: Annotated[Sequence[BaseMessage], operator.add]
    remaining_steps: RemainingSteps

async def main():
    accuweather_tools = \
        await get_accuweather_tools() #B
    tools = [search_travel_info, 
        *accuweather_tools] #C
    llm_model = get_chat_model(use_responses_api=True) #D

    travel_info_agent = create_react_agent( #E
        model=llm_model,
        tools=tools,
        state_schema=AgentState,
        name="travel_info_agent",
        post_model_hook=log_mcp_tool_decisions,
        prompt="""You are a helpful assistant that can 
        search travel information and get the weather forecast. 
        Only use the tools to find destination information, town
        names, and weather. Do not search for hotel, BnB,
        accommodation availability, room availability, or prices.""",
    )
    await chat_loop(travel_info_agent) #F

if __name__ == "__main__":
    asyncio.run(main()) #G

#A - Define the AgentState class
#B - Get the AccuWeather MCP server tools
#C - Combine the local search_travel_info tool with the AccuWeather MCP server tools
#D - Instantiate the LLM model
#E - Create the travel_info_agent
#F - Start the chat loop
#G - Run the main function, asyncronously      
