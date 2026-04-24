# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------

import os
import asyncio
import operator
from typing import Annotated, Sequence, TypedDict
import json
from env_config import load_env

from llm_factory import get_chat_model, get_embeddings_model
from vectorstore_manager import get_travel_info_vectorstore
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition

# -----------------------------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------------------------

ENV = load_env() #A
#A load environment variables from the project root .env file

_model_map = {
    "openai":  (ENV.openai_model,  ENV.openai_embedding_model),
    "ollama":  (ENV.ollama_model,  ENV.ollama_embedding_model),
    "gemini":  (ENV.gemini_model,  ENV.gemini_embedding_model),
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

@tool #A
def search_travel_info(query: str) -> str: #B
    """Search embedded WikiVoyage content for 
    information about destinations in England."""
    docs = ti_retriever.invoke(query) #C
    top = docs[:4] if isinstance(
        docs, list) else docs #C
    return "\n---\n".join(
        d.page_content for d in top) #D

#A Define the tool using the @tool decorator
#B Define the tool function, which takes a query, performs a semantic search and returns a string response from the vectorstore
#C Perform a semantic search on the vectorstore and return the top 4 results
#D Joins the top 4 results into a single string

# ----------------------------------------------------------------------------
# 3. Configure LLM with tool awareness
# ----------------------------------------------------------------------------
TOOLS = [search_travel_info] #A

llm_model = get_chat_model(#B
    use_responses_api=True) #B
llm_with_tools = llm_model.bind_tools(TOOLS) #C

#A Define the tools list (in our case, only one tool)
#B Instantiate the LLM model with the configured provider and the responses API
#C Bind the tools to the LLM model, which will generate a response with the tool calls

# ----------------------------------------------------------------------------
# 4. Initialize the dependencies for the LangGraph graph
# ----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# AgentState: it only contains LLM messages
# -----------------------------------------------------------------------------
class AgentState(TypedDict): #A
    messages: Annotated[Sequence[BaseMessage], operator.add] #B

#A Define the agent state
#B The agent state only contains LLM messages, which are appended to the list of messages

# -----------------------------------------------------------------------------
# CustomToolNode 
# -----------------------------------------------------------------------------

class ToolsExecutionNode: #A
    """Execute tools requested by the LLM in the last AIMessage."""

    def __init__(self, tools: Sequence): #B
        self._tools_by_name = {t.name: t for t in tools}

    def __call__(self, state: dict): #C
        messages: Sequence[BaseMessage] = state.get("messages", [])  

        last_msg = messages[-1] #D
        tool_messages: list[ToolMessage] = [] #E
        tool_calls = getattr(last_msg, 
            "tool_calls", []) #F
        
        for tool_call in tool_calls: #G
            tool_name = tool_call["name"] #H
            tool_args = tool_call["args"] #I
            tool = self._tools_by_name[tool_name] #J
            print(f"🔧 [tools] → {tool_name}  args={tool_args}")
            result = tool.invoke(tool_args) #K
            print(f"📄 [tools] ← {tool_name}  ({len(str(result))} chars)")
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(result), #L
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": tool_messages} #M
    
tools_execution_node = ToolsExecutionNode(TOOLS) #N

#A Define the tools execution node
#B Initialize the tools execution node with the tools list
#C Define the __call__ method, which is called when the node is invoked
#D Get the last message from the messages list
#E Initialize the tool messages list, to gather the results of the tool calls
#F Get the tool calls from the last message
#G Iterate over the tool calls
#H Get the tool name from the tool call
#I Get the tool arguments from the tool call
#J Get the tool from the tools list
#K Invoke the tool with the arguments
#L Add the tool result to the tool messages list
#M Return the tool messages list, which contains the results of the tool calls
#N Instantiate the tools execution node, to be used as a node in the LangGraph graph


# ----------------------------------------------------------------------------
# LLM node
# ----------------------------------------------------------------------------

def llm_node(state: AgentState): #A
    """LLM node that decides whether
    to call the search tool."""
    current_messages = state["messages"] #B
    print(f"🧠 [llm_node] Invoking LLM  (messages in state: {len(current_messages)})")
    respose_message = llm_with_tools.invoke(
        current_messages) #C
    tool_calls = getattr(respose_message, "tool_calls", [])
    if tool_calls:
        names = ", ".join(tc["name"] for tc in tool_calls)
        print(f"🔀 [llm_node] → tool call(s): {names}")
    else:
        print("✅ [llm_node] → final answer ready")
    return {"messages": [respose_message]} #D

#A Define the LLM node
#B Get the current messages from the agent state
#C Invoke the LLM model with the current messages. The LLM will decide whether to call the search tool or return an answer.
#D Return the response message, which contains the tool call or the answer

# ----------------------------------------------------------------------------
# 4. Build the LangGraph graph (llm_node + CustomToolNode)
# ----------------------------------------------------------------------------

builder = StateGraph(AgentState) #A
builder.add_node("llm_node", llm_node) #B
builder.add_node("tools", tools_execution_node) #B

builder.add_conditional_edges("llm_node", 
    tools_condition) #C

builder.add_edge("tools", "llm_node") #D

builder.set_entry_point("llm_node") #E
travel_info_agent = builder.compile() #F

#A Define the graph builder
#B Add the LLM node and the tools node to the graph
#C Add the conditional edges to the graph, to decide whether to execute the tool calls or return an answer and exit the graph
#D Add the edge from the tools node to the LLM node
#E Set the entry point to the LLM node
#F Compile the graph

# ----------------------------------------------------------------------------
# 5. Simple CLI interface
# ----------------------------------------------------------------------------

def chat_loop(): #A
    print("UK Travel Assistant (type 'exit' to quit)")
    while True:
        user_input = input(
            "You: ").strip() #B
        if user_input.lower() in \
            {"exit", "quit"}: #C
                break
        state = {"messages": 
            [HumanMessage(content=user_input)]} #D
        result = travel_info_agent.invoke(
            state) #E
        response_msg = result["messages"][-1] #F
        print(f"Assistant: {response_msg.content}\n") #G

#A Define the chat loop
#B Get the user input
#C Check if the user input is "exit" or "quit" to exit the loop
#D Create the initial state with a HumanMessage containing the user input
#E Invoke the graph with the initial state
#F Get the last message from the result, which contains the final answer
#G Print the assistant's final answer, from the content of the last message


if __name__ == "__main__":
    chat_loop() 
