from langchain.agents import create_agent
from unittest.mock import MagicMock

llm = MagicMock()
tools = []
try:
    agent = create_agent(model=llm, tools=tools, system_prompt="test")
    print("Success")
except Exception as e:
    print(f"Error: {e}")
