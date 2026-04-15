# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv() #A
# openai_api_key = os.getenv("OPENAI_API_KEY") #B

def get_llm(): #C
    # return ChatOpenAI(openai_api_key=openai_api_key,
    #              model_name="gpt-5-nano")
    return ChatOllama(model="gemma4:e2b")
#A Load the environment variables from the .env file
#B Get the OpenAI API key from the environment variables
#C Instantiate and return the selected chat model
