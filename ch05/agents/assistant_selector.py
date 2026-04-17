from models import get_llm, AssistantInfo
from prompts import ASSISTANT_SELECTION_PROMPT_TEMPLATE
from langchain_core.output_parsers import StrOutputParser
import json
from typing import Dict, Any
from graph_logging import invoke_llm, log_dump, log_info

def select_assistant(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select the appropriate research assistant based on the user question.
    """
    user_question = state["user_question"]
    
    # Format the prompt with the user question
    prompt = ASSISTANT_SELECTION_PROMPT_TEMPLATE.format(user_question=user_question)
    
    # Get the LLM response
    llm = get_llm()
    response = invoke_llm(llm, prompt, "assistant selection")
    response_text = response.content
    
    # Parse the response to get the assistant info
    try:
        # Extract the JSON part from the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]
        
        # Parse the JSON
        assistant_info = json.loads(json_str)
        log_dump("Assistant selection JSON object", assistant_info, icon="📦", max_chars=3000)
        
        # Return the updated state
        return {"assistant_info": assistant_info}
    except Exception as e:
        log_info(f"Error parsing assistant selection: {str(e)}", icon="⚠️")
        # Fallback to a default assistant if parsing fails
        default_assistant = {
            "assistant_type": "General research assistant",
            "assistant_instructions": "You are a general research AI assistant. Your main purpose is to draft comprehensive, informative, unbiased, and well-structured reports on given topics.",
            "user_question": user_question
        }
        log_dump("Default assistant selection JSON object", default_assistant, icon="📦", max_chars=3000)
        return {"assistant_info": default_assistant}
