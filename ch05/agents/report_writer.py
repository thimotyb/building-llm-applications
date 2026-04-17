from models import get_llm
from prompts import RESEARCH_REPORT_PROMPT_TEMPLATE
from typing import Dict, Any
from graph_logging import invoke_llm, log_dump

def write_research_report(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write a research report based on the summarized search results.
    """
    research_summary = state["research_summary"]
    user_question = state["user_question"]
    
    # Format the prompt
    prompt = RESEARCH_REPORT_PROMPT_TEMPLATE.format(
        research_summary=research_summary,
        user_question=user_question
    )
    
    # Get the LLM response
    llm = get_llm()
    response = invoke_llm(llm, prompt, "final research report", max_output_chars=5000)
    report = response.content
    log_dump("Final research report", report, icon="📘", max_chars=5000)
    
    # Return the updated state
    return {"final_report": report}
