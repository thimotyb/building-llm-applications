from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, List, Annotated, Tuple
import os
from pathlib import Path

from models import ResearchState
from agents.assistant_selector import select_assistant
from agents.web_researcher import generate_search_queries, perform_web_searches, summarize_search_results, evaluate_search_relevance
from agents.report_writer import write_research_report
from graph_logging import configure_file_logging, log_info, log_node, log_research_state, log_step

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "ch05-research.log"

def create_research_graph() -> StateGraph:
    """
    Create the LangGraph research graph that coordinates the agents.
    """
    # Define the graph
    graph = StateGraph(ResearchState)
    
    # Add nodes to the graph
    graph.add_node("select_assistant", log_node("select_assistant", select_assistant))
    graph.add_node("generate_search_queries", log_node("generate_search_queries", generate_search_queries))
    graph.add_node("perform_web_searches", log_node("perform_web_searches", perform_web_searches))
    graph.add_node("summarize_search_results", log_node("summarize_search_results", summarize_search_results))
    graph.add_node("evaluate_search_relevance", log_node("evaluate_search_relevance", evaluate_search_relevance))
    graph.add_node("write_research_report", log_node("write_research_report", write_research_report))
    
    # Define the conditional routing function for relevance evaluation
    def route_based_on_relevance(state: Dict[str, Any]) -> str:
        """
        Route to either generate new search queries or continue to report writing
        based on the relevance evaluation.
        """
        # This value is persisted by evaluate_search_relevance. Mutating state
        # inside a conditional router is not written back by LangGraph.
        iteration_count = state.get("iteration_count", 0)
        
        # Check if we've reached the maximum number of iterations (3)
        if iteration_count >= 3:
            log_info(
                f"Reached maximum iterations ({iteration_count}). Proceeding to write report with current results.",
                icon="🔁",
            )
            return "write_research_report"
        
        # Otherwise, check if we should regenerate queries
        if state.get("should_regenerate_queries", False):
            log_info(f"Iteration {iteration_count}: Regenerating search queries.", icon="🔁")
            return "generate_search_queries"
        else:
            log_info(
                f"Iteration {iteration_count}: Search results are relevant. Proceeding to write report.",
                icon="✅",
            )
            return "write_research_report"
    
    # Define the flow of the graph
    graph.add_edge("select_assistant", "generate_search_queries")
    graph.add_edge("generate_search_queries", "perform_web_searches")
    graph.add_edge("perform_web_searches", "summarize_search_results")
    graph.add_edge("summarize_search_results", "evaluate_search_relevance")
    
    # Add conditional routing based on relevance evaluation
    graph.add_conditional_edges(
        "evaluate_search_relevance",
        route_based_on_relevance,
        {
            "generate_search_queries": "generate_search_queries",
            "write_research_report": "write_research_report"
        }
    )
    
    graph.add_edge("write_research_report", END)
    
    # Set the entry point
    graph.set_entry_point("select_assistant")
    
    return graph

def run_research(question: str) -> str:
    """
    Run the research graph with a user question.
    
    Args:
        question: The user's research question
        
    Returns:
        The final research report
    """
    log_file = os.getenv("CH05_LOG_FILE") or str(DEFAULT_LOG_FILE)
    log_path = configure_file_logging(log_file, truncate=True)
    log_step("Pipeline start", details=f'Question: "{question}"', icon="🎬")
    log_info(f"Writing execution log to: {log_path}", icon="🧾")

    # Create the graph
    research_graph = create_research_graph()
    
    # Compile the graph
    app = research_graph.compile()
    
    # Initialize the state
    initial_state = {
        "user_question": question,
        "assistant_info": None,
        "search_queries": None,
        "search_results": None,
        "search_summaries": None,
        "research_summary": None,
        "final_report": None,
        "used_fallback_search": False,
        "relevance_evaluation": None,
        "should_regenerate_queries": None,
        "iteration_count": 0
    }
    log_research_state("Initial ResearchState", initial_state, icon="🧭")
    
    # Run the graph
    result = app.invoke(initial_state)
    log_research_state("Final ResearchState", result, icon="🏁")
    
    # Extract and return the final report
    return result["final_report"]

# For testing purposes
if __name__ == "__main__":
    # Example usage
    question = "What can you tell me about Astorga's roman spas"
    report = run_research(question)
    print(report)
