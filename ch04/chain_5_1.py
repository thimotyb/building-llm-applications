from llm_models import get_llm
from prompts import (
    RESEARCH_REPORT_PROMPT_TEMPLATE
)
from chain_1_2 import assistant_instructions_chain
from chain_2_1 import web_searches_chain
from chain_3_1 import search_result_urls_chain
from chain_4_1 import search_result_text_and_summary_chain

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from chain_logging import dump_tap, log_step


def _merge_source_summaries(x):
    log_step(
        "STEP 5.1 - Merge summaries for one web search query",
        details=f"Source summaries: {len(x)}",
        icon="🧱",
    )
    return {
        "summary": "\n".join([i["summary"] for i in x]),
        "user_question": x[0]["user_question"] if len(x) > 0 else "",
    }


def _merge_query_summaries(x):
    log_step(
        "STEP 5.1 - Merge summaries across all queries",
        details=f"Query summary blocks: {len(x)}",
        icon="🧩",
    )
    return {
        "research_summary": "\n\n".join([i["summary"] for i in x]),
        "user_question": x[0]["user_question"] if len(x) > 0 else "",
    }


search_and_summarization_chain = (
    search_result_urls_chain
    | RunnableLambda(
        dump_tap("URL objects for one query", icon="🔗", max_chars=3000)
    )
    | search_result_text_and_summary_chain.map()  # parallelize for each url
    | RunnableLambda(_merge_source_summaries)
    | RunnableLambda(
        dump_tap("Merged summary for one query", icon="📦", max_chars=3000)
    )
)

web_research_chain = (
    assistant_instructions_chain
    | RunnableLambda(
        dump_tap("Assistant selection payload", icon="📦", max_chars=3000)
    )
    | web_searches_chain
    | RunnableLambda(
        dump_tap("Search query list payload", icon="🔎", max_chars=3000)
    )
    | search_and_summarization_chain.map()  # parallelize for each web search
    | RunnableLambda(_merge_query_summaries)
    | RunnableLambda(
        dump_tap("Merged research summary payload", icon="📚", max_chars=3500)
    )
    | RESEARCH_REPORT_PROMPT_TEMPLATE
    | get_llm(provider="ollama")
    | RunnableLambda(
        dump_tap("LLM raw output (final report)", icon="🤖", max_chars=4000)
    )
    | StrOutputParser()
    | RunnableLambda(
        dump_tap("LLM parsed text (final report)", icon="📝", max_chars=4000)
    )
)
