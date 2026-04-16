from llm_models import get_llm
from web_scraping import web_scrape
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from prompts import (
    SUMMARY_PROMPT_TEMPLATE
)
from chain_logging import dump_tap, log_info, log_step

RESULT_TEXT_MAX_CHARACTERS = 10000


def _scrape_and_prepare_payload(x):
    url = x["result_url"]
    log_step("STEP 4.1 - Scrape source page", details=f"URL: {url}", icon="🕸️")
    scraped = web_scrape(url=url)[:RESULT_TEXT_MAX_CHARACTERS]
    log_info(f"Scraped chars: {len(scraped)}", icon="📄")
    return {
        "search_result_text": scraped,
        "result_url": x["result_url"],
        "search_query": x["search_query"],
        "user_question": x["user_question"],
    }


def _to_final_summary_payload(x):
    log_step("STEP 4.1 - Build source summary block", icon="🧱")
    return {
        "summary": f"Source Url: {x['result_url']}\nSummary: {x['text_summary']}",
        "user_question": x["user_question"],
    }


search_result_text_and_summary_chain = (
    RunnableLambda(_scrape_and_prepare_payload)
    | RunnableLambda(
        dump_tap("Prepared scrape payload", icon="📦", max_chars=2500)
    )
    | RunnableParallel(
        {
            "text_summary": SUMMARY_PROMPT_TEMPLATE
            | get_llm(provider="ollama")
            | RunnableLambda(
                dump_tap("LLM raw output (page summary)", icon="🤖", max_chars=2200)
            )
            | StrOutputParser()
            | RunnableLambda(
                dump_tap("LLM parsed text (page summary)", icon="📝", max_chars=2200)
            ),
            "result_url": lambda x: x["result_url"],
            "user_question": lambda x: x["user_question"],
        }
    )
    | RunnableLambda(_to_final_summary_payload)
    | RunnableLambda(dump_tap("Final source summary payload", icon="📦", max_chars=2200))
)
