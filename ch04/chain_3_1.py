from web_searching import web_search
from langchain_core.runnables import RunnableLambda
from chain_logging import dump_tap, log_info, log_step

NUM_SEARCH_RESULTS_PER_QUERY = 3


def _search_and_expand_urls(x):
    query = x["search_query"]
    log_step("STEP 3.1 - Execute web search", details=f'Query: "{query}"', icon="🌐")
    urls = web_search(
        web_query=query,
        num_results=NUM_SEARCH_RESULTS_PER_QUERY,
    )
    log_info(f"URLs found: {len(urls)}", icon="🔗")
    return [
        {
            "result_url": url,
            "search_query": x["search_query"],
            "user_question": x["user_question"],
        }
        for url in urls
    ]


search_result_urls_chain = (
    RunnableLambda(_search_and_expand_urls)
    | RunnableLambda(
        dump_tap("Expanded URL objects (search results)", icon="📦", max_chars=3000)
    )
)
