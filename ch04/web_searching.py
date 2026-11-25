import time
from typing import List

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# The wrapper is created once and reused for every query.
#
# region="us-en" replaces the wrapper's default "wt-wt" ("no region").
# With its default backend="auto", the ddgs library queries several search
# engines, including a Wikipedia engine that builds its URL from the language
# part of the region. With "wt-wt" that URL becomes https://wt.wikipedia.org,
# a host that does not exist, so that engine always fails with a DNS error.
# With "us-en" it queries en.wikipedia.org and returns English results.
_search_wrapper = DuckDuckGoSearchAPIWrapper(region="us-en")

# How many times to try a query before giving up, and the base pause between
# attempts (2 s, then 4 s). Search engines throttle repeated automated
# queries, especially from cloud VMs and from many students on one network,
# so a failed attempt often succeeds a few seconds later.
MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 2


def web_search(web_query: str, num_results: int) -> List[str]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            results = _search_wrapper.results(web_query, num_results)
            # Keep only real results: when nothing is found the wrapper can
            # return a placeholder entry without a "link" key.
            urls = [r["link"] for r in results if r.get("link")]
            if urls:
                return urls
            print(f"[web_search] No results for {web_query!r} "
                  f"(attempt {attempt}/{MAX_ATTEMPTS})")
        except Exception as e:
            # ddgs raises DDGSException when none of the engines it tried
            # returned results (rate limits, timeouts, DNS errors...). The
            # message shows only the last engine's error, which can be
            # misleading, so log it and retry instead of crashing the chain.
            print(f"[web_search] Search failed for {web_query!r} "
                  f"(attempt {attempt}/{MAX_ATTEMPTS}): {e}")
        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_BASE_DELAY_SECONDS * attempt)

    # Every attempt failed: return no URLs for this query. The callers treat
    # an empty list as "nothing found", so the research run continues with
    # the results of the other queries instead of stopping with an exception.
    print(f"[web_search] Giving up on {web_query!r}: returning no URLs")
    return []
