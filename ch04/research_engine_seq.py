from web_searching import web_search
from web_scraping import web_scrape
from llm_models import get_llm
from utilities import to_obj
import json
import re
from prompts import (
    ASSISTANT_SELECTION_PROMPT_TEMPLATE,
    WEB_SEARCH_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    RESEARCH_REPORT_PROMPT_TEMPLATE
)

NUM_SEARCH_QUERIES = 2
NUM_SEARCH_RESULTS_PER_QUERY = 3
RESULT_TEXT_MAX_CHARACTERS = 10000
question = "What can I see and do in the Spanish town of Astorga?"


def _hr():
    print("\n" + "=" * 80)


def log_step(step_number, title, details=None, icon="🚀"):
    _hr()
    print(f"{icon} [STEP {step_number}] {title}")
    if details:
        print(f"  {details}")


def log_info(message, icon="•"):
    print(f"  {icon} {message}")


def log_llm_output(title, text, max_chars=2500):
    _hr()
    print(f"🤖 [LLM OUTPUT] {title}")
    preview = (text or "").strip()
    if len(preview) > max_chars:
        preview = preview[:max_chars] + "\n... [truncated]"
    print(preview if preview else "<empty output>")


def parse_json_with_fallback(raw_text, default_value):
    parsed = to_obj(raw_text)
    if parsed:
        return parsed

    # Try extracting JSON from markdown code fences.
    fence_matches = re.findall(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    for chunk in fence_matches:
        try:
            return json.loads(chunk)
        except Exception:
            pass

    # Try first JSON object/array found in text.
    bracket_starts = [idx for idx in (raw_text.find("{"), raw_text.find("[")) if idx != -1]
    if bracket_starts:
        start = min(bracket_starts)
        candidate = raw_text[start:].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return default_value


###
llm = get_llm()
log_step(0, "Pipeline start", f'Question: "{question}"', icon="🎬")

# select research assistant instructions
log_step(1, "Select assistant instructions", icon="🧠")
assistant_selection_prompt = ASSISTANT_SELECTION_PROMPT_TEMPLATE.format(
    user_question=question
)
assistant_instructions = llm.invoke(assistant_selection_prompt)
log_llm_output("Assistant selection", assistant_instructions.content)
assistant_instructions_dict = parse_json_with_fallback(
    assistant_instructions.content, default_value={}
)

assistant_instructions_text = (
    assistant_instructions_dict.get("assistant_instructions")
    or assistant_instructions_dict.get("instructions")
)
resolved_user_question = (
    assistant_instructions_dict.get("user_question")
    or assistant_instructions_dict.get("question")
)

if not assistant_instructions_text or not resolved_user_question:
    raise RuntimeError(
        "Invalid assistant selection JSON: required keys missing "
        "('assistant_instructions' and/or 'user_question')."
    )
log_info("Assistant instructions generated")

# generate search queries
log_step(2, "Generate web search queries", icon="📝")
web_search_prompt = WEB_SEARCH_PROMPT_TEMPLATE.format(
    assistant_instructions=assistant_instructions_text,
    num_search_queries=NUM_SEARCH_QUERIES,
    user_question=resolved_user_question,
)
web_search_queries = llm.invoke(web_search_prompt)
log_llm_output("Web search query generation", web_search_queries.content)
web_search_queries_list = parse_json_with_fallback(
    web_search_queries.content, default_value=[]
)
if not isinstance(web_search_queries_list, list):
    raise RuntimeError("Invalid web search query JSON: expected a JSON array.")

sanitized_web_search_queries_list = []
for item in web_search_queries_list:
    if isinstance(item, dict) and item.get("search_query"):
        sanitized_web_search_queries_list.append(item)

if not sanitized_web_search_queries_list:
    raise RuntimeError(
        "Invalid web search query JSON: no items with a non-empty 'search_query'."
    )

web_search_queries_list = sanitized_web_search_queries_list
if len(web_search_queries_list) != NUM_SEARCH_QUERIES:
    raise RuntimeError(
        f"Invalid web search query JSON: expected {NUM_SEARCH_QUERIES} queries, "
        f"got {len(web_search_queries_list)}."
    )
log_info(f"Generated {len(web_search_queries_list)} search queries")
for idx, item in enumerate(web_search_queries_list, start=1):
    log_info(f'Query {idx}: {item["search_query"]}')

# find all the search result urls: NUM_SEARCH_QUERIES x NUM_SEARCH_RESULTS_PER_QUERY
log_step(
    3,
    "Search the web for each query",
    f"Top {NUM_SEARCH_RESULTS_PER_QUERY} results per query",
    icon="🔎",
)
searches_and_result_urls = [
    {
        "result_urls": web_search(
            web_query=wq["search_query"],
            num_results=NUM_SEARCH_RESULTS_PER_QUERY,
        ),
        "search_query": wq["search_query"],
    }
    for wq in web_search_queries_list
]
for idx, row in enumerate(searches_and_result_urls, start=1):
    log_info(
        f'Query {idx} -> {len(row["result_urls"])} URLs found '
        f'("{row["search_query"]}")'
    )

# flatten the search result urls
log_step(4, "Flatten URLs from all query results", icon="🧩")
search_query_and_result_url_list = []
for qr in searches_and_result_urls:
    search_query_and_result_url_list.extend(
        [
            {
                "search_query": qr["search_query"],
                "result_url": r,
            }
            for r in qr["result_urls"]
        ]
    )
log_info(f"Total URLs to scrape: {len(search_query_and_result_url_list)}")

# scrape the result text from each result url
log_step(
    5,
    "Scrape page content",
    f"Max chars per page: {RESULT_TEXT_MAX_CHARACTERS}",
    icon="🌐",
)
result_text_list = []
for idx, re in enumerate(search_query_and_result_url_list, start=1):
    log_info(f'Scraping [{idx}/{len(search_query_and_result_url_list)}]: {re["result_url"]}')
    scraped_text = web_scrape(url=re["result_url"])[:RESULT_TEXT_MAX_CHARACTERS]
    result_text_list.append(
        {
            "result_text": scraped_text,
            "result_url": re["result_url"],
            "search_query": re["search_query"],
        }
    )
    log_info(f"Collected {len(scraped_text)} chars")

# summarize each result text
log_step(6, "Summarize each scraped result", icon="✍️")
result_text_summary_list = []
for idx, rt in enumerate(result_text_list, start=1):
    log_info(
        f'Summarizing [{idx}/{len(result_text_list)}] '
        f'for query "{rt["search_query"]}"'
    )
    summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
        search_result_text=rt["result_text"],
        search_query=rt["search_query"],
    )
    text_summary = llm.invoke(summary_prompt)
    log_llm_output(
        f'Summary for {rt["result_url"]}',
        text_summary.content,
        max_chars=1500,
    )
    result_text_summary_list.append(
        {
            "text_summary": text_summary.content,
            "result_url": rt["result_url"],
            "search_query": rt["search_query"],
        }
    )
log_info(f"Summaries created: {len(result_text_summary_list)}")

# create a text including result summary and url from each result
log_step(7, "Build merged summaries block", icon="🧱")
stringified_summary_list = [
    f'Source URL: {sr["result_url"]}\nSummary: {sr["text_summary"]}'
    for sr in result_text_summary_list
]
log_info(f"Summary entries merged: {len(stringified_summary_list)}")

# merge all result summaries
appended_result_summaries = "\n".join(stringified_summary_list)
log_info(f"Total merged summary length: {len(appended_result_summaries)} chars")

# compile report from summaries
log_step(8, "Generate final research report", icon="📘")
research_report_prompt = RESEARCH_REPORT_PROMPT_TEMPLATE.format(
    research_summary=appended_result_summaries,
    user_question=question
)
research_report = llm.invoke(research_report_prompt)
log_llm_output("Final research report", research_report.content, max_chars=4000)

log_step(9, "Pipeline completed", icon="✅")
print("\n📦 [OUTPUT] stringified_summary_list")
print(stringified_summary_list)
print("\n📦 [OUTPUT] merged_result_summaries")
print(appended_result_summaries)
print("\n📦 [OUTPUT] research_report")
print(research_report.content)
