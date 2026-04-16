from llm_models import get_llm
from utilities import to_obj
from prompts import (
    WEB_SEARCH_PROMPT_TEMPLATE
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from chain_logging import dump_tap, step_tap

NUM_SEARCH_QUERIES = 2

web_searches_chain = (
    RunnableLambda(
        step_tap(
            "STEP 2.1 - Generate web search queries",
            icon="🔎",
            details_fn=lambda x: (
                f'Question: "{x.get("user_question", "")}" | '
                f'Assistant: "{x.get("assistant_type", "n/a")}"'
            ),
        )
    )
    | RunnableLambda(lambda x:
        {
            'assistant_instructions': x['assistant_instructions'],
            'num_search_queries': NUM_SEARCH_QUERIES,
            'user_question': x['user_question']
        }
    )
    | WEB_SEARCH_PROMPT_TEMPLATE
    | get_llm(provider="ollama")
    | RunnableLambda(
        dump_tap("LLM raw output (web search queries)", icon="🤖", max_chars=3000)
    )
    | StrOutputParser()
    | RunnableLambda(
        dump_tap("LLM parsed text (web search queries)", icon="📝", max_chars=3000)
    )
    | to_obj
    | RunnableLambda(
        dump_tap("Web search queries JSON list", icon="📦", max_chars=3000)
    )
)
