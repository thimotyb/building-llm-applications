from llm_models import get_llm
from utilities import to_obj
from prompts import (
    ASSISTANT_SELECTION_PROMPT_TEMPLATE,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from chain_logging import dump_tap, step_tap

assistant_instructions_chain = (
    {"user_question": RunnablePassthrough()}
    | RunnableLambda(
        step_tap(
            "STEP 1.2 - Build assistant selection payload",
            icon="🧠",
            details_fn=lambda x: f'Question: "{x.get("user_question", "")}"',
        )
    )
    | ASSISTANT_SELECTION_PROMPT_TEMPLATE
    | get_llm(provider="ollama")
    | RunnableLambda(
        dump_tap("LLM raw output (assistant selection)", icon="🤖", max_chars=3000)
    )
    | StrOutputParser()
    | RunnableLambda(
        dump_tap("LLM parsed text (assistant selection)", icon="📝", max_chars=3000)
    )
    | to_obj
    | RunnableLambda(
        dump_tap("Assistant selection JSON object", icon="📦", max_chars=3000)
    )
)
