from llm_models import get_llm
from prompts import (
    ASSISTANT_SELECTION_PROMPT_TEMPLATE,
)
from langchain_core.runnables import RunnableLambda
from chain_logging import dump_tap, step_tap

assistant_instructions_chain = (
    RunnableLambda(
        step_tap(
            "STEP 1.1 - Assistant selection input",
            icon="🧠",
            details_fn=lambda q: f'Question: "{q}"',
        )
    )
    | ASSISTANT_SELECTION_PROMPT_TEMPLATE
    | get_llm(provider="ollama")
    | RunnableLambda(
        dump_tap("LLM raw output (assistant selection)", icon="🤖", max_chars=3000)
    )
)
