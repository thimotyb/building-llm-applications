from chain_1_1 import assistant_instructions_chain
from chain_logging import log_dump, log_step

# test chain invocation
question = 'What can I see and do in the Spanish town of Astorga?'
log_step("RUN chain_try_1_1", details=f'Question: "{question}"', icon="🧪")

assistant_instructions = assistant_instructions_chain.invoke(question)
log_dump("Assistant instructions response", assistant_instructions, icon="📦")

# content='{\n    "assistant_type": "Tour guide assistant",\n    "assistant_instructions": "You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights.",\n    "user_question": "What can I see and do in the Spanish town of Astorga?"\n}' response_metadata={'token_usage': {'completion_tokens': 82, 'prompt_tokens': 432, 'total_tokens': 514}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_3bc1b5746c', 'finish_reason': 'stop', 'logprobs': None}
