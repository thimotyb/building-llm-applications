from chain_1_2 import assistant_instructions_chain
from chain_logging import log_dump, log_step

# test chain invocation
question = 'What can I see and do in the Spanish town of Astorga?'
log_step("RUN chain_try_1_2", details=f'Question: "{question}"', icon="🧪")

assistant_instructions_dict = assistant_instructions_chain.invoke(question)
log_dump("Assistant instructions JSON", assistant_instructions_dict, icon="📦")

# Result:
# {'assistant_type': 'Tour guide assistant', 'assistant_instructions': 'You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights.', 'user_question': 'What can I see and do in the Spanish town Astorga'}
