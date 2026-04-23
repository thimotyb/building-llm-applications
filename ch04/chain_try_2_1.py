from utilities import to_obj
from chain_2_1 import web_searches_chain
from chain_logging import log_dump, log_step

# test chain invocation
assistant_instruction_str = '{"assistant_type": "Tour guide assistant", "assistant_instructions": "You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights.", "user_question": "What can I see and do in the Spanish town of Astorga?"}'
assistant_instruction_dict = to_obj(assistant_instruction_str)
log_step("RUN chain_try_2_1", icon="🧪")
log_dump("Input assistant instruction JSON", assistant_instruction_dict, icon="📥")
web_searches_list = web_searches_chain.invoke(assistant_instruction_dict)
log_dump("Web searches list", web_searches_list, icon="📦")

# Result:
# [{'search_query': 'Things to do in Astorga Spain', 'user_question': 'What can I see and do in the Spanish town Astorga'}, {'search_query': 'Attractions in Astorga Spain', 'user_question': 'What can I see and do in the Spanish town Astorga'}, {'search_query': 'Historical sites in Astorga Spain', 'user_question': 'What can I see and do in the Spanish town Astorga'}]
