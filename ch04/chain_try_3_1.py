from utilities import to_obj
from chain_3_1 import search_result_urls_chain
from chain_logging import log_dump, log_step

# test chain invocation
web_search_str = '{"search_query": "Astorga Spain attractions", "user_question": "What can I see and do in the Spanish town of Astorga?"}'
web_search_dict = to_obj(web_search_str)
log_step("RUN chain_try_3_1", icon="🧪")
log_dump("Input web search JSON", web_search_dict, icon="📥")
result_urls_list = search_result_urls_chain.invoke(web_search_dict)
log_dump("Result URLs list", result_urls_list, icon="📦")

# Result:
# [{'result_url': 'https://loveatfirstadventure.com/astorga-spain/', 'search_query': 'Astorga Spain attractions', 'user_question': 'What can I see and do in the Spanish town Astorga?'}, {'result_url': 'https://igotospain.com/one-day-in-astorga-on-the-camino-de-santiago/', 'search_query': 'Astorga Spain attractions', 'user_question': 'What can I see and do in the Spanish town Astorga'}, {'result_url': 'https://citiesandattractions.com/spain/astorga-spain-uncovering-the-jewels-of-a-hidden-spanish-gem/', 'search_query': 'Astorga Spain attractions', 'user_question': 'What can I see and do in the Spanish town Astorga'}] 
