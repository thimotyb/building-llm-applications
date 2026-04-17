from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any, TypedDict, Optional
from dotenv import dotenv_values, load_dotenv
from pathlib import Path
import os

PROJECT_ROOT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(PROJECT_ROOT_ENV_FILE)


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value

    if PROJECT_ROOT_ENV_FILE.exists():
        env_data = dotenv_values(PROJECT_ROOT_ENV_FILE)
        for name in names:
            value = env_data.get(name)
            if value:
                return value

    return None


def get_llm(
    provider: str | None = None,
    model_name: str | None = None,
    openai_api_key: str | None = None,
    gemini_api_key: str | None = None,
):
    selected_provider = (provider or _env_value("LLM_PROVIDER") or "ollama").lower().strip()

    if selected_provider == "ollama":
        model = model_name or _env_value("LLM_MODEL", "OLLAMA_MODEL") or "gemma4:e2b"
        return ChatOllama(model=model)

    if selected_provider == "openai":
        api_key = openai_api_key or _env_value("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Pass openai_api_key=..., set env var, "
                "or add it to the project root .env file."
            )
        model = model_name or _env_value("LLM_MODEL", "OPENAI_MODEL") or "gpt-5-nano"
        return ChatOpenAI(
            openai_api_key=api_key,
            model_name=model,
        )

    if selected_provider == "gemini":
        api_key = gemini_api_key or _env_value("GOOGLE_API_KEY", "GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key is missing. Pass gemini_api_key=..., set "
                "GOOGLE_API_KEY/GEMINI_API_KEY, or add one of them to the "
                "project root .env file."
            )
        model = model_name or _env_value("LLM_MODEL", "GEMINI_MODEL") or "gemini-flash-latest"
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
        )

    raise ValueError(
        f"Unsupported provider '{provider}'. Use 'ollama', 'openai', or 'gemini'."
    )

# Define typed dictionaries for state handling
class AssistantInfo(TypedDict):
    assistant_type: str
    assistant_instructions: str
    user_question: str

class SearchQuery(TypedDict):
    search_query: str
    user_question: str

class SearchResult(TypedDict):
    result_url: str
    search_query: str
    user_question: str
    is_fallback: Optional[bool]

class SearchSummary(TypedDict):
    summary: str
    result_url: str
    user_question: str
    is_fallback: Optional[bool]

class ResearchReport(TypedDict):
    report: str

# Graph state
class ResearchState(TypedDict):
    user_question: str
    assistant_info: Optional[AssistantInfo]
    search_queries: Optional[List[SearchQuery]]
    search_results: Optional[List[SearchResult]]
    search_summaries: Optional[List[SearchSummary]]
    research_summary: Optional[str]
    final_report: Optional[str]
    used_fallback_search: Optional[bool]
    relevance_evaluation: Optional[Dict[str, Any]]
    should_regenerate_queries: Optional[bool]
    iteration_count: Optional[int]
