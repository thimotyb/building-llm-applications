from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import dotenv_values, load_dotenv
from pathlib import Path
import os

load_dotenv()  # A
PROJECT_ROOT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def get_llm(
    provider: str = "ollama",
    model_name: str | None = None,
    openai_api_key: str | None = None,
    gemini_api_key: str | None = None,
):  # B
    selected_provider = provider.lower().strip()

    if selected_provider == "ollama":
        return ChatOllama(model=model_name or "gemma4:e2b")

    if selected_provider == "openai":
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key and PROJECT_ROOT_ENV_FILE.exists():
            api_key = dotenv_values(PROJECT_ROOT_ENV_FILE).get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Pass openai_api_key=..., set env var, "
                "or add it to the project root .env file."
            )
        return ChatOpenAI(
            openai_api_key=api_key,
            model_name=model_name or "gpt-5-nano",
        )

    if selected_provider == "gemini":
        api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key and PROJECT_ROOT_ENV_FILE.exists():
            env_data = dotenv_values(PROJECT_ROOT_ENV_FILE)
            api_key = env_data.get("GOOGLE_API_KEY") or env_data.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key is missing. Pass gemini_api_key=..., set "
                "GOOGLE_API_KEY/GEMINI_API_KEY, or add one of them to the "
                "project root .env file."
            )
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model_name or "gemini-flash-latest",
        )

    raise ValueError(
        f"Unsupported provider '{provider}'. Use 'ollama', 'openai', or 'gemini'."
    )


# A Load environment variables from .env
# B Instantiate and return the selected chat model
