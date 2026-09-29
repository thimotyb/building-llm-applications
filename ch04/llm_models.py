from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import dotenv_values, load_dotenv
from pathlib import Path
import os

load_dotenv()  # A
PROJECT_ROOT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class ChatDeepSeek(ChatOpenAI):
    """Use DeepSeek-compatible defaults on top of LangChain's OpenAI client."""

    def with_structured_output(
        self,
        schema=None,
        *,
        method="function_calling",
        include_raw=False,
        strict=None,
        **kwargs,
    ):
        return super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            **kwargs,
        )


def _announce_model(provider: str, model: str) -> None:
    """Show the effective provider and model without exposing credentials."""

    print(f"🤖 Using {provider} model: {model}")


def get_llm(
    provider: str | None = None,
    model_name: str | None = None,
    openai_api_key: str | None = None,
    gemini_api_key: str | None = None,
    deepseek_api_key: str | None = None,
):  # B
    env_data = (
        dotenv_values(PROJECT_ROOT_ENV_FILE)
        if PROJECT_ROOT_ENV_FILE.exists()
        else {}
    )
    selected_provider = (
        provider
        or os.getenv("LLM_PROVIDER")
        or env_data.get("LLM_PROVIDER")
        or "ollama"
    ).lower().strip()

    if selected_provider == "ollama":
        model = (
            model_name
            or os.getenv("OLLAMA_MODEL")
            or env_data.get("OLLAMA_MODEL", "gemma4:e2b")
        )
        _announce_model(selected_provider, model)
        return ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL")
            or env_data.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    if selected_provider == "openai":
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        api_key = api_key or env_data.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Pass openai_api_key=..., set env var, "
                "or add it to the project root .env file."
            )
        model = (
            model_name
            or os.getenv("OPENAI_MODEL")
            or env_data.get("OPENAI_MODEL", "gpt-5-nano")
        )
        _announce_model(selected_provider, model)
        return ChatOpenAI(
            openai_api_key=api_key,
            model_name=model,
        )

    if selected_provider == "gemini":
        api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        api_key = api_key or env_data.get("GOOGLE_API_KEY") or env_data.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key is missing. Pass gemini_api_key=..., set "
                "GOOGLE_API_KEY/GEMINI_API_KEY, or add one of them to the "
                "project root .env file."
            )
        model = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or env_data.get("GEMINI_MODEL", "gemini-flash-latest")
        )
        _announce_model(selected_provider, model)
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
        )

    if selected_provider == "deepseek":
        api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        api_key = api_key or env_data.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is missing. Pass deepseek_api_key=..., set "
                "the environment variable, or add it to the project root .env file."
            )
        thinking = os.getenv("DEEPSEEK_THINKING") or env_data.get(
            "DEEPSEEK_THINKING", "disabled"
        )
        model = (
            model_name
            or os.getenv("DEEPSEEK_MODEL")
            or env_data.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
        )
        _announce_model(selected_provider, model)
        return ChatDeepSeek(
            openai_api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL")
            or env_data.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model_name=model,
            extra_body={"thinking": {"type": thinking}},
        )

    raise ValueError(
        f"Unsupported provider '{selected_provider}'. Use 'ollama', 'openai', 'gemini', or 'deepseek'."
    )


# A Load environment variables from .env
# B Instantiate and return the selected chat model
