import os

from env_config import load_env
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_env()
# Load the shared environment configuration once at import time so every
# chapter 11 entrypoint resolves provider settings in the same way.


class GeminiEmbeddingsOneByOne:
    """Adapt Gemini embeddings to vector stores that expect one vector per text."""

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def embed_documents(self, texts):
        # Gemini can return one embedding per batch with some LangChain
        # versions. Chroma expects one embedding for each input document.
        return [
            self.embeddings.embed_documents([text], batch_size=1)[0]
            for text in texts
        ]

    def embed_query(self, text):
        return self.embeddings.embed_query(text)


def _provider() -> str:
    """Return the configured provider and validate the supported values."""

    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "ollama", "gemini"}:
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER '{provider}'. "
            "Use 'openai', 'ollama', or 'gemini'."
        )
    return provider


def _require_env(*names: str) -> str:
    """Return the first non-empty env var from a list of fallback names."""

    for name in names:
        value = os.getenv(name)
        if value:
            return value
    joined = ", ".join(names)
    raise RuntimeError(f"Missing required environment variable. Expected one of: {joined}.")


def _filter_none(values: dict):
    """Drop unsupported keyword arguments before instantiating a provider model."""

    return {key: value for key, value in values.items() if value is not None}


def get_embeddings_model():
    """
    Build the embeddings model for the configured provider.

    The chapter code can call this function without caring whether embeddings
    come from OpenAI, Ollama, or Gemini.
    """

    provider = _provider()
    if provider == "openai":
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_key=_require_env("OPENAI_API_KEY"),
        )
    if provider == "ollama":
        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview"),
        google_api_key=_require_env("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    return GeminiEmbeddingsOneByOne(embeddings)


def get_chat_model(
    *,
    model_name: str | None = None,
    temperature: float | None = None,
    use_responses_api: bool | None = None,
    use_previous_response_id: bool | None = None,
):
    """
    Build the chat model for the configured provider.

    OpenAI-specific flags are forwarded only when the selected provider
    supports them. Other providers ignore those options cleanly.
    """

    provider = _provider()

    if provider == "openai":
        return ChatOpenAI(
            **_filter_none(
                {
                    "model": model_name or os.getenv("OPENAI_MODEL", "gpt-5-nano"),
                    "temperature": temperature,
                    "use_responses_api": use_responses_api,
                    "use_previous_response_id": use_previous_response_id,
                    "openai_api_key": _require_env("OPENAI_API_KEY"),
                }
            )
        )

    if provider == "ollama":
        return ChatOllama(
            **_filter_none(
                {
                    "model": model_name or os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
                    "temperature": temperature,
                    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                }
            )
        )

    return ChatGoogleGenerativeAI(
        **_filter_none(
            {
                "model": model_name or os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
                "temperature": temperature,
                "google_api_key": _require_env("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            }
        )
    )
