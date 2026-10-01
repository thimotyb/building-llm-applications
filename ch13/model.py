"""Shared model-provider configuration for the chapter 13 ADK agents."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _required(*names: str) -> str:
    """Return the first configured value or explain what must be set."""

    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise RuntimeError(
        "Missing required environment variable. Expected one of: "
        + ", ".join(names)
        + "."
    )


def build_model() -> LiteLlm:
    """Build the model selected by LLM_PROVIDER."""

    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    if provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "gemma4:12b").strip() or "gemma4:12b"
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        # LiteLLM also reads this variable during provider/model inspection.
        os.environ["OLLAMA_API_BASE"] = base_url
        return LiteLlm(model=f"ollama_chat/{model_name}", api_base=base_url)

    if provider == "deepseek":
        model_name = _required("DEEPSEEK_MODEL")
        return LiteLlm(
            model=f"deepseek/{model_name}",
            api_key=_required("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            thinking={"type": os.getenv("DEEPSEEK_THINKING", "disabled")},
        )

    if provider == "gemini":
        model_name = _required("GEMINI_MODEL")
        return LiteLlm(
            model=f"gemini/{model_name}",
            api_key=_required("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        )

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER '{provider}'. Use 'ollama', 'deepseek', or 'gemini'."
    )
