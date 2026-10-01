"""Shared local-model configuration for the chapter 13 ADK agents."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def build_model() -> LiteLlm:
    """Build the course-default Ollama model without requiring a cloud API key."""

    model_name = os.getenv("OLLAMA_MODEL", "gemma4:e4b").strip() or "gemma4:e4b"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    os.environ.setdefault("OLLAMA_API_BASE", base_url)
    return LiteLlm(model=f"ollama_chat/{model_name}", api_base=base_url)
