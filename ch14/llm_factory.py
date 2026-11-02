"""Provider-agnostic LLM access for chapter 14 (Workshop 2), built on LiteLLM.

One interface, several providers: only the model string changes
(`ollama_chat/...`, `deepseek/...`, `gemini/...`, `openai/...`).

Environment (same names as the other chapters, read from the repo-root `.env`):
    LLM_PROVIDER      ollama | deepseek | gemini | openai   (default: ollama)
    OLLAMA_MODEL      default gemma4:e4b
    OLLAMA_BASE_URL   default http://127.0.0.1:11434
    DEEPSEEK_MODEL    default deepseek-v4-pro
    GEMINI_MODEL      default gemini-2.5-flash
    OPENAI_MODEL      default gpt-4o-mini
    LLM_FALLBACK_OLLAMA=1  fall back to Ollama if the cloud key is missing

The judge can use a different provider than the worker: pass `role="judge"`
and set JUDGE_PROVIDER / JUDGE_MODEL (they default to the worker settings).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

PROVIDERS = ("ollama", "deepseek", "gemini", "openai")

_DEFAULT_MODELS = {
    "ollama": "gemma4:e4b",
    "deepseek": "deepseek-v4-pro",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
}
_KEY_VARS = {
    "deepseek": ("DEEPSEEK_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "ollama": (),
}
_LITELLM_PREFIX = {
    "ollama": "ollama_chat",
    "deepseek": "deepseek",
    "gemini": "gemini",
    "openai": "openai",
}


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to call one model through LiteLLM."""

    provider: str
    model: str  # LiteLLM string, e.g. "ollama_chat/gemma4:e4b"
    api_base: str | None = None
    api_key: str | None = None

    @property
    def label(self) -> str:
        return self.model


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _api_key(provider: str) -> str | None:
    for name in _KEY_VARS[provider]:
        value = _env(name)
        if value:
            return value
    return None


def resolve_model(
    provider: str | None = None,
    model: str | None = None,
    role: str = "worker",
) -> ModelSpec:
    """Resolve provider/model from arguments, then role env vars, then defaults."""

    prefix = "JUDGE_" if role == "judge" else ""
    provider = (provider or _env(f"{prefix}PROVIDER") or _env("LLM_PROVIDER", "ollama")).lower()
    if provider not in PROVIDERS:
        raise RuntimeError(f"Unsupported provider '{provider}'. Use one of: {', '.join(PROVIDERS)}.")

    if provider != "ollama" and _KEY_VARS[provider] and not _api_key(provider):
        if _env("LLM_FALLBACK_OLLAMA") == "1":
            provider, model = "ollama", None
        else:
            names = " or ".join(_KEY_VARS[provider])
            raise RuntimeError(
                f"Missing {names} for provider '{provider}'. "
                "Set it in .env or export LLM_FALLBACK_OLLAMA=1 to use Ollama."
            )

    name = model or _env(f"{prefix}MODEL") or _env(f"{provider.upper()}_MODEL", _DEFAULT_MODELS[provider])
    if provider == "ollama":
        base = _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        return ModelSpec(provider, f"ollama_chat/{name}", api_base=base)
    return ModelSpec(provider, f"{_LITELLM_PREFIX[provider]}/{name}", api_key=_api_key(provider))


def complete(
    messages: list[dict[str, str]],
    spec: ModelSpec | None = None,
    *,
    temperature: float = 0.0,
    response_format: Any = None,
    **kwargs: Any,
):
    """Call the model via LiteLLM and return the raw LiteLLM response."""

    import litellm  # imported lazily so config tests need no network stack

    spec = spec or resolve_model()
    params: dict[str, Any] = {"model": spec.model, "messages": messages, "temperature": temperature}
    if spec.api_base:
        params["api_base"] = spec.api_base
    if spec.api_key:
        params["api_key"] = spec.api_key
    if response_format is not None:
        params["response_format"] = response_format
    params.update(kwargs)
    return litellm.completion(**params)


def chat(prompt: str, spec: ModelSpec | None = None, system: str | None = None, **kwargs: Any) -> str:
    """Convenience wrapper: one prompt in, text out."""

    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    response = complete(messages, spec, **kwargs)
    return response.choices[0].message.content or ""
