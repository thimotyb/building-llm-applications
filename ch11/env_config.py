import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

# Silenzia i warning di deprecazione di LangGraph per mantenere l'output pulito durante il corso
warnings.filterwarnings("ignore", message=".*create_react_agent has been moved to.*")
warnings.filterwarnings("ignore", message=".*create_react_agent is deprecated.*")

@dataclass(frozen=True)
class EnvSettings:
    """Normalized environment configuration shared across chapter 11."""

    project_root: Path
    env_path: Path | None
    langsmith_endpoint: str
    langsmith_project: str
    langsmith_tracing: str
    llm_provider: str
    openai_model: str
    openai_embedding_model: str
    ollama_base_url: str
    ollama_model: str
    ollama_embedding_model: str
    gemini_model: str
    gemini_embedding_model: str


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file without depending on python-dotenv."""

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _find_project_root() -> Path:
    """Walk upward until the repository root is found."""

    current = Path(__file__).resolve().parent
    for directory in [current, *current.parents]:
        if (directory / ".env.example").exists() and (directory / ".git").exists():
            return directory
    return Path(__file__).resolve().parents[1]


def _find_env_file(project_root: Path, filename: str = ".env") -> Path | None:
    """Return the root-level env file when present."""

    env_path = project_root / filename
    return env_path if env_path.exists() else None


def load_env(filename: str = ".env", override: bool = True) -> EnvSettings:
    """
    Load the project-root .env file and apply shared defaults.

    This is the single entrypoint used by the chapter 11 examples so all
    variants resolve environment variables in the same way.
    """

    project_root = _find_project_root()
    env_path = _find_env_file(project_root, filename)
    if env_path is not None:
        for key, value in _parse_env_file(env_path).items():
            if override or key not in os.environ:
                os.environ[key] = value

    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGSMITH_PROJECT", "ch11")
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")

    os.environ.setdefault("LLM_PROVIDER", "openai")
    os.environ.setdefault("OPENAI_MODEL", "gpt-5-nano")
    os.environ.setdefault("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
    os.environ.setdefault("OLLAMA_MODEL", "gemma4:e2b")
    os.environ.setdefault("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    os.environ.setdefault("GEMINI_MODEL", "gemini-flash-latest")
    os.environ.setdefault("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")

    # Return a structured snapshot so callers can inspect the effective setup
    # without reading directly from os.environ.
    return EnvSettings(
        project_root=project_root,
        env_path=env_path,
        langsmith_endpoint=os.environ["LANGSMITH_ENDPOINT"],
        langsmith_project=os.environ["LANGSMITH_PROJECT"],
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false"),
        llm_provider=os.environ["LLM_PROVIDER"],
        openai_model=os.environ["OPENAI_MODEL"],
        openai_embedding_model=os.environ["OPENAI_EMBEDDING_MODEL"],
        ollama_base_url=os.environ["OLLAMA_BASE_URL"],
        ollama_model=os.environ["OLLAMA_MODEL"],
        ollama_embedding_model=os.environ["OLLAMA_EMBEDDING_MODEL"],
        gemini_model=os.environ["GEMINI_MODEL"],
        gemini_embedding_model=os.environ["GEMINI_EMBEDDING_MODEL"],
    )


def get_env_dict() -> dict[str, str]:
    """Expose the normalized settings as a plain dict for logging or debugging."""

    settings = load_env()
    data = asdict(settings)
    data["project_root"] = str(data["project_root"])
    data["env_path"] = str(data["env_path"]) if data["env_path"] else ""
    return data
