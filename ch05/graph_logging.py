import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict

_LOG_FILE_HANDLE = None
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr


class _TeeWriter:
    def __init__(self, *writers):
        self._writers = writers

    def write(self, data):
        for writer in self._writers:
            writer.write(data)
            writer.flush()

    def flush(self):
        for writer in self._writers:
            writer.flush()


def configure_file_logging(log_file: str | Path, truncate: bool = True):
    global _LOG_FILE_HANDLE

    if _LOG_FILE_HANDLE is not None:
        return Path(log_file)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if truncate else "a"
    _LOG_FILE_HANDLE = log_path.open(mode, encoding="utf-8", buffering=1)
    sys.stdout = _TeeWriter(_ORIGINAL_STDOUT, _LOG_FILE_HANDLE)
    sys.stderr = _TeeWriter(_ORIGINAL_STDERR, _LOG_FILE_HANDLE)
    return log_path


def _hr():
    print("\n" + "=" * 80)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(_as_text(_strip_signature_data(item)))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return _as_text(_strip_signature_data(content))


def _strip_signature_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_signature_data(item)
            for key, item in value.items()
            if key
            not in {
                "additional_kwargs",
                "extras",
                "invalid_tool_calls",
                "response_metadata",
                "signature",
                "tool_calls",
                "usage_metadata",
            }
        }
    if isinstance(value, list):
        return [_strip_signature_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_signature_data(item) for item in value)
    return value


def _as_text(value: Any) -> str:
    if value is None:
        return "<none>"
    if hasattr(value, "content"):
        return message_text(value)
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(_strip_signature_data(value), ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(value)
    return str(value)


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [truncated]"
    return text if text else "<empty>"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower().strip() in {"1", "true", "yes", "on"}


def log_step(title: str, details: str | None = None, icon: str = "🚀"):
    _hr()
    print(f"{icon} {title}")
    if details:
        print(f"  {details}")


def log_info(message: str, icon: str = "•"):
    print(f"  {icon} {message}")


def log_dump(title: str, value: Any, icon: str = "📦", max_chars: int = 2500):
    _hr()
    print(f"{icon} {title}")
    print(_truncate(_as_text(value), max_chars=max_chars))


def _preview_text(value: str, max_chars: int = 180) -> str:
    value = " ".join(value.split())
    if len(value) > max_chars:
        return value[:max_chars] + "..."
    return value


def _compact_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return {
            "type": "str",
            "chars": len(value),
            "preview": _preview_text(value),
        }
    if isinstance(value, list):
        preview = _compact_value(value[0]) if value else None
        return {
            "type": "list",
            "count": len(value),
            "first": preview,
        }
    if isinstance(value, dict):
        compact = {}
        for key, item in value.items():
            if key in {"assistant_info", "relevance_evaluation"}:
                compact[key] = _strip_signature_data(item)
            else:
                compact[key] = _compact_value(item)
        return compact
    return _preview_text(str(value))


def log_research_state(title: str, state: Dict[str, Any], icon: str = "🧭"):
    if _env_flag("CH05_LOG_FULL_STATE", default=False):
        log_dump(title, state, icon=icon, max_chars=5000)
    else:
        log_dump(f"{title} (compact)", _compact_value(state), icon=icon, max_chars=3500)


def log_compact_dump(title: str, value: Any, icon: str = "📦", max_chars: int = 2500):
    log_dump(f"{title} (compact)", _compact_value(value), icon=icon, max_chars=max_chars)


def log_node_output(node_name: str, output: Dict[str, Any]):
    if _env_flag("CH05_LOG_FULL_STATE", default=False):
        log_dump(f"Node output ({node_name})", output, icon="📤", max_chars=5000)
    else:
        log_dump(f"Node output ({node_name}) (compact)", _compact_value(output), icon="📤", max_chars=3500)


def message_text(message: Any) -> str:
    if hasattr(message, "content"):
        return _content_text(message.content)
    return _as_text(message)


def invoke_llm(llm: Any, prompt: str, label: str, max_prompt_chars: int = 3500, max_output_chars: int = 3500):
    log_llm_dumps = _env_flag("CH05_LOG_LLM_DUMPS", default=False)
    if log_llm_dumps:
        log_dump(f"LLM input ({label})", prompt, icon="📥", max_chars=max_prompt_chars)
    else:
        log_info(f"LLM call: {label}", icon="🤖")

    response = llm.invoke(prompt)

    if log_llm_dumps:
        log_dump(f"LLM raw output ({label})", response, icon="🤖", max_chars=max_output_chars)
        if hasattr(response, "content"):
            log_dump(f"LLM parsed text ({label})", message_text(response), icon="📝", max_chars=max_output_chars)
    elif hasattr(response, "content"):
        log_info(f"LLM response chars ({label}): {len(message_text(response))}", icon="📝")

    return response


def log_node(node_name: str, node_fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
    def _wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        log_step(f"ENTER LangGraph node: {node_name}", icon="➡️")
        log_research_state(f"ResearchState input ({node_name})", state, icon="🧭")
        output = node_fn(state)
        log_node_output(node_name, output)
        log_step(f"EXIT LangGraph node: {node_name}", icon="⬅️")
        return output

    return _wrapped
