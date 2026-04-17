import json
from typing import Any, Callable, Dict


def _hr():
    print("\n" + "=" * 80)


def _as_text(value: Any) -> str:
    if value is None:
        return "<none>"
    if hasattr(value, "content"):
        return str(value.content)
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(value)
    return str(value)


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [truncated]"
    return text if text else "<empty>"


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


def log_research_state(title: str, state: Dict[str, Any], icon: str = "🧭"):
    log_dump(title, state, icon=icon, max_chars=5000)


def invoke_llm(llm: Any, prompt: str, label: str, max_prompt_chars: int = 3500, max_output_chars: int = 3500):
    log_dump(f"LLM input ({label})", prompt, icon="📥", max_chars=max_prompt_chars)
    response = llm.invoke(prompt)
    log_dump(f"LLM raw output ({label})", response, icon="🤖", max_chars=max_output_chars)
    if hasattr(response, "content"):
        log_dump(f"LLM parsed text ({label})", response.content, icon="📝", max_chars=max_output_chars)
    return response


def log_node(node_name: str, node_fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
    def _wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        log_step(f"ENTER LangGraph node: {node_name}", icon="➡️")
        log_research_state(f"ResearchState input ({node_name})", state, icon="🧭")
        output = node_fn(state)
        log_dump(f"Node output ({node_name})", output, icon="📤", max_chars=5000)
        log_step(f"EXIT LangGraph node: {node_name}", icon="⬅️")
        return output

    return _wrapped
