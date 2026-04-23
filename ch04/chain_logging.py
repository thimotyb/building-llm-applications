import json


def _hr():
    print("\n" + "=" * 80)


def _as_text(value):
    if value is None:
        return "<none>"
    if hasattr(value, "content"):
        return str(value.content)
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            return str(value)
    return str(value)


def log_step(title, details=None, icon="🚀"):
    _hr()
    print(f"{icon} {title}")
    if details:
        print(f"  {details}")


def log_info(message, icon="•"):
    print(f"  {icon} {message}")


def log_dump(title, value, icon="📦", max_chars=2500):
    _hr()
    print(f"{icon} {title}")
    text = _as_text(value).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    print(text if text else "<empty>")


def step_tap(title, icon="🚀", details_fn=None):
    def _inner(x):
        details = details_fn(x) if details_fn else None
        log_step(title=title, details=details, icon=icon)
        return x

    return _inner


def dump_tap(title, icon="📦", max_chars=2500):
    def _inner(x):
        log_dump(title=title, value=x, icon=icon, max_chars=max_chars)
        return x

    return _inner
