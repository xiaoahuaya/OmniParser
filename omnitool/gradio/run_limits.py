from __future__ import annotations


def normalize_max_seconds(value: object) -> int | None:
    try:
        if value is None:
            return None
        seconds = int(value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return seconds


def max_seconds_label(value: object) -> str:
    normalized = normalize_max_seconds(value)
    if normalized is None:
        return "不限时"
    return f"{normalized}s"
