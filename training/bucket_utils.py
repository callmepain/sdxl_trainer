from __future__ import annotations

from typing import Any, Optional


def normalize_bucket_key(value: Any) -> Optional[str]:
    if isinstance(value, str):
        token = value.strip().lower().replace(" ", "")
        if "x" in token:
            parts = token.split("x")
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                return f"{int(parts[0])}x{int(parts[1])}"
        return token
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return f"{int(value[0])}x{int(value[1])}"
        except (TypeError, ValueError):
            return None
    return None


def bucket_sort_key(value: str):
    try:
        cleaned = value.strip().lower().replace(" ", "")
        if "x" in cleaned:
            w, h = cleaned.split("x")
            return int(w), int(h)
    except Exception:
        pass
    return (float("inf"), value)


def bucket_key_from_target_size(target_size) -> Optional[str]:
    if target_size is None:
        return None
    try:
        if target_size.ndim == 2:
            height = int(target_size[0, 0])
            width = int(target_size[0, 1])
        elif target_size.ndim == 1 and target_size.numel() >= 2:
            height = int(target_size[0])
            width = int(target_size[1])
        else:
            return None
    except Exception:
        return None
    return normalize_bucket_key((width, height))
