from __future__ import annotations

import re
from typing import Iterable, List


def clamp_text(text: str, max_chars: int) -> str:
    """Clamp text length to a max char count, preserving a readable ending."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n…(truncated)…\n"


def nonempty_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def guess_bullets(text: str, max_lines: int = 12, max_line_chars: int = 180) -> List[str]:
    """Turn a text blob into bullet-ish lines.

    - Prefer existing line breaks
    - Strip common bullet markers
    - Clamp length to avoid runaway text boxes
    """
    lines = nonempty_lines(text)
    cleaned: List[str] = []
    for ln in lines:
        ln = re.sub(r"^[-•\u2022\*\s]+", "", ln).strip()
        if not ln:
            continue
        if len(ln) > max_line_chars:
            ln = ln[: max_line_chars - 1] + "…"
        cleaned.append(ln)
        if len(cleaned) >= max_lines:
            break
    if not cleaned and text.strip():
        cleaned = [text.strip()[: max_line_chars - 1] + "…" if len(text.strip()) > max_line_chars else text.strip()]
    return cleaned


def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    if chunk_size <= 0:
        return [items]
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def safe_cell_value(v) -> str:
    if v is None:
        return ""
    try:
        return str(v)
    except Exception:
        return ""
