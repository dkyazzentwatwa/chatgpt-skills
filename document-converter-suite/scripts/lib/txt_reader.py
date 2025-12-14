#!/usr/bin/env python3
"""Plain text reader for document converter suite."""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TxtContent:
    """Plain text document content."""
    lines: List[str]


def read_txt_content(path: Path, max_chars: int = 300000) -> TxtContent:
    """
    Read plain text file, preserving line structure.

    Args:
        path: Path to text file
        max_chars: Maximum characters to read (default 300,000)

    Returns:
        TxtContent with lines
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read(max_chars)

    return TxtContent(lines=text.splitlines())
