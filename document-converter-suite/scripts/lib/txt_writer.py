#!/usr/bin/env python3
"""Plain text writer for document converter suite."""

from pathlib import Path
from typing import List


def write_txt_from_lines(output_path: Path, lines: List[str]) -> None:
    """
    Write plain text file from lines.

    Args:
        output_path: Path to output text file
        lines: List of text lines to write
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
