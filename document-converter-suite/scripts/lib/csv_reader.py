#!/usr/bin/env python3
"""CSV reader for document converter suite."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class CsvContent:
    """CSV document content."""
    rows: List[List[str]]
    headers: List[str]
    has_header: bool


def read_csv_content(
    path: Path,
    max_rows: int = 200,
    max_cols: int = 50,
    detect_header: bool = True
) -> CsvContent:
    """
    Read CSV file with optional header detection.

    Args:
        path: Path to CSV file
        max_rows: Maximum rows to read (default 200)
        max_cols: Maximum columns per row (default 50)
        detect_header: Whether to auto-detect headers (default True)

    Returns:
        CsvContent with rows, headers, and header detection flag
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        # Try to detect dialect and headers
        sniffer = csv.Sniffer()
        sample = f.read(2048)
        f.seek(0)

        try:
            dialect = sniffer.sniff(sample)
            has_header = detect_header and sniffer.has_header(sample)
        except:
            # Fallback to excel dialect if sniffing fails
            dialect = csv.excel
            has_header = False

        reader = csv.reader(f, dialect=dialect)
        rows = []
        headers = []

        for i, row in enumerate(reader):
            if i >= max_rows:
                break

            # Trim to max columns
            trimmed = row[:max_cols] if len(row) > max_cols else row

            if i == 0 and has_header:
                headers = trimmed
            rows.append(trimmed)

    return CsvContent(rows=rows, headers=headers, has_header=has_header)
