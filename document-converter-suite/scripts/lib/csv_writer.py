#!/usr/bin/env python3
"""CSV writer for document converter suite."""

import csv
from pathlib import Path
from typing import List, Optional


def write_csv_from_rows(
    output_path: Path,
    rows: List[List[str]],
    headers: Optional[List[str]] = None
) -> None:
    """
    Write CSV file from rows with optional headers.

    Args:
        output_path: Path to output CSV file
        rows: List of rows (each row is a list of strings)
        headers: Optional header row to prepend
    """
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        if headers:
            writer.writerow(headers)

        writer.writerows(rows)
