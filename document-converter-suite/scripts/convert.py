#!/usr/bin/env python3
"""Document Converter Suite (single-file)

Convert between PDF, DOCX, PPTX, and XLSX with lightweight, dependency-minimal rules.

This intentionally does *not* attempt pixel-perfect layout conversion. It focuses on:
  - Extracting text and basic structure (headings, bullets, tables)
  - Rebuilding that content in the target format

Supported formats: .pdf, .docx, .pptx, .xlsx

Examples:
  python scripts/convert.py input.pdf --to docx
  python scripts/convert.py deck.pptx --to pdf --out deck.pdf
  python scripts/convert.py sheet.xlsx --to pptx --max-rows 40 --max-cols 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

from lib.conversion import convert_document
from lib.types import SUPPORTED_INPUT_EXTS, SUPPORTED_OUTPUT_EXTS


def _positive_int(value: str) -> int:
    try:
        i = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    if i <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return i


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert a document between PDF/DOCX/PPTX/XLSX")
    p.add_argument("input", type=str, help="Path to input file")
    p.add_argument(
        "--to",
        required=True,
        choices=sorted(SUPPORTED_OUTPUT_EXTS),
        help="Output format (extension without dot), e.g. pdf, docx, pptx, xlsx",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional output path. If omitted, write next to input with new extension.",
    )
    p.add_argument(
        "--max-pages",
        type=_positive_int,
        default=200,
        help="Safety cap for PDF page processing (default: 200)",
    )
    p.add_argument(
        "--max-chars",
        type=_positive_int,
        default=300000,
        help="Safety cap for extracted text size per document (default: 300000)",
    )
    p.add_argument(
        "--max-rows",
        type=_positive_int,
        default=200,
        help="Safety cap for spreadsheet rows per sheet (default: 200)",
    )
    p.add_argument(
        "--max-cols",
        type=_positive_int,
        default=50,
        help="Safety cap for spreadsheet columns per sheet (default: 50)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra progress output",
    )
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    if in_path.suffix.lower() not in SUPPORTED_INPUT_EXTS:
        raise SystemExit(
            f"Unsupported input extension '{in_path.suffix}'. Supported: {sorted(SUPPORTED_INPUT_EXTS)}"
        )

    out_ext = f".{args.to.lower().lstrip('.')}"
    if out_ext not in SUPPORTED_OUTPUT_EXTS:
        raise SystemExit(f"Unsupported output extension '{out_ext}'.")

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = in_path.with_suffix(out_ext)

    if out_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Output exists: {out_path}\nUse --overwrite or provide a different --out path."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    convert_document(
        input_path=in_path,
        output_path=out_path,
        max_pages=args.max_pages,
        max_chars=args.max_chars,
        max_rows=args.max_rows,
        max_cols=args.max_cols,
        verbose=args.verbose,
    )

    if args.verbose:
        print(f"✅ Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
