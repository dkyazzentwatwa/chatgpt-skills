#!/usr/bin/env python3
"""Document Converter Suite (batch)

Batch-convert files in a directory (optionally recursive).

Notes:
  - Uses the same best-effort, structure-first conversions as scripts/convert.py
  - Skips files already in the target extension

Examples:
  python scripts/batch_convert.py ./inbox --to pdf --recursive
  python scripts/batch_convert.py ./inbox --to docx --outdir ./out --overwrite
  python scripts/batch_convert.py ./inbox --to xlsx --pattern "*.pptx"
"""

from __future__ import annotations

import argparse
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
    p = argparse.ArgumentParser(description="Batch convert documents in a folder")
    p.add_argument("indir", type=str, help="Input directory")
    p.add_argument(
        "--to",
        required=True,
        choices=sorted({e.lstrip(".") for e in SUPPORTED_OUTPUT_EXTS}),
        help="Output format: pdf/docx/pptx/xlsx",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory. Defaults to writing next to each input file.",
    )
    p.add_argument("--recursive", action="store_true", help="Search subdirectories")
    p.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern to filter input files (default: '*'). Example: '*.pdf'",
    )
    p.add_argument(
        "--flatten",
        action="store_true",
        help="If --outdir is set, write all outputs directly into that directory (no subfolders).",
    )
    p.add_argument("--max-pages", type=_positive_int, default=200)
    p.add_argument("--max-chars", type=_positive_int, default=300000)
    p.add_argument("--max-rows", type=_positive_int, default=200)
    p.add_argument("--max-cols", type=_positive_int, default=50)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def iter_files(indir: Path, recursive: bool, pattern: str) -> Iterable[Path]:
    if recursive:
        for p in indir.rglob(pattern):
            if p.is_file():
                yield p
    else:
        for p in indir.glob(pattern):
            if p.is_file():
                yield p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    indir = Path(args.indir).expanduser().resolve()
    if not indir.exists() or not indir.is_dir():
        raise SystemExit(f"Input directory not found: {indir}")

    out_ext = f".{args.to.lower().lstrip('.')}"
    if out_ext not in SUPPORTED_OUTPUT_EXTS:
        raise SystemExit(f"Unsupported output extension: {out_ext}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    failed = 0

    for in_path in iter_files(indir, args.recursive, args.pattern):
        if in_path.suffix.lower() not in SUPPORTED_INPUT_EXTS:
            continue
        if in_path.suffix.lower() == out_ext:
            skipped += 1
            continue

        if outdir:
            if args.flatten:
                out_path = outdir / (in_path.stem + out_ext)
            else:
                rel = in_path.relative_to(indir)
                out_path = (outdir / rel).with_suffix(out_ext)
                out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = in_path.with_suffix(out_ext)

        if out_path.exists() and not args.overwrite:
            skipped += 1
            if args.verbose:
                print(f"↷ skip (exists): {out_path}")
            continue

        try:
            convert_document(
                input_path=in_path,
                output_path=out_path,
                max_pages=args.max_pages,
                max_chars=args.max_chars,
                max_rows=args.max_rows,
                max_cols=args.max_cols,
                verbose=args.verbose,
            )
            converted += 1
            if args.verbose:
                print(f"✓ {in_path.name} -> {out_path.name}")
        except Exception as e:
            failed += 1
            if args.verbose:
                print(f"✗ failed: {in_path} ({e})")

    print(f"Done. converted={converted} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
