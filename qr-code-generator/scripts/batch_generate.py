#!/usr/bin/env python3
"""
Batch-generate QR codes from a CSV with columns:
  id,url,label

Example:
  python batch_generate.py --csv inputs.csv --outdir /mnt/data/qrs --format svg --error H

Requires: qrcode, pillow (for PNG)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from generate_qr import generate_qr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV file with id,url,label columns.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--format", default="svg", choices=["png","svg","both"])
    ap.add_argument("--error", default="H", choices=["L","M","Q","H"])
    ap.add_argument("--show-url", action="store_true")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            _id = (row.get("id") or "").strip()
            url = (row.get("url") or "").strip()
            label = (row.get("label") or "").strip()
            if not _id or not url:
                continue

            png = os.path.join(args.outdir, f"{_id}.png") if args.format in ("png","both") else ""
            svg = os.path.join(args.outdir, f"{_id}.svg") if args.format in ("svg","both") else ""

            try:
                final_url = generate_qr(
                    url,
                    png=png,
                    svg=svg,
                    error=args.error,
                    caption_label=label or None,
                    show_url=args.show_url,
                )
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                raise SystemExit(1) from exc
            print(final_url)

if __name__ == "__main__":
    main()
