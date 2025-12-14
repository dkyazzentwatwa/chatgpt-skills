from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader

from .utils import clamp_text


@dataclass
class PdfText:
    pages: List[str]


def read_pdf_text(path: Path, max_pages: int = 200, max_chars: int = 300000) -> PdfText:
    """Extract text from a PDF using pypdf.

    Limitations:
      - No OCR
      - Text extraction quality depends on the PDF (embedded text vs scanned)
    """
    reader = PdfReader(str(path))
    pages: List[str] = []

    page_count = min(len(reader.pages), max_pages)
    remaining = max_chars

    for i in range(page_count):
        page = reader.pages[i]
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        # clamp global budget across all pages
        if remaining <= 0:
            pages.append("…(truncated)…")
            break
        if len(text) > remaining:
            text = text[:remaining]
        remaining -= len(text)
        pages.append(text)

    # If we hit the page cap and there are more pages, signal it
    if len(reader.pages) > page_count:
        pages.append(f"…(stopped at {page_count} pages of {len(reader.pages)})…")

    return PdfText(pages=[clamp_text(p, 100000) for p in pages])
