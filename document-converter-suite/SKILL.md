---
name: document-converter-suite
description: Convert between PDF, DOCX, PPTX, and XLSX in a restricted Python environment. Use for best-effort, text-first conversions, batch conversions, and extracting slide/table/text content into editable formats.
---

# Document Converter Suite

## Overview

Provide a best-effort conversion workflow between **PDF**, **Word (DOCX)**, **PowerPoint (PPTX)**, and **Excel (XLSX)** using `pypdf`, `python-docx`, `python-pptx`, `openpyxl`, and `reportlab`.

Prefer **reliable extraction + rebuild** (text, headings, bullets, basic tables) over pixel-perfect layout.

## When to use

Use when the request involves:

- Converting a file between **.pdf / .docx / .pptx / .xlsx**
- Making a document **more editable** by moving its content into Office formats
- Exporting slide text or spreadsheet cell grids to a different format
- Batch-converting a folder of mixed documents

Avoid promising visual fidelity. Emphasize that output is **clean and structured**, not identical.

## Workflow decision tree

1. **Identify input and desired output** (extensions matter).
2. **Classify the user's goal**:
   - **Editable content** → proceed with this suite.
   - **Visually identical rendering** → explain limitations; suggest external rendering tools.
3. **Pick conversion mode**:
   - Single file → run `scripts/convert.py`.
   - Folder/batch → run `scripts/batch_convert.py`.
4. **Tune safety caps** if needed:
   - PDF: `--max-pages`, `--max-chars`
   - XLSX: `--max-rows`, `--max-cols`
5. **Run conversion**, then sanity-check output size and structure.
6. **Iterate** (e.g., increase max rows/cols, split large docs, or choose a different target format).

## Quick start

### Single-file conversion

Run:

```bash
python scripts/convert.py <input-file> --to <pdf|docx|pptx|xlsx>
```

Examples:

```bash
python scripts/convert.py report.pdf --to docx
python scripts/convert.py deck.pptx --to pdf --out deck_export.pdf
python scripts/convert.py data.xlsx --to pptx --max-rows 40 --max-cols 12
```

### Batch conversion

Run:

```bash
python scripts/batch_convert.py <input-dir> --to <pdf|docx|pptx|xlsx>
```

Examples:

```bash
python scripts/batch_convert.py ./inbox --to docx --recursive
python scripts/batch_convert.py ./inbox --to pdf --outdir ./out --recursive --overwrite
python scripts/batch_convert.py ./inbox --to xlsx --pattern "*.pptx"
```

## Conversion behavior

Follow these defaults (and say them out loud if the user might be expecting magic):

- **PDF → (DOCX/PPTX/XLSX)**: extract text with `pypdf`; no OCR; each page becomes a section/slide block.
- **DOCX → PDF**: rebuild a readable PDF using `reportlab` (text + basic tables).
- **PPTX → (DOCX/PDF/XLSX)**: export slide titles + text frames; export simple tables when present.
- **XLSX → (DOCX/PPTX/PDF)**: export a bounded value grid per sheet (defaults: 200×50) using `openpyxl`.

Load extra detail from:

- `references/conversion_matrix.md`
- `references/limitations.md`

## Guardrails and honesty rules

- State "best-effort" explicitly for any conversion request.
- Do not claim formatting fidelity (fonts, spacing, images, charts, animations).
- Call out scanned PDFs as a likely failure mode (no OCR).
- For giant spreadsheets, prefer increasing caps gradually and/or limiting to specific sheets (if user provides intent).

## Bundled scripts

- `scripts/convert.py`: single-file CLI converter
- `scripts/batch_convert.py`: batch converter for directories
- `scripts/lib/*`: internal readers/writers and conversion orchestration
