# Conversion matrix (best-effort)

This suite supports conversions between these four formats:

- PDF (`.pdf`)
- Word (`.docx`)
- PowerPoint (`.pptx`)
- Excel (`.xlsx`)

## What "best-effort" means

- **Text-first**: prioritize text and basic structure (titles, bullets, simple tables).
- **No OCR**: scanned PDFs are treated as images; extracted text may be empty.
- **No layout engine**: complex pagination, fonts, images, charts, and precise positioning are not preserved.

## Typical outputs

- **PDF → DOCX**: one section per PDF page, with a heading like "Page N".
- **PDF → PPTX**: one slide per PDF page; slide body contains extracted text as bullets.
- **PDF → XLSX**: one sheet with one line per row, grouped by PDF page.

- **DOCX → PDF**: exported text and tables into a simple PDF.
- **DOCX → PPTX**: headings start new slides; paragraphs become bullets.
- **DOCX → XLSX**: document text in a "Text" sheet; each DOCX table becomes its own sheet.

- **PPTX → DOCX**: one section per slide with bullets; PPTX tables exported as DOCX tables.
- **PPTX → PDF**: one page per slide with title/bullets; tables rendered as basic grids.
- **PPTX → XLSX**: a "Slides" sheet (title + combined text) plus one sheet per extracted table.

- **XLSX → DOCX**: one section per sheet; a bounded cell grid exported as a DOCX table.
- **XLSX → PPTX**: one slide per sheet; small sheets become slide tables, large sheets become bullet summaries.
- **XLSX → PDF**: one page per sheet with a bounded cell grid rendered as a table.
