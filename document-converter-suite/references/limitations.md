# Limitations and gotchas

## PDF realities

- **Scanned PDFs**: `pypdf` cannot OCR. If the PDF is basically photos of pages, extracted text will be empty.
- **Weird PDFs**: some PDFs have text split into individual positioned glyphs; extracted text can look scrambled.

## Office format realities

- **DOCX → PDF**: without a full layout engine, the PDF output is a clean-but-simple export (paragraph text + basic tables).
- **PPTX → PDF**: slide visuals (images, complex shapes, animations) are not rendered; the exporter uses slide titles + extracted text + simple tables.
- **XLSX → * **: only a bounded grid of values is exported (defaults: 200 rows × 50 cols). Formulas are exported as their computed values.

## Things to be explicit about

- Ask for the user's priority: "make it editable" vs "make it visually identical".
- If the user needs visually identical output, suggest using a renderer (e.g., LibreOffice/PowerPoint) outside the sandbox.
