# Conversion Matrix (Best-Effort)

This suite supports conversions between **8 document formats**:

**Office Formats**: PDF (`.pdf`), Word (`.docx`), PowerPoint (`.pptx`), Excel (`.xlsx`)
**Text Formats**: Plain Text (`.txt`), CSV (`.csv`), Markdown (`.md`), HTML (`.html`)

**Total conversion paths**: 64 (8×8 matrix)

## What "Best-Effort" Means

- **Text-first**: prioritize text and basic structure (titles, bullets, simple tables).
- **No OCR**: scanned PDFs are treated as images; extracted text may be empty.
- **No layout engine**: complex pagination, fonts, images, charts, and precise positioning are not preserved.
- **Quality varies**: some paths are high-fidelity (MD ↔ HTML, CSV ↔ XLSX), others are lossy (PDF → anything).

## Conversion Tiers

### Tier 1: High-Fidelity (Minimal Data Loss)

These conversions preserve most semantic structure:

- **TXT ↔ MD ↔ HTML**: Natural text format conversions
- **CSV ↔ XLSX**: Direct tabular data mapping
- **MD ↔ DOCX**: Good structure preservation (headings, lists, tables)
- **HTML ↔ DOCX**: Good semantic mapping (tags → styles)

### Tier 2: Good Quality (Some Structure Loss)

These conversions work well but lose some formatting:

- **DOCX ↔ PDF**: Text and tables preserved, no complex layouts
- **MD/HTML → PPTX**: Headings become slide titles
- **PPTX → DOCX**: Slides flatten to sections with bullets
- **XLSX → CSV**: First sheet exported (warns if multiple sheets)
- **TXT → (DOCX/PDF/HTML)**: Lines become paragraphs

### Tier 3: Lossy (Significant Structure Loss)

These conversions extract content but lose significant structure:

- **PDF → (any format)**: Text extraction only, no layout
- **PPTX → XLSX**: Only tables extracted
- **XLSX → PPTX**: Large tables become bullet summaries
- **DOCX → CSV**: Only tables extracted, text content lost

## Format-Specific Outputs

### PDF → *

- **→ DOCX/PPTX/TXT/MD/HTML**: One section/slide/block per PDF page with heading "Page N"
- **→ XLSX**: One sheet with one line per row, grouped by page
- **→ CSV**: Text lines from all pages (very lossy)

**Limitation**: No OCR for scanned PDFs

### DOCX → *

- **→ PDF**: Text and tables in simple PDF layout
- **→ PPTX**: Headings start new slides; paragraphs become bullets; tables on separate slides
- **→ XLSX**: Text in "Text" sheet; each table becomes its own sheet
- **→ TXT**: Paragraphs with heading markers (e.g., "### Heading")
- **→ CSV**: First table only (warns if multiple tables)
- **→ MD/HTML**: Headings, paragraphs, lists, and tables preserved

**Quality Improvement**: Smart heading detection (font size + bold + ALL CAPS)

### PPTX → *

- **→ DOCX**: One section per slide with bullets; tables preserved
- **→ PDF**: One page per slide with title/bullets
- **→ XLSX**: "Slides" sheet + one sheet per table
- **→ TXT/MD/HTML**: Slide titles as headings, bullets as lists
- **→ CSV**: First table only (warns if no tables)

**Quality Improvement**: Multi-table support (one slide per table)

### XLSX → *

- **→ DOCX**: One section per sheet; cell grid as table
- **→ PPTX**: Small sheets (≤20 rows, ≤10 cols) as tables; large sheets as bullet summaries
- **→ PDF**: One page per sheet with table
- **→ TXT/MD/HTML**: Sheet names as headings, cells as tables
- **→ CSV**: First sheet only (warns if multiple sheets)

**Quality Improvement**: Truncation warnings when data exceeds max_rows/max_cols

### TXT → *

- **→ DOCX/PDF**: Lines become paragraphs
- **→ PPTX**: Lines chunked to slides (10-12 lines per slide)
- **→ XLSX**: One line per row
- **→ CSV**: Each line becomes a CSV row
- **→ MD**: Lines with markdown syntax preserved
- **→ HTML**: Lines as `<p>` paragraphs

### CSV → *

- **→ XLSX**: Direct mapping (CSV subset of XLSX)
- **→ DOCX**: First row as heading, rest as table
- **→ PPTX**: Headers + rows as table on slides
- **→ PDF/HTML**: Table rendering
- **→ TXT/MD**: Pipe-separated values

**Quality**: Auto-delimiter detection with `csv.Sniffer`

### MD → *

- **→ DOCX**: Headings → styles, lists → bullets, tables → tables
- **→ HTML**: Near-lossless (both use similar semantic model)
- **→ PPTX**: H2+ starts new slides
- **→ XLSX**: Headings as sheet names, tables as sheets
- **→ PDF**: Rendered via sections
- **→ TXT**: Markdown syntax preserved

**Parsing**: Uses `mistune` with custom renderer

### HTML → *

- **→ MD**: Near-lossless (tags → markdown syntax)
- **→ DOCX**: Semantic tags → Word styles
- **→ PPTX**: H2+ starts new slides
- **→ XLSX**: Tables extracted to sheets
- **→ PDF**: Rendered via sections
- **→ TXT**: Clean text extraction
- **→ CSV**: First table only

**Parsing**: Uses `beautifulsoup4` with `lxml`

## Full Conversion Matrix

|        | PDF | DOCX | PPTX | XLSX | TXT | CSV | MD  | HTML |
|--------|-----|------|------|------|-----|-----|-----|------|
| PDF    | ✓   | ✓    | ✓    | ✓    | ✓   | ✗   | ✓   | ✓    |
| DOCX   | ✓   | ✓    | ✓    | ✓    | ✓   | ✓   | ✓   | ✓    |
| PPTX   | ✓   | ✓    | ✓    | ✓    | ✓   | ✓   | ✓   | ✓    |
| XLSX   | ✓   | ✓    | ✓    | ✓    | ✓   | ✓   | ✓   | ✓    |
| TXT    | ✓   | ✓    | ✓    | ✓    | ✓   | ✓   | ✓   | ✓    |
| CSV    | ✗   | ✓    | ✓    | ✓    | ✓   | ✓   | ✗   | ✓    |
| MD     | ✓   | ✓    | ✓    | ✓    | ✓   | ✗   | ✓   | ✓    |
| HTML   | ✓   | ✓    | ✓    | ✓    | ✓   | ✓   | ✓   | ✓    |

✓ = Supported (59 paths)
✗ = Not recommended (too lossy) (5 paths)

**Note**: All formats can convert to themselves (8 paths = copy operation)

## Edge Cases

- **PDF → CSV**: Not implemented (too lossy)
- **CSV → PDF/MD**: Not implemented (better to use CSV → XLSX → PDF/MD)
- **Multiple tables in PPTX**: Now creates one slide per table (improved in this version)
- **Large XLSX files**: Truncation warnings printed to stderr; use `--max-rows` and `--max-cols` to adjust
- **DOCX headings without styles**: Now detected via font size + bold + ALL CAPS heuristics
