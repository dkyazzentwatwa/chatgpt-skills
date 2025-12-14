# Limitations and Gotchas

## PDF Realities

- **Scanned PDFs**: `pypdf` cannot OCR. If the PDF is basically photos of pages, extracted text will be empty.
- **Weird PDFs**: Some PDFs have text split into individual positioned glyphs; extracted text can look scrambled.
- **No images**: Images are not extracted from PDFs (image extraction infrastructure exists but not yet integrated).
- **No complex layouts**: Multi-column layouts, text boxes, and complex positioning are not preserved.

## Office Format Realities

### DOCX

- **Heading detection improvements**: Now uses font size + bold + ALL CAPS heuristics, but may still miss headings with unconventional formatting.
- **Images**: Not yet extracted (infrastructure exists via `image_handler.py`).
- **Complex formatting**: Font colors, highlighting, borders, and advanced styles are lost.
- **Track changes**: Revision history and comments are not preserved.

### PPTX

- **Multi-table support**: ✅ Improved! Now creates one slide per table when multiple tables exist.
- **Slide visuals**: Images, charts, shapes, animations, and transitions are not rendered.
- **Speaker notes**: Not extracted.
- **Slide layouts**: All slides use basic Title + Content layout in output.

### XLSX

- **Data caps**: Only a bounded grid of values is exported (defaults: 200 rows × 50 cols).
- **Truncation warnings**: ✅ Improved! Warnings now printed to stderr when data is truncated.
- **Formulas**: Exported as computed values (using `data_only=True`).
- **Charts**: Not extracted or rendered.
- **Conditional formatting**: Lost in conversion.
- **Multiple sheets**: When converting to CSV, only first sheet is exported (with warning).

## Text Format Realities

### Plain Text (TXT)

- **No structure**: Very lossy for structured documents; best for simple note-taking or logging.
- **Line-based**: Everything is treated as lines; paragraphs, headings, and formatting are inferred heuristically.
- **No tables**: Tables become pipe-separated text.

### CSV

- **Tables only**: Non-tabular content is lost or forced into tabular format.
- **Delimiter detection**: Uses `csv.Sniffer` for auto-detection, but may fail on unusual delimiters.
- **Single sheet**: Only one table per file; multiple tables require multiple CSV files.
- **No formatting**: All cells are plain text strings.

### Markdown (MD)

- **Parsing variations**: Different Markdown flavors exist; this uses CommonMark via `mistune`.
- **Images**: Rendered as `![alt](url)` but images are not embedded or extracted.
- **Advanced features**: No support for footnotes, definition lists, or extended syntax.
- **Code blocks**: Language detection from class attributes (e.g., `class="language-python"`).

### HTML

- **CSS/JavaScript**: External stylesheets and scripts are not processed or preserved.
- **Complex layouts**: `<div>`-based layouts, flexbox, and grid are flattened to semantic content.
- **Images**: Referenced via `<img>` tags but not embedded or extracted.
- **Forms**: Form elements are extracted as text but not functional in output.
- **Parsing**: Uses `beautifulsoup4` with `lxml` parser; malformed HTML may produce unexpected results.

## Image Extraction

- **Foundation only**: `image_handler.py` provides hash-based deduplication and PIL format detection.
- **Not yet integrated**: Images are not automatically extracted from any format.
- **Placeholders**: Text formats show `[Image: filename.png]` placeholders.
- **Future work**: Full integration requires per-format extraction logic.

## Data Safety

- **Max chars**: Default 300,000 characters for text-based formats to prevent memory issues.
- **Max pages**: Default 200 pages for PDF to prevent long processing times.
- **Max rows/cols**: Default 200 rows × 50 columns for XLSX to prevent memory issues.
- **Truncation behavior**: Data beyond limits is silently dropped (but warnings are now printed for XLSX).

## Conversion Quality by Tier

### Tier 1: High-Fidelity
- **TXT ↔ MD ↔ HTML**: Minimal loss (text structure preserved)
- **CSV ↔ XLSX**: Direct mapping (CSV is subset of XLSX)
- **MD ↔ DOCX**: Good structure preservation
- **HTML ↔ DOCX**: Semantic tags map well to Word styles

### Tier 2: Good Quality
- **DOCX ↔ PDF**: Text and tables preserved
- **PPTX → DOCX**: Slides flatten to sections
- **XLSX → CSV**: First sheet only

### Tier 3: Lossy
- **PDF → anything**: Text extraction only, no layout
- **PPTX → XLSX**: Only tables extracted
- **DOCX → CSV**: Only tables extracted

## Things to Be Explicit About

- Ask for the user's priority: **"make it editable"** vs **"make it visually identical"**.
- If the user needs visually identical output, suggest using a renderer (e.g., LibreOffice/PowerPoint) outside the sandbox.
- For large spreadsheets, recommend increasing `--max-rows` and `--max-cols` gradually.
- For documents with many images, explain that images will show as placeholders.
- For scanned PDFs, recommend using OCR tools before conversion.

## Format Recommendations

**When to use each format:**

- **PDF**: Final, read-only distribution
- **DOCX**: Editable text documents with formatting
- **PPTX**: Presentations with slides
- **XLSX**: Tabular data, spreadsheets, data analysis
- **TXT**: Simple notes, logs, plain text
- **CSV**: Data exchange, database imports, simple tables
- **MD**: Documentation, README files, technical writing
- **HTML**: Web publishing, rich text with links

**Best conversion paths:**

- **Documentation**: MD → HTML (web) or MD → DOCX (printable)
- **Data**: CSV → XLSX (analysis) or XLSX → CSV (portability)
- **Reports**: DOCX → PDF (distribution)
- **Presentations**: PPTX → PDF (handouts)
- **Web to print**: HTML → DOCX → PDF

## Known Issues

- **DOCX tables with merged cells**: May render incorrectly
- **PPTX tables with complex formatting**: Formatting lost, structure preserved
- **XLSX with very wide rows**: May exceed column limit (50 cols default)
- **PDF with vertical text**: May extract in wrong order
- **HTML with nested tables**: May flatten incorrectly
- **Markdown code fences without language**: Rendered as plain code block
