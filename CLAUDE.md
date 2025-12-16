# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a library of AI agent skills for ChatGPT, inspired by Claude's Agent Skills system. Each skill is a self-contained folder with:
- `SKILL.md` (required): Core instructions with YAML frontmatter
- `scripts/` (optional): Executable Python/JavaScript tools
- `references/` (optional): Supporting documentation
- `assets/` (optional): Templates and configurations

Skills use **progressive disclosure**: metadata loads first, then core instructions, then extended resources on-demand. This architecture minimizes context usage while maximizing capability.

## Repository Structure

```
chatgpt-skills/
├── README.md                    # Main documentation
├── RESEARCH_NOTES.md           # Claude Skills architecture research
├── skill-name/                 # Individual skill folders
│   ├── SKILL.md               # Required: frontmatter + instructions
│   ├── scripts/               # Executable tools
│   │   └── requirements.txt   # Python dependencies (if needed)
│   ├── references/            # Extended documentation
│   └── assets/                # Templates, configs
└── skill-creator/             # Meta-skill for building new skills
    └── scripts/
        ├── init_skill.py      # Scaffold new skill
        ├── package_skill.py   # Create distributable ZIP
        └── quick_validate.py  # Validate skill structure
```

## Working with Skills

### Creating a New Skill

Use the skill-creator utilities to scaffold and validate:

```bash
# Initialize new skill structure
python skill-creator/scripts/init_skill.py my-new-skill --path .

# Validate skill structure and SKILL.md
python skill-creator/scripts/quick_validate.py path/to/skill-folder

# Package skill as distributable ZIP
python skill-creator/scripts/package_skill.py path/to/skill-folder
```

The `init_skill.py` script creates the complete folder structure with SKILL.md template, reference placeholders, and script directory.

### SKILL.md Requirements

Every skill MUST have a SKILL.md with this structure:

```markdown
---
name: skill-name
description: Brief description when to use (max 200 chars) - critical for skill activation
---

# Skill Title

[Markdown body with instructions, workflows, examples]
```

**Critical**: The `description` field determines when ChatGPT activates the skill. Make it specific and action-oriented.

### Skill Architecture Patterns

Based on RESEARCH_NOTES.md, skills follow one of four patterns:

1. **Workflow-Based**: Sequential processes (e.g., document-converter-suite)
   - Structure: Overview → Decision Tree → Step-by-step procedures

2. **Task-Based**: Tool collections (e.g., qr-code-generator)
   - Structure: Overview → Quick Start → Individual tasks/operations

3. **Reference/Guidelines**: Standards and specifications (e.g., mcp-builder)
   - Structure: Overview → Guidelines → Specifications → Examples

4. **Capabilities-Based**: Integrated systems (e.g., crypto-ta-analyzer)
   - Structure: Overview → Core Workflow → Individual capabilities

Choose the pattern that best fits the skill's purpose.

### Python Script Conventions

Skills with Python scripts follow these conventions:

**CLI Scripts** (e.g., `scripts/convert.py`, `scripts/svg_cli.py`):
- Include shebang: `#!/usr/bin/env python3`
- Use argparse for command-line interfaces
- Provide `--help` documentation
- Include usage examples in docstring

**Dependencies**:
- Place `requirements.txt` in `scripts/` directory (not root)
- Keep dependencies minimal and version-pinned
- Common deps: `numpy`, `pandas`, `pypdf`, `python-docx`, `openpyxl`

**Module Organization**:
- Simple scripts: single file (e.g., `generate_qr.py`)
- Complex tools: package structure (e.g., `svg_skill/__init__.py`, `svg_skill/core.py`)

## Existing Skills Reference

### crypto-ta-analyzer
Technical analysis with 24+ indicators. Uses CoinGecko integration.
- Scripts: `ta_analyzer.py`, `coingecko_converter.py`
- Dependencies: numpy, pandas

### document-converter-suite
PDF/DOCX/PPTX/XLSX conversion with text-first extraction.
- Main: `scripts/convert.py` (single-file CLI)
- Batch: `scripts/batch_convert.py`
- Library modules in `scripts/lib/`

### svg-precision-skill
Deterministic SVG generation from JSON specs.
- CLI: `scripts/svg_cli.py` (build, validate, render, diff)
- Library: `scripts/svg_skill/` package
- Commands: `build <spec.json> <out.svg>`, `validate <svg>`, `render <svg> <png>`

### qr-code-generator
QR code generation with UTM tracking.
- Single: `scripts/generate_qr.py`
- Batch: `scripts/batch_generate.py`
- Templates in `assets/templates/`

### mcp-builder
Guide for building MCP servers (Model Context Protocol).
- Scripts: `evaluation.py`, `connections.py`
- Extensive reference documentation in `reference/`

### skill-creator
Meta-skill for building new skills.
- `init_skill.py`: Scaffold new skill
- `package_skill.py`: Create ZIP for distribution
- `quick_validate.py`: Validate structure

### data-storyteller
Auto-generate narrative reports from CSV/Excel data.
- Scripts: `data_storyteller.py`
- Dependencies: pandas, numpy, matplotlib, seaborn, scipy, reportlab
- Features: Auto-detection, visualizations, PDF/HTML export, 5 chart styles

### image-enhancement-suite
Batch image processing toolkit.
- Scripts: `image_enhancer.py`
- Dependencies: pillow, opencv-python, numpy
- Features: Resize, crop, watermark, filters, format conversion, social presets

### ocr-document-processor
Extract text from images and PDFs using OCR.
- Scripts: `ocr_processor.py`
- Dependencies: pytesseract, pillow, PyMuPDF, opencv-python
- Features: 100+ languages, table extraction, receipt/card parsing, preprocessing

### video-to-gif
Convert videos to optimized GIFs.
- Scripts: `gif_workshop.py`
- Dependencies: moviepy, pillow, imageio, numpy
- Features: Clipping, speed control, text overlays, size optimization, presets

### financial-calculator
Comprehensive financial calculations.
- Scripts: `financial_calc.py`
- Dependencies: numpy, numpy-financial, pandas, matplotlib, scipy
- Features: Loans, investments, NPV/IRR, retirement, Monte Carlo simulations

### word-cloud-generator
Generate styled word clouds from text.
- Scripts: `wordcloud_gen.py`
- Dependencies: wordcloud, matplotlib, pillow, numpy
- Features: Custom shapes, 20+ colormaps, stopwords, PNG/SVG export

### sankey-diagram-creator
Create interactive Sankey flow diagrams.
- Scripts: `sankey_creator.py`
- Dependencies: plotly, pandas, kaleido
- Features: CSV/dict input, node/link customization, HTML/PNG/SVG export

### org-chart-generator
Generate organizational hierarchy charts.
- Scripts: `orgchart_gen.py`
- Dependencies: graphviz, pandas
- Features: Multiple layouts, department coloring, PNG/SVG/PDF export

### gantt-chart-creator
Create project timeline Gantt charts.
- Scripts: `gantt_creator.py`
- Dependencies: matplotlib, plotly, pandas, kaleido
- Features: Dependencies, milestones, progress tracking, static/interactive output

### flowchart-generator
Generate flowcharts from YAML/JSON or DSL.
- Scripts: `flowchart_gen.py`
- Dependencies: graphviz, PyYAML
- Features: Standard shapes, swimlanes, style presets, PNG/SVG/PDF export

### invoice-generator
Generate professional PDF invoices.
- Scripts: `invoice_gen.py`
- Dependencies: reportlab, pillow
- Features: Tax calculations, branding, batch generation, multiple currencies

### certificate-generator
Create professional certificates.
- Scripts: `certificate_gen.py`
- Dependencies: reportlab, pillow
- Features: Multiple templates, signatures, batch from CSV, PDF export

### resume-builder
Generate professional PDF resumes.
- Scripts: `resume_builder.py`
- Dependencies: reportlab, pillow
- Features: Multiple templates, ATS-friendly, JSON input, flexible sections

### report-generator
Create data-driven reports with charts and tables.
- Scripts: `report_gen.py`
- Dependencies: reportlab, pillow, pandas, matplotlib
- Features: Charts, tables, templates, branding, PDF/HTML export

### meeting-notes-formatter
Format raw meeting notes into structured documents.
- Scripts: `notes_formatter.py`
- Dependencies: reportlab, python-dateutil
- Features: Auto-detection, action items, Markdown/PDF export

### audio-analyzer
Comprehensive audio analysis with visualizations.
- Scripts: `audio_analyzer.py`
- Dependencies: librosa, soundfile, matplotlib, numpy, scipy
- Features: BPM/tempo, key detection, loudness metrics, waveform/spectrogram/chromagram plots

### audio-converter
Convert audio between formats with quality control.
- Scripts: `audio_converter.py`
- Dependencies: pydub, soundfile
- Features: MP3/WAV/FLAC/OGG/M4A, bitrate control, normalization, batch processing

### audio-trimmer
Cut, trim, and edit audio segments.
- Scripts: `audio_trimmer.py`
- Dependencies: pydub
- Features: Timestamp trimming, fades, speed control, concatenation, silence operations

### podcast-splitter
Split audio by silence detection.
- Scripts: `podcast_splitter.py`
- Dependencies: pydub
- Features: Silence detection, auto-split, chapter export, dead air removal

### sound-effects-generator
Generate programmatic audio effects.
- Scripts: `sfx_generator.py`
- Dependencies: numpy, scipy, soundfile
- Features: Tones (sine/square/saw/triangle), noise (white/pink/brown), DTMF, beep sequences

### sentiment-analyzer
Analyze text sentiment with scoring and visualization.
- Scripts: `sentiment_analyzer.py`
- Dependencies: textblob, pandas, matplotlib
- Features: Sentiment classification, polarity scoring, emotion detection, CSV batch, trend plots

### keyword-extractor
Extract keywords using TF-IDF, RAKE, and frequency.
- Scripts: `keyword_extractor.py`
- Dependencies: scikit-learn, nltk, wordcloud, matplotlib
- Features: Multiple algorithms, n-grams, word clouds, batch processing

### text-summarizer
Generate extractive summaries from text.
- Scripts: `text_summarizer.py`
- Dependencies: nltk, numpy, scikit-learn
- Features: TextRank/LSA/frequency methods, key points extraction, batch processing

### readability-scorer
Calculate readability scores and grade levels.
- Scripts: `readability_scorer.py`
- Dependencies: nltk
- Features: Flesch-Kincaid, Gunning Fog, SMOG, Coleman-Liau, ARI, text statistics

### data-anonymizer
Detect and mask PII in text and CSV files.
- Scripts: `data_anonymizer.py`
- Dependencies: pandas, faker
- Features: Multiple PII types, mask/redact/hash/fake strategies, reversible tokens

### data-type-converter
Convert between JSON, CSV, XML, YAML, TOML.
- Scripts: `data_converter.py`
- Dependencies: pyyaml, toml, xmltodict, pandas
- Features: 5 formats, nested data handling, batch conversion

### password-generator
Generate secure passwords and passphrases.
- Scripts: `password_gen.py`
- Dependencies: (standard library)
- Features: Cryptographically secure, custom rules, passphrases, strength checking

### hash-calculator
Calculate cryptographic hashes for text and files.
- Scripts: `hash_calc.py`
- Dependencies: (standard library)
- Features: MD5/SHA1/SHA256/SHA512/BLAKE2, file streaming, checksum files

### barcode-generator
Generate barcodes in multiple formats.
- Scripts: `barcode_gen.py`
- Dependencies: python-barcode, pillow
- Features: Code128/EAN13/UPC/Code39, check digits, batch from CSV

### regex-tester
Test, debug, and explain regular expressions.
- Scripts: `regex_tester.py`
- Dependencies: (standard library)
- Features: Pattern testing, explanation, generation from examples, common patterns library

### geo-visualizer
Create interactive maps with markers, heatmaps, and routes.
- Scripts: `geo_visualizer.py`
- Dependencies: folium, pandas, branca
- Features: Markers, heatmaps, choropleth, routes, clustering, HTML export

### distance-calculator
Calculate geographic distances and find nearby points.
- Scripts: `distance_calc.py`
- Dependencies: geopy
- Features: Haversine/Vincenty, multiple units, nearest search, distance matrix

### geocoder
Convert between addresses and coordinates.
- Scripts: `geocoder.py`
- Dependencies: geopy, pandas
- Features: Geocoding, reverse geocoding, batch processing, multiple providers

### json-schema-validator
Validate JSON against JSON Schema specifications.
- Scripts: `json_validator.py`
- Dependencies: jsonschema
- Features: Draft 4/6/7, error details, schema generation, batch validation

### api-response-mocker
Generate realistic mock API responses.
- Scripts: `api_mocker.py`
- Dependencies: faker
- Features: Schema-based, Faker integration, nested objects, reproducible seeds

### dependency-analyzer
Analyze Python imports and dependencies.
- Scripts: `dependency_analyzer.py`
- Dependencies: (standard library - ast)
- Features: Import extraction, classification, unused detection, requirements generation

### unit-converter
Convert between physical units.
- Scripts: `unit_converter.py`
- Dependencies: (standard library)
- Features: 10 categories, temperature, formulas, batch conversion

### correlation-explorer
Find correlations in datasets with visualization.
- Scripts: `correlation_explorer.py`
- Dependencies: pandas, numpy, scipy, matplotlib, seaborn
- Features: Pearson/Spearman/Kendall, heatmaps, p-values, strong/weak discovery

### outlier-detective
Detect anomalies using statistical and ML methods.
- Scripts: `outlier_detective.py`
- Dependencies: pandas, numpy, scipy, scikit-learn, matplotlib
- Features: IQR, Z-score, Isolation Forest, LOF, visualizations

### data-quality-auditor
Comprehensive data quality assessment.
- Scripts: `data_quality_auditor.py`
- Dependencies: pandas, numpy
- Features: Missing values, duplicates, type validation, quality score, HTML reports

### ab-test-calculator
Statistical significance testing for A/B experiments.
- Scripts: `ab_test_calc.py`
- Dependencies: scipy, numpy
- Features: Chi-square/Z-test, sample size estimation, power analysis, Bayesian analysis, multiple variants

### time-series-decomposer
Extract trend, seasonal, and residual components.
- Scripts: `ts_decomposer.py`
- Dependencies: pandas, numpy, scipy, statsmodels, matplotlib
- Features: Additive/multiplicative decomposition, auto-period detection, forecasting, anomaly detection

### dataset-comparer
Compare two datasets to find differences.
- Scripts: `dataset_comparer.py`
- Dependencies: pandas, numpy
- Features: Row comparison, key-based matching, schema diff, HTML reports, change tracking

### image-metadata-tool
Extract EXIF metadata from images.
- Scripts: `image_metadata.py`
- Dependencies: pillow, folium
- Features: EXIF extraction, GPS coordinates, location maps, metadata stripping, batch processing

### thumbnail-generator
Generate optimized thumbnails with smart cropping.
- Scripts: `thumbnail_gen.py`
- Dependencies: pillow, numpy
- Features: Smart cropping, multiple sizes, presets (web/social/icons), batch processing

### content-similarity-checker
Compare document similarity using multiple algorithms.
- Scripts: `similarity_checker.py`
- Dependencies: scikit-learn, nltk, numpy, pandas
- Features: Cosine/Jaccard/Levenshtein/TF-IDF, similarity matrix, duplicate detection

### named-entity-extractor
Extract named entities from text using NLP.
- Scripts: `entity_extractor.py`
- Dependencies: spacy, pandas
- Features: Person/Org/Location/Date extraction, spaCy and regex modes, batch processing, HTML highlighting

### clustering-analyzer
Cluster data using K-Means, DBSCAN, hierarchical.
- Scripts: `clustering_analyzer.py`
- Dependencies: scikit-learn, pandas, numpy, matplotlib, scipy
- Features: Multiple algorithms, elbow method, silhouette scores, visualization, cluster statistics

### cron-expression-builder
Build and validate cron expressions.
- Scripts: `cron_builder.py`
- Dependencies: croniter
- Features: Natural language to cron, cron to English, validation, next runs preview, presets

### lorem-ipsum-generator
Generate placeholder text for mockups.
- Scripts: `lorem_gen.py`
- Dependencies: faker
- Features: Multiple styles (classic/hipster/corporate/tech), paragraphs/sentences/words, HTML output, templates

### budget-analyzer
Analyze personal or business expenses.
- Scripts: `budget_analyzer.py`
- Dependencies: pandas, numpy, matplotlib, reportlab
- Features: Auto-categorization, trend analysis, budget vs actual, spending score, PDF/HTML reports

### stock-screener
Filter and screen stocks by financial metrics.
- Scripts: `stock_screener.py`
- Dependencies: pandas, numpy
- Features: Multi-metric filtering, presets (value/growth/dividend/quality), ranking, sector analysis

### roi-calculator
Calculate ROI for marketing, investments, and business decisions.
- Scripts: `roi_calculator.py`
- Dependencies: pandas, numpy
- Features: Marketing ROI, CAGR, break-even analysis, payback period, sensitivity analysis

### expense-report-generator
Generate formatted expense reports from receipt data.
- Scripts: `expense_report.py`
- Dependencies: pandas, reportlab, matplotlib
- Features: Auto-categorization, policy compliance, PDF reports, batch processing

### background-remover
Remove backgrounds from images using segmentation.
- Scripts: `background_remover.py`
- Dependencies: pillow, opencv-python, numpy, scikit-image
- Features: Color-based, edge detection, GrabCut, background replacement, batch processing

### image-filter-lab
Apply artistic filters and effects to images.
- Scripts: `image_filter.py`
- Dependencies: pillow, opencv-python, numpy
- Features: Sepia, B&W, vintage, blur, sharpen, vignette, film grain, presets

### photo-collage-maker
Create photo collages with grid layouts.
- Scripts: `collage_maker.py`
- Dependencies: pillow, numpy
- Features: Grid layouts, custom arrangements, templates, borders, text labels

### icon-generator
Generate app icons in multiple sizes from a single image.
- Scripts: `icon_generator.py`
- Dependencies: pillow
- Features: iOS, Android, favicon, macOS, Windows, PWA presets, rounding options

### table-extractor
Extract tables from PDFs and images to CSV/Excel.
- Scripts: `table_extractor.py`
- Dependencies: pdfplumber, pillow, pandas, pytesseract, opencv-python
- Features: PDF and image extraction, OCR support, multi-page, Excel export

### form-filler
Fill PDF forms programmatically with data.
- Scripts: `form_filler.py`
- Dependencies: PyMuPDF, pillow, pandas
- Features: Field detection, text/checkbox/dropdown support, batch filling, flatten option

### statistical-analyzer
Guided statistical analysis with hypothesis testing and plain-English interpretations.
- Scripts: `statistical_analyzer.py`
- Dependencies: scipy, statsmodels, pandas, numpy, matplotlib, seaborn, reportlab
- Features: t-tests, ANOVA, regression, correlation, distribution tests, plain-English results

### survey-analyzer
Analyze survey responses with Likert scales, cross-tabs, and sentiment.
- Scripts: `survey_analyzer.py`
- Dependencies: pandas, numpy, scipy, textblob, matplotlib, seaborn, wordcloud, reportlab
- Features: Likert analysis, cross-tabulation, sentiment scoring, NPS, frequency tables

### color-palette-extractor
Extract dominant colors from images using K-means clustering.
- Scripts: `color_palette_extractor.py`
- Dependencies: pillow, scikit-learn, numpy, matplotlib
- Features: K-means color extraction, CSS/JSON export, color schemes, swatch generation

### sprite-sheet-generator
Combine images into sprite sheets with CSS generation.
- Scripts: `sprite_sheet_generator.py`
- Dependencies: pillow
- Features: Grid layouts, smart packing, CSS sprite maps, padding control

### pdf-toolkit
Comprehensive PDF manipulation toolkit.
- Scripts: `pdf_toolkit.py`
- Dependencies: PyPDF2, PyMuPDF, pillow, reportlab
- Features: Merge, split, rotate, extract pages, watermark, compress, encrypt

### receipt-scanner
Extract structured data from receipt images using OCR.
- Scripts: `receipt_scanner.py`
- Dependencies: pytesseract, pillow, opencv-python, pandas, numpy
- Features: Vendor/date/items extraction, pattern matching, JSON/CSV export

### video-thumbnail-extractor
Extract frames and create thumbnails from videos.
- Scripts: `video_thumbnail_extractor.py`
- Dependencies: moviepy, pillow, numpy
- Features: Timestamp extraction, interval frames, grid previews, best frame detection

### video-clipper
Cut and trim video segments with timestamp control.
- Scripts: `video_clipper.py`
- Dependencies: moviepy, ffmpeg-python
- Features: Segment extraction, split by duration, trim start/end, multi-segment support

### topic-modeler
Extract topics from text using LDA (Latent Dirichlet Allocation).
- Scripts: `topic_modeler.py`
- Dependencies: gensim, nltk, pandas, matplotlib, wordcloud
- Features: LDA topic modeling, keyword extraction, document classification

### language-detector
Detect language of text with confidence scores.
- Scripts: `language_detector.py`
- Dependencies: langdetect, pandas
- Features: 50+ languages, confidence scoring, batch detection, CSV support

### ml-model-explainer
Explain ML model predictions using SHAP values and feature importance.
- Scripts: `ml_model_explainer.py`
- Dependencies: shap, scikit-learn, pandas, numpy, matplotlib
- Features: SHAP values, feature importance, waterfall plots, summary plots, decision paths

### classification-helper
Quick classifier training with automatic model selection and evaluation.
- Scripts: `classification_helper.py`
- Dependencies: scikit-learn, pandas, numpy, matplotlib, seaborn
- Features: Auto model selection, hyperparameter tuning, evaluation metrics, confusion matrix

### feature-engineering-kit
Automated feature engineering with encodings and transformations.
- Scripts: `feature_engineering.py`
- Dependencies: scikit-learn, pandas, numpy
- Features: Encodings (one-hot/label/target), scaling, polynomial features, binning, imputation

### pivot-table-generator
Generate pivot tables with aggregations and visualizations.
- Scripts: `pivot_table_generator.py`
- Dependencies: pandas, numpy, matplotlib, openpyxl
- Features: Multiple aggregations, filtering, grouping, charts, Excel/CSV/HTML export

### image-comparison-tool
Compare images with SSIM similarity scoring and difference visualization.
- Scripts: `image_comparison.py`
- Dependencies: opencv-python, scikit-image, pillow, numpy, matplotlib
- Features: SSIM scoring, pixel differences, side-by-side comparison, diff heatmaps

### timelapse-creator
Create timelapse videos from image sequences.
- Scripts: `timelapse_creator.py`
- Dependencies: moviepy, pillow, numpy
- Features: Image sequence to video, frame rate control, auto-sorting, quality control

### business-card-scanner
Extract contact information from business card images using OCR.
- Scripts: `business_card_scanner.py`
- Dependencies: pytesseract, pillow, opencv-python, pandas, numpy
- Features: Name/company/email/phone/website extraction, pattern matching, JSON export

### address-parser
Parse unstructured addresses into structured components.
- Scripts: `address_parser.py`
- Dependencies: pandas
- Features: Street/city/state/zip extraction, format standardization, batch processing

### phone-number-formatter
Standardize and format phone numbers with validation.
- Scripts: `phone_number_formatter.py`
- Dependencies: phonenumbers, pandas
- Features: E.164/international/national formats, validation, country detection, batch processing

### currency-converter
Convert between currencies with exchange rates.
- Scripts: `currency_converter.py`
- Dependencies: forex-python, pandas
- Features: 150+ currencies, historical rates, batch conversion, locale formatting

### audio-normalizer
Normalize audio volume levels using peak or RMS normalization.
- Scripts: `audio_normalizer.py`
- Dependencies: pydub
- Features: Peak/RMS normalization, loudness matching, LUFS support, batch processing, headroom control

### video-metadata-inspector
Extract comprehensive metadata from video files.
- Scripts: `video_metadata_inspector.py`
- Dependencies: moviepy, ffmpeg-python, pandas
- Features: Duration/resolution/codec info, bitrate analysis, format detection, batch processing, JSON/CSV export

### video-captioner
Add text overlays and subtitles to videos.
- Scripts: `video_captioner.py`
- Dependencies: moviepy, pillow
- Features: Static overlays, timed captions, SRT import, custom styling, position control, style presets, fade effects

### date-normalizer
Parse and normalize dates from various formats.
- Scripts: `date_normalizer.py`
- Dependencies: python-dateutil, pandas
- Features: Smart parsing (100+ formats), ISO8601/US/EU output, ambiguity detection, relative dates, CSV processing

### model-comparison-tool
Compare multiple ML models with cross-validation.
- Scripts: `model_comparison_tool.py`
- Dependencies: scikit-learn, pandas, numpy, matplotlib
- Features: Multi-model testing, k-fold CV, comprehensive metrics, statistical testing, auto-selection

### code-profiler
Analyze Python code performance and identify bottlenecks.
- Scripts: `code_profiler.py`
- Dependencies: (standard library - cProfile, pstats)
- Features: Time profiling, line-by-line analysis, call statistics, function timing, report export

### qr-barcode-reader
Decode QR codes and barcodes from images.
- Scripts: `qr_barcode_reader.py`
- Dependencies: pyzbar, pillow
- Features: Multiple formats (QR/EAN13/Code128/UPC), batch processing, data extraction, JSON export

### uuid-generator
Generate UUIDs in various formats.
- Scripts: `uuid_generator.py`
- Dependencies: (standard library - uuid)
- Features: UUID1/UUID4/UUID5, bulk generation, namespace UUIDs, format options, validation

### statistical-power-calculator
Calculate statistical power and sample sizes for experiments.
- Scripts: `statistical_power_calculator.py`
- Dependencies: scipy, statsmodels, numpy
- Features: Power calculation, sample size determination, effect size analysis, t-test/ANOVA support

### kml-geojson-converter
Convert between KML and GeoJSON geo formats.
- Scripts: `kml_geojson_converter.py`
- Dependencies: geopandas, fiona
- Features: Bidirectional conversion, feature preservation, coordinate system support, validation

### contract-generator
Generate legal contracts from templates with variable substitution.
- Scripts: `contract_generator.py`
- Dependencies: python-docx, pandas
- Features: DOCX templates, variable substitution, conditional sections, validation, batch generation from CSV

### territory-mapper
Visualize sales territories and coverage areas on interactive maps.
- Scripts: `territory_mapper.py`
- Dependencies: folium, geopandas, pandas, shapely
- Features: Territory polygons, color coding, interactive maps, data overlay, GeoJSON import, HTML export

### batch-qr-generator
Generate bulk QR codes from CSV with UTM tracking.
- Scripts: `batch_qr_generator.py`
- Dependencies: qrcode, pillow, pandas
- Features: CSV input, UTM tracking, custom styling, logos, sequential naming, metadata export

### scientific-paper-figure-generator
Create publication-ready scientific figures.
- Scripts: `scientific_paper_figure_generator.py`
- Dependencies: matplotlib, pandas, numpy
- Features: Journal templates (Nature/Science/IEEE), multi-panel figures, 300+ DPI, statistical annotations, vector output

## Key Design Principles

### Progressive Disclosure
From RESEARCH_NOTES.md - skills load information in tiers:
1. **Metadata**: name + description (always visible)
2. **Core**: SKILL.md body (when relevant)
3. **Extended**: references/ files (on-demand)

This prevents context window saturation.

### Skill Activation Criteria
The `description` field in YAML frontmatter is critical. It should:
- Be under 200 characters
- Specify WHEN to use the skill (triggers, file types, tasks)
- Be specific and action-oriented
- Examples: "Use when asked to create or modify SVGs for icons, diagrams..."

### Workflow-Focused Design
Skills should enable complete workflows, not just wrap individual tools:
- Consolidate related operations
- Design for end-to-end task completion
- Include decision trees for complex scenarios
- Provide examples of common usage patterns

### Context Optimization
- Return concise, high-signal information
- Use executable scripts for heavy computation (doesn't consume context)
- Default to human-readable output
- Provide "verbose" vs "concise" options where applicable

## Testing Skills

Before packaging a skill:

1. **Structure validation**:
   ```bash
   python skill-creator/scripts/quick_validate.py path/to/skill
   ```

2. **SKILL.md checks**:
   - YAML frontmatter present and valid
   - `name` under 64 characters
   - `description` under 200 characters, specific about activation
   - Markdown body provides clear instructions

3. **Script testing** (if applicable):
   - Test with minimal input
   - Test with edge cases
   - Verify error messages are clear and actionable
   - Check that dependencies are listed in requirements.txt

4. **Reference completeness**:
   - All referenced files exist
   - Examples are concrete and realistic
   - Limitations are clearly documented

## Packaging for Distribution

Package skills as ZIP files:

```bash
python skill-creator/scripts/package_skill.py path/to/skill [output-dir]
```

The packager:
- Validates skill structure
- Creates ZIP with skill folder as root (not files directly)
- Names ZIP as `skill-name.zip`
- Outputs to specified directory or current directory

## Documentation Standards

- **README.md**: High-level overview, architecture, all skills listed
- **RESEARCH_NOTES.md**: Claude Skills research and design principles
- **SKILL.md**: Per-skill instructions (see format above)
- **references/*.md**: Extended documentation, specs, recipes

Keep documentation focused on what's not obvious from code inspection. Avoid listing every file or generic best practices.

## IMPORTANT: Documentation Maintenance

**Whenever making significant feature updates or changes, ALWAYS update:**

1. **CLAUDE.md** (this file):
   - Update "Existing Skills Reference" section when adding/modifying skills
   - Update any affected conventions or patterns
   - Add new skills to the reference list

2. **README.md** (root):
   - Add new skills to "Available Skills" section with description and features
   - Update existing skill descriptions when features change
   - Keep the skill list current and accurate

3. **Individual SKILL.md files**:
   - Update when adding new features to a skill
   - Keep examples current with actual API/usage
   - Document any breaking changes

**What counts as "significant changes":**
- Adding a new skill
- Adding new indicators/features to existing skills
- Changing output formats or APIs
- Adding new scripts or capabilities
- Bug fixes that change behavior
- New configuration options

**Documentation update checklist:**
- [ ] CLAUDE.md updated (if repo-wide changes)
- [ ] README.md updated (if new/modified skills)
- [ ] SKILL.md updated (if skill features changed)
- [ ] references/*.md updated (if detailed docs affected)
