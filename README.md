# ChatGPT Skills Library

A curated collection of AI agent skills for ChatGPT, inspired by [Claude's Agent Skills system](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills). This library provides specialized capabilities that extend ChatGPT's functionality through structured instructions, scripts, and resources.

## About

With OpenAI's new skill system for ChatGPT, AI agents can now dynamically load specialized capabilities similar to Claude's skills architecture. This repository adapts the proven Claude skills format to work with ChatGPT, enabling:

- **Progressive disclosure**: Skills load only when relevant to the current task
- **Specialized expertise**: Domain-specific workflows and tool orchestration
- **Code execution**: Python and JavaScript scripts for deterministic operations
- **Structured knowledge**: Organized instructions, templates, and reference materials

## How Skills Work

Each skill is a self-contained folder containing:

- **SKILL.md**: Core instructions with YAML frontmatter (`name`, `description`)
- **scripts/**: Executable Python/Node.js tools for specialized operations
- **references/**: Supporting documentation, specifications, and examples
- **assets/**: Templates, configurations, and static resources

When you interact with ChatGPT, the AI evaluates available skills and loads relevant ones dynamically based on your request, similar to how [Claude Skills](https://support.claude.com/en/articles/12512176-what-are-skills) operate.

## Available Skills

### 🪙 Crypto TA Analyzer
**`crypto-ta-analyzer/`**

Comprehensive cryptocurrency and stock technical analysis using **29 proven indicators** including Bollinger Bands, Ichimoku Cloud, and On-Balance Volume with divergence detection. Features a **7-tier signal system** (STRONG_BUY → STRONG_SELL) with volume confirmation, squeeze detection, and divergence warnings.

**Key Features (v2.0)**:
- **29 Indicators**: RSI, MACD, BB, OBV, Ichimoku, EMA, SMA, MFI, KDJ, SAR, VWAP, ATR + 17 more
- **7-Tier Signals**: STRONG_BUY, BUY, WEAK_BUY, NEUTRAL, WEAK_SELL, SELL, STRONG_SELL
- **Divergence Detection**: RSI, MACD, and OBV divergence alerts
- **Bollinger Squeeze**: Low volatility breakout detection
- **Volume Confirmation**: OBV/MFI agreement scoring
- **Generic Data Converter**: Auto-detects CoinGecko, Binance, Yahoo Finance, or any OHLCV format
- **Regime-Aware**: Adapts scoring for trending vs ranging markets

**Scripts**:
- `ta_analyzer.py` - Main analysis engine (29 indicators)
- `data_converter.py` - Generic multi-source data converter with auto-detection

**Use for**: Price analysis, trading signals, trend identification, divergence detection, breakout setups, multi-indicator technical assessments

### 📄 Document Converter Suite
**`document-converter-suite/`**

Convert between **8 document formats** with best-effort text-first extraction. Supports **64 conversion paths** (full 8×8 matrix) with single file or batch processing.

**Supported Formats**:
- **Office**: PDF, Word (DOCX), PowerPoint (PPTX), Excel (XLSX)
- **Text**: Plain Text (TXT), CSV, Markdown (MD), HTML

**Key Features**:
- **64 Conversion Paths**: Any format to any format (8×8 matrix)
- **Smart Heading Detection**: Font size + bold + ALL CAPS heuristics
- **Multi-Table PPTX**: Creates one slide per table (no dropped tables)
- **Data Truncation Warnings**: Alerts when XLSX data exceeds limits
- **Batch Processing**: Recursive directory conversion with pattern matching
- **High Fidelity**: Markdown ↔ HTML and Markdown ↔ DOCX preserve structure well

**Scripts**:
- `convert.py` - Single-file CLI converter
- `batch_convert.py` - Batch converter for directories

**Use for**: Document format conversion, content extraction, batch processing, Markdown/HTML to Office, table extraction to CSV/XLSX

### 🔧 MCP Builder
**`mcp-builder/`**

Comprehensive guide for creating high-quality MCP (Model Context Protocol) servers that enable LLMs to interact with external services. Includes best practices, implementation patterns, and evaluation frameworks for both Python (FastMCP) and Node/TypeScript.

**Use for**: Building MCP servers, API integrations, tool development

### 📱 QR Code Generator
**`qr-code-generator/`**

Generate QR codes with URLs, optional UTM tracking parameters, captions, and export to PNG/SVG formats. Supports batch generation from CSV files with quality controls for print and web use.

**Use for**: QR code generation, UTM tracking, marketing materials, print exports

### 🎨 SVG Precision
**`svg-precision-skill/`**

Deterministic, structurally correct SVG generation from JSON specifications. Includes validation, PNG preview rendering, and templates for icons, diagrams, charts, UI mockups, and technical drawings.

**Use for**: Icons, flowcharts, data visualization, wireframes, technical drawings

### 🛠️ Skill Creator
**`skill-creator/`**

Meta-skill for designing and building new agent skills. Provides templates, best practices, and workflows for creating effective skills that extend AI capabilities.

**Use for**: Creating new skills, skill architecture, workflow design

### 📊 Data Storyteller
**`data-storyteller/`**

Transform CSV/Excel data into narrative reports with auto-generated insights, visualizations, and PDF/HTML export. Automatically detects data types, finds patterns, creates relevant charts, and writes plain-English summaries.

**Key Features**:
- **Auto-Detection**: Column types, patterns, correlations, time series
- **Smart Visualizations**: Distribution plots, correlation heatmaps, trend charts
- **Narrative Insights**: Plain-English findings, recommendations, warnings
- **Multiple Exports**: PDF reports, HTML dashboards, JSON data
- **Chart Styles**: Business, scientific, minimal, dark, colorful themes

**Scripts**:
- `data_storyteller.py` - Main analysis and reporting engine

**Use for**: Data analysis, automated reporting, insight generation, executive summaries, exploratory data analysis

### 🖼️ Image Enhancement Suite
**`image-enhancement-suite/`**

Professional image processing toolkit for batch resize, crop, watermark, color correction, format conversion, and compression. Supports single or batch operations with quality presets for web, print, and social media.

**Key Features**:
- **Resize & Crop**: Smart resizing, aspect ratio preservation, content-aware cropping
- **Watermarking**: Text or image watermarks with positioning and opacity
- **Color Correction**: Brightness, contrast, saturation, sharpness adjustments
- **Format Conversion**: PNG, JPEG, WebP, BMP, TIFF, GIF
- **Presets**: Instagram, Twitter, Facebook, LinkedIn, print sizes
- **Batch Processing**: Process entire directories with consistent settings

**Scripts**:
- `image_enhancer.py` - Main processing engine with CLI

**Use for**: Image optimization, social media assets, watermarking, batch processing, thumbnail generation

### 📝 OCR Document Processor
**`ocr-document-processor/`**

Extract text from images and scanned PDFs using OCR. Supports multiple languages, structured output (Markdown/JSON), table detection, and batch processing.

**Key Features**:
- **Multi-Format Input**: PNG, JPEG, TIFF, PDF
- **100+ Languages**: Full Tesseract language support
- **Structured Output**: Plain text, Markdown, JSON, HTML, searchable PDF
- **Table Extraction**: Detect and export tables to CSV
- **Preprocessing**: Deskew, denoise, threshold for better accuracy
- **Specialized Parsers**: Receipt scanning, business card extraction

**Scripts**:
- `ocr_processor.py` - Main OCR engine with preprocessing

**Use for**: Document digitization, text extraction, receipt scanning, business card parsing, PDF searchability

### 🎬 Video to GIF Workshop
**`video-to-gif/`**

Convert video clips to optimized GIFs with speed control, cropping, text overlays, and file size optimization. Perfect for social media, documentation, and presentations.

**Key Features**:
- **Clip Selection**: Extract specific time ranges
- **Speed Control**: Slow motion, speed up, reverse, boomerang
- **Text Overlays**: Captions, titles, timed text
- **Effects**: Filters, fades, color adjustments
- **Optimization**: Smart compression for target file size
- **Presets**: Twitter, Discord, Slack, Reddit optimized

**Scripts**:
- `gif_workshop.py` - Main video-to-GIF converter

**Use for**: Social media GIFs, documentation animations, tutorial clips, reaction GIFs

### 💰 Financial Calculator Suite
**`financial-calculator/`**

Comprehensive financial calculations including loan amortization, investment projections, NPV/IRR analysis, retirement planning, and Monte Carlo simulations.

**Key Features**:
- **Loan Calculator**: Amortization schedules, prepayment analysis
- **Investment Calculator**: Compound growth, recurring contributions
- **NPV/IRR**: Project valuation, payback period
- **Retirement Planning**: FIRE calculator, withdrawal strategies
- **Monte Carlo**: Risk analysis with probability distributions
- **Mortgage Tools**: Affordability, refinance comparison
- **Visualizations**: Charts for all calculations

**Scripts**:
- `financial_calc.py` - Main calculation engine with CLI

**Use for**: Loan analysis, investment planning, retirement projections, business valuation, risk assessment

### ☁️ Word Cloud Generator
**`word-cloud-generator/`**

Generate visually appealing word clouds from text, files, or word frequency dictionaries with customizable shapes, colors, and fonts.

**Key Features**:
- **Multiple Inputs**: Raw text, text files, or frequency dictionaries
- **Custom Shapes**: Rectangle, circle, or custom mask images
- **Color Schemes**: 20+ matplotlib colormaps or custom colors
- **Stopword Filtering**: Built-in English stopwords + custom additions
- **Export Formats**: PNG, SVG

**Scripts**:
- `wordcloud_gen.py` - Main word cloud generator with CLI

**Use for**: Text visualization, content analysis, presentation graphics, marketing materials

### 📊 Sankey Diagram Creator
**`sankey-diagram-creator/`**

Create interactive Sankey diagrams for flow visualization from CSV, DataFrame, or dictionary data. Perfect for budget flows, energy transfers, and user journeys.

**Key Features**:
- **Multiple Inputs**: CSV, DataFrame, or dict data
- **Interactive Output**: Hover tooltips with flow details
- **Node Customization**: Colors, labels, positions
- **Link Styling**: Colors, opacity, labels
- **Export Formats**: HTML (interactive), PNG, SVG

**Scripts**:
- `sankey_creator.py` - Main Sankey diagram creator with CLI

**Use for**: Budget visualization, energy flows, user journey mapping, data pipeline diagrams

### 🏢 Org Chart Generator
**`org-chart-generator/`**

Generate organizational hierarchy charts from CSV, JSON, or nested data structures. Supports multiple layouts and department coloring.

**Key Features**:
- **Multiple Inputs**: CSV with manager references, JSON nested structure
- **Layout Options**: Top-down, left-right, bottom-up, right-left
- **Department Coloring**: Auto-color by department or level
- **Node Customization**: Shapes, fonts, borders
- **Export Formats**: PNG, SVG, PDF, DOT

**Scripts**:
- `orgchart_gen.py` - Main org chart generator with CLI

**Use for**: Company org charts, team structures, reporting relationships, hierarchy visualization

### 📅 Gantt Chart Creator
**`gantt-chart-creator/`**

Create project timeline Gantt charts with dependencies, milestones, and progress tracking. Supports both static and interactive output.

**Key Features**:
- **Task Properties**: Name, dates, progress, assignee, category
- **Dependencies**: Finish-to-start and other dependency types
- **Milestones**: Diamond markers for key dates
- **Progress Tracking**: Visual progress bars within tasks
- **Today Marker**: Vertical line showing current date
- **Export Formats**: PNG, SVG, PDF, HTML (interactive)

**Scripts**:
- `gantt_creator.py` - Main Gantt chart creator with CLI

**Use for**: Project management, sprint planning, timeline visualization, roadmaps

### 🔀 Flowchart Generator
**`flowchart-generator/`**

Generate flowcharts from YAML/JSON definitions or Python DSL. Supports standard flowchart symbols and swimlanes for multi-actor processes.

**Key Features**:
- **Standard Shapes**: Start/End, Process, Decision, Input/Output, Connector
- **Swimlanes**: Multi-actor/department process diagrams
- **Style Presets**: Business, technical, minimal, colorful themes
- **Layout Options**: Top-bottom, left-right, and more
- **Export Formats**: PNG, SVG, PDF, DOT

**Scripts**:
- `flowchart_gen.py` - Main flowchart generator with CLI

**Use for**: Process documentation, workflow visualization, decision trees, system diagrams

### 🧾 Invoice Generator
**`invoice-generator/`**

Generate professional PDF invoices with customizable templates, tax calculations, and company branding. Supports batch generation from CSV.

**Key Features**:
- **Professional Templates**: Clean, modern invoice designs
- **Tax Calculations**: Multiple tax rates, compound taxes
- **Custom Branding**: Logo, colors, payment details
- **Batch Generation**: Create invoices from CSV data
- **Export Format**: PDF

**Scripts**:
- `invoice_gen.py` - Main invoice generator with CLI

**Use for**: Billing, invoicing, freelance work, small business accounting

### 🏆 Certificate Generator
**`certificate-generator/`**

Create professional certificates for courses, achievements, and awards with multiple template styles and batch generation support.

**Key Features**:
- **Multiple Templates**: Modern, classic, elegant, minimal, academic
- **Custom Branding**: Logo, colors, signatures
- **Batch Generation**: Create hundreds from CSV
- **Auto-Generated IDs**: Unique certificate identifiers
- **Export Format**: PDF

**Scripts**:
- `certificate_gen.py` - Main certificate generator with CLI

**Use for**: Course completion, awards, training programs, events

### 📝 Resume Builder
**`resume-builder/`**

Generate professional PDF resumes from structured JSON data with multiple templates and ATS-friendly formatting.

**Key Features**:
- **Multiple Templates**: Modern, classic, minimal, executive
- **ATS-Friendly**: Clean formatting for applicant tracking systems
- **Flexible Sections**: Experience, education, skills, projects, certifications
- **JSON Input**: Structured data format
- **Export Format**: PDF

**Scripts**:
- `resume_builder.py` - Main resume builder with CLI

**Use for**: Job applications, career documents, professional profiles

### 📑 Report Generator (Deep)
**`report-generator/`**

Create professional data-driven reports with charts, tables, and narrative text. Comprehensive reporting system for business analytics.

**Key Features**:
- **Rich Content**: Text, tables, charts, images, headers
- **Chart Types**: Bar, line, pie, scatter, area
- **Data Integration**: CSV, DataFrame, dict inputs
- **Templates**: Executive summary, quarterly review, project status
- **Branding**: Logo, colors, headers/footers
- **Export Formats**: PDF, HTML

**Scripts**:
- `report_gen.py` - Main report generator with CLI

**Use for**: Business reports, analytics dashboards, status updates, automated reporting

### 📋 Meeting Notes Formatter
**`meeting-notes-formatter/`**

Convert raw meeting notes to structured markdown or PDF with automatic section detection and action item extraction.

**Key Features**:
- **Auto-Detection**: Title, attendees, sections, action items
- **Action Item Extraction**: Tasks with owners and due dates
- **Smart Parsing**: Identifies structure from raw text
- **Export Formats**: Markdown, PDF

**Scripts**:
- `notes_formatter.py` - Main notes formatter with CLI

**Use for**: Meeting documentation, note organization, action tracking

### 🎵 Audio Analyzer (Deep)
**`audio-analyzer/`**

Comprehensive audio analysis toolkit with tempo/BPM detection, musical key identification, frequency analysis, loudness metrics, and professional visualizations.

**Key Features**:
- **Tempo Detection**: BPM and beat position tracking with confidence scores
- **Key Detection**: Musical key and mode (major/minor) identification
- **Loudness Metrics**: RMS, peak, LUFS, dynamic range analysis
- **Frequency Analysis**: Spectrum, dominant frequencies, band energy
- **Visualizations**: Waveform, spectrogram, chromagram, beat grid, dashboard
- **Batch Processing**: Analyze multiple files at once
- **Export Formats**: JSON reports, PNG/SVG visualizations

**Scripts**:
- `audio_analyzer.py` - Main analysis engine with CLI

**Use for**: Music analysis, podcast quality checks, audio forensics, DJ preparation, sound design

### 🔊 Audio Converter
**`audio-converter/`**

Convert audio files between popular formats with control over bitrate, sample rate, and channels. Supports batch processing and volume normalization.

**Key Features**:
- **Format Support**: MP3, WAV, FLAC, OGG, M4A, AIFF
- **Quality Control**: Bitrate, sample rate, channel configuration
- **Volume Normalization**: Automatic level adjustment
- **Batch Processing**: Convert entire directories
- **Metadata Preservation**: Copy tags when possible

**Scripts**:
- `audio_converter.py` - Main converter with CLI

**Use for**: Format conversion, podcast preparation, audio archiving, device compatibility

### ✂️ Audio Trimmer
**`audio-trimmer/`**

Cut, trim, and edit audio segments with fade effects, speed control, concatenation, and basic audio manipulations.

**Key Features**:
- **Precise Trimming**: Cut by timestamp (HH:MM:SS) or milliseconds
- **Fade Effects**: Fade in/out with customizable duration
- **Speed Control**: Speed up or slow down audio
- **Concatenation**: Join multiple files with optional crossfade
- **Effects**: Reverse, loop, overlay, volume adjustment
- **Silence Operations**: Add, remove, or strip silence

**Scripts**:
- `audio_trimmer.py` - Main trimmer with CLI

**Use for**: Podcast editing, ringtone creation, audio clips, lecture editing, highlight extraction

### 🎙️ Podcast Splitter
**`podcast-splitter/`**

Automatically split audio files into segments based on silence detection. Perfect for dividing podcasts into chapters, creating clips, or removing dead air.

**Key Features**:
- **Silence Detection**: Configurable threshold and minimum duration
- **Auto-Split**: Divide at natural breaks/pauses
- **Silence Removal**: Remove or shorten long pauses
- **Chapter Export**: Save individual segments as files
- **Preview Mode**: List silences without splitting
- **Batch Processing**: Process multiple recordings

**Scripts**:
- `podcast_splitter.py` - Main splitter with CLI

**Use for**: Podcast chapters, interview segmentation, dead air removal, audio cleanup

### 🔔 Sound Effects Generator
**`sound-effects-generator/`**

Generate programmatic audio: pure tones, noise types, DTMF signals, and simple sound effects. Perfect for testing, alerts, and placeholder sounds.

**Key Features**:
- **Tones**: Sine, square, sawtooth, triangle waveforms
- **Noise**: White, pink, brown/red noise generation
- **DTMF**: Phone dial tones for any digit sequence
- **Beep Sequences**: Multi-tone patterns and melodies
- **Effects**: Fade in/out, volume control
- **Export**: WAV, MP3 formats

**Scripts**:
- `sfx_generator.py` - Main generator with CLI

**Use for**: Test signals, notification sounds, audio placeholders, DTMF generation

### 💬 Sentiment Analyzer
**`sentiment-analyzer/`**

Analyze text sentiment with detailed scoring, emotion detection, and visualization. Process single texts, CSV files, or track sentiment trends over time.

**Key Features**:
- **Sentiment Classification**: Positive, negative, neutral with confidence scores
- **Polarity Scoring**: -1.0 (negative) to +1.0 (positive) scale
- **Emotion Detection**: Joy, anger, sadness, fear, surprise
- **Batch Processing**: Analyze CSV files with any text column
- **Visualizations**: Distribution plots, trend charts

**Scripts**:
- `sentiment_analyzer.py` - Main analyzer with CLI

**Use for**: Product review analysis, social media monitoring, customer feedback categorization

### 🔑 Keyword Extractor
**`keyword-extractor/`**

Extract important keywords and key phrases from text using TF-IDF, RAKE, and frequency analysis with word cloud visualization.

**Key Features**:
- **Multiple Algorithms**: TF-IDF, RAKE, frequency-based
- **Key Phrases**: Multi-word phrase extraction
- **N-gram Support**: Unigrams, bigrams, trigrams
- **Word Cloud**: Visualize keyword importance
- **Batch Processing**: Process multiple documents

**Scripts**:
- `keyword_extractor.py` - Main extractor with CLI

**Use for**: SEO keyword research, document analysis, content summarization

### 📝 Text Summarizer
**`text-summarizer/`**

Generate extractive summaries from long text documents using TextRank, LSA, and frequency-based algorithms.

**Key Features**:
- **Multiple Algorithms**: TextRank, LSA, frequency
- **Length Control**: By ratio, sentence count, or word count
- **Key Points**: Extract bullet-point summaries
- **Preserve Order**: Maintain original sentence sequence
- **Batch Processing**: Summarize multiple documents

**Scripts**:
- `text_summarizer.py` - Main summarizer with CLI

**Use for**: Article summarization, meeting notes, research papers, content digests

### 📊 Readability Scorer
**`readability-scorer/`**

Calculate readability scores using industry-standard formulas. Get grade level estimates, complexity metrics, and improvement suggestions.

**Key Features**:
- **Multiple Formulas**: Flesch-Kincaid, Gunning Fog, SMOG, Coleman-Liau, ARI
- **Grade Level**: US grade level estimate
- **Reading Ease**: 0-100 ease score
- **Text Statistics**: Words, sentences, syllables analysis
- **Comparison**: Compare readability across texts

**Scripts**:
- `readability_scorer.py` - Main scorer with CLI

**Use for**: Content optimization, documentation review, accessibility compliance

### 🔒 Data Anonymizer
**`data-anonymizer/`**

Detect and mask PII (names, emails, phones, SSN, addresses) in text and CSV files with multiple masking strategies.

**Key Features**:
- **PII Detection**: Names, emails, phones, SSN, credit cards, dates
- **Multiple Strategies**: Mask, redact, hash, fake data replacement
- **CSV Processing**: Anonymize specific columns or auto-detect
- **Reversible Tokens**: Optional mapping for de-anonymization
- **Audit Reports**: List all detected PII with locations

**Scripts**:
- `data_anonymizer.py` - Main anonymizer with CLI

**Use for**: GDPR compliance, test data generation, log sanitization

### 🔄 Data Type Converter
**`data-type-converter/`**

Convert between JSON, CSV, XML, YAML, and TOML formats with nested structure handling and batch processing.

**Key Features**:
- **5 Formats**: JSON, CSV, XML, YAML, TOML
- **Nested Data**: Flatten or preserve nested structures
- **Type Preservation**: Maintain data types where possible
- **Batch Processing**: Convert multiple files

**Scripts**:
- `data_converter.py` - Main converter with CLI

**Use for**: Data migration, config file conversion, API response transformation

### 🔐 Password Generator
**`password-generator/`**

Generate cryptographically secure passwords and memorable passphrases with customizable rules and strength checking.

**Key Features**:
- **Secure Generation**: Uses cryptographically secure random
- **Custom Rules**: Character sets, required types, exclusions
- **Passphrases**: Word-based memorable passwords
- **Strength Check**: Evaluate password strength with entropy
- **Bulk Generation**: Generate multiple passwords

**Scripts**:
- `password_gen.py` - Main generator with CLI

**Use for**: Secure credential generation, bulk account setup, password policies

### #️⃣ Hash Calculator
**`hash-calculator/`**

Calculate cryptographic hashes for text and files with verification and checksum file generation.

**Key Features**:
- **Multiple Algorithms**: MD5, SHA1, SHA256, SHA384, SHA512, BLAKE2
- **File Hashing**: Streaming for large files
- **Verification**: Compare against expected hash
- **Checksum Files**: Generate/verify checksum files
- **Batch Processing**: Hash entire directories

**Scripts**:
- `hash_calc.py` - Main calculator with CLI

**Use for**: File integrity verification, download validation, duplicate detection

### 📊 Barcode Generator
**`barcode-generator/`**

Generate barcodes in multiple formats for retail, inventory, and identification with batch generation support.

**Key Features**:
- **Multiple Formats**: Code128, EAN13, EAN8, UPC-A, Code39, ITF, ISBN
- **Output Formats**: PNG, SVG
- **Customization**: Size, colors, text display
- **Check Digit**: Auto-calculate and verify
- **Batch Generation**: From CSV files

**Scripts**:
- `barcode_gen.py` - Main generator with CLI

**Use for**: Product labels, inventory tags, shipping labels, ISBN barcodes

### 🔍 Regex Tester
**`regex-tester/`**

Test, debug, and explain regular expressions with pattern generation from examples and common patterns library.

**Key Features**:
- **Pattern Testing**: Detailed match results with positions
- **Pattern Explanation**: Plain-English breakdown
- **Pattern Generation**: Create patterns from examples
- **Find & Replace**: Test substitution patterns
- **Common Patterns**: Library of pre-built patterns

**Scripts**:
- `regex_tester.py` - Main tester with CLI

**Use for**: Regex development, data validation, text parsing, pattern learning

### 🗺️ Geo Visualizer
**`geo-visualizer/`**

Create interactive HTML maps with markers, heatmaps, routes, and choropleth layers using Folium. Perfect for location-based data visualization.

**Key Features**:
- **Markers**: Custom icons, popups, tooltips, clustering
- **Heatmaps**: Density visualization with weights
- **Choropleth**: Color regions by data values
- **Routes**: Draw paths between points
- **Circles**: Radius-based coverage areas
- **Export**: Interactive HTML files

**Scripts**:
- `geo_visualizer.py` - Main map generator with CLI

**Use for**: Store locators, sales heatmaps, delivery routes, location analytics

### 📏 Distance Calculator
**`distance-calculator/`**

Calculate geographic distances between coordinates using Haversine or Vincenty methods. Find nearest points, radius search, and distance matrices.

**Key Features**:
- **Methods**: Haversine (fast), Vincenty (accurate)
- **Units**: km, miles, meters, nautical miles, feet
- **Nearest Search**: Find N closest points
- **Radius Search**: Points within distance
- **Distance Matrix**: All pairwise distances
- **CSV Processing**: Batch operations

**Scripts**:
- `distance_calc.py` - Main calculator with CLI

**Use for**: Logistics optimization, delivery zones, store location analysis

### 📍 Geocoder
**`geocoder/`**

Convert between addresses and coordinates (geocoding/reverse geocoding) with multiple provider support and batch processing.

**Key Features**:
- **Geocoding**: Address to coordinates
- **Reverse Geocoding**: Coordinates to address
- **Multiple Providers**: Nominatim (free), Google, Bing
- **Address Components**: Structured parsing
- **Batch Processing**: CSV file operations
- **Rate Limiting**: Built-in request throttling

**Scripts**:
- `geocoder.py` - Main geocoder with CLI

**Use for**: Address validation, location enrichment, coordinate lookups

### ✅ JSON Schema Validator
**`json-schema-validator/`**

Validate JSON data against JSON Schema specifications (Draft 4, 6, 7) with detailed error messages and schema generation.

**Key Features**:
- **Validation**: Validate data against schema
- **Error Details**: Path-specific error messages
- **Schema Generation**: Infer schema from sample data
- **Batch Validation**: Multiple files at once
- **Multiple Drafts**: Support for Draft 4, 6, 7

**Scripts**:
- `json_validator.py` - Main validator with CLI

**Use for**: API validation, config file checks, data quality assurance

### 🎭 API Response Mocker
**`api-response-mocker/`**

Generate realistic mock API responses using Faker for testing, prototyping, and frontend development.

**Key Features**:
- **Schema-Based**: Define response structure
- **Faker Integration**: Realistic fake data
- **Nested Objects**: Complex structures
- **Arrays**: Lists with configurable counts
- **Reproducible**: Seed support for consistency
- **Export**: JSON, XML formats

**Scripts**:
- `api_mocker.py` - Main mocker with CLI

**Use for**: API testing, frontend development, mock servers, sample data

### 🔬 Dependency Analyzer
**`dependency-analyzer/`**

Analyze Python imports and project dependencies. Find unused imports, generate requirements.txt, and detect circular imports.

**Key Features**:
- **Import Extraction**: List all imports
- **Classification**: stdlib, third-party, local
- **Unused Detection**: Find unused imports
- **Requirements Generation**: Auto-generate requirements.txt
- **Circular Import Detection**: Find dependency cycles
- **Project Analysis**: Full directory scanning

**Scripts**:
- `dependency_analyzer.py` - Main analyzer with CLI

**Use for**: Code cleanup, dependency management, project documentation

### ⚖️ Unit Converter
**`unit-converter/`**

Convert between physical units across 10 categories: length, mass, temperature, time, volume, area, speed, digital, energy, and pressure.

**Key Features**:
- **10 Categories**: Comprehensive unit coverage
- **Multiple Units**: Common and scientific units
- **Temperature**: Celsius, Fahrenheit, Kelvin
- **Formulas**: Show conversion formulas
- **Batch Conversion**: Convert lists of values
- **No Dependencies**: Pure Python implementation

**Scripts**:
- `unit_converter.py` - Main converter with CLI

**Use for**: Scientific calculations, data transformation, unit standardization

### 📊 Correlation Explorer
**`correlation-explorer/`**

Analyze correlations between variables in datasets with visualization, significance testing, and strong/weak correlation discovery.

**Key Features**:
- **Multiple Methods**: Pearson, Spearman, Kendall
- **Heatmap Visualization**: Color-coded correlation matrix
- **P-Values**: Statistical significance
- **Strong/Weak Discovery**: Find notable correlations
- **Target Analysis**: Correlations with specific variable
- **Export**: CSV, JSON, PNG

**Scripts**:
- `correlation_explorer.py` - Main analyzer with CLI

**Use for**: Feature selection, data exploration, relationship discovery

### 🔍 Outlier Detective
**`outlier-detective/`**

Detect anomalies and outliers using statistical (IQR, Z-score) and ML methods (Isolation Forest, LOF) with visualization.

**Key Features**:
- **Statistical Methods**: IQR, Z-score, Modified Z-score
- **ML Methods**: Isolation Forest, Local Outlier Factor
- **Visualization**: Box plots, distributions, scatter plots
- **Multi-Column**: Analyze multiple variables
- **Reports**: Detailed outlier analysis
- **Clean Data**: Export data with outliers removed

**Scripts**:
- `outlier_detective.py` - Main detector with CLI

**Use for**: Data cleaning, fraud detection, quality control, anomaly detection

### 📋 Data Quality Auditor
**`data-quality-auditor/`**

Comprehensive data quality assessment with checks for missing values, duplicates, type issues, and quality scoring with HTML reports.

**Key Features**:
- **Completeness**: Missing value analysis
- **Uniqueness**: Duplicate detection
- **Validity**: Type validation
- **Consistency**: Pattern checking
- **Quality Score**: 0-100 overall score
- **Reports**: HTML, JSON export
- **Validation Rules**: Custom rule definitions

**Scripts**:
- `data_quality_auditor.py` - Main auditor with CLI

**Use for**: ETL pipelines, data validation, dataset documentation, quality gates

### 📈 A/B Test Calculator
**`ab-test-calculator/`**

Statistical significance testing for A/B experiments with power analysis, sample size estimation, and Bayesian analysis.

**Key Features**:
- **Significance Testing**: Chi-square, Z-test for conversions
- **Sample Size Estimation**: Calculate required samples for desired power
- **Power Analysis**: Determine test power given sample size
- **Bayesian Analysis**: Probability to beat baseline
- **Multiple Variants**: Support A/B/n testing with correction
- **Duration Estimation**: Based on traffic volume

**Scripts**:
- `ab_test_calc.py` - Main A/B test calculator with CLI

**Use for**: Marketing experiments, product testing, conversion optimization

### 📊 Time Series Decomposer
**`time-series-decomposer/`**

Extract trend, seasonal, and residual components from time series data with visualization and basic forecasting.

**Key Features**:
- **Decomposition**: Additive and multiplicative models
- **Auto-Period Detection**: Automatically find seasonal periods
- **Trend Analysis**: Direction, slope, growth rate
- **Anomaly Detection**: Find outliers in residuals
- **Basic Forecasting**: Trend extrapolation, seasonal naive
- **Visualization**: Component plots, ACF/PACF

**Scripts**:
- `ts_decomposer.py` - Main decomposer with CLI

**Use for**: Sales forecasting, demand planning, seasonality analysis, anomaly detection

### 🔄 Dataset Comparer
**`dataset-comparer/`**

Compare two CSV/Excel datasets to identify differences, additions, deletions, and value changes between versions.

**Key Features**:
- **Row Comparison**: Find added, removed, changed rows
- **Key-Based Matching**: Compare by primary key columns
- **Value Changes**: Detect changed values in matching rows
- **Schema Comparison**: Identify column differences
- **HTML Reports**: Color-coded diff reports
- **Export**: CSV, JSON, HTML formats

**Scripts**:
- `dataset_comparer.py` - Main comparer with CLI

**Use for**: Data migration validation, ETL verification, change tracking

### 📷 Image Metadata Tool
**`image-metadata-tool/`**

Extract EXIF metadata from images including GPS coordinates, camera settings, and timestamps. Generate location maps and strip metadata for privacy.

**Key Features**:
- **EXIF Extraction**: Camera, lens, settings, timestamps
- **GPS Data**: Extract coordinates, generate map URLs
- **Location Maps**: Create interactive maps from photo locations
- **Privacy Features**: Strip metadata for sharing
- **Batch Processing**: Process entire folders
- **Export**: JSON, CSV formats

**Scripts**:
- `image_metadata.py` - Main metadata tool with CLI

**Use for**: Photo organization, privacy audit, location mapping, EXIF analysis

### 🖼️ Thumbnail Generator
**`thumbnail-generator/`**

Generate optimized thumbnails with smart cropping, multiple sizes, and batch processing for web, social media, and app icons.

**Key Features**:
- **Smart Cropping**: Center, edge-detection based
- **Multiple Sizes**: Generate many sizes at once
- **Presets**: Web, social media, app icons, favicon
- **Batch Processing**: Process entire directories
- **Quality Control**: Optimize file size vs quality
- **Formats**: JPEG, PNG, WebP output

**Scripts**:
- `thumbnail_gen.py` - Main generator with CLI

**Use for**: E-commerce images, gallery thumbnails, social media assets, app icons

### 📝 Content Similarity Checker
**`content-similarity-checker/`**

Compare document similarity using TF-IDF, cosine similarity, Jaccard index, and Levenshtein distance.

**Key Features**:
- **Multiple Algorithms**: Cosine, Jaccard, Levenshtein, TF-IDF
- **Similarity Matrix**: Pairwise document comparison
- **Duplicate Detection**: Find near-duplicates by threshold
- **Folder Comparison**: Compare all documents in directory
- **Batch Processing**: CSV export of results

**Scripts**:
- `similarity_checker.py` - Main checker with CLI

**Use for**: Plagiarism detection, content deduplication, document matching

### 🏷️ Named Entity Extractor
**`named-entity-extractor/`**

Extract named entities (people, organizations, locations, dates) from text using spaCy NLP or regex patterns.

**Key Features**:
- **Entity Types**: Person, Org, Location, Date, Money, Percent
- **Multiple Modes**: spaCy (accurate) or regex (fast)
- **Batch Processing**: Process folders of documents
- **Entity Frequency**: Count mentions
- **HTML Highlighting**: Color-coded entity visualization
- **Export**: JSON, CSV formats

**Scripts**:
- `entity_extractor.py` - Main extractor with CLI

**Use for**: Document analysis, information extraction, content tagging

### 📊 Clustering Analyzer
**`clustering-analyzer/`**

Cluster data using K-Means, DBSCAN, and hierarchical clustering with evaluation metrics and visualization.

**Key Features**:
- **K-Means**: Partition-based with elbow method
- **DBSCAN**: Density-based for arbitrary shapes
- **Hierarchical**: Agglomerative with dendrograms
- **Evaluation**: Silhouette scores, cluster statistics
- **Optimal K**: Automatic cluster count selection
- **Visualization**: 2D plots, dendrograms, silhouette charts

**Scripts**:
- `clustering_analyzer.py` - Main analyzer with CLI

**Use for**: Customer segmentation, pattern discovery, anomaly detection, data grouping

### ⏰ Cron Expression Builder
**`cron-expression-builder/`**

Build and validate cron expressions from natural language with next run preview and preset schedules.

**Key Features**:
- **Natural Language**: Convert descriptions to cron
- **Cron Parser**: Explain cron in plain English
- **Validation**: Check syntax and field ranges
- **Next Runs**: Preview upcoming execution times
- **Presets**: Common scheduling patterns
- **Interactive Mode**: Guided expression building

**Scripts**:
- `cron_builder.py` - Main builder with CLI

**Use for**: Job scheduling, task automation, schedule configuration

### 📝 Lorem Ipsum Generator
**`lorem-ipsum-generator/`**

Generate placeholder text in multiple styles (classic, hipster, corporate, tech) for mockups, wireframes, and testing.

**Key Features**:
- **Multiple Styles**: Classic, hipster, corporate, tech
- **Output Types**: Paragraphs, sentences, words, lists
- **HTML Output**: Tags for web mockups
- **Templates**: Fill templates with placeholders
- **Custom Length**: Specify exact word counts
- **Fake Data**: Names, emails, dates

**Scripts**:
- `lorem_gen.py` - Main generator with CLI

**Use for**: Mockups, wireframes, test data, placeholder content

### 💰 Budget Analyzer
**`budget-analyzer/`**

Analyze personal or business expenses with auto-categorization, trend analysis, and budget tracking with spending health scores.

**Key Features**:
- **Auto-Categorization**: Classify expenses by merchant/description
- **Trend Analysis**: Month-over-month spending patterns
- **Budget vs Actual**: Track spending against targets
- **Spending Score**: 0-100 health score with grade
- **Recommendations**: AI-powered savings suggestions
- **Reports**: PDF and HTML export with charts

**Scripts**:
- `budget_analyzer.py` - Main analyzer with CLI

**Use for**: Personal finance, business expense tracking, budget planning

### 📈 Stock Screener
**`stock-screener/`**

Filter and screen stocks by financial metrics like P/E ratio, market cap, dividend yield with preset screens and comparative analysis.

**Key Features**:
- **Multi-Metric Filtering**: P/E, P/B, dividend yield, growth rates
- **Preset Screens**: Value, growth, dividend, quality investing
- **Ranking**: Score and rank stocks by criteria
- **Sector Analysis**: Group by industry
- **Comparison**: Side-by-side stock analysis
- **Export**: CSV, JSON formats

**Scripts**:
- `stock_screener.py` - Main screener with CLI

**Use for**: Stock analysis, portfolio screening, investment research

### 💵 ROI Calculator
**`roi-calculator/`**

Calculate ROI for marketing campaigns, investments, and business decisions with break-even analysis and payback period calculations.

**Key Features**:
- **Marketing ROI**: Campaign performance with ROAS
- **Investment ROI**: CAGR and time-adjusted returns
- **Break-Even Analysis**: Find profit threshold
- **Payback Period**: Time to recover investment
- **Sensitivity Analysis**: What-if scenarios
- **Comparison**: Compare multiple options

**Scripts**:
- `roi_calculator.py` - Main calculator with CLI

**Use for**: Marketing analysis, investment decisions, business planning

### 🧾 Expense Report Generator
**`expense-report-generator/`**

Generate professional expense reports from receipt data with automatic categorization, policy compliance checking, and PDF export.

**Key Features**:
- **Multiple Inputs**: CSV, JSON, manual entry
- **Auto-Categorization**: Classify expenses by type
- **Policy Compliance**: Flag out-of-policy expenses
- **Receipt Tracking**: Link receipts to expenses
- **PDF Reports**: Professional formatted output
- **Batch Processing**: Generate multiple reports

**Scripts**:
- `expense_report.py` - Main generator with CLI

**Use for**: Business expenses, reimbursement requests, expense tracking

### 🖼️ Background Remover
**`background-remover/`**

Remove backgrounds from images using color-based, edge detection, or GrabCut segmentation with batch processing support.

**Key Features**:
- **Color Removal**: Remove solid color backgrounds
- **Edge Detection**: Detect subject edges
- **GrabCut**: Interactive foreground extraction
- **Background Replacement**: Color or image backgrounds
- **Edge Refinement**: Feathering and mask adjustment
- **Batch Processing**: Process multiple images

**Scripts**:
- `background_remover.py` - Main remover with CLI

**Use for**: Product photography, portrait editing, image compositing

### 🎨 Image Filter Lab
**`image-filter-lab/`**

Apply artistic filters and effects to images including vintage, sepia, blur, sharpen, and custom presets with batch processing.

**Key Features**:
- **Color Filters**: Sepia, B&W, negative, tint
- **Blur Effects**: Gaussian, motion, radial blur
- **Artistic**: Vintage, film grain, vignette
- **Adjustments**: Brightness, contrast, saturation
- **Presets**: Vintage, film, noir, dreamy
- **Batch Processing**: Apply to folders

**Scripts**:
- `image_filter.py` - Main filter lab with CLI

**Use for**: Photo editing, batch processing, social media content

### 🖼️ Photo Collage Maker
**`photo-collage-maker/`**

Create photo collages with grid layouts, custom arrangements, borders, and templates for social media and presentations.

**Key Features**:
- **Grid Layouts**: 2x2, 3x3, custom grids
- **Templates**: Magazine, polaroid, pinterest styles
- **Backgrounds**: Solid colors, gradients, images
- **Image Fitting**: Fit, fill, stretch options
- **Text Labels**: Add captions
- **Rounded Corners**: Stylized output

**Scripts**:
- `collage_maker.py` - Main collage maker with CLI

**Use for**: Social media, presentations, photo albums

### 📱 Icon Generator
**`icon-generator/`**

Generate app icons in all required sizes for iOS, Android, web, macOS, and Windows from a single source image.

**Key Features**:
- **iOS Icons**: All sizes including App Store
- **Android Icons**: All density variants
- **Favicon**: Multi-resolution ICO and PNG
- **macOS/Windows**: Desktop icon formats
- **PWA Icons**: Progressive web app sizes
- **Options**: Rounding, padding, backgrounds

**Scripts**:
- `icon_generator.py` - Main generator with CLI

**Use for**: App development, favicon generation, multi-platform icons

### 📋 Table Extractor
**`table-extractor/`**

Extract tables from PDFs and images to CSV or Excel with OCR support for scanned documents.

**Key Features**:
- **PDF Tables**: Extract from digital PDFs
- **Image Tables**: OCR-based extraction
- **Multiple Tables**: Extract all tables
- **Export Formats**: CSV, Excel, JSON
- **Multi-Page**: Process entire documents
- **Table Detection**: Auto-detect boundaries

**Scripts**:
- `table_extractor.py` - Main extractor with CLI

**Use for**: Data extraction, document processing, report analysis

### 📝 Form Filler
**`form-filler/`**

Fill PDF forms programmatically with data from JSON, CSV, or dictionaries with batch filling support.

**Key Features**:
- **Field Detection**: Auto-detect form fields
- **Field Types**: Text, checkbox, dropdown, radio
- **Batch Filling**: Fill multiple forms from data
- **Field Mapping**: Map data keys to fields
- **Flatten**: Convert to non-editable PDF
- **Export**: Filled PDF output

**Scripts**:
- `form_filler.py` - Main filler with CLI

**Use for**: Form automation, batch document generation, data entry

### 67. Statistical Analyzer

Perform statistical hypothesis testing, regression analysis, and ANOVA with plain-English interpretations.

**Features**:
- **Hypothesis Tests**: t-tests, chi-square, ANOVA
- **Regression**: Linear, polynomial, multiple regression
- **Correlation**: Pearson, Spearman with significance
- **Distribution Tests**: Normality, Q-Q plots
- **Visualizations**: Regression plots, residual analysis
- **Plain-English Results**: Interpret statistical outputs

**Scripts**:
- `statistical_analyzer.py` - Statistical analysis with CLI

**Use for**: Academic research, A/B testing analysis, data science

### 68. Survey Analyzer

Analyze survey responses with Likert scales, cross-tabulations, and sentiment analysis.

**Features**:
- **Likert Scales**: Agreement, frequency, satisfaction scales
- **Cross-Tabulation**: Chi-square tests, heatmaps
- **Sentiment Analysis**: Text response sentiment scoring
- **Frequency Tables**: Response distributions
- **NPS Calculation**: Net Promoter Score
- **Report Generation**: PDF/HTML reports

**Scripts**:
- `survey_analyzer.py` - Survey analysis with CLI

**Use for**: Customer feedback analysis, research surveys, satisfaction studies

### 69. Color Palette Extractor

Extract dominant colors from images using K-means clustering and generate color palettes.

**Features**:
- **K-means Extraction**: Extract N dominant colors
- **Multiple Formats**: CSS, JSON, ASE, ACO export
- **Color Info**: RGB, HEX, HSL, HSV values
- **Swatch Generation**: Palette visualization
- **Batch Processing**: Multiple images

**Scripts**:
- `color_palette_extractor.py` - Color extraction with CLI

**Use for**: Design work, brand color analysis, UI development

### 70. Sprite Sheet Generator

Combine multiple images into sprite sheets with automatic CSS generation.

**Features**:
- **Grid Layouts**: Auto or custom arrangements
- **CSS Generation**: Sprite map CSS classes
- **Padding Control**: Spacing between sprites
- **Smart Packing**: Optimize sprite placement
- **Transparent Support**: Preserve alpha channels

**Scripts**:
- `sprite_sheet_generator.py` - Sprite sheet generation

**Use for**: Game development, web optimization, icon management

### 71. PDF Toolkit

Comprehensive PDF manipulation - merge, split, rotate, watermark, and encrypt PDFs.

**Features**:
- **Merge/Split**: Combine or divide PDFs
- **Rotate**: Rotate pages by 90/180/270 degrees
- **Extract**: Extract specific pages
- **Watermark**: Add text/image watermarks
- **Compress**: Reduce file size
- **Encrypt**: Password protection

**Scripts**:
- `pdf_toolkit.py` - PDF operations with CLI

**Use for**: Document management, PDF automation, batch processing

### 72. Receipt Scanner

Extract structured data from receipt images using OCR and pattern matching.

**Features**:
- **OCR Processing**: Extract text from images
- **Data Extraction**: Vendor, date, items, total, tax
- **Pattern Matching**: Smart regex for receipts
- **Multi-Format**: JPG, PNG, PDF support
- **Batch Processing**: Multiple receipts
- **Export**: JSON/CSV output

**Scripts**:
- `receipt_scanner.py` - Receipt OCR with CLI

**Use for**: Expense tracking, accounting automation, receipt digitization

### 73. Video Thumbnail Extractor

Extract frames from videos at timestamps or intervals and create thumbnail grids.

**Features**:
- **Timestamp Extraction**: Extract at specific times
- **Interval Frames**: Extract every N seconds
- **Grid Previews**: Contact sheet thumbnails
- **Best Frame**: Find sharpest frames
- **Batch Processing**: Multiple videos

**Scripts**:
- `video_thumbnail_extractor.py` - Frame extraction with CLI

**Use for**: Video previews, thumbnail generation, frame analysis

### 74. Video Clipper

Cut video segments by timestamp with precise frame control.

**Features**:
- **Segment Extraction**: Cut by start/end times
- **Multi-Segment**: Extract multiple clips
- **Split by Duration**: Auto-split into chunks
- **Trim**: Remove start/end portions
- **Format Preservation**: Maintain quality

**Scripts**:
- `video_clipper.py` - Video clipping with CLI

**Use for**: Video editing, highlight extraction, content creation

### 75. Topic Modeler

Extract topics from text collections using LDA (Latent Dirichlet Allocation).

**Features**:
- **LDA Modeling**: Topic extraction from documents
- **Keyword Extraction**: Representative keywords per topic
- **Document Classification**: Assign docs to topics
- **Coherence Scores**: Evaluate topic quality
- **Visualization**: Word clouds, distributions

**Scripts**:
- `topic_modeler.py` - Topic modeling with CLI

**Use for**: Document analysis, content categorization, research

### 76. Language Detector

Detect language of text with confidence scores for 50+ languages.

**Features**:
- **50+ Languages**: Wide language support
- **Confidence Scores**: Probability estimates
- **Batch Detection**: Process multiple texts
- **CSV Support**: Analyze text columns
- **Alternative Languages**: Multiple candidates

**Scripts**:
- `language_detector.py` - Language detection with CLI

**Use for**: Multilingual content, data preprocessing, text classification

### 77. ML Model Explainer

Explain machine learning model predictions using SHAP values and feature importance.

**Features**:
- **SHAP Values**: Explain individual predictions
- **Feature Importance**: Global feature rankings
- **Decision Paths**: Trace prediction logic
- **Visualizations**: Waterfall, force plots, summary plots
- **Multiple Models**: Tree-based, linear, neural networks

**Scripts**:
- `ml_model_explainer.py` - Model explainability with CLI

**Use for**: Model interpretability, debugging ML models, regulatory compliance

### 78. Classification Helper

Train and evaluate classification models with automatic model selection.

**Features**:
- **Auto Model Selection**: Compare RF, GB, LR, SVM
- **Hyperparameter Tuning**: Grid/random search
- **Evaluation Metrics**: Accuracy, precision, recall, F1, ROC-AUC
- **Cross-Validation**: K-fold validation
- **Confusion Matrix**: Detailed error analysis
- **Model Export**: Save trained models

**Scripts**:
- `classification_helper.py` - Classification training with CLI

**Use for**: Quick ML experiments, model comparison, baseline models

### 79. Feature Engineering Kit

Automated feature engineering with encodings, scaling, and transformations.

**Features**:
- **Encodings**: One-hot, label, target encoding
- **Scaling**: Standard, min-max, robust scaling
- **Polynomial Features**: Generate interactions
- **Binning**: Discretize continuous features
- **Missing Value Handling**: Imputation strategies

**Scripts**:
- `feature_engineering.py` - Feature engineering with CLI

**Use for**: ML preprocessing, data preparation, feature generation

### 80. Pivot Table Generator

Generate pivot tables with aggregations and visualizations.

**Features**:
- **Multiple Aggregations**: Sum, mean, count, min, max
- **Filtering**: Filter data before pivoting
- **Grouping**: Multi-level row/column grouping
- **Charts**: Auto-generate pivot charts
- **Export**: Excel, CSV, HTML output

**Scripts**:
- `pivot_table_generator.py` - Pivot table generation with CLI

**Use for**: Data summarization, business reporting, analysis

### 81. Image Comparison Tool

Compare images with SSIM similarity scoring and difference visualization.

**Features**:
- **SSIM Similarity**: Structural similarity index
- **Pixel Differences**: Highlight changed areas
- **Side-by-Side**: Visual comparison layout
- **Diff Heatmap**: Color-coded differences
- **Batch Comparison**: Compare multiple pairs

**Scripts**:
- `image_comparison.py` - Image comparison with CLI

**Use for**: Quality assurance, A/B testing visuals, change detection

### 82. Timelapse Creator

Create timelapse videos from image sequences.

**Features**:
- **Image Sequence**: Combine images into video
- **Frame Rate Control**: Custom FPS settings
- **Sorting**: Auto-sort by timestamp/filename
- **Quality Control**: Resolution and codec options

**Scripts**:
- `timelapse_creator.py` - Timelapse creation with CLI

**Use for**: Construction progress, nature photography, process documentation

### 83. Business Card Scanner

Extract contact information from business card images using OCR.

**Features**:
- **OCR Extraction**: Extract text from cards
- **Contact Parsing**: Name, company, email, phone, website
- **Pattern Recognition**: Smart regex for fields
- **Batch Processing**: Multiple cards
- **Export**: JSON, vCard output

**Scripts**:
- `business_card_scanner.py` - Card scanning with CLI

**Use for**: Contact management, CRM integration, event networking

### 84. Address Parser

Parse unstructured addresses into structured components.

**Features**:
- **Component Extraction**: Street, city, state, zip, country
- **Format Standardization**: Normalize formats
- **Validation**: Verify address components
- **Batch Processing**: Parse multiple addresses
- **International Support**: Multiple country formats

**Scripts**:
- `address_parser.py` - Address parsing with CLI

**Use for**: Data cleaning, address validation, geocoding preparation

### 85. Phone Number Formatter

Standardize and format phone numbers with validation.

**Features**:
- **Format Standardization**: E.164, national, international
- **Validation**: Check valid numbers
- **Country Detection**: Auto-detect country codes
- **Batch Processing**: Format multiple numbers
- **Type Detection**: Mobile, fixed-line, etc.

**Scripts**:
- `phone_number_formatter.py` - Phone formatting with CLI

**Use for**: Data cleaning, contact management, international calls

### 86. Currency Converter

Convert between currencies with exchange rates.

**Features**:
- **Exchange Rates**: Convert between 150+ currencies
- **Historical Rates**: Date-based conversions
- **Batch Conversion**: Process multiple amounts
- **Currency Formatting**: Locale-specific formatting
- **CSV Support**: Convert currency columns

**Scripts**:
- `currency_converter.py` - Currency conversion with CLI

**Use for**: Financial analysis, e-commerce, expense tracking

### 87. Audio Normalizer

Normalize audio volume levels using peak or RMS normalization for consistent loudness.

**Features**:
- **Peak Normalization**: Target dBFS level with headroom control
- **RMS Normalization**: Average loudness matching
- **LUFS Matching**: Broadcast standard compliance
- **Batch Processing**: Normalize multiple files
- **Format Preservation**: Maintain original audio format
- **Analysis**: Current peak/RMS level reporting

**Scripts**:
- `audio_normalizer.py` - Audio volume normalization with CLI

**Use for**: Podcast production, music playlists, audiobooks, video audio

### 88. Video Metadata Inspector

Extract comprehensive metadata from video files including duration, resolution, codec, and technical specs.

**Features**:
- **Basic Info**: Duration, resolution, frame rate, file size
- **Codec Details**: Video/audio codec, profile, bitrate
- **Format Info**: Container type, metadata tags
- **Batch Analysis**: Process multiple files
- **Export**: JSON, CSV, text reports
- **Comparison**: Side-by-side video comparison

**Scripts**:
- `video_metadata_inspector.py` - Video metadata extraction with CLI

**Use for**: Format verification, quality assessment, transcoding planning, video cataloging

### 89. Video Captioner

Add text overlays and subtitles to videos with customizable styling and timing.

**Features**:
- **Text Overlays**: Static or timed text
- **SRT Import**: Import subtitle files
- **Custom Styling**: Font, size, color, background, outline
- **Positioning**: Top, bottom, center, custom coordinates
- **Style Presets**: Instagram, YouTube, minimal, bold
- **Animations**: Fade in/out effects

**Scripts**:
- `video_captioner.py` - Video captioning with CLI

**Use for**: Social media videos, tutorials, accessibility, branding

### 90. Date Normalizer

Parse and normalize dates from various formats into consistent ISO 8601 or custom formats.

**Features**:
- **Smart Parsing**: Auto-detect 100+ date formats
- **Format Conversion**: ISO8601, US, EU, custom
- **Ambiguity Detection**: Flag dates that need clarification
- **Relative Dates**: Parse "today", "next week", etc.
- **CSV Processing**: Normalize entire columns
- **Validation**: Detect invalid dates

**Scripts**:
- `date_normalizer.py` - Date parsing and normalization with CLI

**Use for**: Data cleaning, ETL pipelines, log parsing, database imports

### 91. Model Comparison Tool

Compare multiple machine learning models with cross-validation and automated selection.

**Features**:
- **Multi-Model Testing**: Compare 5+ algorithms
- **Cross-Validation**: K-fold, stratified splits
- **Comprehensive Metrics**: Accuracy, F1, ROC-AUC, RMSE, R²
- **Statistical Testing**: Paired t-tests
- **Visualization**: Performance charts, ROC curves
- **Auto-Selection**: Best model recommendation

**Scripts**:
- `model_comparison_tool.py` - ML model comparison with CLI

**Use for**: Algorithm selection, hyperparameter tuning, model validation

### 92. Code Profiler

Analyze Python code performance and identify bottlenecks with detailed profiling.

**Features**:
- **Time Profiling**: Function execution time measurement
- **Line-by-Line Analysis**: Profile each code line
- **Call Statistics**: Function call counts and cumulative time
- **Report Export**: HTML and text reports
- **Top Functions**: Identify slowest functions
- **Script Profiling**: Profile entire Python scripts

**Scripts**:
- `code_profiler.py` - Python code profiling with CLI

**Use for**: Performance optimization, bottleneck identification, code analysis

### 93. QR/Barcode Reader

Decode and extract data from QR codes and barcodes in images.

**Features**:
- **Multiple Formats**: QR Code, EAN-13, Code128, Code39, UPC-A
- **Batch Processing**: Scan multiple images
- **Data Extraction**: Decode to text, URLs, structured data
- **Position Info**: Barcode location in image
- **Export**: JSON, CSV output
- **Error Handling**: Graceful failure reporting

**Scripts**:
- `qr_barcode_reader.py` - QR/barcode scanning with CLI

**Use for**: Inventory management, product lookup, event check-in, data entry

### 94. UUID Generator

Generate universally unique identifiers (UUIDs) in various formats.

**Features**:
- **Multiple Versions**: UUID1 (time-based), UUID4 (random), UUID5 (namespace)
- **Bulk Generation**: Generate thousands of UUIDs
- **Custom Formats**: Hyphenated, compact, URN
- **Namespace UUIDs**: Deterministic UUIDs from names
- **Validation**: Check UUID format and version
- **Export**: CSV, JSON, plain text

**Scripts**:
- `uuid_generator.py` - UUID generation with CLI

**Use for**: Database keys, API identifiers, session tokens, file naming

### 95. Statistical Power Calculator

Calculate statistical power and determine required sample sizes for experiments.

**Features**:
- **Power Calculation**: Calculate power for given sample size
- **Sample Size**: Determine required N for desired power
- **Effect Size**: Estimate detectable effect size
- **Multiple Tests**: t-test, proportion test, ANOVA
- **Visualizations**: Power curves, sample size charts
- **Reports**: Detailed analysis with recommendations

**Scripts**:
- `statistical_power_calculator.py` - Power analysis with CLI

**Use for**: Clinical trials, A/B test planning, research study sizing

### 96. KML/GeoJSON Converter

Convert geographic data between KML and GeoJSON formats.

**Features**:
- **Bidirectional Conversion**: KML ↔ GeoJSON
- **Feature Preservation**: Maintain properties and styles
- **Batch Processing**: Convert multiple files
- **Coordinate Systems**: WGS84, UTM support
- **Validation**: Verify output format
- **Simplification**: Reduce polygon complexity

**Scripts**:
- `kml_geojson_converter.py` - Geo format conversion with CLI

**Use for**: Google Maps integration, web mapping, GIS data interchange

### 97. Contract Generator

Generate professional legal contracts and agreements from templates with variable substitution.

**Features**:
- **Template System**: Pre-built contract templates in DOCX format
- **Variable Substitution**: Replace {{placeholders}} with actual values
- **Conditional Sections**: Include/exclude content based on variables
- **Validation**: Check for missing required fields
- **Batch Generation**: Create multiple contracts from CSV
- **Professional Formatting**: Maintain DOCX formatting and styles

**Scripts**:
- `contract_generator.py` - Contract generation with CLI

**Use for**: Employment agreements, NDAs, service contracts, lease agreements

### 98. Territory Mapper

Visualize sales territories, coverage areas, and service regions on interactive maps.

**Features**:
- **Territory Polygons**: Draw custom boundaries with coordinates
- **Color Coding**: Color by performance, team, or status
- **Interactive Maps**: Zoom, pan, tooltips with folium
- **GeoJSON Import**: Load territories from GeoJSON files
- **Data Overlay**: Add markers and additional layers
- **HTML Export**: Interactive HTML maps for sharing

**Scripts**:
- `territory_mapper.py` - Territory visualization with CLI

**Use for**: Sales planning, service coverage, market analysis, delivery zones

### 99. Batch QR Generator

Generate bulk QR codes from CSV data with UTM tracking and customization.

**Features**:
- **CSV Input**: Generate from spreadsheet data
- **UTM Tracking**: Auto-add campaign tracking parameters
- **Custom Styling**: Colors, logos, error correction
- **Sequential Naming**: Auto-generate filenames
- **Metadata Export**: CSV with QR data and filenames
- **Bulk Processing**: Generate thousands of QR codes

**Scripts**:
- `batch_qr_generator.py` - Bulk QR code generation with CLI

**Use for**: Event ticketing, product tracking, marketing campaigns, inventory management

### 100. Scientific Paper Figure Generator

Create publication-ready scientific figures with journal-compliant styling and formatting.

**Features**:
- **Journal Templates**: Nature, Science, IEEE style presets
- **Multi-Panel Figures**: Subfigures with labels (a, b, c)
- **High Resolution**: 300+ DPI for publication requirements
- **Vector Output**: EPS, PDF formats for scalability
- **Statistical Annotations**: p-values, error bars, significance markers
- **Consistent Styling**: Match journal requirements automatically

**Scripts**:
- `scientific_paper_figure_generator.py` - Publication figure creation with CLI

**Use for**: Research papers, conference presentations, thesis figures, grant proposals

## Skill Structure

Each skill follows the Claude skills format:

```
skill-name/
├── SKILL.md              # Core instructions (required)
│   ├── YAML frontmatter  # name + description
│   └── Markdown body     # Detailed instructions
├── scripts/              # Executable tools (optional)
│   ├── *.py             # Python scripts
│   └── *.js             # JavaScript/Node scripts
├── references/           # Documentation (optional)
│   └── *.md             # Specs, guides, examples
└── assets/              # Static resources (optional)
    ├── templates/       # Reusable templates
    └── config/          # Configuration files
```

### SKILL.md Format

```markdown
---
name: skill-name
description: Brief description of when and how to use this skill (200 chars max)
---

# Skill Name

Detailed instructions, workflows, and usage patterns...
```

## Using Skills with ChatGPT

1. **Package the skill**: Create a ZIP file containing the skill folder
2. **Load into ChatGPT**: Upload via the new ChatGPT skills interface
3. **Use naturally**: Simply describe your task - ChatGPT will load relevant skills automatically

Skills activate when ChatGPT determines they're relevant to your request, keeping context focused and efficient.

## Contributing

New skills are added regularly. When building a skill:

1. **Focus on workflows**: Design for complete tasks, not just API wrappers
2. **Optimize for context**: Return concise, high-signal information
3. **Write clear descriptions**: Help the AI understand when to activate your skill
4. **Test thoroughly**: Verify skills work across different use cases
5. **Document well**: Include examples, limitations, and best practices

See the [Skill Creator](#-skill-creator) skill for detailed guidance.

## Inspiration & References

This library is based on Anthropic's Claude Skills system:

- [What are Claude Skills?](https://support.claude.com/en/articles/12512176-what-are-skills) - Overview and concepts
- [How to Create Custom Skills](https://support.claude.com/en/articles/12512198-how-to-create-custom-skills) - Creation guide
- [Equipping Agents for the Real World with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) - Technical deep dive
- [Claude for Financial Services Skills](https://support.claude.com/en/articles/12663107-claude-for-financial-services-skills) - Example implementation

While designed for ChatGPT, these skills follow Claude's proven architecture for maximum compatibility and effectiveness.

## License

Individual skills may have specific licenses. See each skill's directory for details.

---

**Built with inspiration from [Claude Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)**
