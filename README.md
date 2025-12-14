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
