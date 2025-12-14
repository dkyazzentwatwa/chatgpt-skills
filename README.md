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
**`crypto-ta-analyzer2/`**

Comprehensive cryptocurrency and stock technical analysis using 24+ proven indicators. Integrates with CoinGecko for real-time market data and generates scored trading signals (STRONG_UPTREND/NEUTRAL/DOWNTREND) based on consensus across RSI, MACD, moving averages, momentum oscillators, and volume indicators.

**Use for**: Price analysis, trading signals, trend identification, multi-indicator technical assessments

### 📄 Document Converter Suite
**`document-converter-suite/`**

Convert between PDF, DOCX, PPTX, and XLSX formats with best-effort text-first extraction. Handles single files or batch conversions while prioritizing clean, structured output over visual fidelity.

**Use for**: Document format conversion, content extraction, batch processing

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
