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

4. **Capabilities-Based**: Integrated systems (e.g., crypto-ta-analyzer2)
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

### crypto-ta-analyzer2
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
