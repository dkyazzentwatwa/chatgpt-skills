# Claude Agent Skills Research Notes

Research gathered from official Claude/Anthropic documentation to inform the ChatGPT skills library design.

---

## What Are Claude Skills?

**Source**: [https://support.claude.com/en/articles/12512176-what-are-skills](https://support.claude.com/en/articles/12512176-what-are-skills)

### Definition

Claude Skills are dynamic instruction folders that enhance Claude's performance on specialized tasks. According to the documentation: "Skills are folders of instructions, scripts, and resources that Claude loads dynamically to improve performance on specialized tasks."

### How They Work

Skills operate through a **progressive disclosure mechanism**. When you request assistance, Claude:
1. Evaluates available Skills
2. Activates the relevant ones
3. Implements their guidance

This approach prevents context window saturation while ensuring task-specific optimization.

### Key Features

**Two Types of Skills:**

1. **Anthropic Skills**: Pre-built capabilities for document creation (Excel, Word, PowerPoint, PDF) that activate automatically
2. **Custom Skills**: User or organization-created procedures for:
   - Branded workflows
   - Email templates
   - Data analysis
   - Task management

**Main Benefits:**

1. Enhanced performance on specialized tasks
2. Organizational knowledge preservation through packaged workflows
3. Accessible customization requiring only Markdown skills for basic implementations

**Availability:**

- Feature preview for Pro, Max, Team, and Enterprise plan users (requires code execution enabled)
- Beta access for Claude Code users
- All API users with code execution tools

**Distinction from Other Features:**

- **vs Projects**: Skills activate conditionally, while Projects have static knowledge always loaded
- **vs MCP**: Skills provide procedural instructions rather than external connections
- **vs Custom Instructions**: Skills remain task-specific rather than applying broadly

---

## How to Create Custom Claude Skills

**Source**: [https://support.claude.com/en/articles/12512198-how-to-create-custom-skills](https://support.claude.com/en/articles/12512198-how-to-create-custom-skills)

### Core Structure

Every Skill requires a directory with a **SKILL.md file** as its foundation. This file must begin with YAML frontmatter containing required metadata fields.

### Required Metadata

**Name**: A human-friendly identifier limited to 64 characters.

**Description**: A concise explanation of the Skill's purpose and when to use it (200 character maximum). As the documentation states: "Claude uses this to determine when to invoke your Skill."

### Optional Elements

- **Dependencies**: Software packages needed for execution
- **Markdown body**: Detailed instructions and reference information
- **Additional files**: Supplementary resources like REFERENCE.md files
- **Executable scripts**: Python, JavaScript/Node.js, or other code files for advanced functionality

### Packaging Requirements

The folder structure must follow these guidelines:

- Folder name matches the Skill name
- Create a ZIP file containing the folder as its root (not files directly in the ZIP)
- Include all referenced resources in correct locations

### Testing Process

Before uploading:
1. Verify clarity and accuracy
2. After uploading, enable the Skill in Settings > Capabilities
3. Test with relevant prompts
4. Review Claude's thinking to confirm proper loading

### Best Practices

1. **Keep Skills focused** on single workflows rather than attempting multiple tasks
2. **Write clear descriptions** to help Claude understand when to activate
3. **Start simple** with Markdown-based instructions before adding complex code
4. **Include examples** of inputs/outputs for clarity

---

## Claude Agent Skills: Technical Architecture

**Source**: [https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)

### Core Structure

Agent Skills are organized directories containing a `SKILL.md` file with YAML frontmatter specifying `name` and `description` metadata. As described in the article, these skills represent "organized folders of instructions, scripts, and resources that agents can discover and load dynamically to perform better at specific tasks."

### Progressive Disclosure Design

The architecture employs a **tiered information model**:

**1. Metadata Level**:
- Skill names and descriptions load into the system prompt at startup
- Enables Claude to recognize when skills are relevant

**2. Core Level**:
- Full `SKILL.md` content loads when Claude determines relevance to the current task

**3. Extended Level**:
- Additional bundled files (like `reference.md` or `forms.md`) load on-demand as needed

This design "lets Claude load information only as needed," effectively unbounding the context that skills can contain.

### Code Execution Integration

Skills can include executable code scripts. The system allows Claude to:
- Run deterministic operations (such as PDF form field extraction)
- Execute without loading the actual script or processed files into context
- Prioritize efficiency and reliability over token-based generation

### Practical Extension Example

The PDF skill demonstrates real-world application:
- While Claude understands PDFs conceptually, the skill grants it practical manipulation capabilities
- Form-filling through bundled Python scripts and structured instructions
- Separation of knowledge (understanding PDFs) from capability (manipulating them)

### Current Platform Support

Agent Skills are supported across:
- Claude.ai
- Claude Code
- Claude Agent SDK
- Claude Developer Platform

---

## Key Takeaways for ChatGPT Skills

### Architecture Principles

1. **Progressive Disclosure**: Only load what's needed when it's needed
2. **Clear Metadata**: Name and description are critical for skill activation
3. **Structured Organization**: Separate instructions, scripts, and references
4. **Focus**: One skill, one workflow (avoid scope creep)

### File Organization

```
skill-name/
├── SKILL.md              # Required: Core instructions + metadata
├── scripts/              # Optional: Executable tools
├── references/           # Optional: Supporting docs
└── assets/              # Optional: Templates, configs
```

### YAML Frontmatter Requirements

```yaml
---
name: skill-name
description: Brief, clear description of when to use (max 200 chars)
---
```

### Design Guidelines

**DO:**
- Write clear, specific descriptions for skill activation
- Keep skills focused on single workflows
- Use progressive disclosure (metadata → core → extended)
- Include executable scripts for deterministic operations
- Provide examples and templates
- Document limitations and best practices

**DON'T:**
- Create kitchen-sink skills that do too much
- Assume skills are always loaded (they activate conditionally)
- Skip examples and usage patterns
- Forget to test with realistic scenarios
- Neglect to document when NOT to use the skill

### Context Optimization

Claude's progressive disclosure system teaches us:
- Metadata is always visible (keep it concise)
- Core instructions load when relevant (be thorough but focused)
- Extended resources load on-demand (deep reference material belongs here)
- Scripts execute without consuming context (use for heavy lifting)

---

## Application to ChatGPT Skills

While ChatGPT's skill system is new and may differ in implementation details, Claude's proven architecture provides excellent design principles:

1. **Structured folders** with clear metadata
2. **Progressive information** from description → instructions → references
3. **Executable scripts** for complex operations
4. **Workflow-focused** design
5. **Clear activation criteria** in descriptions

These principles make skills more effective, maintainable, and easier for the AI to use correctly.
