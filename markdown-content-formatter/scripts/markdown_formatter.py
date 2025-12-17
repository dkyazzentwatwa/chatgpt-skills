#!/usr/bin/env python3
"""
Markdown Content Formatter

Format and validate markdown documents:
- Auto-generate table of contents
- Add frontmatter (YAML/TOML/JSON)
- Validate structure (headings, links)
- Format code blocks and spacing
- Convert between markdown flavors
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import markdown
except ImportError:
    markdown = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


@dataclass
class ValidationIssue:
    """Validation issue."""
    type: str
    line: int
    message: str


@dataclass
class Heading:
    """Markdown heading."""
    level: int
    text: str
    anchor: str
    line: int


class MarkdownFormatter:
    """Format and validate markdown documents."""

    def __init__(self, file_path: str = None, content: str = None):
        """
        Initialize formatter.

        Args:
            file_path: Path to markdown file
            content: Direct markdown content

        One of file_path or content must be provided.
        """
        if not file_path and not content:
            raise ValueError("Either file_path or content must be provided")

        self.file_path = file_path
        self.content = content or self._read_file()
        self.headings = self._parse_headings()

    def _read_file(self) -> str:
        """Read markdown file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _parse_headings(self) -> List[Heading]:
        """Parse headings from markdown content."""
        headings = []
        lines = self.content.split('\n')

        for i, line in enumerate(lines, 1):
            # Match ATX-style headings (# Heading)
            match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                anchor = self._generate_anchor(text)

                headings.append(Heading(
                    level=level,
                    text=text,
                    anchor=anchor,
                    line=i
                ))

        return headings

    def _generate_anchor(self, heading: str) -> str:
        """
        Generate GitHub-style anchor from heading text.

        Args:
            heading: Heading text

        Returns:
            Anchor string (lowercase, dashes, no special chars)
        """
        # Remove markdown formatting
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', heading)  # Links
        text = re.sub(r'[*_`]', '', text)  # Bold, italic, code

        # Lowercase
        anchor = text.lower()

        # Remove special characters, keep alphanumeric and spaces
        anchor = re.sub(r'[^\w\s-]', '', anchor)

        # Replace spaces and underscores with dashes
        anchor = re.sub(r'[\s_]+', '-', anchor)

        # Remove leading/trailing dashes
        anchor = anchor.strip('-')

        return anchor

    def generate_toc(
        self,
        max_depth: int = 3,
        start_level: int = 2,
        style: str = 'github'
    ) -> str:
        """
        Generate table of contents from headings.

        Args:
            max_depth: Maximum heading level to include (1-6)
            start_level: Start from this heading level (e.g., 2 to skip H1)
            style: TOC style - 'github', 'numbered', or 'bullets'

        Returns:
            TOC markdown string
        """
        toc_lines = ['## Table of Contents', '']

        # Filter headings by level
        toc_headings = [
            h for h in self.headings
            if start_level <= h.level <= max_depth
        ]

        if not toc_headings:
            return ""

        # Determine base indent level
        base_level = min(h.level for h in toc_headings)

        if style == 'github':
            for h in toc_headings:
                indent = '  ' * (h.level - base_level)
                toc_lines.append(f"{indent}- [{h.text}](#{h.anchor})")

        elif style == 'numbered':
            counters = [0] * 7  # For H1-H6

            for h in toc_headings:
                # Increment counter at this level
                counters[h.level] += 1
                # Reset deeper levels
                for i in range(h.level + 1, 7):
                    counters[i] = 0

                # Build number string
                numbers = [str(counters[i]) for i in range(base_level, h.level + 1)]
                number_str = '.'.join(numbers)

                indent = '  ' * (h.level - base_level)
                toc_lines.append(f"{indent}{number_str}. [{h.text}](#{h.anchor})")

        elif style == 'bullets':
            for h in toc_headings:
                indent = '  ' * (h.level - base_level)
                toc_lines.append(f"{indent}* {h.text}")

        return '\n'.join(toc_lines)

    def add_frontmatter(
        self,
        metadata: Dict,
        format: str = 'yaml'
    ) -> str:
        """
        Add frontmatter to markdown content.

        Args:
            metadata: Dictionary of metadata
            format: 'yaml', 'toml', or 'json'

        Returns:
            Markdown content with frontmatter
        """
        if format == 'yaml':
            if not yaml:
                raise ImportError(
                    "pyyaml is required for YAML frontmatter. "
                    "Install with: pip install pyyaml"
                )

            frontmatter = "---\n"
            frontmatter += yaml.dump(metadata, default_flow_style=False, allow_unicode=True)
            frontmatter += "---\n\n"

        elif format == 'toml':
            # Simple TOML generation (without toml library)
            frontmatter = "+++\n"
            for key, value in metadata.items():
                if isinstance(value, str):
                    frontmatter += f'{key} = "{value}"\n'
                elif isinstance(value, bool):
                    frontmatter += f'{key} = {str(value).lower()}\n'
                elif isinstance(value, list):
                    frontmatter += f'{key} = {json.dumps(value)}\n'
                else:
                    frontmatter += f'{key} = {value}\n'
            frontmatter += "+++\n\n"

        elif format == 'json':
            frontmatter = json.dumps(metadata, indent=2)
            frontmatter = f"```json\n{frontmatter}\n```\n\n"

        else:
            raise ValueError(f"Unsupported format: {format}")

        # Remove existing frontmatter if present
        content = self.content
        if content.startswith('---\n'):
            # Remove YAML frontmatter
            parts = content.split('---\n', 2)
            if len(parts) >= 3:
                content = parts[2]
        elif content.startswith('+++\n'):
            # Remove TOML frontmatter
            parts = content.split('+++\n', 2)
            if len(parts) >= 3:
                content = parts[2]

        return frontmatter + content

    def validate_structure(self) -> Dict:
        """
        Validate markdown document structure.

        Returns:
            Dictionary with 'valid', 'errors', and 'warnings'
        """
        errors = []
        warnings = []

        # Check heading hierarchy
        prev_level = 0
        for h in self.headings:
            if prev_level > 0 and h.level > prev_level + 1:
                errors.append(ValidationIssue(
                    type='heading_skip',
                    line=h.line,
                    message=f"Heading level jumps from H{prev_level} to H{h.level} (skipped H{prev_level + 1})"
                ))
            prev_level = h.level

        # Check for duplicate headings
        heading_counts = {}
        for h in self.headings:
            key = h.text.lower()
            if key in heading_counts:
                warnings.append(ValidationIssue(
                    type='duplicate_heading',
                    line=h.line,
                    message=f'Heading "{h.text}" appears multiple times'
                ))
            heading_counts[key] = heading_counts.get(key, 0) + 1

        # Check for broken internal links
        valid_anchors = {h.anchor for h in self.headings}
        lines = self.content.split('\n')

        for i, line in enumerate(lines, 1):
            # Find internal links [text](#anchor)
            internal_links = re.findall(r'\[([^\]]+)\]\(#([^\)]+)\)', line)
            for text, anchor in internal_links:
                if anchor not in valid_anchors:
                    errors.append(ValidationIssue(
                        type='broken_link',
                        line=i,
                        message=f'Link to [#{anchor}] is broken (heading not found)'
                    ))

        return {
            'valid': len(errors) == 0,
            'errors': [{'type': e.type, 'line': e.line, 'message': e.message} for e in errors],
            'warnings': [{'type': w.type, 'line': w.line, 'message': w.message} for w in warnings]
        }

    def format_code_blocks(
        self,
        add_language_tags: bool = True,
        default_language: str = "text"
    ) -> str:
        """
        Format code blocks with language tags.

        Args:
            add_language_tags: Add language tags to untagged blocks
            default_language: Default language for untagged blocks

        Returns:
            Markdown with formatted code blocks
        """
        content = self.content
        lines = content.split('\n')
        result = []
        in_code_block = False
        code_block_lang = None

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for fenced code block
            if line.strip().startswith('```'):
                if not in_code_block:
                    # Starting code block
                    in_code_block = True
                    lang_match = re.match(r'^```(\w+)?', line.strip())
                    code_block_lang = lang_match.group(1) if lang_match and lang_match.group(1) else None

                    if add_language_tags and not code_block_lang:
                        result.append(f"```{default_language}")
                    else:
                        result.append(line)
                else:
                    # Ending code block
                    in_code_block = False
                    code_block_lang = None
                    result.append(line)
            else:
                result.append(line)

            i += 1

        return '\n'.join(result)

    def auto_link_headings(self) -> str:
        """
        Add heading IDs and create cross-reference links.

        Returns:
            Markdown with heading IDs
        """
        content = self.content
        lines = content.split('\n')
        result = []

        for line in lines:
            # Check if line is a heading
            match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if match:
                level = match.group(1)
                text = match.group(2).strip()
                anchor = self._generate_anchor(text)

                # Add anchor before heading
                result.append(f'<a id="{anchor}"></a>')
                result.append(line)
            else:
                result.append(line)

        return '\n'.join(result)

    def fix_spacing(self) -> str:
        """
        Apply consistent spacing rules.

        Returns:
            Markdown with fixed spacing
        """
        content = self.content
        lines = content.split('\n')
        result = []
        prev_line_type = None

        i = 0
        while i < len(lines):
            line = lines[i]
            line_type = self._get_line_type(line)

            # Determine spacing based on current and previous line types
            spacing_needed = 0

            if line_type == 'h1':
                spacing_needed = 2 if prev_line_type and prev_line_type != 'blank' else 0
            elif line_type in ['h2', 'h3', 'h4', 'h5', 'h6']:
                spacing_needed = 1 if prev_line_type and prev_line_type != 'blank' else 0
            elif line_type == 'code_block_start':
                spacing_needed = 1 if prev_line_type and prev_line_type != 'blank' else 0
            elif line_type == 'list':
                spacing_needed = 1 if prev_line_type and prev_line_type not in ['blank', 'list'] else 0

            # Add spacing if needed
            if spacing_needed > 0 and result and result[-1] != '':
                for _ in range(spacing_needed):
                    result.append('')

            result.append(line)

            # Update previous line type (skip blank lines)
            if line_type != 'blank':
                prev_line_type = line_type

            i += 1

        return '\n'.join(result)

    def _get_line_type(self, line: str) -> str:
        """Determine line type."""
        stripped = line.strip()

        if not stripped:
            return 'blank'
        elif re.match(r'^#{1}\s+', stripped):
            return 'h1'
        elif re.match(r'^#{2}\s+', stripped):
            return 'h2'
        elif re.match(r'^#{3}\s+', stripped):
            return 'h3'
        elif re.match(r'^#{4}\s+', stripped):
            return 'h4'
        elif re.match(r'^#{5}\s+', stripped):
            return 'h5'
        elif re.match(r'^#{6}\s+', stripped):
            return 'h6'
        elif stripped.startswith('```'):
            return 'code_block_start'
        elif re.match(r'^[\*\-\+]\s+', stripped) or re.match(r'^\d+\.\s+', stripped):
            return 'list'
        else:
            return 'text'

    def convert_to_flavor(self, target: str) -> str:
        """
        Convert markdown to target flavor.

        Args:
            target: 'github', 'commonmark', 'jekyll', or 'hugo'

        Returns:
            Converted markdown
        """
        content = self.content

        if target == 'github':
            # GitHub Flavored Markdown
            # Preserve task lists, tables, strikethrough
            pass  # GFM is default

        elif target == 'commonmark':
            # CommonMark - remove extensions
            # Remove task lists
            content = re.sub(r'- \[[x ]\] ', '- ', content)

        elif target == 'jekyll':
            # Jekyll - add Liquid compatibility notes
            # Preserve liquid tags {{ }} and {% %}
            pass

        elif target == 'hugo':
            # Hugo - preserve shortcodes
            # Preserve shortcodes {{< >}} and {{< />}}
            pass

        return content

    def export(
        self,
        output_path: str,
        include_toc: bool = True,
        include_frontmatter: bool = True,
        target_flavor: str = 'github'
    ):
        """
        Export formatted markdown.

        Args:
            output_path: Output file path
            include_toc: Add table of contents
            include_frontmatter: Preserve frontmatter
            target_flavor: Target markdown flavor
        """
        content = self.content

        # Apply formatting
        content = self.fix_spacing()
        content = self.format_code_blocks()

        # Add TOC
        if include_toc:
            toc = self.generate_toc()
            if toc:
                # Insert TOC after frontmatter or at beginning
                if content.startswith('---\n') or content.startswith('+++\n'):
                    delimiter = '---' if content.startswith('---') else '+++'
                    parts = content.split(f'{delimiter}\n', 2)
                    if len(parts) >= 3:
                        content = f"{delimiter}\n{parts[1]}{delimiter}\n\n{toc}\n\n{parts[2]}"
                else:
                    content = f"{toc}\n\n{content}"

        # Convert flavor
        content = self.convert_to_flavor(target_flavor)

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Format and validate markdown documents'
    )
    parser.add_argument('--input', '-i', required=True, help='Input markdown file')
    parser.add_argument('--output', '-o', help='Output file path')

    # TOC options
    parser.add_argument('--toc', action='store_true', help='Generate table of contents')
    parser.add_argument('--toc-depth', type=int, default=3, help='Max TOC depth (1-6)')
    parser.add_argument('--toc-style', choices=['github', 'numbered', 'bullets'], default='github')

    # Frontmatter options
    parser.add_argument('--frontmatter', nargs='+', help='Frontmatter key=value pairs')
    parser.add_argument('--frontmatter-file', help='YAML file with frontmatter')

    # Formatting options
    parser.add_argument('--auto-link', action='store_true', help='Auto-link headings')
    parser.add_argument('--fix-spacing', action='store_true', help='Fix spacing')
    parser.add_argument('--flavor', choices=['github', 'commonmark', 'jekyll', 'hugo'], default='github')

    # Validation
    parser.add_argument('--validate', action='store_true', help='Validate structure only')
    parser.add_argument('--format', choices=['json', 'text'], default='text', help='Validation output format')

    args = parser.parse_args()

    try:
        # Load formatter
        formatter = MarkdownFormatter(file_path=args.input)

        if args.validate:
            # Validation mode
            result = formatter.validate_structure()

            if args.format == 'json':
                print(json.dumps(result, indent=2))
            else:
                if result['valid']:
                    print("✓ Document structure is valid")
                else:
                    print("✗ Document has issues:\n")
                    for error in result['errors']:
                        print(f"  Error (line {error['line']}): {error['message']}")
                    for warning in result['warnings']:
                        print(f"  Warning (line {warning['line']}): {warning['message']}")

            sys.exit(0 if result['valid'] else 1)

        # Formatting mode
        content = formatter.content

        # Apply operations
        if args.fix_spacing:
            content = formatter.fix_spacing()
            formatter.content = content

        if args.auto_link:
            content = formatter.auto_link_headings()
            formatter.content = content

        # Add frontmatter
        if args.frontmatter or args.frontmatter_file:
            metadata = {}

            if args.frontmatter_file:
                if not yaml:
                    raise ImportError("pyyaml required for frontmatter file. Install with: pip install pyyaml")
                with open(args.frontmatter_file, 'r') as f:
                    metadata = yaml.safe_load(f)

            if args.frontmatter:
                for pair in args.frontmatter:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        metadata[key] = value

            content = formatter.add_frontmatter(metadata, format='yaml')
            formatter.content = content

        # Add TOC
        if args.toc:
            toc = formatter.generate_toc(max_depth=args.toc_depth, style=args.toc_style)
            if toc:
                content = f"{toc}\n\n{content}"

        # Convert flavor
        content = formatter.convert_to_flavor(args.flavor)

        # Output
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Formatted markdown written to {args.output}")
        else:
            print(content)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
