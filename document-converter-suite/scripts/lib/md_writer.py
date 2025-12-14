#!/usr/bin/env python3
"""Markdown writer for document converter suite."""

from pathlib import Path
from typing import List, Optional


def write_md_from_sections(
    output_path: Path,
    sections: List[dict],
    title: Optional[str] = None
) -> None:
    """
    Write Markdown file from structured sections.

    Args:
        output_path: Path to output markdown file
        sections: List of section dictionaries with type and content
        title: Optional title to prepend as H1
    """
    lines = []

    # Add title if provided
    if title:
        lines.append(f"# {title}")
        lines.append("")

    for section in sections:
        section_type = section.get('type')

        if section_type == 'heading':
            level = section.get('level', 1)
            text = section.get('text', '')
            lines.append(f"{'#' * level} {text}")
            lines.append("")

        elif section_type == 'paragraph':
            text = section.get('text', '')
            if text:
                lines.append(text)
                lines.append("")

        elif section_type == 'list':
            items = section.get('items', [])
            ordered = section.get('ordered', False)

            for idx, item in enumerate(items, start=1):
                if ordered:
                    lines.append(f"{idx}. {item}")
                else:
                    lines.append(f"- {item}")
            lines.append("")

        elif section_type == 'table':
            headers = section.get('headers', [])
            rows = section.get('rows', [])

            if headers:
                # Write header row
                lines.append('| ' + ' | '.join(headers) + ' |')
                # Write separator row
                lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')

            # Write data rows
            for row in rows:
                # Pad row to match header length if needed
                padded_row = row + [''] * (len(headers) - len(row)) if headers else row
                lines.append('| ' + ' | '.join(str(cell) for cell in padded_row) + ' |')

            lines.append("")

        elif section_type == 'code':
            language = section.get('language', '')
            code = section.get('code', '')
            lines.append(f"```{language}")
            lines.append(code.rstrip())
            lines.append("```")
            lines.append("")

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
