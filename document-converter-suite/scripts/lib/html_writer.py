#!/usr/bin/env python3
"""HTML writer for document converter suite."""

from pathlib import Path
from typing import List, Optional


DEFAULT_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    line-height: 1.6;
    max-width: 800px;
    margin: 40px auto;
    padding: 0 20px;
    color: #333;
}

h1, h2, h3, h4, h5, h6 {
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    line-height: 1.3;
}

h1 { font-size: 2em; border-bottom: 2px solid #eee; padding-bottom: 0.3em; }
h2 { font-size: 1.5em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }
h3 { font-size: 1.25em; }

p {
    margin-bottom: 1em;
}

ul, ol {
    margin-bottom: 1em;
    padding-left: 2em;
}

li {
    margin-bottom: 0.25em;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 1em;
    overflow-x: auto;
    display: block;
}

th, td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}

th {
    background-color: #f5f5f5;
    font-weight: 600;
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}

pre {
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 12px;
    overflow-x: auto;
    margin-bottom: 1em;
}

code {
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.9em;
}
"""


def write_html_from_sections(
    output_path: Path,
    sections: List[dict],
    title: Optional[str] = None,
    css: Optional[str] = None
) -> None:
    """
    Write HTML file from structured sections.

    Args:
        output_path: Path to output HTML file
        sections: List of section dictionaries with type and content
        title: Optional title for the document
        css: Optional custom CSS (if None, uses default styles)
    """
    if css is None:
        css = DEFAULT_CSS

    # Build HTML document
    html_parts = []

    # DOCTYPE and head
    html_parts.append('<!DOCTYPE html>')
    html_parts.append('<html lang="en">')
    html_parts.append('<head>')
    html_parts.append('    <meta charset="UTF-8">')
    html_parts.append('    <meta name="viewport" content="width=device-width, initial-scale=1.0">')

    if title:
        html_parts.append(f'    <title>{_escape_html(title)}</title>')
    else:
        html_parts.append('    <title>Document</title>')

    html_parts.append('    <style>')
    html_parts.append(css)
    html_parts.append('    </style>')
    html_parts.append('</head>')
    html_parts.append('<body>')

    # Add title as h1 if provided
    if title:
        html_parts.append(f'    <h1>{_escape_html(title)}</h1>')

    # Process sections
    for section in sections:
        section_type = section.get('type')

        if section_type == 'heading':
            level = section.get('level', 1)
            text = section.get('text', '')
            html_parts.append(f'    <h{level}>{_escape_html(text)}</h{level}>')

        elif section_type == 'paragraph':
            text = section.get('text', '')
            if text:
                html_parts.append(f'    <p>{_escape_html(text)}</p>')

        elif section_type == 'list':
            items = section.get('items', [])
            ordered = section.get('ordered', False)
            tag = 'ol' if ordered else 'ul'

            html_parts.append(f'    <{tag}>')
            for item in items:
                html_parts.append(f'        <li>{_escape_html(item)}</li>')
            html_parts.append(f'    </{tag}>')

        elif section_type == 'table':
            headers = section.get('headers', [])
            rows = section.get('rows', [])

            html_parts.append('    <table>')

            # Write headers
            if headers:
                html_parts.append('        <thead>')
                html_parts.append('            <tr>')
                for header in headers:
                    html_parts.append(f'                <th>{_escape_html(header)}</th>')
                html_parts.append('            </tr>')
                html_parts.append('        </thead>')

            # Write body
            if rows:
                html_parts.append('        <tbody>')
                for row in rows:
                    html_parts.append('            <tr>')
                    for cell in row:
                        html_parts.append(f'                <td>{_escape_html(str(cell))}</td>')
                    html_parts.append('            </tr>')
                html_parts.append('        </tbody>')

            html_parts.append('    </table>')

        elif section_type == 'code':
            language = section.get('language', '')
            code = section.get('code', '')
            lang_class = f' class="language-{language}"' if language else ''

            html_parts.append('    <pre>')
            html_parts.append(f'<code{lang_class}>{_escape_html(code)}</code>')
            html_parts.append('    </pre>')

    # Close HTML
    html_parts.append('</body>')
    html_parts.append('</html>')

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))
