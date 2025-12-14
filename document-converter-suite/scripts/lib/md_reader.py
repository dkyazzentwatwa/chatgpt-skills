#!/usr/bin/env python3
"""Markdown reader for document converter suite."""

import mistune
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class MdContent:
    """Markdown document content."""
    title: Optional[str]
    sections: List[dict] = field(default_factory=list)


class StructuredMarkdownRenderer(mistune.HTMLRenderer):
    """Custom mistune renderer that builds structured sections instead of HTML."""

    def __init__(self):
        super().__init__()
        self.sections = []
        self._list_items = []
        self._list_ordered = False
        self._current_table_headers = []
        self._current_table_rows = []

    def heading(self, text, level, **attrs):
        """Handle heading elements."""
        self.sections.append({
            'type': 'heading',
            'level': level,
            'text': text
        })
        return ''

    def paragraph(self, text):
        """Handle paragraph elements."""
        self.sections.append({
            'type': 'paragraph',
            'text': text
        })
        return ''

    def list(self, text, ordered, **attrs):
        """Handle list elements."""
        if self._list_items:
            self.sections.append({
                'type': 'list',
                'ordered': ordered,
                'items': self._list_items[:]
            })
            self._list_items = []
        return ''

    def list_item(self, text):
        """Handle list item elements."""
        # Strip any HTML tags that might be in the text
        clean_text = text.strip()
        self._list_items.append(clean_text)
        return ''

    def block_code(self, code, info=None):
        """Handle code block elements."""
        self.sections.append({
            'type': 'code',
            'language': info or '',
            'code': code
        })
        return ''

    def table(self, text):
        """Handle table elements."""
        if self._current_table_rows:
            self.sections.append({
                'type': 'table',
                'headers': self._current_table_headers[:],
                'rows': self._current_table_rows[:]
            })
            self._current_table_headers = []
            self._current_table_rows = []
        return ''

    def table_head(self, text):
        """Handle table header."""
        return text

    def table_body(self, text):
        """Handle table body."""
        return text

    def table_row(self, text):
        """Handle table row."""
        return text

    def table_cell(self, text, align=None, is_head=False):
        """Handle table cell."""
        # This is called for each cell - we need to accumulate them
        # mistune calls this in order, so we track state
        clean_text = text.strip()

        if is_head:
            self._current_table_headers.append(clean_text)
        else:
            # For body cells, we need to group them into rows
            # This is a simplified approach - we'll append to the last row
            # or create a new row if needed
            if not self._current_table_rows or len(self._current_table_rows[-1]) >= len(self._current_table_headers):
                self._current_table_rows.append([])
            self._current_table_rows[-1].append(clean_text)

        return ''

    def block_quote(self, text):
        """Handle block quote elements."""
        self.sections.append({
            'type': 'paragraph',
            'text': f"> {text}"
        })
        return ''

    def thematic_break(self):
        """Handle horizontal rule."""
        return ''

    # Inline elements - just return the text
    def emphasis(self, text):
        return text

    def strong(self, text):
        return text

    def link(self, text, url, title=None):
        return f"{text} ({url})" if url else text

    def image(self, alt, url, title=None):
        return f"[Image: {alt or url}]"

    def codespan(self, text):
        return f"`{text}`"


def read_md_content(path: Path, max_chars: int = 300000) -> MdContent:
    """
    Read Markdown file and parse into structured sections.

    Args:
        path: Path to markdown file
        max_chars: Maximum characters to read (default 300,000)

    Returns:
        MdContent with title and sections
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read(max_chars)

    # Create custom renderer
    renderer = StructuredMarkdownRenderer()
    markdown = mistune.create_markdown(renderer=renderer)

    # Parse markdown
    markdown(text)

    # Extract title (first H1 if present)
    title = None
    for section in renderer.sections:
        if section.get('type') == 'heading' and section.get('level') == 1:
            title = section['text']
            break

    return MdContent(title=title, sections=renderer.sections)
