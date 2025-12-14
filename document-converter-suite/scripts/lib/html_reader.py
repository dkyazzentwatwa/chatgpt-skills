#!/usr/bin/env python3
"""HTML reader for document converter suite."""

from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class HtmlContent:
    """HTML document content."""
    title: Optional[str]
    sections: List[dict] = field(default_factory=list)


def read_html_content(path: Path, max_chars: int = 300000) -> HtmlContent:
    """
    Read HTML file and parse into structured sections.

    Args:
        path: Path to HTML file
        max_chars: Maximum characters to read (default 300,000)

    Returns:
        HtmlContent with title and sections
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        html = f.read(max_chars)

    soup = BeautifulSoup(html, 'lxml')

    # Extract title from <title> tag or first <h1>
    title = None
    title_tag = soup.find('title')
    if title_tag and title_tag.string:
        title = title_tag.string.strip()

    # Get body content, or fallback to entire document
    body = soup.find('body')
    if not body:
        body = soup

    sections = []

    # Process elements in order
    for element in body.descendants:
        if not hasattr(element, 'name') or element.name is None:
            continue

        # Headings
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            level = int(element.name[1])
            text = element.get_text(strip=True)
            if text:
                sections.append({
                    'type': 'heading',
                    'level': level,
                    'text': text
                })
                # Set title from first h1 if not already set
                if not title and level == 1:
                    title = text

        # Paragraphs
        elif element.name == 'p':
            text = element.get_text(strip=True)
            if text:
                sections.append({
                    'type': 'paragraph',
                    'text': text
                })

        # Lists
        elif element.name in ['ul', 'ol']:
            ordered = element.name == 'ol'
            items = []
            for li in element.find_all('li', recursive=False):
                item_text = li.get_text(strip=True)
                if item_text:
                    items.append(item_text)

            if items:
                sections.append({
                    'type': 'list',
                    'ordered': ordered,
                    'items': items
                })

        # Tables
        elif element.name == 'table':
            headers = []
            rows = []

            # Extract headers from <thead> or first <tr>
            thead = element.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    for th in header_row.find_all(['th', 'td']):
                        headers.append(th.get_text(strip=True))
            else:
                # Check if first row looks like headers
                first_row = element.find('tr')
                if first_row:
                    first_cells = first_row.find_all('th')
                    if first_cells:
                        for th in first_cells:
                            headers.append(th.get_text(strip=True))

            # Extract body rows
            tbody = element.find('tbody')
            row_container = tbody if tbody else element

            for tr in row_container.find_all('tr'):
                # Skip header row if it was already processed
                if headers and tr == element.find('tr'):
                    first_cells = tr.find_all('th')
                    if first_cells:
                        continue

                row = []
                for td in tr.find_all(['td', 'th']):
                    row.append(td.get_text(strip=True))

                if row:
                    rows.append(row)

            if headers or rows:
                sections.append({
                    'type': 'table',
                    'headers': headers,
                    'rows': rows
                })

        # Code blocks
        elif element.name == 'pre':
            code_element = element.find('code')
            code_text = code_element.get_text() if code_element else element.get_text()

            # Try to detect language from class (common pattern: class="language-python")
            language = ''
            if code_element and code_element.get('class'):
                classes = code_element.get('class')
                for cls in classes:
                    if cls.startswith('language-'):
                        language = cls.replace('language-', '')
                        break

            if code_text.strip():
                sections.append({
                    'type': 'code',
                    'language': language,
                    'code': code_text.rstrip()
                })

    return HtmlContent(title=title, sections=sections)
