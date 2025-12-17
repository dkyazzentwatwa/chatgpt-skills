#!/usr/bin/env python3
"""
Citation & Bibliography Generator

Generate properly formatted citations in multiple academic styles:
- APA, MLA, Chicago, IEEE, Harvard
- Manual entry or automatic DOI/ISBN lookup
- Bibliography management with auto-sort and deduplication
- BibTeX export for LaTeX documents
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Union
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import requests
except ImportError:
    requests = None


@dataclass
class Citation:
    """Data class for citation information."""
    citation_type: str  # book, article, website
    authors: List[str]
    title: str
    year: int
    formatted: str = ""
    # Book-specific
    publisher: str = ""
    city: str = ""
    edition: str = ""
    isbn: str = ""
    # Article-specific
    journal: str = ""
    volume: Optional[int] = None
    issue: Optional[int] = None
    pages: str = ""
    doi: str = ""
    # Website-specific
    url: str = ""
    access_date: str = ""
    publish_date: str = ""


class CitationGenerator:
    """Generate formatted citations in multiple academic styles."""

    SUPPORTED_STYLES = ['apa', 'mla', 'chicago', 'ieee', 'harvard']

    def __init__(self, style: str = 'apa'):
        """
        Initialize citation generator.

        Args:
            style: Citation style - 'apa', 'mla', 'chicago', 'ieee', or 'harvard'
        """
        if style.lower() not in self.SUPPORTED_STYLES:
            raise ValueError(f"Unsupported style: {style}. Choose from {self.SUPPORTED_STYLES}")

        self.style = style.lower()
        self.bibliography: List[Citation] = []

    def _format_authors(self, authors: List[str], max_display: int = None, style: str = None) -> str:
        """
        Format author names according to citation style.

        Args:
            authors: List of author names in "Last, First" format
            max_display: Maximum authors to display before et al.
            style: Override style (for in-text citations)

        Returns:
            Formatted author string
        """
        if not authors:
            return ""

        citation_style = style or self.style

        # Parse authors
        parsed = []
        for author in authors:
            if ',' in author:
                parts = author.split(',', 1)
                last = parts[0].strip()
                first = parts[1].strip() if len(parts) > 1 else ""
            else:
                parts = author.strip().split()
                last = parts[-1] if parts else ""
                first = ' '.join(parts[:-1]) if len(parts) > 1 else ""

            parsed.append({'last': last, 'first': first})

        # Apply et al. rules if max_display is set
        if max_display and len(parsed) > max_display:
            parsed = parsed[:max_display]
            use_et_al = True
        else:
            use_et_al = False

        # Format based on style
        if citation_style == 'apa':
            # Last, F. M.
            formatted = []
            for p in parsed:
                initials = ''.join([f"{n[0]}." for n in p['first'].split() if n])
                formatted.append(f"{p['last']}, {initials}" if initials else p['last'])

            if len(formatted) == 1:
                result = formatted[0]
            elif len(formatted) == 2:
                result = f"{formatted[0]}, & {formatted[1]}"
            else:
                result = ', '.join(formatted[:-1]) + f", & {formatted[-1]}"

            if use_et_al:
                result += ", et al."

        elif citation_style == 'mla':
            # Last, First M.
            formatted = []
            for i, p in enumerate(parsed):
                if i == 0:
                    # First author: Last, First
                    formatted.append(f"{p['last']}, {p['first']}" if p['first'] else p['last'])
                else:
                    # Subsequent: First Last
                    formatted.append(f"{p['first']} {p['last']}" if p['first'] else p['last'])

            if len(formatted) == 1:
                result = formatted[0]
            elif len(formatted) == 2:
                result = f"{formatted[0]}, and {formatted[1]}"
            else:
                result = ', '.join(formatted[:-1]) + f", and {formatted[-1]}"

            if use_et_al:
                result = formatted[0] + ", et al."

        elif citation_style == 'chicago':
            # Last, First
            formatted = []
            for p in parsed:
                formatted.append(f"{p['last']}, {p['first']}" if p['first'] else p['last'])

            if len(formatted) == 1:
                result = formatted[0]
            elif len(formatted) == 2:
                result = f"{formatted[0]}, and {formatted[1]}"
            else:
                result = ', '.join(formatted[:-1]) + f", and {formatted[-1]}"

            if use_et_al:
                result += ", et al."

        elif citation_style == 'ieee':
            # F. M. Last
            formatted = []
            for p in parsed:
                initials = ''.join([f"{n[0]}. " for n in p['first'].split() if n])
                formatted.append(f"{initials}{p['last']}" if initials else p['last'])

            if len(formatted) <= 6:
                result = ', '.join(formatted)
            else:
                result = ', '.join(formatted[:1]) + ", et al."

        elif citation_style == 'harvard':
            # Last, F.M.
            formatted = []
            for p in parsed:
                initials = ''.join([f"{n[0]}." for n in p['first'].split() if n])
                formatted.append(f"{p['last']}, {initials}" if initials else p['last'])

            if len(formatted) == 1:
                result = formatted[0]
            elif len(formatted) == 2:
                result = f"{formatted[0]} and {formatted[1]}"
            else:
                result = ', '.join(formatted[:-1]) + f" and {formatted[-1]}"

            if use_et_al:
                result += " et al."

        else:
            result = ', '.join([p['last'] for p in parsed])

        return result

    def cite_book(
        self,
        authors: List[str],
        title: str,
        year: int,
        publisher: str,
        city: str = "",
        edition: str = "",
        isbn: str = ""
    ) -> str:
        """
        Format book citation.

        Args:
            authors: List of author names in "Last, First" format
            title: Book title
            year: Publication year
            publisher: Publisher name
            city: City of publication (optional)
            edition: Edition (e.g., "2nd ed.") (optional)
            isbn: ISBN number (optional)

        Returns:
            Formatted citation string
        """
        author_str = self._format_authors(authors)

        if self.style == 'apa':
            # Author, A. A. (Year). Title of book (Edition). Publisher.
            citation = f"{author_str} ({year}). "

            # Capitalize only first word and proper nouns (sentence case)
            title_formatted = title[0].upper() + title[1:].lower() if title else title
            # But preserve capitalization after colons
            if ':' in title_formatted:
                parts = title_formatted.split(':', 1)
                title_formatted = f"{parts[0]}: {parts[1].strip()[0].upper()}{parts[1].strip()[1:]}"

            citation += f"{title_formatted}"
            if edition:
                citation += f" ({edition})"
            citation += f". {publisher}."

        elif self.style == 'mla':
            # Author. Title of Book. Edition, Publisher, Year.
            citation = f"{author_str}. "
            citation += f"{title}"
            if edition:
                citation += f". {edition}"
            citation += f", {publisher}, {year}."

        elif self.style == 'chicago':
            # Author. Title of Book. Edition. City: Publisher, Year.
            citation = f"{author_str}. "
            citation += f"{title}."
            if edition:
                citation += f" {edition}."
            if city:
                citation += f" {city}: {publisher}, {year}."
            else:
                citation += f" {publisher}, {year}."

        elif self.style == 'ieee':
            # A. A. Author, Title of Book, Edition. City: Publisher, Year.
            citation = f"{author_str}, "
            citation += f"{title}"
            if edition:
                citation += f", {edition}"
            if city:
                citation += f". {city}: {publisher}, {year}."
            else:
                citation += f". {publisher}, {year}."

        elif self.style == 'harvard':
            # Author, A.A., Year. Title of book. Edition. City: Publisher.
            citation = f"{author_str}, {year}. "
            citation += f"{title}."
            if edition:
                citation += f" {edition}."
            if city:
                citation += f" {city}: {publisher}."
            else:
                citation += f" {publisher}."

        return citation

    def cite_article(
        self,
        authors: List[str],
        title: str,
        journal: str,
        year: int,
        volume: Optional[int] = None,
        issue: Optional[int] = None,
        pages: str = "",
        doi: str = ""
    ) -> str:
        """
        Format journal article citation.

        Args:
            authors: List of author names
            title: Article title
            journal: Journal name
            year: Publication year
            volume: Volume number (optional)
            issue: Issue number (optional)
            pages: Page range (e.g., "45-67") (optional)
            doi: DOI identifier (optional)

        Returns:
            Formatted citation string
        """
        author_str = self._format_authors(authors)

        if self.style == 'apa':
            # Author, A. A. (Year). Title of article. Journal Name, Volume(Issue), pages. https://doi.org/xxx
            citation = f"{author_str} ({year}). "
            citation += f"{title}. "
            citation += f"{journal}"
            if volume:
                citation += f", {volume}"
                if issue:
                    citation += f"({issue})"
            if pages:
                citation += f", {pages}"
            citation += "."
            if doi:
                citation += f" https://doi.org/{doi}"

        elif self.style == 'mla':
            # Author. "Title of Article." Journal Name, vol. X, no. Y, Year, pp. XX-YY. DOI: xxx.
            citation = f"{author_str}. "
            citation += f'"{title}." '
            citation += f"{journal}"
            if volume:
                citation += f", vol. {volume}"
            if issue:
                citation += f", no. {issue}"
            citation += f", {year}"
            if pages:
                citation += f", pp. {pages}"
            citation += "."
            if doi:
                citation += f" DOI: {doi}."

        elif self.style == 'chicago':
            # Author. "Title of Article." Journal Name Volume, no. Issue (Year): pages. https://doi.org/xxx.
            citation = f"{author_str}. "
            citation += f'"{title}." '
            citation += f"{journal}"
            if volume:
                citation += f" {volume}"
            if issue:
                citation += f", no. {issue}"
            citation += f" ({year})"
            if pages:
                citation += f": {pages}"
            citation += "."
            if doi:
                citation += f" https://doi.org/{doi}."

        elif self.style == 'ieee':
            # A. A. Author, "Title of article," Journal Name, vol. X, no. Y, pp. XX-YY, Year. DOI: xxx.
            citation = f"{author_str}, "
            citation += f'"{title}," '
            citation += f"{journal}"
            if volume:
                citation += f", vol. {volume}"
            if issue:
                citation += f", no. {issue}"
            if pages:
                citation += f", pp. {pages}"
            citation += f", {year}."
            if doi:
                citation += f" DOI: {doi}."

        elif self.style == 'harvard':
            # Author, A.A., Year. Title of article. Journal Name, Volume(Issue), pp.XX-YY.
            citation = f"{author_str}, {year}. "
            citation += f"{title}. "
            citation += f"{journal}"
            if volume:
                citation += f", {volume}"
                if issue:
                    citation += f"({issue})"
            if pages:
                citation += f", pp.{pages}"
            citation += "."

        return citation

    def cite_website(
        self,
        authors: List[str],
        title: str,
        url: str,
        access_date: str,
        publish_date: str = ""
    ) -> str:
        """
        Format website citation.

        Args:
            authors: List of author names (can be empty)
            title: Page title
            url: URL
            access_date: Date accessed (YYYY-MM-DD format)
            publish_date: Date published (optional)

        Returns:
            Formatted citation string
        """
        author_str = self._format_authors(authors) if authors else ""

        if self.style == 'apa':
            # Author, A. A. (Year, Month Day). Title. Site Name. URL
            citation = ""
            if author_str:
                citation += f"{author_str} "
            if publish_date:
                citation += f"({publish_date}). "
            citation += f"{title}. "
            citation += f"Retrieved {access_date}, from {url}"

        elif self.style == 'mla':
            # Author. "Title." Website Name. Date, URL. Accessed Day Month Year.
            citation = ""
            if author_str:
                citation += f"{author_str}. "
            citation += f'"{title}." '
            if publish_date:
                citation += f"{publish_date}. "
            citation += f"{url}. "
            citation += f"Accessed {access_date}."

        elif self.style == 'chicago':
            # Author. "Title." Website Name. Last modified Date. Accessed Date. URL.
            citation = ""
            if author_str:
                citation += f"{author_str}. "
            citation += f'"{title}." '
            if publish_date:
                citation += f"Last modified {publish_date}. "
            citation += f"Accessed {access_date}. "
            citation += f"{url}."

        elif self.style == 'ieee':
            # A. A. Author, "Title," Website Name. URL (accessed: Month Day, Year).
            citation = ""
            if author_str:
                citation += f"{author_str}, "
            citation += f'"{title}," '
            citation += f"{url} "
            citation += f"(accessed: {access_date})."

        elif self.style == 'harvard':
            # Author, A.A., Year. Title. [online] Available at: URL [Accessed Date].
            citation = ""
            if author_str:
                citation += f"{author_str}, "
            year = publish_date.split('-')[0] if publish_date else access_date.split('-')[0]
            citation += f"{year}. "
            citation += f"{title}. [online] "
            citation += f"Available at: {url} "
            citation += f"[Accessed {access_date}]."

        return citation

    def cite_from_doi(self, doi: str) -> str:
        """
        Look up article metadata from DOI and generate citation.

        Args:
            doi: DOI identifier

        Returns:
            Formatted citation string
        """
        if not requests:
            raise ImportError(
                "requests library is required for DOI lookup. "
                "Install with: pip install requests"
            )

        url = f"https://api.crossref.org/works/{doi}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()['message']

            # Extract metadata
            authors = []
            for author in data.get('author', []):
                last = author.get('family', '')
                first = author.get('given', '')
                if last:
                    authors.append(f"{last}, {first}" if first else last)

            title = data.get('title', [''])[0] if 'title' in data else ''
            year = data.get('published', {}).get('date-parts', [[0]])[0][0]

            # Determine type and format
            if data.get('type') == 'journal-article':
                journal = data.get('container-title', [''])[0] if 'container-title' in data else ''
                volume = data.get('volume')
                issue = data.get('issue')
                pages = data.get('page', '')

                return self.cite_article(
                    authors=authors,
                    title=title,
                    journal=journal,
                    year=year,
                    volume=int(volume) if volume else None,
                    issue=int(issue) if issue else None,
                    pages=pages,
                    doi=doi
                )
            else:
                # Default to article format
                return self.cite_article(
                    authors=authors,
                    title=title,
                    journal='',
                    year=year,
                    doi=doi
                )

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to retrieve DOI {doi}: {e}")

    def add_to_bibliography(self, citation: Union[str, Citation]):
        """
        Add citation to bibliography.

        Args:
            citation: Citation string or Citation object
        """
        if isinstance(citation, str):
            # Create a minimal Citation object
            self.bibliography.append(Citation(
                citation_type='unknown',
                authors=[],
                title='',
                year=0,
                formatted=citation
            ))
        else:
            self.bibliography.append(citation)

    def generate_bibliography(
        self,
        sort_by: str = 'author',
        deduplicate: bool = True
    ) -> str:
        """
        Generate formatted bibliography from added citations.

        Args:
            sort_by: Sort method - 'author', 'year', or 'title'
            deduplicate: Remove duplicate entries

        Returns:
            Formatted bibliography string
        """
        citations = self.bibliography.copy()

        # Deduplicate
        if deduplicate:
            seen = set()
            unique = []
            for cit in citations:
                formatted = cit.formatted if cit.formatted else str(cit)
                if formatted not in seen:
                    seen.add(formatted)
                    unique.append(cit)
            citations = unique

        # Sort
        if sort_by == 'author':
            citations.sort(key=lambda c: c.authors[0] if c.authors else '')
        elif sort_by == 'year':
            citations.sort(key=lambda c: c.year)
        elif sort_by == 'title':
            citations.sort(key=lambda c: c.title)

        # Format with hanging indent
        lines = []
        for cit in citations:
            formatted = cit.formatted if cit.formatted else str(cit)
            lines.append(formatted)

        return '\n\n'.join(lines)

    def export_bibtex(self, output_path: str):
        """
        Export bibliography as BibTeX format.

        Args:
            output_path: Output file path
        """
        entries = []

        for i, cit in enumerate(self.bibliography, 1):
            # Generate cite key
            if cit.authors:
                first_author = cit.authors[0].split(',')[0].strip()
            else:
                first_author = 'unknown'

            cite_key = f"{first_author.lower()}{cit.year}"

            # Build BibTeX entry
            if cit.citation_type == 'book':
                entry = f"@book{{{cite_key},\n"
                if cit.authors:
                    entry += f"  author = {{{' and '.join(cit.authors)}}},\n"
                entry += f"  title = {{{cit.title}}},\n"
                entry += f"  year = {{{cit.year}}},\n"
                if cit.publisher:
                    entry += f"  publisher = {{{cit.publisher}}},\n"
                if cit.edition:
                    entry += f"  edition = {{{cit.edition}}},\n"
                if cit.isbn:
                    entry += f"  isbn = {{{cit.isbn}}},\n"
                entry += "}\n"

            elif cit.citation_type == 'article':
                entry = f"@article{{{cite_key},\n"
                if cit.authors:
                    entry += f"  author = {{{' and '.join(cit.authors)}}},\n"
                entry += f"  title = {{{cit.title}}},\n"
                if cit.journal:
                    entry += f"  journal = {{{cit.journal}}},\n"
                entry += f"  year = {{{cit.year}}},\n"
                if cit.volume:
                    entry += f"  volume = {{{cit.volume}}},\n"
                if cit.issue:
                    entry += f"  number = {{{cit.issue}}},\n"
                if cit.pages:
                    entry += f"  pages = {{{cit.pages}}},\n"
                if cit.doi:
                    entry += f"  doi = {{{cit.doi}}},\n"
                entry += "}\n"

            else:
                # Misc type for websites
                entry = f"@misc{{{cite_key},\n"
                if cit.authors:
                    entry += f"  author = {{{' and '.join(cit.authors)}}},\n"
                entry += f"  title = {{{cit.title}}},\n"
                entry += f"  year = {{{cit.year}}},\n"
                if cit.url:
                    entry += f"  url = {{{cit.url}}},\n"
                if cit.access_date:
                    entry += f"  note = {{Accessed: {cit.access_date}}},\n"
                entry += "}\n"

            entries.append(entry)

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(entries))

    def in_text_citation(
        self,
        authors: List[str],
        year: int,
        page: str = "",
        narrative: bool = False
    ) -> str:
        """
        Generate in-text citation.

        Args:
            authors: List of author names
            year: Publication year
            page: Page number (optional)
            narrative: True for narrative style, False for parenthetical

        Returns:
            Formatted in-text citation
        """
        # Determine how many authors to show
        if self.style in ['apa', 'harvard']:
            max_authors = 2 if len(authors) <= 2 else 1
        elif self.style == 'mla':
            max_authors = 2 if len(authors) <= 2 else 1
        elif self.style == 'chicago':
            max_authors = 3 if len(authors) <= 3 else 1
        else:
            max_authors = None

        author_str = self._format_authors(authors, max_display=max_authors, style=self.style)

        # Extract just last name(s) for in-text
        if ',' in author_str:
            # "Last, F." format - extract just last name
            last_names = []
            for author in authors[:max_authors]:
                last_name = author.split(',')[0].strip()
                last_names.append(last_name)

            if len(last_names) == 1:
                author_part = last_names[0]
            elif len(last_names) == 2:
                author_part = f"{last_names[0]} & {last_names[1]}" if self.style == 'apa' else f"{last_names[0]} and {last_names[1]}"
            else:
                author_part = last_names[0]

            if len(authors) > max_authors:
                author_part += " et al."
        else:
            author_part = author_str

        # Format citation
        if narrative:
            # Narrative: Smith (2020) or Smith (2020, p. 45)
            if page:
                return f"{author_part} ({year}, p. {page})"
            else:
                return f"{author_part} ({year})"
        else:
            # Parenthetical: (Smith, 2020) or (Smith, 2020, p. 45)
            if page:
                return f"({author_part}, {year}, p. {page})"
            else:
                return f"({author_part}, {year})"

    def import_from_csv(self, csv_path: str) -> List[str]:
        """
        Import citations from CSV file.

        Args:
            csv_path: Path to CSV file

        Returns:
            List of formatted citations
        """
        if not pd:
            raise ImportError(
                "pandas library is required for CSV import. "
                "Install with: pip install pandas"
            )

        df = pd.read_csv(csv_path)
        citations = []

        for _, row in df.iterrows():
            cite_type = row.get('type', '').lower()

            # Parse authors (pipe-separated)
            authors_str = row.get('authors', '')
            authors = [a.strip() for a in authors_str.split('|')] if authors_str else []

            try:
                if cite_type == 'book':
                    citation = self.cite_book(
                        authors=authors,
                        title=row.get('title', ''),
                        year=int(row.get('year', 0)),
                        publisher=row.get('publisher', ''),
                        city=row.get('city', ''),
                        edition=row.get('edition', ''),
                        isbn=row.get('isbn', '')
                    )

                elif cite_type == 'article':
                    citation = self.cite_article(
                        authors=authors,
                        title=row.get('title', ''),
                        journal=row.get('journal', ''),
                        year=int(row.get('year', 0)),
                        volume=int(row['volume']) if pd.notna(row.get('volume')) else None,
                        issue=int(row['issue']) if pd.notna(row.get('issue')) else None,
                        pages=row.get('pages', ''),
                        doi=row.get('doi', '')
                    )

                elif cite_type == 'website':
                    citation = self.cite_website(
                        authors=authors,
                        title=row.get('title', ''),
                        url=row.get('url', ''),
                        access_date=row.get('access_date', ''),
                        publish_date=row.get('publish_date', '')
                    )

                else:
                    continue

                citations.append(citation)
                self.add_to_bibliography(Citation(
                    citation_type=cite_type,
                    authors=authors,
                    title=row.get('title', ''),
                    year=int(row.get('year', 0)),
                    formatted=citation
                ))

            except Exception as e:
                print(f"Warning: Failed to process row: {e}", file=sys.stderr)
                continue

        return citations


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate formatted citations in multiple academic styles'
    )
    parser.add_argument(
        'command',
        choices=['book', 'article', 'website', 'doi', 'batch', 'intext'],
        help='Citation command'
    )
    parser.add_argument('--style', '-s', default='apa',
                        choices=CitationGenerator.SUPPORTED_STYLES,
                        help='Citation style')

    # Common arguments
    parser.add_argument('--authors', '-a', nargs='+', help='Author names (Last, First)')
    parser.add_argument('--title', '-t', help='Title')
    parser.add_argument('--year', '-y', type=int, help='Publication year')

    # Book arguments
    parser.add_argument('--publisher', help='Publisher name')
    parser.add_argument('--city', help='City of publication')
    parser.add_argument('--edition', help='Edition (e.g., "2nd ed.")')
    parser.add_argument('--isbn', help='ISBN number')

    # Article arguments
    parser.add_argument('--journal', '-j', help='Journal name')
    parser.add_argument('--volume', type=int, help='Volume number')
    parser.add_argument('--issue', type=int, help='Issue number')
    parser.add_argument('--pages', help='Page range (e.g., "45-67")')
    parser.add_argument('--doi', help='DOI identifier')

    # Website arguments
    parser.add_argument('--url', help='URL')
    parser.add_argument('--access-date', help='Access date (YYYY-MM-DD)')
    parser.add_argument('--publish-date', help='Publish date (YYYY-MM-DD)')

    # In-text citation arguments
    parser.add_argument('--page', help='Page number for in-text citation')
    parser.add_argument('--narrative', action='store_true',
                        help='Narrative style for in-text citation')

    # Batch arguments
    parser.add_argument('--input', '-i', help='Input CSV file')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--format', '-f', choices=['text', 'bibtex'], default='text',
                        help='Output format')
    parser.add_argument('--sort', choices=['author', 'year', 'title'], default='author',
                        help='Sort bibliography by')

    args = parser.parse_args()

    # Initialize generator
    gen = CitationGenerator(style=args.style)

    try:
        if args.command == 'book':
            if not all([args.authors, args.title, args.year, args.publisher]):
                parser.error('book requires --authors, --title, --year, --publisher')

            citation = gen.cite_book(
                authors=args.authors,
                title=args.title,
                year=args.year,
                publisher=args.publisher,
                city=args.city or '',
                edition=args.edition or '',
                isbn=args.isbn or ''
            )
            print(citation)

        elif args.command == 'article':
            if not all([args.authors, args.title, args.journal, args.year]):
                parser.error('article requires --authors, --title, --journal, --year')

            citation = gen.cite_article(
                authors=args.authors,
                title=args.title,
                journal=args.journal,
                year=args.year,
                volume=args.volume,
                issue=args.issue,
                pages=args.pages or '',
                doi=args.doi or ''
            )
            print(citation)

        elif args.command == 'website':
            if not all([args.title, args.url, args.access_date]):
                parser.error('website requires --title, --url, --access-date')

            citation = gen.cite_website(
                authors=args.authors or [],
                title=args.title,
                url=args.url,
                access_date=args.access_date,
                publish_date=args.publish_date or ''
            )
            print(citation)

        elif args.command == 'doi':
            if not args.doi:
                parser.error('doi command requires --doi')

            citation = gen.cite_from_doi(doi=args.doi)
            print(citation)

        elif args.command == 'batch':
            if not args.input:
                parser.error('batch command requires --input')

            citations = gen.import_from_csv(args.input)

            if args.format == 'bibtex':
                output = args.output or 'references.bib'
                gen.export_bibtex(output)
                print(f"Exported {len(citations)} citations to {output}")
            else:
                bibliography = gen.generate_bibliography(sort_by=args.sort)
                if args.output:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        f.write(bibliography)
                    print(f"Exported {len(citations)} citations to {args.output}")
                else:
                    print(bibliography)

        elif args.command == 'intext':
            if not all([args.authors, args.year]):
                parser.error('intext requires --authors, --year')

            citation = gen.in_text_citation(
                authors=args.authors,
                year=args.year,
                page=args.page or '',
                narrative=args.narrative
            )
            print(citation)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
