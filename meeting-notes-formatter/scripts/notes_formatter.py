#!/usr/bin/env python3
"""
Meeting Notes Formatter - Convert raw notes to structured documents

Features:
- Auto-detection of title, attendees, sections, action items
- Markdown and PDF output
- Clean, professional formatting
"""

import argparse
import re
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dateutil import parser as date_parser

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)


@dataclass
class ActionItem:
    """Represents an action item."""
    task: str
    owner: str = ""
    due: str = ""


@dataclass
class MeetingData:
    """Structured meeting data."""
    title: str = ""
    date: str = ""
    time: str = ""
    location: str = ""
    attendees: List[str] = field(default_factory=list)
    sections: Dict[str, List[str]] = field(default_factory=dict)
    action_items: List[ActionItem] = field(default_factory=list)
    next_meeting: str = ""
    raw_content: str = ""


class MeetingNotesFormatter:
    """
    Format raw meeting notes into structured documents.

    Example:
        formatter = MeetingNotesFormatter(raw_notes)
        formatter.format()
        formatter.save("notes.md")
    """

    def __init__(self, content: str = ""):
        """Initialize with raw notes content."""
        self._raw_content = content
        self._data = MeetingData(raw_content=content)
        self._formatted = False

    @classmethod
    def from_file(cls, filepath: str) -> 'MeetingNotesFormatter':
        """Load notes from file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        content = path.read_text(encoding='utf-8')
        return cls(content)

    def set_title(self, title: str) -> 'MeetingNotesFormatter':
        """Set meeting title."""
        self._data.title = title
        return self

    def set_date(self, date: str) -> 'MeetingNotesFormatter':
        """Set meeting date."""
        self._data.date = date
        return self

    def set_time(self, time: str) -> 'MeetingNotesFormatter':
        """Set meeting time."""
        self._data.time = time
        return self

    def set_location(self, location: str) -> 'MeetingNotesFormatter':
        """Set meeting location."""
        self._data.location = location
        return self

    def set_attendees(self, attendees: List[str]) -> 'MeetingNotesFormatter':
        """Set attendee list."""
        self._data.attendees = attendees
        return self

    def add_section(self, title: str, items: List[str]) -> 'MeetingNotesFormatter':
        """Add a section."""
        self._data.sections[title] = items
        return self

    def add_action_item(
        self,
        task: str,
        owner: str = "",
        due: str = ""
    ) -> 'MeetingNotesFormatter':
        """Add an action item."""
        item = ActionItem(task=task, owner=owner, due=due)
        self._data.action_items.append(item)
        return self

    def format(self) -> 'MeetingNotesFormatter':
        """Auto-format the raw notes."""
        if not self._raw_content:
            return self

        lines = self._raw_content.strip().split('\n')

        # Detect title
        if not self._data.title:
            self._data.title = self._detect_title(lines)

        # Detect date
        if not self._data.date:
            self._data.date = self._detect_date(lines)

        # Detect attendees
        if not self._data.attendees:
            self._data.attendees = self._detect_attendees(lines)

        # Detect sections and content
        if not self._data.sections:
            self._data.sections = self._detect_sections(lines)

        # Detect action items
        if not self._data.action_items:
            self._data.action_items = self._detect_action_items(lines)

        # Detect next meeting
        if not self._data.next_meeting:
            self._data.next_meeting = self._detect_next_meeting(lines)

        self._formatted = True
        return self

    def _detect_title(self, lines: List[str]) -> str:
        """Detect meeting title from lines."""
        # Check first non-empty line
        for line in lines[:5]:
            line = line.strip()
            if not line:
                continue

            # Skip if it looks like attendees or date only
            if any(kw in line.lower() for kw in ['attendees:', 'present:', 'participants:']):
                continue

            # If short and doesn't start with bullet, likely title
            if len(line) < 80 and not line.startswith(('-', '*', '•')):
                # Clean up date from title
                title = re.sub(r'\d{1,2}/\d{1,2}(/\d{2,4})?', '', line)
                title = re.sub(r'\d{4}-\d{2}-\d{2}', '', title)
                return title.strip(' -–')

        return "Meeting Notes"

    def _detect_date(self, lines: List[str]) -> str:
        """Detect meeting date from lines."""
        date_patterns = [
            r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b',
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2})\b',
        ]

        text = '\n'.join(lines[:10])

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    parsed = date_parser.parse(match.group(1))
                    return parsed.strftime("%B %d, %Y")
                except Exception:
                    return match.group(1)

        return datetime.now().strftime("%B %d, %Y")

    def _detect_attendees(self, lines: List[str]) -> List[str]:
        """Detect attendees from lines."""
        attendees = []
        attendee_keywords = ['attendees', 'present', 'participants', 'attending']

        for line in lines:
            line_lower = line.lower().strip()

            # Check for attendee line
            for keyword in attendee_keywords:
                if keyword in line_lower:
                    # Extract names after the keyword
                    parts = re.split(r'[:\-]', line, 1)
                    if len(parts) > 1:
                        names = parts[1]
                    else:
                        continue

                    # Split by common delimiters
                    names = re.split(r'[,;&]|\band\b', names)
                    attendees.extend([
                        n.strip().strip('@')
                        for n in names
                        if n.strip() and len(n.strip()) > 1
                    ])

        return attendees

    def _detect_sections(self, lines: List[str]) -> Dict[str, List[str]]:
        """Detect sections from lines."""
        sections = {}
        current_section = "Notes"
        current_items = []

        section_keywords = [
            'discussion', 'updates', 'agenda', 'topics', 'notes',
            'decisions', 'blockers', 'issues', 'planning', 'review',
            'summary', 'highlights', 'challenges', 'next steps'
        ]

        skip_keywords = [
            'attendees', 'present', 'participants', 'action',
            'todo', 'next meeting'
        ]

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            lower = stripped.lower()

            # Skip certain lines
            if any(kw in lower for kw in skip_keywords):
                continue

            # Check if this is a section header
            is_section = False
            for keyword in section_keywords:
                if keyword in lower and len(stripped) < 50:
                    # Save previous section
                    if current_items:
                        sections[current_section] = current_items

                    # Start new section
                    current_section = stripped.rstrip(':').title()
                    current_items = []
                    is_section = True
                    break

            if not is_section:
                # Check if it's a bullet point or content
                if stripped.startswith(('-', '*', '•', '–')):
                    item = stripped.lstrip('-*•– ').strip()
                    if item:
                        current_items.append(item)
                elif current_items or sections:
                    # Add as content if we're in a section
                    if stripped and not any(kw in lower for kw in ['action', 'todo']):
                        current_items.append(stripped)

        # Save last section
        if current_items:
            sections[current_section] = current_items

        return sections

    def _detect_action_items(self, lines: List[str]) -> List[ActionItem]:
        """Detect action items from lines."""
        items = []

        action_patterns = [
            # "ACTION: task"
            r'(?:action|todo|task)[:\s]+(.+)',
            # "[Owner] to do something"
            r'\[([^\]]+)\]\s*(?:to|will|should)\s+(.+)',
            # "Owner to do something"
            r'(\w+)\s+(?:to|will|should)\s+(.+?)(?:\s+by\s+(.+))?$',
            # "- Owner: task"
            r'^[-*•]\s*(\w+)[:\s]+(.+)',
        ]

        in_action_section = False

        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()

            # Check for action section header
            if any(kw in lower for kw in ['action items', 'actions:', 'todos:', 'tasks:']):
                in_action_section = True
                continue

            # Check for section change
            if in_action_section and stripped and not stripped.startswith(('-', '*', '•')):
                if ':' in stripped and len(stripped) < 30:
                    in_action_section = False

            # Try to extract action items
            for pattern in action_patterns:
                match = re.search(pattern, stripped, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2:
                        owner = groups[0].strip() if groups[0] else ""
                        task = groups[1].strip() if len(groups) > 1 else groups[0].strip()
                        due = groups[2].strip() if len(groups) > 2 and groups[2] else ""

                        # Clean up
                        task = task.rstrip('.')

                        if task and len(task) > 3:
                            items.append(ActionItem(task=task, owner=owner, due=due))
                    break

            # Check for simple action format in action section
            if in_action_section and stripped.startswith(('-', '*', '•')):
                item_text = stripped.lstrip('-*•– ').strip()
                if item_text:
                    # Try to extract owner
                    owner_match = re.match(r'^(\w+)\s*[-:]\s*(.+)', item_text)
                    if owner_match:
                        items.append(ActionItem(
                            task=owner_match.group(2).strip(),
                            owner=owner_match.group(1).strip()
                        ))
                    else:
                        items.append(ActionItem(task=item_text))

        return items

    def _detect_next_meeting(self, lines: List[str]) -> str:
        """Detect next meeting info."""
        for line in lines:
            lower = line.lower()
            if 'next meeting' in lower or 'next sync' in lower or 'next standup' in lower:
                # Extract date/day
                parts = re.split(r'[:\-]', line, 1)
                if len(parts) > 1:
                    return parts[1].strip()
                return line.strip()
        return ""

    def to_markdown(self) -> str:
        """Generate markdown output."""
        if not self._formatted:
            self.format()

        lines = []

        # Title
        lines.append(f"# {self._data.title}\n")

        # Metadata
        if self._data.date:
            lines.append(f"**Date:** {self._data.date}")
        if self._data.time:
            lines.append(f"**Time:** {self._data.time}")
        if self._data.location:
            lines.append(f"**Location:** {self._data.location}")
        if self._data.attendees:
            lines.append(f"**Attendees:** {', '.join(self._data.attendees)}")

        lines.append("\n---\n")

        # Sections
        for section_title, items in self._data.sections.items():
            lines.append(f"## {section_title}\n")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")

        # Action Items
        if self._data.action_items:
            lines.append("## Action Items\n")
            lines.append("| Task | Owner | Due |")
            lines.append("|------|-------|-----|")
            for item in self._data.action_items:
                lines.append(f"| {item.task} | {item.owner} | {item.due} |")
            lines.append("")

        # Next meeting
        if self._data.next_meeting:
            lines.append("---\n")
            lines.append(f"**Next Meeting:** {self._data.next_meeting}")

        return '\n'.join(lines)

    def to_dict(self) -> Dict:
        """Get structured data as dictionary."""
        if not self._formatted:
            self.format()

        return {
            'title': self._data.title,
            'date': self._data.date,
            'time': self._data.time,
            'location': self._data.location,
            'attendees': self._data.attendees,
            'sections': self._data.sections,
            'action_items': [
                {'task': a.task, 'owner': a.owner, 'due': a.due}
                for a in self._data.action_items
            ],
            'next_meeting': self._data.next_meeting
        }

    def to_pdf_bytes(self) -> bytes:
        """Generate PDF as bytes."""
        if not self._formatted:
            self.format()

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch
        )

        styles = getSampleStyleSheet()
        elements = []

        # Title
        title_style = ParagraphStyle(
            'MeetingTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12
        )
        elements.append(Paragraph(self._data.title, title_style))

        # Metadata
        meta_style = ParagraphStyle(
            'Meta',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey
        )

        meta_parts = []
        if self._data.date:
            meta_parts.append(f"<b>Date:</b> {self._data.date}")
        if self._data.attendees:
            meta_parts.append(f"<b>Attendees:</b> {', '.join(self._data.attendees)}")

        for part in meta_parts:
            elements.append(Paragraph(part, meta_style))

        elements.append(Spacer(1, 0.2 * inch))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        elements.append(Spacer(1, 0.2 * inch))

        # Sections
        section_style = ParagraphStyle(
            'Section',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#1e40af'),
            spaceBefore=12,
            spaceAfter=6
        )

        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=10,
            leftIndent=15
        )

        for section_title, items in self._data.sections.items():
            elements.append(Paragraph(section_title, section_style))
            for item in items:
                elements.append(Paragraph(f"• {item}", body_style))

        # Action Items
        if self._data.action_items:
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("Action Items", section_style))

            table_data = [['Task', 'Owner', 'Due']]
            for item in self._data.action_items:
                table_data.append([item.task, item.owner, item.due])

            table = Table(table_data, colWidths=[4 * inch, 1.5 * inch, 1 * inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(table)

        # Next meeting
        if self._data.next_meeting:
            elements.append(Spacer(1, 0.3 * inch))
            elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph(
                f"<b>Next Meeting:</b> {self._data.next_meeting}",
                meta_style
            ))

        doc.build(elements)
        return buffer.getvalue()

    def save(self, path: str) -> str:
        """Save formatted notes to file."""
        output_path = Path(path)

        if output_path.suffix.lower() == '.pdf':
            pdf_bytes = self.to_pdf_bytes()
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)
        else:
            markdown = self.to_markdown()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)

        return str(output_path)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Format raw meeting notes into structured documents'
    )

    parser.add_argument('--input', '-i', required=True, help='Input notes file')
    parser.add_argument('--output', '-o', default='notes.md', help='Output file')
    parser.add_argument('--title', '-t', help='Meeting title override')
    parser.add_argument('--date', '-d', help='Meeting date override')

    args = parser.parse_args()

    try:
        formatter = MeetingNotesFormatter.from_file(args.input)

        if args.title:
            formatter.set_title(args.title)
        if args.date:
            formatter.set_date(args.date)

        formatter.format()
        output_path = formatter.save(args.output)
        print(f"Formatted notes saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
