from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _escape(s: str) -> str:
    """Minimal HTML escaping for ReportLab Paragraph."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def write_pdf_from_sections(
    output_path: Path,
    title: Optional[str],
    sections: List[dict],
) -> None:
    """Write a simple PDF from text and optional tables.

    Each section dict can have:
      - heading: str
      - paragraphs: list[str]
      - tables: list[list[list[str]]]
    """
    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    story = []
    if title:
        story.append(Paragraph(_escape(title), styles["Title"]))
        story.append(Spacer(1, 0.2 * inch))

    for s_i, sec in enumerate(sections):
        heading = (sec.get("heading") or "").strip()
        paras = sec.get("paragraphs") or []
        tables = sec.get("tables") or []

        if heading:
            story.append(Paragraph(_escape(heading), styles["Heading2" if title else "Heading1"]))
            story.append(Spacer(1, 0.12 * inch))

        for p in paras:
            txt = str(p).strip()
            if not txt:
                continue
            story.append(Paragraph(_escape(txt), styles["BodyText"]))
            story.append(Spacer(1, 0.08 * inch))

        for t in tables:
            if not t:
                continue
            data = [[_escape(str(c or "")) for c in row] for row in t]
            if not data:
                continue
            tbl = Table(data, hAlign="LEFT")
            tbl.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(tbl)
            story.append(Spacer(1, 0.15 * inch))

        if s_i < len(sections) - 1:
            story.append(PageBreak())

    doc.build(story)
