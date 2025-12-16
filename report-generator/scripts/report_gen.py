#!/usr/bin/env python3
"""
Report Generator - Create professional PDF/HTML reports

Features:
- Rich content: text, tables, charts, images
- Multiple chart types
- Templates and branding
- PDF and HTML output
"""

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, ListFlowable, ListItem,
    KeepTogether
)


class ReportGenerator:
    """
    Generate professional reports with charts, tables, and text.

    Example:
        report = ReportGenerator("Monthly Report")
        report.add_text("Summary of findings...")
        report.add_table(data)
        report.add_chart(data, "bar", x="Category", y="Value")
        report.generate().save("report.pdf")
    """

    def __init__(
        self,
        title: str = "Report",
        subtitle: str = "",
        config: Dict[str, Any] = None
    ):
        """Initialize report generator."""
        self._title = title
        self._subtitle = subtitle
        self._author = ""
        self._date = ""
        self._organization = ""
        self._logo_path = ""
        self._logo_width = 150

        # Configuration
        config = config or {}
        self._page_size = config.get('page_size', 'letter')
        self._orientation = config.get('orientation', 'portrait')
        self._margins = config.get('margins', {
            'top': 0.75, 'bottom': 0.75, 'left': 0.75, 'right': 0.75
        })

        # Styling
        self._primary_color = "#1e40af"
        self._secondary_color = "#6b7280"
        self._background_color = "#ffffff"
        self._heading_font = "Helvetica-Bold"
        self._body_font = "Helvetica"

        # Header/Footer
        self._header_text = ""
        self._footer_text = ""
        self._watermark = ""

        # Content elements
        self._elements: List[Any] = []
        self._toc_enabled = False
        self._in_appendix = False

        self._pdf_buffer: Optional[BytesIO] = None
        self._styles = None

    @classmethod
    def from_template(cls, template_name: str) -> 'ReportGenerator':
        """Create report from template."""
        templates = {
            'executive_summary': {
                'title': 'Executive Summary',
                'sections': ['overview', 'key_metrics', 'highlights', 'recommendations']
            },
            'quarterly_review': {
                'title': 'Quarterly Review',
                'sections': ['performance', 'financials', 'comparison', 'outlook']
            },
            'project_status': {
                'title': 'Project Status Report',
                'sections': ['overview', 'timeline', 'risks', 'next_steps']
            },
            'analytics_dashboard': {
                'title': 'Analytics Dashboard',
                'sections': ['kpis', 'charts', 'trends', 'insights']
            }
        }

        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}")

        template = templates[template_name]
        report = cls(template['title'])
        report._template = template_name
        return report

    def set_title(self, title: str) -> 'ReportGenerator':
        """Set report title."""
        self._title = title
        return self

    def set_subtitle(self, subtitle: str) -> 'ReportGenerator':
        """Set report subtitle."""
        self._subtitle = subtitle
        return self

    def set_author(self, author: str) -> 'ReportGenerator':
        """Set report author."""
        self._author = author
        return self

    def set_date(self, date: str) -> 'ReportGenerator':
        """Set report date."""
        self._date = date
        return self

    def set_date_auto(self) -> 'ReportGenerator':
        """Set date to today."""
        self._date = datetime.now().strftime("%B %d, %Y")
        return self

    def set_organization(self, name: str) -> 'ReportGenerator':
        """Set organization name."""
        self._organization = name
        return self

    def set_logo(self, path: str, width: int = 150) -> 'ReportGenerator':
        """Set logo image."""
        self._logo_path = path
        self._logo_width = width
        return self

    def set_colors(
        self,
        primary: str = None,
        secondary: str = None,
        background: str = None
    ) -> 'ReportGenerator':
        """Set color scheme."""
        if primary:
            self._primary_color = primary
        if secondary:
            self._secondary_color = secondary
        if background:
            self._background_color = background
        return self

    def set_fonts(
        self,
        heading: str = None,
        body: str = None
    ) -> 'ReportGenerator':
        """Set fonts."""
        if heading:
            self._heading_font = heading
        if body:
            self._body_font = body
        return self

    def set_header(self, text: str) -> 'ReportGenerator':
        """Set page header."""
        self._header_text = text
        return self

    def set_footer(self, text: str) -> 'ReportGenerator':
        """Set page footer."""
        self._footer_text = text
        return self

    def set_watermark(self, text: str) -> 'ReportGenerator':
        """Set watermark text."""
        self._watermark = text
        return self

    def enable_toc(self) -> 'ReportGenerator':
        """Enable table of contents."""
        self._toc_enabled = True
        return self

    def _hex_to_color(self, hex_color: str):
        """Convert hex to reportlab color."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        return colors.Color(r, g, b)

    def _init_styles(self):
        """Initialize paragraph styles."""
        self._styles = getSampleStyleSheet()
        primary = self._hex_to_color(self._primary_color)

        # Title style
        self._styles.add(ParagraphStyle(
            'ReportTitle',
            parent=self._styles['Heading1'],
            fontName=self._heading_font,
            fontSize=24,
            textColor=primary,
            alignment=TA_CENTER,
            spaceAfter=6
        ))

        # Subtitle style
        self._styles.add(ParagraphStyle(
            'ReportSubtitle',
            parent=self._styles['Normal'],
            fontName=self._body_font,
            fontSize=14,
            textColor=self._hex_to_color(self._secondary_color),
            alignment=TA_CENTER,
            spaceAfter=20
        ))

        # Section heading styles
        for i, size in enumerate([18, 14, 12], 1):
            self._styles.add(ParagraphStyle(
                f'Heading{i}Custom',
                parent=self._styles[f'Heading{i}'],
                fontName=self._heading_font,
                fontSize=size,
                textColor=primary,
                spaceBefore=12,
                spaceAfter=6
            ))

        # Body text
        self._styles.add(ParagraphStyle(
            'BodyText',
            parent=self._styles['Normal'],
            fontName=self._body_font,
            fontSize=10,
            textColor=colors.black,
            alignment=TA_JUSTIFY,
            spaceAfter=8
        ))

        # Callout styles
        for style_name, bg_color in [
            ('info', '#e0f2fe'),
            ('warning', '#fef3c7'),
            ('success', '#d1fae5'),
            ('error', '#fee2e2')
        ]:
            self._styles.add(ParagraphStyle(
                f'Callout_{style_name}',
                parent=self._styles['Normal'],
                fontName=self._body_font,
                fontSize=10,
                backColor=self._hex_to_color(bg_color),
                borderPadding=10,
                spaceAfter=12
            ))

    def add_text(self, text: str, style: str = "body") -> 'ReportGenerator':
        """Add text paragraph."""
        style_map = {
            'body': 'BodyText',
            'highlight': 'Callout_info',
            'metric': 'Heading2Custom'
        }
        style_name = style_map.get(style, 'BodyText')
        self._elements.append(('text', text, style_name))
        return self

    def add_heading(self, text: str, level: int = 1) -> 'ReportGenerator':
        """Add section heading."""
        self._elements.append(('heading', text, level))
        return self

    def add_bullets(self, items: List[str]) -> 'ReportGenerator':
        """Add bullet list."""
        self._elements.append(('bullets', items, None))
        return self

    def add_numbered_list(self, items: List[str]) -> 'ReportGenerator':
        """Add numbered list."""
        self._elements.append(('numbered', items, None))
        return self

    def add_table(
        self,
        data: Union[pd.DataFrame, List[Dict], List[List]],
        title: str = "",
        headers: List[str] = None,
        **kwargs
    ) -> 'ReportGenerator':
        """Add data table."""
        self._elements.append(('table', data, {'title': title, 'headers': headers, **kwargs}))
        return self

    def add_chart(
        self,
        data: Union[pd.DataFrame, Dict],
        chart_type: str = "bar",
        x: str = None,
        y: Union[str, List[str]] = None,
        values: str = None,
        labels: str = None,
        title: str = "",
        **kwargs
    ) -> 'ReportGenerator':
        """Add chart."""
        chart_config = {
            'type': chart_type,
            'x': x,
            'y': y,
            'values': values,
            'labels': labels,
            'title': title,
            **kwargs
        }
        self._elements.append(('chart', data, chart_config))
        return self

    def add_image(
        self,
        path: str,
        width: float = None,
        caption: str = ""
    ) -> 'ReportGenerator':
        """Add image."""
        self._elements.append(('image', path, {'width': width, 'caption': caption}))
        return self

    def add_page_break(self) -> 'ReportGenerator':
        """Add page break."""
        self._elements.append(('pagebreak', None, None))
        return self

    def add_divider(self) -> 'ReportGenerator':
        """Add horizontal line."""
        self._elements.append(('divider', None, None))
        return self

    def add_spacer(self, height: float = 0.25) -> 'ReportGenerator':
        """Add vertical space."""
        self._elements.append(('spacer', height, None))
        return self

    def add_callout(self, text: str, style: str = "info") -> 'ReportGenerator':
        """Add callout box."""
        self._elements.append(('callout', text, style))
        return self

    def add_quote(self, text: str, attribution: str = "") -> 'ReportGenerator':
        """Add quote."""
        self._elements.append(('quote', text, attribution))
        return self

    def start_section(self, title: str) -> 'ReportGenerator':
        """Start a new section."""
        self.add_heading(title, level=1)
        return self

    def end_section(self) -> 'ReportGenerator':
        """End current section."""
        self.add_spacer(0.3)
        return self

    def start_appendix(self) -> 'ReportGenerator':
        """Start appendix section."""
        self._in_appendix = True
        self.add_page_break()
        self.add_heading("Appendix", level=1)
        return self

    def _create_chart_image(self, data, config) -> BytesIO:
        """Create chart as image."""
        fig, ax = plt.subplots(figsize=(
            config.get('width', 6),
            config.get('height', 4)
        ))

        # Convert data to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        chart_type = config.get('type', 'bar')
        x = config.get('x')
        y = config.get('y')
        color = config.get('color', self._primary_color)

        if chart_type == 'bar':
            if isinstance(y, list):
                df.plot(kind='bar', x=x, y=y, ax=ax, color=[color, '#6b7280'])
            else:
                df.plot(kind='bar', x=x, y=y, ax=ax, color=color)
            if config.get('horizontal'):
                df.plot(kind='barh', x=x, y=y, ax=ax, color=color)

        elif chart_type == 'line':
            if isinstance(y, list):
                for col in y:
                    ax.plot(df[x], df[col], label=col, marker='o')
                ax.legend()
            else:
                ax.plot(df[x], df[y], color=color, marker='o')

        elif chart_type == 'pie':
            values_col = config.get('values')
            labels_col = config.get('labels')
            ax.pie(df[values_col], labels=df[labels_col], autopct='%1.1f%%')

        elif chart_type == 'scatter':
            ax.scatter(df[x], df[y], color=color, alpha=0.7)

        elif chart_type == 'area':
            ax.fill_between(df[x], df[y], alpha=0.5, color=color)
            ax.plot(df[x], df[y], color=color)

        # Title
        if config.get('title'):
            ax.set_title(config['title'], fontsize=12, fontweight='bold')

        # Labels
        if x:
            ax.set_xlabel(x)
        if y and not isinstance(y, list):
            ax.set_ylabel(y)

        plt.tight_layout()

        # Save to buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)

        return img_buffer

    def _build_table(self, data, config) -> Table:
        """Build table from data."""
        # Convert to list of lists
        if isinstance(data, pd.DataFrame):
            headers = list(data.columns)
            rows = data.values.tolist()
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                headers = config.get('headers') or list(data[0].keys())
                rows = [[row.get(h, '') for h in headers] for row in data]
            else:
                headers = config.get('headers', [])
                rows = data
        else:
            return None

        # Build table data
        table_data = []
        if headers:
            table_data.append(headers)
        table_data.extend(rows)

        # Create table
        table = Table(table_data)

        primary = self._hex_to_color(self._primary_color)

        # Style
        style = [
            ('FONTNAME', (0, 0), (-1, 0), self._heading_font),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTNAME', (0, 1), (-1, -1), self._body_font),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), primary),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]

        # Alternating row colors
        for i in range(2, len(table_data), 2):
            style.append(('BACKGROUND', (0, i), (-1, i), colors.Color(0.95, 0.95, 0.95)))

        table.setStyle(TableStyle(style))

        return table

    def generate(self) -> 'ReportGenerator':
        """Generate the report PDF."""
        self._init_styles()
        self._pdf_buffer = BytesIO()

        # Page size
        if self._page_size == 'a4':
            page = A4
        else:
            page = letter

        if self._orientation == 'landscape':
            page = landscape(page)

        doc = SimpleDocTemplate(
            self._pdf_buffer,
            pagesize=page,
            rightMargin=self._margins['right'] * inch,
            leftMargin=self._margins['left'] * inch,
            topMargin=self._margins['top'] * inch,
            bottomMargin=self._margins['bottom'] * inch
        )

        flowables = []

        # Title page elements
        if self._logo_path and Path(self._logo_path).exists():
            try:
                logo = Image(self._logo_path, width=self._logo_width)
                logo.hAlign = 'CENTER'
                flowables.append(logo)
                flowables.append(Spacer(1, 0.3 * inch))
            except Exception:
                pass

        if self._organization:
            flowables.append(Paragraph(self._organization, self._styles['ReportSubtitle']))

        flowables.append(Paragraph(self._title, self._styles['ReportTitle']))

        if self._subtitle:
            flowables.append(Paragraph(self._subtitle, self._styles['ReportSubtitle']))

        if self._author or self._date:
            meta_parts = []
            if self._author:
                meta_parts.append(f"Author: {self._author}")
            if self._date:
                meta_parts.append(f"Date: {self._date}")
            flowables.append(Paragraph(" | ".join(meta_parts), self._styles['ReportSubtitle']))

        flowables.append(Spacer(1, 0.5 * inch))

        # Process content elements
        for elem_type, content, config in self._elements:
            if elem_type == 'text':
                style = self._styles.get(config, self._styles['BodyText'])
                flowables.append(Paragraph(content, style))

            elif elem_type == 'heading':
                level = min(config, 3)
                style = self._styles[f'Heading{level}Custom']
                flowables.append(Paragraph(content, style))

            elif elem_type == 'bullets':
                bullet_items = [
                    ListItem(Paragraph(item, self._styles['BodyText']))
                    for item in content
                ]
                flowables.append(ListFlowable(bullet_items, bulletType='bullet'))

            elif elem_type == 'numbered':
                num_items = [
                    ListItem(Paragraph(item, self._styles['BodyText']))
                    for item in content
                ]
                flowables.append(ListFlowable(num_items, bulletType='1'))

            elif elem_type == 'table':
                if config.get('title'):
                    flowables.append(Paragraph(
                        f"<b>{config['title']}</b>",
                        self._styles['BodyText']
                    ))
                table = self._build_table(content, config)
                if table:
                    flowables.append(table)
                    flowables.append(Spacer(1, 0.2 * inch))

            elif elem_type == 'chart':
                img_buffer = self._create_chart_image(content, config)
                img = Image(img_buffer, width=config.get('width', 6) * inch)
                img.hAlign = 'CENTER'
                flowables.append(img)
                flowables.append(Spacer(1, 0.2 * inch))

            elif elem_type == 'image':
                if Path(content).exists():
                    width = config.get('width', 4) * inch if config.get('width') else None
                    img = Image(content, width=width)
                    img.hAlign = 'CENTER'
                    flowables.append(img)
                    if config.get('caption'):
                        flowables.append(Paragraph(
                            f"<i>{config['caption']}</i>",
                            self._styles['BodyText']
                        ))

            elif elem_type == 'pagebreak':
                flowables.append(PageBreak())

            elif elem_type == 'divider':
                flowables.append(HRFlowable(
                    width="100%",
                    thickness=1,
                    color=colors.grey,
                    spaceBefore=6,
                    spaceAfter=6
                ))

            elif elem_type == 'spacer':
                flowables.append(Spacer(1, content * inch))

            elif elem_type == 'callout':
                style = self._styles.get(f'Callout_{config}', self._styles['Callout_info'])
                # Create a table for the callout box
                callout_table = Table(
                    [[Paragraph(content, style)]],
                    colWidths=[6 * inch]
                )
                bg_colors = {
                    'info': '#e0f2fe',
                    'warning': '#fef3c7',
                    'success': '#d1fae5',
                    'error': '#fee2e2'
                }
                callout_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), self._hex_to_color(bg_colors.get(config, '#e0f2fe'))),
                    ('BOX', (0, 0), (-1, -1), 1, colors.grey),
                    ('PADDING', (0, 0), (-1, -1), 12),
                ]))
                flowables.append(callout_table)
                flowables.append(Spacer(1, 0.1 * inch))

            elif elem_type == 'quote':
                quote_style = ParagraphStyle(
                    'Quote',
                    parent=self._styles['BodyText'],
                    fontSize=11,
                    leftIndent=30,
                    rightIndent=30,
                    fontName='Times-Italic'
                )
                flowables.append(Paragraph(f'"{content}"', quote_style))
                if config:  # attribution
                    flowables.append(Paragraph(f"— {config}", self._styles['BodyText']))

        # Build PDF
        doc.build(flowables)

        return self

    def save(self, path: str) -> str:
        """Save report to file."""
        if self._pdf_buffer is None:
            self.generate()

        output_path = Path(path)

        if output_path.suffix.lower() == '.html':
            # Basic HTML export
            html = self._to_html()
            with open(output_path, 'w') as f:
                f.write(html)
        else:
            with open(output_path, 'wb') as f:
                f.write(self._pdf_buffer.getvalue())

        return str(output_path)

    def to_bytes(self) -> bytes:
        """Get PDF as bytes."""
        if self._pdf_buffer is None:
            self.generate()
        return self._pdf_buffer.getvalue()

    def _to_html(self) -> str:
        """Generate basic HTML version."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self._title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: {self._primary_color}; }}
        h2, h3 {{ color: {self._primary_color}; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: {self._primary_color}; color: white; }}
        .callout {{ padding: 15px; margin: 15px 0; border-radius: 5px; }}
        .callout-info {{ background: #e0f2fe; }}
        .callout-warning {{ background: #fef3c7; }}
        .callout-success {{ background: #d1fae5; }}
    </style>
</head>
<body>
    <h1>{self._title}</h1>
    {f'<p><em>{self._subtitle}</em></p>' if self._subtitle else ''}
    <hr>
"""
        # Add basic content (simplified)
        html += "<p>Report content would be rendered here in full HTML version.</p>"
        html += "</body></html>"

        return html


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate professional PDF/HTML reports',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--config', '-c', help='Report configuration JSON')
    parser.add_argument('--template', '-t', help='Template name')
    parser.add_argument('--data', '-d', help='Data JSON file')
    parser.add_argument('--csv', help='CSV data file')
    parser.add_argument('--title', default='Report', help='Report title')
    parser.add_argument('--output', '-o', default='report.pdf', help='Output file')
    parser.add_argument('--format', '-f', choices=['pdf', 'html'], default='pdf')

    args = parser.parse_args()

    try:
        if args.template:
            report = ReportGenerator.from_template(args.template)
        else:
            report = ReportGenerator(args.title)

        if args.csv:
            df = pd.read_csv(args.csv)
            report.add_heading("Data Summary", level=1)
            report.add_table(df)

        report.generate().save(args.output)
        print(f"Report saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
