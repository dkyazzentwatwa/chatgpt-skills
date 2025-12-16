#!/usr/bin/env python3
"""
Certificate Generator - Create professional certificates

Features:
- Multiple templates (modern, classic, elegant, minimal, academic)
- Custom branding (logo, colors)
- Multiple signatures
- Batch generation from CSV
"""

import argparse
import csv
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape, portrait
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


@dataclass
class Signature:
    """Signature data."""
    name: str
    title: str
    image_path: Optional[str] = None


class CertificateGenerator:
    """
    Generate professional certificates.

    Example:
        cert = CertificateGenerator()
        cert.set_title("Certificate of Completion")
        cert.set_recipient("John Smith")
        cert.generate().save("certificate.pdf")
    """

    def __init__(self, template: str = "modern"):
        """Initialize certificate generator."""
        self._template = template
        self._orientation = "landscape"

        # Content
        self._title = "Certificate"
        self._recipient = ""
        self._achievement = ""
        self._description = ""
        self._date = ""
        self._certificate_id = ""
        self._organization = ""

        # Branding
        self._logo_path = ""
        self._logo_width = 150
        self._primary_color = "#1e3a5f"
        self._accent_color = "#c9a227"
        self._text_color = "#333333"

        # Signatures
        self._signatures: List[Signature] = []

        self._pdf_buffer: Optional[BytesIO] = None

    def set_template(self, template: str) -> 'CertificateGenerator':
        """Set certificate template."""
        valid = ['modern', 'classic', 'elegant', 'minimal', 'academic']
        if template not in valid:
            raise ValueError(f"Invalid template: {template}. Use {valid}")
        self._template = template
        return self

    def set_orientation(self, orientation: str) -> 'CertificateGenerator':
        """Set page orientation."""
        if orientation not in ['landscape', 'portrait']:
            raise ValueError("Orientation must be 'landscape' or 'portrait'")
        self._orientation = orientation
        return self

    def set_title(self, title: str) -> 'CertificateGenerator':
        """Set certificate title."""
        self._title = title
        return self

    def set_recipient(self, name: str) -> 'CertificateGenerator':
        """Set recipient name."""
        self._recipient = name
        return self

    def set_achievement(self, achievement: str) -> 'CertificateGenerator':
        """Set achievement text."""
        self._achievement = achievement
        return self

    def set_description(self, description: str) -> 'CertificateGenerator':
        """Set description text."""
        self._description = description
        return self

    def set_date(self, date: str) -> 'CertificateGenerator':
        """Set certificate date."""
        self._date = date
        return self

    def set_date_auto(self) -> 'CertificateGenerator':
        """Set date to today."""
        self._date = datetime.now().strftime("%B %d, %Y")
        return self

    def set_certificate_id(self, cert_id: str) -> 'CertificateGenerator':
        """Set certificate ID."""
        self._certificate_id = cert_id
        return self

    def set_organization(self, name: str) -> 'CertificateGenerator':
        """Set organization name."""
        self._organization = name
        return self

    def set_logo(self, path: str, width: int = 150) -> 'CertificateGenerator':
        """Set logo image."""
        self._logo_path = path
        self._logo_width = width
        return self

    def set_colors(
        self,
        primary: str = None,
        accent: str = None,
        text: str = None
    ) -> 'CertificateGenerator':
        """Set color scheme."""
        if primary:
            self._primary_color = primary
        if accent:
            self._accent_color = accent
        if text:
            self._text_color = text
        return self

    def add_signature(
        self,
        name: str,
        title: str,
        signature_image: str = None
    ) -> 'CertificateGenerator':
        """Add a signature."""
        if len(self._signatures) >= 3:
            raise ValueError("Maximum 3 signatures allowed")
        sig = Signature(name=name, title=title, image_path=signature_image)
        self._signatures.append(sig)
        return self

    def _hex_to_color(self, hex_color: str):
        """Convert hex to reportlab color."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        return colors.Color(r, g, b)

    def _get_page_size(self):
        """Get page size based on orientation."""
        if self._orientation == "landscape":
            return landscape(letter)
        return portrait(letter)

    def generate(self) -> 'CertificateGenerator':
        """Generate the certificate PDF."""
        self._pdf_buffer = BytesIO()
        page_size = self._get_page_size()

        doc = SimpleDocTemplate(
            self._pdf_buffer,
            pagesize=page_size,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch
        )

        elements = []
        styles = getSampleStyleSheet()

        primary = self._hex_to_color(self._primary_color)
        accent = self._hex_to_color(self._accent_color)
        text_color = self._hex_to_color(self._text_color)

        # Template-specific styles
        if self._template == "elegant":
            title_font = "Times-Bold"
            name_font = "Times-Italic"
            body_font = "Times-Roman"
        elif self._template == "classic":
            title_font = "Times-Bold"
            name_font = "Times-Bold"
            body_font = "Times-Roman"
        else:  # modern, minimal, academic
            title_font = "Helvetica-Bold"
            name_font = "Helvetica-Bold"
            body_font = "Helvetica"

        # Custom styles
        title_style = ParagraphStyle(
            'CertTitle',
            parent=styles['Heading1'],
            fontName=title_font,
            fontSize=36 if self._template != 'minimal' else 28,
            textColor=primary,
            alignment=TA_CENTER,
            spaceAfter=20
        )

        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontName=body_font,
            fontSize=14,
            textColor=text_color,
            alignment=TA_CENTER,
            spaceAfter=30
        )

        name_style = ParagraphStyle(
            'RecipientName',
            parent=styles['Heading1'],
            fontName=name_font,
            fontSize=32 if self._template == 'elegant' else 28,
            textColor=primary,
            alignment=TA_CENTER,
            spaceBefore=10,
            spaceAfter=10
        )

        achievement_style = ParagraphStyle(
            'Achievement',
            parent=styles['Normal'],
            fontName=body_font,
            fontSize=16,
            textColor=text_color,
            alignment=TA_CENTER,
            spaceAfter=20
        )

        description_style = ParagraphStyle(
            'Description',
            parent=styles['Normal'],
            fontName=body_font,
            fontSize=12,
            textColor=text_color,
            alignment=TA_CENTER,
            spaceAfter=30
        )

        # Add border for elegant/classic templates
        if self._template in ['elegant', 'classic']:
            # Top decorative line
            border_table = Table(
                [['']],
                colWidths=[page_size[0] - 1.5 * inch],
                rowHeights=[3]
            )
            border_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), accent),
            ]))
            elements.append(border_table)
            elements.append(Spacer(1, 0.3 * inch))

        # Logo
        if self._logo_path and Path(self._logo_path).exists():
            try:
                logo = Image(self._logo_path, width=self._logo_width)
                logo.hAlign = 'CENTER'
                elements.append(logo)
                elements.append(Spacer(1, 0.2 * inch))
            except Exception:
                pass

        # Organization
        if self._organization:
            org_style = ParagraphStyle(
                'Organization',
                parent=styles['Normal'],
                fontName=body_font,
                fontSize=14,
                textColor=text_color,
                alignment=TA_CENTER,
                spaceAfter=10
            )
            elements.append(Paragraph(self._organization, org_style))
            elements.append(Spacer(1, 0.1 * inch))

        # Title
        elements.append(Paragraph(self._title, title_style))

        # "This is to certify that" line
        elements.append(Paragraph("This is to certify that", subtitle_style))

        # Recipient name
        elements.append(Paragraph(self._recipient, name_style))

        # Decorative line under name
        if self._template in ['elegant', 'classic', 'academic']:
            line_table = Table(
                [['']],
                colWidths=[4 * inch],
                rowHeights=[1]
            )
            line_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), accent),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            line_wrapper = Table([[line_table]])
            line_wrapper.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            elements.append(line_wrapper)
            elements.append(Spacer(1, 0.2 * inch))

        # Achievement
        if self._achievement:
            achievement_text = f"has successfully completed {self._achievement}" \
                if not self._achievement.startswith("for") and not self._achievement.startswith("has") \
                else self._achievement
            elements.append(Paragraph(achievement_text, achievement_style))

        # Description
        if self._description:
            elements.append(Paragraph(self._description, description_style))

        elements.append(Spacer(1, 0.3 * inch))

        # Date
        if self._date:
            date_style = ParagraphStyle(
                'Date',
                parent=styles['Normal'],
                fontName=body_font,
                fontSize=12,
                textColor=text_color,
                alignment=TA_CENTER
            )
            elements.append(Paragraph(f"Date: {self._date}", date_style))
            elements.append(Spacer(1, 0.3 * inch))

        # Signatures
        if self._signatures:
            sig_data = []
            sig_row = []

            for sig in self._signatures:
                sig_cell = []

                # Signature image or line
                if sig.image_path and Path(sig.image_path).exists():
                    try:
                        sig_img = Image(sig.image_path, width=1.5 * inch, height=0.5 * inch)
                        sig_cell.append(sig_img)
                    except Exception:
                        sig_cell.append(Paragraph("_" * 25, styles['Normal']))
                else:
                    sig_cell.append(Paragraph("_" * 25, styles['Normal']))

                # Name and title
                name_p = Paragraph(
                    f"<b>{sig.name}</b>",
                    ParagraphStyle('SigName', fontSize=11, alignment=TA_CENTER)
                )
                title_p = Paragraph(
                    sig.title,
                    ParagraphStyle('SigTitle', fontSize=10, alignment=TA_CENTER, textColor=text_color)
                )
                sig_cell.extend([name_p, title_p])

                sig_row.append(sig_cell)

            sig_data.append(sig_row)

            # Calculate column widths based on number of signatures
            total_width = page_size[0] - 1.5 * inch
            col_width = total_width / len(self._signatures)

            sig_table = Table(sig_data, colWidths=[col_width] * len(self._signatures))
            sig_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            elements.append(sig_table)

        # Certificate ID
        if self._certificate_id or True:  # Always show ID
            cert_id = self._certificate_id or f"CERT-{uuid.uuid4().hex[:8].upper()}"
            id_style = ParagraphStyle(
                'CertID',
                parent=styles['Normal'],
                fontName=body_font,
                fontSize=8,
                textColor=colors.grey,
                alignment=TA_CENTER
            )
            elements.append(Spacer(1, 0.3 * inch))
            elements.append(Paragraph(f"Certificate ID: {cert_id}", id_style))

        # Bottom border for elegant/classic
        if self._template in ['elegant', 'classic']:
            elements.append(Spacer(1, 0.2 * inch))
            border_table = Table(
                [['']],
                colWidths=[page_size[0] - 1.5 * inch],
                rowHeights=[3]
            )
            border_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), accent),
            ]))
            elements.append(border_table)

        # Build PDF
        doc.build(elements)

        return self

    def save(self, path: str) -> str:
        """Save certificate to PDF file."""
        if self._pdf_buffer is None:
            self.generate()

        output_path = Path(path)
        with open(output_path, 'wb') as f:
            f.write(self._pdf_buffer.getvalue())

        return str(output_path)

    def to_bytes(self) -> bytes:
        """Get PDF as bytes."""
        if self._pdf_buffer is None:
            self.generate()
        return self._pdf_buffer.getvalue()

    @classmethod
    def batch_generate(
        cls,
        csv_file: str,
        template: str = "modern",
        title: str = "Certificate",
        achievement: str = "",
        organization: str = "",
        logo: str = "",
        output_dir: str = "./",
        filename_pattern: str = "{name}.pdf"
    ) -> List[str]:
        """
        Generate certificates from CSV file.

        Args:
            csv_file: Path to CSV file with recipient data
            template: Template style
            title: Certificate title
            achievement: Achievement text (can use {column} placeholders)
            organization: Organization name
            logo: Logo path
            output_dir: Output directory
            filename_pattern: Filename pattern with {column} placeholders

        Returns:
            List of generated file paths
        """
        path = Path(csv_file)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {csv_file}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated = []

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cert = cls(template=template)
                cert.set_title(title)

                # Set recipient (try different column names)
                name = row.get('name') or row.get('recipient') or row.get('Name') or ''
                cert.set_recipient(name)

                # Set achievement (with placeholder replacement)
                ach = achievement
                for key, value in row.items():
                    ach = ach.replace(f"{{{key}}}", str(value))
                if not ach:
                    ach = row.get('course') or row.get('achievement') or ''
                cert.set_achievement(ach)

                # Set date
                date = row.get('date') or row.get('Date') or ''
                if date:
                    cert.set_date(date)
                else:
                    cert.set_date_auto()

                # Set ID if provided
                cert_id = row.get('certificate_id') or row.get('id') or ''
                if cert_id:
                    cert.set_certificate_id(cert_id)

                # Set organization and logo
                if organization:
                    cert.set_organization(organization)
                if logo:
                    cert.set_logo(logo)

                # Generate
                cert.generate()

                # Save with pattern
                filename = filename_pattern
                for key, value in row.items():
                    safe_value = str(value).replace(' ', '_').replace('/', '-')
                    filename = filename.replace(f"{{{key}}}", safe_value)

                filepath = output_path / filename
                cert.save(str(filepath))
                generated.append(str(filepath))

        return generated


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate professional certificates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python certificate_gen.py --recipient "John Smith" --title "Certificate of Completion" --output cert.pdf
  python certificate_gen.py --batch students.csv --title "Certificate of Completion" --output-dir ./certs/
        """
    )

    parser.add_argument('--recipient', '-r', help='Recipient name')
    parser.add_argument('--title', '-t', default='Certificate', help='Certificate title')
    parser.add_argument('--achievement', '-a', help='Achievement text')
    parser.add_argument('--description', '-d', help='Description text')
    parser.add_argument('--organization', '-org', help='Organization name')
    parser.add_argument('--date', help='Certificate date (default: today)')
    parser.add_argument('--template', default='modern',
                        choices=['modern', 'classic', 'elegant', 'minimal', 'academic'],
                        help='Template style')
    parser.add_argument('--logo', help='Logo image path')
    parser.add_argument('--output', '-o', default='certificate.pdf', help='Output PDF path')

    parser.add_argument('--batch', '-b', help='CSV file for batch generation')
    parser.add_argument('--output-dir', default='./', help='Output directory (batch)')

    args = parser.parse_args()

    try:
        if args.batch:
            # Batch mode
            files = CertificateGenerator.batch_generate(
                csv_file=args.batch,
                template=args.template,
                title=args.title,
                achievement=args.achievement or '',
                organization=args.organization or '',
                logo=args.logo or '',
                output_dir=args.output_dir
            )
            print(f"Generated {len(files)} certificates")
            for f in files:
                print(f"  - {f}")

        else:
            # Single certificate
            if not args.recipient:
                parser.error("--recipient is required for single certificate")

            cert = CertificateGenerator(template=args.template)
            cert.set_title(args.title)
            cert.set_recipient(args.recipient)

            if args.achievement:
                cert.set_achievement(args.achievement)
            if args.description:
                cert.set_description(args.description)
            if args.organization:
                cert.set_organization(args.organization)
            if args.logo:
                cert.set_logo(args.logo)
            if args.date:
                cert.set_date(args.date)
            else:
                cert.set_date_auto()

            cert.generate().save(args.output)
            print(f"Certificate saved to: {args.output}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
