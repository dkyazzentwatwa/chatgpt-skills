#!/usr/bin/env python3
"""
Invoice Generator - Create professional PDF invoices

Features:
- Professional templates
- Custom branding (logo, colors)
- Tax calculations
- Discounts
- Batch generation from CSV
"""

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# Currency symbols
CURRENCY_SYMBOLS = {
    'USD': '$',
    'EUR': '€',
    'GBP': '£',
    'JPY': '¥',
    'CAD': 'C$',
    'AUD': 'A$',
    'CHF': 'CHF',
    'INR': '₹',
}


@dataclass
class LineItem:
    """Invoice line item."""
    description: str
    quantity: float
    rate: float
    unit: str = ''
    discount: float = 0  # Percentage

    @property
    def subtotal(self) -> float:
        total = self.quantity * self.rate
        if self.discount > 0:
            total *= (1 - self.discount / 100)
        return total


@dataclass
class Tax:
    """Tax definition."""
    name: str
    rate: float  # Percentage
    compound: bool = False


@dataclass
class CompanyInfo:
    """Company information."""
    name: str
    address: str
    email: str = ''
    phone: str = ''
    website: str = ''
    tax_id: str = ''
    logo_path: str = ''


@dataclass
class ClientInfo:
    """Client information."""
    name: str
    address: str
    company: str = ''
    email: str = ''


class InvoiceGenerator:
    """
    Generate professional PDF invoices.

    Example:
        invoice = InvoiceGenerator()
        invoice.set_company("Acme Corp", "123 Main St")
        invoice.set_client("John Doe", "456 Oak Ave")
        invoice.add_item("Service", 1, 500)
        invoice.generate().save("invoice.pdf")
    """

    def __init__(self):
        """Initialize invoice generator."""
        self._company: Optional[CompanyInfo] = None
        self._client: Optional[ClientInfo] = None
        self._items: List[LineItem] = []
        self._taxes: List[Tax] = []

        self._invoice_number: str = ''
        self._date: str = datetime.now().strftime('%Y-%m-%d')
        self._due_date: str = ''
        self._currency: str = 'USD'
        self._currency_symbol: str = '$'

        self._discount: float = 0
        self._discount_is_percentage: bool = True

        self._payment_terms: str = ''
        self._payment_instructions: str = ''
        self._bank_details: Dict[str, str] = {}
        self._notes: str = ''
        self._terms: str = ''

        # Styling
        self._template: str = 'modern'
        self._primary_color: str = '#2563eb'
        self._secondary_color: str = '#64748b'
        self._background_color: str = '#f8fafc'
        self._font: str = 'Helvetica'

        self._pdf_buffer: Optional[BytesIO] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InvoiceGenerator':
        """Create invoice from dictionary."""
        invoice = cls()

        # Invoice details
        if 'invoice_number' in data:
            invoice.set_invoice_number(data['invoice_number'])
        if 'date' in data:
            invoice.set_date(data['date'])
        if 'due_date' in data:
            invoice.set_due_date(data['due_date'])
        if 'currency' in data:
            invoice.set_currency(data['currency'])

        # Company
        if 'company' in data:
            c = data['company']
            invoice.set_company(
                name=c.get('name', ''),
                address=c.get('address', ''),
                email=c.get('email', ''),
                phone=c.get('phone', ''),
                website=c.get('website', ''),
                tax_id=c.get('tax_id', '')
            )
            if 'logo' in c:
                invoice.set_logo(c['logo'])

        # Client
        if 'client' in data:
            c = data['client']
            invoice.set_client(
                name=c.get('name', ''),
                address=c.get('address', ''),
                company=c.get('company', ''),
                email=c.get('email', '')
            )

        # Items
        for item in data.get('items', []):
            invoice.add_item(
                description=item.get('description', ''),
                quantity=item.get('quantity', 1),
                rate=item.get('rate', 0),
                unit=item.get('unit', ''),
                discount=item.get('discount', 0)
            )

        # Taxes
        for tax in data.get('taxes', []):
            invoice.add_tax(
                name=tax.get('name', 'Tax'),
                rate=tax.get('rate', 0),
                compound=tax.get('compound', False)
            )

        # Other
        if 'discount' in data:
            invoice.set_discount(data['discount'])
        if 'notes' in data:
            invoice.set_notes(data['notes'])
        if 'terms' in data:
            invoice.set_terms(data['terms'])
        if 'payment_terms' in data:
            invoice.set_payment_terms(data['payment_terms'])

        return invoice

    @classmethod
    def from_csv(cls, filepath: str) -> List['InvoiceGenerator']:
        """Create invoices from CSV file (grouped by invoice number)."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        invoices_data: Dict[str, Dict] = {}

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                inv_num = row.get('invoice_number', '')
                if inv_num not in invoices_data:
                    invoices_data[inv_num] = {
                        'invoice_number': inv_num,
                        'date': row.get('date', ''),
                        'due_date': row.get('due_date', ''),
                        'client': {
                            'name': row.get('client_name', ''),
                            'address': row.get('client_address', '')
                        },
                        'items': [],
                        'taxes': []
                    }

                # Add item
                invoices_data[inv_num]['items'].append({
                    'description': row.get('item_description', ''),
                    'quantity': float(row.get('quantity', 1)),
                    'rate': float(row.get('rate', 0))
                })

                # Add tax if present
                if row.get('tax_rate'):
                    tax_rate = float(row['tax_rate'])
                    tax_exists = any(
                        t['rate'] == tax_rate
                        for t in invoices_data[inv_num]['taxes']
                    )
                    if not tax_exists:
                        invoices_data[inv_num]['taxes'].append({
                            'name': row.get('tax_name', 'Tax'),
                            'rate': tax_rate
                        })

        return [cls.from_dict(data) for data in invoices_data.values()]

    def set_company(
        self,
        name: str,
        address: str,
        email: str = '',
        phone: str = '',
        website: str = '',
        tax_id: str = ''
    ) -> 'InvoiceGenerator':
        """Set company information."""
        self._company = CompanyInfo(
            name=name,
            address=address,
            email=email,
            phone=phone,
            website=website,
            tax_id=tax_id
        )
        return self

    def set_logo(self, path: str, width: int = 150) -> 'InvoiceGenerator':
        """Set company logo."""
        if self._company:
            self._company.logo_path = path
        self._logo_width = width
        return self

    def set_client(
        self,
        name: str,
        address: str,
        company: str = '',
        email: str = ''
    ) -> 'InvoiceGenerator':
        """Set client information."""
        self._client = ClientInfo(
            name=name,
            address=address,
            company=company,
            email=email
        )
        return self

    def set_invoice_number(self, number: str) -> 'InvoiceGenerator':
        """Set invoice number."""
        self._invoice_number = number
        return self

    @property
    def invoice_number(self) -> str:
        """Get invoice number."""
        return self._invoice_number

    def set_date(self, date: str) -> 'InvoiceGenerator':
        """Set invoice date."""
        self._date = date
        return self

    def set_due_date(self, date: str) -> 'InvoiceGenerator':
        """Set due date."""
        self._due_date = date
        return self

    def set_due_days(self, days: int) -> 'InvoiceGenerator':
        """Set due date as days from invoice date."""
        try:
            inv_date = datetime.strptime(self._date, '%Y-%m-%d')
            due = inv_date + timedelta(days=days)
            self._due_date = due.strftime('%Y-%m-%d')
        except ValueError:
            pass
        return self

    def set_currency(self, currency: str, symbol_only: bool = False) -> 'InvoiceGenerator':
        """Set currency."""
        if symbol_only:
            self._currency_symbol = currency
        else:
            self._currency = currency
            self._currency_symbol = CURRENCY_SYMBOLS.get(currency, currency)
        return self

    def add_item(
        self,
        description: str,
        quantity: float,
        rate: float,
        unit: str = '',
        discount: float = 0
    ) -> 'InvoiceGenerator':
        """Add line item."""
        item = LineItem(
            description=description,
            quantity=quantity,
            rate=rate,
            unit=unit,
            discount=discount
        )
        self._items.append(item)
        return self

    def add_items(self, items: List[Dict]) -> 'InvoiceGenerator':
        """Add multiple line items."""
        for item in items:
            self.add_item(
                description=item.get('description', ''),
                quantity=item.get('quantity', 1),
                rate=item.get('rate', 0),
                unit=item.get('unit', ''),
                discount=item.get('discount', 0)
            )
        return self

    def add_tax(
        self,
        name: str,
        rate: float,
        compound: bool = False
    ) -> 'InvoiceGenerator':
        """Add tax."""
        tax = Tax(name=name, rate=rate, compound=compound)
        self._taxes.append(tax)
        return self

    def set_discount(
        self,
        amount: float,
        is_percentage: bool = True
    ) -> 'InvoiceGenerator':
        """Set discount on subtotal."""
        self._discount = amount
        self._discount_is_percentage = is_percentage
        return self

    def set_payment_terms(self, terms: str) -> 'InvoiceGenerator':
        """Set payment terms."""
        self._payment_terms = terms
        return self

    def set_payment_instructions(self, instructions: str) -> 'InvoiceGenerator':
        """Set payment instructions."""
        self._payment_instructions = instructions
        return self

    def set_bank_details(
        self,
        bank_name: str = '',
        account_name: str = '',
        account_number: str = '',
        routing_number: str = '',
        swift_code: str = ''
    ) -> 'InvoiceGenerator':
        """Set bank details."""
        self._bank_details = {
            'bank_name': bank_name,
            'account_name': account_name,
            'account_number': account_number,
            'routing_number': routing_number,
            'swift_code': swift_code
        }
        return self

    def set_notes(self, notes: str) -> 'InvoiceGenerator':
        """Set invoice notes."""
        self._notes = notes
        return self

    def set_terms(self, terms: str) -> 'InvoiceGenerator':
        """Set terms and conditions."""
        self._terms = terms
        return self

    def set_colors(
        self,
        primary: str = None,
        secondary: str = None,
        background: str = None
    ) -> 'InvoiceGenerator':
        """Set color theme."""
        if primary:
            self._primary_color = primary
        if secondary:
            self._secondary_color = secondary
        if background:
            self._background_color = background
        return self

    def set_template(self, template: str) -> 'InvoiceGenerator':
        """Set template style."""
        self._template = template
        return self

    def set_font(self, font: str) -> 'InvoiceGenerator':
        """Set font."""
        self._font = font
        return self

    def _hex_to_color(self, hex_color: str):
        """Convert hex color to reportlab color."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        return colors.Color(r, g, b)

    def _calculate_totals(self) -> Dict[str, float]:
        """Calculate invoice totals."""
        subtotal = sum(item.subtotal for item in self._items)

        # Apply discount
        discount_amount = 0
        if self._discount > 0:
            if self._discount_is_percentage:
                discount_amount = subtotal * (self._discount / 100)
            else:
                discount_amount = self._discount

        taxable = subtotal - discount_amount

        # Calculate taxes
        tax_amounts = {}
        total_tax = 0
        current_base = taxable

        # Non-compound taxes first
        for tax in self._taxes:
            if not tax.compound:
                amount = current_base * (tax.rate / 100)
                tax_amounts[tax.name] = amount
                total_tax += amount

        # Compound taxes
        compound_base = current_base + sum(
            amt for name, amt in tax_amounts.items()
            if not any(t.compound and t.name == name for t in self._taxes)
        )
        for tax in self._taxes:
            if tax.compound:
                amount = compound_base * (tax.rate / 100)
                tax_amounts[tax.name] = amount
                total_tax += amount

        total = taxable + total_tax

        return {
            'subtotal': subtotal,
            'discount': discount_amount,
            'taxes': tax_amounts,
            'total_tax': total_tax,
            'total': total
        }

    def _format_currency(self, amount: float) -> str:
        """Format amount with currency."""
        return f"{self._currency_symbol}{amount:,.2f}"

    def generate(self) -> 'InvoiceGenerator':
        """Generate the invoice PDF."""
        self._pdf_buffer = BytesIO()

        doc = SimpleDocTemplate(
            self._pdf_buffer,
            pagesize=letter,
            rightMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch
        )

        elements = []
        styles = getSampleStyleSheet()
        primary_color = self._hex_to_color(self._primary_color)

        # Custom styles
        title_style = ParagraphStyle(
            'InvoiceTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=primary_color,
            spaceAfter=6
        )

        header_style = ParagraphStyle(
            'Header',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey
        )

        normal_style = ParagraphStyle(
            'NormalText',
            parent=styles['Normal'],
            fontSize=10
        )

        # Header section
        header_data = []

        # Logo or company name
        logo_cell = []
        if self._company and self._company.logo_path and Path(self._company.logo_path).exists():
            try:
                logo = Image(self._company.logo_path, width=getattr(self, '_logo_width', 150))
                logo_cell.append(logo)
            except Exception:
                if self._company:
                    logo_cell.append(Paragraph(self._company.name, title_style))
        elif self._company:
            logo_cell.append(Paragraph(self._company.name, title_style))

        # Invoice info
        invoice_info = []
        invoice_info.append(Paragraph("INVOICE", title_style))
        if self._invoice_number:
            invoice_info.append(Paragraph(f"#{self._invoice_number}", header_style))
        invoice_info.append(Paragraph(f"Date: {self._date}", header_style))
        if self._due_date:
            invoice_info.append(Paragraph(f"Due: {self._due_date}", header_style))

        header_data.append([logo_cell, invoice_info])

        header_table = Table(header_data, colWidths=[4 * inch, 3 * inch])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 0.3 * inch))

        # Company and client info
        address_data = []

        # From (company)
        from_cell = []
        from_cell.append(Paragraph("<b>From:</b>", normal_style))
        if self._company:
            from_cell.append(Paragraph(self._company.name, normal_style))
            for line in self._company.address.split('\n'):
                from_cell.append(Paragraph(line, normal_style))
            if self._company.email:
                from_cell.append(Paragraph(self._company.email, normal_style))
            if self._company.phone:
                from_cell.append(Paragraph(self._company.phone, normal_style))

        # To (client)
        to_cell = []
        to_cell.append(Paragraph("<b>Bill To:</b>", normal_style))
        if self._client:
            if self._client.company:
                to_cell.append(Paragraph(self._client.company, normal_style))
            to_cell.append(Paragraph(self._client.name, normal_style))
            for line in self._client.address.split('\n'):
                to_cell.append(Paragraph(line, normal_style))
            if self._client.email:
                to_cell.append(Paragraph(self._client.email, normal_style))

        address_data.append([from_cell, to_cell])

        address_table = Table(address_data, colWidths=[3.5 * inch, 3.5 * inch])
        address_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        elements.append(address_table)
        elements.append(Spacer(1, 0.3 * inch))

        # Line items table
        items_header = ['Description', 'Qty', 'Rate', 'Amount']
        items_data = [items_header]

        for item in self._items:
            qty_str = f"{item.quantity:.0f}" if item.quantity == int(item.quantity) else f"{item.quantity:.2f}"
            if item.unit:
                qty_str += f" {item.unit}"

            row = [
                item.description,
                qty_str,
                self._format_currency(item.rate),
                self._format_currency(item.subtotal)
            ]
            items_data.append(row)

        items_table = Table(
            items_data,
            colWidths=[3.5 * inch, 1 * inch, 1.25 * inch, 1.25 * inch]
        )

        items_table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), f'{self._font}-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),

            # Body
            ('FONTNAME', (0, 1), (-1, -1), self._font),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),

            # Alignment
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),

            # Grid
            ('LINEBELOW', (0, 0), (-1, 0), 1, primary_color),
            ('LINEBELOW', (0, -1), (-1, -1), 1, colors.lightgrey),

            # Alternating rows
            *[('BACKGROUND', (0, i), (-1, i), colors.Color(0.97, 0.97, 0.97))
              for i in range(2, len(items_data), 2)]
        ]))

        elements.append(items_table)
        elements.append(Spacer(1, 0.2 * inch))

        # Totals
        totals = self._calculate_totals()

        totals_data = []
        totals_data.append(['', '', 'Subtotal:', self._format_currency(totals['subtotal'])])

        if totals['discount'] > 0:
            disc_label = f"Discount ({self._discount}%):" if self._discount_is_percentage else "Discount:"
            totals_data.append(['', '', disc_label, f"-{self._format_currency(totals['discount'])}"])

        for tax_name, tax_amount in totals['taxes'].items():
            tax_obj = next((t for t in self._taxes if t.name == tax_name), None)
            rate_str = f" ({tax_obj.rate}%)" if tax_obj else ""
            totals_data.append(['', '', f"{tax_name}{rate_str}:", self._format_currency(tax_amount)])

        totals_data.append(['', '', 'Total:', self._format_currency(totals['total'])])

        totals_table = Table(
            totals_data,
            colWidths=[3.5 * inch, 1 * inch, 1.25 * inch, 1.25 * inch]
        )

        totals_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -2), self._font),
            ('FONTNAME', (0, -1), (-1, -1), f'{self._font}-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
            ('LINEABOVE', (2, -1), (-1, -1), 1, primary_color),
            ('TOPPADDING', (0, -1), (-1, -1), 8),
        ]))

        elements.append(totals_table)
        elements.append(Spacer(1, 0.3 * inch))

        # Payment terms
        if self._payment_terms:
            elements.append(Paragraph(f"<b>Payment Terms:</b> {self._payment_terms}", normal_style))
            elements.append(Spacer(1, 0.1 * inch))

        # Bank details
        if any(self._bank_details.values()):
            elements.append(Paragraph("<b>Bank Details:</b>", normal_style))
            for key, value in self._bank_details.items():
                if value:
                    label = key.replace('_', ' ').title()
                    elements.append(Paragraph(f"{label}: {value}", normal_style))
            elements.append(Spacer(1, 0.1 * inch))

        # Payment instructions
        if self._payment_instructions:
            elements.append(Paragraph("<b>Payment Instructions:</b>", normal_style))
            for line in self._payment_instructions.strip().split('\n'):
                elements.append(Paragraph(line, normal_style))
            elements.append(Spacer(1, 0.1 * inch))

        # Notes
        if self._notes:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph(f"<b>Notes:</b> {self._notes}", normal_style))

        # Terms
        if self._terms:
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("<b>Terms & Conditions:</b>", normal_style))
            for line in self._terms.strip().split('\n'):
                elements.append(Paragraph(line, normal_style))

        # Build PDF
        doc.build(elements)

        return self

    def save(self, path: str) -> str:
        """Save invoice to PDF file."""
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


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate professional PDF invoices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python invoice_gen.py --input invoice.json --output invoice.pdf
  python invoice_gen.py --batch invoices.csv --output-dir ./invoices/
  python invoice_gen.py --quick --company "My Co" --client "Client" --items "Service,1,500" --output inv.pdf
        """
    )

    parser.add_argument('--input', '-i', help='Input JSON file')
    parser.add_argument('--batch', '-b', help='Batch CSV file')
    parser.add_argument('--output', '-o', default='invoice.pdf', help='Output PDF path')
    parser.add_argument('--output-dir', '-d', default='./', help='Output directory (batch)')

    parser.add_argument('--quick', '-q', action='store_true', help='Quick mode')
    parser.add_argument('--company', help='Company name (quick mode)')
    parser.add_argument('--client', help='Client name (quick mode)')
    parser.add_argument('--items', help='Items: "desc,qty,rate;..." (quick mode)')

    parser.add_argument('--template', default='modern', help='Template style')
    parser.add_argument('--currency', default='USD', help='Currency code')
    parser.add_argument('--logo', help='Logo image path')

    args = parser.parse_args()

    try:
        if args.batch:
            # Batch mode
            invoices = InvoiceGenerator.from_csv(args.batch)
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for inv in invoices:
                if args.logo:
                    inv.set_logo(args.logo)
                inv.set_currency(args.currency)
                inv.generate()
                filename = f"{inv.invoice_number or 'invoice'}.pdf"
                inv.save(output_dir / filename)
                print(f"Generated: {filename}")

        elif args.input:
            # JSON mode
            with open(args.input, 'r') as f:
                data = json.load(f)
            invoice = InvoiceGenerator.from_dict(data)
            if args.logo:
                invoice.set_logo(args.logo)
            invoice.set_currency(args.currency)
            invoice.generate().save(args.output)
            print(f"Invoice saved to: {args.output}")

        elif args.quick:
            # Quick mode
            if not args.company or not args.client or not args.items:
                parser.error("Quick mode requires --company, --client, and --items")

            invoice = InvoiceGenerator()
            invoice.set_company(args.company, "")
            invoice.set_client(args.client, "")

            for item_str in args.items.split(';'):
                parts = item_str.split(',')
                if len(parts) >= 3:
                    invoice.add_item(parts[0], float(parts[1]), float(parts[2]))

            if args.logo:
                invoice.set_logo(args.logo)
            invoice.set_currency(args.currency)
            invoice.generate().save(args.output)
            print(f"Invoice saved to: {args.output}")

        else:
            parser.error("Specify --input, --batch, or --quick mode")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
