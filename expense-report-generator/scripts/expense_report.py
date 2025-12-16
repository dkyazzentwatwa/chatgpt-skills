#!/usr/bin/env python3
"""
Expense Report Generator - Create professional expense reports from receipt data.
"""

import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional
import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ExpenseReportGenerator:
    """Generate formatted expense reports."""

    CATEGORIES = [
        "Meals", "Transportation", "Travel", "Lodging", "Supplies",
        "Communication", "Professional", "Entertainment", "Software", "Other"
    ]

    def __init__(self):
        """Initialize the generator."""
        self.expenses = []
        self.employee_name = ""
        self.employee_id = ""
        self.department = ""
        self.period_start = None
        self.period_end = None
        self.approver_name = ""
        self.approver_title = ""
        self.project_name = ""
        self.project_code = ""
        self.policy = {}
        self._expense_counter = 0

    def set_employee(self, name: str, employee_id: str = None,
                    department: str = None) -> 'ExpenseReportGenerator':
        """Set employee information."""
        self.employee_name = name
        self.employee_id = employee_id or ""
        self.department = department or ""
        return self

    def set_period(self, start: str, end: str) -> 'ExpenseReportGenerator':
        """Set reporting period."""
        self.period_start = pd.to_datetime(start)
        self.period_end = pd.to_datetime(end)
        return self

    def set_approver(self, name: str, title: str = None) -> 'ExpenseReportGenerator':
        """Set approver information."""
        self.approver_name = name
        self.approver_title = title or ""
        return self

    def set_project(self, project_name: str, project_code: str = None) -> 'ExpenseReportGenerator':
        """Set project information."""
        self.project_name = project_name
        self.project_code = project_code or ""
        return self

    def add_expense(self, date: str, description: str, category: str,
                   amount: float, receipt: str = None, notes: str = None,
                   reimbursable: bool = True) -> 'ExpenseReportGenerator':
        """Add a single expense."""
        self._expense_counter += 1

        expense = {
            "id": self._expense_counter,
            "date": pd.to_datetime(date),
            "description": description,
            "category": category if category in self.CATEGORIES else "Other",
            "amount": float(amount),
            "receipt": receipt,
            "notes": notes or "",
            "reimbursable": reimbursable
        }

        self.expenses.append(expense)
        return self

    def load_csv(self, filepath: str) -> 'ExpenseReportGenerator':
        """Load expenses from CSV file."""
        df = pd.read_csv(filepath)

        # Map column names
        column_map = {
            'date': ['date', 'expense_date', 'transaction_date'],
            'description': ['description', 'desc', 'merchant', 'vendor'],
            'category': ['category', 'type', 'expense_type'],
            'amount': ['amount', 'total', 'cost'],
            'receipt': ['receipt', 'receipt_file', 'attachment'],
            'notes': ['notes', 'comments', 'memo']
        }

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        for standard, aliases in column_map.items():
            for alias in aliases:
                if alias in df.columns and standard not in df.columns:
                    df = df.rename(columns={alias: standard})
                    break

        # Add expenses
        for _, row in df.iterrows():
            self.add_expense(
                date=row.get('date', datetime.now().strftime('%Y-%m-%d')),
                description=row.get('description', 'Expense'),
                category=row.get('category', 'Other'),
                amount=row.get('amount', 0),
                receipt=row.get('receipt'),
                notes=row.get('notes')
            )

        return self

    def load_json(self, filepath: str) -> 'ExpenseReportGenerator':
        """Load expenses from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        # Set metadata if present
        if 'employee' in data:
            self.employee_name = data['employee']
        if 'employee_id' in data:
            self.employee_id = data['employee_id']
        if 'department' in data:
            self.department = data['department']
        if 'period' in data:
            self.set_period(data['period']['start'], data['period']['end'])

        # Add expenses
        for exp in data.get('expenses', []):
            self.add_expense(
                date=exp.get('date'),
                description=exp.get('description', 'Expense'),
                category=exp.get('category', 'Other'),
                amount=exp.get('amount', 0),
                receipt=exp.get('receipt'),
                notes=exp.get('notes')
            )

        return self

    def set_policy(self, policy: Dict) -> 'ExpenseReportGenerator':
        """Set expense policy for compliance checking."""
        self.policy = policy
        return self

    def check_compliance(self) -> List[Dict]:
        """Check expenses against policy."""
        violations = []

        if not self.policy:
            return violations

        limits = self.policy.get('limits', {})
        receipt_threshold = self.policy.get('requires_receipt', 0)
        approval_threshold = self.policy.get('requires_approval', float('inf'))
        prohibited = self.policy.get('prohibited', [])

        for exp in self.expenses:
            # Check category limits
            if exp['category'] in limits:
                if exp['amount'] > limits[exp['category']]:
                    violations.append({
                        "expense_id": exp['id'],
                        "type": "over_limit",
                        "category": exp['category'],
                        "amount": exp['amount'],
                        "limit": limits[exp['category']],
                        "description": exp['description']
                    })

            # Check receipt requirement
            if exp['amount'] >= receipt_threshold and not exp['receipt']:
                violations.append({
                    "expense_id": exp['id'],
                    "type": "missing_receipt",
                    "amount": exp['amount'],
                    "threshold": receipt_threshold,
                    "description": exp['description']
                })

            # Check approval requirement
            if exp['amount'] >= approval_threshold:
                violations.append({
                    "expense_id": exp['id'],
                    "type": "requires_approval",
                    "amount": exp['amount'],
                    "threshold": approval_threshold,
                    "description": exp['description']
                })

            # Check prohibited categories
            for prohibited_term in prohibited:
                if prohibited_term.lower() in exp['description'].lower():
                    violations.append({
                        "expense_id": exp['id'],
                        "type": "prohibited",
                        "term": prohibited_term,
                        "description": exp['description']
                    })

        return violations

    def get_summary(self) -> Dict:
        """Get expense report summary."""
        if not self.expenses:
            return {}

        df = pd.DataFrame(self.expenses)

        total = df['amount'].sum()
        reimbursable = df[df['reimbursable'] == True]['amount'].sum()
        non_reimbursable = total - reimbursable

        receipts_attached = df['receipt'].notna().sum()
        receipts_missing = len(df) - receipts_attached

        categories = df.groupby('category')['amount'].sum().to_dict()

        return {
            "employee": self.employee_name,
            "employee_id": self.employee_id,
            "department": self.department,
            "period": {
                "start": self.period_start.strftime('%Y-%m-%d') if self.period_start else None,
                "end": self.period_end.strftime('%Y-%m-%d') if self.period_end else None
            },
            "total_expenses": round(total, 2),
            "expense_count": len(self.expenses),
            "categories": {k: round(v, 2) for k, v in categories.items()},
            "reimbursable": round(reimbursable, 2),
            "non_reimbursable": round(non_reimbursable, 2),
            "receipts_attached": int(receipts_attached),
            "receipts_missing": int(receipts_missing)
        }

    def by_category(self) -> Dict[str, float]:
        """Get expenses grouped by category."""
        df = pd.DataFrame(self.expenses)
        return df.groupby('category')['amount'].sum().round(2).to_dict()

    def by_date(self) -> Dict[str, float]:
        """Get expenses grouped by date."""
        df = pd.DataFrame(self.expenses)
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        return df.groupby('date_str')['amount'].sum().round(2).to_dict()

    def get_total(self) -> float:
        """Get total expense amount."""
        return sum(exp['amount'] for exp in self.expenses)

    def generate_pdf(self, output: str) -> str:
        """Generate PDF expense report."""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from io import BytesIO

        doc = SimpleDocTemplate(output, pagesize=letter,
                               leftMargin=0.75*inch, rightMargin=0.75*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=20,
            alignment=1  # Center
        )
        story.append(Paragraph("EXPENSE REPORT", title_style))
        story.append(Spacer(1, 10))

        # Employee Info
        info_data = [
            ["Employee:", self.employee_name, "Employee ID:", self.employee_id or "N/A"],
            ["Department:", self.department or "N/A", "Period:",
             f"{self.period_start.strftime('%m/%d/%Y') if self.period_start else 'N/A'} - "
             f"{self.period_end.strftime('%m/%d/%Y') if self.period_end else 'N/A'}"]
        ]

        if self.project_name:
            info_data.append(["Project:", self.project_name, "Project Code:", self.project_code or "N/A"])

        info_table = Table(info_data, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 20))

        # Summary
        summary = self.get_summary()
        story.append(Paragraph("Summary", styles['Heading2']))

        summary_data = [
            ["Total Expenses:", f"${summary['total_expenses']:,.2f}"],
            ["Number of Items:", str(summary['expense_count'])],
            ["Reimbursable:", f"${summary['reimbursable']:,.2f}"],
            ["Non-Reimbursable:", f"${summary['non_reimbursable']:,.2f}"],
            ["Receipts Attached:", str(summary['receipts_attached'])],
            ["Receipts Missing:", str(summary['receipts_missing'])]
        ]

        summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))

        # Expense Details
        story.append(Paragraph("Expense Details", styles['Heading2']))

        expense_data = [["Date", "Description", "Category", "Amount", "Receipt"]]
        for exp in sorted(self.expenses, key=lambda x: x['date']):
            expense_data.append([
                exp['date'].strftime('%m/%d/%Y'),
                exp['description'][:40] + "..." if len(exp['description']) > 40 else exp['description'],
                exp['category'],
                f"${exp['amount']:,.2f}",
                "Yes" if exp['receipt'] else "No"
            ])

        # Add total row
        expense_data.append(["", "", "TOTAL", f"${self.get_total():,.2f}", ""])

        expense_table = Table(expense_data, colWidths=[0.9*inch, 2.5*inch, 1.2*inch, 1*inch, 0.7*inch])
        expense_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.steelblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
            ('ALIGN', (4, 0), (4, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -2), 0.5, colors.grey),
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('LINEABOVE', (0, -1), (-1, -1), 1, colors.black),
        ]))
        story.append(expense_table)
        story.append(Spacer(1, 20))

        # Category Breakdown
        categories = self.by_category()
        if categories:
            story.append(Paragraph("Category Breakdown", styles['Heading2']))

            cat_data = [["Category", "Amount", "% of Total"]]
            total = self.get_total()
            for cat, amount in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                pct = (amount / total * 100) if total > 0 else 0
                cat_data.append([cat, f"${amount:,.2f}", f"{pct:.1f}%"])

            cat_table = Table(cat_data, colWidths=[2*inch, 1.5*inch, 1*inch])
            cat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.steelblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(cat_table)
            story.append(Spacer(1, 20))

        # Compliance violations
        violations = self.check_compliance()
        if violations:
            story.append(Paragraph("Policy Compliance Notes", styles['Heading2']))
            for v in violations:
                story.append(Paragraph(
                    f"• {v['type'].replace('_', ' ').title()}: {v['description']} (${v.get('amount', 0):,.2f})",
                    styles['Normal']
                ))
            story.append(Spacer(1, 20))

        # Approval Section
        story.append(Paragraph("Approvals", styles['Heading2']))
        story.append(Spacer(1, 10))

        approval_data = [
            ["Employee Signature:", "_" * 30, "Date:", "_" * 15],
            ["", "", "", ""],
            [f"Approver ({self.approver_name or 'Manager'}):", "_" * 30, "Date:", "_" * 15],
        ]

        approval_table = Table(approval_data, colWidths=[2*inch, 2.5*inch, 0.6*inch, 1.2*inch])
        approval_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ]))
        story.append(approval_table)

        doc.build(story)
        return output

    def generate_html(self, output: str) -> str:
        """Generate HTML expense report."""
        summary = self.get_summary()
        categories = self.by_category()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Expense Report - {self.employee_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; text-align: center; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px; }}
        .info-item {{ display: flex; }}
        .info-label {{ font-weight: bold; width: 120px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background-color: #3498db; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .total-row {{ font-weight: bold; background-color: #ecf0f1; }}
        .summary-box {{ background: #ecf0f1; padding: 20px; border-radius: 8px; }}
        .amount {{ text-align: right; }}
        .receipt-yes {{ color: green; }}
        .receipt-no {{ color: red; }}
    </style>
</head>
<body>
    <h1>EXPENSE REPORT</h1>

    <div class="info-grid">
        <div class="info-item"><span class="info-label">Employee:</span> {self.employee_name}</div>
        <div class="info-item"><span class="info-label">Employee ID:</span> {self.employee_id or 'N/A'}</div>
        <div class="info-item"><span class="info-label">Department:</span> {self.department or 'N/A'}</div>
        <div class="info-item"><span class="info-label">Period:</span> {summary['period']['start'] or 'N/A'} - {summary['period']['end'] or 'N/A'}</div>
    </div>

    <div class="summary-box">
        <h2>Summary</h2>
        <p><strong>Total Expenses:</strong> ${summary['total_expenses']:,.2f}</p>
        <p><strong>Number of Items:</strong> {summary['expense_count']}</p>
        <p><strong>Reimbursable:</strong> ${summary['reimbursable']:,.2f}</p>
        <p><strong>Receipts:</strong> {summary['receipts_attached']} attached, {summary['receipts_missing']} missing</p>
    </div>

    <h2>Expense Details</h2>
    <table>
        <tr>
            <th>Date</th>
            <th>Description</th>
            <th>Category</th>
            <th class="amount">Amount</th>
            <th>Receipt</th>
        </tr>
"""

        for exp in sorted(self.expenses, key=lambda x: x['date']):
            receipt_class = "receipt-yes" if exp['receipt'] else "receipt-no"
            receipt_text = "Yes" if exp['receipt'] else "No"
            html += f"""
        <tr>
            <td>{exp['date'].strftime('%m/%d/%Y')}</td>
            <td>{exp['description']}</td>
            <td>{exp['category']}</td>
            <td class="amount">${exp['amount']:,.2f}</td>
            <td class="{receipt_class}">{receipt_text}</td>
        </tr>
"""

        html += f"""
        <tr class="total-row">
            <td colspan="3">TOTAL</td>
            <td class="amount">${self.get_total():,.2f}</td>
            <td></td>
        </tr>
    </table>

    <h2>Category Breakdown</h2>
    <table>
        <tr><th>Category</th><th class="amount">Amount</th><th class="amount">% of Total</th></tr>
"""

        total = self.get_total()
        for cat, amount in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            pct = (amount / total * 100) if total > 0 else 0
            html += f"""
        <tr><td>{cat}</td><td class="amount">${amount:,.2f}</td><td class="amount">{pct:.1f}%</td></tr>
"""

        html += """
    </table>
</body>
</html>
"""

        with open(output, 'w') as f:
            f.write(html)

        return output

    def to_csv(self, output: str) -> str:
        """Export expenses to CSV."""
        df = pd.DataFrame(self.expenses)
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df.to_csv(output, index=False)
        return output

    def to_json(self, output: str) -> str:
        """Export expenses to JSON."""
        data = {
            "employee": self.employee_name,
            "employee_id": self.employee_id,
            "department": self.department,
            "period": {
                "start": self.period_start.strftime('%Y-%m-%d') if self.period_start else None,
                "end": self.period_end.strftime('%Y-%m-%d') if self.period_end else None
            },
            "summary": self.get_summary(),
            "expenses": [
                {**exp, "date": exp['date'].strftime('%Y-%m-%d')}
                for exp in self.expenses
            ]
        }

        with open(output, 'w') as f:
            json.dump(data, f, indent=2)

        return output


def main():
    parser = argparse.ArgumentParser(description="Expense Report Generator")

    parser.add_argument("--input", "-i", help="Input CSV or JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output file (PDF or HTML)")

    parser.add_argument("--employee", "-e", help="Employee name")
    parser.add_argument("--employee-id", help="Employee ID")
    parser.add_argument("--dept", "-d", help="Department")

    parser.add_argument("--start", help="Period start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Period end date (YYYY-MM-DD)")

    parser.add_argument("--approver", help="Approver name")
    parser.add_argument("--project", help="Project name")

    parser.add_argument("--policy", help="Policy JSON file")

    args = parser.parse_args()

    report = ExpenseReportGenerator()

    # Set employee info
    if args.employee:
        report.set_employee(args.employee, args.employee_id, args.dept)

    # Set period
    if args.start and args.end:
        report.set_period(args.start, args.end)

    # Set approver
    if args.approver:
        report.set_approver(args.approver)

    # Set project
    if args.project:
        report.set_project(args.project)

    # Load expenses
    if args.input:
        if args.input.endswith('.json'):
            report.load_json(args.input)
        else:
            report.load_csv(args.input)

    # Load policy
    if args.policy:
        with open(args.policy) as f:
            report.set_policy(json.load(f))

    # Check compliance
    violations = report.check_compliance()
    if violations:
        print(f"Warning: {len(violations)} policy violations found")
        for v in violations:
            print(f"  - {v['type']}: {v['description']}")

    # Generate output
    if args.output.endswith('.html'):
        report.generate_html(args.output)
    else:
        report.generate_pdf(args.output)

    # Show summary
    summary = report.get_summary()
    print(f"\n=== Expense Report Summary ===")
    print(f"Employee: {summary.get('employee', 'N/A')}")
    print(f"Total Expenses: ${summary.get('total_expenses', 0):,.2f}")
    print(f"Number of Items: {summary.get('expense_count', 0)}")
    print(f"Reimbursable: ${summary.get('reimbursable', 0):,.2f}")
    print(f"\nReport generated: {args.output}")


if __name__ == "__main__":
    main()
