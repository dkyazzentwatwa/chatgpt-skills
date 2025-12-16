#!/usr/bin/env python3
"""
Budget Analyzer - Analyze expenses, track budgets, get savings recommendations.
"""

import argparse
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


class BudgetAnalyzer:
    """Analyze personal or business expenses."""

    DEFAULT_CATEGORIES = {
        "Food & Dining": ["restaurant", "cafe", "starbucks", "mcdonald", "uber eats",
                         "doordash", "grubhub", "chipotle", "subway", "pizza", "food"],
        "Transportation": ["uber", "lyft", "gas", "shell", "chevron", "parking",
                          "transit", "metro", "bus", "taxi", "fuel"],
        "Shopping": ["amazon", "walmart", "target", "costco", "best buy", "ebay",
                    "etsy", "shop", "store", "market"],
        "Utilities": ["electric", "water", "gas bill", "internet", "phone", "verizon",
                     "at&t", "comcast", "utility", "power"],
        "Entertainment": ["netflix", "spotify", "hulu", "disney", "movie", "theater",
                         "concert", "game", "steam"],
        "Healthcare": ["pharmacy", "cvs", "walgreens", "doctor", "hospital", "medical",
                      "dental", "vision", "health"],
        "Travel": ["airline", "hotel", "airbnb", "booking", "flight", "vacation",
                  "resort", "travel"],
        "Subscriptions": ["subscription", "membership", "monthly", "annual", "premium"],
        "Insurance": ["insurance", "geico", "state farm", "allstate", "progressive"],
        "Education": ["tuition", "school", "university", "course", "book", "udemy"],
    }

    def __init__(self):
        """Initialize the analyzer."""
        self.df = None
        self.categories = self.DEFAULT_CATEGORIES.copy()
        self.budget = {}
        self.date_col = "date"
        self.amount_col = "amount"
        self.description_col = "description"
        self.category_col = "category"

    def load_csv(self, filepath: str, date_col: str, amount_col: str,
                 description_col: str = None, category_col: str = None) -> 'BudgetAnalyzer':
        """Load transaction data from CSV."""
        self.df = pd.read_csv(filepath)
        self.date_col = date_col
        self.amount_col = amount_col
        self.description_col = description_col or "description"
        self.category_col = category_col or "category"

        # Parse dates
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        # Ensure amount is numeric and positive (expenses)
        self.df[self.amount_col] = pd.to_numeric(self.df[self.amount_col], errors='coerce').abs()

        # Add category column if not present
        if self.category_col not in self.df.columns:
            self.df[self.category_col] = "Uncategorized"

        return self

    def load_dataframe(self, df: pd.DataFrame, date_col: str = "date",
                       amount_col: str = "amount", description_col: str = "description",
                       category_col: str = "category") -> 'BudgetAnalyzer':
        """Load from existing DataFrame."""
        self.df = df.copy()
        self.date_col = date_col
        self.amount_col = amount_col
        self.description_col = description_col
        self.category_col = category_col

        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df[self.amount_col] = pd.to_numeric(self.df[self.amount_col], errors='coerce').abs()

        if self.category_col not in self.df.columns:
            self.df[self.category_col] = "Uncategorized"

        return self

    def set_categories(self, categories: Dict[str, List[str]]) -> 'BudgetAnalyzer':
        """Set custom category mappings."""
        self.categories = categories
        return self

    def auto_categorize(self) -> 'BudgetAnalyzer':
        """Auto-categorize transactions based on description."""
        if self.description_col not in self.df.columns:
            return self

        def categorize(description: str) -> str:
            if pd.isna(description):
                return "Uncategorized"
            desc_lower = str(description).lower()
            for category, keywords in self.categories.items():
                for keyword in keywords:
                    if keyword.lower() in desc_lower:
                        return category
            return "Other"

        self.df[self.category_col] = self.df[self.description_col].apply(categorize)
        return self

    def set_budget(self, budget: Dict[str, float]) -> 'BudgetAnalyzer':
        """Set budget targets by category."""
        self.budget = budget
        return self

    def analyze(self) -> Dict:
        """Get full analysis summary."""
        if self.df is None or len(self.df) == 0:
            return {}

        total = self.df[self.amount_col].sum()
        count = len(self.df)
        avg = self.df[self.amount_col].mean()
        largest_idx = self.df[self.amount_col].idxmax()
        largest = self.df.loc[largest_idx]

        categories = self.df.groupby(self.category_col)[self.amount_col].sum().to_dict()

        return {
            "total_spent": round(total, 2),
            "transaction_count": count,
            "date_range": {
                "start": self.df[self.date_col].min().strftime("%Y-%m-%d"),
                "end": self.df[self.date_col].max().strftime("%Y-%m-%d")
            },
            "average_transaction": round(avg, 2),
            "median_transaction": round(self.df[self.amount_col].median(), 2),
            "largest_expense": {
                "amount": round(largest[self.amount_col], 2),
                "description": largest.get(self.description_col, "N/A") if self.description_col in self.df.columns else "N/A",
                "date": largest[self.date_col].strftime("%Y-%m-%d")
            },
            "categories": {k: round(v, 2) for k, v in categories.items()}
        }

    def by_category(self) -> pd.DataFrame:
        """Get spending breakdown by category."""
        if self.df is None:
            return pd.DataFrame()

        grouped = self.df.groupby(self.category_col).agg({
            self.amount_col: ['sum', 'count', 'mean']
        }).reset_index()

        grouped.columns = ['category', 'amount', 'count', 'average']
        total = grouped['amount'].sum()
        grouped['percentage'] = (grouped['amount'] / total * 100).round(1)
        grouped = grouped.sort_values('amount', ascending=False)
        grouped['amount'] = grouped['amount'].round(2)
        grouped['average'] = grouped['average'].round(2)

        return grouped

    def by_month(self) -> pd.DataFrame:
        """Get spending by month."""
        if self.df is None:
            return pd.DataFrame()

        df = self.df.copy()
        df['month'] = df[self.date_col].dt.to_period('M')

        grouped = df.groupby('month').agg({
            self.amount_col: ['sum', 'count', 'mean']
        }).reset_index()

        grouped.columns = ['month', 'total', 'count', 'avg_transaction']
        grouped['month'] = grouped['month'].astype(str)
        grouped['total'] = grouped['total'].round(2)
        grouped['avg_transaction'] = grouped['avg_transaction'].round(2)

        return grouped

    def by_day_of_week(self) -> pd.DataFrame:
        """Get spending by day of week."""
        if self.df is None:
            return pd.DataFrame()

        df = self.df.copy()
        df['day_of_week'] = df[self.date_col].dt.day_name()

        grouped = df.groupby('day_of_week').agg({
            self.amount_col: ['sum', 'count', 'mean']
        }).reset_index()

        grouped.columns = ['day', 'total', 'count', 'average']

        # Sort by day of week
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        grouped['day'] = pd.Categorical(grouped['day'], categories=days_order, ordered=True)
        grouped = grouped.sort_values('day')
        grouped['total'] = grouped['total'].round(2)
        grouped['average'] = grouped['average'].round(2)

        return grouped

    def top_expenses(self, n: int = 10) -> pd.DataFrame:
        """Get top N expenses."""
        if self.df is None:
            return pd.DataFrame()

        cols = [self.date_col, self.amount_col]
        if self.description_col in self.df.columns:
            cols.append(self.description_col)
        if self.category_col in self.df.columns:
            cols.append(self.category_col)

        return self.df.nlargest(n, self.amount_col)[cols].reset_index(drop=True)

    def recurring_expenses(self, min_occurrences: int = 2) -> pd.DataFrame:
        """Identify recurring expenses."""
        if self.df is None or self.description_col not in self.df.columns:
            return pd.DataFrame()

        # Group by similar descriptions
        grouped = self.df.groupby(self.description_col).agg({
            self.amount_col: ['count', 'sum', 'mean', 'std']
        }).reset_index()

        grouped.columns = ['description', 'occurrences', 'total', 'average', 'std_dev']

        # Filter for recurring (multiple occurrences, similar amounts)
        recurring = grouped[grouped['occurrences'] >= min_occurrences].copy()
        recurring['likely_subscription'] = recurring['std_dev'].fillna(0) < (recurring['average'] * 0.1)
        recurring = recurring.sort_values('occurrences', ascending=False)

        return recurring

    def compare_periods(self, period1: str, period2: str) -> Dict:
        """Compare spending between two periods."""
        if self.df is None:
            return {}

        df = self.df.copy()
        df['period'] = df[self.date_col].dt.to_period('M').astype(str)

        # Support quarter format
        if 'Q' in period1:
            df['period'] = df[self.date_col].dt.to_period('Q').astype(str)

        p1_df = df[df['period'] == period1]
        p2_df = df[df['period'] == period2]

        p1_total = p1_df[self.amount_col].sum()
        p2_total = p2_df[self.amount_col].sum()
        diff = p2_total - p1_total
        pct_change = (diff / p1_total * 100) if p1_total > 0 else 0

        # Category comparison
        p1_cats = p1_df.groupby(self.category_col)[self.amount_col].sum()
        p2_cats = p2_df.groupby(self.category_col)[self.amount_col].sum()

        all_cats = set(p1_cats.index) | set(p2_cats.index)
        category_changes = {}
        for cat in all_cats:
            p1_val = p1_cats.get(cat, 0)
            p2_val = p2_cats.get(cat, 0)
            change = p2_val - p1_val
            pct = (change / p1_val * 100) if p1_val > 0 else (100 if p2_val > 0 else 0)
            category_changes[cat] = {
                "period1": round(p1_val, 2),
                "period2": round(p2_val, 2),
                "change": round(change, 2),
                "percent": round(pct, 1)
            }

        return {
            "period1": period1,
            "period2": period2,
            "period1_total": round(p1_total, 2),
            "period2_total": round(p2_total, 2),
            "difference": round(diff, 2),
            "percent_change": round(pct_change, 1),
            "category_changes": category_changes
        }

    def budget_vs_actual(self) -> pd.DataFrame:
        """Compare actual spending to budget."""
        if not self.budget or self.df is None:
            return pd.DataFrame()

        actual = self.df.groupby(self.category_col)[self.amount_col].sum()

        data = []
        for category, budget_amount in self.budget.items():
            actual_amount = actual.get(category, 0)
            diff = budget_amount - actual_amount
            status = "under" if diff >= 0 else "over"
            if diff >= 0 and actual_amount >= budget_amount * 0.9:
                status = "warning"

            data.append({
                "category": category,
                "budget": budget_amount,
                "actual": round(actual_amount, 2),
                "difference": round(diff, 2),
                "percent_used": round(actual_amount / budget_amount * 100, 1) if budget_amount > 0 else 0,
                "status": status
            })

        return pd.DataFrame(data).sort_values("actual", ascending=False)

    def budget_alerts(self) -> List[Dict]:
        """Get budget alerts for categories over or near budget."""
        comparison = self.budget_vs_actual()
        if comparison.empty:
            return []

        alerts = []
        for _, row in comparison.iterrows():
            if row['status'] == 'over':
                alerts.append({
                    "category": row['category'],
                    "status": "over",
                    "actual": row['actual'],
                    "budget": row['budget'],
                    "percent_over": round((row['actual'] - row['budget']) / row['budget'] * 100, 1)
                })
            elif row['status'] == 'warning':
                alerts.append({
                    "category": row['category'],
                    "status": "warning",
                    "actual": row['actual'],
                    "budget": row['budget'],
                    "percent_used": row['percent_used']
                })

        return alerts

    def get_recommendations(self) -> List[str]:
        """Get spending recommendations."""
        if self.df is None:
            return []

        recommendations = []

        # Category analysis
        categories = self.by_category()
        if not categories.empty:
            top_cat = categories.iloc[0]
            recommendations.append(
                f"Top spending category: {top_cat['category']} (${top_cat['amount']:.2f}, "
                f"{top_cat['percentage']:.1f}% of total)"
            )

        # Month-over-month changes
        monthly = self.by_month()
        if len(monthly) >= 2:
            last_two = monthly.tail(2)
            if len(last_two) == 2:
                change = last_two.iloc[1]['total'] - last_two.iloc[0]['total']
                pct = (change / last_two.iloc[0]['total'] * 100) if last_two.iloc[0]['total'] > 0 else 0
                if pct > 10:
                    recommendations.append(
                        f"Spending increased {pct:.1f}% from {last_two.iloc[0]['month']} to {last_two.iloc[1]['month']}. "
                        "Review recent purchases for unnecessary expenses."
                    )
                elif pct < -10:
                    recommendations.append(
                        f"Great job! Spending decreased {abs(pct):.1f}% from {last_two.iloc[0]['month']}."
                    )

        # Recurring expenses
        recurring = self.recurring_expenses()
        if not recurring.empty:
            subscriptions = recurring[recurring['likely_subscription'] == True]
            if not subscriptions.empty:
                total_subs = subscriptions['average'].sum()
                recommendations.append(
                    f"Found {len(subscriptions)} potential subscriptions totaling ~${total_subs:.2f}/period. "
                    "Review for unused services."
                )

        # Budget alerts
        alerts = self.budget_alerts()
        for alert in alerts:
            if alert['status'] == 'over':
                recommendations.append(
                    f"{alert['category']} is {alert['percent_over']:.1f}% over budget. "
                    f"Actual: ${alert['actual']:.2f}, Budget: ${alert['budget']:.2f}"
                )

        # Day of week patterns
        by_day = self.by_day_of_week()
        if not by_day.empty:
            highest_day = by_day.loc[by_day['total'].idxmax()]
            recommendations.append(
                f"Highest spending day: {highest_day['day']} (${highest_day['total']:.2f}). "
                "Consider planning purchases for other days."
            )

        return recommendations

    def spending_score(self) -> Dict:
        """Calculate overall spending health score."""
        if self.df is None:
            return {}

        scores = []

        # Budget adherence (if budget set)
        if self.budget:
            comparison = self.budget_vs_actual()
            if not comparison.empty:
                under_budget = len(comparison[comparison['status'] == 'under'])
                adherence = (under_budget / len(comparison)) * 100
                scores.append(('budget_adherence', adherence))

        # Spending consistency (low variance = more consistent)
        monthly = self.by_month()
        if len(monthly) >= 3:
            cv = monthly['total'].std() / monthly['total'].mean() if monthly['total'].mean() > 0 else 0
            consistency = max(0, 100 - cv * 100)
            scores.append(('spending_consistency', consistency))

        # Category diversity (not too concentrated)
        categories = self.by_category()
        if not categories.empty:
            top_pct = categories.iloc[0]['percentage']
            diversity = max(0, 100 - (top_pct - 30))  # Penalty if top category > 30%
            scores.append(('category_diversity', diversity))

        if not scores:
            return {"overall_score": 50, "grade": "N/A", "summary": "Insufficient data"}

        overall = np.mean([s[1] for s in scores])
        grade = 'A' if overall >= 90 else 'B' if overall >= 75 else 'C' if overall >= 60 else 'D' if overall >= 40 else 'F'

        return {
            "overall_score": round(overall),
            "factors": {name: round(score) for name, score in scores},
            "grade": grade,
            "summary": self._get_score_summary(grade)
        }

    def _get_score_summary(self, grade: str) -> str:
        """Get summary text for spending score."""
        summaries = {
            'A': "Excellent spending habits. Keep up the great work!",
            'B': "Good spending habits with minor room for improvement.",
            'C': "Average spending habits. Consider reviewing your budget.",
            'D': "Below average. Significant opportunities to improve spending.",
            'F': "Needs attention. Consider creating a strict budget plan."
        }
        return summaries.get(grade, "Unable to assess spending habits.")

    def plot_categories(self, output: str) -> str:
        """Create pie chart of spending by category."""
        categories = self.by_category()
        if categories.empty:
            return ""

        fig, ax = plt.subplots(figsize=(10, 8))

        # Limit to top 8 categories, group rest as "Other"
        if len(categories) > 8:
            top = categories.head(7)
            other_total = categories.iloc[7:]['amount'].sum()
            other_row = pd.DataFrame([{
                'category': 'Other',
                'amount': other_total,
                'percentage': categories.iloc[7:]['percentage'].sum()
            }])
            categories = pd.concat([top, other_row], ignore_index=True)

        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        wedges, texts, autotexts = ax.pie(
            categories['amount'],
            labels=categories['category'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )

        ax.set_title('Spending by Category', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def plot_trends(self, output: str) -> str:
        """Create line chart of monthly spending."""
        monthly = self.by_month()
        if monthly.empty:
            return ""

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(monthly['month'], monthly['total'], marker='o', linewidth=2, markersize=8)
        ax.fill_between(monthly['month'], monthly['total'], alpha=0.3)

        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Total Spending ($)', fontsize=12)
        ax.set_title('Monthly Spending Trend', fontsize=14, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # Add average line
        avg = monthly['total'].mean()
        ax.axhline(y=avg, color='red', linestyle='--', alpha=0.7, label=f'Average: ${avg:.2f}')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def plot_budget_comparison(self, output: str) -> str:
        """Create bar chart comparing budget vs actual."""
        comparison = self.budget_vs_actual()
        if comparison.empty:
            return ""

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(comparison))
        width = 0.35

        bars1 = ax.bar(x - width/2, comparison['budget'], width, label='Budget', color='steelblue')
        bars2 = ax.bar(x + width/2, comparison['actual'], width, label='Actual', color='coral')

        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Amount ($)', fontsize=12)
        ax.set_title('Budget vs Actual Spending', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison['category'], rotation=45, ha='right')
        ax.legend()

        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def generate_report(self, output: str, format: str = "pdf") -> str:
        """Generate comprehensive report."""
        if format == "html":
            return self._generate_html_report(output)
        else:
            return self._generate_pdf_report(output)

    def _generate_pdf_report(self, output: str) -> str:
        """Generate PDF report using reportlab."""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

        doc = SimpleDocTemplate(output, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("Budget Analysis Report", title_style))
        story.append(Spacer(1, 20))

        # Summary
        summary = self.analyze()
        if summary:
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            summary_text = f"""
            <b>Total Spent:</b> ${summary['total_spent']:,.2f}<br/>
            <b>Transactions:</b> {summary['transaction_count']}<br/>
            <b>Average Transaction:</b> ${summary['average_transaction']:,.2f}<br/>
            <b>Date Range:</b> {summary['date_range']['start']} to {summary['date_range']['end']}<br/>
            <b>Largest Expense:</b> ${summary['largest_expense']['amount']:,.2f}
            """
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 20))

        # Category breakdown
        categories = self.by_category()
        if not categories.empty:
            story.append(Paragraph("Spending by Category", styles['Heading2']))

            table_data = [['Category', 'Amount', 'Percentage', 'Count']]
            for _, row in categories.head(10).iterrows():
                table_data.append([
                    row['category'],
                    f"${row['amount']:,.2f}",
                    f"{row['percentage']:.1f}%",
                    str(int(row['count']))
                ])

            table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.steelblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))

        # Budget comparison
        budget_comp = self.budget_vs_actual()
        if not budget_comp.empty:
            story.append(Paragraph("Budget vs Actual", styles['Heading2']))

            table_data = [['Category', 'Budget', 'Actual', 'Difference', 'Status']]
            for _, row in budget_comp.iterrows():
                status_color = 'green' if row['status'] == 'under' else 'orange' if row['status'] == 'warning' else 'red'
                table_data.append([
                    row['category'],
                    f"${row['budget']:,.2f}",
                    f"${row['actual']:,.2f}",
                    f"${row['difference']:,.2f}",
                    row['status'].upper()
                ])

            table = Table(table_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.steelblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))

        # Recommendations
        recommendations = self.get_recommendations()
        if recommendations:
            story.append(Paragraph("Recommendations", styles['Heading2']))
            for rec in recommendations:
                story.append(Paragraph(f"• {rec}", styles['Normal']))
            story.append(Spacer(1, 10))

        # Spending score
        score = self.spending_score()
        if score:
            story.append(Paragraph("Spending Health Score", styles['Heading2']))
            score_text = f"""
            <b>Overall Score:</b> {score['overall_score']}/100 (Grade: {score['grade']})<br/>
            <b>Summary:</b> {score['summary']}
            """
            story.append(Paragraph(score_text, styles['Normal']))

        doc.build(story)
        return output

    def _generate_html_report(self, output: str) -> str:
        """Generate HTML report."""
        summary = self.analyze()
        categories = self.by_category()
        monthly = self.by_month()
        recommendations = self.get_recommendations()
        score = self.spending_score()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Budget Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background-color: #3498db; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .summary {{ background: #ecf0f1; padding: 20px; border-radius: 8px; }}
        .score {{ font-size: 48px; font-weight: bold; color: #27ae60; }}
        .recommendation {{ background: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
        .over {{ color: #e74c3c; }}
        .under {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
    </style>
</head>
<body>
    <h1>Budget Analysis Report</h1>

    <div class="summary">
        <h2>Executive Summary</h2>
        <p><strong>Total Spent:</strong> ${summary.get('total_spent', 0):,.2f}</p>
        <p><strong>Transactions:</strong> {summary.get('transaction_count', 0)}</p>
        <p><strong>Average Transaction:</strong> ${summary.get('average_transaction', 0):,.2f}</p>
        <p><strong>Date Range:</strong> {summary.get('date_range', {}).get('start', 'N/A')} to {summary.get('date_range', {}).get('end', 'N/A')}</p>
    </div>

    <h2>Spending by Category</h2>
    <table>
        <tr><th>Category</th><th>Amount</th><th>Percentage</th><th>Transactions</th></tr>
"""

        for _, row in categories.iterrows():
            html += f"<tr><td>{row['category']}</td><td>${row['amount']:,.2f}</td><td>{row['percentage']:.1f}%</td><td>{int(row['count'])}</td></tr>\n"

        html += """
    </table>

    <h2>Monthly Trends</h2>
    <table>
        <tr><th>Month</th><th>Total</th><th>Transactions</th><th>Average</th></tr>
"""

        for _, row in monthly.iterrows():
            html += f"<tr><td>{row['month']}</td><td>${row['total']:,.2f}</td><td>{int(row['count'])}</td><td>${row['avg_transaction']:,.2f}</td></tr>\n"

        html += """
    </table>

    <h2>Recommendations</h2>
"""
        for rec in recommendations:
            html += f'<div class="recommendation">{rec}</div>\n'

        html += f"""
    <h2>Spending Health Score</h2>
    <div class="score">{score.get('overall_score', 'N/A')}/100</div>
    <p><strong>Grade:</strong> {score.get('grade', 'N/A')}</p>
    <p>{score.get('summary', '')}</p>

</body>
</html>
"""

        with open(output, 'w') as f:
            f.write(html)

        return output

    def to_csv(self, output: str) -> str:
        """Export analyzed data to CSV."""
        if self.df is not None:
            self.df.to_csv(output, index=False)
        return output


def main():
    parser = argparse.ArgumentParser(description="Budget Analyzer - Analyze expenses and track budgets")

    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--date", "-d", default="date", help="Date column name")
    parser.add_argument("--amount", "-a", default="amount", help="Amount column name")
    parser.add_argument("--description", default="description", help="Description column name")
    parser.add_argument("--category-col", help="Category column name (if pre-categorized)")

    parser.add_argument("--categories", help="JSON file with custom categories")
    parser.add_argument("--budget", help="JSON file with budget targets")

    parser.add_argument("--compare", nargs=2, metavar=('PERIOD1', 'PERIOD2'),
                       help="Compare two periods (e.g., 2024-01 2024-02)")

    parser.add_argument("--report", "-r", help="Generate report (PDF or HTML)")
    parser.add_argument("--plot-categories", help="Save category pie chart")
    parser.add_argument("--plot-trends", help="Save trends line chart")
    parser.add_argument("--plot-budget", help="Save budget comparison chart")

    parser.add_argument("--output", "-o", help="Output CSV with categories")

    args = parser.parse_args()

    analyzer = BudgetAnalyzer()

    # Load data
    analyzer.load_csv(
        args.input,
        date_col=args.date,
        amount_col=args.amount,
        description_col=args.description,
        category_col=args.category_col
    )

    # Load custom categories
    if args.categories:
        with open(args.categories) as f:
            analyzer.set_categories(json.load(f))

    # Auto-categorize if no category column
    if not args.category_col:
        analyzer.auto_categorize()

    # Load budget
    if args.budget:
        with open(args.budget) as f:
            analyzer.set_budget(json.load(f))

    # Period comparison
    if args.compare:
        comparison = analyzer.compare_periods(args.compare[0], args.compare[1])
        print(f"\n=== Period Comparison: {args.compare[0]} vs {args.compare[1]} ===")
        print(f"Period 1 Total: ${comparison['period1_total']:,.2f}")
        print(f"Period 2 Total: ${comparison['period2_total']:,.2f}")
        print(f"Difference: ${comparison['difference']:,.2f} ({comparison['percent_change']:+.1f}%)")
    else:
        # Show summary
        summary = analyzer.analyze()
        print("\n=== Budget Analysis Summary ===")
        print(f"Total Spent: ${summary['total_spent']:,.2f}")
        print(f"Transactions: {summary['transaction_count']}")
        print(f"Average: ${summary['average_transaction']:,.2f}")
        print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")

        print("\n=== Top Categories ===")
        categories = analyzer.by_category()
        for _, row in categories.head(5).iterrows():
            print(f"  {row['category']}: ${row['amount']:,.2f} ({row['percentage']:.1f}%)")

        if analyzer.budget:
            print("\n=== Budget Status ===")
            budget_comp = analyzer.budget_vs_actual()
            for _, row in budget_comp.iterrows():
                status_icon = "✓" if row['status'] == 'under' else "⚠" if row['status'] == 'warning' else "✗"
                print(f"  {status_icon} {row['category']}: ${row['actual']:,.2f} / ${row['budget']:,.2f}")

        print("\n=== Recommendations ===")
        for rec in analyzer.get_recommendations():
            print(f"  • {rec}")

        score = analyzer.spending_score()
        print(f"\n=== Spending Score: {score['overall_score']}/100 (Grade: {score['grade']}) ===")
        print(f"  {score['summary']}")

    # Generate visualizations
    if args.plot_categories:
        analyzer.plot_categories(args.plot_categories)
        print(f"\nSaved category chart: {args.plot_categories}")

    if args.plot_trends:
        analyzer.plot_trends(args.plot_trends)
        print(f"Saved trends chart: {args.plot_trends}")

    if args.plot_budget and analyzer.budget:
        analyzer.plot_budget_comparison(args.plot_budget)
        print(f"Saved budget chart: {args.plot_budget}")

    # Generate report
    if args.report:
        fmt = "html" if args.report.endswith(".html") else "pdf"
        analyzer.generate_report(args.report, format=fmt)
        print(f"\nGenerated report: {args.report}")

    # Export
    if args.output:
        analyzer.to_csv(args.output)
        print(f"Exported to: {args.output}")


if __name__ == "__main__":
    main()
