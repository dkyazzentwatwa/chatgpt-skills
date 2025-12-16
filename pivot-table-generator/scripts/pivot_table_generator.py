#!/usr/bin/env python3
"""
Pivot Table Generator - Create pivot tables with charts.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


class PivotTableGenerator:
    """Generate pivot tables."""

    def __init__(self):
        """Initialize generator."""
        self.df = None
        self.pivot = None

    def load_data(self, df: pd.DataFrame) -> 'PivotTableGenerator':
        """Load data."""
        self.df = df
        return self

    def generate(self, rows, columns, values, aggfunc='sum') -> pd.DataFrame:
        """Generate pivot table."""
        self.pivot = pd.pivot_table(
            self.df,
            values=values,
            index=rows,
            columns=columns,
            aggfunc=aggfunc,
            fill_value=0
        )
        return self.pivot

    def plot(self, output: str, kind: str = 'bar') -> str:
        """Plot pivot table."""
        if self.pivot is None:
            raise ValueError("Generate pivot table first")

        self.pivot.plot(kind=kind, figsize=(10, 6))
        plt.title('Pivot Table Visualization')
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()

        return output

    def export(self, output: str, format: str = 'csv') -> str:
        """Export pivot table."""
        if self.pivot is None:
            raise ValueError("Generate pivot table first")

        if format == 'csv':
            self.pivot.to_csv(output)
        elif format == 'excel':
            self.pivot.to_excel(output)
        elif format == 'html':
            self.pivot.to_html(output)

        return output


def main():
    parser = argparse.ArgumentParser(description="Pivot Table Generator")

    parser.add_argument("--data", required=True, help="Input CSV")
    parser.add_argument("--rows", required=True, help="Row field")
    parser.add_argument("--columns", required=True, help="Column field")
    parser.add_argument("--values", required=True, help="Values field")
    parser.add_argument("--agg", default='sum',
                       choices=['sum', 'mean', 'count', 'min', 'max'],
                       help="Aggregation function")
    parser.add_argument("--output", "-o", required=True, help="Output file")

    args = parser.parse_args()

    df = pd.read_csv(args.data)

    gen = PivotTableGenerator()
    gen.load_data(df)

    pivot = gen.generate(args.rows, args.columns, args.values, aggfunc=args.agg)

    print(pivot)

    gen.export(args.output)
    print(f"\nPivot table saved: {args.output}")


if __name__ == "__main__":
    main()
