#!/usr/bin/env python3
"""
Correlation Explorer - Find and visualize correlations in datasets.

Features:
- Correlation matrix computation
- Multiple methods (Pearson, Spearman, Kendall)
- Heatmap visualization
- P-value significance testing
- Strong/weak correlation discovery
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class CorrelationExplorer:
    """Analyze correlations between variables in datasets."""

    def __init__(self):
        """Initialize explorer."""
        self._pd = None
        self._np = None
        self._load_dependencies()
        self.data = None
        self._corr_matrix = None

    def _load_dependencies(self):
        """Load required libraries."""
        try:
            import pandas as pd
            import numpy as np
            self._pd = pd
            self._np = np
        except ImportError:
            raise ImportError("pandas and numpy required. Install with: pip install pandas numpy")

    def load_csv(self, filepath: str, **kwargs) -> 'CorrelationExplorer':
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file
            **kwargs: Additional pandas read_csv args

        Returns:
            self for chaining
        """
        self.data = self._pd.read_csv(filepath, **kwargs)
        # Select only numeric columns
        self.data = self.data.select_dtypes(include=[self._np.number])
        self._corr_matrix = None
        return self

    def load_dataframe(self, df) -> 'CorrelationExplorer':
        """
        Load data from DataFrame.

        Args:
            df: pandas DataFrame

        Returns:
            self for chaining
        """
        self.data = df.select_dtypes(include=[self._np.number])
        self._corr_matrix = None
        return self

    def correlation_matrix(self, method: str = "pearson"):
        """
        Compute correlation matrix.

        Args:
            method: Correlation method (pearson, spearman, kendall)

        Returns:
            Correlation DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_csv() or load_dataframe() first.")

        self._corr_matrix = self.data.corr(method=method)
        return self._corr_matrix

    def correlation_with_pvalues(self, method: str = "pearson") -> Tuple:
        """
        Compute correlation matrix with p-values.

        Args:
            method: Correlation method

        Returns:
            (correlation_matrix, pvalue_matrix)
        """
        from scipy import stats

        if self.data is None:
            raise ValueError("No data loaded.")

        n = len(self.data.columns)
        corr_matrix = self._pd.DataFrame(
            self._np.zeros((n, n)),
            columns=self.data.columns,
            index=self.data.columns
        )
        pval_matrix = corr_matrix.copy()

        for i, col1 in enumerate(self.data.columns):
            for j, col2 in enumerate(self.data.columns):
                # Remove NaN pairs
                mask = ~(self.data[col1].isna() | self.data[col2].isna())
                x = self.data[col1][mask]
                y = self.data[col2][mask]

                if method == 'pearson':
                    corr, pval = stats.pearsonr(x, y)
                elif method == 'spearman':
                    corr, pval = stats.spearmanr(x, y)
                elif method == 'kendall':
                    corr, pval = stats.kendalltau(x, y)
                else:
                    raise ValueError(f"Unknown method: {method}")

                corr_matrix.iloc[i, j] = corr
                pval_matrix.iloc[i, j] = pval

        self._corr_matrix = corr_matrix
        return corr_matrix, pval_matrix

    def correlate_with_target(self, target: str, method: str = "pearson"):
        """
        Get correlations of all variables with a target variable.

        Args:
            target: Target column name
            method: Correlation method

        Returns:
            Series of correlations sorted by absolute value
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        if target not in self.data.columns:
            raise ValueError(f"Target '{target}' not found in data")

        corr_matrix = self.correlation_matrix(method)
        target_corr = corr_matrix[target].drop(target)
        return target_corr.sort_values(key=abs, ascending=False)

    def find_strong_correlations(
        self,
        threshold: float = 0.7,
        method: str = "pearson"
    ) -> List[Dict]:
        """
        Find pairs with strong correlations.

        Args:
            threshold: Minimum absolute correlation
            method: Correlation method

        Returns:
            List of correlation pairs above threshold
        """
        corr_matrix = self.correlation_matrix(method) if self._corr_matrix is None else self._corr_matrix
        strong = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    strong.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': round(corr, 4),
                        'abs_corr': round(abs(corr), 4)
                    })

        return sorted(strong, key=lambda x: x['abs_corr'], reverse=True)

    def find_weak_correlations(
        self,
        threshold: float = 0.3,
        method: str = "pearson"
    ) -> List[Dict]:
        """
        Find pairs with weak correlations.

        Args:
            threshold: Maximum absolute correlation
            method: Correlation method

        Returns:
            List of correlation pairs below threshold
        """
        corr_matrix = self.correlation_matrix(method) if self._corr_matrix is None else self._corr_matrix
        weak = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) <= threshold:
                    weak.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': round(corr, 4),
                        'abs_corr': round(abs(corr), 4)
                    })

        return sorted(weak, key=lambda x: x['abs_corr'])

    def plot_heatmap(
        self,
        output: str,
        method: str = "pearson",
        cmap: str = "coolwarm",
        annot: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        title: str = "Correlation Matrix",
        vmin: float = -1,
        vmax: float = 1
    ) -> str:
        """
        Generate correlation heatmap.

        Args:
            output: Output file path
            method: Correlation method
            cmap: Color map
            annot: Show correlation values
            figsize: Figure size
            title: Plot title
            vmin, vmax: Color scale limits

        Returns:
            Path to saved image
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("matplotlib and seaborn required for plotting")

        corr_matrix = self.correlation_matrix(method) if self._corr_matrix is None else self._corr_matrix

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=annot,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0,
            square=True,
            linewidths=0.5,
            fmt='.2f',
            ax=ax
        )
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(output, dpi=150)
        plt.close()

        return output

    def plot_scatter(
        self,
        var1: str,
        var2: str,
        output: str,
        figsize: Tuple[int, int] = (8, 6)
    ) -> str:
        """
        Create scatter plot of two variables.

        Args:
            var1: First variable
            var2: Second variable
            output: Output file path
            figsize: Figure size

        Returns:
            Path to saved image
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("matplotlib and seaborn required for plotting")

        if self.data is None:
            raise ValueError("No data loaded.")

        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(data=self.data, x=var1, y=var2, ax=ax)

        # Add trend line
        from scipy import stats
        mask = ~(self.data[var1].isna() | self.data[var2].isna())
        x = self.data[var1][mask]
        y = self.data[var2][mask]
        slope, intercept, r, p, se = stats.linregress(x, y)
        ax.plot(x, slope * x + intercept, 'r--', label=f'r={r:.3f}')
        ax.legend()

        ax.set_title(f'{var1} vs {var2}')
        plt.tight_layout()
        plt.savefig(output, dpi=150)
        plt.close()

        return output

    def to_csv(self, output: str, method: str = "pearson") -> str:
        """Save correlation matrix to CSV."""
        corr_matrix = self.correlation_matrix(method) if self._corr_matrix is None else self._corr_matrix
        corr_matrix.to_csv(output)
        return output

    def to_json(self, output: str, method: str = "pearson") -> str:
        """Save correlation matrix to JSON."""
        corr_matrix = self.correlation_matrix(method) if self._corr_matrix is None else self._corr_matrix
        corr_matrix.to_json(output, orient='index', indent=2)
        return output


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Explore correlations in data')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', help='Output file for correlation matrix')
    parser.add_argument('--heatmap', help='Generate heatmap image')
    parser.add_argument('--method', '-m', default='pearson',
                        choices=['pearson', 'spearman', 'kendall'])
    parser.add_argument('--target', '-t', help='Show correlations with target variable')
    parser.add_argument('--strong', action='store_true', help='Find strong correlations')
    parser.add_argument('--weak', action='store_true', help='Find weak correlations')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Threshold for strong/weak (default: 0.7)')
    parser.add_argument('--pvalues', action='store_true', help='Include p-values')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()
    explorer = CorrelationExplorer()
    explorer.load_csv(args.input)

    if args.heatmap:
        explorer.plot_heatmap(args.heatmap, method=args.method)
        print(f"Heatmap saved to: {args.heatmap}")

    elif args.target:
        corr = explorer.correlate_with_target(args.target, method=args.method)
        print(f"\nCorrelations with '{args.target}':")
        print("-" * 40)
        for var, value in corr.items():
            print(f"  {var}: {value:.4f}")

    elif args.strong:
        strong = explorer.find_strong_correlations(args.threshold, method=args.method)
        print(f"\nStrong correlations (|r| >= {args.threshold}):")
        print("-" * 50)
        if strong:
            for pair in strong:
                print(f"  {pair['var1']} <-> {pair['var2']}: {pair['correlation']:.4f}")
        else:
            print("  No strong correlations found")

    elif args.weak:
        weak = explorer.find_weak_correlations(args.threshold, method=args.method)
        print(f"\nWeak correlations (|r| <= {args.threshold}):")
        print("-" * 50)
        for pair in weak[:20]:  # Show first 20
            print(f"  {pair['var1']} <-> {pair['var2']}: {pair['correlation']:.4f}")

    elif args.pvalues:
        corr, pvals = explorer.correlation_with_pvalues(method=args.method)
        print("\nCorrelation Matrix:")
        print(corr.round(4))
        print("\nP-Values:")
        print(pvals.round(4))

    else:
        corr_matrix = explorer.correlation_matrix(method=args.method)

        if args.output:
            if args.json:
                explorer.to_json(args.output)
            else:
                explorer.to_csv(args.output)
            print(f"Correlation matrix saved to: {args.output}")
        else:
            print("\nCorrelation Matrix:")
            print(corr_matrix.round(4))


if __name__ == "__main__":
    main()
