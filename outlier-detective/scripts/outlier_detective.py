#!/usr/bin/env python3
"""
Outlier Detective - Detect anomalies in datasets.

Features:
- Statistical methods (IQR, Z-score, Modified Z-score)
- ML methods (Isolation Forest, LOF)
- Visualization (box plots, distributions)
- Multi-column analysis
"""

import argparse
from typing import Dict, List, Optional, Tuple, Union


class OutlierDetective:
    """Detect outliers and anomalies in data."""

    def __init__(self):
        """Initialize detector."""
        self._pd = None
        self._np = None
        self._load_dependencies()
        self.data = None

    def _load_dependencies(self):
        """Load required libraries."""
        try:
            import pandas as pd
            import numpy as np
            self._pd = pd
            self._np = np
        except ImportError:
            raise ImportError("pandas and numpy required. Install with: pip install pandas numpy")

    def load_csv(self, filepath: str, **kwargs) -> 'OutlierDetective':
        """
        Load data from CSV.

        Args:
            filepath: Path to CSV file
            **kwargs: pandas read_csv arguments

        Returns:
            self for chaining
        """
        self.data = self._pd.read_csv(filepath, **kwargs)
        return self

    def load_dataframe(self, df) -> 'OutlierDetective':
        """
        Load data from DataFrame.

        Args:
            df: pandas DataFrame

        Returns:
            self for chaining
        """
        self.data = df.copy()
        return self

    def detect(
        self,
        column: str,
        method: str = "iqr",
        **kwargs
    ):
        """
        Detect outliers in a column.

        Args:
            column: Column name
            method: Detection method (iqr, zscore, modified_zscore)
            **kwargs: Method-specific parameters

        Returns:
            DataFrame of outlier rows
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found")

        values = self.data[column].dropna()

        if method == "iqr":
            mask, scores, reasons = self._detect_iqr(values, **kwargs)
        elif method == "zscore":
            mask, scores, reasons = self._detect_zscore(values, **kwargs)
        elif method == "modified_zscore":
            mask, scores, reasons = self._detect_modified_zscore(values, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Get outlier rows
        outlier_indices = values[mask].index
        result = self.data.loc[outlier_indices].copy()
        result['outlier_score'] = scores[mask].values
        result['outlier_reason'] = [reasons[i] for i in range(len(mask)) if mask.iloc[i]]

        return result

    def _detect_iqr(self, values, multiplier: float = 1.5):
        """Detect outliers using IQR method."""
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        mask = (values < lower) | (values > upper)

        # Calculate scores (distance from bounds)
        scores = self._pd.Series(0.0, index=values.index)
        scores[values < lower] = (values[values < lower] - lower) / iqr
        scores[values > upper] = (values[values > upper] - upper) / iqr

        # Generate reasons
        reasons = []
        for idx in values.index:
            if values[idx] < lower:
                reasons.append(f"Below Q1 - {multiplier}×IQR ({lower:.2f})")
            elif values[idx] > upper:
                reasons.append(f"Above Q3 + {multiplier}×IQR ({upper:.2f})")
            else:
                reasons.append("")

        return mask, scores, reasons

    def _detect_zscore(self, values, threshold: float = 3.0):
        """Detect outliers using Z-score method."""
        mean = values.mean()
        std = values.std()

        if std == 0:
            return self._pd.Series(False, index=values.index), \
                   self._pd.Series(0.0, index=values.index), \
                   ["" for _ in values]

        scores = (values - mean) / std
        mask = abs(scores) > threshold

        reasons = []
        for idx, score in scores.items():
            if score > threshold:
                reasons.append(f"Z-score {score:.2f} > {threshold}")
            elif score < -threshold:
                reasons.append(f"Z-score {score:.2f} < -{threshold}")
            else:
                reasons.append("")

        return mask, scores, reasons

    def _detect_modified_zscore(self, values, threshold: float = 3.5):
        """Detect outliers using Modified Z-score (MAD-based)."""
        median = values.median()
        mad = (values - median).abs().median()

        if mad == 0:
            return self._pd.Series(False, index=values.index), \
                   self._pd.Series(0.0, index=values.index), \
                   ["" for _ in values]

        # Modified Z-score formula
        scores = 0.6745 * (values - median) / mad
        mask = abs(scores) > threshold

        reasons = []
        for idx, score in scores.items():
            if abs(score) > threshold:
                reasons.append(f"Modified Z-score {score:.2f}, threshold ±{threshold}")
            else:
                reasons.append("")

        return mask, scores, reasons

    def detect_multivariate(
        self,
        columns: Optional[List[str]] = None,
        method: str = "isolation_forest",
        **kwargs
    ):
        """
        Detect multivariate outliers.

        Args:
            columns: Columns to analyze (default: all numeric)
            method: Detection method (isolation_forest, lof)
            **kwargs: Method parameters

        Returns:
            DataFrame of outlier rows
        """
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

        if self.data is None:
            raise ValueError("No data loaded.")

        # Select columns
        if columns is None:
            columns = self.data.select_dtypes(include=[self._np.number]).columns.tolist()

        X = self.data[columns].dropna()

        if method == "isolation_forest":
            contamination = kwargs.get('contamination', 0.1)
            model = IsolationForest(contamination=contamination, random_state=42)
            predictions = model.fit_predict(X)
            scores = model.score_samples(X)
        elif method == "lof":
            n_neighbors = kwargs.get('n_neighbors', 20)
            model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=kwargs.get('contamination', 0.1))
            predictions = model.fit_predict(X)
            scores = model.negative_outlier_factor_
        else:
            raise ValueError(f"Unknown method: {method}")

        # -1 indicates outlier
        outlier_mask = predictions == -1
        result = self.data.loc[X[outlier_mask].index].copy()
        result['outlier_score'] = scores[outlier_mask]

        return result

    def analyze(self, column: str) -> Dict:
        """
        Comprehensive outlier analysis for a column.

        Args:
            column: Column name

        Returns:
            Analysis report dict
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        values = self.data[column].dropna()

        # Statistics
        stats = {
            'mean': values.mean(),
            'median': values.median(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'q1': values.quantile(0.25),
            'q3': values.quantile(0.75),
        }
        stats['iqr'] = stats['q3'] - stats['q1']

        # IQR bounds
        bounds = {
            'lower': stats['q1'] - 1.5 * stats['iqr'],
            'upper': stats['q3'] + 1.5 * stats['iqr']
        }

        # Run each method
        methods = {}
        for method in ['iqr', 'zscore', 'modified_zscore']:
            outliers = self.detect(column, method=method)
            methods[method] = {
                'count': len(outliers),
                'indices': outliers.index.tolist()
            }

        # Primary count (IQR)
        outlier_count = methods['iqr']['count']

        return {
            'column': column,
            'total_rows': len(values),
            'outlier_count': outlier_count,
            'outlier_percent': round(100 * outlier_count / len(values), 2),
            'methods': methods,
            'stats': {k: round(v, 4) for k, v in stats.items()},
            'bounds': {k: round(v, 4) for k, v in bounds.items()}
        }

    def analyze_all(self) -> Dict:
        """
        Analyze all numeric columns.

        Returns:
            Dict of column analyses
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        numeric_cols = self.data.select_dtypes(include=[self._np.number]).columns
        results = {}

        for col in numeric_cols:
            results[col] = self.analyze(col)

        return results

    def get_outliers(self, column: str, method: str = "iqr", **kwargs):
        """Get outlier rows."""
        return self.detect(column, method, **kwargs)

    def get_clean_data(self, column: str, method: str = "iqr", **kwargs):
        """Get data with outliers removed."""
        outliers = self.detect(column, method, **kwargs)
        return self.data.drop(outliers.index)

    def plot_boxplot(
        self,
        column: str,
        output: str,
        figsize: Tuple[int, int] = (8, 6)
    ) -> str:
        """Generate box plot visualization."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot(self.data[column].dropna())
        ax.set_ylabel(column)
        ax.set_title(f'Box Plot: {column}')
        plt.tight_layout()
        plt.savefig(output, dpi=150)
        plt.close()

        return output

    def plot_distribution(
        self,
        column: str,
        output: str,
        figsize: Tuple[int, int] = (10, 6)
    ) -> str:
        """Generate distribution plot with outlier bounds."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        values = self.data[column].dropna()
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(lower, color='r', linestyle='--', label=f'Lower bound ({lower:.2f})')
        ax.axvline(upper, color='r', linestyle='--', label=f'Upper bound ({upper:.2f})')
        ax.axvline(values.median(), color='g', linestyle='-', label=f'Median ({values.median():.2f})')
        ax.legend()
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution: {column}')
        plt.tight_layout()
        plt.savefig(output, dpi=150)
        plt.close()

        return output

    def plot_scatter(
        self,
        col1: str,
        col2: str,
        output: str,
        figsize: Tuple[int, int] = (8, 8)
    ) -> str:
        """Generate scatter plot with outliers highlighted."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        # Detect outliers in both columns
        outliers1 = set(self.detect(col1, method='iqr').index)
        outliers2 = set(self.detect(col2, method='iqr').index)
        all_outliers = outliers1 | outliers2

        fig, ax = plt.subplots(figsize=figsize)

        # Normal points
        normal_mask = ~self.data.index.isin(all_outliers)
        ax.scatter(self.data.loc[normal_mask, col1],
                   self.data.loc[normal_mask, col2],
                   alpha=0.6, label='Normal')

        # Outliers
        if all_outliers:
            ax.scatter(self.data.loc[list(all_outliers), col1],
                       self.data.loc[list(all_outliers), col2],
                       color='red', alpha=0.8, label='Outliers')

        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f'{col1} vs {col2}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output, dpi=150)
        plt.close()

        return output


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Detect outliers in data')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--column', '-c', help='Column to analyze')
    parser.add_argument('--method', '-m', default='iqr',
                        choices=['iqr', 'zscore', 'modified_zscore', 'isolation_forest', 'lof'])
    parser.add_argument('--threshold', '-t', type=float, help='Detection threshold')
    parser.add_argument('--all', '-a', action='store_true', help='Analyze all columns')
    parser.add_argument('--output', '-o', help='Output file for outliers')
    parser.add_argument('--plot', '-p', help='Generate plot')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()
    detective = OutlierDetective()
    detective.load_csv(args.input)

    if args.all:
        results = detective.analyze_all()
        print("Outlier Analysis Summary")
        print("=" * 50)
        for col, report in results.items():
            print(f"\n{col}:")
            print(f"  Outliers: {report['outlier_count']} ({report['outlier_percent']:.1f}%)")
            print(f"  Bounds: [{report['bounds']['lower']:.2f}, {report['bounds']['upper']:.2f}]")

    elif args.column:
        # Method-specific kwargs
        kwargs = {}
        if args.threshold:
            if args.method in ['zscore', 'modified_zscore']:
                kwargs['threshold'] = args.threshold
            elif args.method == 'iqr':
                kwargs['multiplier'] = args.threshold

        if args.method in ['isolation_forest', 'lof']:
            outliers = detective.detect_multivariate(
                columns=[args.column],
                method=args.method
            )
        else:
            outliers = detective.detect(args.column, method=args.method, **kwargs)

        if args.output:
            outliers.to_csv(args.output, index=False)
            print(f"Outliers saved to: {args.output}")
        elif args.plot:
            if args.plot.endswith('.png'):
                detective.plot_boxplot(args.column, args.plot)
                print(f"Plot saved to: {args.plot}")
        else:
            # Show analysis
            report = detective.analyze(args.column)
            print(f"Column: {report['column']}")
            print(f"Total rows: {report['total_rows']}")
            print(f"Outliers: {report['outlier_count']} ({report['outlier_percent']:.1f}%)")
            print(f"\nStatistics:")
            for k, v in report['stats'].items():
                print(f"  {k}: {v}")
            print(f"\nBounds (IQR method):")
            print(f"  Lower: {report['bounds']['lower']}")
            print(f"  Upper: {report['bounds']['upper']}")

            if len(outliers) > 0:
                print(f"\nOutlier indices: {outliers.index.tolist()[:20]}{'...' if len(outliers) > 20 else ''}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
