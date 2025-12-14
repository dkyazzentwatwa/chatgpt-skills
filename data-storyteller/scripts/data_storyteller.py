#!/usr/bin/env python3
"""
Data Storyteller - Transform data into narrative insights
Automatically analyze datasets and generate comprehensive reports with visualizations.
"""

import io
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')


@dataclass
class StorytellerConfig:
    """Configuration for data analysis."""
    max_categories: int = 20
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation'
    outlier_threshold: float = 1.5  # For IQR method
    zscore_threshold: float = 3.0  # For Z-score method
    correlation_threshold: float = 0.5
    significance_level: float = 0.05
    date_format: str = 'auto'
    language: str = 'en'
    chart_style: str = 'business'
    figure_dpi: int = 150
    max_unique_for_categorical: int = 50
    sample_size_for_large_data: int = 100000

    def update(self, settings: Dict[str, Any]) -> None:
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)


# Chart style configurations
CHART_STYLES = {
    'business': {
        'palette': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B'],
        'background': '#FFFFFF',
        'grid_alpha': 0.3,
        'font_family': 'sans-serif',
    },
    'scientific': {
        'palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'background': '#FFFFFF',
        'grid_alpha': 0.5,
        'font_family': 'serif',
    },
    'minimal': {
        'palette': ['#333333', '#666666', '#999999', '#CCCCCC', '#E5E5E5'],
        'background': '#FFFFFF',
        'grid_alpha': 0.2,
        'font_family': 'sans-serif',
    },
    'dark': {
        'palette': ['#00D4FF', '#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3'],
        'background': '#1E1E1E',
        'grid_alpha': 0.2,
        'font_family': 'sans-serif',
    },
    'colorful': {
        'palette': ['#E63946', '#F4A261', '#2A9D8F', '#264653', '#E9C46A'],
        'background': '#FFFFFF',
        'grid_alpha': 0.3,
        'font_family': 'sans-serif',
    },
}


class DataStoryteller:
    """
    Main class for data analysis and report generation.

    Automatically analyzes datasets and generates comprehensive reports
    with visualizations, statistics, and narrative insights.
    """

    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        name: Optional[str] = None
    ):
        """
        Initialize DataStoryteller with data.

        Args:
            data: Path to CSV/Excel file or pandas DataFrame
            name: Optional name for the dataset
        """
        self.config = StorytellerConfig()
        self.name = name
        self._charts: List[Dict[str, Any]] = []
        self._insights: List[str] = []
        self._warnings: List[str] = []

        # Load data
        if isinstance(data, (str, Path)):
            self.df = self._load_file(Path(data))
            self.name = name or Path(data).stem
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
            self.name = name or "Dataset"
        else:
            raise ValueError("Data must be a file path or pandas DataFrame")

        # Initialize analysis results
        self._profile: Optional[Dict] = None
        self._column_analysis: Optional[Dict] = None
        self._correlations: Optional[pd.DataFrame] = None
        self._time_analysis: Optional[Dict] = None

    def _load_file(self, path: Path) -> pd.DataFrame:
        """Load data from file based on extension."""
        suffix = path.suffix.lower()

        if suffix == '.csv':
            # Try to auto-detect delimiter
            with open(path, 'r') as f:
                sample = f.read(1024)
            delimiter = ',' if sample.count(',') > sample.count('\t') else '\t'
            return pd.read_csv(path, delimiter=delimiter)

        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(path)

        elif suffix == '.json':
            return pd.read_json(path)

        elif suffix == '.parquet':
            return pd.read_parquet(path)

        elif suffix == '.tsv':
            return pd.read_csv(path, delimiter='\t')

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _detect_column_type(self, series: pd.Series) -> str:
        """Detect the semantic type of a column."""
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'

        # Try parsing as datetime
        if series.dtype == 'object':
            try:
                pd.to_datetime(series.dropna().head(100))
                return 'datetime'
            except (ValueError, TypeError):
                pass

        # Check numeric
        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.05 and series.nunique() < self.config.max_unique_for_categorical:
                return 'categorical'
            return 'numeric'

        # Check boolean
        if pd.api.types.is_bool_dtype(series):
            return 'boolean'

        # Check categorical vs text
        unique_count = series.nunique()
        if unique_count <= self.config.max_unique_for_categorical:
            return 'categorical'

        # Check if it might be text (long strings)
        avg_len = series.dropna().astype(str).str.len().mean()
        if avg_len > 50:
            return 'text'

        return 'categorical'

    def _profile_data(self) -> Dict[str, Any]:
        """Generate data profile summary."""
        if self._profile is not None:
            return self._profile

        df = self.df
        profile = {
            'name': self.name,
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'duplicates': df.duplicated().sum(),
            'duplicate_pct': (df.duplicated().sum() / len(df)) * 100,
            'column_types': {},
            'missing_summary': {},
            'generated_at': datetime.now().isoformat(),
        }

        # Analyze each column
        for col in df.columns:
            col_type = self._detect_column_type(df[col])
            profile['column_types'][col] = col_type

            missing = df[col].isna().sum()
            profile['missing_summary'][col] = {
                'count': int(missing),
                'percentage': round((missing / len(df)) * 100, 2)
            }

        # Type distribution
        type_counts = {}
        for t in profile['column_types'].values():
            type_counts[t] = type_counts.get(t, 0) + 1
        profile['type_distribution'] = type_counts

        self._profile = profile
        return profile

    def _analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a numeric column."""
        clean = series.dropna()

        if len(clean) == 0:
            return {'error': 'No valid data'}

        analysis = {
            'count': len(clean),
            'missing': series.isna().sum(),
            'mean': float(clean.mean()),
            'median': float(clean.median()),
            'std': float(clean.std()),
            'min': float(clean.min()),
            'max': float(clean.max()),
            'range': float(clean.max() - clean.min()),
            'q1': float(clean.quantile(0.25)),
            'q3': float(clean.quantile(0.75)),
            'iqr': float(clean.quantile(0.75) - clean.quantile(0.25)),
            'skewness': float(clean.skew()),
            'kurtosis': float(clean.kurtosis()),
        }

        # Detect outliers
        if self.config.outlier_method == 'iqr':
            q1, q3 = analysis['q1'], analysis['q3']
            iqr = analysis['iqr']
            lower = q1 - self.config.outlier_threshold * iqr
            upper = q3 + self.config.outlier_threshold * iqr
            outliers = clean[(clean < lower) | (clean > upper)]
        else:  # zscore
            z_scores = np.abs(stats.zscore(clean))
            outliers = clean[z_scores > self.config.zscore_threshold]

        analysis['outlier_count'] = len(outliers)
        analysis['outlier_pct'] = round((len(outliers) / len(clean)) * 100, 2)

        # Distribution assessment
        if abs(analysis['skewness']) < 0.5:
            analysis['distribution'] = 'approximately normal'
        elif analysis['skewness'] > 0.5:
            analysis['distribution'] = 'right-skewed'
        else:
            analysis['distribution'] = 'left-skewed'

        return analysis

    def _analyze_categorical_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a categorical column."""
        clean = series.dropna()

        if len(clean) == 0:
            return {'error': 'No valid data'}

        value_counts = clean.value_counts()

        analysis = {
            'count': len(clean),
            'missing': series.isna().sum(),
            'unique_count': clean.nunique(),
            'top_values': value_counts.head(10).to_dict(),
            'bottom_values': value_counts.tail(5).to_dict() if len(value_counts) > 10 else {},
            'most_common': str(value_counts.index[0]),
            'most_common_count': int(value_counts.iloc[0]),
            'most_common_pct': round((value_counts.iloc[0] / len(clean)) * 100, 2),
        }

        # Category balance assessment
        if len(value_counts) > 1:
            top_pct = value_counts.iloc[0] / len(clean)
            if top_pct > 0.9:
                analysis['balance'] = 'highly imbalanced'
            elif top_pct > 0.7:
                analysis['balance'] = 'moderately imbalanced'
            else:
                analysis['balance'] = 'balanced'
        else:
            analysis['balance'] = 'single value'

        return analysis

    def _analyze_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a datetime column."""
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(series):
            try:
                series = pd.to_datetime(series)
            except (ValueError, TypeError):
                return {'error': 'Could not parse as datetime'}

        clean = series.dropna()

        if len(clean) == 0:
            return {'error': 'No valid data'}

        analysis = {
            'count': len(clean),
            'missing': series.isna().sum(),
            'min_date': str(clean.min()),
            'max_date': str(clean.max()),
            'range_days': (clean.max() - clean.min()).days,
            'unique_count': clean.nunique(),
        }

        # Check for gaps
        sorted_dates = clean.sort_values()
        if len(sorted_dates) > 1:
            diffs = sorted_dates.diff().dropna()
            analysis['median_gap_days'] = diffs.median().days
            analysis['max_gap_days'] = diffs.max().days

        return analysis

    def _analyze_columns(self) -> Dict[str, Dict]:
        """Analyze all columns."""
        if self._column_analysis is not None:
            return self._column_analysis

        profile = self._profile_data()
        analysis = {}

        for col, col_type in profile['column_types'].items():
            if col_type == 'numeric':
                analysis[col] = self._analyze_numeric_column(self.df[col])
            elif col_type == 'categorical':
                analysis[col] = self._analyze_categorical_column(self.df[col])
            elif col_type == 'datetime':
                analysis[col] = self._analyze_datetime_column(self.df[col])
            elif col_type == 'boolean':
                analysis[col] = self._analyze_categorical_column(self.df[col].astype(str))
            else:
                analysis[col] = {'type': 'text', 'count': self.df[col].count()}

            analysis[col]['type'] = col_type

        self._column_analysis = analysis
        return analysis

    def _compute_correlations(self) -> pd.DataFrame:
        """Compute correlation matrix for numeric columns."""
        if self._correlations is not None:
            return self._correlations

        profile = self._profile_data()
        numeric_cols = [
            col for col, t in profile['column_types'].items()
            if t == 'numeric'
        ]

        if len(numeric_cols) < 2:
            self._correlations = pd.DataFrame()
            return self._correlations

        self._correlations = self.df[numeric_cols].corr()
        return self._correlations

    def _find_strong_correlations(self) -> List[Dict]:
        """Find strongly correlated column pairs."""
        corr = self._compute_correlations()

        if corr.empty:
            return []

        strong = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                r = corr.iloc[i, j]
                if abs(r) >= self.config.correlation_threshold:
                    strong.append({
                        'column1': corr.columns[i],
                        'column2': corr.columns[j],
                        'correlation': round(r, 3),
                        'strength': 'strong' if abs(r) >= 0.7 else 'moderate'
                    })

        return sorted(strong, key=lambda x: abs(x['correlation']), reverse=True)

    def _analyze_time_series(self, date_col: str, value_col: str) -> Dict[str, Any]:
        """Analyze time series patterns."""
        df = self.df.copy()

        # Convert date column
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])

        df = df.sort_values(date_col)
        df = df.set_index(date_col)

        series = df[value_col].dropna()

        if len(series) < 10:
            return {'error': 'Insufficient data for time series analysis'}

        analysis = {
            'column': value_col,
            'date_column': date_col,
            'observations': len(series),
        }

        # Trend analysis (simple linear regression)
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)

        analysis['trend'] = {
            'direction': 'increasing' if slope > 0 else 'decreasing',
            'slope': float(slope),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'significant': p_value < self.config.significance_level
        }

        # Growth rate
        first_val = series.iloc[0]
        last_val = series.iloc[-1]
        if first_val != 0:
            analysis['total_change_pct'] = round(((last_val - first_val) / abs(first_val)) * 100, 2)

        # Volatility
        analysis['volatility'] = float(series.std() / series.mean()) if series.mean() != 0 else 0

        return analysis

    def _generate_insights(self) -> List[str]:
        """Generate narrative insights from analysis."""
        insights = []
        profile = self._profile_data()
        col_analysis = self._analyze_columns()

        # Dataset overview insight
        insights.append(
            f"This dataset contains {profile['rows']:,} records across {profile['columns']} columns, "
            f"using approximately {profile['memory_mb']:.1f} MB of memory."
        )

        # Missing data insight
        high_missing = [
            (col, info['percentage'])
            for col, info in profile['missing_summary'].items()
            if info['percentage'] > 5
        ]
        if high_missing:
            cols_str = ', '.join([f"{col} ({pct:.1f}%)" for col, pct in high_missing[:3]])
            insights.append(f"Notable missing data detected in: {cols_str}. Consider imputation or removal.")

        # Duplicate insight
        if profile['duplicate_pct'] > 1:
            insights.append(
                f"Found {profile['duplicates']:,} duplicate rows ({profile['duplicate_pct']:.1f}% of data). "
                "Review for potential data quality issues."
            )

        # Numeric column insights
        for col, analysis in col_analysis.items():
            if analysis.get('type') == 'numeric' and 'error' not in analysis:
                # Outlier insight
                if analysis.get('outlier_pct', 0) > 5:
                    insights.append(
                        f"Column '{col}' has {analysis['outlier_pct']:.1f}% outliers. "
                        f"Range: {analysis['min']:.2f} to {analysis['max']:.2f}."
                    )

                # Distribution insight
                if abs(analysis.get('skewness', 0)) > 1:
                    direction = 'right' if analysis['skewness'] > 0 else 'left'
                    insights.append(
                        f"Column '{col}' shows strong {direction}-skewed distribution "
                        f"(skewness: {analysis['skewness']:.2f}). Consider transformation for modeling."
                    )

        # Correlation insights
        strong_corrs = self._find_strong_correlations()
        if strong_corrs:
            top_corr = strong_corrs[0]
            insights.append(
                f"Strong correlation detected between '{top_corr['column1']}' and '{top_corr['column2']}' "
                f"(r = {top_corr['correlation']:.2f}). This relationship may be worth investigating."
            )

        # Categorical insights
        for col, analysis in col_analysis.items():
            if analysis.get('type') == 'categorical' and 'error' not in analysis:
                if analysis.get('balance') == 'highly imbalanced':
                    insights.append(
                        f"Column '{col}' is highly imbalanced: '{analysis['most_common']}' "
                        f"accounts for {analysis['most_common_pct']:.1f}% of values."
                    )

        self._insights = insights
        return insights

    def _create_distribution_chart(self, col: str) -> Optional[plt.Figure]:
        """Create distribution chart for a numeric column."""
        series = self.df[col].dropna()
        if len(series) == 0:
            return None

        style = CHART_STYLES[self.config.chart_style]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        axes[0].hist(series, bins=30, color=style['palette'][0], edgecolor='white', alpha=0.7)
        axes[0].set_title(f'Distribution of {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=style['grid_alpha'])

        # Box plot
        axes[1].boxplot(series, vert=True)
        axes[1].set_title(f'Box Plot of {col}')
        axes[1].set_ylabel(col)
        axes[1].grid(True, alpha=style['grid_alpha'])

        plt.tight_layout()
        return fig

    def _create_categorical_chart(self, col: str) -> Optional[plt.Figure]:
        """Create bar chart for categorical column."""
        series = self.df[col].dropna()
        if len(series) == 0:
            return None

        value_counts = series.value_counts().head(self.config.max_categories)
        style = CHART_STYLES[self.config.chart_style]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(
            range(len(value_counts)),
            value_counts.values,
            color=style['palette'][0],
            alpha=0.8
        )

        ax.set_yticks(range(len(value_counts)))
        ax.set_yticklabels([str(x)[:30] for x in value_counts.index])
        ax.set_xlabel('Count')
        ax.set_title(f'Distribution of {col}')
        ax.grid(True, axis='x', alpha=style['grid_alpha'])

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, value_counts.values)):
            ax.text(val, i, f' {val:,}', va='center', fontsize=9)

        plt.tight_layout()
        return fig

    def _create_correlation_heatmap(self) -> Optional[plt.Figure]:
        """Create correlation heatmap."""
        corr = self._compute_correlations()
        if corr.empty or len(corr) < 2:
            return None

        style = CHART_STYLES[self.config.chart_style]
        fig, ax = plt.subplots(figsize=(10, 8))

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            ax=ax,
            cbar_kws={'shrink': 0.8}
        )

        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        return fig

    def _create_time_series_chart(self, date_col: str, value_col: str) -> Optional[plt.Figure]:
        """Create time series line chart."""
        df = self.df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except (ValueError, TypeError):
                return None

        df = df.sort_values(date_col)
        style = CHART_STYLES[self.config.chart_style]

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(df[date_col], df[value_col], color=style['palette'][0], linewidth=1.5)
        ax.fill_between(df[date_col], df[value_col], alpha=0.3, color=style['palette'][0])

        ax.set_xlabel('Date')
        ax.set_ylabel(value_col)
        ax.set_title(f'{value_col} Over Time')
        ax.grid(True, alpha=style['grid_alpha'])

        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def generate_report(
        self,
        include_correlations: bool = True,
        include_outliers: bool = True,
        include_trends: bool = True,
        time_column: Optional[str] = None,
        value_columns: Optional[List[str]] = None,
        chart_style: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.

        Args:
            include_correlations: Include correlation analysis
            include_outliers: Include outlier detection
            include_trends: Include trend analysis (if time column available)
            time_column: Column to use for time series analysis
            value_columns: Columns to analyze (default: all)
            chart_style: Chart styling ('business', 'scientific', 'minimal', 'dark', 'colorful')

        Returns:
            Dictionary containing complete analysis report
        """
        if chart_style:
            self.config.chart_style = chart_style

        # Run all analyses
        profile = self._profile_data()
        col_analysis = self._analyze_columns()
        insights = self._generate_insights()

        report = {
            'name': self.name,
            'generated_at': datetime.now().isoformat(),
            'profile': profile,
            'column_analysis': col_analysis,
            'insights': insights,
            'warnings': self._warnings,
        }

        # Correlations
        if include_correlations:
            report['correlations'] = {
                'matrix': self._compute_correlations().to_dict(),
                'strong_pairs': self._find_strong_correlations()
            }

        # Time series analysis
        if include_trends:
            # Auto-detect time column if not specified
            if time_column is None:
                for col, t in profile['column_types'].items():
                    if t == 'datetime':
                        time_column = col
                        break

            if time_column:
                numeric_cols = [
                    col for col, t in profile['column_types'].items()
                    if t == 'numeric'
                ]
                if value_columns:
                    numeric_cols = [c for c in numeric_cols if c in value_columns]

                report['time_series'] = {}
                for col in numeric_cols[:5]:  # Limit to top 5
                    report['time_series'][col] = self._analyze_time_series(time_column, col)

        # Generate executive summary
        report['summary'] = self._generate_executive_summary(report)

        return report

    def _generate_executive_summary(self, report: Dict) -> str:
        """Generate executive summary paragraph."""
        profile = report['profile']

        summary_parts = [
            f"This analysis covers '{self.name}' containing {profile['rows']:,} records "
            f"across {profile['columns']} columns."
        ]

        # Add type distribution
        type_dist = profile['type_distribution']
        types_str = ', '.join([f"{v} {k}" for k, v in type_dist.items()])
        summary_parts.append(f"The dataset includes {types_str} columns.")

        # Add key findings
        if report.get('insights'):
            key_insights = report['insights'][:3]
            if key_insights:
                summary_parts.append("Key findings: " + " ".join(key_insights))

        # Add correlations summary
        if report.get('correlations', {}).get('strong_pairs'):
            n_corr = len(report['correlations']['strong_pairs'])
            summary_parts.append(f"Found {n_corr} notable correlation(s) between variables.")

        # Add time series summary
        if report.get('time_series'):
            trends = []
            for col, ts in report['time_series'].items():
                if 'trend' in ts:
                    direction = ts['trend']['direction']
                    if ts['trend']['significant']:
                        trends.append(f"{col} ({direction})")
            if trends:
                summary_parts.append(f"Significant trends detected in: {', '.join(trends[:3])}.")

        return ' '.join(summary_parts)

    def generate_charts(self) -> List[Tuple[str, plt.Figure]]:
        """Generate all relevant charts."""
        charts = []
        profile = self._profile_data()

        # Distribution charts for numeric columns
        for col, col_type in profile['column_types'].items():
            if col_type == 'numeric':
                fig = self._create_distribution_chart(col)
                if fig:
                    charts.append((f'distribution_{col}', fig))

        # Bar charts for categorical columns (limit to top 5)
        cat_count = 0
        for col, col_type in profile['column_types'].items():
            if col_type == 'categorical' and cat_count < 5:
                fig = self._create_categorical_chart(col)
                if fig:
                    charts.append((f'categorical_{col}', fig))
                    cat_count += 1

        # Correlation heatmap
        fig = self._create_correlation_heatmap()
        if fig:
            charts.append(('correlation_heatmap', fig))

        # Time series charts
        time_col = None
        for col, col_type in profile['column_types'].items():
            if col_type == 'datetime':
                time_col = col
                break

        if time_col:
            for col, col_type in profile['column_types'].items():
                if col_type == 'numeric':
                    fig = self._create_time_series_chart(time_col, col)
                    if fig:
                        charts.append((f'timeseries_{col}', fig))
                    break  # Just one time series chart

        self._charts = charts
        return charts

    def export_charts(self, output_dir: Union[str, Path], format: str = 'png') -> List[str]:
        """Export all charts to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self._charts:
            self.generate_charts()

        exported = []
        for name, fig in self._charts:
            filepath = output_dir / f"{name}.{format}"
            fig.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
            exported.append(str(filepath))
            plt.close(fig)

        return exported

    def export_html(self, filepath: Union[str, Path]) -> str:
        """Export report as HTML file."""
        report = self.generate_report()

        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<meta charset='utf-8'>",
            f"<title>Data Analysis: {self.name}</title>",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; ",
            "       max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.6; }",
            "h1 { color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px; }",
            "h2 { color: #333; margin-top: 30px; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }",
            "th { background-color: #2E86AB; color: white; }",
            "tr:nth-child(even) { background-color: #f9f9f9; }",
            ".insight { background: #f0f7fa; padding: 15px; border-left: 4px solid #2E86AB; margin: 10px 0; }",
            ".stat-box { display: inline-block; background: #f5f5f5; padding: 15px; margin: 10px; border-radius: 5px; }",
            "</style>",
            "</head><body>",
        ]

        # Header
        html_parts.append(f"<h1>Data Analysis Report: {self.name}</h1>")
        html_parts.append(f"<p><em>Generated: {report['generated_at']}</em></p>")

        # Executive Summary
        html_parts.append("<h2>Executive Summary</h2>")
        html_parts.append(f"<p>{report['summary']}</p>")

        # Data Profile
        profile = report['profile']
        html_parts.append("<h2>Data Profile</h2>")
        html_parts.append("<div>")
        html_parts.append(f"<div class='stat-box'><strong>Rows:</strong> {profile['rows']:,}</div>")
        html_parts.append(f"<div class='stat-box'><strong>Columns:</strong> {profile['columns']}</div>")
        html_parts.append(f"<div class='stat-box'><strong>Memory:</strong> {profile['memory_mb']:.1f} MB</div>")
        html_parts.append(f"<div class='stat-box'><strong>Duplicates:</strong> {profile['duplicate_pct']:.1f}%</div>")
        html_parts.append("</div>")

        # Key Insights
        html_parts.append("<h2>Key Insights</h2>")
        for insight in report['insights']:
            html_parts.append(f"<div class='insight'>{insight}</div>")

        # Column Analysis Table
        html_parts.append("<h2>Column Analysis</h2>")
        html_parts.append("<table><tr><th>Column</th><th>Type</th><th>Missing</th><th>Key Stats</th></tr>")

        for col, analysis in report['column_analysis'].items():
            missing_pct = profile['missing_summary'][col]['percentage']
            col_type = analysis.get('type', 'unknown')

            if col_type == 'numeric' and 'error' not in analysis:
                stats_str = f"Mean: {analysis['mean']:.2f}, Std: {analysis['std']:.2f}"
            elif col_type == 'categorical' and 'error' not in analysis:
                stats_str = f"Unique: {analysis['unique_count']}, Top: {analysis['most_common']}"
            elif col_type == 'datetime' and 'error' not in analysis:
                stats_str = f"Range: {analysis['min_date']} to {analysis['max_date']}"
            else:
                stats_str = "N/A"

            html_parts.append(f"<tr><td>{col}</td><td>{col_type}</td><td>{missing_pct:.1f}%</td><td>{stats_str}</td></tr>")

        html_parts.append("</table>")

        # Correlations
        if report.get('correlations', {}).get('strong_pairs'):
            html_parts.append("<h2>Notable Correlations</h2>")
            html_parts.append("<table><tr><th>Column 1</th><th>Column 2</th><th>Correlation</th><th>Strength</th></tr>")
            for corr in report['correlations']['strong_pairs']:
                html_parts.append(
                    f"<tr><td>{corr['column1']}</td><td>{corr['column2']}</td>"
                    f"<td>{corr['correlation']:.3f}</td><td>{corr['strength']}</td></tr>"
                )
            html_parts.append("</table>")

        html_parts.append("</body></html>")

        html_content = '\n'.join(html_parts)

        filepath = Path(filepath)
        filepath.write_text(html_content)
        return str(filepath)

    def export_pdf(self, filepath: Union[str, Path]) -> str:
        """Export report as PDF file."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError:
            raise ImportError("reportlab is required for PDF export. Install with: pip install reportlab")

        report = self.generate_report()
        filepath = Path(filepath)

        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2E86AB')
        )
        story.append(Paragraph(f"Data Analysis Report: {self.name}", title_style))
        story.append(Paragraph(f"<i>Generated: {report['generated_at']}</i>", styles['Normal']))
        story.append(Spacer(1, 20))

        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        story.append(Paragraph(report['summary'], styles['Normal']))
        story.append(Spacer(1, 15))

        # Key Insights
        story.append(Paragraph("Key Insights", styles['Heading2']))
        for insight in report['insights'][:5]:
            story.append(Paragraph(f"• {insight}", styles['Normal']))
            story.append(Spacer(1, 5))
        story.append(Spacer(1, 15))

        # Data Profile Table
        story.append(Paragraph("Data Profile", styles['Heading2']))
        profile = report['profile']
        profile_data = [
            ['Metric', 'Value'],
            ['Total Rows', f"{profile['rows']:,}"],
            ['Total Columns', str(profile['columns'])],
            ['Memory Usage', f"{profile['memory_mb']:.1f} MB"],
            ['Duplicate Rows', f"{profile['duplicates']:,} ({profile['duplicate_pct']:.1f}%)"],
        ]

        profile_table = Table(profile_data, colWidths=[2.5 * inch, 2.5 * inch])
        profile_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DDDDDD')),
        ]))
        story.append(profile_table)
        story.append(Spacer(1, 15))

        # Column Summary Table
        story.append(Paragraph("Column Summary", styles['Heading2']))
        col_data = [['Column', 'Type', 'Missing %', 'Notes']]

        for col, analysis in list(report['column_analysis'].items())[:15]:  # Limit rows
            col_type = analysis.get('type', 'unknown')
            missing_pct = profile['missing_summary'][col]['percentage']

            if col_type == 'numeric' and 'error' not in analysis:
                notes = f"Mean: {analysis['mean']:.1f}"
            elif col_type == 'categorical' and 'error' not in analysis:
                notes = f"Unique: {analysis['unique_count']}"
            else:
                notes = "-"

            col_data.append([col[:20], col_type, f"{missing_pct:.1f}%", notes])

        col_table = Table(col_data, colWidths=[2 * inch, 1.2 * inch, 1 * inch, 2 * inch])
        col_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DDDDDD')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9F9F9')]),
        ]))
        story.append(col_table)

        doc.build(story)
        return str(filepath)

    def analyze_columns(self, columns: List[str]) -> 'DataStoryteller':
        """Filter analysis to specific columns."""
        available = [c for c in columns if c in self.df.columns]
        if available:
            self.df = self.df[available]
            # Reset cached analyses
            self._profile = None
            self._column_analysis = None
            self._correlations = None
        return self

    def __repr__(self) -> str:
        return f"DataStoryteller(name='{self.name}', rows={len(self.df)}, cols={len(self.df.columns)})"


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Data Storyteller - Transform data into insights')
    parser.add_argument('input', help='Input data file (CSV, Excel, JSON)')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-f', '--format', choices=['html', 'pdf', 'json'], default='html',
                        help='Output format (default: html)')
    parser.add_argument('-s', '--style', choices=['business', 'scientific', 'minimal', 'dark', 'colorful'],
                        default='business', help='Chart style')
    parser.add_argument('--charts', help='Directory to export charts')

    args = parser.parse_args()

    storyteller = DataStoryteller(args.input)
    report = storyteller.generate_report(chart_style=args.style)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = input_path.with_suffix(f'.{args.format}')

    # Export
    if args.format == 'html':
        storyteller.export_html(output_path)
        print(f"Report exported to: {output_path}")
    elif args.format == 'pdf':
        storyteller.export_pdf(output_path)
        print(f"Report exported to: {output_path}")
    else:  # json
        Path(output_path).write_text(json.dumps(report, indent=2, default=str))
        print(f"Report exported to: {output_path}")

    # Export charts if requested
    if args.charts:
        exported = storyteller.export_charts(args.charts)
        print(f"Charts exported: {len(exported)} files to {args.charts}")
