#!/usr/bin/env python3
"""
Data Quality Auditor - Comprehensive data quality assessment.

Features:
- Missing values analysis
- Duplicate detection
- Type validation
- Pattern checking
- Quality scoring
- Report generation
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DataQualityAuditor:
    """Assess data quality in datasets."""

    def __init__(self):
        """Initialize auditor."""
        self._pd = None
        self._np = None
        self._load_dependencies()
        self.data = None
        self._audit_cache = None

    def _load_dependencies(self):
        """Load required libraries."""
        try:
            import pandas as pd
            import numpy as np
            self._pd = pd
            self._np = np
        except ImportError:
            raise ImportError("pandas and numpy required. Install with: pip install pandas numpy")

    def load_csv(self, filepath: str, **kwargs) -> 'DataQualityAuditor':
        """
        Load data from CSV.

        Args:
            filepath: Path to CSV file
            **kwargs: pandas read_csv arguments

        Returns:
            self for chaining
        """
        self.data = self._pd.read_csv(filepath, **kwargs)
        self._filepath = filepath
        self._audit_cache = None
        return self

    def load_dataframe(self, df) -> 'DataQualityAuditor':
        """
        Load data from DataFrame.

        Args:
            df: pandas DataFrame

        Returns:
            self for chaining
        """
        self.data = df.copy()
        self._filepath = "DataFrame"
        self._audit_cache = None
        return self

    def audit(self) -> Dict:
        """
        Run full data quality audit.

        Returns:
            Comprehensive audit report
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        # Run all checks
        missing = self.check_missing()
        duplicates = self.check_duplicates()
        types = self.check_types()
        uniqueness = self.check_uniqueness()

        # Calculate scores
        completeness_score = 100 - missing['missing_percent']
        uniqueness_score = 100 - duplicates['duplicate_percent']
        validity_score = self._calculate_validity_score(types)
        consistency_score = self._calculate_consistency_score()

        # Overall quality score (weighted average)
        quality_score = (
            completeness_score * 0.30 +
            uniqueness_score * 0.25 +
            validity_score * 0.25 +
            consistency_score * 0.20
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            missing, duplicates, types
        )

        self._audit_cache = {
            'file': getattr(self, '_filepath', 'Unknown'),
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'quality_score': round(quality_score, 1),
            'completeness': {
                'score': round(completeness_score, 1),
                'missing_cells': missing['missing_cells'],
                'details': missing
            },
            'uniqueness': {
                'score': round(uniqueness_score, 1),
                'duplicate_rows': duplicates['duplicate_rows'],
                'details': duplicates
            },
            'validity': {
                'score': round(validity_score, 1),
                'details': types
            },
            'consistency': {
                'score': round(consistency_score, 1),
                'details': uniqueness
            },
            'recommendations': recommendations
        }

        return self._audit_cache

    def quality_score(self) -> float:
        """Get overall quality score."""
        if self._audit_cache is None:
            self.audit()
        return self._audit_cache['quality_score']

    def check_missing(self) -> Dict:
        """
        Analyze missing values.

        Returns:
            Missing values report
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        total_cells = self.data.size
        missing_cells = self.data.isna().sum().sum()
        missing_percent = 100 * missing_cells / total_cells if total_cells > 0 else 0

        by_column = {}
        for col in self.data.columns:
            col_missing = self.data[col].isna().sum()
            col_total = len(self.data)
            by_column[col] = {
                'count': int(col_missing),
                'percent': round(100 * col_missing / col_total, 2) if col_total > 0 else 0
            }

        rows_with_missing = self.data.isna().any(axis=1).sum()

        return {
            'total_cells': int(total_cells),
            'missing_cells': int(missing_cells),
            'missing_percent': round(missing_percent, 2),
            'by_column': by_column,
            'rows_with_missing': int(rows_with_missing)
        }

    def check_duplicates(self, subset: Optional[List[str]] = None) -> Dict:
        """
        Detect duplicate rows.

        Args:
            subset: Columns to check for duplicates

        Returns:
            Duplicate analysis report
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        total_rows = len(self.data)
        duplicates = self.data.duplicated(subset=subset, keep='first')
        duplicate_rows = duplicates.sum()
        duplicate_percent = 100 * duplicate_rows / total_rows if total_rows > 0 else 0

        # Find duplicate groups
        duplicate_groups = []
        if duplicate_rows > 0:
            dup_mask = self.data.duplicated(subset=subset, keep=False)
            dup_df = self.data[dup_mask]
            if subset:
                groups = dup_df.groupby(subset).size()
            else:
                groups = dup_df.groupby(list(self.data.columns)).size()
            duplicate_groups = groups[groups > 1].head(10).to_dict()

        # Check duplicates by individual columns
        by_columns = {}
        for col in self.data.columns:
            col_dups = self.data[col].duplicated(keep='first').sum()
            if col_dups > 0:
                by_columns[col] = {'duplicates': int(col_dups)}

        return {
            'total_rows': total_rows,
            'duplicate_rows': int(duplicate_rows),
            'duplicate_percent': round(duplicate_percent, 2),
            'duplicate_groups': duplicate_groups,
            'by_columns': by_columns
        }

    def check_types(self) -> Dict:
        """
        Analyze column types.

        Returns:
            Type analysis report
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        columns = {}
        for col in self.data.columns:
            series = self.data[col]
            detected_type = str(series.dtype)
            unique_values = series.nunique()
            sample_values = series.dropna().head(5).tolist()

            issues = []

            # Check for mixed types in object columns
            if series.dtype == 'object':
                type_counts = series.apply(type).value_counts()
                if len(type_counts) > 1:
                    issues.append("Mixed types detected")

                # Check for potential numeric columns stored as strings
                try:
                    numeric_count = self._pd.to_numeric(series, errors='coerce').notna().sum()
                    if numeric_count > len(series) * 0.8:
                        issues.append("Potential numeric column stored as string")
                except:
                    pass

                # Check for potential date columns
                try:
                    date_count = self._pd.to_datetime(series, errors='coerce').notna().sum()
                    if date_count > len(series) * 0.8:
                        issues.append("Potential date column stored as string")
                except:
                    pass

            # Check for outliers in numeric columns
            if self._np.issubdtype(series.dtype, self._np.number):
                if series.std() > 0:
                    z_scores = (series - series.mean()) / series.std()
                    outlier_count = (abs(z_scores) > 3).sum()
                    if outlier_count > 0:
                        issues.append(f"{outlier_count} potential outliers (|z| > 3)")

            columns[col] = {
                'detected_type': detected_type,
                'unique_values': int(unique_values),
                'sample_values': sample_values,
                'issues': issues
            }

        return {'columns': columns}

    def check_uniqueness(self) -> Dict:
        """
        Check column uniqueness (cardinality).

        Returns:
            Uniqueness analysis
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        total_rows = len(self.data)
        columns = {}

        for col in self.data.columns:
            unique = self.data[col].nunique()
            uniqueness_ratio = unique / total_rows if total_rows > 0 else 0

            columns[col] = {
                'unique_values': int(unique),
                'uniqueness_ratio': round(uniqueness_ratio, 4),
                'is_unique': unique == total_rows,
                'is_constant': unique == 1
            }

        # Identify potential ID columns (high uniqueness)
        potential_ids = [
            col for col, info in columns.items()
            if info['uniqueness_ratio'] > 0.95
        ]

        # Identify low cardinality columns (potential categoricals)
        potential_categoricals = [
            col for col, info in columns.items()
            if info['unique_values'] < 20 and info['uniqueness_ratio'] < 0.1
        ]

        return {
            'columns': columns,
            'potential_ids': potential_ids,
            'potential_categoricals': potential_categoricals
        }

    def check_patterns(self, column: str, pattern: str) -> Dict:
        """
        Check column values against regex pattern.

        Args:
            column: Column name
            pattern: Regex pattern

        Returns:
            Pattern match report
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        series = self.data[column].astype(str)
        regex = re.compile(pattern)
        matches = series.apply(lambda x: bool(regex.match(str(x))))

        match_count = matches.sum()
        non_match_count = (~matches).sum()

        non_matching_samples = series[~matches].head(10).tolist()

        return {
            'column': column,
            'pattern': pattern,
            'total': len(series),
            'matches': int(match_count),
            'non_matches': int(non_match_count),
            'match_percent': round(100 * match_count / len(series), 2),
            'non_matching_samples': non_matching_samples
        }

    def validate_column(self, column: str, rules: Dict) -> Dict:
        """
        Validate column against rules.

        Args:
            column: Column name
            rules: Validation rules

        Returns:
            Validation result
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        if column not in self.data.columns:
            return {'valid': False, 'issues': [f"Column '{column}' not found"]}

        series = self.data[column]
        issues = []

        # Required check
        if rules.get('required'):
            missing = series.isna().sum()
            if missing > 0:
                issues.append(f"{missing} missing values")

        # Unique check
        if rules.get('unique'):
            dups = series.duplicated().sum()
            if dups > 0:
                issues.append(f"{dups} duplicate values")

        # Pattern check
        if 'pattern' in rules:
            pattern_result = self.check_patterns(column, rules['pattern'])
            if pattern_result['non_matches'] > 0:
                issues.append(f"{pattern_result['non_matches']} values don't match pattern")

        # Type check
        if 'type' in rules:
            expected_type = rules['type']
            if expected_type == 'integer':
                non_int = series.dropna().apply(lambda x: not isinstance(x, (int, self._np.integer)))
                if non_int.sum() > 0:
                    issues.append(f"{non_int.sum()} non-integer values")
            elif expected_type == 'date':
                try:
                    self._pd.to_datetime(series, format=rules.get('format'))
                except:
                    issues.append("Date parsing failed")

        # Range checks
        if 'min' in rules:
            below_min = (series < rules['min']).sum()
            if below_min > 0:
                issues.append(f"{below_min} values below minimum {rules['min']}")
        if 'max' in rules:
            above_max = (series > rules['max']).sum()
            if above_max > 0:
                issues.append(f"{above_max} values above maximum {rules['max']}")

        # Allowed values
        if 'allowed_values' in rules:
            invalid = ~series.isin(rules['allowed_values'])
            if invalid.sum() > 0:
                issues.append(f"{invalid.sum()} values not in allowed list")

        return {
            'column': column,
            'valid': len(issues) == 0,
            'issues': issues
        }

    def validate_dataset(self, rules: Dict) -> Dict:
        """
        Validate entire dataset against rules.

        Args:
            rules: Dict with column rules

        Returns:
            Validation results
        """
        results = {}
        all_valid = True

        for column, col_rules in rules.get('columns', {}).items():
            result = self.validate_column(column, col_rules)
            results[column] = result
            if not result['valid']:
                all_valid = False

        return {
            'valid': all_valid,
            'columns': results
        }

    def _calculate_validity_score(self, types: Dict) -> float:
        """Calculate validity score from type analysis."""
        if not types.get('columns'):
            return 100.0

        total_issues = sum(
            len(col_info['issues'])
            for col_info in types['columns'].values()
        )
        max_issues = len(types['columns']) * 3  # Assume max 3 issues per column

        return max(0, 100 - (100 * total_issues / max_issues if max_issues > 0 else 0))

    def _calculate_consistency_score(self) -> float:
        """Calculate consistency score."""
        uniqueness = self.check_uniqueness()
        total_cols = len(uniqueness['columns'])
        if total_cols == 0:
            return 100.0

        # Penalize constant columns
        constant_cols = sum(1 for c in uniqueness['columns'].values() if c['is_constant'])
        penalty = (constant_cols / total_cols) * 20

        return max(0, 100 - penalty)

    def _generate_recommendations(
        self,
        missing: Dict,
        duplicates: Dict,
        types: Dict
    ) -> List[str]:
        """Generate recommendations based on audit results."""
        recommendations = []

        # Missing value recommendations
        for col, info in missing['by_column'].items():
            if info['percent'] >= 5:
                recommendations.append(
                    f"Column '{col}' has {info['percent']}% missing values"
                )

        # Duplicate recommendations
        if duplicates['duplicate_rows'] > 0:
            recommendations.append(
                f"{duplicates['duplicate_rows']} duplicate rows detected"
            )

        # Type recommendations
        for col, info in types['columns'].items():
            for issue in info['issues']:
                recommendations.append(f"Column '{col}': {issue}")

        return recommendations[:10]  # Limit to top 10

    def generate_report(self, output: str, format: str = "html") -> str:
        """
        Generate quality report file.

        Args:
            output: Output file path
            format: Report format (html, json)

        Returns:
            Path to generated report
        """
        if self._audit_cache is None:
            self.audit()

        if format == 'json':
            with open(output, 'w') as f:
                json.dump(self._audit_cache, f, indent=2, default=str)
        elif format == 'html':
            html = self._generate_html_report()
            with open(output, 'w') as f:
                f.write(html)
        else:
            raise ValueError(f"Unknown format: {format}")

        return output

    def _generate_html_report(self) -> str:
        """Generate HTML report."""
        report = self._audit_cache

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .score {{ font-size: 48px; font-weight: bold; }}
        .score.good {{ color: #28a745; }}
        .score.warning {{ color: #ffc107; }}
        .score.bad {{ color: #dc3545; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>Data Quality Report</h1>
    <p>File: {report['file']}</p>
    <p>Rows: {report['rows']:,} | Columns: {report['columns']}</p>

    <h2>Overall Quality Score</h2>
    <div class="score {'good' if report['quality_score'] >= 80 else 'warning' if report['quality_score'] >= 60 else 'bad'}">
        {report['quality_score']}/100
    </div>

    <h2>Component Scores</h2>
    <table>
        <tr><th>Component</th><th>Score</th><th>Details</th></tr>
        <tr><td>Completeness</td><td>{report['completeness']['score']}</td>
            <td>{report['completeness']['missing_cells']} missing cells</td></tr>
        <tr><td>Uniqueness</td><td>{report['uniqueness']['score']}</td>
            <td>{report['uniqueness']['duplicate_rows']} duplicate rows</td></tr>
        <tr><td>Validity</td><td>{report['validity']['score']}</td><td>Type issues checked</td></tr>
        <tr><td>Consistency</td><td>{report['consistency']['score']}</td><td>Pattern consistency</td></tr>
    </table>

    <h2>Recommendations</h2>
    {''.join(f'<div class="recommendation">{r}</div>' for r in report['recommendations']) or '<p>No issues found</p>'}
</body>
</html>
"""
        return html

    def summary(self) -> str:
        """Get text summary of audit."""
        if self._audit_cache is None:
            self.audit()

        report = self._audit_cache
        lines = [
            f"Data Quality Report: {report['file']}",
            f"{'=' * 50}",
            f"Rows: {report['rows']:,} | Columns: {report['columns']}",
            f"",
            f"Quality Score: {report['quality_score']}/100",
            f"  - Completeness: {report['completeness']['score']}/100",
            f"  - Uniqueness: {report['uniqueness']['score']}/100",
            f"  - Validity: {report['validity']['score']}/100",
            f"  - Consistency: {report['consistency']['score']}/100",
            f"",
            f"Issues:",
        ]
        for rec in report['recommendations']:
            lines.append(f"  - {rec}")

        if not report['recommendations']:
            lines.append("  No issues found")

        return '\n'.join(lines)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Audit data quality')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--report', '-r', help='Generate report file')
    parser.add_argument('--format', '-f', default='html', choices=['html', 'json'])
    parser.add_argument('--missing', action='store_true', help='Check missing values')
    parser.add_argument('--duplicates', action='store_true', help='Check duplicates')
    parser.add_argument('--types', action='store_true', help='Check types')
    parser.add_argument('--rules', help='Validation rules JSON file')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()
    auditor = DataQualityAuditor()
    auditor.load_csv(args.input)

    if args.missing:
        result = auditor.check_missing()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Missing Values Analysis")
            print("=" * 40)
            print(f"Total cells: {result['total_cells']:,}")
            print(f"Missing cells: {result['missing_cells']:,} ({result['missing_percent']}%)")
            print(f"Rows with missing: {result['rows_with_missing']:,}")
            print("\nBy Column:")
            for col, info in result['by_column'].items():
                if info['count'] > 0:
                    print(f"  {col}: {info['count']} ({info['percent']}%)")

    elif args.duplicates:
        result = auditor.check_duplicates()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Duplicate Analysis")
            print("=" * 40)
            print(f"Total rows: {result['total_rows']:,}")
            print(f"Duplicate rows: {result['duplicate_rows']:,} ({result['duplicate_percent']}%)")

    elif args.types:
        result = auditor.check_types()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Type Analysis")
            print("=" * 40)
            for col, info in result['columns'].items():
                print(f"\n{col}:")
                print(f"  Type: {info['detected_type']}")
                print(f"  Unique values: {info['unique_values']}")
                if info['issues']:
                    print(f"  Issues: {', '.join(info['issues'])}")

    elif args.rules:
        with open(args.rules) as f:
            rules = json.load(f)
        result = auditor.validate_dataset(rules)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Validation: {'PASSED' if result['valid'] else 'FAILED'}")
            for col, info in result['columns'].items():
                status = 'OK' if info['valid'] else 'FAIL'
                print(f"  {col}: {status}")
                for issue in info['issues']:
                    print(f"    - {issue}")

    elif args.report:
        auditor.generate_report(args.report, format=args.format)
        print(f"Report saved to: {args.report}")

    else:
        # Full audit
        report = auditor.audit()
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print(auditor.summary())


if __name__ == "__main__":
    main()
