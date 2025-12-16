#!/usr/bin/env python3
"""
Dataset Comparer - Compare two datasets to find differences.
"""

import argparse
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import html

import numpy as np
import pandas as pd


class DatasetComparer:
    """Compare two datasets to identify differences."""

    def __init__(self):
        """Initialize the comparer."""
        self.old_df: Optional[pd.DataFrame] = None
        self.new_df: Optional[pd.DataFrame] = None
        self.comparison_result: Optional[Dict] = None
        self.key_columns: List[str] = []
        self._added: Optional[pd.DataFrame] = None
        self._removed: Optional[pd.DataFrame] = None
        self._changed: Optional[pd.DataFrame] = None
        self._unchanged: Optional[pd.DataFrame] = None

    def load(self, old_path: str, new_path: str) -> 'DatasetComparer':
        """
        Load datasets from files.

        Args:
            old_path: Path to old/baseline dataset
            new_path: Path to new dataset

        Returns:
            Self for method chaining
        """
        # Detect file type and load
        self.old_df = self._load_file(old_path)
        self.new_df = self._load_file(new_path)
        return self

    def _load_file(self, filepath: str) -> pd.DataFrame:
        """Load file based on extension."""
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            return pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        else:
            # Try CSV as default
            return pd.read_csv(filepath)

    def load_dataframes(self, old_df: pd.DataFrame,
                       new_df: pd.DataFrame) -> 'DatasetComparer':
        """
        Load from pandas DataFrames.

        Args:
            old_df: Old/baseline DataFrame
            new_df: New DataFrame

        Returns:
            Self for method chaining
        """
        self.old_df = old_df.copy()
        self.new_df = new_df.copy()
        return self

    def compare_schema(self) -> Dict:
        """
        Compare column structure between datasets.

        Returns:
            Dictionary with schema comparison
        """
        if self.old_df is None or self.new_df is None:
            raise ValueError("No data loaded")

        old_cols = set(self.old_df.columns)
        new_cols = set(self.new_df.columns)

        common = old_cols & new_cols
        added = new_cols - old_cols
        removed = old_cols - new_cols

        # Check type changes in common columns
        type_changes = []
        for col in common:
            old_type = str(self.old_df[col].dtype)
            new_type = str(self.new_df[col].dtype)
            if old_type != new_type:
                type_changes.append({
                    "column": col,
                    "old_type": old_type,
                    "new_type": new_type
                })

        return {
            "old_columns": list(self.old_df.columns),
            "new_columns": list(self.new_df.columns),
            "common_columns": list(common),
            "added_columns": list(added),
            "removed_columns": list(removed),
            "type_changes": type_changes,
            "old_row_count": len(self.old_df),
            "new_row_count": len(self.new_df)
        }

    def compare(self, key_columns: List[str] = None,
               ignore_columns: List[str] = None,
               compare_columns: List[str] = None) -> Dict:
        """
        Compare datasets and identify differences.

        Args:
            key_columns: Columns to use as row identifiers
            ignore_columns: Columns to exclude from comparison
            compare_columns: Only compare these columns (if specified)

        Returns:
            Dictionary with comparison results
        """
        if self.old_df is None or self.new_df is None:
            raise ValueError("No data loaded")

        self.key_columns = key_columns or []
        ignore_columns = ignore_columns or []

        # Get common columns
        common_cols = set(self.old_df.columns) & set(self.new_df.columns)

        # Determine columns to compare
        if compare_columns:
            cols_to_compare = [c for c in compare_columns if c in common_cols]
        else:
            cols_to_compare = [c for c in common_cols
                            if c not in ignore_columns and c not in self.key_columns]

        # Schema comparison
        schema_changes = self.compare_schema()

        if key_columns:
            self._compare_by_key(key_columns, cols_to_compare)
        else:
            self._compare_by_position(cols_to_compare)

        self.comparison_result = {
            "summary": {
                "old_rows": len(self.old_df),
                "new_rows": len(self.new_df),
                "added_count": len(self._added),
                "removed_count": len(self._removed),
                "changed_count": len(self._changed),
                "unchanged_count": len(self._unchanged) if self._unchanged is not None else 0,
                "total_differences": len(self._added) + len(self._removed) + len(self._changed)
            },
            "schema_changes": {
                "added_columns": schema_changes["added_columns"],
                "removed_columns": schema_changes["removed_columns"],
                "type_changes": schema_changes["type_changes"]
            },
            "key_columns": self.key_columns,
            "compared_columns": cols_to_compare,
            "ignored_columns": ignore_columns
        }

        return self.comparison_result

    def _compare_by_key(self, key_columns: List[str], compare_cols: List[str]):
        """Compare datasets by key columns."""
        # Create key column
        old_df = self.old_df.copy()
        new_df = self.new_df.copy()

        # Create composite key
        old_df['_key'] = old_df[key_columns].astype(str).agg('|'.join, axis=1)
        new_df['_key'] = new_df[key_columns].astype(str).agg('|'.join, axis=1)

        old_keys = set(old_df['_key'])
        new_keys = set(new_df['_key'])

        # Find added/removed
        added_keys = new_keys - old_keys
        removed_keys = old_keys - new_keys
        common_keys = old_keys & new_keys

        self._added = new_df[new_df['_key'].isin(added_keys)].drop('_key', axis=1)
        self._removed = old_df[old_df['_key'].isin(removed_keys)].drop('_key', axis=1)

        # Compare common rows
        old_common = old_df[old_df['_key'].isin(common_keys)].set_index('_key')
        new_common = new_df[new_df['_key'].isin(common_keys)].set_index('_key')

        # Find changed rows
        changed_rows = []
        unchanged_indices = []

        for key in common_keys:
            old_row = old_common.loc[key]
            new_row = new_common.loc[key]

            changes_in_row = []
            for col in compare_cols:
                if col in old_row.index and col in new_row.index:
                    old_val = old_row[col]
                    new_val = new_row[col]

                    # Handle NaN comparisons
                    if pd.isna(old_val) and pd.isna(new_val):
                        continue
                    elif pd.isna(old_val) or pd.isna(new_val) or old_val != new_val:
                        changes_in_row.append({
                            "_key": key,
                            "_column": col,
                            "_old_value": old_val,
                            "_new_value": new_val
                        })

            if changes_in_row:
                changed_rows.extend(changes_in_row)
            else:
                unchanged_indices.append(key)

        self._changed = pd.DataFrame(changed_rows) if changed_rows else pd.DataFrame()
        self._unchanged = old_common.loc[old_common.index.isin(unchanged_indices)].reset_index(drop=True)

    def _compare_by_position(self, compare_cols: List[str]):
        """Compare datasets by row position."""
        min_rows = min(len(self.old_df), len(self.new_df))

        # Added = new rows beyond old count
        if len(self.new_df) > len(self.old_df):
            self._added = self.new_df.iloc[len(self.old_df):].copy()
        else:
            self._added = pd.DataFrame()

        # Removed = old rows beyond new count
        if len(self.old_df) > len(self.new_df):
            self._removed = self.old_df.iloc[len(self.new_df):].copy()
        else:
            self._removed = pd.DataFrame()

        # Compare overlapping rows
        changed_rows = []
        unchanged_indices = []

        for i in range(min_rows):
            old_row = self.old_df.iloc[i]
            new_row = self.new_df.iloc[i]

            changes_in_row = []
            for col in compare_cols:
                if col in old_row.index and col in new_row.index:
                    old_val = old_row[col]
                    new_val = new_row[col]

                    if pd.isna(old_val) and pd.isna(new_val):
                        continue
                    elif pd.isna(old_val) or pd.isna(new_val) or old_val != new_val:
                        changes_in_row.append({
                            "_key": i,
                            "_column": col,
                            "_old_value": old_val,
                            "_new_value": new_val
                        })

            if changes_in_row:
                changed_rows.extend(changes_in_row)
            else:
                unchanged_indices.append(i)

        self._changed = pd.DataFrame(changed_rows) if changed_rows else pd.DataFrame()
        self._unchanged = self.old_df.iloc[unchanged_indices].copy() if unchanged_indices else pd.DataFrame()

    def get_added_rows(self) -> pd.DataFrame:
        """Get rows that were added in the new dataset."""
        if self._added is None:
            raise ValueError("Run compare() first")
        return self._added.copy()

    def get_removed_rows(self) -> pd.DataFrame:
        """Get rows that were removed from the old dataset."""
        if self._removed is None:
            raise ValueError("Run compare() first")
        return self._removed.copy()

    def get_changed_rows(self) -> pd.DataFrame:
        """Get details of changed values."""
        if self._changed is None:
            raise ValueError("Run compare() first")
        return self._changed.copy()

    def get_unchanged_rows(self) -> pd.DataFrame:
        """Get rows that remained unchanged."""
        if self._unchanged is None:
            raise ValueError("Run compare() first")
        return self._unchanged.copy()

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all differences to a single DataFrame.

        Returns:
            DataFrame with all differences
        """
        if self.comparison_result is None:
            raise ValueError("Run compare() first")

        rows = []

        # Added rows
        for _, row in self._added.iterrows():
            for col in row.index:
                rows.append({
                    "change_type": "added",
                    "key": None,
                    "column": col,
                    "old_value": None,
                    "new_value": row[col]
                })

        # Removed rows
        for _, row in self._removed.iterrows():
            for col in row.index:
                rows.append({
                    "change_type": "removed",
                    "key": None,
                    "column": col,
                    "old_value": row[col],
                    "new_value": None
                })

        # Changed values
        for _, row in self._changed.iterrows():
            rows.append({
                "change_type": "modified",
                "key": row["_key"],
                "column": row["_column"],
                "old_value": row["_old_value"],
                "new_value": row["_new_value"]
            })

        return pd.DataFrame(rows)

    def generate_report(self, output: str, format: str = "html") -> str:
        """
        Generate comparison report.

        Args:
            output: Output file path
            format: "html", "csv", or "json"

        Returns:
            Output file path
        """
        if self.comparison_result is None:
            raise ValueError("Run compare() first")

        if format == "html":
            self._generate_html_report(output)
        elif format == "csv":
            self._generate_csv_report(output)
        elif format == "json":
            self._generate_json_report(output)
        else:
            raise ValueError(f"Unknown format: {format}")

        return output

    def _generate_html_report(self, output: str):
        """Generate HTML report."""
        summary = self.comparison_result["summary"]
        schema = self.comparison_result["schema_changes"]

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Dataset Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat {{ background: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .stat-label {{ color: #666; font-size: 12px; }}
        .added {{ background: #e8f5e9; }}
        .removed {{ background: #ffebee; }}
        .changed {{ background: #fff3e0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 11px; }}
        .badge-added {{ background: #4CAF50; color: white; }}
        .badge-removed {{ background: #f44336; color: white; }}
        .badge-changed {{ background: #ff9800; color: white; }}
        .timestamp {{ color: #999; font-size: 12px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Dataset Comparison Report</h1>

        <h2>Summary</h2>
        <div class="summary">
            <div class="stat">
                <div class="stat-value">{summary['old_rows']:,}</div>
                <div class="stat-label">OLD ROWS</div>
            </div>
            <div class="stat">
                <div class="stat-value">{summary['new_rows']:,}</div>
                <div class="stat-label">NEW ROWS</div>
            </div>
            <div class="stat added">
                <div class="stat-value">{summary['added_count']:,}</div>
                <div class="stat-label">ADDED</div>
            </div>
            <div class="stat removed">
                <div class="stat-value">{summary['removed_count']:,}</div>
                <div class="stat-label">REMOVED</div>
            </div>
            <div class="stat changed">
                <div class="stat-value">{summary['changed_count']:,}</div>
                <div class="stat-label">CHANGED</div>
            </div>
            <div class="stat">
                <div class="stat-value">{summary['unchanged_count']:,}</div>
                <div class="stat-label">UNCHANGED</div>
            </div>
        </div>

        <h2>Schema Changes</h2>
        <table>
            <tr>
                <th>Change Type</th>
                <th>Details</th>
            </tr>
            <tr>
                <td>Added Columns</td>
                <td>{', '.join(schema['added_columns']) or 'None'}</td>
            </tr>
            <tr>
                <td>Removed Columns</td>
                <td>{', '.join(schema['removed_columns']) or 'None'}</td>
            </tr>
            <tr>
                <td>Type Changes</td>
                <td>{self._format_type_changes(schema['type_changes'])}</td>
            </tr>
        </table>
"""

        # Added rows
        if len(self._added) > 0:
            html_content += """
        <h2><span class="badge badge-added">ADDED</span> New Rows</h2>
        <table>
"""
            html_content += "<tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in self._added.columns) + "</tr>\n"
            for _, row in self._added.head(100).iterrows():
                html_content += "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>\n"
            html_content += "</table>\n"
            if len(self._added) > 100:
                html_content += f"<p>... and {len(self._added) - 100} more rows</p>\n"

        # Removed rows
        if len(self._removed) > 0:
            html_content += """
        <h2><span class="badge badge-removed">REMOVED</span> Deleted Rows</h2>
        <table>
"""
            html_content += "<tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in self._removed.columns) + "</tr>\n"
            for _, row in self._removed.head(100).iterrows():
                html_content += "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>\n"
            html_content += "</table>\n"
            if len(self._removed) > 100:
                html_content += f"<p>... and {len(self._removed) - 100} more rows</p>\n"

        # Changed rows
        if len(self._changed) > 0:
            html_content += """
        <h2><span class="badge badge-changed">CHANGED</span> Modified Values</h2>
        <table>
            <tr>
                <th>Key</th>
                <th>Column</th>
                <th>Old Value</th>
                <th>New Value</th>
            </tr>
"""
            for _, row in self._changed.head(100).iterrows():
                html_content += f"""            <tr>
                <td>{html.escape(str(row['_key']))}</td>
                <td>{html.escape(str(row['_column']))}</td>
                <td>{html.escape(str(row['_old_value']))}</td>
                <td>{html.escape(str(row['_new_value']))}</td>
            </tr>
"""
            html_content += "</table>\n"
            if len(self._changed) > 100:
                html_content += f"<p>... and {len(self._changed) - 100} more changes</p>\n"

        html_content += f"""
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>"""

        with open(output, 'w') as f:
            f.write(html_content)

    def _format_type_changes(self, changes: List[Dict]) -> str:
        """Format type changes for display."""
        if not changes:
            return "None"
        return "; ".join(f"{c['column']}: {c['old_type']} → {c['new_type']}" for c in changes)

    def _generate_csv_report(self, output: str):
        """Generate CSV report."""
        df = self.to_dataframe()
        df.to_csv(output, index=False)

    def _generate_json_report(self, output: str):
        """Generate JSON report."""
        result = {
            "comparison": self.comparison_result,
            "added_rows": self._added.to_dict(orient="records"),
            "removed_rows": self._removed.to_dict(orient="records"),
            "changed_values": self._changed.to_dict(orient="records") if len(self._changed) > 0 else []
        }

        with open(output, 'w') as f:
            json.dump(result, f, indent=2, default=str)

    def summary(self) -> str:
        """
        Generate text summary.

        Returns:
            Summary string
        """
        if self.comparison_result is None:
            raise ValueError("Run compare() first")

        s = self.comparison_result["summary"]

        lines = [
            "=" * 50,
            "DATASET COMPARISON SUMMARY",
            "=" * 50,
            f"Old dataset: {s['old_rows']:,} rows",
            f"New dataset: {s['new_rows']:,} rows",
            "",
            "DIFFERENCES",
            "-" * 30,
            f"Added rows:     {s['added_count']:,}",
            f"Removed rows:   {s['removed_count']:,}",
            f"Changed rows:   {s['changed_count']:,}",
            f"Unchanged rows: {s['unchanged_count']:,}",
            "",
            f"Total differences: {s['total_differences']:,}",
            "",
            "SCHEMA CHANGES",
            "-" * 30
        ]

        schema = self.comparison_result["schema_changes"]
        if schema["added_columns"]:
            lines.append(f"Added columns: {', '.join(schema['added_columns'])}")
        if schema["removed_columns"]:
            lines.append(f"Removed columns: {', '.join(schema['removed_columns'])}")
        if schema["type_changes"]:
            for tc in schema["type_changes"]:
                lines.append(f"Type change: {tc['column']} ({tc['old_type']} -> {tc['new_type']})")

        if not any([schema["added_columns"], schema["removed_columns"], schema["type_changes"]]):
            lines.append("No schema changes")

        lines.append("=" * 50)
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Comparer - Compare two datasets to find differences"
    )

    parser.add_argument("--old", required=True, help="Old/baseline dataset path")
    parser.add_argument("--new", required=True, help="New dataset path")
    parser.add_argument("--key", "-k", help="Key column(s) for matching (comma-separated)")
    parser.add_argument("--ignore", help="Columns to ignore (comma-separated)")
    parser.add_argument("--columns", "-c", help="Only compare these columns (comma-separated)")
    parser.add_argument("--report", "-r", help="Output report file (html/csv/json based on extension)")
    parser.add_argument("--output", "-o", help="Output CSV file for differences")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    comparer = DatasetComparer()
    comparer.load(args.old, args.new)

    # Parse arguments
    key_columns = args.key.split(',') if args.key else None
    ignore_columns = args.ignore.split(',') if args.ignore else None
    compare_columns = args.columns.split(',') if args.columns else None

    # Run comparison
    result = comparer.compare(
        key_columns=key_columns,
        ignore_columns=ignore_columns,
        compare_columns=compare_columns
    )

    if args.json:
        output = {
            "comparison": result,
            "added_rows": comparer.get_added_rows().to_dict(orient="records"),
            "removed_rows": comparer.get_removed_rows().to_dict(orient="records"),
            "changed_values": comparer.get_changed_rows().to_dict(orient="records")
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print(comparer.summary())

    # Generate report
    if args.report:
        ext = args.report.split('.')[-1].lower()
        format_map = {'html': 'html', 'csv': 'csv', 'json': 'json'}
        fmt = format_map.get(ext, 'html')
        comparer.generate_report(args.report, format=fmt)
        print(f"\nReport saved to: {args.report}")

    # Export differences
    if args.output:
        df = comparer.to_dataframe()
        df.to_csv(args.output, index=False)
        print(f"Differences exported to: {args.output}")


if __name__ == "__main__":
    main()
