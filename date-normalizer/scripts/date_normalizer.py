#!/usr/bin/env python3
"""
Date Normalizer - Parse and normalize dates from various formats.

Features:
- Smart date parsing (100+ formats)
- Format conversion (ISO8601, US, EU, custom)
- Batch CSV processing
- Ambiguity detection
- Relative date parsing
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dateutil import parser
from dateutil.relativedelta import relativedelta
import pandas as pd


class DateNormalizer:
    """Parse and normalize dates from various formats."""

    OUTPUT_FORMATS = {
        'iso8601': '%Y-%m-%d',
        'iso8601_time': '%Y-%m-%dT%H:%M:%S',
        'us': '%m/%d/%Y',
        'eu': '%d/%m/%Y',
        'long': '%B %d, %Y',
        'short': '%b %d, %Y',
        'compact': '%Y%m%d',
        'unix': 'unix'  # Special case
    }

    RELATIVE_PATTERNS = {
        r'today': lambda: datetime.now(),
        r'yesterday': lambda: datetime.now() - timedelta(days=1),
        r'tomorrow': lambda: datetime.now() + timedelta(days=1),
        r'(\d+)\s+days?\s+ago': lambda m: datetime.now() - timedelta(days=int(m.group(1))),
        r'(\d+)\s+weeks?\s+ago': lambda m: datetime.now() - timedelta(weeks=int(m.group(1))),
        r'(\d+)\s+months?\s+ago': lambda m: datetime.now() - relativedelta(months=int(m.group(1))),
        r'in\s+(\d+)\s+days?': lambda m: datetime.now() + timedelta(days=int(m.group(1))),
        r'in\s+(\d+)\s+weeks?': lambda m: datetime.now() + timedelta(weeks=int(m.group(1))),
        r'next\s+week': lambda: datetime.now() + timedelta(weeks=1),
        r'last\s+week': lambda: datetime.now() - timedelta(weeks=1),
        r'next\s+month': lambda: datetime.now() + relativedelta(months=1),
        r'last\s+month': lambda: datetime.now() - relativedelta(months=1),
    }

    def __init__(self):
        pass

    def normalize(self, date_string: str, output_format: str = 'iso8601',
                 dayfirst: bool = False, yearfirst: bool = False) -> Dict[str, Any]:
        """
        Normalize date string to specified format.

        Args:
            date_string: Date string to normalize
            output_format: Output format ('iso8601', 'us', 'eu', etc.)
            dayfirst: Interpret DD/MM/YY format (default: False for MM/DD/YY)
            yearfirst: Interpret YY/MM/DD format (default: False)

        Returns:
            Dictionary with normalized date and metadata
        """
        if not date_string or not isinstance(date_string, str):
            return {
                'original': date_string,
                'normalized': None,
                'valid': False,
                'error': 'Empty or invalid input'
            }

        date_string = str(date_string).strip()

        try:
            # Try parsing relative dates first
            dt = self._parse_relative(date_string.lower())

            if dt is None:
                # Use dateutil parser for standard dates
                dt = parser.parse(
                    date_string,
                    dayfirst=dayfirst,
                    yearfirst=yearfirst,
                    fuzzy=True
                )

            # Format output
            if output_format in self.OUTPUT_FORMATS:
                if output_format == 'unix':
                    normalized = str(int(dt.timestamp()))
                else:
                    format_str = self.OUTPUT_FORMATS[output_format]
                    normalized = dt.strftime(format_str)
            else:
                # Custom format string
                normalized = dt.strftime(output_format)

            return {
                'original': date_string,
                'normalized': normalized,
                'parsed_datetime': dt.isoformat(),
                'valid': True,
                'format': output_format,
                'ambiguous': self.is_ambiguous(date_string)
            }

        except Exception as e:
            return {
                'original': date_string,
                'normalized': None,
                'valid': False,
                'error': str(e)
            }

    def _parse_relative(self, date_string: str) -> Optional[datetime]:
        """Parse relative date expressions."""
        for pattern, func in self.RELATIVE_PATTERNS.items():
            match = re.match(pattern, date_string)
            if match:
                if callable(func):
                    return func(match) if '(' in str(func.__code__.co_varnames) else func()
        return None

    def is_valid(self, date_string: str) -> bool:
        """
        Check if date string is valid.

        Args:
            date_string: Date string to validate

        Returns:
            True if valid, False otherwise
        """
        result = self.normalize(date_string)
        return result.get('valid', False)

    def is_ambiguous(self, date_string: str) -> bool:
        """
        Check if date string is ambiguous.

        Dates like "01/02/03" are ambiguous.

        Args:
            date_string: Date string to check

        Returns:
            True if ambiguous, False otherwise
        """
        # Check for numeric date formats that could be interpreted multiple ways
        patterns = [
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # 01/02/03
            r'^\d{1,2}-\d{1,2}-\d{2,4}$',  # 01-02-03
            r'^\d{1,2}\.\d{1,2}\.\d{2,4}$',  # 01.02.03
        ]

        for pattern in patterns:
            if re.match(pattern, date_string.strip()):
                # Check if all parts are <= 12 (could be month or day)
                parts = re.split(r'[/\-.]', date_string.strip())
                if len(parts) == 3:
                    try:
                        parts_int = [int(p) for p in parts]
                        # If first two parts are both <= 12, it's ambiguous
                        if parts_int[0] <= 12 and parts_int[1] <= 12:
                            return True
                    except ValueError:
                        pass

        return False

    def detect_format(self, date_string: str) -> str:
        """
        Detect the format of a date string.

        Args:
            date_string: Date string to analyze

        Returns:
            Detected format description
        """
        if not self.is_valid(date_string):
            return 'invalid'

        # Common patterns
        patterns = {
            r'^\d{4}-\d{2}-\d{2}$': 'ISO 8601 (YYYY-MM-DD)',
            r'^\d{2}/\d{2}/\d{4}$': 'US or EU (MM/DD/YYYY or DD/MM/YYYY)',
            r'^\d{4}/\d{2}/\d{2}$': 'Japanese (YYYY/MM/DD)',
            r'^\d{8}$': 'Compact (YYYYMMDD)',
            r'^[A-Za-z]+\s+\d{1,2},\s+\d{4}$': 'Long US (Month DD, YYYY)',
            r'^\d{1,2}\s+[A-Za-z]+\s+\d{4}$': 'Long EU (DD Month YYYY)',
        }

        for pattern, format_name in patterns.items():
            if re.match(pattern, date_string.strip()):
                return format_name

        return 'Mixed or textual format'

    def normalize_batch(self, dates: List[str], **kwargs) -> List[Dict]:
        """
        Normalize multiple dates.

        Args:
            dates: List of date strings
            **kwargs: Arguments to pass to normalize()

        Returns:
            List of normalization results
        """
        return [self.normalize(date, **kwargs) for date in dates]

    def normalize_csv(self, csv_path: str, date_column: str,
                     output: str = None, output_format: str = 'iso8601',
                     **kwargs) -> str:
        """
        Normalize dates in CSV file.

        Args:
            csv_path: Input CSV file path
            date_column: Column containing dates
            output: Output CSV file path (default: input_normalized.csv)
            output_format: Output date format
            **kwargs: Arguments to pass to normalize()

        Returns:
            Path to output CSV file
        """
        # Read CSV
        df = pd.read_csv(csv_path)

        if date_column not in df.columns:
            raise ValueError(f"Column '{date_column}' not found in CSV")

        # Normalize dates
        print(f"Normalizing {len(df)} dates from column '{date_column}'...")

        results = []
        errors = []

        for i, date_val in enumerate(df[date_column], 1):
            result = self.normalize(str(date_val), output_format=output_format, **kwargs)
            results.append(result)

            if not result['valid']:
                errors.append((i, date_val, result.get('error')))

        # Update column with normalized dates
        df[date_column] = [r['normalized'] for r in results]

        # Add validation column
        df[f'{date_column}_valid'] = [r['valid'] for r in results]

        # Save
        if output is None:
            output = csv_path.replace('.csv', '_normalized.csv')

        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        df.to_csv(output, index=False)

        # Report errors
        if errors:
            print(f"\nWarning: {len(errors)} invalid dates found:")
            for row, value, error in errors[:10]:  # Show first 10
                print(f"  Row {row}: '{value}' - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")

        return output


def main():
    parser = argparse.ArgumentParser(
        description='Parse and normalize dates from various formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normalize single date
  python date_normalizer.py --date "March 14, 2024"

  # Convert to US format
  python date_normalizer.py --date "14/03/2024" --format us

  # Normalize CSV column
  python date_normalizer.py --csv data.csv --column created_at --format iso8601 --output normalized.csv

  # Check if date is ambiguous
  python date_normalizer.py --date "01/02/03" --detect-ambiguous
        """
    )

    parser.add_argument('--date', '-d', help='Single date string to normalize')
    parser.add_argument('--csv', help='CSV file to process')
    parser.add_argument('--column', '-c', help='CSV column containing dates')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--format', '-f', default='iso8601',
                       help='Output format (iso8601, us, eu, long, etc.)')
    parser.add_argument('--dayfirst', action='store_true',
                       help='Interpret DD/MM/YY format (EU style)')
    parser.add_argument('--yearfirst', action='store_true',
                       help='Interpret YY/MM/DD format')
    parser.add_argument('--detect-ambiguous', action='store_true',
                       help='Check if date is ambiguous')
    parser.add_argument('--detect-format', action='store_true',
                       help='Detect date format')

    args = parser.parse_args()

    normalizer = DateNormalizer()

    # Single date mode
    if args.date:
        result = normalizer.normalize(
            args.date,
            output_format=args.format,
            dayfirst=args.dayfirst,
            yearfirst=args.yearfirst
        )

        print(f"\nOriginal: {result['original']}")

        if result['valid']:
            print(f"Normalized: {result['normalized']}")
            print(f"Format: {result['format']}")

            if args.detect_ambiguous:
                if result.get('ambiguous'):
                    print("⚠ Warning: Date is ambiguous. Specify --dayfirst or --yearfirst")
                else:
                    print("✓ Date is unambiguous")

            if args.detect_format:
                fmt = normalizer.detect_format(args.date)
                print(f"Detected Format: {fmt}")

        else:
            print(f"✗ Invalid date: {result.get('error')}")

    # CSV mode
    elif args.csv:
        if not args.column:
            print("Error: --column required for CSV processing")
            return

        print(f"Processing CSV: {args.csv}")
        output = normalizer.normalize_csv(
            args.csv,
            date_column=args.column,
            output=args.output,
            output_format=args.format,
            dayfirst=args.dayfirst,
            yearfirst=args.yearfirst
        )

        print(f"\n✓ Normalized dates saved to: {output}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
