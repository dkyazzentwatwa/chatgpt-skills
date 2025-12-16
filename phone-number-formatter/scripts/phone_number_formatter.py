#!/usr/bin/env python3
"""
Phone Number Formatter - Standardize phone numbers.
"""

import argparse
import pandas as pd
import phonenumbers
from phonenumbers import NumberParseException


class PhoneNumberFormatter:
    """Format and validate phone numbers."""

    def __init__(self, default_region: str = 'US'):
        """Initialize formatter."""
        self.default_region = default_region

    def format_number(self, number: str, format_type: str = 'international') -> str:
        """Format single phone number."""
        try:
            parsed = phonenumbers.parse(number, self.default_region)

            if not phonenumbers.is_valid_number(parsed):
                return None

            if format_type == 'international':
                return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
            elif format_type == 'e164':
                return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
            elif format_type == 'national':
                return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL)
            else:
                return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL)

        except NumberParseException:
            return None

    def validate(self, number: str) -> bool:
        """Validate phone number."""
        try:
            parsed = phonenumbers.parse(number, self.default_region)
            return phonenumbers.is_valid_number(parsed)
        except NumberParseException:
            return False

    def get_info(self, number: str) -> dict:
        """Get phone number information."""
        try:
            parsed = phonenumbers.parse(number, self.default_region)

            return {
                'number': number,
                'valid': phonenumbers.is_valid_number(parsed),
                'country_code': parsed.country_code,
                'national_number': parsed.national_number,
                'carrier': None,  # Requires carrier data
                'type': phonenumbers.number_type(parsed)
            }

        except NumberParseException:
            return {'number': number, 'valid': False}

    def format_batch(self, numbers: list, format_type: str = 'international') -> list:
        """Format multiple phone numbers."""
        return [self.format_number(num, format_type) for num in numbers]


def main():
    parser = argparse.ArgumentParser(description="Phone Number Formatter")

    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--column", required=True, help="Phone column name")
    parser.add_argument("--format", default='international',
                       choices=['international', 'e164', 'national'],
                       help="Output format")
    parser.add_argument("--region", default='US', help="Default region (e.g., US, GB)")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file")

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    phones = df[args.column].tolist()

    formatter = PhoneNumberFormatter(default_region=args.region)

    # Format numbers
    formatted = formatter.format_batch(phones, format_type=args.format)

    # Add to dataframe
    df['formatted_phone'] = formatted
    df['valid'] = [formatter.validate(str(p)) for p in phones]

    df.to_csv(args.output, index=False)

    valid_count = df['valid'].sum()
    print(f"Processed {len(phones)} phone numbers")
    print(f"Valid: {valid_count}/{len(phones)}")
    print(f"Output saved: {args.output}")


if __name__ == "__main__":
    main()
