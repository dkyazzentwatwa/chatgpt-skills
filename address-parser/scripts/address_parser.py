#!/usr/bin/env python3
"""
Address Parser - Parse unstructured addresses.
"""

import argparse
import re
import pandas as pd


class AddressParser:
    """Parse addresses into components."""

    US_STATES = {
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
    }

    def __init__(self):
        """Initialize parser."""
        pass

    def parse(self, address: str) -> dict:
        """Parse address into components."""
        result = {
            'original': address,
            'street': None,
            'city': None,
            'state': None,
            'zip': None,
            'country': 'USA'
        }

        # Extract ZIP code
        zip_match = re.search(r'\b\d{5}(?:-\d{4})?\b', address)
        if zip_match:
            result['zip'] = zip_match.group()
            address = address.replace(zip_match.group(), '').strip()

        # Extract state
        for state in self.US_STATES:
            if re.search(rf'\b{state}\b', address, re.IGNORECASE):
                result['state'] = state
                address = re.sub(rf'\b{state}\b', '', address, flags=re.IGNORECASE).strip()
                break

        # Split remaining by comma
        parts = [p.strip() for p in address.split(',')]

        if len(parts) >= 2:
            result['street'] = parts[0]
            result['city'] = parts[1]
        elif len(parts) == 1:
            result['street'] = parts[0]

        # Clean up
        for key in result:
            if isinstance(result[key], str):
                result[key] = result[key].strip().strip(',').strip()

        return result

    def parse_batch(self, addresses: list) -> pd.DataFrame:
        """Parse multiple addresses."""
        results = [self.parse(addr) for addr in addresses]
        return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Address Parser")

    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--column", required=True, help="Address column name")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file")

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    addresses = df[args.column].tolist()

    address_parser = AddressParser()
    parsed = address_parser.parse_batch(addresses)

    # Combine with original data
    result = pd.concat([df, parsed.drop(columns=['original'])], axis=1)

    result.to_csv(args.output, index=False)
    print(f"Parsed {len(addresses)} addresses")
    print(f"Output saved: {args.output}")


if __name__ == "__main__":
    main()
