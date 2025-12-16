#!/usr/bin/env python3
"""
Currency Converter - Convert between currencies.
"""

import argparse
import pandas as pd
from forex_python.converter import CurrencyRates, CurrencyCodes
from datetime import datetime


class CurrencyConverter:
    """Convert currencies."""

    def __init__(self):
        """Initialize converter."""
        self.rates = CurrencyRates()
        self.codes = CurrencyCodes()

    def convert(self, amount: float, from_currency: str, to_currency: str,
               date: datetime = None) -> float:
        """Convert currency amount."""
        try:
            if date:
                rate = self.rates.get_rate(from_currency, to_currency, date)
            else:
                rate = self.rates.get_rate(from_currency, to_currency)

            return amount * rate

        except Exception as e:
            print(f"Error converting {from_currency} to {to_currency}: {e}")
            return None

    def get_rate(self, from_currency: str, to_currency: str) -> float:
        """Get exchange rate."""
        try:
            return self.rates.get_rate(from_currency, to_currency)
        except:
            return None

    def get_symbol(self, currency_code: str) -> str:
        """Get currency symbol."""
        try:
            return self.codes.get_symbol(currency_code)
        except:
            return currency_code

    def convert_batch(self, amounts: list, from_currency: str, to_currency: str) -> list:
        """Convert multiple amounts."""
        return [self.convert(amt, from_currency, to_currency) for amt in amounts]


def main():
    parser = argparse.ArgumentParser(description="Currency Converter")

    # Single conversion
    parser.add_argument("--amount", type=float, help="Amount to convert")
    parser.add_argument("--from", dest='from_currency', help="From currency (e.g., USD)")
    parser.add_argument("--to", dest='to_currency', help="To currency (e.g., EUR)")

    # Batch conversion
    parser.add_argument("--file", help="Input CSV file")
    parser.add_argument("--from_col", help="From currency column")
    parser.add_argument("--to_col", help="To currency code")
    parser.add_argument("--output", "-o", help="Output file")

    args = parser.parse_args()

    converter = CurrencyConverter()

    if args.amount and args.from_currency and args.to_currency:
        # Single conversion
        result = converter.convert(args.amount, args.from_currency.upper(), args.to_currency.upper())

        if result:
            from_sym = converter.get_symbol(args.from_currency.upper())
            to_sym = converter.get_symbol(args.to_currency.upper())

            print(f"{from_sym}{args.amount:,.2f} {args.from_currency.upper()} = "
                  f"{to_sym}{result:,.2f} {args.to_currency.upper()}")

            rate = converter.get_rate(args.from_currency.upper(), args.to_currency.upper())
            print(f"Exchange rate: 1 {args.from_currency.upper()} = {rate:.4f} {args.to_currency.upper()}")

    elif args.file and args.from_col and args.to_col:
        # Batch conversion
        df = pd.read_csv(args.file)
        amounts = df[args.from_col].tolist()

        # Detect from currency (assume first row or from column name)
        from_curr = args.from_col.upper()[-3:] if len(args.from_col) >= 3 else 'USD'

        converted = converter.convert_batch(amounts, from_curr, args.to_col.upper())

        df[f'{args.to_col.upper()}_converted'] = converted

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Converted {len(amounts)} amounts")
            print(f"Output saved: {args.output}")


if __name__ == "__main__":
    main()
