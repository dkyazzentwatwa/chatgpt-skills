#!/usr/bin/env python3
"""
Receipt Scanner - Extract data from receipt images using OCR.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pytesseract
import cv2
import numpy as np
from PIL import Image
import pandas as pd


class ReceiptScanner:
    """Scan and extract data from receipts."""

    def __init__(self):
        """Initialize scanner."""
        self.raw_text = ""
        self.data = {}

    def scan(self, filepath: str, preprocess: bool = True) -> 'ReceiptScanner':
        """Scan receipt image."""
        img = cv2.imread(filepath)

        if preprocess:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Denoise
            denoised = cv2.fastNlMeansDenoising(thresh)

            # OCR
            self.raw_text = pytesseract.image_to_string(denoised)
        else:
            self.raw_text = pytesseract.image_to_string(filepath)

        # Extract data
        self.extract_data()

        return self

    def extract_data(self):
        """Extract structured data from text."""
        lines = self.raw_text.split('\n')

        # Extract vendor (usually first line)
        self.data['vendor'] = lines[0].strip() if lines else "Unknown"

        # Extract date
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        for line in lines:
            match = re.search(date_pattern, line)
            if match:
                self.data['date'] = match.group()
                break

        # Extract total
        total_patterns = [
            r'total[:\s]*\$?\s*(\d+\.\d{2})',
            r'amount[:\s]*\$?\s*(\d+\.\d{2})',
            r'\$\s*(\d+\.\d{2})'
        ]

        for pattern in total_patterns:
            match = re.search(pattern, self.raw_text, re.IGNORECASE)
            if match:
                self.data['total'] = float(match.group(1))
                break

        # Extract line items
        items = []
        item_pattern = r'(.*?)\s+\$?\s*(\d+\.\d{2})'

        for line in lines:
            match = re.search(item_pattern, line)
            if match and len(match.group(1)) > 2:
                item_name = match.group(1).strip()
                amount = float(match.group(2))

                if amount < 1000:  # Filter out likely totals
                    items.append({'name': item_name, 'amount': amount})

        self.data['items'] = items

    def get_data(self) -> Dict:
        """Get extracted data."""
        return self.data

    def to_json(self, output: str) -> str:
        """Export to JSON."""
        with open(output, 'w') as f:
            json.dump(self.data, f, indent=2)
        return output


def main():
    parser = argparse.ArgumentParser(description="Receipt Scanner")

    parser.add_argument("--input", "-i", required=True, help="Receipt image")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument("--no-preprocess", action="store_true",
                       help="Skip image preprocessing")

    args = parser.parse_args()

    scanner = ReceiptScanner()
    scanner.scan(args.input, preprocess=not args.no_preprocess)

    data = scanner.get_data()
    print(f"Vendor: {data.get('vendor', 'Unknown')}")
    print(f"Date: {data.get('date', 'Not found')}")
    print(f"Total: ${data.get('total', 0):.2f}")
    print(f"Items: {len(data.get('items', []))}")

    scanner.to_json(args.output)
    print(f"\nData saved: {args.output}")


if __name__ == "__main__":
    main()
