#!/usr/bin/env python3
"""
Business Card Scanner - Extract contact info from cards.
"""

import argparse
import json
import re

import pytesseract
import cv2
import numpy as np
from PIL import Image


class BusinessCardScanner:
    """Scan business cards."""

    def __init__(self):
        """Initialize scanner."""
        self.raw_text = ""
        self.data = {}

    def scan(self, filepath: str) -> 'BusinessCardScanner':
        """Scan business card image."""
        img = cv2.imread(filepath)

        # Preprocess
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # OCR
        self.raw_text = pytesseract.image_to_string(thresh)

        # Extract data
        self.extract_contact_info()

        return self

    def extract_contact_info(self):
        """Extract contact information."""
        lines = [line.strip() for line in self.raw_text.split('\n') if line.strip()]

        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, self.raw_text)
        self.data['email'] = emails[0] if emails else None

        # Extract phone
        phone_patterns = [
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
        ]
        for pattern in phone_patterns:
            phones = re.findall(pattern, self.raw_text)
            if phones:
                self.data['phone'] = phones[0]
                break

        # Extract name (usually first line)
        self.data['name'] = lines[0] if lines else None

        # Extract company (heuristic: look for Inc, LLC, Ltd)
        company_keywords = ['Inc', 'LLC', 'Ltd', 'Corp', 'Company']
        for line in lines:
            if any(kw in line for kw in company_keywords):
                self.data['company'] = line
                break

        # Extract website
        url_pattern = r'www\.[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
        urls = re.findall(url_pattern, self.raw_text, re.IGNORECASE)
        self.data['website'] = urls[0] if urls else None

    def get_data(self) -> dict:
        """Get extracted data."""
        return self.data

    def to_json(self, output: str) -> str:
        """Export to JSON."""
        with open(output, 'w') as f:
            json.dump(self.data, f, indent=2)
        return output


def main():
    parser = argparse.ArgumentParser(description="Business Card Scanner")

    parser.add_argument("--input", "-i", required=True, help="Business card image")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")

    args = parser.parse_args()

    scanner = BusinessCardScanner()
    scanner.scan(args.input)

    data = scanner.get_data()

    print("Extracted Contact Information:")
    print(f"  Name: {data.get('name', 'Not found')}")
    print(f"  Company: {data.get('company', 'Not found')}")
    print(f"  Email: {data.get('email', 'Not found')}")
    print(f"  Phone: {data.get('phone', 'Not found')}")
    print(f"  Website: {data.get('website', 'Not found')}")

    scanner.to_json(args.output)
    print(f"\nData saved: {args.output}")


if __name__ == "__main__":
    main()
