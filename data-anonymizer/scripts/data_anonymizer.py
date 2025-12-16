#!/usr/bin/env python3
"""
Data Anonymizer - Detect and mask PII in text and data.

Features:
- PII detection (names, emails, phones, SSN, etc.)
- Multiple masking strategies
- CSV processing
- Reversible tokenization
"""

import argparse
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PIIMatch:
    """Represents a detected PII match."""
    type: str
    value: str
    start: int
    end: int
    masked: str


class DataAnonymizer:
    """Detect and mask PII in text and data."""

    # Regex patterns for PII detection
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        'ssn': r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        'date_of_birth': r'\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b',
        'zipcode': r'\b\d{5}(?:[-\s]\d{4})?\b',
    }

    # Mask labels
    LABELS = {
        'email': '[EMAIL]',
        'phone': '[PHONE]',
        'ssn': '[SSN]',
        'credit_card': '[CREDIT_CARD]',
        'ip_address': '[IP_ADDRESS]',
        'date_of_birth': '[DOB]',
        'zipcode': '[ZIPCODE]',
        'name': '[NAME]',
        'address': '[ADDRESS]',
    }

    def __init__(self, strategy: str = "mask", reversible: bool = False):
        """
        Initialize anonymizer.

        Args:
            strategy: Masking strategy (mask, redact, hash, fake)
            reversible: Enable token mapping for de-anonymization
        """
        self.strategy = strategy
        self.reversible = reversible
        self._mapping: Dict[str, str] = {}
        self._reverse_mapping: Dict[str, str] = {}
        self._faker = None

    def _get_faker(self):
        """Lazy load Faker."""
        if self._faker is None:
            try:
                from faker import Faker
                self._faker = Faker()
            except ImportError:
                raise ImportError("Faker is required for 'fake' strategy. Install with: pip install faker")
        return self._faker

    def _mask_value(self, pii_type: str, value: str) -> str:
        """Apply masking strategy to value."""
        if self.strategy == "mask":
            masked = self.LABELS.get(pii_type, f'[{pii_type.upper()}]')

        elif self.strategy == "redact":
            masked = '*' * len(value)

        elif self.strategy == "hash":
            hash_val = hashlib.sha256(value.encode()).hexdigest()[:12]
            masked = hash_val

        elif self.strategy == "fake":
            faker = self._get_faker()
            fake_generators = {
                'email': faker.email,
                'phone': faker.phone_number,
                'ssn': lambda: faker.ssn(),
                'name': faker.name,
                'credit_card': faker.credit_card_number,
                'ip_address': faker.ipv4,
                'date_of_birth': lambda: faker.date_of_birth().strftime('%m/%d/%Y'),
                'zipcode': faker.zipcode,
                'address': faker.address,
            }
            gen = fake_generators.get(pii_type)
            masked = gen() if gen else f'[{pii_type.upper()}]'

        else:
            masked = self.LABELS.get(pii_type, f'[{pii_type.upper()}]')

        # Store mapping if reversible
        if self.reversible:
            self._mapping[value] = masked
            self._reverse_mapping[masked] = value

        return masked

    def _detect_names(self, text: str) -> List[Tuple[int, int, str]]:
        """Detect names using simple heuristics."""
        # Simple pattern: Capitalized words that look like names
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        matches = []

        # Common non-name words to exclude
        exclude = {'The', 'This', 'That', 'These', 'Those', 'What', 'When', 'Where',
                   'Which', 'While', 'However', 'Therefore', 'Furthermore'}

        for match in re.finditer(pattern, text):
            name = match.group()
            words = name.split()
            if not any(w in exclude for w in words):
                matches.append((match.start(), match.end(), name))

        return matches

    def detect_pii(self, text: str, pii_types: Optional[List[str]] = None) -> List[PIIMatch]:
        """
        Detect PII in text.

        Args:
            text: Text to analyze
            pii_types: Specific PII types to detect (or all if None)

        Returns:
            List of PIIMatch objects
        """
        matches = []

        # Detect regex-based PII
        for pii_type, pattern in self.PATTERNS.items():
            if pii_types and pii_type not in pii_types:
                continue

            for match in re.finditer(pattern, text):
                masked = self._mask_value(pii_type, match.group())
                matches.append(PIIMatch(
                    type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    masked=masked
                ))

        # Detect names if requested
        if pii_types is None or 'name' in pii_types:
            for start, end, name in self._detect_names(text):
                # Check if not already matched
                if not any(m.start <= start < m.end for m in matches):
                    masked = self._mask_value('name', name)
                    matches.append(PIIMatch(
                        type='name',
                        value=name,
                        start=start,
                        end=end,
                        masked=masked
                    ))

        # Sort by position (reverse for replacement)
        matches.sort(key=lambda m: m.start, reverse=True)
        return matches

    def anonymize(
        self,
        text: str,
        pii_types: Optional[List[str]] = None,
        return_report: bool = False
    ):
        """
        Anonymize text by masking detected PII.

        Args:
            text: Text to anonymize
            pii_types: Specific PII types (or all)
            return_report: Also return detection report

        Returns:
            Anonymized text (and report if requested)
        """
        matches = self.detect_pii(text, pii_types)

        # Replace matches (already sorted in reverse)
        result = text
        for match in matches:
            result = result[:match.start] + match.masked + result[match.end:]

        if return_report:
            report = [{'type': m.type, 'original': m.value, 'masked': m.masked,
                       'position': (m.start, m.end)} for m in matches]
            return result, report

        return result

    def anonymize_csv(
        self,
        input_path: str,
        output_path: str,
        columns: Optional[List[str]] = None,
        column_strategies: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Anonymize CSV file.

        Args:
            input_path: Input CSV path
            output_path: Output CSV path
            columns: Columns to process (auto-detect if None)
            column_strategies: Strategy per column

        Returns:
            Output path
        """
        import pandas as pd

        df = pd.read_csv(input_path)

        # Auto-detect columns with potential PII
        if columns is None:
            pii_keywords = ['name', 'email', 'phone', 'ssn', 'address', 'dob', 'birth']
            columns = [col for col in df.columns
                       if any(kw in col.lower() for kw in pii_keywords)]

        # Process each column
        for col in columns:
            if col not in df.columns:
                continue

            # Get strategy for this column
            if column_strategies and col in column_strategies:
                original_strategy = self.strategy
                self.strategy = column_strategies[col]

            df[col] = df[col].astype(str).apply(self.anonymize)

            if column_strategies and col in column_strategies:
                self.strategy = original_strategy

        df.to_csv(output_path, index=False)
        return output_path

    def get_mapping(self) -> Dict[str, str]:
        """Get token mapping (for reversible mode)."""
        return self._mapping.copy()

    def save_mapping(self, path: str, encrypt: bool = False, password: str = None):
        """Save mapping to file."""
        data = json.dumps(self._mapping)

        if encrypt and password:
            # Simple XOR encryption (use proper encryption in production)
            key = password.encode()
            encrypted = bytes([d ^ key[i % len(key)] for i, d in enumerate(data.encode())])
            with open(path, 'wb') as f:
                f.write(encrypted)
        else:
            with open(path, 'w') as f:
                f.write(data)

    def load_mapping(self, path: str, password: str = None):
        """Load mapping from file."""
        with open(path, 'rb') as f:
            data = f.read()

        if password:
            key = password.encode()
            decrypted = bytes([d ^ key[i % len(key)] for i, d in enumerate(data)])
            data = decrypted.decode()
        else:
            data = data.decode()

        self._mapping = json.loads(data)
        self._reverse_mapping = {v: k for k, v in self._mapping.items()}

    def deanonymize(self, text: str) -> str:
        """Reverse anonymization using stored mapping."""
        if not self._reverse_mapping:
            raise ValueError("No mapping available. Load mapping first.")

        result = text
        for masked, original in self._reverse_mapping.items():
            result = result.replace(masked, original)

        return result

    def add_pattern(self, name: str, pattern: str, label: str = None):
        """Add custom PII pattern."""
        self.PATTERNS[name] = pattern
        self.LABELS[name] = label or f'[{name.upper()}]'


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Detect and mask PII in text')
    parser.add_argument('--input', '-i', help='Input file')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--text', '-t', help='Text to anonymize')
    parser.add_argument('--strategy', '-s', choices=['mask', 'redact', 'hash', 'fake'], default='mask')
    parser.add_argument('--types', nargs='+', help='PII types to detect')
    parser.add_argument('--columns', nargs='+', help='CSV columns to process')
    parser.add_argument('--report', '-r', help='Generate audit report')
    parser.add_argument('--reversible', action='store_true', help='Enable token mapping')

    args = parser.parse_args()
    anonymizer = DataAnonymizer(strategy=args.strategy, reversible=args.reversible)

    if args.text:
        result = anonymizer.anonymize(args.text, pii_types=args.types)
        print(result)

    elif args.input:
        input_path = Path(args.input)

        if input_path.suffix.lower() == '.csv':
            # CSV mode
            if not args.output:
                args.output = str(input_path.stem) + '_anonymized.csv'

            anonymizer.anonymize_csv(args.input, args.output, columns=args.columns)
            print(f"Anonymized CSV saved to: {args.output}")

        else:
            # Text file mode
            text = input_path.read_text()

            if args.report:
                result, report = anonymizer.anonymize(text, return_report=True)
                with open(args.report, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"Report saved to: {args.report}")
            else:
                result = anonymizer.anonymize(text, pii_types=args.types)

            if args.output:
                Path(args.output).write_text(result)
                print(f"Anonymized text saved to: {args.output}")
            else:
                print(result)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
