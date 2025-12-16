#!/usr/bin/env python3
"""
API Response Mocker - Generate realistic mock API responses.

Features:
- Schema-based generation
- Faker integration for realistic data
- Nested objects and arrays
- Custom value generators
- Reproducible with seeds
"""

import argparse
import json
import random
from typing import Any, Dict, List, Optional, Union


class APIMocker:
    """Generate mock API responses with fake data."""

    def __init__(self, locale: str = "en_US", seed: Optional[int] = None):
        """
        Initialize mocker.

        Args:
            locale: Faker locale
            seed: Random seed for reproducibility
        """
        self._faker_lib = None
        self._load_faker()

        self.faker = self._faker_lib.Faker(locale)
        if seed is not None:
            self.set_seed(seed)

    def _load_faker(self):
        """Load Faker library."""
        try:
            import faker
            self._faker_lib = faker
        except ImportError:
            raise ImportError("faker required. Install with: pip install faker")

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._faker_lib.Faker.seed(seed)
        random.seed(seed)

    def get_faker(self):
        """Get the Faker instance for direct use."""
        return self.faker

    def generate(self, schema: Dict) -> Dict:
        """
        Generate data based on schema.

        Args:
            schema: Schema dict defining structure

        Returns:
            Generated data dict
        """
        return self._process_schema(schema)

    def generate_list(self, schema: Dict, count: int = 10) -> List[Dict]:
        """
        Generate list of objects.

        Args:
            schema: Schema for each item
            count: Number of items

        Returns:
            List of generated objects
        """
        return [self._process_schema(schema) for _ in range(count)]

    def _process_schema(self, schema: Any) -> Any:
        """Process schema recursively."""
        if isinstance(schema, dict):
            # Check for special directives
            if '_array' in schema:
                count = schema.get('_count', 5)
                item_schema = schema.get('_item', 'word')
                return [self._process_schema(item_schema) for _ in range(count)]

            if '_choice' in schema:
                return random.choice(schema['_choice'])

            if '_range' in schema:
                min_val, max_val = schema['_range']
                return random.randint(min_val, max_val)

            if '_float' in schema:
                config = schema['_float']
                min_val = config.get('min', 0)
                max_val = config.get('max', 100)
                decimals = config.get('decimals', 2)
                return round(random.uniform(min_val, max_val), decimals)

            if '_constant' in schema:
                return schema['_constant']

            # Regular object - process each key
            return {key: self._process_schema(value) for key, value in schema.items()}

        if isinstance(schema, list):
            return [self._process_schema(item) for item in schema]

        if isinstance(schema, str):
            return self._generate_value(schema)

        # Return as-is (numbers, booleans, None)
        return schema

    def _generate_value(self, faker_type: str) -> Any:
        """Generate a single value using Faker."""
        # Handle special types
        if faker_type == 'price':
            return round(random.uniform(0.99, 999.99), 2)

        if faker_type == 'rating':
            return round(random.uniform(1, 5), 1)

        if faker_type == 'percentage':
            return random.randint(0, 100)

        # Try to get Faker method
        if hasattr(self.faker, faker_type):
            method = getattr(self.faker, faker_type)
            try:
                return method()
            except TypeError:
                # Some methods need arguments
                return str(method)

        # Fallback to string
        return faker_type

    def from_schema_file(self, filepath: str) -> Dict:
        """
        Load schema from file and generate.

        Args:
            filepath: Path to JSON schema file

        Returns:
            Generated data
        """
        with open(filepath) as f:
            schema = json.load(f)
        return self.generate(schema)

    def save(
        self,
        data: Any,
        filepath: str,
        format: str = "json",
        indent: int = 2
    ) -> str:
        """
        Save generated data to file.

        Args:
            data: Data to save
            filepath: Output path
            format: Output format (json, xml)
            indent: JSON indentation

        Returns:
            Path to saved file
        """
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=indent, default=str)
        elif format == 'xml':
            xml_str = self._to_xml(data)
            with open(filepath, 'w') as f:
                f.write(xml_str)
        else:
            raise ValueError(f"Unknown format: {format}")

        return filepath

    def _to_xml(self, data: Any, root: str = "response") -> str:
        """Convert data to XML string."""
        def convert(obj, name):
            if isinstance(obj, dict):
                children = ''.join(convert(v, k) for k, v in obj.items())
                return f"<{name}>{children}</{name}>"
            elif isinstance(obj, list):
                items = ''.join(convert(item, 'item') for item in obj)
                return f"<{name}>{items}</{name}>"
            else:
                return f"<{name}>{obj}</{name}>"

        return f'<?xml version="1.0"?>\n{convert(data, root)}'


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Generate mock API responses')
    parser.add_argument('--schema', '-s', help='Schema file (JSON)')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--count', '-n', type=int, default=1, help='Number of items to generate')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--locale', default='en_US', help='Faker locale')
    parser.add_argument('--format', default='json', choices=['json', 'xml'])
    parser.add_argument('--preview', action='store_true', help='Preview without saving')
    parser.add_argument('--indent', type=int, default=2, help='JSON indentation')

    args = parser.parse_args()
    mocker = APIMocker(locale=args.locale, seed=args.seed)

    if args.schema:
        with open(args.schema) as f:
            schema = json.load(f)

        if args.count > 1:
            data = mocker.generate_list(schema, args.count)
        else:
            data = mocker.generate(schema)

        if args.preview or not args.output:
            print(json.dumps(data, indent=args.indent, default=str))
        else:
            mocker.save(data, args.output, format=args.format, indent=args.indent)
            print(f"Generated data saved to: {args.output}")

    else:
        # Demo mode - generate sample user
        demo_schema = {
            "id": "uuid",
            "username": "user_name",
            "email": "email",
            "profile": {
                "first_name": "first_name",
                "last_name": "last_name",
                "phone": "phone_number",
                "company": "company"
            },
            "address": {
                "street": "street_address",
                "city": "city",
                "state": "state_abbr",
                "zip": "zipcode"
            },
            "created_at": "iso8601",
            "is_active": "boolean"
        }

        print("Demo - Generated User:")
        print(json.dumps(mocker.generate(demo_schema), indent=2, default=str))
        print("\n---")
        print("Usage: python api_mocker.py --schema schema.json --output data.json")


if __name__ == "__main__":
    main()
