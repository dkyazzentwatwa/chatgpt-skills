#!/usr/bin/env python3
"""
JSON Schema Validator - Validate JSON data against JSON Schema.

Features:
- Schema validation (Draft 4, 6, 7)
- Detailed error messages
- Schema generation from samples
- Batch validation
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class JSONValidator:
    """Validate JSON against JSON Schema."""

    def __init__(self, draft: str = "draft7"):
        """
        Initialize validator.

        Args:
            draft: JSON Schema draft version (draft4, draft6, draft7)
        """
        self._jsonschema = None
        self._load_jsonschema()
        self.draft = draft

    def _load_jsonschema(self):
        """Load jsonschema library."""
        try:
            import jsonschema
            self._jsonschema = jsonschema
        except ImportError:
            raise ImportError("jsonschema required. Install with: pip install jsonschema")

    def _get_validator_class(self):
        """Get validator class for draft version."""
        if self.draft == 'draft4':
            return self._jsonschema.Draft4Validator
        elif self.draft == 'draft6':
            return self._jsonschema.Draft6Validator
        else:  # draft7
            return self._jsonschema.Draft7Validator

    def validate(self, data: Any, schema: Dict) -> Dict:
        """
        Validate data against schema.

        Args:
            data: JSON data to validate
            schema: JSON Schema

        Returns:
            Validation result with errors
        """
        validator_cls = self._get_validator_class()

        # Check schema first
        try:
            validator_cls.check_schema(schema)
        except self._jsonschema.SchemaError as e:
            return {
                'valid': False,
                'errors': [{
                    'message': f"Invalid schema: {e.message}",
                    'path': '$',
                    'schema_path': str(e.schema_path)
                }]
            }

        validator = validator_cls(schema)
        errors = list(validator.iter_errors(data))

        if not errors:
            return {'valid': True, 'errors': [], 'path': '$'}

        error_details = []
        for error in errors:
            path = '$' + ''.join(
                f'.{p}' if isinstance(p, str) else f'[{p}]'
                for p in error.absolute_path
            )
            schema_path = '.'.join(str(p) for p in error.schema_path)

            error_details.append({
                'message': error.message,
                'path': path,
                'schema_path': schema_path,
                'validator': error.validator,
                'value': error.instance if not isinstance(error.instance, (dict, list)) else type(error.instance).__name__
            })

        return {
            'valid': False,
            'errors': error_details,
            'path': '$'
        }

    def validate_file(self, data_path: str, schema_path: str) -> Dict:
        """
        Validate JSON file against schema file.

        Args:
            data_path: Path to JSON data file
            schema_path: Path to schema file

        Returns:
            Validation result
        """
        with open(data_path) as f:
            data = json.load(f)

        with open(schema_path) as f:
            schema = json.load(f)

        result = self.validate(data, schema)
        result['data_file'] = data_path
        result['schema_file'] = schema_path
        return result

    def validate_batch(
        self,
        data_files: List[str],
        schema: Union[Dict, str]
    ) -> List[Dict]:
        """
        Validate multiple files against schema.

        Args:
            data_files: List of JSON file paths
            schema: Schema dict or path to schema file

        Returns:
            List of validation results
        """
        if isinstance(schema, str):
            with open(schema) as f:
                schema = json.load(f)

        results = []
        for filepath in data_files:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                result = self.validate(data, schema)
                result['file'] = filepath
            except json.JSONDecodeError as e:
                result = {
                    'valid': False,
                    'file': filepath,
                    'errors': [{'message': f"Invalid JSON: {e.msg}"}]
                }
            except Exception as e:
                result = {
                    'valid': False,
                    'file': filepath,
                    'errors': [{'message': str(e)}]
                }
            results.append(result)

        return results

    def check_schema(self, schema: Dict) -> Dict:
        """
        Validate that a schema is well-formed.

        Args:
            schema: JSON Schema to check

        Returns:
            Result with valid flag and errors
        """
        validator_cls = self._get_validator_class()
        try:
            validator_cls.check_schema(schema)
            return {'valid': True, 'errors': []}
        except self._jsonschema.SchemaError as e:
            return {
                'valid': False,
                'errors': [{
                    'message': e.message,
                    'path': str(e.schema_path)
                }]
            }

    def generate_schema(self, data: Any, require_all: bool = False) -> Dict:
        """
        Generate schema from sample data.

        Args:
            data: Sample JSON data
            require_all: Make all properties required

        Returns:
            Generated JSON Schema
        """
        schema = self._infer_schema(data)
        if require_all and schema.get('type') == 'object':
            schema['required'] = list(schema.get('properties', {}).keys())
        return schema

    def _infer_schema(self, data: Any) -> Dict:
        """Recursively infer schema from data."""
        if data is None:
            return {'type': 'null'}

        if isinstance(data, bool):
            return {'type': 'boolean'}

        if isinstance(data, int):
            return {'type': 'integer'}

        if isinstance(data, float):
            return {'type': 'number'}

        if isinstance(data, str):
            schema = {'type': 'string'}
            # Detect common formats
            if '@' in data and '.' in data:
                schema['format'] = 'email'
            elif data.startswith('http'):
                schema['format'] = 'uri'
            elif len(data) == 10 and data[4] == '-' and data[7] == '-':
                schema['format'] = 'date'
            return schema

        if isinstance(data, list):
            if not data:
                return {'type': 'array', 'items': {}}

            # Infer items schema from first element
            items_schema = self._infer_schema(data[0])
            return {
                'type': 'array',
                'items': items_schema
            }

        if isinstance(data, dict):
            properties = {}
            for key, value in data.items():
                properties[key] = self._infer_schema(value)

            return {
                'type': 'object',
                'properties': properties
            }

        return {}


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Validate JSON against JSON Schema')
    parser.add_argument('--data', '-d', help='JSON data file to validate')
    parser.add_argument('--schema', '-s', help='JSON Schema file or inline JSON')
    parser.add_argument('--data-dir', help='Directory of JSON files to validate')
    parser.add_argument('--generate', '-g', help='Generate schema from sample JSON file')
    parser.add_argument('--output', '-o', help='Output file for generated schema')
    parser.add_argument('--draft', default='draft7', choices=['draft4', 'draft6', 'draft7'])
    parser.add_argument('--require-all', action='store_true', help='Make all properties required in generated schema')
    parser.add_argument('--check-schema', help='Check if schema is valid')

    args = parser.parse_args()
    validator = JSONValidator(draft=args.draft)

    if args.generate:
        # Generate schema from sample
        with open(args.generate) as f:
            data = json.load(f)
        schema = validator.generate_schema(data, require_all=args.require_all)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(schema, f, indent=2)
            print(f"Schema saved to: {args.output}")
        else:
            print(json.dumps(schema, indent=2))

    elif args.check_schema:
        # Validate schema itself
        with open(args.check_schema) as f:
            schema = json.load(f)
        result = validator.check_schema(schema)
        if result['valid']:
            print("Schema is valid")
        else:
            print("Schema is invalid:")
            for error in result['errors']:
                print(f"  - {error['message']}")

    elif args.data_dir and args.schema:
        # Batch validation
        data_files = list(Path(args.data_dir).glob('*.json'))
        results = validator.validate_batch([str(f) for f in data_files], args.schema)

        valid_count = sum(1 for r in results if r['valid'])
        print(f"Results: {valid_count}/{len(results)} valid\n")

        for result in results:
            status = "VALID" if result['valid'] else "INVALID"
            print(f"{result['file']}: {status}")
            if not result['valid']:
                for error in result['errors'][:3]:  # Show first 3 errors
                    print(f"  - {error.get('path', '')}: {error['message']}")

    elif args.data and args.schema:
        # Single file validation
        # Schema can be file or inline JSON
        try:
            with open(args.schema) as f:
                schema = json.load(f)
        except (FileNotFoundError, IsADirectoryError):
            schema = json.loads(args.schema)

        with open(args.data) as f:
            data = json.load(f)

        result = validator.validate(data, schema)

        if result['valid']:
            print("VALID")
        else:
            print(f"INVALID - {len(result['errors'])} error(s):\n")
            for error in result['errors']:
                print(f"  Path: {error['path']}")
                print(f"  Error: {error['message']}")
                print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
