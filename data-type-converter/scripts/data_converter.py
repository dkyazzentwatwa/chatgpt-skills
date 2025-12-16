#!/usr/bin/env python3
"""
Data Type Converter - Convert between JSON, CSV, XML, YAML, TOML.

Features:
- Multi-format conversion
- Nested data handling
- Batch processing
- Pretty output
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DataTypeConverter:
    """Convert between data formats."""

    SUPPORTED_FORMATS = {'json', 'csv', 'xml', 'yaml', 'toml'}

    def __init__(self):
        """Initialize converter."""
        pass

    def _detect_format(self, filepath: str) -> str:
        """Detect format from file extension."""
        ext = Path(filepath).suffix.lower().lstrip('.')
        if ext in self.SUPPORTED_FORMATS:
            return ext
        if ext == 'yml':
            return 'yaml'
        raise ValueError(f"Unknown format: {ext}")

    def load(self, filepath: str, format: Optional[str] = None) -> Any:
        """
        Load data from file.

        Args:
            filepath: Path to file
            format: Format (auto-detect if None)

        Returns:
            Loaded data
        """
        format = format or self._detect_format(filepath)
        text = Path(filepath).read_text()

        if format == 'json':
            return json.loads(text)

        elif format == 'csv':
            import io
            reader = csv.DictReader(io.StringIO(text))
            return list(reader)

        elif format == 'yaml':
            try:
                import yaml
                return yaml.safe_load(text)
            except ImportError:
                raise ImportError("PyYAML required. Install with: pip install pyyaml")

        elif format == 'toml':
            try:
                import toml
                return toml.loads(text)
            except ImportError:
                raise ImportError("toml required. Install with: pip install toml")

        elif format == 'xml':
            try:
                import xmltodict
                return xmltodict.parse(text)
            except ImportError:
                raise ImportError("xmltodict required. Install with: pip install xmltodict")

        raise ValueError(f"Unsupported format: {format}")

    def save(
        self,
        data: Any,
        filepath: str,
        format: Optional[str] = None,
        indent: int = 2,
        root: str = "root",
        flatten: bool = False
    ) -> str:
        """
        Save data to file.

        Args:
            data: Data to save
            filepath: Output path
            format: Format (auto-detect if None)
            indent: Indentation for pretty output
            root: Root element name for XML
            flatten: Flatten nested data for CSV

        Returns:
            Output path
        """
        format = format or self._detect_format(filepath)

        if format == 'json':
            output = json.dumps(data, indent=indent, default=str)

        elif format == 'csv':
            output = self._to_csv(data, flatten=flatten)

        elif format == 'yaml':
            try:
                import yaml
                output = yaml.dump(data, default_flow_style=False, sort_keys=False)
            except ImportError:
                raise ImportError("PyYAML required")

        elif format == 'toml':
            try:
                import toml
                output = toml.dumps(data)
            except ImportError:
                raise ImportError("toml required")

        elif format == 'xml':
            try:
                import xmltodict
                if not isinstance(data, dict):
                    data = {root: data}
                elif len(data) != 1:
                    data = {root: data}
                output = xmltodict.unparse(data, pretty=True)
            except ImportError:
                raise ImportError("xmltodict required")

        else:
            raise ValueError(f"Unsupported format: {format}")

        Path(filepath).write_text(output)
        return filepath

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self._flatten_dict(item, f"{new_key}[{i}]", sep=sep).items())
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, v))
        return dict(items)

    def _to_csv(self, data: Any, flatten: bool = False) -> str:
        """Convert data to CSV string."""
        import io

        # Ensure data is a list
        if isinstance(data, dict):
            if flatten:
                data = [self._flatten_dict(data)]
            else:
                data = [data]
        elif not isinstance(data, list):
            data = [data]

        # Flatten if requested
        if flatten and data:
            data = [self._flatten_dict(d) if isinstance(d, dict) else d for d in data]

        # Get all keys
        if data and isinstance(data[0], dict):
            keys = []
            for row in data:
                for k in row.keys():
                    if k not in keys:
                        keys.append(k)
        else:
            return str(data)

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

        return output.getvalue()

    def convert(
        self,
        input_path: str,
        output_path: str,
        indent: int = 2,
        root: str = "root",
        flatten: bool = False
    ) -> str:
        """
        Convert file from one format to another.

        Args:
            input_path: Input file path
            output_path: Output file path
            indent: Indentation for output
            root: XML root element
            flatten: Flatten nested data for CSV

        Returns:
            Output path
        """
        data = self.load(input_path)
        return self.save(data, output_path, indent=indent, root=root, flatten=flatten)

    def convert_string(
        self,
        data_str: str,
        from_format: str,
        to_format: str,
        indent: int = 2
    ) -> str:
        """
        Convert data string between formats.

        Args:
            data_str: Input data string
            from_format: Source format
            to_format: Target format
            indent: Indentation

        Returns:
            Converted string
        """
        # Parse input
        if from_format == 'json':
            data = json.loads(data_str)
        elif from_format == 'yaml':
            import yaml
            data = yaml.safe_load(data_str)
        elif from_format == 'toml':
            import toml
            data = toml.loads(data_str)
        elif from_format == 'xml':
            import xmltodict
            data = xmltodict.parse(data_str)
        else:
            raise ValueError(f"Unsupported source format: {from_format}")

        # Generate output
        if to_format == 'json':
            return json.dumps(data, indent=indent, default=str)
        elif to_format == 'yaml':
            import yaml
            return yaml.dump(data, default_flow_style=False)
        elif to_format == 'toml':
            import toml
            return toml.dumps(data)
        elif to_format == 'xml':
            import xmltodict
            if not isinstance(data, dict) or len(data) != 1:
                data = {'root': data}
            return xmltodict.unparse(data, pretty=True)
        else:
            raise ValueError(f"Unsupported target format: {to_format}")

    def batch_convert(
        self,
        input_dir: str,
        output_dir: str,
        output_format: str,
        flatten: bool = False
    ) -> List[str]:
        """
        Batch convert all files in directory.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            output_format: Target format
            flatten: Flatten nested data

        Returns:
            List of output paths
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        converted = []
        for ext in self.SUPPORTED_FORMATS:
            for filepath in input_path.glob(f'*.{ext}'):
                try:
                    output_file = output_path / f"{filepath.stem}.{output_format}"
                    self.convert(str(filepath), str(output_file), flatten=flatten)
                    converted.append(str(output_file))
                    print(f"Converted: {filepath.name} -> {output_file.name}")
                except Exception as e:
                    print(f"Failed: {filepath.name} - {e}")

        return converted


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Convert between data formats')
    parser.add_argument('--input', '-i', help='Input file')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--input-dir', help='Input directory for batch')
    parser.add_argument('--output-dir', help='Output directory for batch')
    parser.add_argument('--format', '-f', help='Output format')
    parser.add_argument('--flatten', action='store_true', help='Flatten nested data')
    parser.add_argument('--indent', type=int, default=2, help='Indentation')
    parser.add_argument('--root', default='root', help='XML root element')

    args = parser.parse_args()
    converter = DataTypeConverter()

    if args.input_dir and args.output_dir:
        if not args.format:
            parser.error("--format required for batch conversion")

        converted = converter.batch_convert(
            args.input_dir,
            args.output_dir,
            args.format,
            flatten=args.flatten
        )
        print(f"\nConverted {len(converted)} files")

    elif args.input and args.output:
        converter.convert(
            args.input,
            args.output,
            indent=args.indent,
            root=args.root,
            flatten=args.flatten
        )
        print(f"Converted: {args.input} -> {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
