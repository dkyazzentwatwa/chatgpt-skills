#!/usr/bin/env python3
"""
Barcode Generator - Generate various barcode formats.

Features:
- Multiple formats (Code128, EAN13, UPC, Code39, etc.)
- Customization (size, colors, text)
- Batch generation from CSV
- Check digit calculation
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional


class BarcodeGenerator:
    """Generate barcodes in various formats."""

    FORMATS = {
        'code128': 'code128',
        'code39': 'code39',
        'ean13': 'ean13',
        'ean8': 'ean8',
        'upca': 'upca',
        'isbn13': 'isbn13',
        'isbn10': 'isbn10',
        'itf': 'itf',
        'pzn': 'pzn',
    }

    def __init__(self):
        """Initialize generator."""
        self._barcode = None
        self._load_barcode()

    def _load_barcode(self):
        """Load python-barcode library."""
        try:
            import barcode
            self._barcode = barcode
        except ImportError:
            raise ImportError("python-barcode required. Install with: pip install python-barcode")

    def generate(
        self,
        code: str,
        format: str = "code128",
        output: str = "barcode.png",
        width: int = 300,
        height: int = 150,
        show_text: bool = True,
        font_size: int = 10,
        foreground: str = "black",
        background: str = "white",
        auto_check_digit: bool = False
    ) -> str:
        """
        Generate a barcode.

        Args:
            code: Data to encode
            format: Barcode format
            output: Output file path
            width: Image width (approximate)
            height: Image height (approximate)
            show_text: Show code text below barcode
            font_size: Text size
            foreground: Bar color
            background: Background color
            auto_check_digit: Calculate and append check digit

        Returns:
            Path to generated file
        """
        format = format.lower()
        if format not in self.FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported: {list(self.FORMATS.keys())}")

        # Auto check digit for specific formats
        if auto_check_digit and format in ('ean13', 'ean8', 'upca'):
            check = self.calculate_check_digit(code, format)
            if check is not None:
                code = code + str(check)

        # Get barcode class
        barcode_class = self._barcode.get_barcode_class(self.FORMATS[format])

        # Writer options
        output_path = Path(output)
        ext = output_path.suffix.lower()

        if ext == '.svg':
            from barcode.writer import SVGWriter
            writer = SVGWriter()
        else:
            try:
                from barcode.writer import ImageWriter
                writer = ImageWriter()
            except ImportError:
                raise ImportError("Pillow required for PNG output. Install with: pip install Pillow")

        # Generate barcode
        bc = barcode_class(code, writer=writer)

        # Writer options
        options = {
            'module_width': 0.2,
            'module_height': 15.0,
            'quiet_zone': 6.5,
            'font_size': font_size,
            'text_distance': 5.0,
            'write_text': show_text,
        }

        if ext != '.svg':
            options['foreground'] = foreground
            options['background'] = background

        # Save (barcode library adds extension automatically)
        filename = str(output_path.with_suffix(''))
        full_path = bc.save(filename, options=options)

        return full_path

    def validate(self, code: str, format: str) -> bool:
        """
        Validate barcode format and check digit.

        Args:
            code: Code to validate
            format: Barcode format

        Returns:
            True if valid
        """
        format = format.lower()

        try:
            if format == 'ean13':
                if len(code) != 13 or not code.isdigit():
                    return False
                return self._verify_ean_check(code)

            elif format == 'ean8':
                if len(code) != 8 or not code.isdigit():
                    return False
                return self._verify_ean_check(code)

            elif format == 'upca':
                if len(code) != 12 or not code.isdigit():
                    return False
                return self._verify_ean_check(code)

            elif format in ('code128', 'code39'):
                # These formats are more flexible
                return len(code) > 0

            return True

        except Exception:
            return False

    def _verify_ean_check(self, code: str) -> bool:
        """Verify EAN/UPC check digit."""
        digits = [int(d) for d in code]
        check = digits[-1]

        # Calculate expected check digit
        total = 0
        for i, d in enumerate(digits[:-1]):
            weight = 1 if i % 2 == 0 else 3
            total += d * weight

        expected = (10 - (total % 10)) % 10
        return check == expected

    def calculate_check_digit(self, code: str, format: str) -> Optional[int]:
        """
        Calculate check digit for code.

        Args:
            code: Code without check digit
            format: Barcode format

        Returns:
            Check digit or None
        """
        format = format.lower()

        if format == 'ean13' and len(code) == 12:
            return self._calc_ean_check(code)
        elif format == 'ean8' and len(code) == 7:
            return self._calc_ean_check(code)
        elif format == 'upca' and len(code) == 11:
            return self._calc_ean_check(code)

        return None

    def _calc_ean_check(self, code: str) -> int:
        """Calculate EAN/UPC check digit."""
        digits = [int(d) for d in code]
        total = 0
        for i, d in enumerate(digits):
            weight = 1 if i % 2 == 0 else 3
            total += d * weight
        return (10 - (total % 10)) % 10

    def batch_generate(
        self,
        csv_file: str,
        code_column: str = "code",
        format: str = "code128",
        output_dir: str = ".",
        filename_column: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate barcodes from CSV file.

        Args:
            csv_file: Path to CSV file
            code_column: Column containing codes
            format: Barcode format
            output_dir: Output directory
            filename_column: Column for filenames (or use code)
            **kwargs: Additional options for generate()

        Returns:
            List of generated file paths
        """
        import csv

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated = []

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get(code_column)
                if not code:
                    continue

                # Determine filename
                if filename_column and filename_column in row:
                    filename = row[filename_column]
                    # Clean filename
                    filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
                else:
                    filename = code

                output_file = output_path / f"{filename}.png"

                try:
                    path = self.generate(code, format=format, output=str(output_file), **kwargs)
                    generated.append(path)
                    print(f"Generated: {path}")
                except Exception as e:
                    print(f"Failed for {code}: {e}")

        return generated

    def batch_generate_list(
        self,
        codes: List[str],
        format: str = "code128",
        output_dir: str = ".",
        **kwargs
    ) -> List[str]:
        """
        Generate barcodes from list of codes.

        Args:
            codes: List of codes
            format: Barcode format
            output_dir: Output directory
            **kwargs: Additional options

        Returns:
            List of generated file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated = []
        for i, code in enumerate(codes):
            output_file = output_path / f"barcode_{i+1:04d}.png"
            try:
                path = self.generate(code, format=format, output=str(output_file), **kwargs)
                generated.append(path)
            except Exception as e:
                print(f"Failed for {code}: {e}")

        return generated


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Generate barcodes')
    parser.add_argument('--code', '-c', help='Code to encode')
    parser.add_argument('--format', '-f', default='code128',
                        choices=list(BarcodeGenerator.FORMATS.keys()),
                        help='Barcode format')
    parser.add_argument('--output', '-o', default='barcode.png', help='Output file')
    parser.add_argument('--width', type=int, default=300, help='Width')
    parser.add_argument('--height', type=int, default=150, help='Height')
    parser.add_argument('--no-text', action='store_true', help='Hide code text')
    parser.add_argument('--batch', help='CSV file for batch generation')
    parser.add_argument('--column', default='code', help='Code column in CSV')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--validate', help='Validate code')
    parser.add_argument('--check-digit', action='store_true', help='Auto-calculate check digit')

    args = parser.parse_args()
    gen = BarcodeGenerator()

    if args.validate:
        is_valid = gen.validate(args.validate, args.format)
        print(f"{args.validate}: {'VALID' if is_valid else 'INVALID'}")
        if not is_valid and args.format in ('ean13', 'ean8', 'upca'):
            # Show what check digit should be
            code_without_check = args.validate[:-1] if len(args.validate) in (8, 12, 13) else args.validate
            check = gen.calculate_check_digit(code_without_check, args.format)
            if check is not None:
                print(f"Expected check digit: {check}")

    elif args.batch:
        generated = gen.batch_generate(
            args.batch,
            code_column=args.column,
            format=args.format,
            output_dir=args.output_dir,
            show_text=not args.no_text
        )
        print(f"\nGenerated {len(generated)} barcodes")

    elif args.code:
        path = gen.generate(
            args.code,
            format=args.format,
            output=args.output,
            width=args.width,
            height=args.height,
            show_text=not args.no_text,
            auto_check_digit=args.check_digit
        )
        print(f"Generated: {path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
