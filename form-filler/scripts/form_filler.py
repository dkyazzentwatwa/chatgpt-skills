#!/usr/bin/env python3
"""
Form Filler - Fill PDF forms programmatically.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class FormFiller:
    """Fill PDF forms with data."""

    def __init__(self):
        """Initialize the form filler."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) is required for form filling")

        self.doc = None
        self.filepath = None
        self.mapping = {}

    def load(self, filepath: str) -> 'FormFiller':
        """Load a PDF form."""
        self.filepath = filepath
        self.doc = fitz.open(filepath)
        return self

    def list_fields(self) -> List[Dict]:
        """List all form fields."""
        if not self.doc:
            raise ValueError("No document loaded")

        fields = []

        for page_num, page in enumerate(self.doc):
            widgets = page.widgets()
            if widgets:
                for widget in widgets:
                    field_info = {
                        "name": widget.field_name,
                        "type": self._get_field_type(widget.field_type),
                        "page": page_num,
                        "value": widget.field_value,
                        "rect": list(widget.rect)
                    }

                    # Get options for choice fields
                    if widget.field_type in [fitz.PDF_WIDGET_TYPE_LISTBOX,
                                             fitz.PDF_WIDGET_TYPE_COMBOBOX]:
                        field_info["options"] = widget.choice_values

                    fields.append(field_info)

        return fields

    def _get_field_type(self, field_type: int) -> str:
        """Convert field type constant to string."""
        type_map = {
            fitz.PDF_WIDGET_TYPE_TEXT: "text",
            fitz.PDF_WIDGET_TYPE_CHECKBOX: "checkbox",
            fitz.PDF_WIDGET_TYPE_RADIOBUTTON: "radio",
            fitz.PDF_WIDGET_TYPE_LISTBOX: "listbox",
            fitz.PDF_WIDGET_TYPE_COMBOBOX: "dropdown",
            fitz.PDF_WIDGET_TYPE_BUTTON: "button",
            fitz.PDF_WIDGET_TYPE_SIGNATURE: "signature"
        }
        return type_map.get(field_type, "unknown")

    def get_field_info(self, field_name: str) -> Optional[Dict]:
        """Get information about a specific field."""
        fields = self.list_fields()
        for field in fields:
            if field["name"] == field_name:
                return field
        return None

    def get_field_value(self, field_name: str) -> Any:
        """Get current value of a field."""
        field_info = self.get_field_info(field_name)
        if field_info:
            return field_info.get("value")
        return None

    def set_mapping(self, mapping: Dict[str, str]) -> 'FormFiller':
        """Set data key to field name mapping."""
        self.mapping = mapping
        return self

    def fill(self, data: Dict) -> 'FormFiller':
        """Fill form with data dictionary."""
        if not self.doc:
            raise ValueError("No document loaded")

        for key, value in data.items():
            # Apply mapping if exists
            field_name = self.mapping.get(key, key)
            self.fill_field(field_name, value)

        return self

    def fill_field(self, name: str, value: Any) -> 'FormFiller':
        """Fill a specific form field."""
        if not self.doc:
            raise ValueError("No document loaded")

        for page in self.doc:
            widgets = page.widgets()
            if widgets:
                for widget in widgets:
                    if widget.field_name == name:
                        self._set_widget_value(widget, value)

        return self

    def _set_widget_value(self, widget, value: Any):
        """Set value on a widget based on its type."""
        field_type = widget.field_type

        if field_type == fitz.PDF_WIDGET_TYPE_TEXT:
            widget.field_value = str(value) if value is not None else ""
            widget.update()

        elif field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
            # Checkboxes: True/False or "Yes"/"Off"
            if isinstance(value, bool):
                widget.field_value = "Yes" if value else "Off"
            else:
                widget.field_value = str(value)
            widget.update()

        elif field_type == fitz.PDF_WIDGET_TYPE_RADIOBUTTON:
            widget.field_value = str(value)
            widget.update()

        elif field_type in [fitz.PDF_WIDGET_TYPE_LISTBOX, fitz.PDF_WIDGET_TYPE_COMBOBOX]:
            widget.field_value = str(value)
            widget.update()

    def fill_from_json(self, filepath: str) -> 'FormFiller':
        """Fill form from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        return self.fill(data)

    def fill_from_csv_row(self, row: Dict) -> 'FormFiller':
        """Fill form from a CSV row dictionary."""
        return self.fill(row)

    def flatten(self) -> 'FormFiller':
        """Flatten form (make fields non-editable)."""
        if not self.doc:
            raise ValueError("No document loaded")

        for page in self.doc:
            # Get all annotations (widgets are annotations)
            annots = page.annots()
            if annots:
                for annot in annots:
                    # Set the annotation to be read-only
                    annot.set_flags(fitz.ANNOT_BF_ReadOnly)

        return self

    def save(self, filepath: str, flatten: bool = False) -> str:
        """Save filled form to file."""
        if not self.doc:
            raise ValueError("No document loaded")

        if flatten:
            self.flatten()

        self.doc.save(filepath, garbage=4, deflate=True)
        return filepath

    def close(self):
        """Close the document."""
        if self.doc:
            self.doc.close()

    def batch_fill(self, input_form: str, data_file: str,
                  output_dir: str, name_field: str = None) -> List[str]:
        """
        Fill multiple copies of form from data file.

        Args:
            input_form: Path to PDF form template
            data_file: Path to JSON or CSV data file
            output_dir: Output directory for filled forms
            name_field: Field to use for naming output files

        Returns:
            List of generated file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load data
        if data_file.endswith('.json'):
            with open(data_file) as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
        else:
            df = pd.read_csv(data_file)
            data = df.to_dict(orient='records')

        generated = []
        base_name = Path(input_form).stem

        for i, row in enumerate(data):
            # Load fresh copy of form
            self.load(input_form)

            # Fill with data
            self.fill(row)

            # Determine output filename
            if name_field and name_field in row:
                filename = f"{base_name}_{row[name_field]}.pdf"
            else:
                filename = f"{base_name}_{i}.pdf"

            # Clean filename
            filename = "".join(c if c.isalnum() or c in '._-' else '_' for c in filename)

            output_file = output_path / filename
            self.save(str(output_file))
            generated.append(str(output_file))

            self.close()
            print(f"Generated: {filename}")

        return generated


def main():
    parser = argparse.ArgumentParser(description="Form Filler")

    parser.add_argument("--input", "-i", required=True, help="Input PDF form")
    parser.add_argument("--output", "-o", help="Output PDF file")
    parser.add_argument("--data", "-d", help="Data file (JSON)")

    parser.add_argument("--list-fields", action="store_true", help="List form fields")

    parser.add_argument("--batch", help="Batch fill from CSV/JSON data file")
    parser.add_argument("--output-dir", help="Output directory for batch")
    parser.add_argument("--name-field", help="Field to use for output filenames")

    parser.add_argument("--mapping", help="Field mapping JSON file")
    parser.add_argument("--flatten", action="store_true", help="Flatten output PDF")

    args = parser.parse_args()

    filler = FormFiller()

    # List fields mode
    if args.list_fields:
        filler.load(args.input)
        fields = filler.list_fields()

        print(f"Form: {args.input}")
        print(f"Total fields: {len(fields)}\n")

        for field in fields:
            print(f"Name: {field['name']}")
            print(f"  Type: {field['type']}")
            print(f"  Page: {field['page'] + 1}")
            if field.get('value'):
                print(f"  Current value: {field['value']}")
            if field.get('options'):
                print(f"  Options: {field['options']}")
            print()

        filler.close()
        return

    # Batch mode
    if args.batch:
        output_dir = args.output_dir or "filled_forms"
        generated = filler.batch_fill(
            args.input,
            args.batch,
            output_dir,
            name_field=args.name_field
        )
        print(f"\nGenerated {len(generated)} filled forms in: {output_dir}")
        return

    # Single form filling
    if args.data and args.output:
        filler.load(args.input)

        # Load mapping if provided
        if args.mapping:
            with open(args.mapping) as f:
                filler.set_mapping(json.load(f))

        # Fill from data
        filler.fill_from_json(args.data)

        # Save
        filler.save(args.output, flatten=args.flatten)
        print(f"Filled form saved: {args.output}")

        # Show summary
        with open(args.data) as f:
            data = json.load(f)
        print(f"Filled {len(data)} fields")

        filler.close()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
