#!/usr/bin/env python3
"""
Table Extractor - Extract tables from PDFs and images.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from PIL import Image

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import pytesseract
    import cv2
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class TableExtractor:
    """Extract tables from PDFs and images."""

    def __init__(self):
        """Initialize the extractor."""
        self.pdf = None
        self.image = None
        self.filepath = None
        self.pages = None
        self.ocr_enabled = False
        self.ocr_lang = "eng"

    def load_pdf(self, filepath: str, pages: List[int] = None) -> 'TableExtractor':
        """Load a PDF document."""
        if not PDF_AVAILABLE:
            raise ImportError("pdfplumber is required for PDF extraction")

        self.filepath = filepath
        self.pdf = pdfplumber.open(filepath)
        self.pages = pages
        self.image = None
        return self

    def load_image(self, filepath: str) -> 'TableExtractor':
        """Load an image file."""
        self.filepath = filepath
        self.image = Image.open(filepath)
        self.pdf = None
        return self

    def set_ocr(self, enabled: bool = True, lang: str = "eng") -> 'TableExtractor':
        """Enable or disable OCR for scanned documents."""
        if enabled and not OCR_AVAILABLE:
            raise ImportError("pytesseract and opencv-python are required for OCR")
        self.ocr_enabled = enabled
        self.ocr_lang = lang
        return self

    def get_page_count(self) -> int:
        """Get number of pages in PDF."""
        if self.pdf:
            return len(self.pdf.pages)
        return 1 if self.image else 0

    def detect_tables(self, page: int = 0) -> List[Dict]:
        """Detect tables on a page without extracting."""
        tables_info = []

        if self.pdf:
            pdf_page = self.pdf.pages[page]
            tables = pdf_page.find_tables()

            for i, table in enumerate(tables):
                bbox = table.bbox
                # Estimate rows/cols from table structure
                tables_info.append({
                    "index": i,
                    "bbox": bbox,
                    "page": page
                })

        elif self.image:
            # For images, detect table regions using line detection
            tables_info = self._detect_image_tables()

        return tables_info

    def _detect_image_tables(self) -> List[Dict]:
        """Detect table regions in image using line detection."""
        if not OCR_AVAILABLE:
            return [{"index": 0, "bbox": (0, 0, self.image.width, self.image.height)}]

        img_array = np.array(self.image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Detect horizontal and vertical lines
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)

        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

        # Combine lines
        table_mask = cv2.add(horizontal_lines, vertical_lines)

        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tables = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 100:  # Filter small contours
                tables.append({
                    "index": i,
                    "bbox": (x, y, x + w, y + h)
                })

        if not tables:
            # Return full image as single table
            tables = [{"index": 0, "bbox": (0, 0, self.image.width, self.image.height)}]

        return tables

    def extract_table(self, page: int = 0, table_index: int = 0) -> pd.DataFrame:
        """Extract a single table from specified page."""
        if self.pdf:
            return self._extract_pdf_table(page, table_index)
        elif self.image:
            return self._extract_image_table(table_index)
        else:
            raise ValueError("No document loaded")

    def extract_all(self) -> List[pd.DataFrame]:
        """Extract all tables from the document."""
        tables = []

        if self.pdf:
            page_range = range(len(self.pdf.pages))
            if self.pages:
                page_range = self.pages

            for page_num in page_range:
                page_tables = self.extract_page(page_num)
                tables.extend(page_tables)

        elif self.image:
            table = self._extract_image_table(0)
            if not table.empty:
                tables.append(table)

        return tables

    def extract_page(self, page: int) -> List[pd.DataFrame]:
        """Extract all tables from a specific page."""
        tables = []

        if self.pdf:
            pdf_page = self.pdf.pages[page]
            page_tables = pdf_page.extract_tables()

            for table_data in page_tables:
                if table_data:
                    # Convert to DataFrame
                    df = self._table_to_dataframe(table_data)
                    if not df.empty:
                        tables.append(df)

        return tables

    def _extract_pdf_table(self, page: int, table_index: int) -> pd.DataFrame:
        """Extract specific table from PDF page."""
        pdf_page = self.pdf.pages[page]
        page_tables = pdf_page.extract_tables()

        if table_index < len(page_tables):
            return self._table_to_dataframe(page_tables[table_index])

        return pd.DataFrame()

    def _table_to_dataframe(self, table_data: List[List]) -> pd.DataFrame:
        """Convert raw table data to DataFrame."""
        if not table_data:
            return pd.DataFrame()

        # Clean data
        cleaned = []
        for row in table_data:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    cleaned_row.append(str(cell).strip())
            cleaned_row.append(cleaned_row)

        # Use first row as header if it looks like one
        if cleaned:
            df = pd.DataFrame(cleaned[1:], columns=cleaned[0])
        else:
            df = pd.DataFrame(cleaned)

        return df

    def _extract_image_table(self, table_index: int) -> pd.DataFrame:
        """Extract table from image using OCR."""
        if not OCR_AVAILABLE:
            raise ImportError("pytesseract required for image table extraction")

        img_array = np.array(self.image.convert('RGB'))

        # Preprocess image
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Get table regions
        tables = self._detect_image_tables()
        if table_index >= len(tables):
            return pd.DataFrame()

        bbox = tables[table_index]["bbox"]
        x1, y1, x2, y2 = map(int, bbox)

        # Crop to table region
        table_img = thresh[y1:y2, x1:x2]

        # OCR with table structure detection
        custom_config = r'--oem 3 --psm 6'
        ocr_data = pytesseract.image_to_data(
            table_img,
            lang=self.ocr_lang,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )

        # Group text by lines and columns
        rows_data = self._group_ocr_by_rows(ocr_data)

        if not rows_data:
            return pd.DataFrame()

        # Convert to DataFrame
        max_cols = max(len(row) for row in rows_data)
        for row in rows_data:
            while len(row) < max_cols:
                row.append("")

        if rows_data:
            df = pd.DataFrame(rows_data[1:], columns=rows_data[0]) if len(rows_data) > 1 else pd.DataFrame(rows_data)
        else:
            df = pd.DataFrame()

        return df

    def _group_ocr_by_rows(self, ocr_data: Dict) -> List[List[str]]:
        """Group OCR output by rows based on position."""
        # Get word positions
        words = []
        n_boxes = len(ocr_data['text'])

        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 30:  # Confidence threshold
                text = ocr_data['text'][i].strip()
                if text:
                    words.append({
                        'text': text,
                        'left': ocr_data['left'][i],
                        'top': ocr_data['top'][i],
                        'height': ocr_data['height'][i]
                    })

        if not words:
            return []

        # Sort by vertical position
        words.sort(key=lambda w: w['top'])

        # Group by rows (similar y position)
        rows = []
        current_row = []
        current_y = None
        row_threshold = 10

        for word in words:
            if current_y is None:
                current_y = word['top']
                current_row = [word]
            elif abs(word['top'] - current_y) < row_threshold:
                current_row.append(word)
            else:
                # Sort row by x position
                current_row.sort(key=lambda w: w['left'])
                rows.append([w['text'] for w in current_row])
                current_row = [word]
                current_y = word['top']

        if current_row:
            current_row.sort(key=lambda w: w['left'])
            rows.append([w['text'] for w in current_row])

        return rows

    def to_csv(self, tables: List[pd.DataFrame], output_dir: str) -> List[str]:
        """Export tables to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = []
        for i, table in enumerate(tables):
            filepath = output_path / f"table_{i}.csv"
            table.to_csv(filepath, index=False)
            files.append(str(filepath))

        return files

    def to_excel(self, tables: List[pd.DataFrame], output: str) -> str:
        """Export tables to Excel with multiple sheets."""
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for i, table in enumerate(tables):
                table.to_excel(writer, sheet_name=f"Table_{i+1}", index=False)

        return output

    def to_json(self, tables: List[pd.DataFrame], output: str) -> str:
        """Export tables to JSON."""
        data = [table.to_dict(orient='records') for table in tables]

        with open(output, 'w') as f:
            json.dump(data, f, indent=2)

        return output

    def get_table_count(self) -> int:
        """Get total number of tables in document."""
        count = 0

        if self.pdf:
            for page in self.pdf.pages:
                count += len(page.find_tables())
        elif self.image:
            count = len(self._detect_image_tables())

        return count

    def close(self):
        """Close opened resources."""
        if self.pdf:
            self.pdf.close()


def parse_pages(pages_str: str) -> List[int]:
    """Parse page range string like '1-3,5,7-9'."""
    pages = []
    for part in pages_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            pages.extend(range(start - 1, end))  # Convert to 0-indexed
        else:
            pages.append(int(part) - 1)
    return pages


def main():
    parser = argparse.ArgumentParser(description="Table Extractor")

    parser.add_argument("--input", "-i", required=True, help="Input PDF or image")
    parser.add_argument("--output", "-o", required=True, help="Output file or directory")

    parser.add_argument("--pages", help="Page range (e.g., 1-3,5)")
    parser.add_argument("--format", choices=['csv', 'xlsx', 'json'],
                       default='csv', help="Output format")

    parser.add_argument("--ocr", action="store_true", help="Enable OCR")
    parser.add_argument("--lang", default="eng", help="OCR language")

    parser.add_argument("--list", action="store_true", help="List tables without extracting")

    args = parser.parse_args()

    extractor = TableExtractor()

    # Enable OCR if requested
    if args.ocr:
        extractor.set_ocr(enabled=True, lang=args.lang)

    # Load document
    if args.input.lower().endswith('.pdf'):
        pages = parse_pages(args.pages) if args.pages else None
        extractor.load_pdf(args.input, pages=pages)
    else:
        extractor.load_image(args.input)

    # List mode
    if args.list:
        page_count = extractor.get_page_count()
        print(f"Document: {args.input}")
        print(f"Pages: {page_count}")

        total_tables = 0
        for page in range(page_count):
            tables = extractor.detect_tables(page)
            if tables:
                print(f"\nPage {page + 1}: {len(tables)} table(s)")
                for t in tables:
                    print(f"  Table {t['index']}: bbox={t['bbox']}")
            total_tables += len(tables)

        print(f"\nTotal tables: {total_tables}")
        return

    # Extract tables
    print(f"Extracting tables from: {args.input}")
    tables = extractor.extract_all()

    if not tables:
        print("No tables found")
        return

    print(f"Found {len(tables)} table(s)")

    # Export
    if args.format == 'csv':
        files = extractor.to_csv(tables, args.output)
        print(f"Exported {len(files)} CSV files to: {args.output}")
    elif args.format == 'xlsx':
        extractor.to_excel(tables, args.output)
        print(f"Exported to Excel: {args.output}")
    elif args.format == 'json':
        extractor.to_json(tables, args.output)
        print(f"Exported to JSON: {args.output}")

    # Show preview of first table
    if tables:
        print(f"\nFirst table preview ({len(tables[0])} rows):")
        print(tables[0].head().to_string())

    extractor.close()


if __name__ == "__main__":
    main()
