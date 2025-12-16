#!/usr/bin/env python3
"""
PDF Toolkit - Comprehensive PDF manipulation.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


class PDFToolkit:
    """PDF manipulation toolkit."""

    def __init__(self):
        """Initialize toolkit."""
        self.doc = None
        self.filepath = None

    def load(self, filepath: str) -> 'PDFToolkit':
        """Load PDF document."""
        self.filepath = filepath
        self.doc = fitz.open(filepath)
        return self

    def merge(self, pdf_files: List[str], output: str) -> str:
        """Merge multiple PDFs."""
        merger = PyPDF2.PdfMerger()

        for pdf in pdf_files:
            merger.append(pdf)

        merger.write(output)
        merger.close()

        return output

    def split(self, pages_per_chunk: int, output_dir: str) -> List[str]:
        """Split PDF into chunks."""
        if not self.doc:
            raise ValueError("No document loaded")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        total_pages = len(self.doc)
        chunks = []

        for i in range(0, total_pages, pages_per_chunk):
            end = min(i + pages_per_chunk, total_pages)

            # Create new PDF
            new_doc = fitz.open()
            new_doc.insert_pdf(self.doc, from_page=i, to_page=end-1)

            output_file = output_path / f"chunk_{i//pages_per_chunk + 1}.pdf"
            new_doc.save(str(output_file))
            new_doc.close()

            chunks.append(str(output_file))

        return chunks

    def extract_pages(self, pages: List[int], output: str) -> str:
        """Extract specific pages (0-indexed)."""
        if not self.doc:
            raise ValueError("No document loaded")

        new_doc = fitz.open()

        for page_num in pages:
            new_doc.insert_pdf(self.doc, from_page=page_num, to_page=page_num)

        new_doc.save(output)
        new_doc.close()

        return output

    def rotate(self, angle: int, pages: Optional[List[int]] = None,
              output: str = None) -> str:
        """Rotate pages by angle (90, 180, 270)."""
        if not self.doc:
            raise ValueError("No document loaded")

        if pages is None:
            pages = range(len(self.doc))

        for page_num in pages:
            page = self.doc[page_num]
            page.set_rotation(angle)

        output = output or self.filepath
        self.doc.save(output)

        return output

    def watermark(self, text: str, output: str, opacity: float = 0.3,
                 angle: int = 45, font_size: int = 60) -> str:
        """Add text watermark."""
        if not self.doc:
            raise ValueError("No document loaded")

        for page in self.doc:
            # Get page dimensions
            rect = page.rect
            center_x = rect.width / 2
            center_y = rect.height / 2

            # Add watermark text
            page.insert_text(
                (center_x, center_y),
                text,
                fontsize=font_size,
                rotate=angle,
                color=(0.7, 0.7, 0.7),
                overlay=True
            )

        self.doc.save(output)

        return output

    def compress(self, output: str, quality: int = 2) -> str:
        """Compress PDF."""
        if not self.doc:
            raise ValueError("No document loaded")

        # Deflate and compress images
        self.doc.save(output, garbage=quality, deflate=True, clean=True)

        return output

    def encrypt(self, output: str, user_password: str,
               owner_password: str = None) -> str:
        """Encrypt PDF with password."""
        if not self.doc:
            raise ValueError("No document loaded")

        owner_pwd = owner_password or user_password

        perm = fitz.PDF_PERM_PRINT | fitz.PDF_PERM_COPY | fitz.PDF_PERM_ANNOTATE

        self.doc.save(output, encryption=fitz.PDF_ENCRYPT_AES_256,
                     user_pw=user_password, owner_pw=owner_pwd,
                     permissions=perm)

        return output

    def get_info(self) -> dict:
        """Get PDF metadata."""
        if not self.doc:
            raise ValueError("No document loaded")

        return {
            'pages': len(self.doc),
            'title': self.doc.metadata.get('title', ''),
            'author': self.doc.metadata.get('author', ''),
            'subject': self.doc.metadata.get('subject', ''),
            'creator': self.doc.metadata.get('creator', '')
        }

    def close(self):
        """Close document."""
        if self.doc:
            self.doc.close()


def main():
    parser = argparse.ArgumentParser(description="PDF Toolkit")

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Merge
    merge_parser = subparsers.add_parser('merge', help='Merge PDFs')
    merge_parser.add_argument('files', nargs='+', help='PDF files to merge')
    merge_parser.add_argument('--output', '-o', required=True, help='Output file')

    # Split
    split_parser = subparsers.add_parser('split', help='Split PDF')
    split_parser.add_argument('input', help='Input PDF')
    split_parser.add_argument('--pages', type=int, required=True, help='Pages per chunk')
    split_parser.add_argument('--output', '-o', required=True, help='Output directory')

    # Rotate
    rotate_parser = subparsers.add_parser('rotate', help='Rotate pages')
    rotate_parser.add_argument('input', help='Input PDF')
    rotate_parser.add_argument('--angle', type=int, choices=[90, 180, 270],
                             required=True, help='Rotation angle')
    rotate_parser.add_argument('--output', '-o', required=True, help='Output file')

    # Watermark
    watermark_parser = subparsers.add_parser('watermark', help='Add watermark')
    watermark_parser.add_argument('input', help='Input PDF')
    watermark_parser.add_argument('--text', required=True, help='Watermark text')
    watermark_parser.add_argument('--output', '-o', required=True, help='Output file')

    args = parser.parse_args()

    toolkit = PDFToolkit()

    if args.command == 'merge':
        output = toolkit.merge(args.files, args.output)
        print(f"Merged {len(args.files)} PDFs → {output}")

    elif args.command == 'split':
        toolkit.load(args.input)
        chunks = toolkit.split(args.pages, args.output)
        print(f"Split into {len(chunks)} chunks in {args.output}/")

    elif args.command == 'rotate':
        toolkit.load(args.input)
        output = toolkit.rotate(args.angle, output=args.output)
        print(f"Rotated {args.angle}° → {output}")

    elif args.command == 'watermark':
        toolkit.load(args.input)
        output = toolkit.watermark(args.text, args.output)
        print(f"Watermarked → {output}")


if __name__ == "__main__":
    main()
