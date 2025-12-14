#!/usr/bin/env python3
"""
OCR Document Processor - Extract text from images and PDFs
Supports multiple languages, structured output, and batch processing.
"""

import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Try to import optional dependencies
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class OCRError(Exception):
    """Custom exception for OCR errors."""
    pass


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    psm: int = 3  # Page segmentation mode
    oem: int = 3  # OCR engine mode
    dpi: int = 300
    timeout: int = 30
    min_confidence: int = 60
    preserve_layout: bool = True

    def update(self, settings: Dict[str, Any]) -> None:
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)


class OCRProcessor:
    """
    Main class for OCR processing.

    Extracts text from images and PDFs using Tesseract OCR.
    """

    def __init__(
        self,
        source: Union[str, Path, Image.Image, bytes],
        lang: str = 'eng'
    ):
        """
        Initialize OCR processor.

        Args:
            source: Image path, PDF path, PIL Image, or bytes
            lang: OCR language code (e.g., 'eng', 'deu', 'eng+fra')
        """
        if not HAS_TESSERACT:
            raise ImportError("pytesseract is required. Install with: pip install pytesseract")

        self.config = OCRConfig()
        self.lang = lang
        self._images: List[Image.Image] = []
        self._preprocessed = False
        self._source_path: Optional[Path] = None

        # Load source
        if isinstance(source, (str, Path)):
            self._source_path = Path(source)
            self._load_file(self._source_path)
        elif isinstance(source, Image.Image):
            self._images = [source.copy()]
        elif isinstance(source, bytes):
            self._images = [Image.open(io.BytesIO(source))]
        else:
            raise OCRError(f"Unsupported source type: {type(source)}")

    def _load_file(self, path: Path) -> None:
        """Load image or PDF file."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()

        if suffix == '.pdf':
            self._load_pdf(path)
        elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif']:
            self._images = [Image.open(path)]
        else:
            raise OCRError(f"Unsupported file format: {suffix}")

    def _load_pdf(self, path: Path) -> None:
        """Load PDF and convert to images."""
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install PyMuPDF")

        doc = fitz.open(path)
        self._images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render at higher DPI for better OCR
            mat = fitz.Matrix(self.config.dpi / 72, self.config.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            self._images.append(img)

        doc.close()

    def preprocess(
        self,
        deskew: bool = False,
        denoise: bool = False,
        threshold: bool = False,
        threshold_method: str = 'otsu',
        contrast: float = 1.0,
        sharpen: float = 0,
        scale: float = 1.0,
        remove_shadows: bool = False
    ) -> 'OCRProcessor':
        """
        Preprocess images for better OCR accuracy.

        Args:
            deskew: Correct rotation
            denoise: Remove noise
            threshold: Convert to binary
            threshold_method: 'otsu', 'adaptive', 'simple'
            contrast: Contrast adjustment (1.0 = no change)
            sharpen: Sharpening factor
            scale: Upscale factor for small text
            remove_shadows: Remove shadow artifacts

        Returns:
            self for chaining
        """
        if not HAS_CV2:
            raise ImportError("OpenCV is required for preprocessing. Install with: pip install opencv-python")

        processed = []

        for img in self._images:
            # Convert to numpy array
            img_array = np.array(img)

            # Convert to grayscale if color
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Scale up if needed
            if scale > 1.0:
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # Denoise
            if denoise:
                gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            # Contrast adjustment
            if contrast != 1.0:
                gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=0)

            # Remove shadows
            if remove_shadows:
                dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
                bg = cv2.medianBlur(dilated, 21)
                gray = 255 - cv2.absdiff(gray, bg)

            # Threshold
            if threshold:
                if threshold_method == 'otsu':
                    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                elif threshold_method == 'adaptive':
                    gray = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                    )
                else:  # simple
                    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Sharpen
            if sharpen > 0:
                kernel = np.array([[-1, -1, -1], [-1, 9 + sharpen, -1], [-1, -1, -1]])
                gray = cv2.filter2D(gray, -1, kernel)

            # Deskew
            if deskew:
                gray = self._deskew_image(gray)

            # Convert back to PIL Image
            processed.append(Image.fromarray(gray))

        self._images = processed
        self._preprocessed = True
        return self

    def _deskew_image(self, img: np.ndarray) -> np.ndarray:
        """Deskew a grayscale image."""
        # Find all non-zero points
        coords = np.column_stack(np.where(img > 0))

        if len(coords) == 0:
            return img

        # Get rotation angle
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        # Rotate image
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    def extract_text(self, pages: Optional[List[int]] = None) -> str:
        """
        Extract text from all pages/images.

        Args:
            pages: Specific pages to extract (1-indexed), None for all

        Returns:
            Extracted text
        """
        texts = []

        images_to_process = self._images
        if pages:
            images_to_process = [self._images[p - 1] for p in pages if 0 < p <= len(self._images)]

        for img in images_to_process:
            custom_config = f'--psm {self.config.psm} --oem {self.config.oem}'
            text = pytesseract.image_to_string(
                img,
                lang=self.lang,
                config=custom_config,
                timeout=self.config.timeout
            )
            texts.append(text)

        return '\n\n'.join(texts)

    def extract_structured(self) -> Dict[str, Any]:
        """
        Extract text with structure information.

        Returns:
            Dict containing text, blocks, lines, words, confidence
        """
        all_blocks = []
        all_lines = []
        all_words = []
        full_text = []
        total_conf = []

        for page_idx, img in enumerate(self._images):
            custom_config = f'--psm {self.config.psm} --oem {self.config.oem}'
            data = pytesseract.image_to_data(
                img,
                lang=self.lang,
                config=custom_config,
                output_type=pytesseract.Output.DICT,
                timeout=self.config.timeout
            )

            # Process data
            current_block = None
            current_line = None

            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])

                if not text:
                    continue

                word = {
                    'text': text,
                    'confidence': conf,
                    'bbox': [
                        data['left'][i],
                        data['top'][i],
                        data['width'][i],
                        data['height'][i]
                    ],
                    'page': page_idx + 1,
                    'block_num': data['block_num'][i],
                    'line_num': data['line_num'][i],
                    'word_num': data['word_num'][i]
                }
                all_words.append(word)

                if conf >= 0:
                    total_conf.append(conf)

            # Get full text for this page
            page_text = pytesseract.image_to_string(
                img,
                lang=self.lang,
                config=custom_config,
                timeout=self.config.timeout
            )
            full_text.append(page_text)

        # Calculate average confidence
        avg_conf = sum(total_conf) / len(total_conf) if total_conf else 0

        # Group words into blocks and lines
        blocks = self._group_into_blocks(all_words)
        lines = self._group_into_lines(all_words)

        return {
            'text': '\n\n'.join(full_text),
            'blocks': blocks,
            'lines': lines,
            'words': all_words,
            'confidence': round(avg_conf, 1),
            'language': self.lang,
            'pages': len(self._images),
            'source': str(self._source_path) if self._source_path else None,
            'preprocessed': self._preprocessed
        }

    def _group_into_blocks(self, words: List[Dict]) -> List[Dict]:
        """Group words into text blocks."""
        blocks = {}
        for word in words:
            key = (word['page'], word['block_num'])
            if key not in blocks:
                blocks[key] = {
                    'page': word['page'],
                    'block_num': word['block_num'],
                    'words': [],
                    'text': ''
                }
            blocks[key]['words'].append(word)

        # Build block text
        for block in blocks.values():
            block['text'] = ' '.join(w['text'] for w in block['words'])
            # Calculate bounding box
            if block['words']:
                min_x = min(w['bbox'][0] for w in block['words'])
                min_y = min(w['bbox'][1] for w in block['words'])
                max_x = max(w['bbox'][0] + w['bbox'][2] for w in block['words'])
                max_y = max(w['bbox'][1] + w['bbox'][3] for w in block['words'])
                block['bbox'] = [min_x, min_y, max_x - min_x, max_y - min_y]

        return list(blocks.values())

    def _group_into_lines(self, words: List[Dict]) -> List[Dict]:
        """Group words into lines."""
        lines = {}
        for word in words:
            key = (word['page'], word['block_num'], word['line_num'])
            if key not in lines:
                lines[key] = {
                    'page': word['page'],
                    'block_num': word['block_num'],
                    'line_num': word['line_num'],
                    'words': [],
                    'text': ''
                }
            lines[key]['words'].append(word)

        for line in lines.values():
            line['text'] = ' '.join(w['text'] for w in line['words'])

        return list(lines.values())

    def extract_by_page(self) -> Dict[int, str]:
        """Extract text page by page."""
        results = {}
        for idx, img in enumerate(self._images):
            custom_config = f'--psm {self.config.psm} --oem {self.config.oem}'
            text = pytesseract.image_to_string(
                img,
                lang=self.lang,
                config=custom_config,
                timeout=self.config.timeout
            )
            results[idx + 1] = text
        return results

    def extract_tables(self) -> List[List[List[str]]]:
        """
        Attempt to extract tabular data.

        Returns:
            List of tables, each table is a list of rows
        """
        tables = []

        for img in self._images:
            # Use table detection mode
            custom_config = f'--psm 6 --oem {self.config.oem}'
            data = pytesseract.image_to_data(
                img,
                lang=self.lang,
                config=custom_config,
                output_type=pytesseract.Output.DICT,
                timeout=self.config.timeout
            )

            # Simple heuristic: group by Y position for rows
            lines_by_y = {}
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if not text:
                    continue

                y = data['top'][i]
                # Round Y to group nearby items
                y_bucket = (y // 20) * 20

                if y_bucket not in lines_by_y:
                    lines_by_y[y_bucket] = []

                lines_by_y[y_bucket].append({
                    'text': text,
                    'x': data['left'][i]
                })

            # Sort each row by X position
            if lines_by_y:
                table = []
                for y in sorted(lines_by_y.keys()):
                    row = sorted(lines_by_y[y], key=lambda x: x['x'])
                    table.append([item['text'] for item in row])
                tables.append(table)

        return tables

    def export_markdown(self, filepath: Union[str, Path]) -> str:
        """Export extracted text as Markdown."""
        result = self.extract_structured()
        filepath = Path(filepath)

        lines = [
            f"# OCR Extracted Document",
            f"",
            f"**Source:** {result['source'] or 'Unknown'}",
            f"**Pages:** {result['pages']}",
            f"**Language:** {result['language']}",
            f"**Confidence:** {result['confidence']}%",
            f"**Extracted:** {datetime.now().isoformat()}",
            f"",
            "---",
            "",
        ]

        # Add text content
        lines.append(result['text'])

        content = '\n'.join(lines)
        filepath.write_text(content, encoding='utf-8')
        return str(filepath)

    def export_json(self, filepath: Union[str, Path]) -> str:
        """Export structured data as JSON."""
        result = self.extract_structured()
        result['extracted_at'] = datetime.now().isoformat()

        filepath = Path(filepath)
        filepath.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        return str(filepath)

    def export_html(self, filepath: Union[str, Path]) -> str:
        """Export as formatted HTML."""
        result = self.extract_structured()
        filepath = Path(filepath)

        html = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<meta charset='utf-8'>",
            "<title>OCR Extracted Document</title>",
            "<style>",
            "body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }",
            "h1 { color: #333; }",
            ".meta { background: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }",
            ".meta span { display: inline-block; margin-right: 20px; }",
            ".content { white-space: pre-wrap; line-height: 1.6; }",
            ".low-conf { background: #fff3cd; }",
            "</style>",
            "</head><body>",
            "<h1>OCR Extracted Document</h1>",
            "<div class='meta'>",
            f"<span><strong>Source:</strong> {result['source'] or 'Unknown'}</span>",
            f"<span><strong>Pages:</strong> {result['pages']}</span>",
            f"<span><strong>Confidence:</strong> {result['confidence']}%</span>",
            "</div>",
            "<div class='content'>",
            result['text'].replace('<', '&lt;').replace('>', '&gt;'),
            "</div>",
            "</body></html>"
        ]

        content = '\n'.join(html)
        filepath.write_text(content, encoding='utf-8')
        return str(filepath)

    def export_searchable_pdf(self, filepath: Union[str, Path]) -> str:
        """Create searchable PDF with OCR text layer."""
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF required for PDF export")

        filepath = Path(filepath)
        doc = fitz.open()

        for img in self._images:
            # Get text and positions
            custom_config = f'--psm {self.config.psm} --oem {self.config.oem}'
            data = pytesseract.image_to_data(
                img,
                lang=self.lang,
                config=custom_config,
                output_type=pytesseract.Output.DICT,
                timeout=self.config.timeout
            )

            # Create PDF page from image
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            img_doc = fitz.open(stream=img_bytes, filetype="png")
            rect = img_doc[0].rect
            page = doc.new_page(width=rect.width, height=rect.height)
            page.insert_image(rect, stream=img_bytes.getvalue())

            # Add invisible text layer
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if not text:
                    continue

                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]

                # Add text annotation
                text_rect = fitz.Rect(x, y, x + w, y + h)
                page.insert_text(
                    (x, y + h * 0.8),  # Baseline position
                    text,
                    fontsize=max(6, h * 0.7),
                    render_mode=3  # Invisible
                )

            img_doc.close()

        doc.save(filepath)
        doc.close()
        return str(filepath)

    def export_tables_csv(self, output_dir: Union[str, Path]) -> List[str]:
        """Export detected tables as CSV files."""
        import csv

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tables = self.extract_tables()
        exported = []

        for idx, table in enumerate(tables):
            filepath = output_dir / f"table_{idx + 1}.csv"
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(table)
            exported.append(str(filepath))

        return exported

    def export_tables_json(self, filepath: Union[str, Path]) -> str:
        """Export detected tables as JSON."""
        tables = self.extract_tables()
        filepath = Path(filepath)
        filepath.write_text(json.dumps(tables, indent=2, ensure_ascii=False), encoding='utf-8')
        return str(filepath)

    def parse_receipt(self) -> Dict[str, Any]:
        """Parse receipt/invoice structure."""
        text = self.extract_text()

        receipt = {
            'vendor': None,
            'date': None,
            'items': [],
            'subtotal': None,
            'tax': None,
            'total': None,
            'raw_text': text
        }

        lines = text.split('\n')

        # Try to find vendor (usually first non-empty line)
        for line in lines[:5]:
            if line.strip():
                receipt['vendor'] = line.strip()
                break

        # Find date patterns
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{1,2}-\d{1,2}-\d{2,4}',
            r'\d{4}-\d{2}-\d{2}',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                receipt['date'] = match.group()
                break

        # Find total (look for common patterns)
        total_patterns = [
            r'total[:\s]*\$?([\d,]+\.?\d*)',
            r'amount[:\s]*\$?([\d,]+\.?\d*)',
            r'grand total[:\s]*\$?([\d,]+\.?\d*)',
        ]
        for pattern in total_patterns:
            match = re.search(pattern, text.lower())
            if match:
                receipt['total'] = match.group(1)
                break

        # Find tax
        tax_patterns = [
            r'tax[:\s]*\$?([\d,]+\.?\d*)',
            r'vat[:\s]*\$?([\d,]+\.?\d*)',
        ]
        for pattern in tax_patterns:
            match = re.search(pattern, text.lower())
            if match:
                receipt['tax'] = match.group(1)
                break

        return receipt

    def parse_business_card(self) -> Dict[str, Any]:
        """Parse business card information."""
        text = self.extract_text()

        contact = {
            'name': None,
            'title': None,
            'company': None,
            'email': [],
            'phone': [],
            'address': None,
            'website': [],
            'raw_text': text
        }

        # Find emails
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        contact['email'] = re.findall(email_pattern, text)

        # Find phone numbers
        phone_pattern = r'[\+\d][\d\s\-\(\)]{8,}'
        contact['phone'] = re.findall(phone_pattern, text)

        # Find websites
        url_pattern = r'(?:www\.)?[\w\-]+\.(?:com|org|net|io|co)(?:/[\w\-]*)?'
        contact['website'] = re.findall(url_pattern, text.lower())

        # First line often contains name
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            # Name is usually the first or second line
            for line in lines[:3]:
                # Skip if it looks like email/phone/url
                if '@' in line or 'www' in line.lower() or re.search(r'\d{5,}', line):
                    continue
                # Likely a name if it's 2-4 words
                words = line.split()
                if 1 < len(words) <= 4 and all(w[0].isupper() for w in words if w):
                    contact['name'] = line
                    break

        return contact


# ==================== BATCH PROCESSING ====================

def batch_ocr(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    output_format: str = 'text',
    lang: str = 'eng',
    recursive: bool = False,
    preprocess: bool = False
) -> Dict[str, Any]:
    """
    Process multiple documents with OCR.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        output_format: 'text', 'markdown', 'json', 'html'
        lang: OCR language
        recursive: Include subdirectories
        preprocess: Apply preprocessing

    Returns:
        Results with success/failed counts
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find files
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp', '*.pdf']
    files = []
    for ext in extensions:
        if recursive:
            files.extend(input_dir.rglob(ext))
        else:
            files.extend(input_dir.glob(ext))

    results = {'success': 0, 'failed': 0, 'errors': []}

    for filepath in files:
        try:
            processor = OCRProcessor(filepath, lang=lang)

            if preprocess:
                processor.preprocess(denoise=True, threshold=True)

            # Determine output path
            rel_path = filepath.relative_to(input_dir) if recursive else filepath.name
            out_name = Path(rel_path).stem

            if output_format == 'text':
                out_path = output_dir / f"{out_name}.txt"
                out_path.write_text(processor.extract_text(), encoding='utf-8')
            elif output_format == 'markdown':
                out_path = output_dir / f"{out_name}.md"
                processor.export_markdown(out_path)
            elif output_format == 'json':
                out_path = output_dir / f"{out_name}.json"
                processor.export_json(out_path)
            elif output_format == 'html':
                out_path = output_dir / f"{out_name}.html"
                processor.export_html(out_path)

            results['success'] += 1

        except Exception as e:
            results['failed'] += 1
            results['errors'].append({'file': str(filepath), 'error': str(e)})

    return results


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='OCR Document Processor')
    parser.add_argument('input', help='Input image, PDF, or directory')
    parser.add_argument('-o', '--output', help='Output path')
    parser.add_argument('--format', choices=['text', 'markdown', 'json', 'html'],
                        default='text', help='Output format')
    parser.add_argument('--lang', default='eng', help='OCR language code')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    parser.add_argument('--preprocess', action='store_true', help='Apply preprocessing')
    parser.add_argument('--deskew', action='store_true', help='Deskew images')
    parser.add_argument('--denoise', action='store_true', help='Denoise images')

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.batch or input_path.is_dir():
        # Batch mode
        output_dir = args.output or 'ocr_output'
        results = batch_ocr(
            input_path,
            output_dir,
            output_format=args.format,
            lang=args.lang,
            preprocess=args.preprocess
        )
        print(f"Processed: {results['success']} files")
        print(f"Failed: {results['failed']} files")
    else:
        # Single file mode
        processor = OCRProcessor(input_path, lang=args.lang)

        if args.preprocess or args.deskew or args.denoise:
            processor.preprocess(deskew=args.deskew, denoise=args.denoise)

        if args.format == 'text':
            text = processor.extract_text()
            if args.output:
                Path(args.output).write_text(text, encoding='utf-8')
            else:
                print(text)
        elif args.format == 'markdown':
            output = args.output or input_path.with_suffix('.md')
            processor.export_markdown(output)
            print(f"Saved to: {output}")
        elif args.format == 'json':
            output = args.output or input_path.with_suffix('.json')
            processor.export_json(output)
            print(f"Saved to: {output}")
        elif args.format == 'html':
            output = args.output or input_path.with_suffix('.html')
            processor.export_html(output)
            print(f"Saved to: {output}")
