#!/usr/bin/env python3
"""QR/Barcode Reader - Decode QR codes and barcodes from images."""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
from pyzbar.pyzbar import decode
from PIL import Image


class QRBarcodeReader:
    """Read QR codes and barcodes from images."""

    def __init__(self):
        self.results = []

    def read_image(self, filepath: str) -> List[Dict]:
        """Read barcodes from single image."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image not found: {filepath}")

        image = Image.open(filepath)
        decoded_objects = decode(image)

        results = []
        for obj in decoded_objects:
            results.append({
                'data': obj.data.decode('utf-8'),
                'type': obj.type,
                'rect': {
                    'x': obj.rect.left,
                    'y': obj.rect.top,
                    'width': obj.rect.width,
                    'height': obj.rect.height
                }
            })

        return results

    def read_directory(self, pattern: str) -> Dict:
        """Read barcodes from multiple images."""
        from glob import glob

        files = glob(pattern)
        all_results = {}

        for filepath in files:
            print(f"Reading {os.path.basename(filepath)}...")
            try:
                results = self.read_image(filepath)
                all_results[filepath] = results
                if results:
                    print(f"  Found {len(results)} barcode(s)")
                else:
                    print(f"  No barcodes found")
            except Exception as e:
                print(f"  Error: {e}")
                all_results[filepath] = {'error': str(e)}

        return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read QR codes and barcodes from images')
    parser.add_argument('input', nargs='+', help='Image file(s) or pattern')
    parser.add_argument('--output', '-o', help='Output JSON file')

    args = parser.parse_args()

    reader = QRBarcodeReader()

    if len(args.input) == 1 and '*' not in args.input[0]:
        # Single file
        results = reader.read_image(args.input[0])
        print(f"\nFound {len(results)} barcode(s):")
        for i, result in enumerate(results, 1):
            print(f"{i}. Type: {result['type']}, Data: {result['data']}")
    else:
        # Multiple files
        results = {}
        for pattern in args.input:
            results.update(reader.read_directory(pattern))

        print(f"\n✓ Processed {len(results)} image(s)")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {args.output}")
