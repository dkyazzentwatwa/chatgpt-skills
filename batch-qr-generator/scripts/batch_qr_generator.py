#!/usr/bin/env python3
"""Batch QR Generator - Generate bulk QR codes from CSV."""

import argparse
import os
from urllib.parse import urlencode
from typing import Dict, Optional
import pandas as pd
import qrcode
from PIL import Image


class BatchQRGenerator:
    """Generate multiple QR codes from CSV data."""

    def __init__(self):
        self.data = None
        self.utm_params = {}

    def load_csv(self, filepath: str, url_column: str):
        """Load CSV with URLs."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV not found: {filepath}")

        self.data = pd.read_csv(filepath)

        if url_column not in self.data.columns:
            raise ValueError(f"Column '{url_column}' not found in CSV")

        self.url_column = url_column
        return self

    def add_utm_params(self, source: str = None, medium: str = None,
                      campaign: str = None, term: str = None, content: str = None):
        """Add UTM tracking parameters."""
        if source:
            self.utm_params['utm_source'] = source
        if medium:
            self.utm_params['utm_medium'] = medium
        if campaign:
            self.utm_params['utm_campaign'] = campaign
        if term:
            self.utm_params['utm_term'] = term
        if content:
            self.utm_params['utm_content'] = content
        return self

    def _add_utm_to_url(self, url: str) -> str:
        """Add UTM parameters to URL."""
        if not self.utm_params:
            return url

        separator = '&' if '?' in url else '?'
        utm_string = urlencode(self.utm_params)
        return f"{url}{separator}{utm_string}"

    def generate_batch(self, output_dir: str, logo_path: str = None,
                      box_size: int = 10, border: int = 4):
        """Generate QR codes for all URLs."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")

        os.makedirs(output_dir, exist_ok=True)

        outputs = []
        metadata = []

        for i, row in self.data.iterrows():
            url = row[self.url_column]
            url_with_utm = self._add_utm_to_url(url)

            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_H if logo_path else qrcode.constants.ERROR_CORRECT_L,
                box_size=box_size,
                border=border
            )
            qr.add_data(url_with_utm)
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")

            # Add logo if provided
            if logo_path and os.path.exists(logo_path):
                logo = Image.open(logo_path)
                # Calculate logo size (10% of QR code)
                qr_width, qr_height = img.size
                logo_size = int(qr_width * 0.1)
                logo = logo.resize((logo_size, logo_size))

                # Paste logo in center
                logo_pos = ((qr_width - logo_size) // 2, (qr_height - logo_size) // 2)
                img.paste(logo, logo_pos)

            # Generate filename
            if 'filename' in row and pd.notna(row['filename']):
                filename = f"{row['filename']}.png"
            else:
                filename = f"qr_{i+1:04d}.png"

            output_path = os.path.join(output_dir, filename)
            img.save(output_path)

            outputs.append(output_path)
            metadata.append({
                'index': i,
                'filename': filename,
                'original_url': url,
                'qr_url': url_with_utm,
                **{k: v for k, v in row.items() if k != self.url_column}
            })

            print(f"Generated {i+1}/{len(self.data)}: {filename}")

        # Save metadata CSV
        metadata_path = os.path.join(output_dir, 'qr_metadata.csv')
        pd.DataFrame(metadata).to_csv(metadata_path, index=False)

        return outputs, metadata_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate bulk QR codes from CSV')
    parser.add_argument('--csv', required=True, help='CSV file with URLs')
    parser.add_argument('--url-column', default='url', help='Column containing URLs')
    parser.add_argument('--output-dir', '-o', default='qr_codes', help='Output directory')
    parser.add_argument('--logo', help='Logo image to embed in QR codes')
    parser.add_argument('--utm-source', help='UTM source parameter')
    parser.add_argument('--utm-medium', help='UTM medium parameter')
    parser.add_argument('--utm-campaign', help='UTM campaign parameter')
    parser.add_argument('--box-size', type=int, default=10, help='QR box size')
    parser.add_argument('--border', type=int, default=4, help='QR border size')

    args = parser.parse_args()

    generator = BatchQRGenerator()
    generator.load_csv(args.csv, url_column=args.url_column)

    if any([args.utm_source, args.utm_medium, args.utm_campaign]):
        generator.add_utm_params(
            source=args.utm_source,
            medium=args.utm_medium,
            campaign=args.utm_campaign
        )

    outputs, metadata_path = generator.generate_batch(
        output_dir=args.output_dir,
        logo_path=args.logo,
        box_size=args.box_size,
        border=args.border
    )

    print(f"\n✓ Generated {len(outputs)} QR codes in {args.output_dir}")
    print(f"✓ Metadata saved to: {metadata_path}")
