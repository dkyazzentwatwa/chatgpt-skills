#!/usr/bin/env python3
"""KML/GeoJSON Converter - Convert between geo formats."""

import argparse
import json
import os
from pathlib import Path
import geopandas as gpd


class GeoConverter:
    """Convert between KML and GeoJSON formats."""

    def __init__(self):
        self.gdf = None

    def load_kml(self, filepath: str):
        """Load KML file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"KML file not found: {filepath}")

        # Enable KML driver
        gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
        self.gdf = gpd.read_file(filepath, driver='KML')
        return self

    def load_geojson(self, filepath: str):
        """Load GeoJSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GeoJSON file not found: {filepath}")

        self.gdf = gpd.read_file(filepath)
        return self

    def save_kml(self, output: str):
        """Save as KML file."""
        if self.gdf is None:
            raise ValueError("No data loaded")

        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

        # Enable KML driver
        gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
        self.gdf.to_file(output, driver='KML')
        return output

    def save_geojson(self, output: str):
        """Save as GeoJSON file."""
        if self.gdf is None:
            raise ValueError("No data loaded")

        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        self.gdf.to_file(output, driver='GeoJSON')
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert between KML and GeoJSON formats')
    parser.add_argument('input', help='Input file (KML or GeoJSON)')
    parser.add_argument('--to', choices=['kml', 'geojson'], required=True, help='Output format')
    parser.add_argument('--output', '-o', required=True, help='Output file path')

    args = parser.parse_args()

    converter = GeoConverter()

    # Detect input format
    input_ext = Path(args.input).suffix.lower()

    print(f"Converting {args.input} to {args.to.upper()}...")

    # Load input
    if input_ext == '.kml':
        converter.load_kml(args.input)
    elif input_ext in ['.geojson', '.json']:
        converter.load_geojson(args.input)
    else:
        print(f"Error: Unknown input format '{input_ext}'")
        sys.exit(1)

    # Save output
    if args.to == 'kml':
        converter.save_kml(args.output)
    else:
        converter.save_geojson(args.output)

    print(f"✓ Converted to: {args.output}")
