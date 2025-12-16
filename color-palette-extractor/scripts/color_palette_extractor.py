#!/usr/bin/env python3
"""
Color Palette Extractor - Extract dominant colors from images.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class ColorPaletteExtractor:
    """Extract color palettes from images."""

    def __init__(self):
        """Initialize extractor."""
        self.image = None
        self.colors = []
        self.rgb_colors = []

    def load(self, filepath: str) -> 'ColorPaletteExtractor':
        """Load image."""
        self.image = Image.open(filepath).convert('RGB')
        return self

    def extract_colors(self, n_colors: int = 5, sample_size: int = 10000) -> List[str]:
        """
        Extract dominant colors using K-means clustering.

        Args:
            n_colors: Number of colors to extract
            sample_size: Number of pixels to sample (for performance)
        """
        if not self.image:
            raise ValueError("No image loaded")

        # Get pixel data
        pixels = np.array(self.image)
        h, w, c = pixels.shape
        pixels_flat = pixels.reshape(-1, 3)

        # Sample if image is large
        if len(pixels_flat) > sample_size:
            indices = np.random.choice(len(pixels_flat), sample_size, replace=False)
            pixels_flat = pixels_flat[indices]

        # K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels_flat)

        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)
        self.rgb_colors = [tuple(color) for color in colors]

        # Convert to HEX
        self.colors = [self._rgb_to_hex(color) for color in colors]

        # Sort by frequency
        labels = kmeans.labels_
        counts = np.bincount(labels)
        sorted_indices = np.argsort(-counts)

        self.colors = [self.colors[i] for i in sorted_indices]
        self.rgb_colors = [self.rgb_colors[i] for i in sorted_indices]

        return self.colors

    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to HEX."""
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert HEX to RGB."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def get_color_info(self, color_hex: str) -> Dict:
        """Get color information in multiple formats."""
        rgb = self._hex_to_rgb(color_hex)

        # Convert to HSL
        r, g, b = [x / 255.0 for x in rgb]
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        l = (max_c + min_c) / 2

        if max_c == min_c:
            h = s = 0
        else:
            d = max_c - min_c
            s = d / (2 - max_c - min_c) if l > 0.5 else d / (max_c + min_c)

            if max_c == r:
                h = (g - b) / d + (6 if g < b else 0)
            elif max_c == g:
                h = (b - r) / d + 2
            else:
                h = (r - g) / d + 4
            h /= 6

        return {
            'hex': color_hex,
            'rgb': rgb,
            'hsl': (int(h * 360), int(s * 100), int(l * 100))
        }

    def export_css(self, output: str, prefix: str = 'color') -> str:
        """Export as CSS custom properties."""
        css = ":root {\n"
        for i, color in enumerate(self.colors, 1):
            css += f"  --{prefix}-{i}: {color};\n"
        css += "}\n"

        with open(output, 'w') as f:
            f.write(css)

        return output

    def export_json(self, output: str) -> str:
        """Export as JSON."""
        palette = []
        for i, (hex_color, rgb) in enumerate(zip(self.colors, self.rgb_colors), 1):
            info = self.get_color_info(hex_color)
            palette.append({
                'name': f'Color {i}',
                'hex': hex_color,
                'rgb': list(rgb),
                'hsl': info['hsl']
            })

        with open(output, 'w') as f:
            json.dump(palette, f, indent=2)

        return output

    def save_swatch(self, output: str, width: int = 800, height: int = 100) -> str:
        """Generate swatch image."""
        n_colors = len(self.colors)
        color_width = width // n_colors

        swatch = Image.new('RGB', (width, height))
        pixels = swatch.load()

        for i, rgb in enumerate(self.rgb_colors):
            x_start = i * color_width
            x_end = (i + 1) * color_width if i < n_colors - 1 else width

            for x in range(x_start, x_end):
                for y in range(height):
                    pixels[x, y] = rgb

        swatch.save(output)
        return output

    def visualize(self, output: str) -> str:
        """Create visualization with color info."""
        n_colors = len(self.colors)

        fig, axes = plt.subplots(1, n_colors, figsize=(n_colors * 2, 3))

        if n_colors == 1:
            axes = [axes]

        for i, (hex_color, rgb) in enumerate(zip(self.colors, self.rgb_colors)):
            # Color swatch
            axes[i].add_patch(plt.Rectangle((0, 0), 1, 1, color=np.array(rgb)/255))
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)
            axes[i].axis('off')

            # Info text
            info = self.get_color_info(hex_color)
            text = f"{hex_color}\nRGB: {rgb}\nHSL: {info['hsl']}"
            axes[i].text(0.5, -0.1, text, ha='center', va='top',
                        fontsize=8, transform=axes[i].transAxes)

        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()

        return output


def main():
    parser = argparse.ArgumentParser(description="Color Palette Extractor")

    parser.add_argument("--input", "-i", required=True, help="Input image")
    parser.add_argument("--output", "-o", help="Output file (JSON)")
    parser.add_argument("--colors", "-n", type=int, default=5,
                       help="Number of colors to extract")

    parser.add_argument("--css", help="Export CSS file")
    parser.add_argument("--swatch", help="Save swatch image")
    parser.add_argument("--viz", help="Save visualization")

    args = parser.parse_args()

    extractor = ColorPaletteExtractor()
    extractor.load(args.input)

    print(f"Extracting {args.colors} colors from {args.input}...")
    colors = extractor.extract_colors(n_colors=args.colors)

    print("\nExtracted Colors:")
    for i, color in enumerate(colors, 1):
        print(f"  {i}. {color}")

    # Exports
    if args.output:
        extractor.export_json(args.output)
        print(f"\nJSON saved: {args.output}")

    if args.css:
        extractor.export_css(args.css)
        print(f"CSS saved: {args.css}")

    if args.swatch:
        extractor.save_swatch(args.swatch)
        print(f"Swatch saved: {args.swatch}")

    if args.viz:
        extractor.visualize(args.viz)
        print(f"Visualization saved: {args.viz}")


if __name__ == "__main__":
    main()
