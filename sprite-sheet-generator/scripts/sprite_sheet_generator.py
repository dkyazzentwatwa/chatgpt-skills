#!/usr/bin/env python3
"""
Sprite Sheet Generator - Combine images into sprite sheets.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image


class SpriteSheetGenerator:
    """Generate sprite sheets from multiple images."""

    def __init__(self):
        """Initialize generator."""
        self.images = []
        self.image_names = []

    def add_image(self, filepath: str, name: str = None) -> 'SpriteSheetGenerator':
        """Add single image."""
        img = Image.open(filepath).convert('RGBA')
        self.images.append(img)
        self.image_names.append(name or Path(filepath).stem)
        return self

    def add_images_from_dir(self, directory: str, pattern: str = '*') -> 'SpriteSheetGenerator':
        """Add all images from directory."""
        path = Path(directory)
        for filepath in sorted(path.glob(pattern)):
            if filepath.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                self.add_image(str(filepath))
        return self

    def generate(self, output: str, grid: Tuple[int, int] = None,
                sprite_size: Tuple[int, int] = None, padding: int = 0) -> str:
        """
        Generate sprite sheet.

        Args:
            output: Output filepath
            grid: (cols, rows) or None for auto
            sprite_size: (width, height) or None to use max size
            padding: Pixels between sprites
        """
        if not self.images:
            raise ValueError("No images added")

        # Determine sprite size
        if sprite_size:
            w, h = sprite_size
        else:
            # Use max dimensions
            w = max(img.width for img in self.images)
            h = max(img.height for img in self.images)

        # Resize images
        resized = []
        for img in self.images:
            # Create canvas
            canvas = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            # Center image
            x = (w - img.width) // 2
            y = (h - img.height) // 2
            canvas.paste(img, (x, y))
            resized.append(canvas)

        # Determine grid
        if grid:
            cols, rows = grid
        else:
            cols = int(len(resized) ** 0.5) + 1
            rows = (len(resized) + cols - 1) // cols

        # Calculate sheet size
        sheet_width = cols * w + (cols + 1) * padding
        sheet_height = rows * h + (rows + 1) * padding

        # Create sheet
        sheet = Image.new('RGBA', (sheet_width, sheet_height), (0, 0, 0, 0))

        # Place sprites
        for i, img in enumerate(resized):
            row = i // cols
            col = i % cols

            x = padding + col * (w + padding)
            y = padding + row * (h + padding)

            sheet.paste(img, (x, y))

        sheet.save(output)

        # Store metadata
        self.sprite_width = w
        self.sprite_height = h
        self.cols = cols
        self.rows = rows
        self.padding = padding

        return output

    def generate_css(self, output: str, class_prefix: str = 'sprite',
                    sprite_sheet_path: str = 'sprites.png') -> str:
        """Generate CSS file."""
        if not hasattr(self, 'sprite_width'):
            raise ValueError("Call generate() first")

        css = f".{class_prefix} {{\n"
        css += f"  background-image: url('{sprite_sheet_path}');\n"
        css += f"  display: inline-block;\n"
        css += f"  width: {self.sprite_width}px;\n"
        css += f"  height: {self.sprite_height}px;\n"
        css += "}\n\n"

        for i, name in enumerate(self.image_names):
            row = i // self.cols
            col = i % self.cols

            x = -(self.padding + col * (self.sprite_width + self.padding))
            y = -(self.padding + row * (self.sprite_height + self.padding))

            css += f".{class_prefix}-{name} {{\n"
            css += f"  background-position: {x}px {y}px;\n"
            css += "}\n\n"

        with open(output, 'w') as f:
            f.write(css)

        return output


def main():
    parser = argparse.ArgumentParser(description="Sprite Sheet Generator")

    parser.add_argument("--input", "-i", required=True, help="Input directory")
    parser.add_argument("--output", "-o", required=True, help="Output sprite sheet")

    parser.add_argument("--grid", help="Grid size (e.g., 4x4)")
    parser.add_argument("--size", help="Sprite size (e.g., 64x64)")
    parser.add_argument("--padding", type=int, default=0, help="Padding between sprites")
    parser.add_argument("--css", help="Generate CSS file")

    args = parser.parse_args()

    gen = SpriteSheetGenerator()
    gen.add_images_from_dir(args.input)

    print(f"Found {len(gen.images)} images")

    # Parse grid
    grid = None
    if args.grid:
        cols, rows = map(int, args.grid.split('x'))
        grid = (cols, rows)

    # Parse size
    sprite_size = None
    if args.size:
        w, h = map(int, args.size.split('x'))
        sprite_size = (w, h)

    # Generate
    gen.generate(args.output, grid=grid, sprite_size=sprite_size, padding=args.padding)
    print(f"Sprite sheet saved: {args.output}")

    if args.css:
        gen.generate_css(args.css, sprite_sheet_path=Path(args.output).name)
        print(f"CSS saved: {args.css}")


if __name__ == "__main__":
    main()
