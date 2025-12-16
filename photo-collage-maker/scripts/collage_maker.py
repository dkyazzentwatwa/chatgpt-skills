#!/usr/bin/env python3
"""
Photo Collage Maker - Create photo collages with grid layouts and custom arrangements.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import glob

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class CollageMaker:
    """Create photo collages from multiple images."""

    TEMPLATES = {
        'grid_2x2': {'rows': 2, 'cols': 2, 'gap': 10},
        'grid_3x3': {'rows': 3, 'cols': 3, 'gap': 10},
        'grid_2x3': {'rows': 2, 'cols': 3, 'gap': 10},
        'magazine': {
            'layout': [
                {'x': 0, 'y': 0, 'w': 0.6, 'h': 1.0},
                {'x': 0.6, 'y': 0, 'w': 0.4, 'h': 0.5},
                {'x': 0.6, 'y': 0.5, 'w': 0.4, 'h': 0.5}
            ]
        },
        'pinterest': {
            'cols': 3,
            'aspect_ratio': 'varied'
        },
        'polaroid': {
            'rows': 2,
            'cols': 3,
            'padding': 20,
            'border': 40
        }
    }

    def __init__(self):
        """Initialize the collage maker."""
        self.canvas = None
        self.width = 1200
        self.height = 800
        self.bg_color = (255, 255, 255)
        self.gap = 10
        self.border = 0
        self.border_color = (255, 255, 255)
        self.images = []
        self.layout = []

    def set_canvas(self, width: int, height: int,
                  bg_color: Tuple[int, int, int] = (255, 255, 255)) -> 'CollageMaker':
        """Set canvas dimensions and background."""
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.canvas = Image.new('RGB', (width, height), bg_color)
        return self

    def grid(self, rows: int, cols: int, gap: int = 10) -> 'CollageMaker':
        """Set up a grid layout."""
        self.gap = gap
        self.layout = []

        # Calculate cell dimensions
        total_gap_w = gap * (cols + 1)
        total_gap_h = gap * (rows + 1)
        cell_w = (self.width - total_gap_w) // cols
        cell_h = (self.height - total_gap_h) // rows

        for row in range(rows):
            for col in range(cols):
                x = gap + col * (cell_w + gap)
                y = gap + row * (cell_h + gap)
                self.layout.append({
                    'x': x, 'y': y, 'width': cell_w, 'height': cell_h
                })

        return self

    def template(self, name: str) -> 'CollageMaker':
        """Apply a predefined template."""
        if name not in self.TEMPLATES:
            raise ValueError(f"Unknown template: {name}")

        template = self.TEMPLATES[name]

        if 'rows' in template and 'cols' in template:
            self.grid(template['rows'], template['cols'], template.get('gap', 10))
        elif 'layout' in template:
            # Custom layout with proportions
            self.layout = []
            for slot in template['layout']:
                self.layout.append({
                    'x': int(slot['x'] * self.width),
                    'y': int(slot['y'] * self.height),
                    'width': int(slot['w'] * self.width),
                    'height': int(slot['h'] * self.height)
                })

        return self

    def add_images(self, image_paths: List[str], fit: str = "fill") -> 'CollageMaker':
        """Add multiple images to the layout."""
        if not self.layout:
            raise ValueError("No layout defined. Call grid() or template() first.")

        if not self.canvas:
            self.canvas = Image.new('RGB', (self.width, self.height), self.bg_color)

        for i, path in enumerate(image_paths):
            if i >= len(self.layout):
                break

            slot = self.layout[i]
            self._add_image_to_slot(path, slot, fit)

        return self

    def add_image(self, path: str, x: int, y: int, width: int, height: int,
                 fit: str = "fill") -> 'CollageMaker':
        """Add a single image at specific position."""
        if not self.canvas:
            self.canvas = Image.new('RGB', (self.width, self.height), self.bg_color)

        slot = {'x': x, 'y': y, 'width': width, 'height': height}
        self._add_image_to_slot(path, slot, fit)

        return self

    def _add_image_to_slot(self, path: str, slot: Dict, fit: str = "fill"):
        """Add an image to a specific slot."""
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return

        target_w = slot['width']
        target_h = slot['height']

        if fit == "fill":
            # Crop to fill entire slot
            img = self._crop_to_fill(img, target_w, target_h)
        elif fit == "fit":
            # Fit within slot (may have letterboxing)
            img = self._fit_within(img, target_w, target_h)
        elif fit == "stretch":
            # Stretch to fill
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        self.canvas.paste(img, (slot['x'], slot['y']))

    def _crop_to_fill(self, img: Image, target_w: int, target_h: int) -> Image:
        """Crop image to fill target dimensions."""
        img_w, img_h = img.size
        target_ratio = target_w / target_h
        img_ratio = img_w / img_h

        if img_ratio > target_ratio:
            # Image is wider - crop sides
            new_w = int(img_h * target_ratio)
            left = (img_w - new_w) // 2
            img = img.crop((left, 0, left + new_w, img_h))
        else:
            # Image is taller - crop top/bottom
            new_h = int(img_w / target_ratio)
            top = (img_h - new_h) // 2
            img = img.crop((0, top, img_w, top + new_h))

        return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    def _fit_within(self, img: Image, target_w: int, target_h: int) -> Image:
        """Fit image within target dimensions (letterboxed)."""
        img_w, img_h = img.size
        ratio = min(target_w / img_w, target_h / img_h)

        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create letterboxed image
        result = Image.new('RGB', (target_w, target_h), self.bg_color)
        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2
        result.paste(img, (x, y))

        return result

    def set_background(self, color: Tuple[int, int, int] = None,
                      image: str = None) -> 'CollageMaker':
        """Set background color or image."""
        if image:
            bg = Image.open(image).convert('RGB')
            bg = bg.resize((self.width, self.height), Image.Resampling.LANCZOS)
            self.canvas = bg
        elif color:
            self.bg_color = color
            if self.canvas:
                # Redraw with new background
                new_canvas = Image.new('RGB', (self.width, self.height), color)
                self.canvas = new_canvas

        return self

    def set_border(self, width: int, color: Tuple[int, int, int] = (255, 255, 255)) -> 'CollageMaker':
        """Set border width and color."""
        self.border = width
        self.border_color = color
        return self

    def set_gap(self, gap: int) -> 'CollageMaker':
        """Set gap between images."""
        self.gap = gap
        return self

    def rounded_corners(self, radius: int) -> 'CollageMaker':
        """Apply rounded corners to the collage."""
        if not self.canvas:
            return self

        # Create mask with rounded corners
        mask = Image.new('L', (self.width, self.height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), (self.width, self.height)],
                               radius=radius, fill=255)

        # Apply mask
        result = Image.new('RGB', (self.width, self.height), self.bg_color)
        result.paste(self.canvas, mask=mask)
        self.canvas = result

        return self

    def add_text(self, text: str, x: int, y: int, font_size: int = 24,
                color: Tuple[int, int, int] = (0, 0, 0)) -> 'CollageMaker':
        """Add text to the collage."""
        if not self.canvas:
            return self

        draw = ImageDraw.Draw(self.canvas)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

        draw.text((x, y), text, fill=color, font=font)

        return self

    def save(self, filepath: str, quality: int = 95) -> str:
        """Save the collage."""
        if not self.canvas:
            raise ValueError("No collage created")

        # Add border if specified
        if self.border > 0:
            bordered = Image.new('RGB',
                (self.width + 2 * self.border, self.height + 2 * self.border),
                self.border_color)
            bordered.paste(self.canvas, (self.border, self.border))
            bordered.save(filepath, quality=quality)
        else:
            self.canvas.save(filepath, quality=quality)

        return filepath

    def get_image(self) -> Image:
        """Get the PIL Image object."""
        return self.canvas


def main():
    parser = argparse.ArgumentParser(description="Photo Collage Maker")

    parser.add_argument("--images", nargs='+', help="Input image files")
    parser.add_argument("--output", "-o", required=True, help="Output file")

    parser.add_argument("--grid", help="Grid layout (e.g., 2x2, 3x3)")
    parser.add_argument("--template", choices=list(CollageMaker.TEMPLATES.keys()),
                       help="Use template layout")
    parser.add_argument("--gap", type=int, default=10, help="Gap between images")

    parser.add_argument("--width", type=int, default=1200, help="Canvas width")
    parser.add_argument("--height", type=int, default=800, help="Canvas height")

    parser.add_argument("--bg-color", help="Background color (R,G,B)")
    parser.add_argument("--bg-image", help="Background image")

    parser.add_argument("--fit", choices=['fill', 'fit', 'stretch'],
                       default='fill', help="Image fit mode")

    parser.add_argument("--border", type=int, default=0, help="Border width")
    parser.add_argument("--border-color", help="Border color (R,G,B)")

    parser.add_argument("--rounded", type=int, help="Rounded corner radius")

    args = parser.parse_args()

    collage = CollageMaker()

    # Set canvas
    bg_color = (255, 255, 255)
    if args.bg_color:
        bg_color = tuple(int(x) for x in args.bg_color.split(','))

    collage.set_canvas(args.width, args.height, bg_color)

    # Background image
    if args.bg_image:
        collage.set_background(image=args.bg_image)

    # Set layout
    if args.grid:
        rows, cols = map(int, args.grid.lower().split('x'))
        collage.grid(rows, cols, args.gap)
    elif args.template:
        collage.template(args.template)
    else:
        # Default to 2x2 grid
        collage.grid(2, 2, args.gap)

    # Get images
    images = []
    if args.images:
        for pattern in args.images:
            if '*' in pattern:
                images.extend(glob.glob(pattern))
            else:
                images.append(pattern)

    if not images:
        print("No images specified")
        parser.print_help()
        return

    # Add images
    collage.add_images(images, fit=args.fit)

    # Apply effects
    if args.rounded:
        collage.rounded_corners(args.rounded)

    if args.border > 0:
        border_color = (255, 255, 255)
        if args.border_color:
            border_color = tuple(int(x) for x in args.border_color.split(','))
        collage.set_border(args.border, border_color)

    # Save
    collage.save(args.output)
    print(f"Collage saved: {args.output}")


if __name__ == "__main__":
    main()
