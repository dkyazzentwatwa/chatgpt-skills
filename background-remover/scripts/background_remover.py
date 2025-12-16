#!/usr/bin/env python3
"""
Background Remover - Remove backgrounds from images using segmentation.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import cv2


class BackgroundRemover:
    """Remove backgrounds from images using various methods."""

    def __init__(self):
        """Initialize the remover."""
        self.image = None
        self.original = None
        self.mask = None
        self.filepath = None

    def load(self, filepath: str) -> 'BackgroundRemover':
        """Load an image from file."""
        self.filepath = filepath
        self.image = Image.open(filepath).convert('RGBA')
        self.original = self.image.copy()
        self.mask = None
        return self

    def load_array(self, array: np.ndarray) -> 'BackgroundRemover':
        """Load from numpy array."""
        if array.shape[-1] == 3:
            self.image = Image.fromarray(array).convert('RGBA')
        else:
            self.image = Image.fromarray(array)
        self.original = self.image.copy()
        self.mask = None
        return self

    def remove_background(self, method: str = "auto") -> 'BackgroundRemover':
        """
        Remove background using specified method.

        Args:
            method: "auto", "edge", "grabcut", or "color"
        """
        if method == "auto":
            # Try to detect best method
            method = self._detect_best_method()

        if method == "edge":
            return self.remove_edges()
        elif method == "grabcut":
            return self.grabcut()
        elif method == "color":
            # Try to detect dominant background color
            bg_color = self._detect_background_color()
            return self.remove_color(bg_color, tolerance=30)
        else:
            return self.grabcut()

    def _detect_best_method(self) -> str:
        """Detect best removal method based on image characteristics."""
        img_array = np.array(self.image.convert('RGB'))

        # Check if corners have similar colors (likely solid background)
        h, w = img_array.shape[:2]
        corners = [
            img_array[0, 0],
            img_array[0, w-1],
            img_array[h-1, 0],
            img_array[h-1, w-1]
        ]

        # Calculate variance of corner colors
        corner_variance = np.var([np.mean(c) for c in corners])

        if corner_variance < 100:
            return "color"
        else:
            return "grabcut"

    def _detect_background_color(self) -> Tuple[int, int, int]:
        """Detect dominant background color from image edges."""
        img_array = np.array(self.image.convert('RGB'))
        h, w = img_array.shape[:2]

        # Sample colors from edges
        edge_pixels = []
        edge_pixels.extend(img_array[0, :].tolist())  # Top
        edge_pixels.extend(img_array[h-1, :].tolist())  # Bottom
        edge_pixels.extend(img_array[:, 0].tolist())  # Left
        edge_pixels.extend(img_array[:, w-1].tolist())  # Right

        # Find most common color
        edge_pixels = np.array(edge_pixels)
        mean_color = np.mean(edge_pixels, axis=0).astype(int)

        return tuple(mean_color)

    def remove_color(self, color: Tuple[int, int, int],
                    tolerance: int = 20) -> 'BackgroundRemover':
        """
        Remove specific color from image.

        Args:
            color: RGB tuple of color to remove
            tolerance: Color matching tolerance (0-255)
        """
        if self.image is None:
            raise ValueError("No image loaded")

        img_array = np.array(self.image.convert('RGB'))

        # Calculate distance from target color
        diff = np.abs(img_array.astype(int) - np.array(color))
        distance = np.sqrt(np.sum(diff ** 2, axis=2))

        # Create mask
        mask = (distance > tolerance * np.sqrt(3)).astype(np.uint8) * 255
        self.mask = Image.fromarray(mask, mode='L')

        # Apply mask to image
        self.image.putalpha(self.mask)

        return self

    def remove_edges(self, threshold: int = 50) -> 'BackgroundRemover':
        """
        Remove background using edge detection.

        Args:
            threshold: Edge detection threshold
        """
        if self.image is None:
            raise ValueError("No image loaded")

        img_array = np.array(self.image.convert('RGB'))

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, threshold, threshold * 2)

        # Dilate edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Fill holes
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(gray)
        if contours:
            # Find largest contour
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest], -1, 255, -1)

        self.mask = Image.fromarray(mask, mode='L')
        self.image.putalpha(self.mask)

        return self

    def grabcut(self, rect: Tuple[int, int, int, int] = None,
               iterations: int = 5) -> 'BackgroundRemover':
        """
        Remove background using GrabCut algorithm.

        Args:
            rect: Bounding rectangle (x, y, width, height) containing foreground
            iterations: Number of GrabCut iterations
        """
        if self.image is None:
            raise ValueError("No image loaded")

        img_array = np.array(self.image.convert('RGB'))
        h, w = img_array.shape[:2]

        # Default rectangle: slightly smaller than full image
        if rect is None:
            margin = 10
            rect = (margin, margin, w - 2*margin, h - 2*margin)

        # Initialize mask
        mask = np.zeros((h, w), np.uint8)

        # GrabCut models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Run GrabCut
        cv2.grabCut(img_array, mask, rect, bgd_model, fgd_model,
                   iterations, cv2.GC_INIT_WITH_RECT)

        # Create binary mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')

        self.mask = Image.fromarray(mask2, mode='L')
        self.image.putalpha(self.mask)

        return self

    def replace_background(self, color: Tuple[int, int, int] = None,
                          image: str = None) -> 'BackgroundRemover':
        """
        Replace transparent background with color or image.

        Args:
            color: RGB tuple for solid color background
            image: Path to background image
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Ensure we have RGBA
        if self.image.mode != 'RGBA':
            self.image = self.image.convert('RGBA')

        w, h = self.image.size

        if image:
            # Use image background
            bg = Image.open(image).convert('RGB')
            bg = bg.resize((w, h), Image.Resampling.LANCZOS)
        elif color:
            # Use solid color
            bg = Image.new('RGB', (w, h), color)
        else:
            # Default white
            bg = Image.new('RGB', (w, h), (255, 255, 255))

        # Composite foreground onto background
        bg.paste(self.image, (0, 0), self.image)
        self.image = bg.convert('RGBA')

        return self

    def add_shadow(self, offset: Tuple[int, int] = (5, 5),
                  blur: int = 10, opacity: int = 128) -> 'BackgroundRemover':
        """
        Add drop shadow to subject.

        Args:
            offset: Shadow offset (x, y)
            blur: Shadow blur radius
            opacity: Shadow opacity (0-255)
        """
        if self.image is None or self.mask is None:
            return self

        w, h = self.image.size

        # Create shadow from mask
        shadow = self.mask.copy()

        # Apply blur
        shadow = shadow.filter(ImageFilter.GaussianBlur(blur))

        # Create shadow layer
        shadow_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        shadow_rgba = Image.new('RGBA', (w, h), (0, 0, 0, opacity))
        shadow_layer.paste(shadow_rgba, offset, shadow)

        # Composite: background + shadow + subject
        result = Image.new('RGBA', (w, h), (255, 255, 255, 255))
        result = Image.alpha_composite(result, shadow_layer)
        result = Image.alpha_composite(result, self.image)

        self.image = result
        return self

    def refine_edges(self, feather: int = 2) -> 'BackgroundRemover':
        """
        Refine mask edges with feathering.

        Args:
            feather: Feather radius in pixels
        """
        if self.mask is None:
            return self

        # Apply gaussian blur to mask for feathering
        blurred_mask = self.mask.filter(ImageFilter.GaussianBlur(feather))

        # Update alpha channel
        if self.image.mode == 'RGBA':
            r, g, b, _ = self.image.split()
            self.image = Image.merge('RGBA', (r, g, b, blurred_mask))

        self.mask = blurred_mask
        return self

    def expand_mask(self, pixels: int = 2) -> 'BackgroundRemover':
        """Expand mask by specified pixels."""
        if self.mask is None:
            return self

        mask_array = np.array(self.mask)
        kernel = np.ones((pixels*2+1, pixels*2+1), np.uint8)
        expanded = cv2.dilate(mask_array, kernel, iterations=1)

        self.mask = Image.fromarray(expanded, mode='L')

        if self.image.mode == 'RGBA':
            r, g, b, _ = self.image.split()
            self.image = Image.merge('RGBA', (r, g, b, self.mask))

        return self

    def contract_mask(self, pixels: int = 2) -> 'BackgroundRemover':
        """Contract mask by specified pixels."""
        if self.mask is None:
            return self

        mask_array = np.array(self.mask)
        kernel = np.ones((pixels*2+1, pixels*2+1), np.uint8)
        contracted = cv2.erode(mask_array, kernel, iterations=1)

        self.mask = Image.fromarray(contracted, mode='L')

        if self.image.mode == 'RGBA':
            r, g, b, _ = self.image.split()
            self.image = Image.merge('RGBA', (r, g, b, self.mask))

        return self

    def save(self, filepath: str, quality: int = 95) -> str:
        """Save the processed image."""
        if self.image is None:
            raise ValueError("No image to save")

        # Determine format from extension
        ext = Path(filepath).suffix.lower()

        if ext in ['.jpg', '.jpeg']:
            # JPEG doesn't support alpha, convert to RGB
            rgb_image = Image.new('RGB', self.image.size, (255, 255, 255))
            if self.image.mode == 'RGBA':
                rgb_image.paste(self.image, mask=self.image.split()[3])
            else:
                rgb_image.paste(self.image)
            rgb_image.save(filepath, quality=quality)
        else:
            self.image.save(filepath, quality=quality)

        return filepath

    def get_image(self) -> Image:
        """Get the processed PIL Image."""
        return self.image

    def get_mask(self) -> Image:
        """Get the mask as PIL Image."""
        return self.mask

    def batch_process(self, input_dir: str, output_dir: str,
                     method: str = "auto", **kwargs) -> List[str]:
        """
        Process multiple images.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            method: Removal method
            **kwargs: Additional arguments for removal method

        Returns:
            List of processed file paths
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        processed = []
        extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

        for img_file in input_path.iterdir():
            if img_file.suffix.lower() not in extensions:
                continue

            try:
                self.load(str(img_file))

                if method == "color" and 'color' in kwargs:
                    self.remove_color(kwargs['color'], kwargs.get('tolerance', 30))
                elif method == "grabcut":
                    self.grabcut(iterations=kwargs.get('iterations', 5))
                elif method == "edge":
                    self.remove_edges(threshold=kwargs.get('threshold', 50))
                else:
                    self.remove_background(method=method)

                # Refine if specified
                if kwargs.get('feather'):
                    self.refine_edges(kwargs['feather'])

                # Save as PNG to preserve transparency
                output_file = output_path / f"{img_file.stem}_nobg.png"
                self.save(str(output_file))
                processed.append(str(output_file))

                print(f"Processed: {img_file.name}")

            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")

        return processed


def parse_color(color_str: str) -> Tuple[int, int, int]:
    """Parse color string like '255,255,255' to tuple."""
    parts = color_str.split(',')
    return tuple(int(p.strip()) for p in parts)


def main():
    parser = argparse.ArgumentParser(description="Background Remover")

    parser.add_argument("--input", "-i", help="Input image file")
    parser.add_argument("--output", "-o", help="Output image file")

    parser.add_argument("--method", choices=['auto', 'color', 'edge', 'grabcut'],
                       default='auto', help="Removal method")
    parser.add_argument("--color", help="Color to remove (R,G,B)")
    parser.add_argument("--tolerance", type=int, default=30, help="Color tolerance")
    parser.add_argument("--threshold", type=int, default=50, help="Edge threshold")
    parser.add_argument("--iterations", type=int, default=5, help="GrabCut iterations")

    parser.add_argument("--replace-color", help="Replace background with color (R,G,B)")
    parser.add_argument("--replace-image", help="Replace background with image")

    parser.add_argument("--feather", type=int, help="Edge feather radius")
    parser.add_argument("--expand", type=int, help="Expand mask by pixels")
    parser.add_argument("--contract", type=int, help="Contract mask by pixels")

    parser.add_argument("--batch", help="Batch process directory")
    parser.add_argument("--output-dir", help="Output directory for batch")

    args = parser.parse_args()

    remover = BackgroundRemover()

    if args.batch:
        # Batch processing
        output_dir = args.output_dir or f"{args.batch}_processed"

        kwargs = {
            'tolerance': args.tolerance,
            'threshold': args.threshold,
            'iterations': args.iterations
        }
        if args.color:
            kwargs['color'] = parse_color(args.color)
        if args.feather:
            kwargs['feather'] = args.feather

        processed = remover.batch_process(
            args.batch, output_dir,
            method=args.method, **kwargs
        )
        print(f"\nProcessed {len(processed)} images to {output_dir}")

    elif args.input:
        # Single image processing
        remover.load(args.input)

        # Apply removal method
        if args.method == "color" and args.color:
            remover.remove_color(parse_color(args.color), args.tolerance)
        elif args.method == "edge":
            remover.remove_edges(args.threshold)
        elif args.method == "grabcut":
            remover.grabcut(iterations=args.iterations)
        else:
            if args.color:
                remover.remove_color(parse_color(args.color), args.tolerance)
            else:
                remover.remove_background(method=args.method)

        # Refinement
        if args.feather:
            remover.refine_edges(args.feather)
        if args.expand:
            remover.expand_mask(args.expand)
        if args.contract:
            remover.contract_mask(args.contract)

        # Background replacement
        if args.replace_color:
            remover.replace_background(color=parse_color(args.replace_color))
        elif args.replace_image:
            remover.replace_background(image=args.replace_image)

        # Save
        output = args.output or args.input.rsplit('.', 1)[0] + '_nobg.png'
        remover.save(output)
        print(f"Saved: {output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
