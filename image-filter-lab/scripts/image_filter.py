#!/usr/bin/env python3
"""
Image Filter Lab - Apply artistic filters and effects to images.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import cv2


class ImageFilterLab:
    """Apply professional filters and effects to images."""

    PRESETS = {
        'vintage': ['sepia', 'saturation:0.8', 'vignette'],
        'film': ['saturation:0.9', 'grain:20', 'contrast:1.1'],
        'instagram': ['contrast:1.2', 'temperature:10', 'vignette'],
        'noir': ['grayscale', 'contrast:1.4', 'vignette:0.9'],
        'warm': ['temperature:20', 'saturation:1.2'],
        'cool': ['temperature:-20', 'saturation:0.9'],
        'dramatic': ['contrast:1.3', 'saturation:1.1', 'brightness:0.9'],
        'dreamy': ['blur:2', 'brightness:1.1', 'contrast:0.9', 'saturation:0.8']
    }

    def __init__(self):
        """Initialize the filter lab."""
        self.image = None
        self.original = None
        self.filepath = None

    def load(self, filepath: str) -> 'ImageFilterLab':
        """Load an image from file."""
        self.filepath = filepath
        self.image = Image.open(filepath).convert('RGB')
        self.original = self.image.copy()
        return self

    def reset(self) -> 'ImageFilterLab':
        """Reset to original image."""
        if self.original:
            self.image = self.original.copy()
        return self

    # Color Filters

    def grayscale(self) -> 'ImageFilterLab':
        """Convert to grayscale."""
        self.image = ImageOps.grayscale(self.image).convert('RGB')
        return self

    def sepia(self, intensity: float = 1.0) -> 'ImageFilterLab':
        """Apply sepia tone."""
        img_array = np.array(self.image, dtype=np.float32)

        # Sepia matrix
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])

        sepia_img = np.dot(img_array, sepia_matrix.T)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)

        # Blend with original based on intensity
        if intensity < 1.0:
            original = np.array(self.image)
            sepia_img = (sepia_img * intensity + original * (1 - intensity)).astype(np.uint8)

        self.image = Image.fromarray(sepia_img)
        return self

    def negative(self) -> 'ImageFilterLab':
        """Invert colors."""
        self.image = ImageOps.invert(self.image)
        return self

    def tint(self, color: Tuple[int, int, int], intensity: float = 0.3) -> 'ImageFilterLab':
        """Apply color tint."""
        img_array = np.array(self.image, dtype=np.float32)
        tint_layer = np.full_like(img_array, color)

        blended = img_array * (1 - intensity) + tint_layer * intensity
        self.image = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
        return self

    # Adjustments

    def brightness(self, factor: float) -> 'ImageFilterLab':
        """Adjust brightness. Factor > 1 brightens, < 1 darkens."""
        enhancer = ImageEnhance.Brightness(self.image)
        self.image = enhancer.enhance(factor)
        return self

    def contrast(self, factor: float) -> 'ImageFilterLab':
        """Adjust contrast. Factor > 1 increases contrast."""
        enhancer = ImageEnhance.Contrast(self.image)
        self.image = enhancer.enhance(factor)
        return self

    def saturation(self, factor: float) -> 'ImageFilterLab':
        """Adjust saturation. Factor > 1 increases, < 1 decreases."""
        enhancer = ImageEnhance.Color(self.image)
        self.image = enhancer.enhance(factor)
        return self

    def temperature(self, value: int) -> 'ImageFilterLab':
        """Adjust color temperature. Positive = warm, negative = cool."""
        img_array = np.array(self.image, dtype=np.float32)

        # Adjust red and blue channels
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] + value, 0, 255)  # Red
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] - value, 0, 255)  # Blue

        self.image = Image.fromarray(img_array.astype(np.uint8))
        return self

    def hue(self, shift: int) -> 'ImageFilterLab':
        """Shift hue by degrees (0-360)."""
        hsv = self.image.convert('HSV')
        h, s, v = hsv.split()

        h_array = np.array(h, dtype=np.int32)
        h_array = (h_array + shift) % 256
        h = Image.fromarray(h_array.astype(np.uint8), mode='L')

        self.image = Image.merge('HSV', (h, s, v)).convert('RGB')
        return self

    # Blur Effects

    def blur(self, radius: int = 5) -> 'ImageFilterLab':
        """Apply Gaussian blur."""
        self.image = self.image.filter(ImageFilter.GaussianBlur(radius))
        return self

    def motion_blur(self, size: int = 15, angle: int = 0) -> 'ImageFilterLab':
        """Apply motion blur."""
        img_array = np.array(self.image)

        # Create motion blur kernel
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = 1.0 / size

        # Rotate kernel for angle
        if angle != 0:
            M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (size, size))

        blurred = cv2.filter2D(img_array, -1, kernel)
        self.image = Image.fromarray(blurred)
        return self

    def radial_blur(self, amount: int = 10) -> 'ImageFilterLab':
        """Apply radial blur from center."""
        # Simplified radial blur using multiple rotated copies
        img_array = np.array(self.image, dtype=np.float32)
        h, w = img_array.shape[:2]
        center = (w // 2, h // 2)

        result = img_array.copy()

        for i in range(1, amount + 1):
            angle = i * 0.5
            M = cv2.getRotationMatrix2D(center, angle, 1)
            rotated = cv2.warpAffine(img_array, M, (w, h))
            result = result * 0.9 + rotated * 0.1

        self.image = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
        return self

    # Sharpen

    def sharpen(self, factor: float = 1.0) -> 'ImageFilterLab':
        """Sharpen image."""
        enhancer = ImageEnhance.Sharpness(self.image)
        self.image = enhancer.enhance(1 + factor)
        return self

    def unsharp_mask(self, radius: int = 2, percent: int = 150, threshold: int = 3) -> 'ImageFilterLab':
        """Apply unsharp mask."""
        self.image = self.image.filter(
            ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
        )
        return self

    # Artistic Effects

    def vintage(self) -> 'ImageFilterLab':
        """Apply vintage filter."""
        self.sepia(0.6)
        self.saturation(0.8)
        self.contrast(1.1)
        self.vignette(0.7, 0.4)
        return self

    def film_grain(self, amount: int = 25) -> 'ImageFilterLab':
        """Add film grain effect."""
        img_array = np.array(self.image, dtype=np.float32)

        # Generate noise
        noise = np.random.normal(0, amount, img_array.shape)
        noisy = img_array + noise

        self.image = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
        return self

    def vignette(self, radius: float = 0.8, intensity: float = 0.5) -> 'ImageFilterLab':
        """Add vignette effect."""
        w, h = self.image.size

        # Create radial gradient
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)

        # Create vignette mask
        vignette_mask = 1 - np.clip((distance - radius) / (1 - radius), 0, 1) * intensity
        vignette_mask = np.stack([vignette_mask] * 3, axis=2)

        img_array = np.array(self.image, dtype=np.float32)
        result = img_array * vignette_mask

        self.image = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
        return self

    def posterize(self, levels: int = 4) -> 'ImageFilterLab':
        """Reduce color levels for poster effect."""
        self.image = ImageOps.posterize(self.image, levels)
        return self

    def solarize(self, threshold: int = 128) -> 'ImageFilterLab':
        """Apply solarize effect."""
        self.image = ImageOps.solarize(self.image, threshold)
        return self

    def emboss(self) -> 'ImageFilterLab':
        """Apply emboss effect."""
        self.image = self.image.filter(ImageFilter.EMBOSS)
        return self

    def edge_enhance(self) -> 'ImageFilterLab':
        """Enhance edges."""
        self.image = self.image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        return self

    # Presets

    def apply_preset(self, preset: str) -> 'ImageFilterLab':
        """Apply a named preset."""
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}")

        for operation in self.PRESETS[preset]:
            self._apply_operation(operation)

        return self

    def _apply_operation(self, operation: str):
        """Apply a single operation string."""
        if ':' in operation:
            name, value = operation.split(':')
            value = float(value)
        else:
            name = operation
            value = None

        method = getattr(self, name, None)
        if method:
            if value is not None:
                method(value)
            else:
                method()

    # Output

    def save(self, filepath: str, quality: int = 95) -> str:
        """Save the processed image."""
        self.image.save(filepath, quality=quality)
        return filepath

    def get_image(self) -> Image:
        """Get the PIL Image object."""
        return self.image

    # Batch Processing

    def batch_process(self, input_dir: str, output_dir: str,
                     filter_func: Callable = None, preset: str = None) -> List[str]:
        """Process multiple images."""
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

                if preset:
                    self.apply_preset(preset)
                elif filter_func:
                    filter_func(self)

                output_file = output_path / f"{img_file.stem}_filtered{img_file.suffix}"
                self.save(str(output_file))
                processed.append(str(output_file))

                print(f"Processed: {img_file.name}")

            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")

        return processed


def main():
    parser = argparse.ArgumentParser(description="Image Filter Lab")

    parser.add_argument("--input", "-i", help="Input image file")
    parser.add_argument("--output", "-o", help="Output image file")

    # Filter presets
    parser.add_argument("--filter", "-f", choices=list(ImageFilterLab.PRESETS.keys()),
                       help="Apply preset filter")

    # Individual filters
    parser.add_argument("--grayscale", action="store_true", help="Convert to grayscale")
    parser.add_argument("--sepia", action="store_true", help="Apply sepia tone")
    parser.add_argument("--negative", action="store_true", help="Invert colors")
    parser.add_argument("--vintage", action="store_true", help="Apply vintage filter")
    parser.add_argument("--vignette", action="store_true", help="Add vignette")

    # Adjustments
    parser.add_argument("--brightness", type=float, help="Brightness factor")
    parser.add_argument("--contrast", type=float, help="Contrast factor")
    parser.add_argument("--saturation", type=float, help="Saturation factor")
    parser.add_argument("--temperature", type=int, help="Color temperature adjustment")
    parser.add_argument("--sharpen", type=float, help="Sharpen factor")

    # Effects
    parser.add_argument("--blur", type=int, help="Blur radius")
    parser.add_argument("--grain", type=int, help="Film grain amount")
    parser.add_argument("--posterize", type=int, help="Posterize levels")

    # Batch
    parser.add_argument("--batch", help="Batch process directory")
    parser.add_argument("--output-dir", help="Output directory for batch")

    args = parser.parse_args()

    lab = ImageFilterLab()

    if args.batch:
        output_dir = args.output_dir or f"{args.batch}_filtered"
        processed = lab.batch_process(args.batch, output_dir, preset=args.filter)
        print(f"\nProcessed {len(processed)} images to {output_dir}")

    elif args.input:
        lab.load(args.input)

        # Apply preset
        if args.filter:
            lab.apply_preset(args.filter)

        # Apply individual filters
        if args.grayscale:
            lab.grayscale()
        if args.sepia:
            lab.sepia()
        if args.negative:
            lab.negative()
        if args.vintage:
            lab.vintage()
        if args.vignette:
            lab.vignette()

        # Adjustments
        if args.brightness:
            lab.brightness(args.brightness)
        if args.contrast:
            lab.contrast(args.contrast)
        if args.saturation:
            lab.saturation(args.saturation)
        if args.temperature:
            lab.temperature(args.temperature)
        if args.sharpen:
            lab.sharpen(args.sharpen)

        # Effects
        if args.blur:
            lab.blur(args.blur)
        if args.grain:
            lab.film_grain(args.grain)
        if args.posterize:
            lab.posterize(args.posterize)

        # Save
        output = args.output or args.input.rsplit('.', 1)[0] + '_filtered.jpg'
        lab.save(output)
        print(f"Saved: {output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
