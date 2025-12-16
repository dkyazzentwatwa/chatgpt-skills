#!/usr/bin/env python3
"""
Thumbnail Generator - Create optimized thumbnails with smart cropping.
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from io import BytesIO

from PIL import Image, ImageFilter
import numpy as np


class ThumbnailGenerator:
    """Generate thumbnails with various cropping and sizing options."""

    PRESETS = {
        "web": [
            (150, 150, "small"),
            (300, 300, "medium"),
            (600, 600, "large")
        ],
        "social": [
            (1080, 1080, "instagram_square"),
            (1080, 1350, "instagram_portrait"),
            (1080, 566, "instagram_landscape"),
            (1200, 675, "twitter"),
            (1200, 630, "facebook"),
            (1200, 627, "linkedin")
        ],
        "icons": [
            (16, 16, "icon_16"),
            (32, 32, "icon_32"),
            (48, 48, "icon_48"),
            (64, 64, "icon_64"),
            (128, 128, "icon_128"),
            (256, 256, "icon_256"),
            (512, 512, "icon_512")
        ],
        "favicon": [
            (16, 16, "favicon_16"),
            (32, 32, "favicon_32"),
            (180, 180, "apple_touch"),
            (192, 192, "android_192"),
            (512, 512, "android_512")
        ]
    }

    def __init__(self):
        """Initialize the generator."""
        self.image: Optional[Image.Image] = None
        self.original: Optional[Image.Image] = None
        self.filepath: Optional[str] = None

    def load(self, filepath: str) -> 'ThumbnailGenerator':
        """
        Load an image file.

        Args:
            filepath: Path to image file

        Returns:
            Self for method chaining
        """
        self.filepath = filepath
        self.original = Image.open(filepath)
        self.image = self.original.copy()

        # Convert to RGB if necessary (for JPEG output)
        if self.image.mode in ('RGBA', 'P'):
            self.image = self.image.convert('RGB')

        return self

    def resize(self, width: int, height: int, crop: str = "fit") -> 'ThumbnailGenerator':
        """
        Resize image to specified dimensions.

        Args:
            width: Target width
            height: Target height
            crop: Resize mode - "fit", "fill", or "stretch"

        Returns:
            Self for method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")

        if crop == "fit":
            # Maintain aspect ratio, fit within bounds
            self.image.thumbnail((width, height), Image.Resampling.LANCZOS)
        elif crop == "fill":
            # Maintain aspect ratio, fill bounds (crop excess)
            self._resize_and_crop(width, height)
        elif crop == "stretch":
            # Ignore aspect ratio
            self.image = self.image.resize((width, height), Image.Resampling.LANCZOS)
        else:
            raise ValueError(f"Unknown crop mode: {crop}")

        return self

    def _resize_and_crop(self, width: int, height: int):
        """Resize to fill and crop to exact dimensions."""
        img = self.image
        img_ratio = img.width / img.height
        target_ratio = width / height

        if img_ratio > target_ratio:
            # Image is wider - resize by height, crop width
            new_height = height
            new_width = int(height * img_ratio)
        else:
            # Image is taller - resize by width, crop height
            new_width = width
            new_height = int(width / img_ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Crop to center
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        right = left + width
        bottom = top + height

        self.image = img.crop((left, top, right, bottom))

    def resize_width(self, width: int) -> 'ThumbnailGenerator':
        """
        Resize to specific width, maintaining aspect ratio.

        Args:
            width: Target width

        Returns:
            Self for method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")

        ratio = width / self.image.width
        height = int(self.image.height * ratio)
        self.image = self.image.resize((width, height), Image.Resampling.LANCZOS)
        return self

    def resize_height(self, height: int) -> 'ThumbnailGenerator':
        """
        Resize to specific height, maintaining aspect ratio.

        Args:
            height: Target height

        Returns:
            Self for method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")

        ratio = height / self.image.height
        width = int(self.image.width * ratio)
        self.image = self.image.resize((width, height), Image.Resampling.LANCZOS)
        return self

    def crop_center(self, width: int, height: int) -> 'ThumbnailGenerator':
        """
        Crop to specified size from center.

        Args:
            width: Crop width
            height: Crop height

        Returns:
            Self for method chaining
        """
        return self.crop_position(width, height, "center")

    def crop_position(self, width: int, height: int,
                     position: str = "center") -> 'ThumbnailGenerator':
        """
        Crop to specified size from given position.

        Args:
            width: Crop width
            height: Crop height
            position: Crop position

        Returns:
            Self for method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")

        img_w, img_h = self.image.size

        # First, resize if image is smaller than crop size
        if img_w < width or img_h < height:
            scale = max(width / img_w, height / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            self.image = self.image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img_w, img_h = self.image.size

        # Calculate crop coordinates
        positions = {
            "center": ((img_w - width) // 2, (img_h - height) // 2),
            "top": ((img_w - width) // 2, 0),
            "bottom": ((img_w - width) // 2, img_h - height),
            "left": (0, (img_h - height) // 2),
            "right": (img_w - width, (img_h - height) // 2),
            "top-left": (0, 0),
            "top-right": (img_w - width, 0),
            "bottom-left": (0, img_h - height),
            "bottom-right": (img_w - width, img_h - height)
        }

        if position not in positions:
            raise ValueError(f"Unknown position: {position}")

        left, top = positions[position]
        left = max(0, left)
        top = max(0, top)

        self.image = self.image.crop((left, top, left + width, top + height))
        return self

    def crop_smart(self, width: int, height: int) -> 'ThumbnailGenerator':
        """
        Smart crop using edge detection to find interesting region.

        Args:
            width: Crop width
            height: Crop height

        Returns:
            Self for method chaining
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # First resize to be slightly larger than target
        img = self.image.copy()
        img_ratio = img.width / img.height
        target_ratio = width / height

        if img_ratio > target_ratio:
            new_height = height
            new_width = int(height * img_ratio)
        else:
            new_width = width
            new_height = int(width / img_ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Use edge detection to find interesting region
        gray = img.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_array = np.array(edges)

        # Find center of mass of edges
        y_indices, x_indices = np.mgrid[0:edge_array.shape[0], 0:edge_array.shape[1]]

        total = edge_array.sum()
        if total > 0:
            center_x = int((x_indices * edge_array).sum() / total)
            center_y = int((y_indices * edge_array).sum() / total)
        else:
            center_x = new_width // 2
            center_y = new_height // 2

        # Calculate crop box centered on point of interest
        left = max(0, min(center_x - width // 2, new_width - width))
        top = max(0, min(center_y - height // 2, new_height - height))

        self.image = img.crop((left, top, left + width, top + height))
        return self

    def save(self, output: str, quality: int = 85, format: str = None) -> str:
        """
        Save thumbnail to file.

        Args:
            output: Output file path
            quality: JPEG/WebP quality (1-100)
            format: Output format (auto-detect from extension if None)

        Returns:
            Output file path
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Detect format from extension
        if format is None:
            ext = Path(output).suffix.lower()
            format_map = {
                '.jpg': 'JPEG',
                '.jpeg': 'JPEG',
                '.png': 'PNG',
                '.webp': 'WEBP',
                '.gif': 'GIF'
            }
            format = format_map.get(ext, 'JPEG')

        # Convert to RGB for JPEG
        img = self.image
        if format == 'JPEG' and img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        # Save with appropriate options
        save_kwargs = {}
        if format in ('JPEG', 'WEBP'):
            save_kwargs['quality'] = quality
        if format == 'PNG':
            save_kwargs['optimize'] = True

        img.save(output, format=format, **save_kwargs)
        return output

    def to_bytes(self, format: str = "JPEG", quality: int = 85) -> bytes:
        """
        Get thumbnail as bytes.

        Args:
            format: Image format
            quality: Quality setting

        Returns:
            Image bytes
        """
        if self.image is None:
            raise ValueError("No image loaded")

        buffer = BytesIO()
        img = self.image

        if format == 'JPEG' and img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        save_kwargs = {}
        if format in ('JPEG', 'WEBP'):
            save_kwargs['quality'] = quality

        img.save(buffer, format=format, **save_kwargs)
        return buffer.getvalue()

    def generate_sizes(self, sizes: List[Tuple[int, int]], output_dir: str,
                      prefix: str = None) -> List[Dict]:
        """
        Generate multiple thumbnail sizes.

        Args:
            sizes: List of (width, height) tuples
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            List of generated file info
        """
        if self.original is None:
            raise ValueError("No image loaded")

        os.makedirs(output_dir, exist_ok=True)

        if prefix is None:
            prefix = Path(self.filepath).stem if self.filepath else "thumb"

        results = []
        ext = ".jpg"

        for width, height in sizes:
            # Reset to original
            self.image = self.original.copy()
            if self.image.mode in ('RGBA', 'P'):
                self.image = self.image.convert('RGB')

            # Resize
            self.resize(width, height, crop="fill")

            # Save
            filename = f"{prefix}_{width}x{height}{ext}"
            output_path = os.path.join(output_dir, filename)
            self.save(output_path)

            results.append({
                "size": f"{width}x{height}",
                "path": output_path,
                "file_size": os.path.getsize(output_path)
            })

        return results

    def apply_preset(self, preset: str, output_dir: str) -> List[Dict]:
        """
        Apply a preset configuration.

        Args:
            preset: Preset name
            output_dir: Output directory

        Returns:
            List of generated file info
        """
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(self.PRESETS.keys())}")

        os.makedirs(output_dir, exist_ok=True)
        results = []

        # Determine format based on preset
        ext = ".png" if preset in ("icons", "favicon") else ".jpg"

        for width, height, name in self.PRESETS[preset]:
            # Reset to original
            self.image = self.original.copy()
            if ext == ".jpg" and self.image.mode in ('RGBA', 'P'):
                self.image = self.image.convert('RGB')

            # Resize
            self.resize(width, height, crop="fill")

            # Save
            output_path = os.path.join(output_dir, f"{name}{ext}")
            self.save(output_path)

            results.append({
                "name": name,
                "size": f"{width}x{height}",
                "path": output_path,
                "file_size": os.path.getsize(output_path)
            })

        return results

    def process_folder(self, input_dir: str, output_dir: str,
                      width: int, height: int,
                      recursive: bool = False) -> List[Dict]:
        """
        Process all images in a folder.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            width: Thumbnail width
            height: Thumbnail height
            recursive: Include subdirectories

        Returns:
            List of processed file info
        """
        supported = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}
        os.makedirs(output_dir, exist_ok=True)

        results = []
        input_path = Path(input_dir)

        if recursive:
            files = input_path.rglob("*")
        else:
            files = input_path.glob("*")

        for file in files:
            if file.suffix.lower() in supported:
                try:
                    self.load(str(file))
                    self.resize(width, height, crop="fill")

                    output_file = os.path.join(output_dir, f"thumb_{file.name}")
                    self.save(output_file)

                    results.append({
                        "input": str(file),
                        "output": output_file,
                        "success": True
                    })
                except Exception as e:
                    results.append({
                        "input": str(file),
                        "error": str(e),
                        "success": False
                    })

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Thumbnail Generator - Create optimized thumbnails"
    )

    parser.add_argument("--input", "-i", required=True, help="Input image or folder")
    parser.add_argument("--output", "-o", help="Output file or folder")
    parser.add_argument("--size", "-s", help="Size as WxH (e.g., 200x200)")
    parser.add_argument("--sizes", help="Multiple sizes as comma-separated WxH")
    parser.add_argument("--preset", "-p", choices=list(ThumbnailGenerator.PRESETS.keys()),
                       help="Use preset sizes")
    parser.add_argument("--crop", "-c", choices=["fit", "fill", "stretch", "smart"],
                       default="fill", help="Crop mode (default: fill)")
    parser.add_argument("--format", "-f", choices=["jpeg", "png", "webp"],
                       help="Output format")
    parser.add_argument("--quality", "-q", type=int, default=85,
                       help="Quality (1-100, default: 85)")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="Process subfolders")

    args = parser.parse_args()

    gen = ThumbnailGenerator()
    input_path = Path(args.input)

    if input_path.is_dir():
        # Batch processing
        if not args.size:
            parser.error("--size required for batch processing")

        w, h = map(int, args.size.split('x'))
        output_dir = args.output or str(input_path / "thumbnails")

        results = gen.process_folder(
            str(input_path), output_dir, w, h,
            recursive=args.recursive
        )

        success = sum(1 for r in results if r["success"])
        print(f"Processed {success}/{len(results)} images")
        print(f"Output: {output_dir}")

    else:
        # Single file
        gen.load(args.input)

        if args.preset:
            output_dir = args.output or "./"
            results = gen.apply_preset(args.preset, output_dir)
            print(f"Generated {len(results)} thumbnails:")
            for r in results:
                print(f"  {r['name']}: {r['size']} ({r['file_size']} bytes)")

        elif args.sizes:
            sizes = [tuple(map(int, s.split('x'))) for s in args.sizes.split(',')]
            output_dir = args.output or "./"
            results = gen.generate_sizes(sizes, output_dir)
            print(f"Generated {len(results)} thumbnails:")
            for r in results:
                print(f"  {r['size']}: {r['path']}")

        elif args.size:
            w, h = map(int, args.size.split('x'))

            if args.crop == "smart":
                gen.crop_smart(w, h)
            else:
                gen.resize(w, h, crop=args.crop)

            output = args.output or f"thumb_{input_path.name}"

            format_arg = args.format.upper() if args.format else None
            gen.save(output, quality=args.quality, format=format_arg)
            print(f"Saved: {output}")

        else:
            parser.error("Specify --size, --sizes, or --preset")


if __name__ == "__main__":
    main()
