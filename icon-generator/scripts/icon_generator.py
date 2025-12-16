#!/usr/bin/env python3
"""
Icon Generator - Generate app icons in multiple sizes from a single source image.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from PIL import Image, ImageDraw


class IconGenerator:
    """Generate app icons for multiple platforms."""

    # Platform-specific sizes
    IOS_SIZES = [
        (20, 1), (20, 2), (20, 3),     # Notification
        (29, 1), (29, 2), (29, 3),     # Settings
        (40, 1), (40, 2), (40, 3),     # Spotlight
        (60, 2), (60, 3),              # iPhone App
        (76, 1), (76, 2),              # iPad App
        (83.5, 2),                      # iPad Pro App
        (1024, 1),                      # App Store
    ]

    ANDROID_SIZES = {
        'ldpi': 36,
        'mdpi': 48,
        'hdpi': 72,
        'xhdpi': 96,
        'xxhdpi': 144,
        'xxxhdpi': 192,
        'play_store': 512
    }

    FAVICON_SIZES = [16, 32, 48, 180, 192, 512]

    MACOS_SIZES = [16, 32, 64, 128, 256, 512, 1024]

    WINDOWS_SIZES = [16, 32, 48, 256]

    PWA_SIZES = [72, 96, 128, 144, 152, 192, 384, 512]

    def __init__(self):
        """Initialize the generator."""
        self.image = None
        self.filepath = None
        self.rounding = 0
        self.padding = 0
        self.background = None

    def load(self, filepath: str) -> 'IconGenerator':
        """Load source image."""
        self.filepath = filepath
        self.image = Image.open(filepath).convert('RGBA')
        return self

    def set_rounding(self, radius_percent: float) -> 'IconGenerator':
        """Set corner rounding as percentage of size."""
        self.rounding = radius_percent
        return self

    def set_padding(self, padding_percent: float) -> 'IconGenerator':
        """Set padding as percentage of size."""
        self.padding = padding_percent
        return self

    def set_background(self, color: Tuple[int, int, int, int]) -> 'IconGenerator':
        """Set background color (RGBA)."""
        self.background = color
        return self

    def _resize_image(self, size: int) -> Image:
        """Resize image to target size with optional effects."""
        # Calculate padding
        if self.padding > 0:
            padding = int(size * self.padding / 100)
            inner_size = size - (2 * padding)
        else:
            padding = 0
            inner_size = size

        # Resize
        resized = self.image.resize((inner_size, inner_size), Image.Resampling.LANCZOS)

        # Create canvas
        if self.background:
            canvas = Image.new('RGBA', (size, size), self.background)
        else:
            canvas = Image.new('RGBA', (size, size), (0, 0, 0, 0))

        # Apply rounding if specified
        if self.rounding > 0:
            resized = self._apply_rounding(resized, inner_size)

        # Paste centered
        canvas.paste(resized, (padding, padding), resized)

        return canvas

    def _apply_rounding(self, img: Image, size: int) -> Image:
        """Apply rounded corners to image."""
        radius = int(size * self.rounding / 100)

        # Create rounded mask
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), (size, size)], radius=radius, fill=255)

        # Apply mask
        result = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        result.paste(img, (0, 0), mask)

        return result

    def generate_ios(self, output_dir: str) -> List[str]:
        """Generate iOS app icons."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated = []

        for base_size, scale in self.IOS_SIZES:
            size = int(base_size * scale)
            filename = f"Icon-{base_size}@{scale}x.png"

            if base_size == int(base_size):
                filename = f"Icon-{int(base_size)}@{scale}x.png"
            else:
                filename = f"Icon-{base_size}@{scale}x.png"

            img = self._resize_image(size)
            filepath = output_path / filename
            img.save(filepath, 'PNG')
            generated.append(str(filepath))

        return generated

    def generate_android(self, output_dir: str) -> List[str]:
        """Generate Android app icons."""
        output_path = Path(output_dir)
        generated = []

        for density, size in self.ANDROID_SIZES.items():
            if density == 'play_store':
                folder = output_path
            else:
                folder = output_path / f"mipmap-{density}"

            folder.mkdir(parents=True, exist_ok=True)

            img = self._resize_image(size)

            if density == 'play_store':
                filepath = folder / "ic_launcher_playstore.png"
            else:
                filepath = folder / "ic_launcher.png"

            img.save(filepath, 'PNG')
            generated.append(str(filepath))

        return generated

    def generate_favicon(self, output_dir: str) -> List[str]:
        """Generate favicon files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated = []

        for size in self.FAVICON_SIZES:
            img = self._resize_image(size)

            if size == 180:
                filename = "apple-touch-icon.png"
            elif size in [192, 512]:
                filename = f"icon-{size}x{size}.png"
            else:
                filename = f"favicon-{size}x{size}.png"

            filepath = output_path / filename
            img.save(filepath, 'PNG')
            generated.append(str(filepath))

        # Generate ICO file (multi-resolution)
        ico_sizes = [16, 32, 48]
        ico_images = [self._resize_image(s).convert('RGBA') for s in ico_sizes]

        ico_path = output_path / "favicon.ico"
        ico_images[0].save(ico_path, format='ICO',
                          sizes=[(s, s) for s in ico_sizes])
        generated.append(str(ico_path))

        return generated

    def generate_macos(self, output_dir: str) -> List[str]:
        """Generate macOS app icons."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated = []

        for size in self.MACOS_SIZES:
            img = self._resize_image(size)
            filename = f"icon_{size}x{size}.png"
            filepath = output_path / filename
            img.save(filepath, 'PNG')
            generated.append(str(filepath))

            # @2x variant
            if size <= 512:
                img_2x = self._resize_image(size * 2)
                filename_2x = f"icon_{size}x{size}@2x.png"
                filepath_2x = output_path / filename_2x
                img_2x.save(filepath_2x, 'PNG')
                generated.append(str(filepath_2x))

        return generated

    def generate_windows(self, output_dir: str) -> List[str]:
        """Generate Windows icons."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated = []

        # Individual PNGs
        for size in self.WINDOWS_SIZES:
            img = self._resize_image(size)
            filename = f"icon_{size}x{size}.png"
            filepath = output_path / filename
            img.save(filepath, 'PNG')
            generated.append(str(filepath))

        # ICO with all sizes
        ico_images = [self._resize_image(s).convert('RGBA') for s in self.WINDOWS_SIZES]
        ico_path = output_path / "icon.ico"
        ico_images[0].save(ico_path, format='ICO',
                          sizes=[(s, s) for s in self.WINDOWS_SIZES])
        generated.append(str(ico_path))

        return generated

    def generate_pwa(self, output_dir: str) -> List[str]:
        """Generate PWA manifest icons."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated = []

        for size in self.PWA_SIZES:
            img = self._resize_image(size)
            filename = f"icon-{size}x{size}.png"
            filepath = output_path / filename
            img.save(filepath, 'PNG')
            generated.append(str(filepath))

        return generated

    def generate_all(self, output_dir: str) -> Dict[str, List[str]]:
        """Generate icons for all platforms."""
        output_path = Path(output_dir)

        return {
            'ios': self.generate_ios(output_path / 'ios'),
            'android': self.generate_android(output_path / 'android'),
            'favicon': self.generate_favicon(output_path / 'favicon'),
            'macos': self.generate_macos(output_path / 'macos'),
            'windows': self.generate_windows(output_path / 'windows'),
            'pwa': self.generate_pwa(output_path / 'pwa')
        }

    def generate_sizes(self, sizes: List[int], output_dir: str,
                      prefix: str = "icon") -> List[str]:
        """Generate custom sizes."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated = []

        for size in sizes:
            img = self._resize_image(size)
            filename = f"{prefix}_{size}x{size}.png"
            filepath = output_path / filename
            img.save(filepath, 'PNG')
            generated.append(str(filepath))

        return generated

    def generate_single(self, size: int, output: str) -> str:
        """Generate a single icon."""
        img = self._resize_image(size)
        img.save(output, 'PNG')
        return output


def main():
    parser = argparse.ArgumentParser(description="Icon Generator")

    parser.add_argument("--input", "-i", required=True, help="Input image file")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")

    parser.add_argument("--preset", choices=['ios', 'android', 'favicon', 'macos',
                                             'windows', 'pwa', 'all'],
                       help="Platform preset")
    parser.add_argument("--sizes", nargs='+', type=int, help="Custom sizes")

    parser.add_argument("--rounding", type=float, default=0,
                       help="Corner rounding (percentage)")
    parser.add_argument("--padding", type=float, default=0,
                       help="Padding (percentage)")
    parser.add_argument("--background", help="Background color (R,G,B or R,G,B,A)")

    args = parser.parse_args()

    gen = IconGenerator()
    gen.load(args.input)

    # Apply options
    if args.rounding > 0:
        gen.set_rounding(args.rounding)

    if args.padding > 0:
        gen.set_padding(args.padding)

    if args.background:
        parts = [int(x) for x in args.background.split(',')]
        if len(parts) == 3:
            parts.append(255)
        gen.set_background(tuple(parts))

    # Generate icons
    if args.sizes:
        generated = gen.generate_sizes(args.sizes, args.output_dir)
        print(f"Generated {len(generated)} custom icons")

    elif args.preset:
        if args.preset == 'all':
            result = gen.generate_all(args.output_dir)
            total = sum(len(v) for v in result.values())
            print(f"Generated {total} icons for all platforms")
            for platform, files in result.items():
                print(f"  {platform}: {len(files)} icons")
        elif args.preset == 'ios':
            generated = gen.generate_ios(args.output_dir)
            print(f"Generated {len(generated)} iOS icons")
        elif args.preset == 'android':
            generated = gen.generate_android(args.output_dir)
            print(f"Generated {len(generated)} Android icons")
        elif args.preset == 'favicon':
            generated = gen.generate_favicon(args.output_dir)
            print(f"Generated {len(generated)} favicon files")
        elif args.preset == 'macos':
            generated = gen.generate_macos(args.output_dir)
            print(f"Generated {len(generated)} macOS icons")
        elif args.preset == 'windows':
            generated = gen.generate_windows(args.output_dir)
            print(f"Generated {len(generated)} Windows icons")
        elif args.preset == 'pwa':
            generated = gen.generate_pwa(args.output_dir)
            print(f"Generated {len(generated)} PWA icons")

    else:
        # Default to all platforms
        result = gen.generate_all(args.output_dir)
        total = sum(len(v) for v in result.values())
        print(f"Generated {total} icons for all platforms")

    print(f"\nOutput directory: {args.output_dir}")


if __name__ == "__main__":
    main()
