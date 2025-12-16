#!/usr/bin/env python3
"""
Timelapse Creator - Create timelapse videos from images.
"""

import argparse
from pathlib import Path
from typing import List

from moviepy.editor import ImageSequenceClip
from PIL import Image


class TimelapseCreator:
    """Create timelapse videos."""

    def __init__(self):
        """Initialize creator."""
        self.images = []

    def load_images_from_dir(self, directory: str, pattern: str = '*') -> 'TimelapseCreator':
        """Load images from directory."""
        path = Path(directory)
        self.images = sorted([
            str(p) for p in path.glob(pattern)
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])

        return self

    def create_timelapse(self, output: str, fps: int = 30) -> str:
        """Create timelapse video."""
        if not self.images:
            raise ValueError("No images loaded")

        print(f"Creating timelapse from {len(self.images)} images at {fps} FPS...")

        clip = ImageSequenceClip(self.images, fps=fps)
        clip.write_videofile(output, logger=None)

        return output


def main():
    parser = argparse.ArgumentParser(description="Timelapse Creator")

    parser.add_argument("--input", "-i", required=True, help="Input directory with images")
    parser.add_argument("--output", "-o", required=True, help="Output video file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--pattern", default="*", help="File pattern (e.g., '*.jpg')")

    args = parser.parse_args()

    creator = TimelapseCreator()
    creator.load_images_from_dir(args.input, pattern=args.pattern)

    print(f"Found {len(creator.images)} images")

    creator.create_timelapse(args.output, fps=args.fps)

    print(f"\nTimelapse created: {args.output}")


if __name__ == "__main__":
    main()
