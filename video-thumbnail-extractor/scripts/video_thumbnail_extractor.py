#!/usr/bin/env python3
"""
Video Thumbnail Extractor - Extract frames from videos.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np


class VideoThumbnailExtractor:
    """Extract thumbnails from videos."""

    def __init__(self):
        """Initialize extractor."""
        self.clip = None

    def load(self, filepath: str) -> 'VideoThumbnailExtractor':
        """Load video."""
        self.clip = VideoFileClip(filepath)
        return self

    def extract_at_time(self, time: float, output: str) -> str:
        """Extract frame at specific time (seconds)."""
        frame = self.clip.get_frame(time)
        img = Image.fromarray(frame)
        img.save(output)
        return output

    def extract_interval(self, interval: float, output_dir: str) -> List[str]:
        """Extract frames at regular intervals."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        duration = self.clip.duration
        times = np.arange(0, duration, interval)

        frames = []
        for i, t in enumerate(times):
            frame = self.clip.get_frame(t)
            output_file = output_path / f"frame_{i:04d}.jpg"
            img = Image.fromarray(frame)
            img.save(str(output_file))
            frames.append(str(output_file))

        return frames

    def create_grid(self, grid: Tuple[int, int], output: str) -> str:
        """Create thumbnail grid preview."""
        cols, rows = grid
        n_thumbs = cols * rows

        duration = self.clip.duration
        times = np.linspace(0, duration * 0.95, n_thumbs)

        # Get first frame to determine size
        first_frame = self.clip.get_frame(times[0])
        h, w = first_frame.shape[:2]

        # Calculate thumbnail size
        thumb_w = w // 4
        thumb_h = h // 4

        # Create grid image
        grid_w = cols * thumb_w
        grid_h = rows * thumb_h
        grid_img = Image.new('RGB', (grid_w, grid_h))

        for i, t in enumerate(times):
            row = i // cols
            col = i % cols

            frame = self.clip.get_frame(t)
            thumb = Image.fromarray(frame).resize((thumb_w, thumb_h))

            x = col * thumb_w
            y = row * thumb_h
            grid_img.paste(thumb, (x, y))

        grid_img.save(output)
        return output

    def close(self):
        """Close video clip."""
        if self.clip:
            self.clip.close()


def parse_time(time_str: str) -> float:
    """Parse time string (HH:MM:SS or seconds)."""
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(float, parts)
            return m * 60 + s
    return float(time_str)


def main():
    parser = argparse.ArgumentParser(description="Video Thumbnail Extractor")

    parser.add_argument("--input", "-i", required=True, help="Input video")
    parser.add_argument("--output", "-o", required=True, help="Output file/directory")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--time", help="Extract at time (HH:MM:SS or seconds)")
    group.add_argument("--interval", type=float, help="Interval in seconds")
    group.add_argument("--grid", help="Grid size (e.g., 4x4)")

    args = parser.parse_args()

    extractor = VideoThumbnailExtractor()
    extractor.load(args.input)

    if args.time:
        time_sec = parse_time(args.time)
        extractor.extract_at_time(time_sec, args.output)
        print(f"Frame extracted at {args.time} → {args.output}")

    elif args.interval:
        frames = extractor.extract_interval(args.interval, args.output)
        print(f"Extracted {len(frames)} frames to {args.output}/")

    elif args.grid:
        cols, rows = map(int, args.grid.split('x'))
        extractor.create_grid((cols, rows), args.output)
        print(f"Grid preview ({args.grid}) → {args.output}")

    extractor.close()


if __name__ == "__main__":
    main()
