#!/usr/bin/env python3
"""
Video Clipper - Cut and trim video segments.
"""

import argparse
from pathlib import Path
from typing import List

from moviepy.editor import VideoFileClip


class VideoClipper:
    """Clip and trim videos."""

    def __init__(self):
        """Initialize clipper."""
        self.clip = None

    def load(self, filepath: str) -> 'VideoClipper':
        """Load video."""
        self.clip = VideoFileClip(filepath)
        return self

    def extract_segment(self, start: float, end: float, output: str) -> str:
        """Extract segment between start and end times."""
        subclip = self.clip.subclip(start, end)
        subclip.write_videofile(output, logger=None)
        subclip.close()
        return output

    def split_by_duration(self, chunk_duration: float, output_dir: str) -> List[str]:
        """Split video into chunks of specified duration."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        duration = self.clip.duration
        chunks = []

        i = 0
        current_time = 0

        while current_time < duration:
            end_time = min(current_time + chunk_duration, duration)

            output_file = output_path / f"chunk_{i:03d}.mp4"
            subclip = self.clip.subclip(current_time, end_time)
            subclip.write_videofile(str(output_file), logger=None)
            subclip.close()

            chunks.append(str(output_file))
            current_time = end_time
            i += 1

        return chunks

    def trim(self, start: float = None, end: float = None, output: str = None) -> str:
        """Trim start and/or end of video."""
        start = start or 0
        end = end or self.clip.duration

        trimmed = self.clip.subclip(start, end)
        trimmed.write_videofile(output, logger=None)
        trimmed.close()

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
    parser = argparse.ArgumentParser(description="Video Clipper")

    parser.add_argument("--input", "-i", required=True, help="Input video")
    parser.add_argument("--output", "-o", required=True, help="Output file/directory")

    parser.add_argument("--start", help="Start time (HH:MM:SS or seconds)")
    parser.add_argument("--end", help="End time (HH:MM:SS or seconds)")
    parser.add_argument("--split", type=float, help="Split into chunks (duration in seconds)")

    args = parser.parse_args()

    clipper = VideoClipper()
    clipper.load(args.input)

    if args.split:
        chunks = clipper.split_by_duration(args.split, args.output)
        print(f"Split into {len(chunks)} chunks in {args.output}/")

    elif args.start or args.end:
        start = parse_time(args.start) if args.start else 0
        end = parse_time(args.end) if args.end else clipper.clip.duration

        clipper.extract_segment(start, end, args.output)
        print(f"Clip extracted ({args.start or '0'} - {args.end or 'end'}) → {args.output}")

    clipper.close()


if __name__ == "__main__":
    main()
