#!/usr/bin/env python3
"""
Podcast Splitter - Split audio files by detecting silence.

Features:
- Silence detection with configurable threshold
- Auto-split into segments/chapters
- Silence removal and shortening
- Batch processing
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class PodcastSplitter:
    """Split audio files based on silence detection."""

    def __init__(
        self,
        filepath: str,
        silence_thresh: float = -40,
        min_silence_len: int = 1000,
        keep_silence: int = 300
    ):
        """
        Initialize splitter with audio file.

        Args:
            filepath: Path to input audio file
            silence_thresh: Silence threshold in dBFS (default -40)
            min_silence_len: Minimum silence length to detect in ms (default 1000)
            keep_silence: Silence to keep at segment edges in ms (default 300)
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        self.silence_thresh = silence_thresh
        self.min_silence_len = min_silence_len
        self.keep_silence = keep_silence

        self._audio = None
        self._segments: List[Dict] = []
        self._silences: List[Tuple[int, int]] = []

        self._load_audio()

    def _load_audio(self):
        """Load audio file using pydub."""
        try:
            from pydub import AudioSegment
            self._audio = AudioSegment.from_file(str(self.filepath))
        except ImportError:
            raise ImportError("pydub is required. Install with: pip install pydub")
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {e}")

    def detect_silence(self) -> List[Tuple[int, int]]:
        """
        Detect silence regions in the audio.

        Returns:
            List of (start_ms, end_ms) tuples for each silence region
        """
        from pydub.silence import detect_silence

        self._silences = detect_silence(
            self._audio,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh
        )

        return self._silences

    def print_silence_report(self):
        """Print a report of detected silences."""
        if not self._silences:
            self.detect_silence()

        total_duration = len(self._audio)
        total_silence = sum(end - start for start, end in self._silences)
        content_duration = total_duration - total_silence

        print(f"\nSilence Detection Report: {self.filepath.name}")
        print("=" * 50)
        print(f"Total Duration:   {self._format_time(total_duration)}")
        print(f"Content Duration: {self._format_time(content_duration)}")
        print(f"Total Silence:    {self._format_time(total_silence)}")
        print(f"Silence Regions:  {len(self._silences)}")
        print(f"\nSettings:")
        print(f"  Threshold:      {self.silence_thresh} dBFS")
        print(f"  Min Length:     {self.min_silence_len} ms")

        if self._silences:
            print(f"\nDetected Silences:")
            for i, (start, end) in enumerate(self._silences[:20]):  # Show first 20
                duration = end - start
                print(f"  {i+1:3d}. {self._format_time(start)} - {self._format_time(end)} ({duration}ms)")

            if len(self._silences) > 20:
                print(f"  ... and {len(self._silences) - 20} more")

    @staticmethod
    def _format_time(ms: int) -> str:
        """Format milliseconds as MM:SS."""
        seconds = ms / 1000
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"

    def split_by_silence(
        self,
        min_silence_len: Optional[int] = None,
        max_segments: Optional[int] = None
    ) -> List[Dict]:
        """
        Split audio at silence regions.

        Args:
            min_silence_len: Override minimum silence length for splitting
            max_segments: Maximum number of segments to create

        Returns:
            List of segment info dicts with start, end, duration
        """
        from pydub.silence import split_on_silence

        # Use overrides if provided
        silence_len = min_silence_len or self.min_silence_len

        # Perform split
        chunks = split_on_silence(
            self._audio,
            min_silence_len=silence_len,
            silence_thresh=self.silence_thresh,
            keep_silence=self.keep_silence
        )

        # Build segment info
        self._segments = []
        position = 0

        for i, chunk in enumerate(chunks):
            if max_segments and i >= max_segments:
                # Merge remaining chunks
                remaining = chunks[i:]
                merged = sum(remaining[1:], remaining[0])
                self._segments.append({
                    'index': i,
                    'start': position,
                    'end': position + len(merged),
                    'duration': len(merged),
                    'audio': merged
                })
                break

            self._segments.append({
                'index': i,
                'start': position,
                'end': position + len(chunk),
                'duration': len(chunk),
                'audio': chunk
            })
            position += len(chunk)

        return [
            {'index': s['index'], 'start': s['start'], 'end': s['end'], 'duration': s['duration']}
            for s in self._segments
        ]

    def remove_silence(self, min_length: int = 2000) -> 'PodcastSplitter':
        """
        Remove silences longer than specified length.

        Args:
            min_length: Remove silences longer than this (ms)

        Returns:
            Self for chaining
        """
        from pydub import AudioSegment

        # Detect silences
        if not self._silences:
            self.detect_silence()

        # Build new audio without long silences
        result = AudioSegment.empty()
        last_end = 0

        for start, end in self._silences:
            duration = end - start

            # Add content before this silence
            result += self._audio[last_end:start]

            # If silence is short enough, keep it
            if duration < min_length:
                result += self._audio[start:end]

            last_end = end

        # Add remaining content
        result += self._audio[last_end:]

        self._audio = result
        self._silences = []  # Reset for re-detection

        return self

    def shorten_silence(self, max_length: int = 500) -> 'PodcastSplitter':
        """
        Shorten all silences to maximum length.

        Args:
            max_length: Maximum silence length (ms)

        Returns:
            Self for chaining
        """
        from pydub import AudioSegment

        # Detect silences
        if not self._silences:
            self.detect_silence()

        # Build new audio with shortened silences
        result = AudioSegment.empty()
        last_end = 0

        for start, end in self._silences:
            duration = end - start

            # Add content before this silence
            result += self._audio[last_end:start]

            # Add shortened silence
            silence_to_keep = min(duration, max_length)
            result += self._audio[start:start + silence_to_keep]

            last_end = end

        # Add remaining content
        result += self._audio[last_end:]

        self._audio = result
        self._silences = []  # Reset for re-detection

        return self

    def strip_silence(self, threshold: Optional[float] = None) -> 'PodcastSplitter':
        """
        Remove leading and trailing silence only.

        Args:
            threshold: Silence threshold (optional, uses instance default)

        Returns:
            Self for chaining
        """
        from pydub.silence import detect_leading_silence

        thresh = threshold or self.silence_thresh

        # Strip leading silence
        start_trim = detect_leading_silence(self._audio, silence_threshold=thresh)

        # Strip trailing silence
        reversed_audio = self._audio.reverse()
        end_trim = detect_leading_silence(reversed_audio, silence_threshold=thresh)

        if start_trim + end_trim < len(self._audio):
            self._audio = self._audio[start_trim:len(self._audio) - end_trim]

        return self

    def export_segments(
        self,
        output_dir: str,
        prefix: str = "segment",
        format: str = "mp3",
        bitrate: int = 192
    ) -> List[str]:
        """
        Export all segments to files.

        Args:
            output_dir: Output directory path
            prefix: Filename prefix
            format: Output format
            bitrate: Bitrate for lossy formats

        Returns:
            List of exported file paths
        """
        if not self._segments:
            self.split_by_silence()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported = []
        for segment in self._segments:
            filename = f"{prefix}_{segment['index'] + 1:02d}.{format}"
            filepath = output_path / filename

            params = {}
            if format in ('mp3', 'ogg', 'm4a'):
                params['bitrate'] = f"{bitrate}k"

            segment['audio'].export(str(filepath), format=format, **params)
            exported.append(str(filepath))

            duration_str = self._format_time(segment['duration'])
            print(f"Exported: {filename} ({duration_str})")

        return exported

    def export_segment(
        self,
        index: int,
        output: str,
        format: Optional[str] = None,
        bitrate: int = 192
    ) -> str:
        """
        Export a specific segment.

        Args:
            index: Segment index (0-based)
            output: Output file path
            format: Output format (from extension if not specified)
            bitrate: Bitrate for lossy formats

        Returns:
            Path to exported file
        """
        if not self._segments:
            self.split_by_silence()

        if index < 0 or index >= len(self._segments):
            raise IndexError(f"Segment index {index} out of range (0-{len(self._segments)-1})")

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = output_path.suffix.lstrip('.').lower()

        params = {}
        if format in ('mp3', 'ogg', 'm4a'):
            params['bitrate'] = f"{bitrate}k"

        self._segments[index]['audio'].export(str(output_path), format=format, **params)

        return str(output_path)

    def save(
        self,
        output: str,
        format: Optional[str] = None,
        bitrate: int = 192
    ) -> str:
        """
        Save the (possibly modified) audio.

        Args:
            output: Output file path
            format: Output format
            bitrate: Bitrate for lossy formats

        Returns:
            Path to saved file
        """
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = output_path.suffix.lstrip('.').lower()

        params = {}
        if format in ('mp3', 'ogg', 'm4a'):
            params['bitrate'] = f"{bitrate}k"

        self._audio.export(str(output_path), format=format, **params)

        return str(output_path)

    def get_segment_count(self) -> int:
        """Get number of segments after splitting."""
        if not self._segments:
            self.split_by_silence()
        return len(self._segments)

    def get_duration(self) -> int:
        """Get current audio duration in milliseconds."""
        return len(self._audio)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Split audio files by detecting silence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input episode.mp3 --output-dir ./chapters/
  %(prog)s --input episode.mp3 --detect-only
  %(prog)s --input raw.mp3 --output clean.mp3 --remove-silence 2000
  %(prog)s --input episode.mp3 --output-dir ./chapters/ --threshold -35 --min-silence 2000
        """
    )

    parser.add_argument('--input', '-i', required=True, help='Input audio file')
    parser.add_argument('--output', '-o', help='Output file (for silence removal)')
    parser.add_argument('--output-dir', '-d', help='Output directory for segments')
    parser.add_argument('--detect-only', action='store_true', help='Only detect/report silences')
    parser.add_argument('--threshold', '-t', type=float, default=-40, help='Silence threshold (dBFS)')
    parser.add_argument('--min-silence', '-m', type=int, default=1000, help='Minimum silence length (ms)')
    parser.add_argument('--keep-silence', '-k', type=int, default=300, help='Silence to keep at edges (ms)')
    parser.add_argument('--max-segments', type=int, help='Maximum segments to create')
    parser.add_argument('--remove-silence', type=int, help='Remove silences longer than (ms)')
    parser.add_argument('--shorten-silence', type=int, help='Cap silence length at (ms)')
    parser.add_argument('--strip', action='store_true', help='Strip leading/trailing silence')
    parser.add_argument('--prefix', default='segment', help='Output filename prefix')
    parser.add_argument('--format', '-f', default='mp3', help='Output format')
    parser.add_argument('--bitrate', '-b', type=int, default=192, help='Output bitrate (kbps)')

    args = parser.parse_args()

    # Initialize splitter
    splitter = PodcastSplitter(
        args.input,
        silence_thresh=args.threshold,
        min_silence_len=args.min_silence,
        keep_silence=args.keep_silence
    )

    # Detect-only mode
    if args.detect_only:
        splitter.print_silence_report()
        return

    # Silence modification modes
    if args.remove_silence:
        print(f"Removing silences > {args.remove_silence}ms...")
        splitter.remove_silence(args.remove_silence)

    if args.shorten_silence:
        print(f"Shortening silences to max {args.shorten_silence}ms...")
        splitter.shorten_silence(args.shorten_silence)

    if args.strip:
        print("Stripping leading/trailing silence...")
        splitter.strip_silence()

    # Output handling
    if args.output:
        # Save modified audio
        output = splitter.save(args.output, format=args.format, bitrate=args.bitrate)
        duration = splitter._format_time(splitter.get_duration())
        print(f"Saved: {output} ({duration})")

    elif args.output_dir:
        # Split and export segments
        segments = splitter.split_by_silence(max_segments=args.max_segments)
        print(f"Found {len(segments)} segments")

        exported = splitter.export_segments(
            args.output_dir,
            prefix=args.prefix,
            format=args.format,
            bitrate=args.bitrate
        )
        print(f"\nExported {len(exported)} segments to {args.output_dir}")

    else:
        # Just show detection results
        splitter.print_silence_report()
        segments = splitter.split_by_silence()
        print(f"\nWould create {len(segments)} segments")


if __name__ == "__main__":
    main()
