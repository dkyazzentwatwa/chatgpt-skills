#!/usr/bin/env python3
"""
Audio Trimmer - Cut, trim, and edit audio segments.

Features:
- Precise trimming by timestamp
- Fade in/out effects
- Speed control
- Concatenation with crossfade
- Basic audio effects
- Volume adjustment
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union


class AudioTrimmer:
    """Cut, trim, and edit audio segments."""

    def __init__(self, filepath: str):
        """
        Initialize trimmer with audio file.

        Args:
            filepath: Path to input audio file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        self._audio = None
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

    @staticmethod
    def _parse_timestamp(ts: str) -> int:
        """
        Parse timestamp string to milliseconds.

        Args:
            ts: Timestamp string (HH:MM:SS, MM:SS, or seconds)

        Returns:
            Milliseconds
        """
        if isinstance(ts, (int, float)):
            return int(ts * 1000) if ts < 1000 else int(ts)

        ts = str(ts).strip()

        # Try HH:MM:SS.ms or HH:MM:SS
        match = re.match(r'^(\d+):(\d{2}):(\d{2})(?:\.(\d+))?$', ts)
        if match:
            h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
            ms = int(match.group(4) or 0)
            return (h * 3600 + m * 60 + s) * 1000 + ms

        # Try MM:SS.ms or MM:SS
        match = re.match(r'^(\d+):(\d{2})(?:\.(\d+))?$', ts)
        if match:
            m, s = int(match.group(1)), int(match.group(2))
            ms = int(match.group(3) or 0)
            return (m * 60 + s) * 1000 + ms

        # Try seconds.ms
        match = re.match(r'^(\d+)(?:\.(\d+))?$', ts)
        if match:
            s = int(match.group(1))
            ms = int(match.group(2) or 0)
            return s * 1000 + ms

        raise ValueError(f"Invalid timestamp format: {ts}")

    def trim(
        self,
        start: Optional[Union[str, int]] = None,
        end: Optional[Union[str, int]] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None
    ) -> 'AudioTrimmer':
        """
        Trim audio to segment.

        Args:
            start: Start timestamp (HH:MM:SS, MM:SS, or seconds)
            end: End timestamp
            start_ms: Start position in milliseconds
            end_ms: End position in milliseconds

        Returns:
            Self for chaining
        """
        # Parse timestamps
        if start is not None:
            start_ms = self._parse_timestamp(start)
        if end is not None:
            end_ms = self._parse_timestamp(end)

        # Default values
        if start_ms is None:
            start_ms = 0
        if end_ms is None:
            end_ms = len(self._audio)

        # Validate
        if start_ms < 0:
            start_ms = 0
        if end_ms > len(self._audio):
            end_ms = len(self._audio)
        if start_ms >= end_ms:
            raise ValueError("Start must be before end")

        self._audio = self._audio[start_ms:end_ms]
        return self

    def fade_in(self, duration_ms: int) -> 'AudioTrimmer':
        """
        Apply fade in effect.

        Args:
            duration_ms: Fade duration in milliseconds

        Returns:
            Self for chaining
        """
        if duration_ms > len(self._audio):
            duration_ms = len(self._audio)

        self._audio = self._audio.fade_in(duration_ms)
        return self

    def fade_out(self, duration_ms: int) -> 'AudioTrimmer':
        """
        Apply fade out effect.

        Args:
            duration_ms: Fade duration in milliseconds

        Returns:
            Self for chaining
        """
        if duration_ms > len(self._audio):
            duration_ms = len(self._audio)

        self._audio = self._audio.fade_out(duration_ms)
        return self

    def speed(self, factor: float) -> 'AudioTrimmer':
        """
        Change playback speed (affects pitch).

        Args:
            factor: Speed multiplier (1.5 = 50% faster, 0.5 = half speed)

        Returns:
            Self for chaining
        """
        if factor <= 0:
            raise ValueError("Speed factor must be positive")

        # Change frame rate to adjust speed
        new_frame_rate = int(self._audio.frame_rate * factor)
        self._audio = self._audio._spawn(
            self._audio.raw_data,
            overrides={'frame_rate': new_frame_rate}
        ).set_frame_rate(self._audio.frame_rate)

        return self

    def reverse(self) -> 'AudioTrimmer':
        """
        Reverse the audio.

        Returns:
            Self for chaining
        """
        self._audio = self._audio.reverse()
        return self

    def loop(self, times: int) -> 'AudioTrimmer':
        """
        Loop the audio N times.

        Args:
            times: Number of times to repeat

        Returns:
            Self for chaining
        """
        if times < 1:
            raise ValueError("Times must be at least 1")

        self._audio = self._audio * times
        return self

    def gain(self, db: float) -> 'AudioTrimmer':
        """
        Adjust volume by dB.

        Args:
            db: Volume change in decibels (positive = louder)

        Returns:
            Self for chaining
        """
        self._audio = self._audio + db
        return self

    def normalize(self, target_dbfs: float = -3.0) -> 'AudioTrimmer':
        """
        Normalize audio to target level.

        Args:
            target_dbfs: Target level in dBFS

        Returns:
            Self for chaining
        """
        change_in_dbfs = target_dbfs - self._audio.dBFS
        self._audio = self._audio.apply_gain(change_in_dbfs)
        return self

    def add_silence_start(self, duration_ms: int) -> 'AudioTrimmer':
        """
        Add silence at the start.

        Args:
            duration_ms: Silence duration in milliseconds

        Returns:
            Self for chaining
        """
        from pydub import AudioSegment
        silence = AudioSegment.silent(
            duration=duration_ms,
            frame_rate=self._audio.frame_rate
        )
        self._audio = silence + self._audio
        return self

    def add_silence_end(self, duration_ms: int) -> 'AudioTrimmer':
        """
        Add silence at the end.

        Args:
            duration_ms: Silence duration in milliseconds

        Returns:
            Self for chaining
        """
        from pydub import AudioSegment
        silence = AudioSegment.silent(
            duration=duration_ms,
            frame_rate=self._audio.frame_rate
        )
        self._audio = self._audio + silence
        return self

    def strip_silence(
        self,
        threshold: float = -50.0,
        chunk_size: int = 10,
        min_silence_len: int = 100
    ) -> 'AudioTrimmer':
        """
        Strip leading and trailing silence.

        Args:
            threshold: Silence threshold in dBFS
            chunk_size: Analysis chunk size in ms
            min_silence_len: Minimum silence length to strip

        Returns:
            Self for chaining
        """
        from pydub.silence import detect_leading_silence

        # Strip leading silence
        start_trim = detect_leading_silence(self._audio, silence_threshold=threshold, chunk_size=chunk_size)

        # Strip trailing silence
        reversed_audio = self._audio.reverse()
        end_trim = detect_leading_silence(reversed_audio, silence_threshold=threshold, chunk_size=chunk_size)

        if start_trim + end_trim < len(self._audio):
            self._audio = self._audio[start_trim:len(self._audio) - end_trim]

        return self

    def overlay(
        self,
        other_file: str,
        position_ms: int = 0,
        volume: float = 0,
        loop: bool = False
    ) -> 'AudioTrimmer':
        """
        Overlay another audio file.

        Args:
            other_file: Path to audio file to overlay
            position_ms: Position to start overlay
            volume: Volume adjustment for overlay (dB)
            loop: Loop overlay to fill duration

        Returns:
            Self for chaining
        """
        from pydub import AudioSegment

        other = AudioSegment.from_file(other_file)

        # Adjust volume
        if volume != 0:
            other = other + volume

        # Loop if needed
        if loop:
            remaining = len(self._audio) - position_ms
            if len(other) < remaining:
                repetitions = (remaining // len(other)) + 1
                other = other * repetitions
            other = other[:remaining]

        self._audio = self._audio.overlay(other, position=position_ms)
        return self

    def get_duration_ms(self) -> int:
        """Get current audio duration in milliseconds."""
        return len(self._audio)

    def get_duration_str(self) -> str:
        """Get current audio duration as formatted string."""
        total_seconds = len(self._audio) / 1000
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:05.2f}"
        else:
            return f"{minutes}:{seconds:05.2f}"

    def save(
        self,
        output: str,
        format: Optional[str] = None,
        bitrate: int = 192
    ) -> str:
        """
        Save audio to file.

        Args:
            output: Output file path
            format: Output format (optional, from extension)
            bitrate: Bitrate for lossy formats (kbps)

        Returns:
            Path to saved file
        """
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = output_path.suffix.lstrip('.').lower()

        # Export parameters
        params = {}
        if format in ('mp3', 'ogg', 'm4a'):
            params['bitrate'] = f"{bitrate}k"

        self._audio.export(str(output_path), format=format, **params)
        return str(output_path)

    @classmethod
    def concatenate(
        cls,
        files: List[str],
        output: str,
        format: Optional[str] = None,
        bitrate: int = 192
    ) -> str:
        """
        Concatenate multiple audio files.

        Args:
            files: List of input file paths
            output: Output file path
            format: Output format
            bitrate: Bitrate for lossy formats

        Returns:
            Path to saved file
        """
        from pydub import AudioSegment

        if not files:
            raise ValueError("No files to concatenate")

        combined = AudioSegment.empty()
        for filepath in files:
            segment = AudioSegment.from_file(filepath)
            combined += segment

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = output_path.suffix.lstrip('.').lower()

        params = {}
        if format in ('mp3', 'ogg', 'm4a'):
            params['bitrate'] = f"{bitrate}k"

        combined.export(str(output_path), format=format, **params)
        return str(output_path)

    @classmethod
    def concatenate_with_crossfade(
        cls,
        files: List[str],
        output: str,
        crossfade_ms: int = 1000,
        format: Optional[str] = None,
        bitrate: int = 192
    ) -> str:
        """
        Concatenate files with crossfade transitions.

        Args:
            files: List of input file paths
            output: Output file path
            crossfade_ms: Crossfade duration in milliseconds
            format: Output format
            bitrate: Bitrate for lossy formats

        Returns:
            Path to saved file
        """
        from pydub import AudioSegment

        if not files:
            raise ValueError("No files to concatenate")

        combined = AudioSegment.from_file(files[0])

        for filepath in files[1:]:
            segment = AudioSegment.from_file(filepath)
            combined = combined.append(segment, crossfade=crossfade_ms)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = output_path.suffix.lstrip('.').lower()

        params = {}
        if format in ('mp3', 'ogg', 'm4a'):
            params['bitrate'] = f"{bitrate}k"

        combined.export(str(output_path), format=format, **params)
        return str(output_path)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Cut, trim, and edit audio segments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input podcast.mp3 --output segment.mp3 --start 05:30 --end 10:00
  %(prog)s --input song.mp3 --output faded.mp3 --fade-in 3000 --fade-out 5000
  %(prog)s --input lecture.mp3 --output fast.mp3 --speed 1.5
  %(prog)s --concat file1.mp3 file2.mp3 file3.mp3 --output merged.mp3
        """
    )

    parser.add_argument('--input', '-i', help='Input audio file')
    parser.add_argument('--output', '-o', required=True, help='Output file path')
    parser.add_argument('--start', '-s', help='Start timestamp (HH:MM:SS or MM:SS)')
    parser.add_argument('--end', '-e', help='End timestamp')
    parser.add_argument('--fade-in', type=int, help='Fade in duration (ms)')
    parser.add_argument('--fade-out', type=int, help='Fade out duration (ms)')
    parser.add_argument('--speed', type=float, default=1.0, help='Speed multiplier')
    parser.add_argument('--gain', type=float, default=0, help='Volume adjustment (dB)')
    parser.add_argument('--normalize', type=float, help='Normalize to dBFS level')
    parser.add_argument('--reverse', action='store_true', help='Reverse audio')
    parser.add_argument('--loop', type=int, help='Loop N times')
    parser.add_argument('--concat', nargs='+', help='Files to concatenate')
    parser.add_argument('--crossfade', type=int, default=0, help='Crossfade duration (ms)')
    parser.add_argument('--bitrate', type=int, default=192, help='Output bitrate (kbps)')
    parser.add_argument('--segments', help='Multiple segments: "00:00-05:00,10:00-15:00"')
    parser.add_argument('--output-dir', help='Output directory for multiple segments')

    args = parser.parse_args()

    # Concatenation mode
    if args.concat:
        if args.crossfade > 0:
            output = AudioTrimmer.concatenate_with_crossfade(
                args.concat, args.output,
                crossfade_ms=args.crossfade,
                bitrate=args.bitrate
            )
        else:
            output = AudioTrimmer.concatenate(
                args.concat, args.output,
                bitrate=args.bitrate
            )
        print(f"Concatenated {len(args.concat)} files -> {output}")
        return

    # Single file mode
    if not args.input:
        parser.error("--input is required (unless using --concat)")

    # Multiple segments mode
    if args.segments:
        if not args.output_dir:
            parser.error("--output-dir is required when using --segments")

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        segment_pairs = args.segments.split(',')

        for i, segment in enumerate(segment_pairs):
            start, end = segment.strip().split('-')
            trimmer = AudioTrimmer(args.input)
            trimmer.trim(start=start.strip(), end=end.strip())

            if args.fade_in:
                trimmer.fade_in(args.fade_in)
            if args.fade_out:
                trimmer.fade_out(args.fade_out)

            output_file = Path(args.output_dir) / f"segment_{i+1:02d}.mp3"
            trimmer.save(str(output_file), bitrate=args.bitrate)
            print(f"Segment {i+1}: {start.strip()} - {end.strip()} -> {output_file}")

        return

    # Standard trimming mode
    trimmer = AudioTrimmer(args.input)

    # Apply operations
    if args.start or args.end:
        trimmer.trim(start=args.start, end=args.end)

    if args.speed != 1.0:
        trimmer.speed(args.speed)

    if args.reverse:
        trimmer.reverse()

    if args.loop:
        trimmer.loop(args.loop)

    if args.gain != 0:
        trimmer.gain(args.gain)

    if args.normalize is not None:
        trimmer.normalize(args.normalize)

    if args.fade_in:
        trimmer.fade_in(args.fade_in)

    if args.fade_out:
        trimmer.fade_out(args.fade_out)

    # Save
    output = trimmer.save(args.output, bitrate=args.bitrate)
    print(f"Saved: {output} (duration: {trimmer.get_duration_str()})")


if __name__ == "__main__":
    main()
