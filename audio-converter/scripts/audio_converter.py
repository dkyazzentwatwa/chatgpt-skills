#!/usr/bin/env python3
"""
Audio Converter - Convert audio files between formats.

Features:
- Format conversion (MP3, WAV, FLAC, OGG, M4A, AIFF)
- Bitrate and sample rate control
- Batch processing
- Volume normalization
"""

import argparse
import os
from pathlib import Path
from typing import Optional, List


class AudioConverter:
    """Convert audio files between formats."""

    SUPPORTED_FORMATS = {
        'mp3': {'codec': 'mp3', 'lossy': True},
        'wav': {'codec': 'pcm_s16le', 'lossy': False},
        'flac': {'codec': 'flac', 'lossy': False},
        'ogg': {'codec': 'libvorbis', 'lossy': True},
        'm4a': {'codec': 'aac', 'lossy': True},
        'aiff': {'codec': 'pcm_s16be', 'lossy': False}
    }

    def __init__(self, filepath: str):
        """
        Initialize converter with audio file.

        Args:
            filepath: Path to input audio file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        self._bitrate: Optional[int] = None
        self._sample_rate: Optional[int] = None
        self._channels: Optional[int] = None
        self._normalize: bool = False

        # Load audio
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

    def bitrate(self, kbps: int) -> 'AudioConverter':
        """
        Set output bitrate for lossy formats.

        Args:
            kbps: Bitrate in kilobits per second (e.g., 128, 192, 320)

        Returns:
            Self for chaining
        """
        self._bitrate = kbps
        return self

    def sample_rate(self, hz: int) -> 'AudioConverter':
        """
        Set output sample rate.

        Args:
            hz: Sample rate in Hz (e.g., 44100, 48000)

        Returns:
            Self for chaining
        """
        self._sample_rate = hz
        return self

    def channels(self, num: int) -> 'AudioConverter':
        """
        Set number of output channels.

        Args:
            num: Number of channels (1=mono, 2=stereo)

        Returns:
            Self for chaining
        """
        self._channels = num
        return self

    def normalize(self, enable: bool = True) -> 'AudioConverter':
        """
        Enable/disable volume normalization.

        Args:
            enable: Whether to normalize

        Returns:
            Self for chaining
        """
        self._normalize = enable
        return self

    def convert(self, output: str, format: Optional[str] = None) -> str:
        """
        Convert audio to specified format.

        Args:
            output: Output file path
            format: Output format (optional, inferred from extension)

        Returns:
            Path to converted file
        """
        from pydub import AudioSegment

        output_path = Path(output)

        # Determine format
        if format is None:
            format = output_path.suffix.lstrip('.').lower()

        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported: {list(self.SUPPORTED_FORMATS.keys())}")

        # Prepare audio
        audio = self._audio

        # Apply sample rate
        if self._sample_rate:
            audio = audio.set_frame_rate(self._sample_rate)

        # Apply channels
        if self._channels:
            if self._channels == 1:
                audio = audio.set_channels(1)
            elif self._channels == 2:
                audio = audio.set_channels(2)

        # Normalize
        if self._normalize:
            # Normalize to -3 dB
            change_in_dBFS = -3 - audio.dBFS
            audio = audio.apply_gain(change_in_dBFS)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build export parameters
        export_params = {}

        if self.SUPPORTED_FORMATS[format]['lossy']:
            # Lossy format - set bitrate
            bitrate = self._bitrate or 192
            export_params['bitrate'] = f"{bitrate}k"

        # Export
        audio.export(
            str(output_path),
            format=format,
            **export_params
        )

        return str(output_path)

    def get_info(self) -> dict:
        """
        Get information about the input audio.

        Returns:
            Dict with audio properties
        """
        return {
            'file': str(self.filepath),
            'format': self.filepath.suffix.lstrip('.'),
            'duration_ms': len(self._audio),
            'duration_sec': len(self._audio) / 1000,
            'channels': self._audio.channels,
            'sample_rate': self._audio.frame_rate,
            'sample_width': self._audio.sample_width,
            'frame_count': self._audio.frame_count(),
            'dBFS': self._audio.dBFS
        }

    @classmethod
    def batch_convert(
        cls,
        input_dir: str,
        output_dir: str,
        format: str,
        bitrate: Optional[int] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        normalize: bool = False
    ) -> List[str]:
        """
        Batch convert all audio files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            format: Output format
            bitrate: Bitrate in kbps (for lossy formats)
            sample_rate: Sample rate in Hz
            channels: Number of channels
            normalize: Whether to normalize volume

        Returns:
            List of converted file paths
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        output_path.mkdir(parents=True, exist_ok=True)

        # Find audio files
        audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aiff', '.aac', '.wma'}
        files = [f for f in input_path.iterdir() if f.suffix.lower() in audio_extensions]

        converted = []
        for filepath in files:
            try:
                converter = cls(str(filepath))

                if bitrate:
                    converter.bitrate(bitrate)
                if sample_rate:
                    converter.sample_rate(sample_rate)
                if channels:
                    converter.channels(channels)
                if normalize:
                    converter.normalize(True)

                output_file = output_path / f"{filepath.stem}.{format}"
                converter.convert(str(output_file), format=format)
                converted.append(str(output_file))
                print(f"Converted: {filepath.name} -> {output_file.name}")

            except Exception as e:
                print(f"Failed to convert {filepath.name}: {e}")

        return converted


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Convert audio files between formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input song.wav --output song.mp3
  %(prog)s --input song.flac --output song.mp3 --bitrate 320
  %(prog)s --input-dir ./wavs --output-dir ./mp3s --format mp3 --bitrate 192
        """
    )

    parser.add_argument('--input', '-i', help='Input audio file')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--input-dir', help='Input directory for batch processing')
    parser.add_argument('--output-dir', help='Output directory for batch processing')
    parser.add_argument('--format', '-f', help='Output format')
    parser.add_argument('--bitrate', '-b', type=int, help='Bitrate in kbps (for lossy formats)')
    parser.add_argument('--sample-rate', '-s', type=int, help='Sample rate in Hz')
    parser.add_argument('--channels', '-c', type=int, choices=[1, 2], help='Number of channels')
    parser.add_argument('--normalize', '-n', action='store_true', help='Normalize volume')
    parser.add_argument('--info', action='store_true', help='Show info about input file')

    args = parser.parse_args()

    # Validate arguments
    if args.input_dir:
        # Batch mode
        if not args.output_dir:
            parser.error("--output-dir is required for batch processing")
        if not args.format:
            parser.error("--format is required for batch processing")

        converted = AudioConverter.batch_convert(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            format=args.format,
            bitrate=args.bitrate,
            sample_rate=args.sample_rate,
            channels=args.channels,
            normalize=args.normalize
        )
        print(f"\nConverted {len(converted)} files")

    elif args.input:
        # Single file mode
        converter = AudioConverter(args.input)

        if args.info:
            info = converter.get_info()
            print(f"File: {info['file']}")
            print(f"Format: {info['format']}")
            print(f"Duration: {info['duration_sec']:.2f} seconds")
            print(f"Channels: {info['channels']}")
            print(f"Sample Rate: {info['sample_rate']} Hz")
            print(f"Level: {info['dBFS']:.1f} dBFS")
            return

        if not args.output:
            parser.error("--output is required for conversion")

        if args.bitrate:
            converter.bitrate(args.bitrate)
        if args.sample_rate:
            converter.sample_rate(args.sample_rate)
        if args.channels:
            converter.channels(args.channels)
        if args.normalize:
            converter.normalize(True)

        output = converter.convert(args.output, format=args.format)
        print(f"Converted: {args.input} -> {output}")

    else:
        parser.error("Either --input or --input-dir is required")


if __name__ == "__main__":
    main()
