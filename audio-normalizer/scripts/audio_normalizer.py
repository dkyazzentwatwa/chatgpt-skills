#!/usr/bin/env python3
"""
Audio Normalizer - Normalize audio volume using peak or RMS methods.

Features:
- Peak normalization to target dBFS
- RMS normalization for average loudness
- Loudness analysis
- Batch processing
- Format preservation
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from pydub import AudioSegment
from pydub.utils import db_to_float, ratio_to_db


class AudioNormalizer:
    """Normalize audio volume levels."""

    def __init__(self):
        self.audio = None
        self.filepath = None
        self.sample_rate = None

    def load(self, filepath: str) -> 'AudioNormalizer':
        """
        Load audio file.

        Args:
            filepath: Path to audio file

        Returns:
            Self for method chaining
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        self.filepath = filepath
        # Auto-detect format from extension
        ext = Path(filepath).suffix[1:].lower()
        self.audio = AudioSegment.from_file(filepath, format=ext)
        self.sample_rate = self.audio.frame_rate

        return self

    def analyze_levels(self) -> Dict[str, float]:
        """
        Analyze current audio levels.

        Returns:
            Dictionary with peak and RMS levels in dBFS
        """
        if self.audio is None:
            raise ValueError("No audio loaded. Call load() first.")

        # Peak level (max dBFS)
        peak_dbfs = self.audio.max_dBFS

        # RMS level (average loudness)
        rms_dbfs = self.audio.dBFS

        # Crest factor (peak to RMS ratio)
        crest_factor = peak_dbfs - rms_dbfs

        return {
            'peak_dbfs': round(peak_dbfs, 2),
            'rms_dbfs': round(rms_dbfs, 2),
            'crest_factor': round(crest_factor, 2),
            'duration_seconds': len(self.audio) / 1000.0,
            'sample_rate': self.sample_rate,
            'channels': self.audio.channels
        }

    def normalize_peak(self, target_dbfs: float = -1.0, headroom: float = 0.1) -> 'AudioNormalizer':
        """
        Normalize audio to target peak level.

        Args:
            target_dbfs: Target peak level in dBFS (default: -1.0)
            headroom: Safety headroom in dB to prevent clipping (default: 0.1)

        Returns:
            Self for method chaining
        """
        if self.audio is None:
            raise ValueError("No audio loaded. Call load() first.")

        # Calculate current peak
        current_peak = self.audio.max_dBFS

        # Calculate gain needed (with headroom)
        gain_needed = (target_dbfs - headroom) - current_peak

        # Apply gain
        self.audio = self.audio.apply_gain(gain_needed)

        return self

    def normalize_rms(self, target_dbfs: float = -20.0) -> 'AudioNormalizer':
        """
        Normalize audio to target RMS (average loudness) level.

        Args:
            target_dbfs: Target RMS level in dBFS (default: -20.0)

        Returns:
            Self for method chaining
        """
        if self.audio is None:
            raise ValueError("No audio loaded. Call load() first.")

        # Calculate current RMS
        current_rms = self.audio.dBFS

        # Calculate gain needed
        gain_needed = target_dbfs - current_rms

        # Apply gain
        self.audio = self.audio.apply_gain(gain_needed)

        # Check for clipping after normalization
        if self.audio.max_dBFS > -0.1:
            print(f"Warning: Audio may clip (peak at {self.audio.max_dBFS:.2f} dBFS). "
                  f"Consider using peak normalization or lower target.")

        return self

    def match_loudness(self, reference_audio: 'AudioSegment') -> 'AudioNormalizer':
        """
        Match loudness to reference audio.

        Args:
            reference_audio: AudioSegment to match loudness to

        Returns:
            Self for method chaining
        """
        if self.audio is None:
            raise ValueError("No audio loaded. Call load() first.")

        # Get reference RMS
        reference_rms = reference_audio.dBFS

        # Normalize to match
        return self.normalize_rms(target_dbfs=reference_rms)

    def save(self, output: str, format: str = None, bitrate: str = '192k') -> str:
        """
        Save normalized audio.

        Args:
            output: Output filepath
            format: Output format (auto-detected from extension if None)
            bitrate: Output bitrate for compressed formats

        Returns:
            Path to saved file
        """
        if self.audio is None:
            raise ValueError("No audio loaded. Call load() first.")

        # Auto-detect format from extension
        if format is None:
            format = Path(output).suffix[1:].lower()

        # Create output directory if needed
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

        # Export with parameters
        if format in ['mp3', 'ogg', 'm4a']:
            self.audio.export(output, format=format, bitrate=bitrate)
        else:
            self.audio.export(output, format=format)

        return output

    def batch_normalize(self, input_files: List[str], output_dir: str,
                       method: str = 'rms', target_dbfs: float = -20.0,
                       format: str = None) -> List[str]:
        """
        Normalize multiple audio files to same level.

        Args:
            input_files: List of input file paths
            output_dir: Output directory for normalized files
            method: Normalization method ('peak' or 'rms')
            target_dbfs: Target level in dBFS
            format: Output format (preserves input format if None)

        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []

        for i, input_file in enumerate(input_files, 1):
            print(f"Processing {i}/{len(input_files)}: {os.path.basename(input_file)}")

            # Load and analyze
            self.load(input_file)
            levels = self.analyze_levels()
            print(f"  Current: Peak={levels['peak_dbfs']:.2f} dB, RMS={levels['rms_dbfs']:.2f} dB")

            # Normalize
            if method == 'peak':
                self.normalize_peak(target_dbfs=target_dbfs)
            elif method == 'rms':
                self.normalize_rms(target_dbfs=target_dbfs)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Save
            output_format = format or Path(input_file).suffix[1:]
            output_filename = Path(input_file).stem + f'_normalized.{output_format}'
            output_path = os.path.join(output_dir, output_filename)
            self.save(output_path, format=output_format)

            # Verify
            self.load(output_path)
            new_levels = self.analyze_levels()
            print(f"  Normalized: Peak={new_levels['peak_dbfs']:.2f} dB, RMS={new_levels['rms_dbfs']:.2f} dB")

            output_files.append(output_path)

        return output_files


def main():
    parser = argparse.ArgumentParser(
        description='Normalize audio volume levels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Peak normalization to -1 dBFS
  python audio_normalizer.py input.mp3 --output normalized.mp3 --method peak --target -1.0

  # RMS normalization for podcasts
  python audio_normalizer.py input.mp3 --output normalized.mp3 --method rms --target -19.0

  # Analyze current levels
  python audio_normalizer.py input.mp3 --analyze-only

  # Batch normalize all MP3s
  python audio_normalizer.py *.mp3 --output-dir normalized/ --method rms --target -20.0
        """
    )

    parser.add_argument('input', nargs='+', help='Input audio file(s)')
    parser.add_argument('--output', '-o', help='Output file (single file mode)')
    parser.add_argument('--output-dir', '-d', help='Output directory (batch mode)')
    parser.add_argument('--method', '-m', choices=['peak', 'rms'], default='rms',
                       help='Normalization method (default: rms)')
    parser.add_argument('--target', '-t', type=float, default=-20.0,
                       help='Target level in dBFS (default: -20.0 for RMS, -1.0 for peak)')
    parser.add_argument('--format', '-f', help='Output format (auto-detected from extension)')
    parser.add_argument('--bitrate', '-b', default='192k', help='Bitrate for compressed formats (default: 192k)')
    parser.add_argument('--analyze-only', '-a', action='store_true',
                       help='Analyze levels without normalizing')

    args = parser.parse_args()

    normalizer = AudioNormalizer()

    # Single file mode
    if len(args.input) == 1 and not args.output_dir:
        input_file = args.input[0]

        # Load and analyze
        normalizer.load(input_file)
        levels = normalizer.analyze_levels()

        print(f"\nAudio Analysis: {os.path.basename(input_file)}")
        print(f"  Peak Level: {levels['peak_dbfs']:.2f} dBFS")
        print(f"  RMS Level: {levels['rms_dbfs']:.2f} dBFS")
        print(f"  Crest Factor: {levels['crest_factor']:.2f} dB")
        print(f"  Duration: {levels['duration_seconds']:.2f} seconds")
        print(f"  Sample Rate: {levels['sample_rate']} Hz")
        print(f"  Channels: {levels['channels']}")

        if args.analyze_only:
            return

        if not args.output:
            print("\nError: --output required for single file normalization")
            return

        # Normalize
        if args.method == 'peak':
            target = args.target if args.target != -20.0 else -1.0
            normalizer.normalize_peak(target_dbfs=target)
        else:
            normalizer.normalize_rms(target_dbfs=args.target)

        # Save
        normalizer.save(args.output, format=args.format, bitrate=args.bitrate)

        # Verify
        normalizer.load(args.output)
        new_levels = normalizer.analyze_levels()
        print(f"\nNormalized Audio:")
        print(f"  Peak Level: {new_levels['peak_dbfs']:.2f} dBFS")
        print(f"  RMS Level: {new_levels['rms_dbfs']:.2f} dBFS")
        print(f"\nSaved: {args.output}")

    # Batch mode
    else:
        if not args.output_dir:
            print("Error: --output-dir required for batch processing")
            return

        target = args.target
        if args.method == 'peak' and target == -20.0:
            target = -1.0

        print(f"\nBatch normalizing {len(args.input)} files...")
        print(f"Method: {args.method.upper()} to {target:.1f} dBFS")
        print(f"Output directory: {args.output_dir}\n")

        output_files = normalizer.batch_normalize(
            input_files=args.input,
            output_dir=args.output_dir,
            method=args.method,
            target_dbfs=target,
            format=args.format
        )

        print(f"\n✓ Normalized {len(output_files)} files to {args.output_dir}")


if __name__ == '__main__':
    main()
