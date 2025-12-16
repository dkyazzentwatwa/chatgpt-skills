#!/usr/bin/env python3
"""
Video Metadata Inspector - Extract comprehensive metadata from video files.

Features:
- Duration, resolution, frame rate, file size
- Video/audio codec details
- Format and container information
- Batch processing
- Export to JSON/CSV
"""

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from moviepy.editor import VideoFileClip
import ffmpeg


class VideoMetadataInspector:
    """Extract and analyze video file metadata."""

    def __init__(self):
        self.filepath = None
        self.clip = None
        self.probe_data = None

    def load(self, filepath: str) -> 'VideoMetadataInspector':
        """
        Load video file.

        Args:
            filepath: Path to video file

        Returns:
            Self for method chaining
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Video file not found: {filepath}")

        self.filepath = filepath

        # Load with moviepy for basic info
        try:
            self.clip = VideoFileClip(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load video: {e}")

        # Probe with ffprobe for detailed codec info
        try:
            self.probe_data = ffmpeg.probe(filepath)
        except Exception as e:
            print(f"Warning: FFprobe failed: {e}")
            self.probe_data = None

        return self

    def get_basic_info(self) -> Dict[str, Any]:
        """
        Get basic video information.

        Returns:
            Dictionary with duration, resolution, FPS, file size
        """
        if self.clip is None:
            raise ValueError("No video loaded. Call load() first.")

        file_size = os.path.getsize(self.filepath)

        info = {
            'filename': os.path.basename(self.filepath),
            'filepath': self.filepath,
            'duration_seconds': round(self.clip.duration, 2),
            'duration_formatted': str(timedelta(seconds=int(self.clip.duration))),
            'width': self.clip.w,
            'height': self.clip.h,
            'fps': round(self.clip.fps, 2),
            'aspect_ratio': f"{self.clip.w}:{self.clip.h}",
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2)
        }

        # Calculate simplified aspect ratio
        from math import gcd
        divisor = gcd(self.clip.w, self.clip.h)
        info['aspect_ratio_simple'] = f"{self.clip.w//divisor}:{self.clip.h//divisor}"

        # Resolution category
        if self.clip.w >= 7680:
            info['resolution_category'] = '8K'
        elif self.clip.w >= 3840:
            info['resolution_category'] = '4K'
        elif self.clip.w >= 2560:
            info['resolution_category'] = '2K'
        elif self.clip.w >= 1920:
            info['resolution_category'] = '1080p'
        elif self.clip.w >= 1280:
            info['resolution_category'] = '720p'
        elif self.clip.w >= 854:
            info['resolution_category'] = '480p'
        else:
            info['resolution_category'] = 'SD'

        return info

    def get_video_info(self) -> Dict[str, Any]:
        """
        Get video stream information.

        Returns:
            Dictionary with video codec, bitrate, pixel format
        """
        if self.probe_data is None:
            return {}

        # Find video stream
        video_stream = next((s for s in self.probe_data['streams']
                           if s['codec_type'] == 'video'), None)

        if video_stream is None:
            return {}

        info = {
            'video_codec': video_stream.get('codec_name', 'unknown'),
            'video_codec_long': video_stream.get('codec_long_name', 'unknown'),
            'profile': video_stream.get('profile', 'unknown'),
            'pixel_format': video_stream.get('pix_fmt', 'unknown'),
            'color_space': video_stream.get('color_space', 'unknown'),
            'color_range': video_stream.get('color_range', 'unknown'),
        }

        # Bitrate
        if 'bit_rate' in video_stream:
            bitrate_bps = int(video_stream['bit_rate'])
            info['bitrate_kbps'] = round(bitrate_bps / 1000, 2)
            info['bitrate_mbps'] = round(bitrate_bps / 1_000_000, 2)
        else:
            info['bitrate_kbps'] = None
            info['bitrate_mbps'] = None

        # Level
        if 'level' in video_stream:
            info['level'] = video_stream['level']

        return info

    def get_audio_info(self) -> Dict[str, Any]:
        """
        Get audio stream information.

        Returns:
            Dictionary with audio codec, sample rate, channels
        """
        if self.probe_data is None:
            return {}

        # Find audio stream
        audio_stream = next((s for s in self.probe_data['streams']
                           if s['codec_type'] == 'audio'), None)

        if audio_stream is None:
            return {'has_audio': False}

        info = {
            'has_audio': True,
            'audio_codec': audio_stream.get('codec_name', 'unknown'),
            'audio_codec_long': audio_stream.get('codec_long_name', 'unknown'),
            'sample_rate': int(audio_stream.get('sample_rate', 0)),
            'channels': audio_stream.get('channels', 0),
            'channel_layout': audio_stream.get('channel_layout', 'unknown'),
        }

        # Sample rate in kHz
        if info['sample_rate']:
            info['sample_rate_khz'] = round(info['sample_rate'] / 1000, 1)

        # Bitrate
        if 'bit_rate' in audio_stream:
            bitrate_bps = int(audio_stream['bit_rate'])
            info['audio_bitrate_kbps'] = round(bitrate_bps / 1000, 2)
        else:
            info['audio_bitrate_kbps'] = None

        return info

    def get_format_info(self) -> Dict[str, Any]:
        """
        Get container format information.

        Returns:
            Dictionary with container type, creation date, metadata
        """
        if self.probe_data is None:
            return {}

        format_data = self.probe_data.get('format', {})

        info = {
            'container_format': format_data.get('format_name', 'unknown'),
            'container_long_name': format_data.get('format_long_name', 'unknown'),
            'num_streams': format_data.get('nb_streams', 0),
        }

        # Metadata tags
        tags = format_data.get('tags', {})
        if tags:
            info['metadata'] = {
                'title': tags.get('title'),
                'artist': tags.get('artist'),
                'album': tags.get('album'),
                'date': tags.get('date'),
                'comment': tags.get('comment'),
                'encoder': tags.get('encoder'),
            }
            # Remove None values
            info['metadata'] = {k: v for k, v in info['metadata'].items() if v}

        # Overall bitrate
        if 'bit_rate' in format_data:
            bitrate_bps = int(format_data['bit_rate'])
            info['overall_bitrate_kbps'] = round(bitrate_bps / 1000, 2)
            info['overall_bitrate_mbps'] = round(bitrate_bps / 1_000_000, 2)

        return info

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get all metadata combined.

        Returns:
            Dictionary with all metadata categories
        """
        metadata = {}
        metadata.update(self.get_basic_info())
        metadata.update(self.get_video_info())
        metadata.update(self.get_audio_info())
        metadata.update(self.get_format_info())

        return metadata

    def export_report(self, output: str, format: str = 'json') -> str:
        """
        Export metadata report.

        Args:
            output: Output filepath
            format: Export format ('json', 'text')

        Returns:
            Path to exported file
        """
        metadata = self.get_metadata()

        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

        if format == 'json':
            with open(output, 'w') as f:
                json.dump(metadata, f, indent=2)

        elif format == 'text':
            with open(output, 'w') as f:
                f.write(f"Video Metadata Report: {metadata['filename']}\n")
                f.write("=" * 60 + "\n\n")

                f.write("BASIC INFORMATION\n")
                f.write("-" * 60 + "\n")
                f.write(f"Duration: {metadata['duration_formatted']} ({metadata['duration_seconds']}s)\n")
                f.write(f"Resolution: {metadata['width']}x{metadata['height']} ({metadata['resolution_category']})\n")
                f.write(f"Aspect Ratio: {metadata['aspect_ratio_simple']}\n")
                f.write(f"Frame Rate: {metadata['fps']} fps\n")
                f.write(f"File Size: {metadata['file_size_mb']} MB\n\n")

                if metadata.get('video_codec'):
                    f.write("VIDEO STREAM\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"Codec: {metadata['video_codec']} ({metadata.get('video_codec_long', 'N/A')})\n")
                    f.write(f"Profile: {metadata.get('profile', 'N/A')}\n")
                    f.write(f"Pixel Format: {metadata.get('pixel_format', 'N/A')}\n")
                    if metadata.get('bitrate_mbps'):
                        f.write(f"Bitrate: {metadata['bitrate_mbps']} Mbps\n")
                    f.write("\n")

                if metadata.get('has_audio'):
                    f.write("AUDIO STREAM\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"Codec: {metadata.get('audio_codec', 'N/A')}\n")
                    f.write(f"Sample Rate: {metadata.get('sample_rate_khz', 'N/A')} kHz\n")
                    f.write(f"Channels: {metadata.get('channels', 'N/A')} ({metadata.get('channel_layout', 'N/A')})\n")
                    if metadata.get('audio_bitrate_kbps'):
                        f.write(f"Bitrate: {metadata['audio_bitrate_kbps']} kbps\n")
                    f.write("\n")

                f.write("CONTAINER FORMAT\n")
                f.write("-" * 60 + "\n")
                f.write(f"Format: {metadata.get('container_format', 'N/A')}\n")
                f.write(f"Streams: {metadata.get('num_streams', 'N/A')}\n")
                if metadata.get('overall_bitrate_mbps'):
                    f.write(f"Overall Bitrate: {metadata['overall_bitrate_mbps']} Mbps\n")

        return output

    def batch_inspect(self, input_files: List[str], output: str = None,
                     format: str = 'csv') -> pd.DataFrame:
        """
        Inspect multiple video files.

        Args:
            input_files: List of video file paths
            output: Output filepath (optional)
            format: Export format ('csv', 'json')

        Returns:
            DataFrame with metadata for all files
        """
        results = []

        for i, filepath in enumerate(input_files, 1):
            print(f"Processing {i}/{len(input_files)}: {os.path.basename(filepath)}")

            try:
                self.load(filepath)
                metadata = self.get_metadata()
                results.append(metadata)
            except Exception as e:
                print(f"  Error: {e}")
                results.append({'filename': os.path.basename(filepath), 'error': str(e)})

        df = pd.DataFrame(results)

        if output:
            os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
            if format == 'csv':
                df.to_csv(output, index=False)
            elif format == 'json':
                df.to_json(output, orient='records', indent=2)

        return df

    def compare_videos(self, video_files: List[str]) -> pd.DataFrame:
        """
        Compare metadata of multiple videos side-by-side.

        Args:
            video_files: List of video file paths

        Returns:
            DataFrame with comparison
        """
        df = self.batch_inspect(video_files)

        # Select key comparison columns
        compare_cols = [
            'filename', 'duration_seconds', 'width', 'height', 'fps',
            'file_size_mb', 'video_codec', 'bitrate_mbps',
            'audio_codec', 'sample_rate_khz', 'channels'
        ]

        # Keep only columns that exist
        compare_cols = [c for c in compare_cols if c in df.columns]

        return df[compare_cols]

    def close(self):
        """Release video resources."""
        if self.clip:
            self.clip.close()
            self.clip = None


def main():
    parser = argparse.ArgumentParser(
        description='Extract and analyze video file metadata',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic metadata
  python video_metadata_inspector.py video.mp4

  # Export to JSON
  python video_metadata_inspector.py video.mp4 --output metadata.json --format json

  # Batch inspect directory
  python video_metadata_inspector.py *.mp4 --output metadata.csv --format csv

  # Compare videos
  python video_metadata_inspector.py video1.mp4 video2.mp4 --compare
        """
    )

    parser.add_argument('input', nargs='+', help='Input video file(s)')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--format', '-f', choices=['json', 'csv', 'text'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed metadata')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare multiple videos')

    args = parser.parse_args()

    inspector = VideoMetadataInspector()

    try:
        # Single file mode
        if len(args.input) == 1 and not args.compare:
            filepath = args.input[0]
            inspector.load(filepath)

            if args.verbose:
                # Full metadata
                metadata = inspector.get_metadata()
                print(json.dumps(metadata, indent=2))
            else:
                # Basic summary
                basic = inspector.get_basic_info()
                video = inspector.get_video_info()
                audio = inspector.get_audio_info()

                print(f"\n{basic['filename']}")
                print("=" * 60)
                print(f"Duration:    {basic['duration_formatted']} ({basic['duration_seconds']}s)")
                print(f"Resolution:  {basic['width']}x{basic['height']} ({basic['resolution_category']})")
                print(f"Frame Rate:  {basic['fps']} fps")
                print(f"File Size:   {basic['file_size_mb']} MB")

                if video.get('video_codec'):
                    print(f"\nVideo Codec: {video['video_codec']}")
                    if video.get('bitrate_mbps'):
                        print(f"Video Bitrate: {video['bitrate_mbps']} Mbps")

                if audio.get('has_audio'):
                    print(f"\nAudio Codec: {audio['audio_codec']}")
                    print(f"Sample Rate: {audio.get('sample_rate_khz', 'N/A')} kHz")
                    print(f"Channels:    {audio['channels']} ({audio['channel_layout']})")

            # Export if requested
            if args.output:
                inspector.export_report(args.output, format=args.format)
                print(f"\n✓ Exported to: {args.output}")

        # Compare mode
        elif args.compare:
            print(f"\nComparing {len(args.input)} videos...\n")
            df = inspector.compare_videos(args.input)
            print(df.to_string(index=False))

            if args.output:
                if args.format == 'csv':
                    df.to_csv(args.output, index=False)
                elif args.format == 'json':
                    df.to_json(args.output, orient='records', indent=2)
                print(f"\n✓ Exported to: {args.output}")

        # Batch mode
        else:
            print(f"\nInspecting {len(args.input)} videos...\n")
            df = inspector.batch_inspect(args.input, output=args.output, format=args.format)

            # Print summary
            print(f"\nSummary:")
            print(f"  Total files: {len(df)}")
            if 'error' not in df.columns:
                print(f"  Total duration: {df['duration_seconds'].sum():.0f}s")
                print(f"  Total size: {df['file_size_mb'].sum():.2f} MB")
                print(f"  Resolutions: {df['resolution_category'].value_counts().to_dict()}")

            if args.output:
                print(f"\n✓ Exported to: {args.output}")

    finally:
        inspector.close()


if __name__ == '__main__':
    main()
