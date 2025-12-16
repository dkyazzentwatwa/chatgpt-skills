#!/usr/bin/env python3
"""
Video Captioner - Add text overlays and subtitles to videos.

Features:
- Static text overlays
- Timed captions with SRT support
- Custom styling (font, color, position)
- Style presets for social media
- Batch captioning
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx.all import fadein, fadeout


class VideoCaptioner:
    """Add text overlays and captions to videos."""

    POSITIONS = {
        'top': lambda w, h: ('center', 50),
        'bottom': lambda w, h: ('center', h - 100),
        'center': lambda w, h: ('center', 'center'),
        'top-left': lambda w, h: (50, 50),
        'top-right': lambda w, h: (w - 50, 50),
        'bottom-left': lambda w, h: (50, h - 100),
        'bottom-right': lambda w, h: (w - 50, h - 100),
    }

    PRESETS = {
        'instagram-story': {
            'font': 'Arial-Bold',
            'font_size': 60,
            'color': 'white',
            'stroke_color': 'black',
            'stroke_width': 3,
            'position': 'top'
        },
        'youtube': {
            'font': 'Arial',
            'font_size': 48,
            'color': 'yellow',
            'bg_color': 'black',
            'position': 'bottom'
        },
        'minimal': {
            'font': 'Arial',
            'font_size': 42,
            'color': 'white',
            'stroke_color': 'black',
            'stroke_width': 1,
            'position': 'bottom'
        },
        'bold': {
            'font': 'Arial-Bold',
            'font_size': 72,
            'color': 'white',
            'bg_color': 'black',
            'position': 'center'
        }
    }

    def __init__(self):
        self.video = None
        self.filepath = None
        self.captions = []
        self.current_preset = 'minimal'

    def load(self, filepath: str) -> 'VideoCaptioner':
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
        self.video = VideoFileClip(filepath)

        return self

    def _parse_position(self, position) -> Tuple:
        """Parse position string or tuple to coordinates."""
        if isinstance(position, tuple):
            return position

        if position in self.POSITIONS:
            return self.POSITIONS[position](self.video.w, self.video.h)

        raise ValueError(f"Unknown position: {position}")

    def add_text(self, text: str, position: str = 'bottom',
                font: str = 'Arial', font_size: int = 48,
                color: str = 'white', bg_color: str = None,
                stroke_color: str = None, stroke_width: int = 0,
                duration: float = None) -> 'VideoCaptioner':
        """
        Add static text overlay to entire video.

        Args:
            text: Text to display
            position: Position ('top', 'bottom', 'center', or tuple)
            font: Font name
            font_size: Font size in pixels
            color: Text color
            bg_color: Background color (optional)
            stroke_color: Outline color (optional)
            stroke_width: Outline width (optional)
            duration: Duration in seconds (None = full video)

        Returns:
            Self for method chaining
        """
        if self.video is None:
            raise ValueError("No video loaded. Call load() first.")

        pos = self._parse_position(position)

        # Create text clip
        txt_clip = TextClip(
            text,
            fontsize=font_size,
            color=color,
            font=font,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            bg_color=bg_color,
            method='caption',
            size=(self.video.w - 100, None)  # Leave margins
        ).set_position(pos)

        # Set duration
        if duration is None:
            txt_clip = txt_clip.set_duration(self.video.duration)
        else:
            txt_clip = txt_clip.set_duration(duration)

        self.captions.append({
            'clip': txt_clip,
            'start': 0,
            'end': duration or self.video.duration
        })

        return self

    def add_caption(self, text: str, start: float, end: float,
                   position: str = 'bottom', font: str = 'Arial',
                   font_size: int = 48, color: str = 'white',
                   bg_color: str = None, stroke_color: str = None,
                   stroke_width: int = 0, fade: bool = False) -> 'VideoCaptioner':
        """
        Add timed caption that appears during specified time range.

        Args:
            text: Caption text
            start: Start time in seconds
            end: End time in seconds
            position: Position ('top', 'bottom', 'center', or tuple)
            font: Font name
            font_size: Font size in pixels
            color: Text color
            bg_color: Background color (optional)
            stroke_color: Outline color (optional)
            stroke_width: Outline width (optional)
            fade: Apply fade in/out effect

        Returns:
            Self for method chaining
        """
        if self.video is None:
            raise ValueError("No video loaded. Call load() first.")

        pos = self._parse_position(position)
        duration = end - start

        # Create text clip
        txt_clip = TextClip(
            text,
            fontsize=font_size,
            color=color,
            font=font,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            bg_color=bg_color,
            method='caption',
            size=(self.video.w - 100, None)
        ).set_position(pos).set_start(start).set_duration(duration)

        # Apply fade effect
        if fade:
            fade_duration = min(0.5, duration / 4)
            txt_clip = fadein(txt_clip, fade_duration)
            txt_clip = fadeout(txt_clip, fade_duration)

        self.captions.append({
            'clip': txt_clip,
            'start': start,
            'end': end
        })

        return self

    def import_srt(self, srt_filepath: str, **style_kwargs) -> 'VideoCaptioner':
        """
        Import subtitles from SRT file.

        Args:
            srt_filepath: Path to SRT subtitle file
            **style_kwargs: Style parameters (font_size, color, etc.)

        Returns:
            Self for method chaining
        """
        if not os.path.exists(srt_filepath):
            raise FileNotFoundError(f"SRT file not found: {srt_filepath}")

        # Parse SRT
        captions = self._parse_srt(srt_filepath)

        # Add captions
        for cap in captions:
            self.add_caption(
                text=cap['text'],
                start=cap['start'],
                end=cap['end'],
                **style_kwargs
            )

        return self

    def _parse_srt(self, filepath: str) -> List[Dict]:
        """Parse SRT subtitle file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # SRT format: number, timestamp, text, blank line
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:.*\n?)+?)(?:\n\n|\Z)'
        matches = re.findall(pattern, content)

        captions = []
        for match in matches:
            start = self._parse_srt_time(match[1])
            end = self._parse_srt_time(match[2])
            text = match[3].strip()

            captions.append({
                'start': start,
                'end': end,
                'text': text
            })

        return captions

    def _parse_srt_time(self, time_str: str) -> float:
        """Convert SRT timestamp to seconds."""
        # Format: HH:MM:SS,mmm
        h, m, s = time_str.split(':')
        s, ms = s.split(',')
        total_seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
        return total_seconds

    def import_captions_json(self, json_filepath: str) -> 'VideoCaptioner':
        """
        Import captions from JSON file.

        JSON format:
        {
          "captions": [
            {"text": "...", "start": 0.0, "end": 3.0, "position": "bottom", ...}
          ]
        }

        Args:
            json_filepath: Path to JSON file

        Returns:
            Self for method chaining
        """
        with open(json_filepath, 'r') as f:
            data = json.load(f)

        for cap in data.get('captions', []):
            self.add_caption(**cap)

        return self

    def style_preset(self, preset: str) -> 'VideoCaptioner':
        """
        Apply style preset.

        Args:
            preset: Preset name ('instagram-story', 'youtube', 'minimal', 'bold')

        Returns:
            Self for method chaining
        """
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(self.PRESETS.keys())}")

        self.current_preset = preset

        return self

    def preview_frame(self, time: float, output: str) -> str:
        """
        Generate preview frame with captions at specific time.

        Args:
            time: Time in seconds
            output: Output image path

        Returns:
            Path to saved image
        """
        if self.video is None:
            raise ValueError("No video loaded. Call load() first.")

        # Composite all captions active at this time
        active_captions = [
            cap['clip'] for cap in self.captions
            if cap['start'] <= time < cap['end']
        ]

        if active_captions:
            composite = CompositeVideoClip([self.video] + active_captions)
        else:
            composite = self.video

        # Save frame
        composite.save_frame(output, t=time)

        return output

    def save(self, output: str, codec: str = 'libx264',
             fps: int = None, bitrate: str = None) -> str:
        """
        Save video with captions.

        Args:
            output: Output video path
            codec: Video codec (default: libx264)
            fps: Frame rate (None = preserve original)
            bitrate: Video bitrate (None = auto)

        Returns:
            Path to saved video
        """
        if self.video is None:
            raise ValueError("No video loaded. Call load() first.")

        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

        # Composite video with all caption clips
        if self.captions:
            caption_clips = [cap['clip'] for cap in self.captions]
            final_video = CompositeVideoClip([self.video] + caption_clips)
        else:
            final_video = self.video

        # Write video
        write_params = {
            'codec': codec,
            'audio_codec': 'aac',
            'logger': None
        }

        if fps:
            write_params['fps'] = fps
        if bitrate:
            write_params['bitrate'] = bitrate

        final_video.write_videofile(output, **write_params)

        return output

    def clear_captions(self) -> 'VideoCaptioner':
        """Clear all captions."""
        self.captions = []
        return self

    def close(self):
        """Release video resources."""
        if self.video:
            self.video.close()
            self.video = None


def main():
    parser = argparse.ArgumentParser(
        description='Add text overlays and captions to videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple text overlay
  python video_captioner.py input.mp4 --text "Subscribe!" --position bottom --output captioned.mp4

  # Add SRT subtitles
  python video_captioner.py input.mp4 --srt subtitles.srt --output subtitled.mp4

  # Custom styling
  python video_captioner.py input.mp4 --text "Sale!" --font-size 72 --color red --bg-color black --output promo.mp4

  # Use preset
  python video_captioner.py input.mp4 --text "Check this out!" --preset instagram-story --output story.mp4
        """
    )

    parser.add_argument('input', help='Input video file')
    parser.add_argument('--output', '-o', required=True, help='Output video file')
    parser.add_argument('--text', '-t', help='Text to overlay')
    parser.add_argument('--srt', help='SRT subtitle file')
    parser.add_argument('--captions', help='JSON captions file')
    parser.add_argument('--position', '-p', default='bottom',
                       help='Text position (top, bottom, center, etc.)')
    parser.add_argument('--font', '-f', default='Arial', help='Font name')
    parser.add_argument('--font-size', type=int, default=48, help='Font size')
    parser.add_argument('--color', '-c', default='white', help='Text color')
    parser.add_argument('--bg-color', help='Background color (optional)')
    parser.add_argument('--stroke-color', help='Outline color (optional)')
    parser.add_argument('--stroke-width', type=int, default=0, help='Outline width')
    parser.add_argument('--preset', choices=['instagram-story', 'youtube', 'minimal', 'bold'],
                       help='Use style preset')
    parser.add_argument('--preview', type=float, help='Generate preview frame at time (seconds)')

    args = parser.parse_args()

    captioner = VideoCaptioner()

    try:
        # Load video
        print(f"Loading video: {args.input}")
        captioner.load(args.input)

        # Apply preset if specified
        if args.preset:
            captioner.style_preset(args.preset)
            preset_style = VideoCaptioner.PRESETS[args.preset]

            # Use preset values as defaults
            if not args.text and not args.srt and not args.captions:
                print(f"Error: --text, --srt, or --captions required")
                return

            style_kwargs = {
                'font': args.font or preset_style.get('font', 'Arial'),
                'font_size': args.font_size or preset_style.get('font_size', 48),
                'color': args.color or preset_style.get('color', 'white'),
                'bg_color': args.bg_color or preset_style.get('bg_color'),
                'stroke_color': args.stroke_color or preset_style.get('stroke_color'),
                'stroke_width': args.stroke_width or preset_style.get('stroke_width', 0),
                'position': args.position or preset_style.get('position', 'bottom')
            }
        else:
            style_kwargs = {
                'font': args.font,
                'font_size': args.font_size,
                'color': args.color,
                'bg_color': args.bg_color,
                'stroke_color': args.stroke_color,
                'stroke_width': args.stroke_width,
                'position': args.position
            }

        # Add captions
        if args.text:
            print(f"Adding text overlay: {args.text}")
            captioner.add_text(args.text, **style_kwargs)

        if args.srt:
            print(f"Importing SRT subtitles: {args.srt}")
            captioner.import_srt(args.srt, **style_kwargs)

        if args.captions:
            print(f"Importing JSON captions: {args.captions}")
            captioner.import_captions_json(args.captions)

        # Preview frame
        if args.preview is not None:
            preview_path = args.output.replace('.mp4', '_preview.png')
            print(f"Generating preview frame at {args.preview}s: {preview_path}")
            captioner.preview_frame(args.preview, preview_path)
            print(f"✓ Preview saved: {preview_path}")
            return

        # Save video
        print(f"Rendering video with captions...")
        captioner.save(args.output)
        print(f"✓ Saved: {args.output}")

    finally:
        captioner.close()


if __name__ == '__main__':
    main()
