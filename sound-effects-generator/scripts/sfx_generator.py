#!/usr/bin/env python3
"""
Sound Effects Generator - Generate programmatic audio.

Features:
- Tone generation (sine, square, sawtooth, triangle)
- Noise generation (white, pink, brown)
- DTMF tones
- Beep sequences
- Fade effects
"""

import argparse
from pathlib import Path
from typing import List, Union, Optional
import numpy as np


class SoundEffectsGenerator:
    """Generate programmatic audio effects."""

    # DTMF frequencies (row, column)
    DTMF_FREQ = {
        '1': (697, 1209), '2': (697, 1336), '3': (697, 1477), 'A': (697, 1633),
        '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), 'B': (770, 1633),
        '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), 'C': (852, 1633),
        '*': (941, 1209), '0': (941, 1336), '#': (941, 1477), 'D': (941, 1633),
    }

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize generator.

        Args:
            sample_rate: Sample rate in Hz (default 44100)
        """
        self.sample_rate = sample_rate
        self._audio = np.array([], dtype=np.float32)

    def _generate_waveform(
        self,
        frequency: float,
        duration_ms: int,
        waveform: str = "sine"
    ) -> np.ndarray:
        """Generate a waveform at given frequency."""
        num_samples = int(self.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, dtype=np.float32)

        if waveform == "sine":
            signal = np.sin(2 * np.pi * frequency * t)
        elif waveform == "square":
            signal = np.sign(np.sin(2 * np.pi * frequency * t))
        elif waveform == "sawtooth":
            signal = 2 * (t * frequency - np.floor(0.5 + t * frequency))
        elif waveform == "triangle":
            signal = 2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1
        else:
            raise ValueError(f"Unknown waveform: {waveform}")

        return signal.astype(np.float32)

    def tone(
        self,
        frequency: float,
        duration: int = 1000,
        waveform: str = "sine",
        volume: float = 0.8
    ) -> 'SoundEffectsGenerator':
        """
        Generate a tone.

        Args:
            frequency: Frequency in Hz
            duration: Duration in milliseconds
            waveform: Waveform type (sine, square, sawtooth, triangle)
            volume: Volume level (0.0 to 1.0)

        Returns:
            Self for chaining
        """
        signal = self._generate_waveform(frequency, duration, waveform)
        signal *= volume
        self._audio = np.concatenate([self._audio, signal])
        return self

    def noise(
        self,
        noise_type: str = "white",
        duration: int = 1000,
        volume: float = 0.5
    ) -> 'SoundEffectsGenerator':
        """
        Generate noise.

        Args:
            noise_type: Type of noise (white, pink, brown)
            duration: Duration in milliseconds
            volume: Volume level (0.0 to 1.0)

        Returns:
            Self for chaining
        """
        num_samples = int(self.sample_rate * duration / 1000)

        if noise_type == "white":
            signal = np.random.uniform(-1, 1, num_samples)

        elif noise_type == "pink":
            # Generate pink noise using Voss-McCartney algorithm
            white = np.random.randn(num_samples)
            # Apply 1/f filter
            from scipy import signal as scipy_signal
            b, a = scipy_signal.butter(1, 0.01)
            pink = scipy_signal.lfilter(b, a, white)
            # Normalize
            signal = pink / np.max(np.abs(pink))

        elif noise_type == "brown":
            # Brown noise is integrated white noise
            white = np.random.randn(num_samples)
            brown = np.cumsum(white)
            # Normalize
            signal = brown / np.max(np.abs(brown))

        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        signal = signal.astype(np.float32) * volume
        self._audio = np.concatenate([self._audio, signal])
        return self

    def silence(self, duration: int = 1000) -> 'SoundEffectsGenerator':
        """
        Generate silence.

        Args:
            duration: Duration in milliseconds

        Returns:
            Self for chaining
        """
        num_samples = int(self.sample_rate * duration / 1000)
        signal = np.zeros(num_samples, dtype=np.float32)
        self._audio = np.concatenate([self._audio, signal])
        return self

    def dtmf(
        self,
        digit: str,
        duration: int = 200,
        volume: float = 0.7
    ) -> 'SoundEffectsGenerator':
        """
        Generate a DTMF tone for a single digit.

        Args:
            digit: DTMF digit (0-9, *, #, A-D)
            duration: Duration in milliseconds
            volume: Volume level

        Returns:
            Self for chaining
        """
        digit = digit.upper()
        if digit not in self.DTMF_FREQ:
            raise ValueError(f"Invalid DTMF digit: {digit}")

        f1, f2 = self.DTMF_FREQ[digit]

        num_samples = int(self.sample_rate * duration / 1000)
        t = np.linspace(0, duration / 1000, num_samples, dtype=np.float32)

        # DTMF is sum of two frequencies
        signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
        signal = signal / 2 * volume  # Normalize

        self._audio = np.concatenate([self._audio, signal])
        return self

    def dtmf_sequence(
        self,
        digits: str,
        tone_duration: int = 150,
        gap: int = 50
    ) -> 'SoundEffectsGenerator':
        """
        Generate DTMF sequence for multiple digits.

        Args:
            digits: String of DTMF digits
            tone_duration: Duration of each tone (ms)
            gap: Gap between tones (ms)

        Returns:
            Self for chaining
        """
        for i, digit in enumerate(digits):
            if digit == ' ':
                self.silence(gap)
                continue

            self.dtmf(digit, tone_duration)

            if i < len(digits) - 1:
                self.silence(gap)

        return self

    def beep(
        self,
        frequency: float = 800,
        duration: int = 200,
        waveform: str = "sine",
        volume: float = 0.8
    ) -> 'SoundEffectsGenerator':
        """
        Generate a single beep.

        Args:
            frequency: Beep frequency in Hz
            duration: Duration in milliseconds
            waveform: Waveform type
            volume: Volume level

        Returns:
            Self for chaining
        """
        return self.tone(frequency, duration, waveform, volume)

    def beep_sequence(
        self,
        frequencies: List[float],
        durations: Union[int, List[int]] = 200,
        gap: int = 100,
        waveform: str = "sine",
        volume: float = 0.8
    ) -> 'SoundEffectsGenerator':
        """
        Generate a sequence of beeps.

        Args:
            frequencies: List of frequencies for each beep
            durations: Duration (or list of durations) in ms
            gap: Gap between beeps (ms)
            waveform: Waveform type
            volume: Volume level

        Returns:
            Self for chaining
        """
        if isinstance(durations, int):
            durations = [durations] * len(frequencies)

        if len(durations) != len(frequencies):
            raise ValueError("Durations list must match frequencies list")

        for i, (freq, dur) in enumerate(zip(frequencies, durations)):
            self.tone(freq, dur, waveform, volume)

            if i < len(frequencies) - 1:
                self.silence(gap)

        return self

    def fade_in(self, duration_ms: int) -> 'SoundEffectsGenerator':
        """
        Apply fade in effect.

        Args:
            duration_ms: Fade duration in milliseconds

        Returns:
            Self for chaining
        """
        if len(self._audio) == 0:
            return self

        num_samples = int(self.sample_rate * duration_ms / 1000)
        num_samples = min(num_samples, len(self._audio))

        fade = np.linspace(0, 1, num_samples, dtype=np.float32)
        self._audio[:num_samples] *= fade

        return self

    def fade_out(self, duration_ms: int) -> 'SoundEffectsGenerator':
        """
        Apply fade out effect.

        Args:
            duration_ms: Fade duration in milliseconds

        Returns:
            Self for chaining
        """
        if len(self._audio) == 0:
            return self

        num_samples = int(self.sample_rate * duration_ms / 1000)
        num_samples = min(num_samples, len(self._audio))

        fade = np.linspace(1, 0, num_samples, dtype=np.float32)
        self._audio[-num_samples:] *= fade

        return self

    def volume(self, level: float) -> 'SoundEffectsGenerator':
        """
        Adjust volume.

        Args:
            level: Volume level (0.0 to 1.0+)

        Returns:
            Self for chaining
        """
        self._audio *= level
        return self

    def get_duration_ms(self) -> int:
        """Get current audio duration in milliseconds."""
        return int(len(self._audio) / self.sample_rate * 1000)

    def clear(self) -> 'SoundEffectsGenerator':
        """Clear all audio data."""
        self._audio = np.array([], dtype=np.float32)
        return self

    def save(
        self,
        output: str,
        bitrate: int = 192
    ) -> str:
        """
        Save audio to file.

        Args:
            output: Output file path
            bitrate: Bitrate for MP3 (kbps)

        Returns:
            Path to saved file
        """
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        format_ext = output_path.suffix.lower()

        if format_ext == '.wav':
            # Save directly using soundfile
            import soundfile as sf
            sf.write(str(output_path), self._audio, self.sample_rate)

        elif format_ext == '.mp3':
            # Use pydub for MP3
            try:
                from pydub import AudioSegment
                import io

                # Convert to 16-bit PCM
                audio_int = (self._audio * 32767).astype(np.int16)

                # Create AudioSegment
                audio_segment = AudioSegment(
                    audio_int.tobytes(),
                    frame_rate=self.sample_rate,
                    sample_width=2,  # 16-bit
                    channels=1
                )

                audio_segment.export(str(output_path), format='mp3', bitrate=f'{bitrate}k')

            except ImportError:
                raise ImportError("pydub is required for MP3 export. Install with: pip install pydub")

        else:
            # Try soundfile for other formats
            import soundfile as sf
            sf.write(str(output_path), self._audio, self.sample_rate)

        return str(output_path)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate programmatic audio effects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --tone 440 --duration 1000 --output tone.wav
  %(prog)s --noise white --duration 2000 --output noise.wav
  %(prog)s --dtmf "5551234" --output phone.wav
  %(prog)s --beeps "800,800,800" --duration 100 --gap 100 --output alert.wav
        """
    )

    parser.add_argument('--tone', '-t', type=float, help='Generate tone at frequency (Hz)')
    parser.add_argument('--noise', '-n', choices=['white', 'pink', 'brown'], help='Generate noise')
    parser.add_argument('--dtmf', help='Generate DTMF tones for digits')
    parser.add_argument('--beeps', help='Comma-separated frequencies for beep sequence')
    parser.add_argument('--duration', '-d', type=int, default=1000, help='Duration in ms')
    parser.add_argument('--gap', '-g', type=int, default=100, help='Gap between sounds (ms)')
    parser.add_argument('--waveform', '-w', default='sine',
                        choices=['sine', 'square', 'sawtooth', 'triangle'],
                        help='Waveform type for tones')
    parser.add_argument('--volume', '-v', type=float, default=0.8, help='Volume (0.0-1.0)')
    parser.add_argument('--sample-rate', type=int, default=44100, help='Sample rate (Hz)')
    parser.add_argument('--fade-in', type=int, help='Fade in duration (ms)')
    parser.add_argument('--fade-out', type=int, help='Fade out duration (ms)')
    parser.add_argument('--output', '-o', required=True, help='Output file')

    args = parser.parse_args()

    sfx = SoundEffectsGenerator(sample_rate=args.sample_rate)

    # Generate audio based on options
    if args.tone:
        sfx.tone(args.tone, args.duration, args.waveform, args.volume)
        print(f"Generated {args.waveform} tone at {args.tone}Hz")

    elif args.noise:
        sfx.noise(args.noise, args.duration, args.volume)
        print(f"Generated {args.noise} noise")

    elif args.dtmf:
        sfx.dtmf_sequence(args.dtmf, tone_duration=args.duration, gap=args.gap)
        print(f"Generated DTMF for: {args.dtmf}")

    elif args.beeps:
        frequencies = [float(f) for f in args.beeps.split(',')]
        sfx.beep_sequence(frequencies, args.duration, args.gap, args.waveform, args.volume)
        print(f"Generated beep sequence: {frequencies}")

    else:
        parser.error("Must specify --tone, --noise, --dtmf, or --beeps")

    # Apply effects
    if args.fade_in:
        sfx.fade_in(args.fade_in)

    if args.fade_out:
        sfx.fade_out(args.fade_out)

    # Save
    output = sfx.save(args.output)
    print(f"Saved: {output} ({sfx.get_duration_ms()}ms)")


if __name__ == "__main__":
    main()
