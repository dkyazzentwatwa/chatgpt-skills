#!/usr/bin/env python3
"""
Audio Analyzer - Comprehensive audio analysis toolkit.

Features:
- Tempo/BPM detection
- Key detection
- Frequency analysis
- Loudness metrics
- Waveform visualization
- Spectrogram generation
- Chromagram plotting
- Beat grid visualization
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class TempoResult:
    """Tempo analysis results."""
    bpm: float
    confidence: float
    beats: List[float]
    beat_count: int


@dataclass
class KeyResult:
    """Key detection results."""
    key: str
    mode: str
    confidence: float
    profile: Dict[str, float]


@dataclass
class LoudnessResult:
    """Loudness analysis results."""
    rms_db: float
    peak_db: float
    lufs: float
    dynamic_range_db: float
    crest_factor: float


@dataclass
class FrequencyResult:
    """Frequency analysis results."""
    dominant_freq: float
    spectral_centroid: float
    spectral_rolloff: float
    bands: Dict[str, float]


class AudioAnalyzer:
    """Comprehensive audio analysis toolkit."""

    # Key names
    KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Frequency bands (Hz)
    FREQ_BANDS = {
        'sub_bass': (20, 60),
        'bass': (60, 250),
        'low_mid': (250, 500),
        'mid': (500, 2000),
        'high_mid': (2000, 4000),
        'high': (4000, 20000)
    }

    def __init__(self, filepath: str, sr: int = 22050):
        """
        Initialize analyzer with audio file.

        Args:
            filepath: Path to audio file
            sr: Sample rate for analysis (default 22050)
        """
        self.filepath = Path(filepath)
        self.sr = sr
        self.y = None
        self.duration = 0.0

        # Results
        self._tempo: Optional[TempoResult] = None
        self._key: Optional[KeyResult] = None
        self._loudness: Optional[LoudnessResult] = None
        self._frequency: Optional[FrequencyResult] = None

        # Load audio
        self._load_audio()

    def _load_audio(self):
        """Load audio file using librosa."""
        try:
            import librosa
            self.y, self.sr = librosa.load(self.filepath, sr=self.sr, mono=True)
            self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        except ImportError:
            raise ImportError("librosa is required. Install with: pip install librosa")
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {e}")

    def analyze(self) -> 'AudioAnalyzer':
        """Run all analyses."""
        self.analyze_tempo()
        self.analyze_key()
        self.analyze_loudness()
        self.analyze_frequency()
        return self

    def analyze_tempo(self) -> 'AudioAnalyzer':
        """Detect tempo and beat positions."""
        import librosa

        # Get tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)

        # Convert frames to time
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)

        # Calculate confidence based on beat regularity
        if len(beat_times) > 1:
            intervals = np.diff(beat_times)
            expected_interval = 60.0 / float(tempo)
            interval_variance = np.var(intervals - expected_interval)
            confidence = max(0, min(1, 1 - interval_variance * 10))
        else:
            confidence = 0.0

        self._tempo = TempoResult(
            bpm=float(tempo),
            confidence=float(confidence),
            beats=beat_times.tolist(),
            beat_count=len(beat_times)
        )

        return self

    def analyze_key(self) -> 'AudioAnalyzer':
        """Detect musical key."""
        import librosa

        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)

        # Average chroma over time
        chroma_avg = np.mean(chroma, axis=1)

        # Key profiles (Krumhansl-Schmuckler)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # Correlate with all keys
        correlations = []
        for i in range(12):
            # Rotate profiles
            major_rot = np.roll(major_profile, i)
            minor_rot = np.roll(minor_profile, i)

            # Compute correlations
            corr_major = np.corrcoef(chroma_avg, major_rot)[0, 1]
            corr_minor = np.corrcoef(chroma_avg, minor_rot)[0, 1]

            correlations.append((i, 'major', corr_major))
            correlations.append((i, 'minor', corr_minor))

        # Find best match
        best = max(correlations, key=lambda x: x[2])
        key_idx, mode, confidence = best

        # Build profile dict
        profile = {self.KEY_NAMES[i]: float(chroma_avg[i]) for i in range(12)}

        self._key = KeyResult(
            key=self.KEY_NAMES[key_idx],
            mode=mode,
            confidence=float(max(0, confidence)),
            profile=profile
        )

        return self

    def analyze_loudness(self) -> 'AudioAnalyzer':
        """Analyze loudness metrics."""
        # RMS
        rms = np.sqrt(np.mean(self.y ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        # Peak
        peak = np.max(np.abs(self.y))
        peak_db = 20 * np.log10(peak + 1e-10)

        # Simplified LUFS (approximation)
        # Real LUFS requires K-weighting filter
        lufs = rms_db - 0.691  # Rough approximation

        # Dynamic range (difference between 95th and 5th percentile)
        y_abs = np.abs(self.y)
        y_abs = y_abs[y_abs > 1e-10]  # Remove silence
        if len(y_abs) > 0:
            high = np.percentile(y_abs, 95)
            low = np.percentile(y_abs, 5)
            dynamic_range = 20 * np.log10(high / low + 1e-10)
        else:
            dynamic_range = 0.0

        # Crest factor
        crest_factor = peak / (rms + 1e-10)

        self._loudness = LoudnessResult(
            rms_db=float(rms_db),
            peak_db=float(peak_db),
            lufs=float(lufs),
            dynamic_range_db=float(dynamic_range),
            crest_factor=float(crest_factor)
        )

        return self

    def analyze_frequency(self) -> 'AudioAnalyzer':
        """Analyze frequency content."""
        import librosa

        # Compute spectrum
        D = np.abs(librosa.stft(self.y))
        freqs = librosa.fft_frequencies(sr=self.sr)

        # Average magnitude spectrum
        mag_avg = np.mean(D, axis=1)

        # Dominant frequency
        dominant_idx = np.argmax(mag_avg)
        dominant_freq = freqs[dominant_idx]

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        spectral_centroid = float(np.mean(centroid))

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr, roll_percent=0.85)
        spectral_rolloff = float(np.mean(rolloff))

        # Frequency band energy
        bands = {}
        for band_name, (low, high) in self.FREQ_BANDS.items():
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                band_energy = np.mean(mag_avg[mask])
                band_db = 20 * np.log10(band_energy + 1e-10)
            else:
                band_db = -60.0
            bands[band_name] = float(band_db)

        self._frequency = FrequencyResult(
            dominant_freq=float(dominant_freq),
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            bands=bands
        )

        return self

    def get_tempo(self) -> Dict[str, Any]:
        """Get tempo analysis results."""
        if self._tempo is None:
            self.analyze_tempo()
        return asdict(self._tempo)

    def get_key(self) -> Dict[str, Any]:
        """Get key detection results."""
        if self._key is None:
            self.analyze_key()
        return asdict(self._key)

    def get_loudness(self) -> Dict[str, Any]:
        """Get loudness analysis results."""
        if self._loudness is None:
            self.analyze_loudness()
        return asdict(self._loudness)

    def get_frequency(self) -> Dict[str, Any]:
        """Get frequency analysis results."""
        if self._frequency is None:
            self.analyze_frequency()
        return asdict(self._frequency)

    def get_results(self) -> Dict[str, Any]:
        """Get all analysis results."""
        return {
            'file': str(self.filepath),
            'duration': self.duration,
            'sample_rate': self.sr,
            'tempo': self.get_tempo(),
            'key': self.get_key(),
            'loudness': self.get_loudness(),
            'frequency': self.get_frequency()
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        results = self.get_results()

        lines = [
            f"Audio Analysis: {self.filepath.name}",
            f"Duration: {results['duration']:.1f} seconds",
            "",
            f"Tempo: {results['tempo']['bpm']:.1f} BPM (confidence: {results['tempo']['confidence']:.0%})",
            f"Key: {results['key']['key']} {results['key']['mode']} (confidence: {results['key']['confidence']:.0%})",
            "",
            "Loudness:",
            f"  RMS: {results['loudness']['rms_db']:.1f} dB",
            f"  Peak: {results['loudness']['peak_db']:.1f} dB",
            f"  LUFS: {results['loudness']['lufs']:.1f}",
            f"  Dynamic Range: {results['loudness']['dynamic_range_db']:.1f} dB",
            "",
            "Frequency:",
            f"  Dominant: {results['frequency']['dominant_freq']:.1f} Hz",
            f"  Centroid: {results['frequency']['spectral_centroid']:.1f} Hz",
            f"  Rolloff: {results['frequency']['spectral_rolloff']:.1f} Hz"
        ]

        return "\n".join(lines)

    def plot_waveform(
        self,
        output: str = "waveform.png",
        figsize: tuple = (12, 4),
        color: str = "#1f77b4",
        show_rms: bool = True
    ) -> str:
        """
        Plot waveform visualization.

        Args:
            output: Output file path
            figsize: Figure size (width, height)
            color: Waveform color
            show_rms: Show RMS envelope

        Returns:
            Path to saved file
        """
        import matplotlib.pyplot as plt
        import librosa.display

        fig, ax = plt.subplots(figsize=figsize)

        # Time axis
        times = np.linspace(0, self.duration, len(self.y))

        # Plot waveform
        ax.plot(times, self.y, color=color, alpha=0.7, linewidth=0.5)

        # RMS envelope
        if show_rms:
            import librosa
            rms = librosa.feature.rms(y=self.y)[0]
            rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=self.sr)
            ax.plot(rms_times, rms, color='red', alpha=0.8, linewidth=1.5, label='RMS')
            ax.plot(rms_times, -rms, color='red', alpha=0.8, linewidth=1.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Waveform: {self.filepath.name}')
        ax.set_xlim(0, self.duration)
        ax.axhline(0, color='gray', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def plot_spectrogram(
        self,
        output: str = "spectrogram.png",
        figsize: tuple = (12, 6),
        cmap: str = "magma",
        freq_scale: str = "log",
        max_freq: int = 8000
    ) -> str:
        """
        Plot spectrogram visualization.

        Args:
            output: Output file path
            figsize: Figure size
            cmap: Colormap (viridis, plasma, inferno, magma)
            freq_scale: Frequency scale (linear, log, mel)
            max_freq: Maximum frequency to display (Hz)

        Returns:
            Path to saved file
        """
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display

        fig, ax = plt.subplots(figsize=figsize)

        if freq_scale == 'mel':
            # Mel spectrogram
            S = librosa.feature.melspectrogram(y=self.y, sr=self.sr, fmax=max_freq)
            S_db = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(
                S_db, x_axis='time', y_axis='mel',
                sr=self.sr, fmax=max_freq, cmap=cmap, ax=ax
            )
        else:
            # Regular spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max)
            img = librosa.display.specshow(
                D, x_axis='time', y_axis=freq_scale,
                sr=self.sr, cmap=cmap, ax=ax
            )

        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(f'Spectrogram: {self.filepath.name}')

        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def plot_chromagram(
        self,
        output: str = "chromagram.png",
        figsize: tuple = (12, 4)
    ) -> str:
        """
        Plot chromagram (pitch class distribution over time).

        Args:
            output: Output file path
            figsize: Figure size

        Returns:
            Path to saved file
        """
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display

        fig, ax = plt.subplots(figsize=figsize)

        chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
        img = librosa.display.specshow(
            chroma, x_axis='time', y_axis='chroma',
            sr=self.sr, cmap='coolwarm', ax=ax
        )

        fig.colorbar(img, ax=ax)
        ax.set_title(f'Chromagram: {self.filepath.name}')

        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def plot_beats(
        self,
        output: str = "beats.png",
        figsize: tuple = (12, 4),
        show_strength: bool = True
    ) -> str:
        """
        Plot beat grid with onset strength.

        Args:
            output: Output file path
            figsize: Figure size
            show_strength: Show onset strength envelope

        Returns:
            Path to saved file
        """
        import matplotlib.pyplot as plt
        import librosa

        fig, ax = plt.subplots(figsize=figsize)

        # Ensure tempo is analyzed
        if self._tempo is None:
            self.analyze_tempo()

        times = np.linspace(0, self.duration, len(self.y))

        # Plot waveform lightly
        ax.plot(times, self.y, color='gray', alpha=0.3, linewidth=0.5)

        # Onset strength
        if show_strength:
            onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
            onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=self.sr)
            # Normalize
            onset_env = onset_env / np.max(onset_env) * np.max(np.abs(self.y))
            ax.plot(onset_times, onset_env, color='blue', alpha=0.6, label='Onset strength')

        # Plot beat markers
        for beat_time in self._tempo.beats:
            ax.axvline(beat_time, color='red', alpha=0.7, linewidth=1)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Beat Grid ({self._tempo.bpm:.1f} BPM): {self.filepath.name}')
        ax.set_xlim(0, self.duration)

        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def plot_dashboard(
        self,
        output: str = "dashboard.png",
        figsize: tuple = (14, 10)
    ) -> str:
        """
        Plot comprehensive analysis dashboard.

        Args:
            output: Output file path
            figsize: Figure size

        Returns:
            Path to saved file
        """
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display

        # Ensure all analyses are done
        self.analyze()

        fig = plt.figure(figsize=figsize)

        # Create grid
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Waveform (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        times = np.linspace(0, self.duration, len(self.y))
        ax1.plot(times, self.y, color='#1f77b4', alpha=0.7, linewidth=0.5)
        for beat_time in self._tempo.beats[:50]:  # Limit beats shown
            ax1.axvline(beat_time, color='red', alpha=0.3, linewidth=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Waveform with Beats ({self._tempo.bpm:.1f} BPM)')
        ax1.set_xlim(0, self.duration)

        # 2. Spectrogram (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max)
        img = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=self.sr, cmap='magma', ax=ax2)
        fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        ax2.set_title('Spectrogram')

        # 3. Chromagram (middle right)
        ax3 = fig.add_subplot(gs[1, 1])
        chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
        img2 = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', sr=self.sr, cmap='coolwarm', ax=ax3)
        fig.colorbar(img2, ax=ax3)
        ax3.set_title(f'Chromagram (Key: {self._key.key} {self._key.mode})')

        # 4. Frequency bands (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        bands = list(self._frequency.bands.keys())
        values = list(self._frequency.bands.values())
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(bands)))
        ax4.barh(bands, values, color=colors)
        ax4.set_xlabel('Energy (dB)')
        ax4.set_title('Frequency Bands')
        ax4.axvline(0, color='gray', linewidth=0.5)

        # 5. Info panel (bottom right)
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        info_text = f"""
FILE: {self.filepath.name}
Duration: {self.duration:.1f}s

TEMPO
BPM: {self._tempo.bpm:.1f}
Confidence: {self._tempo.confidence:.0%}
Beats: {self._tempo.beat_count}

KEY
Key: {self._key.key} {self._key.mode}
Confidence: {self._key.confidence:.0%}

LOUDNESS
RMS: {self._loudness.rms_db:.1f} dB
Peak: {self._loudness.peak_db:.1f} dB
LUFS: {self._loudness.lufs:.1f}
Dynamic Range: {self._loudness.dynamic_range_db:.1f} dB

FREQUENCY
Dominant: {self._frequency.dominant_freq:.0f} Hz
Centroid: {self._frequency.spectral_centroid:.0f} Hz
"""
        ax5.text(0.1, 0.95, info_text, transform=ax5.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Audio Analysis Dashboard', fontsize=14, fontweight='bold')
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def save_report(self, path: str) -> str:
        """
        Save analysis report to JSON.

        Args:
            path: Output file path

        Returns:
            Path to saved file
        """
        results = self.get_results()

        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

        return path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Comprehensive audio analysis toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input song.mp3 --output-dir ./analysis/
  %(prog)s --input song.mp3 --analyze tempo key --output report.json
  %(prog)s --input song.mp3 --plot spectrogram --output spec.png
  %(prog)s --input song.mp3 --dashboard --output dashboard.png
        """
    )

    parser.add_argument('--input', '-i', required=True, help='Input audio file')
    parser.add_argument('--input-dir', help='Directory of audio files for batch processing')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--output-dir', default='.', help='Output directory (default: current)')
    parser.add_argument('--analyze', nargs='+',
                        choices=['tempo', 'key', 'loudness', 'frequency', 'all'],
                        default=['all'], help='Analysis types to run')
    parser.add_argument('--plot', choices=['waveform', 'spectrogram', 'chromagram', 'beats', 'dashboard'],
                        help='Generate specific plot')
    parser.add_argument('--dashboard', action='store_true', help='Generate full dashboard')
    parser.add_argument('--format', choices=['json', 'txt'], default='json', help='Output format')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate for analysis')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process files
    if args.input_dir:
        # Batch mode
        audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aiff'}
        files = [f for f in Path(args.input_dir).iterdir()
                 if f.suffix.lower() in audio_extensions]
    else:
        files = [Path(args.input)]

    for filepath in files:
        print(f"Analyzing: {filepath.name}")

        try:
            analyzer = AudioAnalyzer(str(filepath), sr=args.sr)

            # Run analyses
            if 'all' in args.analyze:
                analyzer.analyze()
            else:
                if 'tempo' in args.analyze:
                    analyzer.analyze_tempo()
                if 'key' in args.analyze:
                    analyzer.analyze_key()
                if 'loudness' in args.analyze:
                    analyzer.analyze_loudness()
                if 'frequency' in args.analyze:
                    analyzer.analyze_frequency()

            # Generate outputs
            base_name = filepath.stem

            if args.dashboard or args.plot == 'dashboard':
                out_path = args.output or os.path.join(args.output_dir, f"{base_name}_dashboard.png")
                analyzer.plot_dashboard(out_path)
                print(f"  Dashboard: {out_path}")

            elif args.plot:
                out_path = args.output or os.path.join(args.output_dir, f"{base_name}_{args.plot}.png")
                if args.plot == 'waveform':
                    analyzer.plot_waveform(out_path)
                elif args.plot == 'spectrogram':
                    analyzer.plot_spectrogram(out_path)
                elif args.plot == 'chromagram':
                    analyzer.plot_chromagram(out_path)
                elif args.plot == 'beats':
                    analyzer.plot_beats(out_path)
                print(f"  Plot: {out_path}")

            else:
                # Output report
                if args.format == 'json':
                    out_path = args.output or os.path.join(args.output_dir, f"{base_name}_analysis.json")
                    analyzer.save_report(out_path)
                    print(f"  Report: {out_path}")
                else:
                    summary = analyzer.get_summary()
                    if args.output:
                        with open(args.output, 'w') as f:
                            f.write(summary)
                        print(f"  Summary: {args.output}")
                    else:
                        print(summary)

        except Exception as e:
            print(f"  Error: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
