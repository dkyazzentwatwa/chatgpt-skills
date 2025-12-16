#!/usr/bin/env python3
"""
Word Cloud Generator - Generate styled word clouds from text

Features:
- Multiple input sources (text, file, frequency dict)
- Custom shapes (rectangle, circle, mask image)
- Color schemes (matplotlib colormaps or custom)
- Font selection
- Stopword filtering
- PNG/SVG export
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import Counter

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


# Default configuration
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 400
DEFAULT_MAX_WORDS = 200
DEFAULT_MIN_WORD_LENGTH = 1
DEFAULT_COLORMAP = 'viridis'
DEFAULT_BACKGROUND = 'white'


class WordCloudGenerator:
    """
    Generate word clouds from text or frequency dictionaries.

    Example:
        wc = WordCloudGenerator("Hello world hello")
        wc.colors("plasma").shape("circle").generate().save("output.png")
    """

    def __init__(self, text: str = None, frequencies: Dict[str, int] = None):
        """
        Initialize with text or frequency dictionary.

        Args:
            text: Input text string
            frequencies: Dictionary of word -> frequency
        """
        self._text = text
        self._frequencies = frequencies
        self._wordcloud = None

        # Configuration
        self._width = DEFAULT_WIDTH
        self._height = DEFAULT_HEIGHT
        self._max_words = DEFAULT_MAX_WORDS
        self._min_word_length = DEFAULT_MIN_WORD_LENGTH
        self._colormap = DEFAULT_COLORMAP
        self._custom_colors = None
        self._background_color = DEFAULT_BACKGROUND
        self._font_path = None
        self._mask = None
        self._stopwords = set(STOPWORDS)
        self._use_stopwords = True

    @classmethod
    def from_file(cls, filepath: str) -> 'WordCloudGenerator':
        """
        Create generator from text file.

        Args:
            filepath: Path to text file

        Returns:
            WordCloudGenerator instance
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        text = path.read_text(encoding='utf-8')
        return cls(text=text)

    def shape(self, shape: str = 'rectangle', mask: str = None) -> 'WordCloudGenerator':
        """
        Set the shape of the word cloud.

        Args:
            shape: 'rectangle', 'circle', or 'square'
            mask: Path to mask image (white areas filled with words)

        Returns:
            Self for chaining
        """
        if mask:
            # Load custom mask image
            mask_path = Path(mask)
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask image not found: {mask}")

            mask_img = np.array(Image.open(mask_path).convert('L'))
            # Invert if needed (wordcloud expects white=words)
            self._mask = mask_img
            self._width = mask_img.shape[1]
            self._height = mask_img.shape[0]
        elif shape == 'circle':
            self._mask = self._create_circle_mask()
        elif shape == 'square':
            self._mask = self._create_square_mask()
        else:
            self._mask = None  # Rectangle (default)

        return self

    def _create_circle_mask(self) -> np.ndarray:
        """Create a circular mask."""
        x, y = np.ogrid[:self._height, :self._width]
        center_x, center_y = self._width // 2, self._height // 2
        radius = min(center_x, center_y)
        mask = ((x - center_y) ** 2 + (y - center_x) ** 2) > radius ** 2
        mask = 255 * mask.astype(int)
        return mask

    def _create_square_mask(self) -> np.ndarray:
        """Create a square mask."""
        size = min(self._width, self._height)
        mask = np.ones((self._height, self._width), dtype=np.uint8) * 255

        start_x = (self._width - size) // 2
        start_y = (self._height - size) // 2
        mask[start_y:start_y + size, start_x:start_x + size] = 0

        return mask

    def colors(self, colormap: str = None, custom: List[str] = None) -> 'WordCloudGenerator':
        """
        Set color scheme.

        Args:
            colormap: Matplotlib colormap name (e.g., 'viridis', 'plasma')
            custom: List of custom colors (hex or named)

        Returns:
            Self for chaining
        """
        if custom:
            self._custom_colors = custom
            self._colormap = None
        else:
            self._colormap = colormap or DEFAULT_COLORMAP
            self._custom_colors = None

        return self

    def font(self, font_path: str) -> 'WordCloudGenerator':
        """
        Set custom font.

        Args:
            font_path: Path to TrueType font file

        Returns:
            Self for chaining
        """
        path = Path(font_path)
        if not path.exists():
            raise FileNotFoundError(f"Font file not found: {font_path}")

        self._font_path = str(path)
        return self

    def stopwords(self, words: List[str] = None, use_default: bool = True) -> 'WordCloudGenerator':
        """
        Configure stopwords filtering.

        Args:
            words: Additional stopwords to filter
            use_default: Whether to use built-in stopwords

        Returns:
            Self for chaining
        """
        if use_default:
            self._stopwords = set(STOPWORDS)
        else:
            self._stopwords = set()

        if words:
            self._stopwords.update(w.lower() for w in words)

        self._use_stopwords = True
        return self

    def no_stopwords(self) -> 'WordCloudGenerator':
        """
        Disable stopword filtering.

        Returns:
            Self for chaining
        """
        self._use_stopwords = False
        self._stopwords = set()
        return self

    def max_words(self, count: int) -> 'WordCloudGenerator':
        """
        Set maximum number of words.

        Args:
            count: Maximum words to display

        Returns:
            Self for chaining
        """
        self._max_words = count
        return self

    def min_word_length(self, length: int) -> 'WordCloudGenerator':
        """
        Set minimum word length.

        Args:
            length: Minimum character count for words

        Returns:
            Self for chaining
        """
        self._min_word_length = length
        return self

    def size(self, width: int, height: int) -> 'WordCloudGenerator':
        """
        Set output image size.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Self for chaining
        """
        self._width = width
        self._height = height

        # Recreate shape mask if using circle/square
        if self._mask is not None and not hasattr(self, '_custom_mask'):
            # Check if it's a generated mask (circle/square) vs custom
            pass  # Keep existing mask

        return self

    def background(self, color: str) -> 'WordCloudGenerator':
        """
        Set background color.

        Args:
            color: Background color (name or hex)

        Returns:
            Self for chaining
        """
        self._background_color = color
        return self

    def _get_color_func(self):
        """Get color function for word cloud."""
        if self._custom_colors:
            colors = self._custom_colors

            def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                return colors[hash(word) % len(colors)]

            return color_func
        return None

    def generate(self) -> 'WordCloudGenerator':
        """
        Generate the word cloud.

        Returns:
            Self for chaining
        """
        # Build wordcloud parameters
        params = {
            'width': self._width,
            'height': self._height,
            'max_words': self._max_words,
            'min_word_length': self._min_word_length,
            'background_color': self._background_color,
            'colormap': self._colormap,
        }

        if self._mask is not None:
            params['mask'] = self._mask
            params['contour_width'] = 0

        if self._font_path:
            params['font_path'] = self._font_path

        if self._use_stopwords:
            params['stopwords'] = self._stopwords
        else:
            params['stopwords'] = set()

        # Create WordCloud instance
        wc = WordCloud(**params)

        # Apply custom colors if specified
        color_func = self._get_color_func()

        # Generate from text or frequencies
        if self._frequencies:
            self._wordcloud = wc.generate_from_frequencies(self._frequencies)
        elif self._text:
            self._wordcloud = wc.generate(self._text)
        else:
            raise ValueError("No text or frequencies provided")

        # Recolor if custom colors
        if color_func:
            self._wordcloud.recolor(color_func=color_func)

        return self

    def save(self, path: str, format: str = None) -> str:
        """
        Save word cloud to file.

        Args:
            path: Output file path
            format: 'png' or 'svg' (auto-detected from extension)

        Returns:
            Path to saved file
        """
        if self._wordcloud is None:
            self.generate()

        output_path = Path(path)

        # Determine format
        if format is None:
            format = output_path.suffix.lower().lstrip('.')

        if format == 'svg':
            # SVG export via matplotlib
            fig, ax = plt.subplots(figsize=(self._width / 100, self._height / 100), dpi=100)
            ax.imshow(self._wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(str(output_path), format='svg', bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            # PNG export
            self._wordcloud.to_file(str(output_path))

        return str(output_path)

    def show(self) -> None:
        """Display word cloud using matplotlib."""
        if self._wordcloud is None:
            self.generate()

        plt.figure(figsize=(self._width / 100, self._height / 100), dpi=100)
        plt.imshow(self._wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()

    def get_frequencies(self) -> Dict[str, int]:
        """
        Get word frequencies from generated cloud.

        Returns:
            Dictionary of word -> frequency
        """
        if self._wordcloud is None:
            self.generate()

        return dict(self._wordcloud.words_)

    def to_array(self) -> np.ndarray:
        """
        Get word cloud as numpy array.

        Returns:
            RGB image array
        """
        if self._wordcloud is None:
            self.generate()

        return self._wordcloud.to_array()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate word clouds from text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wordcloud_gen.py --text "Hello world" --output cloud.png
  python wordcloud_gen.py --file article.txt --colors plasma --output cloud.png
  python wordcloud_gen.py --file data.txt --shape circle --max-words 100 --output cloud.png
  python wordcloud_gen.py --file speech.txt --mask logo.png --output branded.png
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', '-t', help='Input text string')
    input_group.add_argument('--file', '-f', help='Input text file')

    # Output options
    parser.add_argument('--output', '-o', default='wordcloud.png',
                        help='Output file path (default: wordcloud.png)')

    # Shape options
    parser.add_argument('--shape', '-s', choices=['rectangle', 'circle', 'square'],
                        default='rectangle', help='Word cloud shape')
    parser.add_argument('--mask', '-m', help='Custom mask image path')

    # Color options
    parser.add_argument('--colors', '-c', default='viridis',
                        help='Colormap name (default: viridis)')
    parser.add_argument('--background', '-bg', default='white',
                        help='Background color (default: white)')

    # Size options
    parser.add_argument('--width', '-W', type=int, default=800,
                        help='Image width (default: 800)')
    parser.add_argument('--height', '-H', type=int, default=400,
                        help='Image height (default: 400)')

    # Word options
    parser.add_argument('--max-words', type=int, default=200,
                        help='Maximum words (default: 200)')
    parser.add_argument('--min-length', type=int, default=1,
                        help='Minimum word length (default: 1)')

    # Font and stopwords
    parser.add_argument('--font', help='Custom font file path')
    parser.add_argument('--no-stopwords', action='store_true',
                        help='Disable stopword filtering')
    parser.add_argument('--stopwords', help='Additional stopwords (comma-separated)')

    # Frequency output
    parser.add_argument('--show-frequencies', action='store_true',
                        help='Print word frequencies')

    args = parser.parse_args()

    try:
        # Create generator
        if args.file:
            wc = WordCloudGenerator.from_file(args.file)
        else:
            wc = WordCloudGenerator(text=args.text)

        # Configure
        wc.size(args.width, args.height)
        wc.colors(args.colors)
        wc.background(args.background)
        wc.max_words(args.max_words)
        wc.min_word_length(args.min_length)

        # Shape
        if args.mask:
            wc.shape(mask=args.mask)
        else:
            wc.shape(args.shape)

        # Font
        if args.font:
            wc.font(args.font)

        # Stopwords
        if args.no_stopwords:
            wc.no_stopwords()
        elif args.stopwords:
            additional = [w.strip() for w in args.stopwords.split(',')]
            wc.stopwords(words=additional)

        # Generate and save
        wc.generate()
        output_path = wc.save(args.output)
        print(f"Word cloud saved to: {output_path}")

        # Show frequencies if requested
        if args.show_frequencies:
            print("\nTop words:")
            freqs = wc.get_frequencies()
            for word, freq in list(freqs.items())[:20]:
                print(f"  {word}: {freq:.4f}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
