#!/usr/bin/env python3
"""
Readability Scorer - Calculate readability metrics for text.

Features:
- Flesch-Kincaid Grade Level
- Flesch Reading Ease
- Gunning Fog Index
- SMOG Index
- Coleman-Liau Index
- Automated Readability Index
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


class ReadabilityScorer:
    """Calculate readability metrics for text."""

    # Common vowels for syllable counting
    VOWELS = 'aeiouy'

    def __init__(self):
        """Initialize scorer."""
        pass

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower().strip()
        if not word:
            return 0

        count = 0
        prev_vowel = False

        for char in word:
            is_vowel = char in self.VOWELS
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Handle silent e
        if word.endswith('e') and count > 1:
            count -= 1

        # Handle -le endings
        if word.endswith('le') and len(word) > 2 and word[-3] not in self.VOWELS:
            count += 1

        return max(1, count)

    def _get_stats(self, text: str) -> Dict:
        """Get text statistics."""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()

        # Count sentences (split by .!?)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(1, len(sentences))

        # Count words
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = max(1, len(words))

        # Count syllables
        syllable_count = sum(self._count_syllables(w) for w in words)

        # Count complex words (3+ syllables)
        complex_words = [w for w in words if self._count_syllables(w) >= 3]
        complex_word_count = len(complex_words)

        # Count characters in words
        char_count = sum(len(w) for w in words)

        return {
            'words': word_count,
            'sentences': sentence_count,
            'syllables': syllable_count,
            'complex_words': complex_word_count,
            'characters': char_count,
            'avg_words_per_sentence': round(word_count / sentence_count, 2),
            'avg_syllables_per_word': round(syllable_count / word_count, 2),
            'avg_chars_per_word': round(char_count / word_count, 2)
        }

    def flesch_reading_ease(self, text: str) -> float:
        """
        Calculate Flesch Reading Ease score.
        Higher = easier to read (0-100 scale, can exceed)
        """
        stats = self._get_stats(text)
        score = 206.835 - 1.015 * stats['avg_words_per_sentence'] - 84.6 * stats['avg_syllables_per_word']
        return round(score, 1)

    def flesch_kincaid_grade(self, text: str) -> float:
        """
        Calculate Flesch-Kincaid Grade Level.
        Returns US grade level (e.g., 8.0 = 8th grade)
        """
        stats = self._get_stats(text)
        grade = 0.39 * stats['avg_words_per_sentence'] + 11.8 * stats['avg_syllables_per_word'] - 15.59
        return round(grade, 1)

    def gunning_fog(self, text: str) -> float:
        """
        Calculate Gunning Fog Index.
        Returns years of education needed to understand.
        """
        stats = self._get_stats(text)
        complex_ratio = stats['complex_words'] / stats['words'] * 100
        index = 0.4 * (stats['avg_words_per_sentence'] + complex_ratio)
        return round(index, 1)

    def smog_index(self, text: str) -> float:
        """
        Calculate SMOG Index.
        Simple Measure of Gobbledygook.
        """
        stats = self._get_stats(text)
        if stats['sentences'] < 30:
            # Adjustment for short texts
            polysyllables = stats['complex_words'] * (30 / stats['sentences'])
        else:
            polysyllables = stats['complex_words']

        import math
        index = 1.0430 * math.sqrt(polysyllables) + 3.1291
        return round(index, 1)

    def coleman_liau(self, text: str) -> float:
        """
        Calculate Coleman-Liau Index.
        Based on characters per word rather than syllables.
        """
        stats = self._get_stats(text)
        L = stats['characters'] / stats['words'] * 100  # avg letters per 100 words
        S = stats['sentences'] / stats['words'] * 100   # avg sentences per 100 words
        index = 0.0588 * L - 0.296 * S - 15.8
        return round(index, 1)

    def ari(self, text: str) -> float:
        """
        Calculate Automated Readability Index.
        """
        stats = self._get_stats(text)
        index = 4.71 * stats['avg_chars_per_word'] + 0.5 * stats['avg_words_per_sentence'] - 21.43
        return round(index, 1)

    def analyze(self, text: str) -> Dict:
        """
        Calculate all readability metrics.

        Returns:
            Dict with all scores and statistics
        """
        stats = self._get_stats(text)

        fre = self.flesch_reading_ease(text)
        fkg = self.flesch_kincaid_grade(text)
        fog = self.gunning_fog(text)
        smog = self.smog_index(text)
        cl = self.coleman_liau(text)
        ari = self.ari(text)

        # Average grade level
        grades = [fkg, fog, smog, cl, ari]
        avg_grade = round(sum(grades) / len(grades), 1)

        # Estimate reading time (avg 200-250 wpm)
        reading_time = round(stats['words'] / 225, 1)

        return {
            'flesch_reading_ease': fre,
            'flesch_kincaid_grade': fkg,
            'gunning_fog': fog,
            'smog_index': smog,
            'coleman_liau': cl,
            'ari': ari,
            'grade_level': avg_grade,
            'reading_time_minutes': reading_time,
            'stats': stats
        }

    def compare(self, text1: str, text2: str) -> Dict:
        """Compare readability of two texts."""
        return {
            'text1': self.analyze(text1),
            'text2': self.analyze(text2)
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]

    def analyze_files(self, filepaths: List[str]) -> Dict:
        """Analyze multiple files."""
        results = {}
        for filepath in filepaths:
            text = Path(filepath).read_text()
            results[filepath] = self.analyze(text)
        return results

    def get_reading_level_label(self, grade: float) -> str:
        """Get human-readable label for grade level."""
        if grade < 6:
            return "Elementary"
        elif grade < 9:
            return "Middle School"
        elif grade < 13:
            return "High School"
        elif grade < 17:
            return "College"
        else:
            return "Graduate"

    def get_ease_label(self, ease: float) -> str:
        """Get label for Flesch Reading Ease score."""
        if ease >= 90:
            return "Very Easy"
        elif ease >= 80:
            return "Easy"
        elif ease >= 70:
            return "Fairly Easy"
        elif ease >= 60:
            return "Standard"
        elif ease >= 50:
            return "Fairly Difficult"
        elif ease >= 30:
            return "Difficult"
        else:
            return "Very Difficult"


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Calculate readability scores')
    parser.add_argument('--text', '-t', help='Text to analyze')
    parser.add_argument('--input', '-i', help='Input file')
    parser.add_argument('--input-dir', help='Directory of files')
    parser.add_argument('--output', '-o', help='Output file (json/csv)')
    parser.add_argument('--compare', nargs=2, help='Compare two files')
    parser.add_argument('--formula', choices=['flesch', 'fog', 'smog', 'all'], default='all')

    args = parser.parse_args()
    scorer = ReadabilityScorer()

    if args.compare:
        text1 = Path(args.compare[0]).read_text()
        text2 = Path(args.compare[1]).read_text()
        result = scorer.compare(text1, text2)

        print(f"\n{args.compare[0]}:")
        print(f"  Grade Level: {result['text1']['grade_level']}")
        print(f"  Reading Ease: {result['text1']['flesch_reading_ease']}")

        print(f"\n{args.compare[1]}:")
        print(f"  Grade Level: {result['text2']['grade_level']}")
        print(f"  Reading Ease: {result['text2']['flesch_reading_ease']}")

    elif args.text or args.input:
        if args.text:
            text = args.text
        else:
            text = Path(args.input).read_text()

        result = scorer.analyze(text)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {args.output}")
        else:
            print("\nReadability Analysis")
            print("=" * 40)
            print(f"Flesch Reading Ease: {result['flesch_reading_ease']} ({scorer.get_ease_label(result['flesch_reading_ease'])})")
            print(f"Flesch-Kincaid Grade: {result['flesch_kincaid_grade']}")
            print(f"Gunning Fog Index: {result['gunning_fog']}")
            print(f"SMOG Index: {result['smog_index']}")
            print(f"Coleman-Liau Index: {result['coleman_liau']}")
            print(f"Automated Readability Index: {result['ari']}")
            print(f"\nAverage Grade Level: {result['grade_level']} ({scorer.get_reading_level_label(result['grade_level'])})")
            print(f"Estimated Reading Time: {result['reading_time_minutes']} minutes")
            print(f"\nStatistics:")
            print(f"  Words: {result['stats']['words']}")
            print(f"  Sentences: {result['stats']['sentences']}")
            print(f"  Avg words/sentence: {result['stats']['avg_words_per_sentence']}")
            print(f"  Avg syllables/word: {result['stats']['avg_syllables_per_word']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
