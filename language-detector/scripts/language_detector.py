#!/usr/bin/env python3
"""
Language Detector - Detect language of text.
"""

import argparse
import json
from typing import Dict, List

from langdetect import detect, detect_langs, LangDetectException
import pandas as pd


class LanguageDetector:
    """Detect language of text."""

    LANGUAGE_NAMES = {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'pl': 'Polish',
        'ru': 'Russian', 'ja': 'Japanese', 'zh-cn': 'Chinese (Simplified)',
        'zh-tw': 'Chinese (Traditional)', 'ko': 'Korean', 'ar': 'Arabic',
        'hi': 'Hindi', 'tr': 'Turkish', 'vi': 'Vietnamese', 'th': 'Thai'
    }

    def __init__(self):
        """Initialize detector."""
        pass

    def detect_language(self, text: str) -> Dict:
        """Detect language of text."""
        try:
            # Simple detection
            lang_code = detect(text)

            # With probabilities
            probs = detect_langs(text)

            result = {
                'language': lang_code,
                'language_name': self.LANGUAGE_NAMES.get(lang_code, lang_code),
                'confidence': float(probs[0].prob) if probs else 0,
                'alternatives': [
                    {
                        'language': str(p.lang),
                        'language_name': self.LANGUAGE_NAMES.get(str(p.lang), str(p.lang)),
                        'probability': float(p.prob)
                    }
                    for p in probs[:3]
                ]
            }

            return result

        except LangDetectException:
            return {
                'language': 'unknown',
                'language_name': 'Unknown',
                'confidence': 0,
                'alternatives': []
            }

    def detect_batch(self, texts: List[str]) -> List[Dict]:
        """Detect language for multiple texts."""
        return [self.detect_language(text) for text in texts]

    def analyze_dataframe(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Analyze text column in DataFrame."""
        results = []

        for text in df[column]:
            if pd.isna(text):
                result = {'language': 'unknown', 'confidence': 0}
            else:
                result = self.detect_language(str(text))

            results.append(result)

        # Add to dataframe
        df_copy = df.copy()
        df_copy['detected_language'] = [r['language'] for r in results]
        df_copy['language_name'] = [r['language_name'] for r in results]
        df_copy['confidence'] = [r['confidence'] for r in results]

        return df_copy


def main():
    parser = argparse.ArgumentParser(description="Language Detector")

    parser.add_argument("--text", help="Text to analyze")
    parser.add_argument("--file", help="CSV file with text column")
    parser.add_argument("--column", help="Text column name (for CSV)")
    parser.add_argument("--output", "-o", required=True, help="Output file")

    args = parser.parse_args()

    detector = LanguageDetector()

    if args.text:
        # Single text
        result = detector.detect_language(args.text)

        print(f"Language: {result['language_name']} ({result['language']})")
        print(f"Confidence: {result['confidence']:.2%}")

        if result['alternatives']:
            print("\nAlternatives:")
            for alt in result['alternatives']:
                print(f"  {alt['language_name']}: {alt['probability']:.2%}")

        # Export
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\nResult saved: {args.output}")

    elif args.file and args.column:
        # CSV file
        df = pd.read_csv(args.file)
        result_df = detector.analyze_dataframe(df, args.column)

        # Summary
        lang_counts = result_df['language_name'].value_counts()
        print("Language distribution:")
        for lang, count in lang_counts.items():
            print(f"  {lang}: {count} ({count/len(result_df)*100:.1f}%)")

        # Export
        result_df.to_csv(args.output, index=False)
        print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
