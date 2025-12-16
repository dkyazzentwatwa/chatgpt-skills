#!/usr/bin/env python3
"""
Sentiment Analyzer - Analyze text sentiment with scoring and visualization.

Features:
- Sentiment classification (positive/negative/neutral)
- Polarity and subjectivity scoring
- Batch CSV processing
- Trend analysis and visualization
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    text: str
    sentiment: str
    score: float
    confidence: float
    subjectivity: float
    emotions: Dict[str, float]


class SentimentAnalyzer:
    """Analyze text sentiment."""

    def __init__(self):
        """Initialize analyzer."""
        self._results: List[SentimentResult] = []
        self._textblob = None
        self._load_textblob()

    def _load_textblob(self):
        """Load TextBlob library."""
        try:
            from textblob import TextBlob
            self._textblob = TextBlob
        except ImportError:
            raise ImportError("textblob is required. Install with: pip install textblob")

    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Dict with sentiment results
        """
        blob = self._textblob(text)

        # Get polarity (-1 to 1) and subjectivity (0 to 1)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Classify sentiment
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Calculate confidence (higher for extreme values)
        confidence = min(abs(polarity) * 1.5 + 0.3, 1.0)

        # Simple emotion detection based on polarity and words
        emotions = self._detect_emotions(text, polarity)

        result = SentimentResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            sentiment=sentiment,
            score=round(polarity, 3),
            confidence=round(confidence, 3),
            subjectivity=round(subjectivity, 3),
            emotions=emotions
        )

        self._results.append(result)
        return asdict(result)

    def _detect_emotions(self, text: str, polarity: float) -> Dict[str, float]:
        """Simple emotion detection."""
        text_lower = text.lower()

        # Simple keyword-based emotion detection
        joy_words = ['love', 'happy', 'joy', 'excited', 'wonderful', 'great', 'amazing', 'fantastic']
        anger_words = ['angry', 'hate', 'furious', 'annoyed', 'frustrated', 'terrible']
        sadness_words = ['sad', 'depressed', 'unhappy', 'disappointed', 'sorry', 'regret']
        fear_words = ['afraid', 'scared', 'worried', 'anxious', 'nervous', 'terrified']
        surprise_words = ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected']

        def score_emotion(words):
            count = sum(1 for w in words if w in text_lower)
            return min(count * 0.3, 1.0)

        emotions = {
            'joy': score_emotion(joy_words) if polarity > 0 else 0,
            'anger': score_emotion(anger_words) if polarity < 0 else 0,
            'sadness': score_emotion(sadness_words),
            'fear': score_emotion(fear_words),
            'surprise': score_emotion(surprise_words)
        }

        # Normalize
        total = sum(emotions.values()) or 1
        return {k: round(v / total, 3) if total > 1 else round(v, 3) for k, v in emotions.items()}

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]

    def analyze_csv(
        self,
        filepath: str,
        text_column: str = "text",
        output: Optional[str] = None
    ) -> List[Dict]:
        """
        Analyze CSV file.

        Args:
            filepath: Path to CSV file
            text_column: Column containing text
            output: Optional output CSV path

        Returns:
            List of results
        """
        import pandas as pd

        df = pd.read_csv(filepath)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")

        results = []
        for text in df[text_column].dropna():
            results.append(self.analyze(str(text)))

        if output:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output, index=False)

        return results

    def plot_distribution(self, output: str = "sentiment_distribution.png"):
        """Plot sentiment distribution."""
        import matplotlib.pyplot as plt

        if not self._results:
            raise ValueError("No results to plot. Run analyze first.")

        sentiments = [r.sentiment for r in self._results]
        counts = {
            'positive': sentiments.count('positive'),
            'neutral': sentiments.count('neutral'),
            'negative': sentiments.count('negative')
        }

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#2ecc71', '#95a5a6', '#e74c3c']
        bars = ax.bar(counts.keys(), counts.values(), color=colors)

        ax.set_ylabel('Count')
        ax.set_title('Sentiment Distribution')

        for bar, count in zip(bars, counts.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output, dpi=150)
        plt.close()

        return output

    def plot_trend(self, output: str = "sentiment_trend.png", title: str = "Sentiment Over Time"):
        """Plot sentiment trend over results."""
        import matplotlib.pyplot as plt

        if not self._results:
            raise ValueError("No results to plot.")

        scores = [r.score for r in self._results]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(scores, marker='o', markersize=3, linewidth=1)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.fill_between(range(len(scores)), scores, 0, alpha=0.3)

        ax.set_xlabel('Index')
        ax.set_ylabel('Sentiment Score')
        ax.set_title(title)
        ax.set_ylim(-1.1, 1.1)

        plt.tight_layout()
        plt.savefig(output, dpi=150)
        plt.close()

        return output


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Analyze text sentiment')
    parser.add_argument('--text', '-t', help='Text to analyze')
    parser.add_argument('--input', '-i', help='Input CSV file')
    parser.add_argument('--column', '-c', default='text', help='Text column name')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--plot', '-p', help='Save distribution plot')
    parser.add_argument('--trend', help='Save trend plot')
    parser.add_argument('--format', '-f', choices=['json', 'csv'], default='json')

    args = parser.parse_args()
    analyzer = SentimentAnalyzer()

    if args.text:
        result = analyzer.analyze(args.text)
        print(json.dumps(result, indent=2))

    elif args.input:
        results = analyzer.analyze_csv(args.input, args.column, args.output)

        if args.format == 'json' and not args.output:
            print(json.dumps(results, indent=2))
        else:
            print(f"Analyzed {len(results)} texts")

            # Summary
            sentiments = [r['sentiment'] for r in results]
            print(f"Positive: {sentiments.count('positive')}")
            print(f"Neutral: {sentiments.count('neutral')}")
            print(f"Negative: {sentiments.count('negative')}")

        if args.plot:
            analyzer.plot_distribution(args.plot)
            print(f"Distribution plot saved: {args.plot}")

        if args.trend:
            analyzer.plot_trend(args.trend)
            print(f"Trend plot saved: {args.trend}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
