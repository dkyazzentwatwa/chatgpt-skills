#!/usr/bin/env python3
"""
Keyword Extractor - Extract keywords using TF-IDF, RAKE, and frequency analysis.

Features:
- Multiple extraction algorithms
- Key phrase extraction
- Word cloud generation
- Batch processing
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional, Set
from collections import Counter


class KeywordExtractor:
    """Extract keywords from text."""

    # Default English stopwords
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
        'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 'can', 'just', 'should', 'now', 'i', 'you', 'we', 'your', 'our'
    }

    def __init__(
        self,
        method: str = "tfidf",
        max_keywords: int = 20,
        min_word_length: int = 3,
        ngram_range: Tuple[int, int] = (1, 3)
    ):
        """
        Initialize extractor.

        Args:
            method: Algorithm (tfidf, rake, frequency)
            max_keywords: Maximum keywords to return
            min_word_length: Minimum word length
            ngram_range: Range of n-grams to consider
        """
        self.method = method
        self.max_keywords = max_keywords
        self.min_word_length = min_word_length
        self.ngram_range = ngram_range
        self.stopwords: Set[str] = self.STOPWORDS.copy()
        self._last_keywords: List[Tuple[str, float]] = []

    def add_stopwords(self, words: List[str]):
        """Add custom stopwords."""
        self.stopwords.update(w.lower() for w in words)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        return [w for w in words if len(w) >= self.min_word_length and w not in self.stopwords]

    def _get_ngrams(self, words: List[str]) -> List[str]:
        """Generate n-grams from words."""
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngrams.append(ngram)
        return ngrams

    def extract(
        self,
        text: str,
        method: Optional[str] = None,
        top_n: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords from text.

        Args:
            text: Text to analyze
            method: Algorithm to use (overrides default)
            top_n: Number of keywords (overrides default)

        Returns:
            List of (keyword, score) tuples
        """
        method = method or self.method
        top_n = top_n or self.max_keywords

        if method == "tfidf":
            keywords = self._extract_tfidf(text)
        elif method == "rake":
            keywords = self._extract_rake(text)
        elif method == "frequency":
            keywords = self._extract_frequency(text)
        else:
            raise ValueError(f"Unknown method: {method}")

        self._last_keywords = keywords[:top_n]
        return self._last_keywords

    def _extract_frequency(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords by frequency."""
        words = self._tokenize(text)
        ngrams = self._get_ngrams(words)
        counts = Counter(ngrams)

        max_count = max(counts.values()) if counts else 1
        return [(kw, count/max_count) for kw, count in counts.most_common()]

    def _extract_tfidf(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError("scikit-learn is required for TF-IDF. Install with: pip install scikit-learn")

        vectorizer = TfidfVectorizer(
            stop_words=list(self.stopwords),
            ngram_range=self.ngram_range,
            min_df=1,
            max_features=self.max_keywords * 5
        )

        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            keywords = [(feature_names[i], scores[i]) for i in range(len(scores)) if scores[i] > 0]
            keywords.sort(key=lambda x: x[1], reverse=True)
            return keywords
        except ValueError:
            return self._extract_frequency(text)

    def _extract_rake(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords using RAKE-like algorithm."""
        # Split into phrases by stopwords and punctuation
        text_lower = text.lower()
        phrase_pattern = r'[^a-z\s]+'
        phrases = re.split(phrase_pattern, text_lower)

        # Filter and clean phrases
        clean_phrases = []
        for phrase in phrases:
            words = phrase.split()
            # Remove leading/trailing stopwords
            while words and words[0] in self.stopwords:
                words.pop(0)
            while words and words[-1] in self.stopwords:
                words.pop()

            if words and len(' '.join(words)) >= self.min_word_length:
                clean_phrases.append(' '.join(words))

        # Score phrases
        phrase_scores = Counter(clean_phrases)

        # Calculate word scores
        word_freq = Counter()
        word_degree = Counter()
        for phrase in clean_phrases:
            words = phrase.split()
            for word in words:
                word_freq[word] += 1
                word_degree[word] += len(words)

        word_score = {w: word_degree[w] / word_freq[w] for w in word_freq}

        # Score phrases by sum of word scores
        results = []
        for phrase, count in phrase_scores.items():
            words = phrase.split()
            score = sum(word_score.get(w, 0) for w in words) * count
            results.append((phrase, score))

        # Normalize
        max_score = max(r[1] for r in results) if results else 1
        results = [(kw, score/max_score) for kw, score in results]
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def get_keywords(self, text: str, top_n: Optional[int] = None) -> List[str]:
        """Get just the keyword strings."""
        keywords = self.extract(text, top_n=top_n)
        return [kw for kw, _ in keywords]

    def extract_from_file(self, filepath: str) -> List[Tuple[str, float]]:
        """Extract keywords from file."""
        text = Path(filepath).read_text()
        return self.extract(text)

    def to_wordcloud(self, output: str = "wordcloud.png", colormap: str = "viridis"):
        """Generate word cloud from last extraction."""
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("wordcloud and matplotlib required. Install with: pip install wordcloud matplotlib")

        if not self._last_keywords:
            raise ValueError("No keywords extracted. Run extract() first.")

        freq_dict = {kw: score for kw, score in self._last_keywords}

        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap=colormap,
            max_words=100
        ).generate_from_frequencies(freq_dict)

        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()

        return output

    def plot_keywords(self, output: str = "keywords.png", top_n: int = 15):
        """Plot bar chart of top keywords."""
        import matplotlib.pyplot as plt

        keywords = self._last_keywords[:top_n]
        if not keywords:
            raise ValueError("No keywords to plot.")

        labels = [kw for kw, _ in keywords]
        scores = [score for _, score in keywords]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(labels[::-1], scores[::-1])

        ax.set_xlabel('Score')
        ax.set_title('Top Keywords')

        plt.tight_layout()
        plt.savefig(output, dpi=150)
        plt.close()

        return output

    def to_json(self, output: str):
        """Export keywords to JSON."""
        data = [{'keyword': kw, 'score': score} for kw, score in self._last_keywords]
        with open(output, 'w') as f:
            json.dump(data, f, indent=2)
        return output

    def to_csv(self, output: str):
        """Export keywords to CSV."""
        with open(output, 'w') as f:
            f.write("keyword,score\n")
            for kw, score in self._last_keywords:
                f.write(f'"{kw}",{score}\n')
        return output


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Extract keywords from text')
    parser.add_argument('--text', '-t', help='Text to analyze')
    parser.add_argument('--input', '-i', help='Input file')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--method', '-m', choices=['tfidf', 'rake', 'frequency'], default='tfidf')
    parser.add_argument('--top', '-n', type=int, default=20, help='Number of keywords')
    parser.add_argument('--ngrams', default='1,3', help='N-gram range (e.g., "1,2")')
    parser.add_argument('--wordcloud', '-w', help='Generate word cloud')
    parser.add_argument('--stopwords', help='Custom stopwords file')

    args = parser.parse_args()

    ngram_range = tuple(int(x) for x in args.ngrams.split(','))
    extractor = KeywordExtractor(
        method=args.method,
        max_keywords=args.top,
        ngram_range=ngram_range
    )

    if args.stopwords:
        with open(args.stopwords) as f:
            custom = [line.strip() for line in f if line.strip()]
            extractor.add_stopwords(custom)

    if args.text:
        text = args.text
    elif args.input:
        text = Path(args.input).read_text()
    else:
        parser.print_help()
        return

    keywords = extractor.extract(text)

    print(f"\nTop {len(keywords)} Keywords ({args.method}):")
    for kw, score in keywords:
        print(f"  {score:.3f}: {kw}")

    if args.output:
        if args.output.endswith('.json'):
            extractor.to_json(args.output)
        else:
            extractor.to_csv(args.output)
        print(f"\nSaved to: {args.output}")

    if args.wordcloud:
        extractor.to_wordcloud(args.wordcloud)
        print(f"Word cloud saved: {args.wordcloud}")


if __name__ == "__main__":
    main()
