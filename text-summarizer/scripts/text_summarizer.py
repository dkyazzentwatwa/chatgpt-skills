#!/usr/bin/env python3
"""
Text Summarizer - Generate extractive summaries from text.

Features:
- TextRank, LSA, and frequency-based summarization
- Control by ratio, sentence count, or word count
- Key points extraction
- Batch processing
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Union
import numpy as np


class TextSummarizer:
    """Generate extractive text summaries."""

    def __init__(self, method: str = "textrank", language: str = "english"):
        """
        Initialize summarizer.

        Args:
            method: Algorithm (textrank, lsa, frequency)
            language: Text language
        """
        self.method = method
        self.language = language
        self.min_sentence_length = 10

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) >= self.min_sentence_length]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\b\w+\b', text.lower())

    def summarize(
        self,
        text: str,
        ratio: Optional[float] = None,
        num_sentences: Optional[int] = None,
        max_words: Optional[int] = None,
        preserve_order: bool = True,
        include_first: bool = False
    ) -> str:
        """
        Generate summary of text.

        Args:
            text: Text to summarize
            ratio: Proportion of sentences to keep (0.0-1.0)
            num_sentences: Exact number of sentences
            max_words: Maximum words in summary
            preserve_order: Keep original sentence order
            include_first: Always include first sentence

        Returns:
            Summary text
        """
        sentences = self._split_sentences(text)

        if not sentences:
            return text

        # Default to ratio if nothing specified
        if ratio is None and num_sentences is None and max_words is None:
            ratio = 0.2

        # Calculate number of sentences to keep
        if num_sentences is not None:
            n_keep = min(num_sentences, len(sentences))
        elif ratio is not None:
            n_keep = max(1, int(len(sentences) * ratio))
        else:
            n_keep = len(sentences)  # Will filter by words later

        # Score sentences
        if self.method == "textrank":
            scores = self._score_textrank(sentences)
        elif self.method == "lsa":
            scores = self._score_lsa(sentences)
        elif self.method == "frequency":
            scores = self._score_frequency(sentences)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Select top sentences
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        selected_indices = set()

        # Always include first if requested
        if include_first:
            selected_indices.add(0)

        # Add top-scoring sentences
        for idx, _ in indexed_scores:
            if len(selected_indices) >= n_keep:
                break
            selected_indices.add(idx)

        # Filter by word count if specified
        if max_words:
            selected_sentences = []
            word_count = 0
            for idx in sorted(selected_indices):
                sentence_words = len(self._tokenize(sentences[idx]))
                if word_count + sentence_words <= max_words:
                    selected_sentences.append(sentences[idx])
                    word_count += sentence_words
            return ' '.join(selected_sentences)

        # Preserve order or rank order
        if preserve_order:
            selected_indices = sorted(selected_indices)

        return ' '.join(sentences[i] for i in selected_indices)

    def _score_frequency(self, sentences: List[str]) -> List[float]:
        """Score sentences by word frequency."""
        # Count word frequencies across all sentences
        word_freq = {}
        for sentence in sentences:
            for word in self._tokenize(sentence):
                word_freq[word] = word_freq.get(word, 0) + 1

        # Score each sentence by sum of word frequencies
        scores = []
        for sentence in sentences:
            words = self._tokenize(sentence)
            if not words:
                scores.append(0)
            else:
                score = sum(word_freq.get(w, 0) for w in words) / len(words)
                scores.append(score)

        # Normalize
        max_score = max(scores) if scores else 1
        return [s / max_score for s in scores]

    def _score_textrank(self, sentences: List[str]) -> List[float]:
        """Score sentences using TextRank algorithm."""
        n = len(sentences)
        if n <= 1:
            return [1.0] * n

        # Build similarity matrix
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            words_i = set(self._tokenize(sentences[i]))
            for j in range(n):
                if i == j:
                    continue
                words_j = set(self._tokenize(sentences[j]))

                # Jaccard similarity
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)
                if union > 0:
                    similarity_matrix[i][j] = intersection / union

        # Normalize
        row_sums = similarity_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        similarity_matrix = similarity_matrix / row_sums[:, np.newaxis]

        # Power iteration
        scores = np.ones(n) / n
        damping = 0.85

        for _ in range(100):
            new_scores = (1 - damping) / n + damping * similarity_matrix.T @ scores
            if np.allclose(scores, new_scores):
                break
            scores = new_scores

        return scores.tolist()

    def _score_lsa(self, sentences: List[str]) -> List[float]:
        """Score sentences using LSA (Latent Semantic Analysis)."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
        except ImportError:
            # Fall back to frequency
            return self._score_frequency(sentences)

        if len(sentences) < 2:
            return [1.0] * len(sentences)

        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            return self._score_frequency(sentences)

        # SVD
        n_components = min(5, len(sentences) - 1, tfidf_matrix.shape[1] - 1)
        if n_components < 1:
            return self._score_frequency(sentences)

        svd = TruncatedSVD(n_components=n_components)
        svd_matrix = svd.fit_transform(tfidf_matrix)

        # Score by sum of absolute concept weights
        scores = np.abs(svd_matrix).sum(axis=1)

        # Normalize
        max_score = scores.max() if scores.max() > 0 else 1
        return (scores / max_score).tolist()

    def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """Extract key points as bullet points."""
        summary = self.summarize(text, num_sentences=num_points, preserve_order=True)
        sentences = self._split_sentences(summary)
        return sentences[:num_points]

    def summarize_file(
        self,
        filepath: str,
        ratio: Optional[float] = None,
        num_sentences: Optional[int] = None
    ) -> str:
        """Summarize text file."""
        text = Path(filepath).read_text()
        return self.summarize(text, ratio=ratio, num_sentences=num_sentences)

    def summarize_batch(
        self,
        texts: List[str],
        ratio: float = 0.2
    ) -> List[str]:
        """Summarize multiple texts."""
        return [self.summarize(text, ratio=ratio) for text in texts]

    def summarize_directory(
        self,
        input_dir: str,
        ratio: float = 0.2,
        pattern: str = "*.txt"
    ) -> dict:
        """Summarize all text files in directory."""
        results = {}
        for filepath in Path(input_dir).glob(pattern):
            text = filepath.read_text()
            results[filepath.name] = self.summarize(text, ratio=ratio)
        return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Generate text summaries')
    parser.add_argument('--input', '-i', help='Input file')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--input-dir', help='Input directory for batch')
    parser.add_argument('--output-dir', help='Output directory for batch')
    parser.add_argument('--ratio', '-r', type=float, help='Summary ratio (0.0-1.0)')
    parser.add_argument('--sentences', '-s', type=int, help='Number of sentences')
    parser.add_argument('--words', '-w', type=int, help='Maximum words')
    parser.add_argument('--points', '-p', type=int, help='Extract N key points')
    parser.add_argument('--method', '-m', choices=['textrank', 'lsa', 'frequency'], default='textrank')
    parser.add_argument('--preserve-order', action='store_true', default=True)

    args = parser.parse_args()
    summarizer = TextSummarizer(method=args.method)

    if args.input_dir and args.output_dir:
        # Batch mode
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        results = summarizer.summarize_directory(args.input_dir, ratio=args.ratio or 0.2)

        for filename, summary in results.items():
            output_path = Path(args.output_dir) / filename
            output_path.write_text(summary)
            print(f"Summarized: {filename}")

        print(f"\nSummarized {len(results)} files")

    elif args.input:
        text = Path(args.input).read_text()

        if args.points:
            points = summarizer.extract_key_points(text, args.points)
            output = "\n".join(f"• {p}" for p in points)
        else:
            output = summarizer.summarize(
                text,
                ratio=args.ratio,
                num_sentences=args.sentences,
                max_words=args.words,
                preserve_order=args.preserve_order
            )

        if args.output:
            Path(args.output).write_text(output)
            print(f"Summary saved to: {args.output}")
        else:
            print(output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
