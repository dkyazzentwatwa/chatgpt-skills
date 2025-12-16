#!/usr/bin/env python3
"""
Survey Analyzer - Analyze survey responses with Likert scales, cross-tabs, and sentiment.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter

import pandas as pd
import numpy as np
from scipy import stats
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class SurveyAnalyzer:
    """Analyze survey data."""

    LIKERT_SCALES = {
        'agreement': {
            1: 'Strongly Disagree',
            2: 'Disagree',
            3: 'Neutral',
            4: 'Agree',
            5: 'Strongly Agree'
        },
        'frequency': {
            1: 'Never',
            2: 'Rarely',
            3: 'Sometimes',
            4: 'Often',
            5: 'Always'
        },
        'satisfaction': {
            1: 'Very Dissatisfied',
            2: 'Dissatisfied',
            3: 'Neutral',
            4: 'Satisfied',
            5: 'Very Satisfied'
        },
        'quality': {
            1: 'Poor',
            2: 'Fair',
            3: 'Good',
            4: 'Very Good',
            5: 'Excellent'
        }
    }

    def __init__(self):
        """Initialize the analyzer."""
        self.data = None
        self.original_data = None
        self.results = []

    def load_csv(self, filepath: str, **kwargs) -> 'SurveyAnalyzer':
        """Load survey data from CSV."""
        self.data = pd.read_csv(filepath, **kwargs)
        self.original_data = self.data.copy()
        return self

    def load_data(self, data: pd.DataFrame) -> 'SurveyAnalyzer':
        """Load survey data from DataFrame."""
        self.data = data.copy()
        self.original_data = self.data.copy()
        return self

    def likert_analysis(self, column: str, scale_type: str = 'agreement',
                       custom_labels: Dict = None) -> Dict:
        """
        Analyze Likert scale question.

        Args:
            column: Column name
            scale_type: 'agreement', 'frequency', 'satisfaction', 'quality', or 'custom'
            custom_labels: Custom labels dict (if scale_type='custom')
        """
        data = self.data[column].dropna()

        if custom_labels:
            labels = custom_labels
        else:
            labels = self.LIKERT_SCALES.get(scale_type, self.LIKERT_SCALES['agreement'])

        # Calculate statistics
        mean_score = data.mean()
        median = data.median()
        mode = data.mode()[0] if len(data.mode()) > 0 else None

        # Distribution
        value_counts = data.value_counts().to_dict()
        total = len(data)
        percentages = {k: (v / total) * 100 for k, v in value_counts.items()}

        # Top-2 box (positive responses)
        top_2 = sum(v for k, v in value_counts.items() if k >= 4) / total * 100 if total > 0 else 0

        # Bottom-2 box (negative responses)
        bottom_2 = sum(v for k, v in value_counts.items() if k <= 2) / total * 100 if total > 0 else 0

        result = {
            'column': column,
            'scale_type': scale_type,
            'labels': labels,
            'mean_score': float(mean_score),
            'median': float(median),
            'mode': float(mode) if mode else None,
            'std_dev': float(data.std()),
            'distribution': value_counts,
            'percentages': percentages,
            'top_2_box': round(top_2, 1),
            'bottom_2_box': round(bottom_2, 1),
            'response_count': total
        }

        self.results.append(result)
        return result

    def frequency_table(self, column: str) -> pd.DataFrame:
        """Generate frequency table for categorical variable."""
        freq = self.data[column].value_counts()
        pct = (freq / len(self.data) * 100).round(1)

        df = pd.DataFrame({
            'Count': freq,
            'Percentage': pct
        })

        return df.sort_values('Count', ascending=False)

    def multiple_choice(self, column: str, delimiter: str = ',') -> pd.DataFrame:
        """
        Analyze multiple-choice question (multiple selections allowed).

        Args:
            column: Column with delimited responses (e.g., "Option A, Option B")
            delimiter: Delimiter separating choices
        """
        # Split and flatten
        all_choices = []
        for val in self.data[column].dropna():
            choices = [c.strip() for c in str(val).split(delimiter)]
            all_choices.extend(choices)

        # Count
        counts = Counter(all_choices)
        total_responses = len(self.data[column].dropna())

        df = pd.DataFrame({
            'Choice': list(counts.keys()),
            'Count': list(counts.values())
        })
        df['Percentage'] = (df['Count'] / total_responses * 100).round(1)

        return df.sort_values('Count', ascending=False)

    def crosstab(self, row_var: str, col_var: str,
                normalize: Optional[str] = None) -> pd.DataFrame:
        """
        Create cross-tabulation between two variables.

        Args:
            row_var: Row variable
            col_var: Column variable
            normalize: None, 'index' (row %), 'columns' (col %), or 'all' (total %)
        """
        ct = pd.crosstab(
            self.data[row_var],
            self.data[col_var],
            normalize=normalize,
            margins=True
        )

        if normalize:
            ct = (ct * 100).round(1)

        return ct

    def chi_square_test(self, row_var: str, col_var: str) -> Dict:
        """Perform chi-square test of independence."""
        ct = pd.crosstab(self.data[row_var], self.data[col_var])

        chi2, p_value, dof, expected = stats.chi2_contingency(ct)

        result = {
            'test': 'Chi-square test of independence',
            'variables': [row_var, col_var],
            'chi2': float(chi2),
            'p_value': float(p_value),
            'dof': int(dof),
            'significant': p_value < 0.05,
            'interpretation': f"There is {'a' if p_value < 0.05 else 'no'} significant "
                            f"relationship between {row_var} and {col_var} (χ²={chi2:.2f}, p={p_value:.4f})"
        }

        self.results.append(result)
        return result

    def sentiment_analysis(self, column: str) -> pd.DataFrame:
        """Analyze sentiment of text responses."""
        def get_sentiment(text):
            if pd.isna(text):
                return None, None
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                sentiment = 'Positive'
            elif polarity < -0.1:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'

            return polarity, sentiment

        results = self.data[column].apply(get_sentiment)
        df = pd.DataFrame(results.tolist(), columns=['polarity', 'sentiment'])
        df['comment'] = self.data[column]

        return df[['comment', 'polarity', 'sentiment']].dropna()

    def sentiment_summary(self, column: str) -> Dict:
        """Get sentiment distribution summary."""
        sentiment_df = self.sentiment_analysis(column)

        total = len(sentiment_df)
        counts = sentiment_df['sentiment'].value_counts().to_dict()

        summary = {
            'positive': round(counts.get('Positive', 0) / total * 100, 1) if total > 0 else 0,
            'neutral': round(counts.get('Neutral', 0) / total * 100, 1) if total > 0 else 0,
            'negative': round(counts.get('Negative', 0) / total * 100, 1) if total > 0 else 0,
            'avg_polarity': round(sentiment_df['polarity'].mean(), 3),
            'total_responses': total
        }

        return summary

    def word_frequency(self, column: str, top_n: int = 20,
                      min_length: int = 3) -> pd.DataFrame:
        """Get most frequent words in text responses."""
        # Combine all text
        all_text = ' '.join(self.data[column].dropna().astype(str))

        # Tokenize and count
        words = all_text.lower().split()

        # Filter
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                    'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were',
                    'it', 'this', 'that', 'i', 'you', 'we', 'they'}

        filtered_words = [w for w in words
                         if len(w) >= min_length and w not in stopwords]

        # Count
        counts = Counter(filtered_words)

        df = pd.DataFrame(counts.most_common(top_n), columns=['Word', 'Frequency'])
        return df

    def nps_score(self, column: str) -> Dict:
        """
        Calculate Net Promoter Score.

        Args:
            column: Column with 0-10 rating
        """
        data = self.data[column].dropna()

        promoters = (data >= 9).sum()
        passives = ((data >= 7) & (data <= 8)).sum()
        detractors = (data <= 6).sum()
        total = len(data)

        promoters_pct = promoters / total * 100 if total > 0 else 0
        detractors_pct = detractors / total * 100 if total > 0 else 0

        nps = promoters_pct - detractors_pct

        return {
            'promoters': round(promoters_pct, 1),
            'passives': round(passives / total * 100, 1) if total > 0 else 0,
            'detractors': round(detractors_pct, 1),
            'nps': round(nps, 1),
            'interpretation': self._interpret_nps(nps)
        }

    def _interpret_nps(self, nps: float) -> str:
        """Interpret NPS score."""
        if nps >= 70:
            return f"Excellent (NPS: {nps})"
        elif nps >= 50:
            return f"Great (NPS: {nps})"
        elif nps >= 30:
            return f"Good (NPS: {nps})"
        elif nps >= 0:
            return f"Needs improvement (NPS: {nps})"
        else:
            return f"Critical (NPS: {nps})"

    def satisfaction_score(self, columns: List[str]) -> Dict:
        """Calculate overall satisfaction from multiple columns."""
        scores = []
        category_scores = {}

        for col in columns:
            col_data = self.data[col].dropna()
            mean_score = col_data.mean()
            category_scores[col] = round(mean_score, 2)
            scores.append(mean_score)

        overall = np.mean(scores)

        # Satisfaction rate (% scoring 4-5 on 5-point scale)
        all_data = pd.concat([self.data[col] for col in columns])
        satisfaction_rate = (all_data >= 4).sum() / len(all_data) * 100 if len(all_data) > 0 else 0

        return {
            'overall_score': round(overall, 2),
            'category_scores': category_scores,
            'satisfaction_rate': round(satisfaction_rate, 1)
        }

    def plot_likert(self, column: str, output: str,
                   scale_type: str = 'agreement') -> str:
        """Create stacked bar chart for Likert scale."""
        result = self.likert_analysis(column, scale_type)
        labels = result['labels']
        percentages = result['percentages']

        # Create stacked bar
        fig, ax = plt.subplots(figsize=(10, 2))

        colors = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
        left = 0

        for value in sorted(labels.keys()):
            pct = percentages.get(value, 0)
            ax.barh([0], [pct], left=left, color=colors[value-1],
                   label=f"{value}: {labels[value]}", edgecolor='white')
            if pct > 5:  # Only show label if segment is wide enough
                ax.text(left + pct/2, 0, f"{pct:.0f}%",
                       ha='center', va='center', fontweight='bold')
            left += pct

        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel('Percentage')
        ax.set_title(f'{column}\n(Mean: {result["mean_score"]:.2f})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()

        return output

    def plot_frequencies(self, column: str, output: str, top_n: int = None) -> str:
        """Plot frequency distribution."""
        freq = self.frequency_table(column)

        if top_n:
            freq = freq.head(top_n)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(freq)), freq['Count'])
        plt.yticks(range(len(freq)), freq.index)
        plt.xlabel('Count')
        plt.title(f'Frequency Distribution: {column}')
        plt.grid(True, alpha=0.3, axis='x')

        # Add percentage labels
        for i, (count, pct) in enumerate(zip(freq['Count'], freq['Percentage'])):
            plt.text(count, i, f' {pct}%', va='center')

        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()

        return output

    def plot_crosstab(self, row_var: str, col_var: str, output: str) -> str:
        """Create heatmap for cross-tabulation."""
        ct = pd.crosstab(self.data[row_var], self.data[col_var])

        plt.figure(figsize=(10, 6))
        sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
        plt.title(f'{row_var} vs {col_var}')
        plt.ylabel(row_var)
        plt.xlabel(col_var)
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()

        return output

    def plot_sentiment(self, column: str, output: str) -> str:
        """Plot sentiment distribution."""
        summary = self.sentiment_summary(column)

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [summary['positive'], summary['neutral'], summary['negative']]
        colors = ['#2ecc71', '#95a5a6', '#e74c3c']

        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90)
        plt.title(f'Sentiment Distribution: {column}\n(Avg Polarity: {summary["avg_polarity"]})')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()

        return output

    def word_cloud(self, column: str, output: str) -> str:
        """Generate word cloud from text responses."""
        all_text = ' '.join(self.data[column].dropna().astype(str))

        wc = WordCloud(width=800, height=400, background_color='white',
                      max_words=100, colormap='viridis').generate(all_text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud: {column}')
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()

        return output

    def summary(self) -> str:
        """Get summary of all analyses."""
        summary = "Survey Analysis Summary\n"
        summary += "=" * 50 + "\n\n"
        summary += f"Total Responses: {len(self.data)}\n\n"

        for result in self.results:
            if 'column' in result:
                summary += f"{result['column']}\n"
                summary += f"  Mean: {result['mean_score']:.2f}\n"
                summary += f"  Top-2 Box: {result['top_2_box']}%\n\n"
            elif 'interpretation' in result:
                summary += f"{result.get('test', 'Analysis')}\n"
                summary += f"  {result['interpretation']}\n\n"

        return summary


def main():
    parser = argparse.ArgumentParser(description="Survey Analyzer")

    parser.add_argument("--data", "-d", required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output file")

    # Analysis type
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--likert", help="Likert scale column")
    group.add_argument("--crosstab", nargs=2, metavar=('ROW', 'COL'),
                      help="Cross-tabulation variables")
    group.add_argument("--sentiment", help="Sentiment analysis column")
    group.add_argument("--frequencies", help="Frequency table column")

    # Options
    parser.add_argument("--scale", default="agreement",
                       choices=['agreement', 'frequency', 'satisfaction', 'quality'],
                       help="Likert scale type")

    args = parser.parse_args()

    analyzer = SurveyAnalyzer()
    analyzer.load_csv(args.data)

    if args.likert:
        result = analyzer.likert_analysis(args.likert, scale_type=args.scale)
        print(f"Mean score: {result['mean_score']:.2f}")
        print(f"Top-2 box: {result['top_2_box']}%")
        analyzer.plot_likert(args.likert, args.output, scale_type=args.scale)
        print(f"Likert chart saved: {args.output}")

    elif args.crosstab:
        row_var, col_var = args.crosstab
        crosstab = analyzer.crosstab(row_var, col_var)
        print(crosstab)

        test = analyzer.chi_square_test(row_var, col_var)
        print(f"\n{test['interpretation']}")

        analyzer.plot_crosstab(row_var, col_var, args.output)
        print(f"Crosstab heatmap saved: {args.output}")

    elif args.sentiment:
        summary = analyzer.sentiment_summary(args.sentiment)
        print(f"Positive: {summary['positive']}%")
        print(f"Neutral: {summary['neutral']}%")
        print(f"Negative: {summary['negative']}%")

        analyzer.plot_sentiment(args.sentiment, args.output)
        print(f"Sentiment plot saved: {args.output}")

    elif args.frequencies:
        freq = analyzer.frequency_table(args.frequencies)
        print(freq)

        analyzer.plot_frequencies(args.frequencies, args.output)
        print(f"Frequency chart saved: {args.output}")


if __name__ == "__main__":
    main()
