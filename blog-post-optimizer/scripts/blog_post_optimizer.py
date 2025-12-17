#!/usr/bin/env python3
"""
Blog Post Optimizer

Analyze blog posts for SEO, readability, headline quality, and content structure:
- Headline analysis with power words
- SEO optimization recommendations
- Content structure validation
- Readability scoring
- Meta tag generation
- HTML reporting
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
except ImportError:
    nltk = None

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None


@dataclass
class AnalysisResults:
    """Complete analysis results."""
    overall_score: int
    headline: Dict
    seo: Dict
    structure: Dict
    readability: Dict
    stats: Dict
    recommendations: List[Dict]


class BlogPostOptimizer:
    """Analyze and optimize blog post content."""

    def __init__(self):
        """Initialize optimizer."""
        self._load_power_words()
        self._ensure_nltk_data()

    def _load_power_words(self):
        """Load power words database."""
        power_words_path = Path(__file__).parent / 'power_words.json'
        try:
            with open(power_words_path) as f:
                self.power_words_db = json.load(f)
        except FileNotFoundError:
            # Fallback minimal database
            self.power_words_db = {
                'urgency': ['now', 'today', 'fast', 'quick'],
                'value': ['best', 'proven', 'ultimate', 'essential'],
                'curiosity': ['secret', 'discover', 'revealed', 'truth']
            }

    def _ensure_nltk_data(self):
        """Ensure NLTK data is downloaded."""
        if not nltk:
            return

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except:
                pass

    def analyze_headline(self, headline: str) -> Dict:
        """
        Analyze headline effectiveness.

        Args:
            headline: Headline text

        Returns:
            Dictionary with score, power words, suggestions
        """
        score = 0
        power_words_found = []
        char_count = len(headline)

        # Character count score (50-60 is ideal)
        if 50 <= char_count <= 60:
            score += 30
        elif 40 <= char_count <= 70:
            score += 20
        else:
            score += 10

        # Power words detection
        headline_lower = headline.lower()
        for category, words in self.power_words_db.items():
            for word in words:
                if re.search(r'\b' + re.escape(word) + r'\b', headline_lower):
                    power_words_found.append(word)

        power_word_score = min(len(power_words_found) * 15, 40)
        score += power_word_score

        # Numbers in headline (adds specificity)
        if re.search(r'\d+', headline):
            score += 15

        # Question format
        if '?' in headline:
            score += 10

        # Emotional impact (based on power words)
        emotional_impact = min(len(power_words_found) * 20, 100)

        # Generate suggestions
        suggestions = []
        if char_count < 50:
            suggestions.append(f"{headline} - Complete Guide")
        if not power_words_found:
            suggestions.append(f"The Ultimate Guide to {headline}")
        if not re.search(r'\d+', headline):
            suggestions.append(f"10 Essential Tips: {headline}")

        return {
            'score': min(score, 100),
            'character_count': char_count,
            'power_words': list(set(power_words_found)),
            'emotional_impact': emotional_impact,
            'suggestions': suggestions[:3]
        }

    def analyze_seo(self, content: str, keywords: List[str]) -> Dict:
        """
        Analyze SEO elements.

        Args:
            content: Content text
            keywords: List of target keywords

        Returns:
            Dictionary with SEO metrics
        """
        score = 0
        content_lower = content.lower()
        word_count = len(content.split())

        # Calculate keyword density
        keyword_density = {}
        for keyword in keywords:
            keyword_lower = keyword.lower()
            count = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', content_lower))
            density = (count / word_count * 100) if word_count > 0 else 0
            keyword_density[keyword] = round(density, 2)

            # Score based on density (1-2% is ideal)
            if 1.0 <= density <= 2.0:
                score += 30
            elif 0.5 <= density <= 3.0:
                score += 20
            else:
                score += 10

        # Keyword prominence (in first 100 words)
        first_100_words = ' '.join(content_lower.split()[:100])
        keyword_prominence = any(
            re.search(r'\b' + re.escape(kw.lower()) + r'\b', first_100_words)
            for kw in keywords
        )
        if keyword_prominence:
            score += 20

        # Generate meta description (150-160 chars)
        sentences = sent_tokenize(content) if nltk else content.split('. ')
        meta_description = sentences[0] if sentences else ''
        if len(meta_description) > 160:
            meta_description = meta_description[:157] + '...'

        # Generate title tag
        primary_keyword = keywords[0] if keywords else ''
        title_tag = f"{primary_keyword} - Complete Guide"[:60]

        # Generate URL slug
        slug_words = primary_keyword.lower().split()
        url_slug = '-'.join(slug_words[:5])

        return {
            'score': min(score, 100),
            'keyword_density': keyword_density,
            'keyword_prominence': keyword_prominence,
            'meta_description': meta_description,
            'title_tag': title_tag,
            'url_slug': url_slug
        }

    def analyze_structure(self, content: str) -> Dict:
        """
        Analyze content structure.

        Args:
            content: Content text (markdown or plain text)

        Returns:
            Dictionary with structure metrics
        """
        score = 70  # Base score
        warnings = []

        # Count headings (markdown format)
        h1_count = len(re.findall(r'^#\s+', content, re.MULTILINE))
        h2_count = len(re.findall(r'^##\s+', content, re.MULTILINE))
        h3_count = len(re.findall(r'^###\s+', content, re.MULTILINE))

        # H1 should be exactly 1
        if h1_count == 1:
            score += 10
        elif h1_count == 0:
            warnings.append("Missing H1 heading")
            score -= 10
        else:
            warnings.append(f"Multiple H1 headings ({h1_count})")
            score -= 5

        # Count paragraphs and lists
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        sentences = sent_tokenize(content) if nltk else content.split('. ')

        avg_paragraph_length = len(sentences) / len(paragraphs) if paragraphs else 0

        # Ideal paragraph length: 3-5 sentences
        if 3 <= avg_paragraph_length <= 5:
            score += 10
        elif avg_paragraph_length > 8:
            warnings.append(f"Paragraphs too long (avg {avg_paragraph_length:.1f} sentences)")

        # Count lists
        list_count = len(re.findall(r'^\s*[-*+]\s+', content, re.MULTILINE))
        list_count += len(re.findall(r'^\s*\d+\.\s+', content, re.MULTILINE))

        if list_count > 0:
            score += 10

        return {
            'score': min(max(score, 0), 100),
            'h1_count': h1_count,
            'h2_count': h2_count,
            'h3_count': h3_count,
            'avg_paragraph_length': round(avg_paragraph_length, 1),
            'list_count': list_count,
            'warnings': warnings
        }

    def analyze_readability(self, content: str) -> Dict:
        """
        Calculate readability scores.

        Args:
            content: Content text

        Returns:
            Dictionary with readability metrics
        """
        # Clean content
        clean_text = re.sub(r'[#*`\[\]]', '', content)
        clean_text = re.sub(r'\n+', ' ', clean_text)

        sentences = sent_tokenize(clean_text) if nltk else clean_text.split('. ')
        words = clean_text.split()

        sentence_count = len(sentences)
        word_count = len(words)

        # Calculate average sentence length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # Calculate syllables (rough estimate)
        syllable_count = sum(self._count_syllables(word) for word in words)

        # Flesch Reading Ease
        if sentence_count > 0 and word_count > 0:
            reading_ease = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
            reading_ease = max(0, min(100, reading_ease))
        else:
            reading_ease = 0

        # Flesch-Kincaid Grade Level
        if sentence_count > 0 and word_count > 0:
            grade_level = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
            grade_level = max(0, grade_level)
        else:
            grade_level = 0

        # Passive voice detection (simple)
        passive_voice_count = len(re.findall(r'\b(was|were|been|being)\s+\w+ed\b', clean_text, re.IGNORECASE))
        passive_voice_pct = (passive_voice_count / sentence_count * 100) if sentence_count > 0 else 0

        # Complexity score (inverse of reading ease)
        complexity_score = 100 - reading_ease

        return {
            'flesch_kincaid_grade': round(grade_level, 1),
            'reading_ease': round(reading_ease, 1),
            'avg_sentence_length': round(avg_sentence_length, 1),
            'passive_voice_pct': round(passive_voice_pct, 1),
            'complexity_score': round(complexity_score, 0)
        }

    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word."""
        word = word.lower()
        vowels = 'aeiouy'
        count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        # Adjust for silent e
        if word.endswith('e'):
            count -= 1

        return max(1, count)

    def calculate_content_stats(self, content: str) -> Dict:
        """
        Calculate content statistics.

        Args:
            content: Content text

        Returns:
            Dictionary with content stats
        """
        clean_text = re.sub(r'[#*`]', '', content)
        words = clean_text.split()
        sentences = sent_tokenize(clean_text) if nltk else clean_text.split('. ')
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        word_count = len(words)
        reading_time = max(1, word_count // 265)  # 265 words per minute

        return {
            'word_count': word_count,
            'reading_time_minutes': reading_time,
            'character_count': len(content),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs)
        }

    def generate_meta_tags(
        self,
        title: str,
        description: str,
        keywords: List[str]
    ) -> Dict:
        """
        Generate meta tags for social media and SEO.

        Args:
            title: Page title
            description: Meta description
            keywords: List of keywords

        Returns:
            Dictionary with meta tag groups
        """
        return {
            'open_graph': {
                'og:title': title,
                'og:description': description,
                'og:type': 'article',
                'og:locale': 'en_US'
            },
            'twitter_card': {
                'twitter:card': 'summary_large_image',
                'twitter:title': title,
                'twitter:description': description
            },
            'schema_org': {
                '@context': 'https://schema.org',
                '@type': 'Article',
                'headline': title,
                'description': description,
                'keywords': ', '.join(keywords) if keywords else ''
            }
        }

    def analyze_full(
        self,
        content: str,
        headline: str,
        keywords: List[str]
    ) -> Dict:
        """
        Perform complete analysis.

        Args:
            content: Content text
            headline: Headline text
            keywords: List of target keywords

        Returns:
            Complete analysis results
        """
        # Run all analyses
        headline_analysis = self.analyze_headline(headline)
        seo_analysis = self.analyze_seo(content, keywords)
        structure_analysis = self.analyze_structure(content)
        readability_analysis = self.analyze_readability(content)
        stats = self.calculate_content_stats(content)

        # Calculate overall score
        overall_score = (
            headline_analysis['score'] * 0.2 +
            seo_analysis['score'] * 0.3 +
            structure_analysis['score'] * 0.2 +
            (100 - readability_analysis['complexity_score']) * 0.3
        )

        # Generate recommendations
        recommendations = []

        if headline_analysis['score'] < 70:
            recommendations.append({
                'priority': 'high',
                'issue': 'Headline needs improvement',
                'fix': f"Try: {headline_analysis['suggestions'][0]}" if headline_analysis['suggestions'] else 'Add power words and numbers'
            })

        if not seo_analysis['keyword_prominence']:
            recommendations.append({
                'priority': 'high',
                'issue': 'Keywords not in first 100 words',
                'fix': 'Include primary keyword in introduction'
            })

        if structure_analysis['h1_count'] != 1:
            recommendations.append({
                'priority': 'medium',
                'issue': f'H1 count is {structure_analysis["h1_count"]}',
                'fix': 'Use exactly one H1 heading'
            })

        if readability_analysis['flesch_kincaid_grade'] > 12:
            recommendations.append({
                'priority': 'medium',
                'issue': 'Content is too complex',
                'fix': 'Simplify sentences and use shorter words'
            })

        if stats['word_count'] < 300:
            recommendations.append({
                'priority': 'low',
                'issue': 'Content is too short',
                'fix': 'Aim for at least 1000 words for SEO'
            })

        return {
            'overall_score': round(overall_score, 0),
            'headline': headline_analysis,
            'seo': seo_analysis,
            'structure': structure_analysis,
            'readability': readability_analysis,
            'stats': stats,
            'recommendations': recommendations
        }

    def export_html_report(self, results: Dict, output_path: str):
        """
        Export results as HTML report.

        Args:
            results: Analysis results
            output_path: Output file path
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Blog Post Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .score {{
            font-size: 48px;
            font-weight: bold;
            text-align: center;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        .good {{ color: #27ae60; font-weight: bold; }}
        .warning {{ color: #f39c12; font-weight: bold; }}
        .bad {{ color: #e74c3c; font-weight: bold; }}
        .recommendation {{
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #3498db;
            background-color: #ecf0f1;
        }}
        .priority-high {{
            border-left-color: #e74c3c;
        }}
        .priority-medium {{
            border-left-color: #f39c12;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Blog Post Analysis Report</h1>
        <div class="score">{results['overall_score']}/100</div>
    </div>

    <div class="section">
        <h2>Headline Analysis (Score: {results['headline']['score']}/100)</h2>
        <div class="metric">
            <span>Character Count:</span>
            <span>{results['headline']['character_count']} (ideal: 50-60)</span>
        </div>
        <div class="metric">
            <span>Power Words:</span>
            <span>{', '.join(results['headline']['power_words']) if results['headline']['power_words'] else 'None found'}</span>
        </div>
        <div class="metric">
            <span>Emotional Impact:</span>
            <span>{results['headline']['emotional_impact']}/100</span>
        </div>
    </div>

    <div class="section">
        <h2>SEO Analysis (Score: {results['seo']['score']}/100)</h2>
        <div class="metric">
            <span>Keyword Density:</span>
            <span>{', '.join(f"{k}: {v}%" for k, v in results['seo']['keyword_density'].items())}</span>
        </div>
        <div class="metric">
            <span>Keyword Prominence:</span>
            <span class="{'good' if results['seo']['keyword_prominence'] else 'bad'}">
                {'✓ Yes' if results['seo']['keyword_prominence'] else '✗ No'}
            </span>
        </div>
        <div class="metric">
            <span>Meta Description:</span>
            <span>{results['seo']['meta_description']}</span>
        </div>
    </div>

    <div class="section">
        <h2>Content Structure (Score: {results['structure']['score']}/100)</h2>
        <div class="metric">
            <span>H1 Headings:</span>
            <span class="{'good' if results['structure']['h1_count'] == 1 else 'warning'}">
                {results['structure']['h1_count']} (should be 1)
            </span>
        </div>
        <div class="metric">
            <span>H2 Headings:</span>
            <span>{results['structure']['h2_count']}</span>
        </div>
        <div class="metric">
            <span>Avg Paragraph Length:</span>
            <span>{results['structure']['avg_paragraph_length']} sentences</span>
        </div>
        <div class="metric">
            <span>Lists:</span>
            <span>{results['structure']['list_count']}</span>
        </div>
    </div>

    <div class="section">
        <h2>Readability</h2>
        <div class="metric">
            <span>Grade Level:</span>
            <span>{results['readability']['flesch_kincaid_grade']} (target: 8-10)</span>
        </div>
        <div class="metric">
            <span>Reading Ease:</span>
            <span>{results['readability']['reading_ease']} (target: 60-70)</span>
        </div>
        <div class="metric">
            <span>Passive Voice:</span>
            <span class="{'good' if results['readability']['passive_voice_pct'] < 10 else 'warning'}">
                {results['readability']['passive_voice_pct']}% (target: <10%)
            </span>
        </div>
    </div>

    <div class="section">
        <h2>Content Statistics</h2>
        <div class="metric">
            <span>Word Count:</span>
            <span>{results['stats']['word_count']}</span>
        </div>
        <div class="metric">
            <span>Reading Time:</span>
            <span>{results['stats']['reading_time_minutes']} minutes</span>
        </div>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        """

        for rec in results['recommendations']:
            html += f"""
        <div class="recommendation priority-{rec['priority']}">
            <strong>{rec['issue']}</strong><br>
            {rec['fix']}
        </div>
            """

        html += """
    </div>
</body>
</html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze and optimize blog post content'
    )
    parser.add_argument('--input', '-i', help='Input file (txt/md/html)')
    parser.add_argument('--headline', help='Blog post headline')
    parser.add_argument('--keywords', '-k', help='Comma-separated keywords')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--format', '-f', choices=['json', 'html'], default='json')
    parser.add_argument('--headline-only', help='Analyze headline only')

    args = parser.parse_args()

    optimizer = BlogPostOptimizer()

    try:
        if args.headline_only:
            # Headline analysis only
            results = optimizer.analyze_headline(args.headline_only)
            print(json.dumps(results, indent=2))
            return

        if not args.input:
            parser.error('--input required (or use --headline-only)')

        # Read content
        with open(args.input, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract headline if not provided
        headline = args.headline
        if not headline:
            # Try to extract from first H1
            match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            headline = match.group(1) if match else "Untitled"

        # Parse keywords
        keywords = [k.strip() for k in args.keywords.split(',')] if args.keywords else []

        # Full analysis
        results = optimizer.analyze_full(content, headline, keywords)

        # Output
        if args.format == 'html':
            output_path = args.output or 'analysis_report.html'
            optimizer.export_html_report(results, output_path)
            print(f"HTML report saved to {output_path}")
        else:
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Analysis saved to {args.output}")
            else:
                print(json.dumps(results, indent=2))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
