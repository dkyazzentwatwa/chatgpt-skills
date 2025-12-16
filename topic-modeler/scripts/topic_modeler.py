#!/usr/bin/env python3
"""
Topic Modeler - Extract topics using LDA.
"""

import argparse
import json
from typing import List, Dict

import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric
import nltk
from nltk.corpus import stopwords

# Download stopwords
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')


class TopicModeler:
    """Extract topics from text using LDA."""

    def __init__(self, n_topics: int = 5):
        """Initialize modeler."""
        self.n_topics = n_topics
        self.lda_model = None
        self.dictionary = None
        self.corpus = None
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text: str) -> List[str]:
        """Preprocess text."""
        # Remove punctuation and numbers
        text = strip_punctuation(text)
        text = strip_numeric(text)

        # Tokenize and lowercase
        tokens = text.lower().split()

        # Remove stopwords
        tokens = [w for w in tokens if w not in self.stop_words and len(w) > 3]

        return tokens

    def fit(self, documents: List[str]) -> 'TopicModeler':
        """Fit LDA model on documents."""
        # Preprocess
        processed_docs = [self.preprocess(doc) for doc in documents]

        # Create dictionary
        self.dictionary = corpora.Dictionary(processed_docs)

        # Filter extremes
        self.dictionary.filter_extremes(no_below=2, no_above=0.5)

        # Create corpus
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]

        # Train LDA
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            random_state=42,
            passes=10,
            alpha='auto'
        )

        return self

    def get_topics(self, n_words: int = 10) -> List[Dict]:
        """Get topics with keywords."""
        if not self.lda_model:
            raise ValueError("Model not fitted")

        topics = []

        for topic_id in range(self.n_topics):
            topic_words = self.lda_model.show_topic(topic_id, topn=n_words)

            topics.append({
                'topic_id': topic_id,
                'keywords': [{'word': word, 'weight': float(weight)}
                           for word, weight in topic_words]
            })

        return topics

    def predict(self, document: str) -> List[Dict]:
        """Predict topic distribution for document."""
        if not self.lda_model:
            raise ValueError("Model not fitted")

        # Preprocess
        tokens = self.preprocess(document)
        bow = self.dictionary.doc2bow(tokens)

        # Get topic distribution
        topic_dist = self.lda_model.get_document_topics(bow)

        return [{'topic_id': tid, 'probability': float(prob)}
               for tid, prob in topic_dist]

    def export_json(self, output: str) -> str:
        """Export topics to JSON."""
        topics = self.get_topics()

        with open(output, 'w') as f:
            json.dump(topics, f, indent=2)

        return output


def main():
    parser = argparse.ArgumentParser(description="Topic Modeler")

    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--column", required=True, help="Text column name")
    parser.add_argument("--topics", "-t", type=int, default=5, help="Number of topics")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    documents = df[args.column].dropna().tolist()

    print(f"Analyzing {len(documents)} documents...")

    # Fit model
    modeler = TopicModeler(n_topics=args.topics)
    modeler.fit(documents)

    # Get topics
    topics = modeler.get_topics()

    print(f"\nExtracted {len(topics)} topics:\n")
    for topic in topics:
        print(f"Topic {topic['topic_id']}:")
        keywords = [kw['word'] for kw in topic['keywords'][:5]]
        print(f"  Keywords: {', '.join(keywords)}\n")

    # Export
    modeler.export_json(args.output)
    print(f"Topics saved: {args.output}")


if __name__ == "__main__":
    main()
