#!/usr/bin/env python3
"""
Named Entity Extractor - Extract entities from text using NLP.
"""

import argparse
import json
import re
from typing import Dict, List, Optional
from pathlib import Path
from collections import Counter

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class EntityExtractor:
    """Extract named entities from text."""

    # Regex patterns for common entities
    REGEX_PATTERNS = {
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "PHONE": r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}',
        "URL": r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
        "DATE": r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b',
        "MONEY": r'\$[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|dollars?|euros?|pounds?)',
        "PERCENTAGE": r'\b\d+(?:\.\d+)?%\b'
    }

    def __init__(self, mode: str = "spacy", model: str = "en_core_web_sm"):
        """
        Initialize the entity extractor.

        Args:
            mode: Extraction mode ("spacy" or "regex")
            model: spaCy model name
        """
        self.mode = mode
        self.nlp = None

        if mode == "spacy":
            if not SPACY_AVAILABLE:
                raise ImportError("spaCy is not installed. Install with: pip install spacy")
            try:
                self.nlp = spacy.load(model)
            except OSError:
                print(f"Model {model} not found. Downloading...")
                spacy.cli.download(model)
                self.nlp = spacy.load(model)

    def extract(self, text: str) -> List[Dict]:
        """
        Extract entities from text.

        Args:
            text: Input text

        Returns:
            List of entity dictionaries
        """
        if self.mode == "spacy":
            return self._extract_spacy(text)
        else:
            return self._extract_regex(text)

    def _extract_spacy(self, text: str) -> List[Dict]:
        """Extract entities using spaCy."""
        if self.nlp is None:
            raise RuntimeError("spaCy model not loaded")

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "description": spacy.explain(ent.label_) or ent.label_
            })

        return entities

    def _extract_regex(self, text: str) -> List[Dict]:
        """Extract entities using regex patterns."""
        entities = []

        for entity_type, pattern in self.REGEX_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(),
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end()
                })

        # Sort by position
        entities.sort(key=lambda x: x["start"])
        return entities

    def extract_file(self, filepath: str) -> List[Dict]:
        """
        Extract entities from a file.

        Args:
            filepath: Path to text file

        Returns:
            List of entities
        """
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return self.extract(text)

    def extract_batch(self, folder: str, recursive: bool = False) -> Dict:
        """
        Extract entities from all files in folder.

        Args:
            folder: Folder path
            recursive: Include subdirectories

        Returns:
            Dictionary mapping filenames to results
        """
        supported = {'.txt', '.md', '.html', '.htm'}
        results = {}

        path = Path(folder)
        files = path.rglob("*") if recursive else path.glob("*")

        for file in files:
            if file.suffix.lower() in supported:
                try:
                    entities = self.extract_file(str(file))
                    results[file.name] = {
                        "entities": entities,
                        "summary": self._create_summary(entities)
                    }
                except Exception as e:
                    results[file.name] = {"error": str(e)}

        return results

    def _create_summary(self, entities: List[Dict]) -> Dict:
        """Create summary of extracted entities."""
        by_type = {}
        for ent in entities:
            ent_type = ent["type"]
            if ent_type not in by_type:
                by_type[ent_type] = []
            by_type[ent_type].append(ent["text"])

        unique_texts = set(ent["text"] for ent in entities)

        return {
            "total_entities": len(entities),
            "unique_entities": len(unique_texts),
            "by_type": {k: len(v) for k, v in by_type.items()}
        }

    def filter_entities(self, entities: List[Dict], types: List[str]) -> List[Dict]:
        """
        Filter entities by type.

        Args:
            entities: List of entities
            types: Entity types to keep

        Returns:
            Filtered entity list
        """
        types_upper = [t.upper() for t in types]
        return [e for e in entities if e["type"].upper() in types_upper]

    def get_unique_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Get unique entities (deduplicated).

        Args:
            entities: List of entities

        Returns:
            List of unique entities
        """
        seen = set()
        unique = []

        for ent in entities:
            key = (ent["text"].lower(), ent["type"])
            if key not in seen:
                seen.add(key)
                unique.append(ent)

        return unique

    def group_by_type(self, entities: List[Dict]) -> Dict[str, List[str]]:
        """
        Group entities by type.

        Args:
            entities: List of entities

        Returns:
            Dictionary mapping types to entity texts
        """
        grouped = {}

        for ent in entities:
            ent_type = ent["type"]
            if ent_type not in grouped:
                grouped[ent_type] = []
            if ent["text"] not in grouped[ent_type]:
                grouped[ent_type].append(ent["text"])

        return grouped

    def entity_frequency(self, text: str) -> Dict:
        """
        Get frequency of each entity in text.

        Args:
            text: Input text

        Returns:
            Dictionary with entity frequencies
        """
        entities = self.extract(text)
        freq = {}

        for ent in entities:
            key = ent["text"]
            if key not in freq:
                freq[key] = {"count": 0, "type": ent["type"]}
            freq[key]["count"] += 1

        return freq

    def highlight_text(self, text: str) -> str:
        """
        Generate HTML with highlighted entities.

        Args:
            text: Input text

        Returns:
            HTML string with entity highlighting
        """
        entities = self.extract(text)

        # Sort by position (reverse to process from end)
        entities.sort(key=lambda x: x["start"], reverse=True)

        # Color map for entity types
        colors = {
            "PERSON": "#ff9999",
            "ORG": "#99ff99",
            "GPE": "#9999ff",
            "LOC": "#99ffff",
            "DATE": "#ffff99",
            "TIME": "#ffcc99",
            "MONEY": "#cc99ff",
            "PERCENT": "#99ccff",
            "EMAIL": "#ffcccc",
            "PHONE": "#ccffcc",
            "URL": "#ccccff"
        }

        html_text = text

        for ent in entities:
            color = colors.get(ent["type"], "#dddddd")
            replacement = f'<span style="background-color:{color}" title="{ent["type"]}">{ent["text"]}</span>'
            html_text = html_text[:ent["start"]] + replacement + html_text[ent["end"]:]

        return f'<div style="font-family:sans-serif;line-height:1.6">{html_text}</div>'

    def to_csv(self, results: Dict, output: str) -> str:
        """
        Export results to CSV.

        Args:
            results: Batch extraction results
            output: Output file path

        Returns:
            Output path
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for CSV export")

        rows = []
        for filename, data in results.items():
            if "error" in data:
                continue
            for ent in data.get("entities", []):
                rows.append({
                    "filename": filename,
                    "entity_text": ent["text"],
                    "entity_type": ent["type"],
                    "start": ent.get("start"),
                    "end": ent.get("end")
                })

        df = pd.DataFrame(rows)
        df.to_csv(output, index=False)
        return output

    def to_json(self, entities: List[Dict], output: str) -> str:
        """
        Export entities to JSON.

        Args:
            entities: List of entities
            output: Output file path

        Returns:
            Output path
        """
        with open(output, 'w') as f:
            json.dump(entities, f, indent=2)
        return output


def main():
    parser = argparse.ArgumentParser(
        description="Named Entity Extractor - Extract entities from text"
    )

    parser.add_argument("--text", "-t", help="Text to analyze")
    parser.add_argument("--input", "-i", help="Input file or folder")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--mode", "-m", choices=["spacy", "regex"],
                       default="spacy", help="Extraction mode")
    parser.add_argument("--types", help="Filter by entity types (comma-separated)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--highlight", action="store_true", help="Generate highlighted HTML")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="Process subfolders")

    args = parser.parse_args()

    extractor = EntityExtractor(mode=args.mode)

    if args.text:
        if args.highlight:
            html = extractor.highlight_text(args.text)
            print(html)
        else:
            entities = extractor.extract(args.text)

            if args.types:
                types = [t.strip() for t in args.types.split(',')]
                entities = extractor.filter_entities(entities, types)

            if args.json:
                print(json.dumps(entities, indent=2))
            else:
                print("\n=== Extracted Entities ===")
                grouped = extractor.group_by_type(entities)
                for ent_type, texts in grouped.items():
                    print(f"\n{ent_type}:")
                    for text in texts:
                        print(f"  - {text}")

    elif args.input:
        input_path = Path(args.input)

        if input_path.is_dir():
            results = extractor.extract_batch(args.input, recursive=args.recursive)

            if args.output and args.output.endswith('.csv'):
                extractor.to_csv(results, args.output)
                print(f"Saved to: {args.output}")

            elif args.json:
                print(json.dumps(results, indent=2))

            else:
                print(f"\n=== Batch Extraction Results ===")
                for filename, data in results.items():
                    if "error" in data:
                        print(f"\n{filename}: Error - {data['error']}")
                    else:
                        summary = data.get("summary", {})
                        print(f"\n{filename}:")
                        print(f"  Total entities: {summary.get('total_entities', 0)}")
                        print(f"  Unique entities: {summary.get('unique_entities', 0)}")
                        if summary.get("by_type"):
                            print("  By type:", dict(summary["by_type"]))

        else:
            entities = extractor.extract_file(args.input)

            if args.types:
                types = [t.strip() for t in args.types.split(',')]
                entities = extractor.filter_entities(entities, types)

            if args.highlight:
                text = open(args.input, 'r').read()
                html = extractor.highlight_text(text)
                if args.output:
                    with open(args.output, 'w') as f:
                        f.write(html)
                    print(f"Highlighted HTML saved to: {args.output}")
                else:
                    print(html)

            elif args.json:
                print(json.dumps(entities, indent=2))

            elif args.output:
                extractor.to_json(entities, args.output)
                print(f"Saved to: {args.output}")

            else:
                print(f"\n=== Entities in {args.input} ===")
                grouped = extractor.group_by_type(entities)
                for ent_type, texts in grouped.items():
                    print(f"\n{ent_type} ({len(texts)}):")
                    for text in texts[:10]:
                        print(f"  - {text}")
                    if len(texts) > 10:
                        print(f"  ... and {len(texts) - 10} more")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
