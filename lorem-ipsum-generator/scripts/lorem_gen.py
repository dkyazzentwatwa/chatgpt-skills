#!/usr/bin/env python3
"""
Lorem Ipsum Generator - Generate placeholder text for mockups.
"""

import argparse
import random
import re
from typing import List, Optional
from datetime import datetime, timedelta

try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False


class LoremGenerator:
    """Generate lorem ipsum and other placeholder text."""

    CLASSIC_WORDS = [
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
        "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
        "et", "dolore", "magna", "aliqua", "enim", "ad", "minim", "veniam",
        "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi",
        "aliquip", "ex", "ea", "commodo", "consequat", "duis", "aute", "irure",
        "in", "reprehenderit", "voluptate", "velit", "esse", "cillum", "fugiat",
        "nulla", "pariatur", "excepteur", "sint", "occaecat", "cupidatat", "non",
        "proident", "sunt", "culpa", "qui", "officia", "deserunt", "mollit",
        "anim", "id", "est", "laborum", "perspiciatis", "unde", "omnis", "iste",
        "natus", "error", "voluptatem", "accusantium", "doloremque", "laudantium",
        "totam", "rem", "aperiam", "eaque", "ipsa", "quae", "ab", "illo",
        "inventore", "veritatis", "quasi", "architecto", "beatae", "vitae",
        "dicta", "explicabo", "nemo", "ipsam", "quia", "voluptas", "aspernatur",
        "aut", "odit", "fugit", "consequuntur", "magni", "dolores", "eos",
        "ratione", "sequi", "nesciunt", "neque", "porro", "quisquam"
    ]

    HIPSTER_WORDS = [
        "artisan", "sustainable", "organic", "craft", "ethical", "vintage",
        "raw", "denim", "cold-pressed", "pour-over", "vinyl", "aesthetic",
        "microdosing", "polaroid", "typewriter", "Brooklyn", "kombucha",
        "avocado", "toast", "flexitarian", "vegan", "gluten-free", "kale",
        "quinoa", "activated", "charcoal", "matcha", "turmeric", "adaptogenic",
        "wellness", "mindful", "curated", "bespoke", "handcrafted", "locally",
        "sourced", "farm-to-table", "slow", "food", "sourdough", "fermented",
        "probiotic", "meditation", "yoga", "breathwork", "intentional", "minimal",
        "hygge", "wabi-sabi", "kinfolk", "wanderlust", "authentic", "genuine"
    ]

    CORPORATE_WORDS = [
        "leverage", "synergy", "paradigm", "ecosystem", "bandwidth", "scalable",
        "agile", "disruptive", "innovative", "strategic", "robust", "seamless",
        "holistic", "proactive", "stakeholder", "alignment", "deliverables",
        "milestone", "optimize", "streamline", "empower", "incentivize",
        "monetize", "pivot", "ideate", "iterate", "sprint", "roadmap",
        "actionable", "insights", "analytics", "metrics", "KPIs", "ROI",
        "value-add", "best-practice", "core-competency", "thought-leadership",
        "mission-critical", "customer-centric", "data-driven", "results-oriented",
        "team-player", "win-win", "low-hanging-fruit", "move-the-needle",
        "circle-back", "deep-dive", "touch-base", "bandwidth", "vertical"
    ]

    TECH_WORDS = [
        "API", "microservices", "kubernetes", "docker", "serverless", "cloud",
        "DevOps", "CI/CD", "pipeline", "deployment", "infrastructure", "scalable",
        "distributed", "async", "webhook", "endpoint", "middleware", "backend",
        "frontend", "fullstack", "framework", "library", "dependency", "package",
        "repository", "commit", "branch", "merge", "pull-request", "code-review",
        "refactor", "optimize", "debug", "compile", "runtime", "exception",
        "database", "query", "index", "cache", "Redis", "MongoDB", "PostgreSQL",
        "GraphQL", "REST", "JSON", "authentication", "authorization", "OAuth",
        "encryption", "SSL", "HTTPS", "latency", "throughput", "monitoring"
    ]

    FIRST_NAMES = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
        "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan",
        "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Emma",
        "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia"
    ]

    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
    ]

    def __init__(self, style: str = "classic"):
        """
        Initialize the generator.

        Args:
            style: Text style ("classic", "hipster", "corporate", "tech")
        """
        self.style = style
        self._set_word_list()

        if FAKER_AVAILABLE:
            self.faker = Faker()
        else:
            self.faker = None

    def _set_word_list(self):
        """Set word list based on style."""
        style_map = {
            "classic": self.CLASSIC_WORDS,
            "hipster": self.HIPSTER_WORDS,
            "corporate": self.CORPORATE_WORDS,
            "tech": self.TECH_WORDS
        }
        self.words_list = style_map.get(self.style, self.CLASSIC_WORDS)

    def set_style(self, style: str) -> 'LoremGenerator':
        """Change text style."""
        self.style = style
        self._set_word_list()
        return self

    def words(self, count: int = 50) -> str:
        """
        Generate specified number of words.

        Args:
            count: Number of words to generate

        Returns:
            String of words
        """
        selected = []
        for _ in range(count):
            selected.append(random.choice(self.words_list))
        return " ".join(selected)

    def sentences(self, count: int = 5) -> str:
        """
        Generate specified number of sentences.

        Args:
            count: Number of sentences

        Returns:
            String of sentences
        """
        result = []
        for _ in range(count):
            word_count = random.randint(8, 15)
            words = self.words(word_count).split()

            # Capitalize first word
            words[0] = words[0].capitalize()

            # Add punctuation
            sentence = " ".join(words) + "."
            result.append(sentence)

        return " ".join(result)

    def paragraphs(self, count: int = 3, words_per: int = None) -> str:
        """
        Generate specified number of paragraphs.

        Args:
            count: Number of paragraphs
            words_per: Words per paragraph (random if None)

        Returns:
            String of paragraphs
        """
        result = []

        for _ in range(count):
            if words_per:
                # Generate exactly words_per words
                words = self.words(words_per).split()
            else:
                # Random sentence count
                sentence_count = random.randint(4, 8)
                words = self.sentences(sentence_count).split()

            # Make it look like sentences
            paragraph_words = []
            current_sentence = []

            for i, word in enumerate(words):
                current_sentence.append(word)

                # End sentence every 8-15 words
                if len(current_sentence) >= random.randint(8, 15):
                    current_sentence[0] = current_sentence[0].capitalize()
                    paragraph_words.append(" ".join(current_sentence) + ".")
                    current_sentence = []

            # Handle remaining words
            if current_sentence:
                current_sentence[0] = current_sentence[0].capitalize()
                paragraph_words.append(" ".join(current_sentence) + ".")

            result.append(" ".join(paragraph_words))

        return "\n\n".join(result)

    def list_items(self, count: int = 5, ordered: bool = False) -> str:
        """
        Generate a list.

        Args:
            count: Number of items
            ordered: Use numbers instead of bullets

        Returns:
            List as string
        """
        items = []

        for i in range(count):
            word_count = random.randint(4, 10)
            words = self.words(word_count).split()
            words[0] = words[0].capitalize()
            item = " ".join(words)

            if ordered:
                items.append(f"{i + 1}. {item}")
            else:
                items.append(f"- {item}")

        return "\n".join(items)

    def heading(self, level: int = 1) -> str:
        """
        Generate a heading.

        Args:
            level: Heading level (1-6)

        Returns:
            Heading text
        """
        word_count = random.randint(2, 5)
        words = self.words(word_count).split()
        return " ".join(w.capitalize() for w in words)

    def title(self, words: int = 4) -> str:
        """
        Generate a title.

        Args:
            words: Number of words

        Returns:
            Title string
        """
        word_list = self.words(words).split()
        return " ".join(w.capitalize() for w in word_list)

    def html_paragraphs(self, count: int = 3) -> str:
        """
        Generate paragraphs wrapped in HTML tags.

        Args:
            count: Number of paragraphs

        Returns:
            HTML string
        """
        paragraphs = self.paragraphs(count).split("\n\n")
        return "\n".join(f"<p>{p}</p>" for p in paragraphs)

    def html_list(self, count: int = 5, ordered: bool = False) -> str:
        """
        Generate an HTML list.

        Args:
            count: Number of items
            ordered: Ordered list (<ol>) or unordered (<ul>)

        Returns:
            HTML string
        """
        tag = "ol" if ordered else "ul"
        items = []

        for _ in range(count):
            word_count = random.randint(4, 10)
            words = self.words(word_count).split()
            words[0] = words[0].capitalize()
            items.append(f"  <li>{' '.join(words)}</li>")

        return f"<{tag}>\n" + "\n".join(items) + f"\n</{tag}>"

    def html_article(self, sections: int = 3) -> str:
        """
        Generate an HTML article structure.

        Args:
            sections: Number of sections

        Returns:
            HTML string
        """
        lines = ["<article>", f"  <h1>{self.heading(1)}</h1>"]
        lines.append(f"  {self.html_paragraphs(1)}")

        for i in range(sections - 1):
            lines.append(f"  <h2>{self.heading(2)}</h2>")
            lines.append(f"  {self.html_paragraphs(random.randint(1, 2))}")

        lines.append("</article>")
        return "\n".join(lines)

    def name(self) -> str:
        """Generate a random name."""
        if self.faker:
            return self.faker.name()
        return f"{random.choice(self.FIRST_NAMES)} {random.choice(self.LAST_NAMES)}"

    def email(self) -> str:
        """Generate a random email."""
        if self.faker:
            return self.faker.email()
        first = random.choice(self.FIRST_NAMES).lower()
        last = random.choice(self.LAST_NAMES).lower()
        domains = ["email.com", "mail.com", "example.com", "test.com"]
        return f"{first}.{last}@{random.choice(domains)}"

    def date(self) -> str:
        """Generate a random date."""
        if self.faker:
            return self.faker.date()
        days_ago = random.randint(0, 365)
        date = datetime.now() - timedelta(days=days_ago)
        return date.strftime("%Y-%m-%d")

    def fill_template(self, template: str) -> str:
        """
        Fill a template with placeholder text.

        Args:
            template: Template with {{placeholders}}

        Returns:
            Filled template
        """
        result = template

        # Replace {{title}}
        result = re.sub(r'\{\{title\}\}', self.title(), result)

        # Replace {{heading}}
        result = re.sub(r'\{\{heading\}\}', self.heading(), result)

        # Replace {{paragraph}}
        while '{{paragraph}}' in result:
            result = result.replace('{{paragraph}}', self.paragraphs(1), 1)

        # Replace {{sentence}}
        while '{{sentence}}' in result:
            result = result.replace('{{sentence}}', self.sentences(1), 1)

        # Replace {{words:N}}
        words_pattern = re.compile(r'\{\{words:(\d+)\}\}')
        for match in words_pattern.finditer(result):
            count = int(match.group(1))
            result = result.replace(match.group(0), self.words(count), 1)

        # Replace {{list:N}}
        list_pattern = re.compile(r'\{\{list:(\d+)\}\}')
        for match in list_pattern.finditer(result):
            count = int(match.group(1))
            result = result.replace(match.group(0), self.list_items(count), 1)

        # Replace {{name}}
        result = re.sub(r'\{\{name\}\}', lambda m: self.name(), result)

        # Replace {{email}}
        result = re.sub(r'\{\{email\}\}', lambda m: self.email(), result)

        # Replace {{date}}
        result = re.sub(r'\{\{date\}\}', lambda m: self.date(), result)

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Lorem Ipsum Generator - Generate placeholder text"
    )

    parser.add_argument("--paragraphs", "-p", type=int, help="Number of paragraphs")
    parser.add_argument("--sentences", "-s", type=int, help="Number of sentences")
    parser.add_argument("--words", "-w", type=int, help="Number of words")
    parser.add_argument("--words-per", type=int, help="Words per paragraph")
    parser.add_argument("--list", "-l", type=int, help="Number of list items")
    parser.add_argument("--ordered", action="store_true", help="Use ordered list")
    parser.add_argument("--html", action="store_true", help="Output HTML")
    parser.add_argument("--style", choices=["classic", "hipster", "corporate", "tech"],
                       default="classic", help="Text style")
    parser.add_argument("--title", action="store_true", help="Generate a title")
    parser.add_argument("--heading", action="store_true", help="Generate a heading")
    parser.add_argument("--article", type=int, help="Generate HTML article with N sections")
    parser.add_argument("--output", "-o", help="Output file")

    args = parser.parse_args()

    gen = LoremGenerator(style=args.style)
    output = ""

    if args.paragraphs:
        if args.html:
            output = gen.html_paragraphs(args.paragraphs)
        else:
            output = gen.paragraphs(args.paragraphs, words_per=args.words_per)

    elif args.sentences:
        output = gen.sentences(args.sentences)

    elif args.words:
        output = gen.words(args.words)

    elif args.list:
        if args.html:
            output = gen.html_list(args.list, ordered=args.ordered)
        else:
            output = gen.list_items(args.list, ordered=args.ordered)

    elif args.title:
        output = gen.title()

    elif args.heading:
        output = gen.heading()

    elif args.article:
        output = gen.html_article(sections=args.article)

    else:
        # Default: 3 paragraphs
        output = gen.paragraphs(3)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Saved to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
