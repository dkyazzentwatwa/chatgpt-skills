#!/usr/bin/env python3
"""
Regex Tester - Test, debug, and explain regular expressions.

Features:
- Pattern testing with detailed results
- Pattern explanation
- Pattern generation from examples
- Find and replace
- Common patterns library
"""

import argparse
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class RegexTester:
    """Test and debug regular expressions."""

    # Common patterns library
    PATTERNS = {
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'phone_us': r'\d{3}[-.]?\d{3}[-.]?\d{4}',
        'url': r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*',
        'ipv4': r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
        'date_iso': r'\d{4}-\d{2}-\d{2}',
        'date_us': r'\d{1,2}/\d{1,2}/\d{4}',
        'time_24h': r'\d{2}:\d{2}(?::\d{2})?',
        'hex_color': r'#[0-9A-Fa-f]{6}',
        'zipcode_us': r'\d{5}(?:-\d{4})?',
        'ssn': r'\d{3}-\d{2}-\d{4}',
        'credit_card': r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',
        'username': r'[a-zA-Z][a-zA-Z0-9_]{2,}',
        'slug': r'[a-z0-9]+(?:-[a-z0-9]+)*',
    }

    # Regex token explanations
    TOKEN_EXPLANATIONS = {
        r'\d': 'digit (0-9)',
        r'\D': 'non-digit',
        r'\w': 'word character (a-z, A-Z, 0-9, _)',
        r'\W': 'non-word character',
        r'\s': 'whitespace',
        r'\S': 'non-whitespace',
        r'\b': 'word boundary',
        r'\B': 'non-word boundary',
        r'^': 'start of string/line',
        r'$': 'end of string/line',
        r'.': 'any character (except newline)',
        r'*': 'zero or more (greedy)',
        r'+': 'one or more (greedy)',
        r'?': 'zero or one (optional)',
        r'*?': 'zero or more (lazy)',
        r'+?': 'one or more (lazy)',
        r'??': 'zero or one (lazy)',
    }

    def __init__(self):
        """Initialize tester."""
        self.patterns = self.PATTERNS.copy()

    def validate(self, pattern: str) -> Tuple[bool, Optional[str]]:
        """
        Validate regex pattern syntax.

        Args:
            pattern: Regex pattern

        Returns:
            (is_valid, error_message)
        """
        try:
            re.compile(pattern)
            return True, None
        except re.error as e:
            return False, str(e)

    def test(
        self,
        pattern: str,
        text: str,
        ignore_case: bool = False,
        multiline: bool = False,
        dotall: bool = False
    ) -> Dict:
        """
        Test pattern against text.

        Args:
            pattern: Regex pattern
            text: Text to test against
            ignore_case: Case insensitive
            multiline: Multiline mode
            dotall: Dot matches newline

        Returns:
            Dict with matches, positions, groups
        """
        # Build flags
        flags = 0
        if ignore_case:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL

        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return {
                'pattern': pattern,
                'text': text,
                'error': str(e),
                'matches': [],
                'match_count': 0,
                'positions': [],
                'groups': []
            }

        matches = []
        positions = []
        groups = []

        for match in compiled.finditer(text):
            matches.append(match.group())
            positions.append((match.start(), match.end()))
            if match.groups():
                groups.append(match.groups())

        return {
            'pattern': pattern,
            'text': text[:200] + '...' if len(text) > 200 else text,
            'matches': matches,
            'match_count': len(matches),
            'positions': positions,
            'groups': groups
        }

    def explain(self, pattern: str) -> str:
        """
        Explain a regex pattern in plain English.

        Args:
            pattern: Regex pattern

        Returns:
            Explanation string
        """
        explanations = []
        i = 0
        in_charset = False
        in_group = 0

        while i < len(pattern):
            char = pattern[i]
            remaining = pattern[i:]

            # Character class
            if char == '[' and not in_charset:
                end = remaining.find(']')
                if end > 0:
                    charset = remaining[:end+1]
                    if charset.startswith('[^'):
                        explanations.append(f"{charset} - any character NOT in {charset[2:-1]}")
                    else:
                        explanations.append(f"{charset} - any character in {charset[1:-1]}")
                    i += end + 1
                    continue

            # Escaped characters
            if char == '\\' and i + 1 < len(pattern):
                two_char = pattern[i:i+2]
                if two_char in self.TOKEN_EXPLANATIONS:
                    explanations.append(f"{two_char} - {self.TOKEN_EXPLANATIONS[two_char]}")
                    i += 2
                    continue
                else:
                    explanations.append(f"{two_char} - literal '{pattern[i+1]}'")
                    i += 2
                    continue

            # Groups
            if char == '(':
                if remaining.startswith('(?:'):
                    explanations.append("(?:...) - non-capturing group")
                    i += 3
                    in_group += 1
                    continue
                elif remaining.startswith('(?='):
                    explanations.append("(?=...) - positive lookahead")
                    i += 3
                    continue
                elif remaining.startswith('(?!'):
                    explanations.append("(?!...) - negative lookahead")
                    i += 3
                    continue
                elif remaining.startswith('(?<='):
                    explanations.append("(?<=...) - positive lookbehind")
                    i += 4
                    continue
                elif remaining.startswith('(?<!'):
                    explanations.append("(?<!...) - negative lookbehind")
                    i += 4
                    continue
                else:
                    in_group += 1
                    explanations.append("( - start capturing group")
                    i += 1
                    continue

            if char == ')':
                explanations.append(") - end group")
                in_group = max(0, in_group - 1)
                i += 1
                continue

            # Quantifiers
            if char == '{':
                end = remaining.find('}')
                if end > 0:
                    quant = remaining[:end+1]
                    if ',' in quant:
                        parts = quant[1:-1].split(',')
                        if parts[1]:
                            explanations.append(f"{quant} - {parts[0]} to {parts[1]} times")
                        else:
                            explanations.append(f"{quant} - {parts[0]} or more times")
                    else:
                        explanations.append(f"{quant} - exactly {quant[1:-1]} times")
                    i += end + 1
                    continue

            # Single character tokens
            if char in self.TOKEN_EXPLANATIONS:
                explanations.append(f"{char} - {self.TOKEN_EXPLANATIONS[char]}")
            elif char == '|':
                explanations.append("| - OR (alternative)")
            elif char.isalnum():
                explanations.append(f"{char} - literal '{char}'")
            else:
                explanations.append(f"{char} - literal '{char}'")

            i += 1

        return '\n'.join(explanations)

    def replace(
        self,
        pattern: str,
        replacement: str,
        text: str,
        count: int = 0,
        ignore_case: bool = False
    ) -> str:
        """
        Find and replace using regex.

        Args:
            pattern: Search pattern
            replacement: Replacement string
            text: Input text
            count: Max replacements (0 = all)
            ignore_case: Case insensitive

        Returns:
            Text with replacements
        """
        flags = re.IGNORECASE if ignore_case else 0
        return re.sub(pattern, replacement, text, count=count, flags=flags)

    def generate_pattern(
        self,
        positive: List[str],
        negative: Optional[List[str]] = None
    ) -> str:
        """
        Generate regex pattern from examples.

        Args:
            positive: Examples that should match
            negative: Examples that should NOT match

        Returns:
            Suggested pattern
        """
        if not positive:
            return ""

        # Analyze common structure
        lengths = [len(s) for s in positive]
        same_length = len(set(lengths)) == 1

        # Check if all digits
        all_digits = all(s.isdigit() for s in positive)
        if all_digits:
            if same_length:
                return f"\\d{{{lengths[0]}}}"
            return r"\d+"

        # Check if all alphanumeric
        all_alnum = all(s.isalnum() for s in positive)
        if all_alnum:
            if same_length:
                return f"[a-zA-Z0-9]{{{lengths[0]}}}"
            return r"[a-zA-Z0-9]+"

        # Check for common patterns with separators
        # e.g., "555-1234", "123-4567"
        sample = positive[0]
        pattern_parts = []
        i = 0
        while i < len(sample):
            char = sample[i]
            if char.isdigit():
                # Count consecutive digits
                j = i
                while j < len(sample) and sample[j].isdigit():
                    j += 1
                digit_len = j - i
                # Check if consistent across all examples
                if all(len(s) > i and s[i:j].isdigit() for s in positive):
                    pattern_parts.append(f"\\d{{{digit_len}}}")
                else:
                    pattern_parts.append(r"\d+")
                i = j
            elif char.isalpha():
                j = i
                while j < len(sample) and sample[j].isalpha():
                    j += 1
                alpha_len = j - i
                pattern_parts.append(f"[a-zA-Z]{{{alpha_len}}}")
                i = j
            else:
                # Literal character (escape if special)
                if char in r'\.^$*+?{}[]|()':
                    pattern_parts.append(f"\\{char}")
                else:
                    pattern_parts.append(char)
                i += 1

        pattern = ''.join(pattern_parts)

        # Validate against examples
        try:
            compiled = re.compile(f"^{pattern}$")
            if all(compiled.match(s) for s in positive):
                if negative is None or not any(compiled.match(s) for s in negative):
                    return pattern
        except re.error:
            pass

        return pattern

    def get_pattern(self, name: str) -> Optional[str]:
        """Get pattern from library by name."""
        return self.patterns.get(name)

    def add_pattern(self, name: str, pattern: str):
        """Add pattern to library."""
        self.patterns[name] = pattern


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Test and debug regular expressions')
    parser.add_argument('--pattern', '-p', help='Regex pattern to test')
    parser.add_argument('--text', '-t', help='Text to test against')
    parser.add_argument('--file', '-f', help='File to test against')
    parser.add_argument('--explain', '-e', help='Explain pattern')
    parser.add_argument('--replace', '-r', help='Replacement string')
    parser.add_argument('--generate', '-g', help='Generate pattern from comma-separated examples')
    parser.add_argument('--ignore-case', '-i', action='store_true')
    parser.add_argument('--multiline', '-m', action='store_true')
    parser.add_argument('--list-patterns', action='store_true', help='List common patterns')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()
    tester = RegexTester()

    if args.list_patterns:
        print("Common Patterns:")
        for name, pattern in tester.patterns.items():
            print(f"  {name}: {pattern}")
        return

    if args.explain:
        explanation = tester.explain(args.explain)
        print(f"Pattern: {args.explain}\n")
        print("Explanation:")
        print(explanation)
        return

    if args.generate:
        examples = [e.strip() for e in args.generate.split(',')]
        pattern = tester.generate_pattern(examples)
        print(f"Examples: {examples}")
        print(f"Suggested pattern: {pattern}")

        # Test it
        for ex in examples:
            matches = bool(re.fullmatch(pattern, ex))
            print(f"  {ex}: {'MATCH' if matches else 'NO MATCH'}")
        return

    if args.pattern:
        # Get text
        if args.file:
            text = Path(args.file).read_text()
        elif args.text:
            text = args.text
        else:
            parser.error("--text or --file required with --pattern")
            return

        # Replace mode
        if args.replace:
            result = tester.replace(
                args.pattern,
                args.replace,
                text,
                ignore_case=args.ignore_case
            )
            print(result)
            return

        # Test mode
        result = tester.test(
            args.pattern,
            text,
            ignore_case=args.ignore_case,
            multiline=args.multiline
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Pattern: {result['pattern']}")
            print(f"Matches: {result['match_count']}")

            if result.get('error'):
                print(f"Error: {result['error']}")
            elif result['matches']:
                print("\nMatches:")
                for i, (match, pos) in enumerate(zip(result['matches'], result['positions'])):
                    print(f"  {i+1}. '{match}' at position {pos[0]}-{pos[1]}")

                if result['groups']:
                    print("\nCapture Groups:")
                    for i, groups in enumerate(result['groups']):
                        print(f"  Match {i+1}: {groups}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
