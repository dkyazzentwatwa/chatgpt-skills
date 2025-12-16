#!/usr/bin/env python3
"""
Password Generator - Generate secure passwords and passphrases.

Features:
- Cryptographically secure random generation
- Custom character rules
- Memorable passphrases
- Strength checking
- Bulk generation
"""

import argparse
import secrets
import string
import math
from typing import List, Optional, Dict


class PasswordGenerator:
    """Generate secure passwords and passphrases."""

    # Character sets
    UPPERCASE = string.ascii_uppercase
    LOWERCASE = string.ascii_lowercase
    DIGITS = string.digits
    SYMBOLS = "!@#$%^&*()_+-=[]{}|;:,.<>?"

    # Ambiguous characters
    AMBIGUOUS = "0O1lI"

    # Common word list for passphrases (small embedded list)
    WORDLIST = [
        "apple", "banana", "cherry", "dragon", "eagle", "falcon", "garden",
        "harbor", "island", "jungle", "kitten", "lemon", "marble", "needle",
        "orange", "panda", "quartz", "rabbit", "silver", "tiger", "umbrella",
        "violet", "window", "yellow", "zebra", "anchor", "bridge", "castle",
        "desert", "forest", "glacier", "helmet", "jacket", "lantern", "magnet",
        "nature", "ocean", "planet", "puzzle", "rocket", "stream", "temple",
        "voyage", "winter", "summit", "breeze", "canyon", "meadow", "valley",
        "thunder", "crystal", "phoenix", "diamond", "emerald", "golden", "copper",
        "bronze", "granite", "coral", "ember", "frost", "spark", "shadow",
        "sunset", "horizon", "morning", "evening", "midnight", "daylight"
    ]

    def __init__(self):
        """Initialize generator."""
        pass

    def generate(
        self,
        length: int = 16,
        uppercase: bool = True,
        lowercase: bool = True,
        digits: bool = True,
        symbols: bool = True,
        exclude_ambiguous: bool = False,
        charset: Optional[str] = None,
        exclude: str = "",
        min_uppercase: int = 0,
        min_lowercase: int = 0,
        min_digits: int = 0,
        min_symbols: int = 0
    ) -> str:
        """
        Generate a secure password.

        Args:
            length: Password length
            uppercase: Include uppercase letters
            lowercase: Include lowercase letters
            digits: Include digits
            symbols: Include symbols
            exclude_ambiguous: Exclude 0, O, 1, l, I
            charset: Custom character set (overrides other options)
            exclude: Characters to exclude
            min_uppercase: Minimum uppercase characters
            min_lowercase: Minimum lowercase characters
            min_digits: Minimum digits
            min_symbols: Minimum symbols

        Returns:
            Generated password
        """
        # Build character set
        if charset:
            chars = charset
        else:
            chars = ""
            if uppercase:
                chars += self.UPPERCASE
            if lowercase:
                chars += self.LOWERCASE
            if digits:
                chars += self.DIGITS
            if symbols:
                chars += self.SYMBOLS

        # Remove excluded characters
        if exclude_ambiguous:
            chars = ''.join(c for c in chars if c not in self.AMBIGUOUS)
        if exclude:
            chars = ''.join(c for c in chars if c not in exclude)

        if not chars:
            raise ValueError("No characters available for password generation")

        # Generate with requirements
        password = []
        requirements = []

        # Add required characters first
        if min_uppercase > 0 and uppercase:
            req_chars = ''.join(c for c in self.UPPERCASE if c in chars)
            for _ in range(min_uppercase):
                if req_chars:
                    requirements.append(secrets.choice(req_chars))

        if min_lowercase > 0 and lowercase:
            req_chars = ''.join(c for c in self.LOWERCASE if c in chars)
            for _ in range(min_lowercase):
                if req_chars:
                    requirements.append(secrets.choice(req_chars))

        if min_digits > 0 and digits:
            req_chars = ''.join(c for c in self.DIGITS if c in chars)
            for _ in range(min_digits):
                if req_chars:
                    requirements.append(secrets.choice(req_chars))

        if min_symbols > 0 and symbols:
            req_chars = ''.join(c for c in self.SYMBOLS if c in chars)
            for _ in range(min_symbols):
                if req_chars:
                    requirements.append(secrets.choice(req_chars))

        # Fill remaining length
        remaining = length - len(requirements)
        for _ in range(remaining):
            password.append(secrets.choice(chars))

        # Add requirements and shuffle
        password.extend(requirements)

        # Secure shuffle
        result = list(password)
        for i in range(len(result) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            result[i], result[j] = result[j], result[i]

        return ''.join(result[:length])

    def passphrase(
        self,
        words: int = 4,
        separator: str = "-",
        capitalize: bool = False,
        include_number: bool = False,
        wordlist: Optional[List[str]] = None
    ) -> str:
        """
        Generate a memorable passphrase.

        Args:
            words: Number of words
            separator: Word separator
            capitalize: Capitalize each word
            include_number: Include a random number
            wordlist: Custom word list

        Returns:
            Generated passphrase
        """
        word_source = wordlist or self.WORDLIST

        # Select random words
        selected = [secrets.choice(word_source) for _ in range(words)]

        # Capitalize if requested
        if capitalize:
            selected = [w.capitalize() for w in selected]

        # Include number if requested
        if include_number:
            num = str(secrets.randbelow(100))
            pos = secrets.randbelow(len(selected) + 1)
            selected.insert(pos, num)

        return separator.join(selected)

    def check_strength(self, password: str) -> Dict:
        """
        Check password strength.

        Args:
            password: Password to check

        Returns:
            Dict with score, label, entropy, feedback
        """
        # Character set analysis
        has_upper = any(c in self.UPPERCASE for c in password)
        has_lower = any(c in self.LOWERCASE for c in password)
        has_digit = any(c in self.DIGITS for c in password)
        has_symbol = any(c in self.SYMBOLS for c in password)

        # Calculate charset size
        charset_size = 0
        if has_upper:
            charset_size += 26
        if has_lower:
            charset_size += 26
        if has_digit:
            charset_size += 10
        if has_symbol:
            charset_size += len(self.SYMBOLS)

        # Calculate entropy
        if charset_size > 0:
            entropy = len(password) * math.log2(charset_size)
        else:
            entropy = 0

        # Generate feedback
        feedback = []

        if len(password) >= 12:
            feedback.append("Good length")
        elif len(password) >= 8:
            feedback.append("Acceptable length")
        else:
            feedback.append("Too short - use 12+ characters")

        if has_upper and has_lower:
            feedback.append("Has mixed case")
        else:
            feedback.append("Add mixed case")

        if has_digit:
            feedback.append("Has numbers")

        if has_symbol:
            feedback.append("Has symbols")
        else:
            feedback.append("Add symbols for more security")

        # Check for common patterns
        if password.lower() in ['password', '123456', 'qwerty', 'admin']:
            feedback.append("WARNING: Common password")
            entropy = 0

        # Score (0-4)
        if entropy < 28:
            score = 0
            label = "Very Weak"
        elif entropy < 36:
            score = 1
            label = "Weak"
        elif entropy < 60:
            score = 2
            label = "Fair"
        elif entropy < 128:
            score = 3
            label = "Strong"
        else:
            score = 4
            label = "Very Strong"

        return {
            'score': score,
            'label': label,
            'entropy': round(entropy, 1),
            'feedback': feedback
        }

    def generate_bulk(self, count: int = 10, **kwargs) -> List[str]:
        """Generate multiple passwords."""
        return [self.generate(**kwargs) for _ in range(count)]

    def generate_to_csv(
        self,
        filepath: str,
        count: int = 100,
        **kwargs
    ) -> str:
        """Generate passwords to CSV file."""
        passwords = self.generate_bulk(count, **kwargs)

        with open(filepath, 'w') as f:
            f.write("password,strength_score,entropy\n")
            for pwd in passwords:
                strength = self.check_strength(pwd)
                f.write(f'"{pwd}",{strength["score"]},{strength["entropy"]}\n')

        return filepath


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Generate secure passwords')
    parser.add_argument('--length', '-l', type=int, default=16, help='Password length')
    parser.add_argument('--count', '-c', type=int, default=1, help='Number to generate')
    parser.add_argument('--passphrase', '-p', action='store_true', help='Generate passphrase')
    parser.add_argument('--words', '-w', type=int, default=4, help='Words in passphrase')
    parser.add_argument('--no-uppercase', action='store_true')
    parser.add_argument('--no-lowercase', action='store_true')
    parser.add_argument('--no-digits', action='store_true')
    parser.add_argument('--no-symbols', action='store_true')
    parser.add_argument('--exclude-ambiguous', action='store_true')
    parser.add_argument('--check', help='Check password strength')
    parser.add_argument('--output', '-o', help='Output file')

    args = parser.parse_args()
    gen = PasswordGenerator()

    if args.check:
        strength = gen.check_strength(args.check)
        print(f"\nPassword Strength: {strength['label']}")
        print(f"Score: {strength['score']}/4")
        print(f"Entropy: {strength['entropy']} bits")
        print("\nFeedback:")
        for item in strength['feedback']:
            print(f"  - {item}")
        return

    if args.passphrase:
        for _ in range(args.count):
            phrase = gen.passphrase(words=args.words)
            print(phrase)
    else:
        passwords = gen.generate_bulk(
            count=args.count,
            length=args.length,
            uppercase=not args.no_uppercase,
            lowercase=not args.no_lowercase,
            digits=not args.no_digits,
            symbols=not args.no_symbols,
            exclude_ambiguous=args.exclude_ambiguous
        )

        if args.output:
            gen.generate_to_csv(args.output, count=args.count, length=args.length)
            print(f"Saved {args.count} passwords to: {args.output}")
        else:
            for pwd in passwords:
                strength = gen.check_strength(pwd)
                print(f"{pwd}  ({strength['label']}, {strength['entropy']} bits)")


if __name__ == "__main__":
    main()
