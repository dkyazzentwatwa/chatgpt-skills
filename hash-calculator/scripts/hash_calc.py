#!/usr/bin/env python3
"""
Hash Calculator - Calculate cryptographic hashes for text and files.

Features:
- Multiple algorithms (MD5, SHA1, SHA256, SHA512, BLAKE2)
- File hashing with streaming
- Hash verification
- Checksum file generation/verification
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Union


class HashCalculator:
    """Calculate cryptographic hashes."""

    ALGORITHMS = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        'sha256': hashlib.sha256,
        'sha384': hashlib.sha384,
        'sha512': hashlib.sha512,
        'blake2b': hashlib.blake2b,
        'blake2s': hashlib.blake2s,
    }

    def __init__(self):
        """Initialize calculator."""
        pass

    def hash_text(
        self,
        text: str,
        algorithm: Optional[str] = None,
        algorithms: Optional[List[str]] = None
    ) -> Union[str, Dict[str, str]]:
        """
        Hash text string.

        Args:
            text: Text to hash
            algorithm: Single algorithm to use
            algorithms: Multiple algorithms

        Returns:
            Hash string or dict of hashes
        """
        data = text.encode('utf-8')

        if algorithm:
            if algorithm not in self.ALGORITHMS:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            return self.ALGORITHMS[algorithm](data).hexdigest()

        if algorithms:
            return {alg: self.ALGORITHMS[alg](data).hexdigest()
                    for alg in algorithms if alg in self.ALGORITHMS}

        # Return all
        return {alg: func(data).hexdigest()
                for alg, func in self.ALGORITHMS.items()}

    def hash_file(
        self,
        filepath: str,
        algorithm: Optional[str] = None,
        algorithms: Optional[List[str]] = None,
        chunk_size: int = 65536
    ) -> Union[str, Dict[str, str]]:
        """
        Hash file with streaming (memory efficient for large files).

        Args:
            filepath: Path to file
            algorithm: Single algorithm
            algorithms: Multiple algorithms
            chunk_size: Read chunk size

        Returns:
            Hash string or dict of hashes
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Determine algorithms to use
        if algorithm:
            algs_to_use = [algorithm]
        elif algorithms:
            algs_to_use = algorithms
        else:
            algs_to_use = list(self.ALGORITHMS.keys())

        # Initialize hashers
        hashers = {alg: self.ALGORITHMS[alg]() for alg in algs_to_use
                   if alg in self.ALGORITHMS}

        # Stream file
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                for hasher in hashers.values():
                    hasher.update(chunk)

        results = {alg: hasher.hexdigest() for alg, hasher in hashers.items()}

        if algorithm:
            return results.get(algorithm, "")

        return results

    def verify_file(
        self,
        filepath: str,
        expected_hash: str,
        algorithm: str = "sha256"
    ) -> bool:
        """
        Verify file against expected hash.

        Args:
            filepath: Path to file
            expected_hash: Expected hash value
            algorithm: Hash algorithm

        Returns:
            True if hash matches
        """
        actual = self.hash_file(filepath, algorithm=algorithm)
        return actual.lower() == expected_hash.lower()

    def verify_text(
        self,
        text: str,
        expected_hash: str,
        algorithm: str = "sha256"
    ) -> bool:
        """Verify text against expected hash."""
        actual = self.hash_text(text, algorithm=algorithm)
        return actual.lower() == expected_hash.lower()

    def hash_directory(
        self,
        directory: str,
        algorithm: str = "sha256",
        recursive: bool = False,
        pattern: str = "*"
    ) -> Dict[str, str]:
        """
        Hash all files in directory.

        Args:
            directory: Directory path
            algorithm: Hash algorithm
            recursive: Include subdirectories
            pattern: File pattern

        Returns:
            Dict of {filename: hash}
        """
        path = Path(directory)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        results = {}
        glob_func = path.rglob if recursive else path.glob

        for filepath in glob_func(pattern):
            if filepath.is_file():
                rel_path = filepath.relative_to(path)
                results[str(rel_path)] = self.hash_file(str(filepath), algorithm=algorithm)

        return results

    def generate_checksums(
        self,
        directory: str,
        output: str,
        algorithm: str = "sha256",
        recursive: bool = False
    ) -> str:
        """
        Generate checksum file for directory.

        Args:
            directory: Directory to hash
            output: Output checksum file
            algorithm: Hash algorithm
            recursive: Include subdirectories

        Returns:
            Output file path
        """
        hashes = self.hash_directory(directory, algorithm=algorithm, recursive=recursive)

        with open(output, 'w') as f:
            for filename, hash_val in sorted(hashes.items()):
                f.write(f"{hash_val}  {filename}\n")

        return output

    def verify_checksums(self, checksum_file: str, base_dir: Optional[str] = None) -> Dict[str, bool]:
        """
        Verify files against checksum file.

        Args:
            checksum_file: Path to checksum file
            base_dir: Base directory for relative paths

        Returns:
            Dict of {filename: is_valid}
        """
        path = Path(checksum_file)
        base = Path(base_dir) if base_dir else path.parent

        # Detect algorithm from filename
        algorithm = "sha256"  # default
        for alg in self.ALGORITHMS:
            if alg in path.name.lower():
                algorithm = alg
                break

        results = {}
        with open(checksum_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Parse "hash  filename" or "hash *filename"
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue

                expected_hash = parts[0]
                filename = parts[1].lstrip('*').strip()
                filepath = base / filename

                if filepath.exists():
                    is_valid = self.verify_file(str(filepath), expected_hash, algorithm)
                else:
                    is_valid = False

                results[filename] = is_valid

        return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Calculate cryptographic hashes')
    parser.add_argument('--text', '-t', help='Text to hash')
    parser.add_argument('--file', '-f', help='File to hash')
    parser.add_argument('--directory', '-d', help='Directory to hash')
    parser.add_argument('--algorithm', '-a', default='sha256',
                        choices=list(HashCalculator.ALGORITHMS.keys()),
                        help='Hash algorithm')
    parser.add_argument('--all', action='store_true', help='Use all algorithms')
    parser.add_argument('--verify', '-v', help='Expected hash to verify against')
    parser.add_argument('--output', '-o', help='Output file for checksums')
    parser.add_argument('--verify-checksums', help='Verify against checksum file')
    parser.add_argument('--recursive', '-r', action='store_true', help='Recursive directory')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()
    calc = HashCalculator()

    if args.verify_checksums:
        results = calc.verify_checksums(args.verify_checksums)
        all_valid = all(results.values())

        for filename, is_valid in results.items():
            status = "OK" if is_valid else "FAILED"
            print(f"{filename}: {status}")

        print(f"\n{'All files verified' if all_valid else 'Some files FAILED'}")
        exit(0 if all_valid else 1)

    elif args.text:
        if args.verify:
            is_valid = calc.verify_text(args.text, args.verify, args.algorithm)
            print("VALID" if is_valid else "INVALID")
            exit(0 if is_valid else 1)

        result = calc.hash_text(args.text, algorithm=None if args.all else args.algorithm)

        if args.json:
            print(json.dumps({'input': args.text, **result} if isinstance(result, dict)
                             else {'input': args.text, args.algorithm: result}, indent=2))
        elif isinstance(result, dict):
            for alg, hash_val in result.items():
                print(f"{alg.upper():8}: {hash_val}")
        else:
            print(result)

    elif args.file:
        if args.verify:
            is_valid = calc.verify_file(args.file, args.verify, args.algorithm)
            print("VALID" if is_valid else "INVALID")
            exit(0 if is_valid else 1)

        result = calc.hash_file(args.file, algorithm=None if args.all else args.algorithm)

        if args.json:
            print(json.dumps({'file': args.file, **result} if isinstance(result, dict)
                             else {'file': args.file, args.algorithm: result}, indent=2))
        elif isinstance(result, dict):
            print(f"File: {args.file}")
            for alg, hash_val in result.items():
                print(f"{alg.upper():8}: {hash_val}")
        else:
            print(result)

    elif args.directory:
        if args.output:
            calc.generate_checksums(args.directory, args.output, args.algorithm, args.recursive)
            print(f"Checksums saved to: {args.output}")
        else:
            results = calc.hash_directory(args.directory, args.algorithm, args.recursive)
            for filename, hash_val in sorted(results.items()):
                print(f"{hash_val}  {filename}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
