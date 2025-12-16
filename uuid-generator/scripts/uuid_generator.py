#!/usr/bin/env python3
"""UUID Generator - Generate UUIDs in various formats."""

import argparse
import uuid
import sys


class UUIDGenerator:
    """Generate UUIDs."""

    def generate(self, version: int = 4, namespace: str = None, name: str = None) -> str:
        """Generate UUID."""
        if version == 1:
            return str(uuid.uuid1())
        elif version == 4:
            return str(uuid.uuid4())
        elif version == 5:
            if not namespace or not name:
                raise ValueError("UUID5 requires namespace and name")
            ns = getattr(uuid, f'NAMESPACE_{namespace.upper()}', uuid.NAMESPACE_DNS)
            return str(uuid.uuid5(ns, name))
        else:
            raise ValueError(f"Unsupported UUID version: {version}")

    def generate_bulk(self, count: int, version: int = 4) -> List[str]:
        """Generate multiple UUIDs."""
        return [self.generate(version=version) for _ in range(count)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate UUIDs')
    parser.add_argument('--version', '-v', type=int, default=4, choices=[1, 4, 5], help='UUID version')
    parser.add_argument('--count', '-c', type=int, default=1, help='Number of UUIDs to generate')
    parser.add_argument('--namespace', choices=['dns', 'url', 'oid', 'x500'], help='Namespace for UUID5')
    parser.add_argument('--name', '-n', help='Name for UUID5')
    parser.add_argument('--output', '-o', help='Output file')

    args = parser.parse_args()

    generator = UUIDGenerator()

    if args.count == 1:
        uid = generator.generate(version=args.version, namespace=args.namespace, name=args.name)
        print(uid)
    else:
        uuids = generator.generate_bulk(count=args.count, version=args.version)
        if args.output:
            with open(args.output, 'w') as f:
                f.write('\n'.join(uuids))
            print(f"✓ Generated {args.count} UUIDs to {args.output}")
        else:
            for uid in uuids:
                print(uid)
