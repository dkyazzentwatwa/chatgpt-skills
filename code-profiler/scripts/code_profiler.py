#!/usr/bin/env python3
"""Code Profiler - Analyze Python code performance."""

import argparse
import cProfile
import pstats
import os
import sys
from io import StringIO


class CodeProfiler:
    """Profile Python code performance."""

    def __init__(self):
        self.profiler = cProfile.Profile()
        self.stats = None

    def profile_function(self, func, args=(), kwargs=None):
        """Profile a function call."""
        if kwargs is None:
            kwargs = {}

        self.profiler.enable()
        result = func(*args, **kwargs)
        self.profiler.disable()

        self.stats = pstats.Stats(self.profiler)
        return result

    def profile_script(self, script_path: str):
        """Profile a Python script."""
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        self.profiler.enable()
        with open(script_path) as f:
            code = compile(f.read(), script_path, 'exec')
            exec(code)
        self.profiler.disable()

        self.stats = pstats.Stats(self.profiler)
        return self

    def print_stats(self, top: int = 20, sort_by: str = 'cumulative'):
        """Print profiling statistics."""
        if self.stats is None:
            print("No profiling data available")
            return

        print(f"\nTop {top} functions by {sort_by} time:")
        print("=" * 80)
        self.stats.sort_stats(sort_by).print_stats(top)

    def export_report(self, output: str):
        """Export profiling report."""
        if self.stats is None:
            raise ValueError("No profiling data available")

        stream = StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats()

        with open(output, 'w') as f:
            f.write(stream.getvalue())

        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile Python code')
    parser.add_argument('script', help='Python script to profile')
    parser.add_argument('--top', type=int, default=20, help='Show top N functions')
    parser.add_argument('--sort', default='cumulative', help='Sort by metric')
    parser.add_argument('--output', '-o', help='Export report to file')

    args = parser.parse_args()

    profiler = CodeProfiler()
    profiler.profile_script(args.script)
    profiler.print_stats(top=args.top, sort_by=args.sort)

    if args.output:
        profiler.export_report(args.output)
        print(f"\n✓ Report exported to: {args.output}")
