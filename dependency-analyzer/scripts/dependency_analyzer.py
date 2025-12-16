#!/usr/bin/env python3
"""
Dependency Analyzer - Analyze Python imports and dependencies.

Features:
- Extract imports from Python files
- Classify as stdlib/third-party/local
- Generate requirements.txt
- Find unused imports
- Detect circular imports
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class DependencyAnalyzer:
    """Analyze Python file imports and dependencies."""

    # Standard library modules (Python 3.8+)
    STDLIB_MODULES = {
        'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio',
        'asyncore', 'atexit', 'audioop', 'base64', 'bdb', 'binascii',
        'binhex', 'bisect', 'builtins', 'bz2', 'calendar', 'cgi', 'cgitb',
        'chunk', 'cmath', 'cmd', 'code', 'codecs', 'codeop', 'collections',
        'colorsys', 'compileall', 'concurrent', 'configparser', 'contextlib',
        'contextvars', 'copy', 'copyreg', 'cProfile', 'crypt', 'csv',
        'ctypes', 'curses', 'dataclasses', 'datetime', 'dbm', 'decimal',
        'difflib', 'dis', 'distutils', 'doctest', 'email', 'encodings',
        'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput',
        'fnmatch', 'fractions', 'ftplib', 'functools', 'gc', 'getopt',
        'getpass', 'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib',
        'heapq', 'hmac', 'html', 'http', 'idlelib', 'imaplib', 'imghdr',
        'imp', 'importlib', 'inspect', 'io', 'ipaddress', 'itertools',
        'json', 'keyword', 'lib2to3', 'linecache', 'locale', 'logging',
        'lzma', 'mailbox', 'mailcap', 'marshal', 'math', 'mimetypes',
        'mmap', 'modulefinder', 'multiprocessing', 'netrc', 'nis',
        'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev',
        'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil',
        'platform', 'plistlib', 'poplib', 'posix', 'posixpath', 'pprint',
        'profile', 'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr',
        'pydoc', 'queue', 'quopri', 'random', 're', 'readline', 'reprlib',
        'resource', 'rlcompleter', 'runpy', 'sched', 'secrets', 'select',
        'selectors', 'shelve', 'shlex', 'shutil', 'signal', 'site',
        'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'spwd',
        'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep',
        'struct', 'subprocess', 'sunau', 'symtable', 'sys', 'sysconfig',
        'syslog', 'tabnanny', 'tarfile', 'telnetlib', 'tempfile', 'termios',
        'test', 'textwrap', 'threading', 'time', 'timeit', 'tkinter',
        'token', 'tokenize', 'trace', 'traceback', 'tracemalloc', 'tty',
        'turtle', 'turtledemo', 'types', 'typing', 'unicodedata', 'unittest',
        'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref',
        'webbrowser', 'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml',
        'xmlrpc', 'zipapp', 'zipfile', 'zipimport', 'zlib', 'zoneinfo',
        '_thread', '__future__', 'typing_extensions'
    }

    def __init__(self):
        """Initialize analyzer."""
        pass

    def analyze_file(self, filepath: str) -> Dict:
        """
        Analyze imports in a single Python file.

        Args:
            filepath: Path to Python file

        Returns:
            Dict with imports categorized
        """
        with open(filepath, 'r') as f:
            source = f.read()

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {'error': f"Syntax error: {e}"}

        imports = []
        from_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    imports.append({
                        'module': alias.name,
                        'alias': alias.asname,
                        'type': self._classify_module(module),
                        'line': node.lineno
                    })

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    from_imports.append({
                        'module': node.module,
                        'names': [alias.name for alias in node.names],
                        'type': self._classify_module(module),
                        'line': node.lineno
                    })

        return {
            'file': filepath,
            'imports': imports,
            'from_imports': from_imports
        }

    def _classify_module(self, module: str) -> str:
        """Classify module as stdlib, third_party, or local."""
        if module in self.STDLIB_MODULES:
            return 'stdlib'
        # Assume anything not in stdlib is third-party
        # (local detection requires project context)
        return 'third_party'

    def analyze_project(self, directory: str) -> Dict:
        """
        Analyze all Python files in directory.

        Args:
            directory: Project directory

        Returns:
            Aggregated analysis results
        """
        path = Path(directory)
        py_files = list(path.rglob('*.py'))

        stdlib = set()
        third_party = set()
        local = set()
        by_file = {}

        # Get local module names
        local_modules = self._get_local_modules(path)

        for py_file in py_files:
            result = self.analyze_file(str(py_file))
            if 'error' in result:
                continue

            by_file[str(py_file)] = result

            for imp in result['imports']:
                module = imp['module'].split('.')[0]
                if module in local_modules:
                    local.add(module)
                elif imp['type'] == 'stdlib':
                    stdlib.add(module)
                else:
                    third_party.add(module)

            for imp in result['from_imports']:
                module = imp['module'].split('.')[0]
                if module in local_modules:
                    local.add(module)
                elif imp['type'] == 'stdlib':
                    stdlib.add(module)
                else:
                    third_party.add(module)

        return {
            'directory': directory,
            'files_analyzed': len(py_files),
            'stdlib': sorted(stdlib),
            'third_party': sorted(third_party),
            'local': sorted(local),
            'by_file': by_file
        }

    def _get_local_modules(self, path: Path) -> Set[str]:
        """Get names of local Python modules."""
        modules = set()

        for py_file in path.rglob('*.py'):
            relative = py_file.relative_to(path)
            # Get top-level module name
            parts = relative.parts
            if parts[0] != '__pycache__':
                modules.add(parts[0].replace('.py', ''))

        # Add directory names (packages)
        for subdir in path.iterdir():
            if subdir.is_dir() and (subdir / '__init__.py').exists():
                modules.add(subdir.name)

        return modules

    def find_unused_imports(self, filepath: str) -> List[Dict]:
        """
        Find imports that are not used in file.

        Args:
            filepath: Path to Python file

        Returns:
            List of unused imports
        """
        with open(filepath, 'r') as f:
            source = f.read()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        # Collect all imports
        imports = {}  # name -> import info

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports[name] = {
                        'module': alias.name,
                        'line': node.lineno,
                        'used': False
                    }

            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports[name] = {
                        'module': f"{node.module}.{alias.name}",
                        'line': node.lineno,
                        'used': False
                    }

        # Find usages
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id in imports:
                    imports[node.id]['used'] = True
            elif isinstance(node, ast.Attribute):
                # Check for module.attr usage
                if isinstance(node.value, ast.Name):
                    if node.value.id in imports:
                        imports[node.value.id]['used'] = True

        # Return unused
        return [
            {'name': name, **info}
            for name, info in imports.items()
            if not info['used']
        ]

    def find_circular_imports(self, directory: str) -> List[List[str]]:
        """
        Detect circular import dependencies.

        Args:
            directory: Project directory

        Returns:
            List of circular dependency chains
        """
        # Build import graph
        graph = {}
        path = Path(directory)
        local_modules = self._get_local_modules(path)

        for py_file in path.rglob('*.py'):
            result = self.analyze_file(str(py_file))
            if 'error' in result:
                continue

            module_name = py_file.stem
            if module_name not in graph:
                graph[module_name] = set()

            for imp in result['imports'] + result['from_imports']:
                dep = imp['module'].split('.')[0]
                if dep in local_modules:
                    graph[module_name].add(dep)

        # Find cycles using DFS
        cycles = []
        visited = set()
        rec_stack = []

        def dfs(node, path):
            if node in path:
                # Found cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return

            if node in visited:
                return

            visited.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                dfs(neighbor, path.copy())

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def generate_requirements(self, directory: str) -> List[str]:
        """
        Generate list of third-party dependencies.

        Args:
            directory: Project directory

        Returns:
            List of package names
        """
        result = self.analyze_project(directory)
        return result['third_party']

    def save_requirements(self, deps: List[str], output: str) -> str:
        """
        Save requirements to file.

        Args:
            deps: List of dependencies
            output: Output file path

        Returns:
            Path to saved file
        """
        with open(output, 'w') as f:
            for dep in sorted(deps):
                f.write(f"{dep}\n")
        return output

    def is_stdlib(self, module: str) -> bool:
        """Check if module is in standard library."""
        return module.split('.')[0] in self.STDLIB_MODULES


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Analyze Python dependencies')
    parser.add_argument('--file', '-f', help='Single Python file to analyze')
    parser.add_argument('--dir', '-d', help='Directory to analyze')
    parser.add_argument('--requirements', '-r', action='store_true',
                        help='Generate requirements.txt')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--unused', action='store_true', help='Find unused imports')
    parser.add_argument('--circular', action='store_true', help='Find circular imports')
    parser.add_argument('--graph', action='store_true', help='Show dependency graph')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()
    analyzer = DependencyAnalyzer()

    if args.file:
        if args.unused:
            unused = analyzer.find_unused_imports(args.file)
            if unused:
                print(f"Unused imports in {args.file}:")
                for imp in unused:
                    print(f"  Line {imp['line']}: {imp['name']} ({imp['module']})")
            else:
                print("No unused imports found")
        else:
            result = analyzer.analyze_file(args.file)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"File: {result['file']}")
                print("\nImports:")
                for imp in result['imports']:
                    print(f"  Line {imp['line']}: {imp['module']} [{imp['type']}]")
                print("\nFrom Imports:")
                for imp in result['from_imports']:
                    names = ', '.join(imp['names'])
                    print(f"  Line {imp['line']}: from {imp['module']} import {names} [{imp['type']}]")

    elif args.dir:
        if args.requirements:
            deps = analyzer.generate_requirements(args.dir)
            if args.output:
                analyzer.save_requirements(deps, args.output)
                print(f"Requirements saved to: {args.output}")
            else:
                print("Third-party dependencies:")
                for dep in deps:
                    print(dep)

        elif args.circular:
            cycles = analyzer.find_circular_imports(args.dir)
            if cycles:
                print("Circular imports detected:")
                for cycle in cycles:
                    print(f"  {' -> '.join(cycle)}")
            else:
                print("No circular imports found")

        else:
            result = analyzer.analyze_project(args.dir)
            if args.json:
                # Don't include by_file in JSON (too verbose)
                result_summary = {k: v for k, v in result.items() if k != 'by_file'}
                print(json.dumps(result_summary, indent=2))
            else:
                print(f"Directory: {result['directory']}")
                print(f"Files analyzed: {result['files_analyzed']}")
                print(f"\nStandard Library ({len(result['stdlib'])}):")
                print(f"  {', '.join(result['stdlib'][:10])}{'...' if len(result['stdlib']) > 10 else ''}")
                print(f"\nThird-Party ({len(result['third_party'])}):")
                for dep in result['third_party']:
                    print(f"  - {dep}")
                print(f"\nLocal Modules ({len(result['local'])}):")
                for mod in result['local']:
                    print(f"  - {mod}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
