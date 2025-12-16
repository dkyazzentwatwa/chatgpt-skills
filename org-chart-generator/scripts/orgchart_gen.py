#!/usr/bin/env python3
"""
Org Chart Generator - Create organizational hierarchy charts

Features:
- Multiple input sources (CSV, JSON, dict)
- Multiple layout options (top-down, left-right, etc.)
- Department/level coloring
- PNG/SVG/PDF/DOT export
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import pandas as pd
from graphviz import Digraph


# Layout direction mapping
LAYOUT_DIRECTIONS = {
    'top-down': 'TB',
    'bottom-up': 'BT',
    'left-right': 'LR',
    'right-left': 'RL',
    'TB': 'TB',
    'BT': 'BT',
    'LR': 'LR',
    'RL': 'RL',
}

# Default colors for levels
DEFAULT_LEVEL_COLORS = {
    0: '#e74c3c',  # Red - Executive
    1: '#f39c12',  # Orange - Senior leadership
    2: '#3498db',  # Blue - Directors
    3: '#2ecc71',  # Green - Managers
    4: '#9b59b6',  # Purple - Team leads
    5: '#1abc9c',  # Teal - Staff
}


@dataclass
class Node:
    """Represents a node in the org chart."""
    id: str
    name: str
    title: Optional[str] = None
    department: Optional[str] = None
    parent_id: Optional[str] = None
    level: int = 0
    children: List['Node'] = field(default_factory=list)


class OrgChartGenerator:
    """
    Generate organizational hierarchy charts.

    Example:
        org = OrgChartGenerator()
        org.from_csv("employees.csv", name="name", manager="reports_to")
        org.generate().save("org_chart.png")
    """

    def __init__(self):
        """Initialize org chart generator."""
        self._nodes: Dict[str, Node] = {}
        self._roots: List[str] = []
        self._graph: Optional[Digraph] = None

        # Layout settings
        self._direction: str = 'TB'
        self._rank_sep: float = 1.0
        self._node_sep: float = 0.5

        # Node styling
        self._node_shape: str = 'box'
        self._node_font: str = 'Arial'
        self._node_font_size: int = 12
        self._node_border_width: float = 1.0
        self._node_color: Optional[str] = None
        self._dept_colors: Dict[str, str] = {}
        self._level_colors: Dict[int, str] = {}

        # Edge styling
        self._edge_style: str = 'ortho'  # ortho, line, spline
        self._edge_color: str = '#666666'
        self._edge_width: float = 1.0

    def from_csv(
        self,
        filepath: str,
        name: str,
        manager: str,
        title: str = None,
        department: str = None
    ) -> 'OrgChartGenerator':
        """
        Load org structure from CSV file.

        Args:
            filepath: Path to CSV file
            name: Column containing employee name
            manager: Column containing manager name (empty for root)
            title: Optional column for job title
            department: Optional column for department

        Returns:
            Self for chaining
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)

        # Create nodes
        for _, row in df.iterrows():
            node_name = str(row[name])
            node_id = self._make_id(node_name)

            manager_name = row.get(manager)
            parent_id = None
            if pd.notna(manager_name) and str(manager_name).strip():
                parent_id = self._make_id(str(manager_name))

            node = Node(
                id=node_id,
                name=node_name,
                title=str(row[title]) if title and pd.notna(row.get(title)) else None,
                department=str(row[department]) if department and pd.notna(row.get(department)) else None,
                parent_id=parent_id
            )
            self._nodes[node_id] = node

        # Build tree structure
        self._build_tree()

        return self

    def from_json(self, filepath: str) -> 'OrgChartGenerator':
        """
        Load org structure from JSON file.

        Args:
            filepath: Path to JSON file (nested structure)

        Returns:
            Self for chaining
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return self.from_dict(data)

    def from_dict(self, data: Dict[str, Any]) -> 'OrgChartGenerator':
        """
        Load org structure from nested dictionary.

        Args:
            data: Nested dict with 'name', optional 'title', 'department', 'children'

        Returns:
            Self for chaining
        """
        self._nodes = {}
        self._roots = []
        self._load_nested(data, None, 0)
        return self

    def _load_nested(
        self,
        data: Dict[str, Any],
        parent_id: Optional[str],
        level: int
    ) -> str:
        """Recursively load nested dictionary structure."""
        name = data.get('name', 'Unknown')
        node_id = self._make_id(name)

        # Handle duplicate names
        counter = 1
        original_id = node_id
        while node_id in self._nodes:
            node_id = f"{original_id}_{counter}"
            counter += 1

        node = Node(
            id=node_id,
            name=name,
            title=data.get('title'),
            department=data.get('department'),
            parent_id=parent_id,
            level=level
        )
        self._nodes[node_id] = node

        if parent_id is None:
            self._roots.append(node_id)
        else:
            self._nodes[parent_id].children.append(node)

        # Process children
        for child_data in data.get('children', []):
            self._load_nested(child_data, node_id, level + 1)

        return node_id

    def add_node(
        self,
        name: str,
        title: str = None,
        department: str = None
    ) -> 'OrgChartGenerator':
        """
        Add a node programmatically.

        Args:
            name: Node name
            title: Optional title
            department: Optional department

        Returns:
            Self for chaining
        """
        node_id = self._make_id(name)
        node = Node(
            id=node_id,
            name=name,
            title=title,
            department=department
        )
        self._nodes[node_id] = node
        return self

    def add_relationship(
        self,
        child_name: str,
        parent_name: str
    ) -> 'OrgChartGenerator':
        """
        Add parent-child relationship.

        Args:
            child_name: Name of child node
            parent_name: Name of parent node

        Returns:
            Self for chaining
        """
        child_id = self._make_id(child_name)
        parent_id = self._make_id(parent_name)

        if child_id not in self._nodes:
            raise ValueError(f"Child node not found: {child_name}")
        if parent_id not in self._nodes:
            raise ValueError(f"Parent node not found: {parent_name}")

        self._nodes[child_id].parent_id = parent_id
        self._build_tree()
        return self

    def _make_id(self, name: str) -> str:
        """Create valid node ID from name."""
        return name.lower().replace(' ', '_').replace('.', '').replace(',', '')

    def _build_tree(self) -> None:
        """Build tree structure and calculate levels."""
        # Find roots (nodes without parents)
        self._roots = []
        for node_id, node in self._nodes.items():
            if node.parent_id is None:
                self._roots.append(node_id)
            elif node.parent_id in self._nodes:
                # Add as child of parent
                parent = self._nodes[node.parent_id]
                if node not in parent.children:
                    parent.children.append(node)

        # Calculate levels
        def set_level(node_id: str, level: int):
            self._nodes[node_id].level = level
            for child in self._nodes[node_id].children:
                set_level(child.id, level + 1)

        for root_id in self._roots:
            set_level(root_id, 0)

    def layout(self, direction: str) -> 'OrgChartGenerator':
        """
        Set layout direction.

        Args:
            direction: 'top-down', 'bottom-up', 'left-right', 'right-left'

        Returns:
            Self for chaining
        """
        if direction not in LAYOUT_DIRECTIONS:
            raise ValueError(f"Invalid direction: {direction}")
        self._direction = LAYOUT_DIRECTIONS[direction]
        return self

    def spacing(self, rank: float = None, node: float = None) -> 'OrgChartGenerator':
        """
        Set spacing between nodes.

        Args:
            rank: Vertical spacing between ranks
            node: Horizontal spacing between nodes

        Returns:
            Self for chaining
        """
        if rank is not None:
            self._rank_sep = rank
        if node is not None:
            self._node_sep = node
        return self

    def node_color(self, color: str) -> 'OrgChartGenerator':
        """
        Set uniform node color.

        Args:
            color: Color (hex or name)

        Returns:
            Self for chaining
        """
        self._node_color = color
        return self

    def colors_by_department(self, colors: Dict[str, str]) -> 'OrgChartGenerator':
        """
        Set colors by department.

        Args:
            colors: Dict mapping department names to colors

        Returns:
            Self for chaining
        """
        self._dept_colors = colors
        return self

    def colors_by_level(self, colors: Dict[int, str]) -> 'OrgChartGenerator':
        """
        Set colors by level.

        Args:
            colors: Dict mapping level (0=root) to colors

        Returns:
            Self for chaining
        """
        self._level_colors = colors
        return self

    def node_style(
        self,
        shape: str = None,
        font: str = None,
        font_size: int = None,
        border_width: float = None
    ) -> 'OrgChartGenerator':
        """
        Set node styling.

        Args:
            shape: 'box', 'ellipse', 'diamond', 'record'
            font: Font name
            font_size: Font size
            border_width: Border width

        Returns:
            Self for chaining
        """
        if shape:
            self._node_shape = shape
        if font:
            self._node_font = font
        if font_size:
            self._node_font_size = font_size
        if border_width is not None:
            self._node_border_width = border_width
        return self

    def edge_style(
        self,
        style: str = None,
        color: str = None,
        width: float = None
    ) -> 'OrgChartGenerator':
        """
        Set edge styling.

        Args:
            style: 'orthogonal', 'straight', 'curved'
            color: Edge color
            width: Edge width

        Returns:
            Self for chaining
        """
        if style:
            style_map = {
                'orthogonal': 'ortho',
                'straight': 'line',
                'curved': 'spline',
                'ortho': 'ortho',
                'line': 'line',
                'spline': 'spline'
            }
            self._edge_style = style_map.get(style, 'ortho')
        if color:
            self._edge_color = color
        if width is not None:
            self._edge_width = width
        return self

    def _get_node_color(self, node: Node) -> str:
        """Get color for a node."""
        # Priority: uniform color > department > level > default
        if self._node_color:
            return self._node_color
        if node.department and node.department in self._dept_colors:
            return self._dept_colors[node.department]
        if node.level in self._level_colors:
            return self._level_colors[node.level]
        if node.level in DEFAULT_LEVEL_COLORS:
            return DEFAULT_LEVEL_COLORS[node.level]
        return '#3498db'

    def _get_node_label(self, node: Node) -> str:
        """Get label for a node."""
        if node.title:
            return f"{node.name}\\n{node.title}"
        return node.name

    def generate(self) -> 'OrgChartGenerator':
        """
        Generate the org chart.

        Returns:
            Self for chaining
        """
        if not self._nodes:
            raise ValueError("No data provided. Use from_csv, from_json, from_dict, or add_node first.")

        # Create graph
        self._graph = Digraph(
            name='org_chart',
            format='png',
            engine='dot'
        )

        # Set graph attributes
        self._graph.attr(
            rankdir=self._direction,
            ranksep=str(self._rank_sep),
            nodesep=str(self._node_sep),
            splines=self._edge_style
        )

        # Set default node attributes
        self._graph.attr('node',
            shape=self._node_shape,
            fontname=self._node_font,
            fontsize=str(self._node_font_size),
            penwidth=str(self._node_border_width),
            style='filled'
        )

        # Set default edge attributes
        self._graph.attr('edge',
            color=self._edge_color,
            penwidth=str(self._edge_width)
        )

        # Add nodes
        for node_id, node in self._nodes.items():
            color = self._get_node_color(node)
            label = self._get_node_label(node)

            # Determine text color based on background
            fontcolor = self._get_contrast_color(color)

            self._graph.node(
                node_id,
                label=label,
                fillcolor=color,
                fontcolor=fontcolor
            )

        # Add edges
        for node_id, node in self._nodes.items():
            if node.parent_id and node.parent_id in self._nodes:
                self._graph.edge(node.parent_id, node_id)

        return self

    def _get_contrast_color(self, hex_color: str) -> str:
        """Get contrasting text color (black or white) for background."""
        if not hex_color.startswith('#'):
            return 'black'  # Default for named colors

        # Convert hex to RGB
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        # Calculate luminance
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

        return 'black' if luminance > 0.5 else 'white'

    def save(self, path: str, dpi: int = 96) -> str:
        """
        Save org chart to file.

        Args:
            path: Output file path
            dpi: Image resolution (for PNG)

        Returns:
            Path to saved file
        """
        if self._graph is None:
            self.generate()

        output_path = Path(path)
        format_ext = output_path.suffix.lower().lstrip('.')

        # Map extensions to graphviz formats
        format_map = {
            'png': 'png',
            'svg': 'svg',
            'pdf': 'pdf',
            'dot': 'dot',
            'gv': 'dot'
        }

        if format_ext not in format_map:
            raise ValueError(f"Unsupported format: {format_ext}")

        gv_format = format_map[format_ext]
        self._graph.format = gv_format

        if gv_format == 'png':
            self._graph.attr(dpi=str(dpi))

        # Render to file (graphviz adds extension automatically)
        output_base = str(output_path.with_suffix(''))
        self._graph.render(output_base, cleanup=True)

        return str(output_path)

    def to_dot(self) -> str:
        """
        Get DOT source code.

        Returns:
            DOT source string
        """
        if self._graph is None:
            self.generate()
        return self._graph.source

    def show(self) -> None:
        """Display chart in system viewer."""
        if self._graph is None:
            self.generate()
        self._graph.view()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate organizational hierarchy charts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orgchart_gen.py --input employees.csv --name name --manager reports_to --output org.png
  python orgchart_gen.py --input structure.json --output org.svg
  python orgchart_gen.py --input team.csv --name emp --manager mgr --layout left-right --output team.png
        """
    )

    # Input options
    parser.add_argument('--input', '-i', required=True, help='Input CSV or JSON file')
    parser.add_argument('--name', '-n', help='Name column (CSV only)')
    parser.add_argument('--manager', '-m', help='Manager column (CSV only)')
    parser.add_argument('--title', '-t', help='Title column (CSV only)')
    parser.add_argument('--department', '-d', help='Department column (CSV only)')

    # Output options
    parser.add_argument('--output', '-o', default='orgchart.png',
                        help='Output file path (default: orgchart.png)')

    # Layout options
    parser.add_argument('--layout', '-l',
                        choices=['top-down', 'bottom-up', 'left-right', 'right-left'],
                        default='top-down',
                        help='Layout direction (default: top-down)')

    # Style options
    parser.add_argument('--shape',
                        choices=['box', 'ellipse', 'diamond'],
                        default='box',
                        help='Node shape (default: box)')
    parser.add_argument('--dpi', type=int, default=96,
                        help='Image resolution (default: 96)')

    args = parser.parse_args()

    try:
        org = OrgChartGenerator()

        # Load data
        input_path = Path(args.input)
        if input_path.suffix.lower() == '.json':
            org.from_json(args.input)
        else:
            if not args.name or not args.manager:
                parser.error("CSV input requires --name and --manager arguments")
            org.from_csv(
                args.input,
                name=args.name,
                manager=args.manager,
                title=args.title,
                department=args.department
            )

        # Configure
        org.layout(args.layout)
        org.node_style(shape=args.shape)

        # Generate and save
        org.generate()
        output_path = org.save(args.output, dpi=args.dpi)
        print(f"Org chart saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
