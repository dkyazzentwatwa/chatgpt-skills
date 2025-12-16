#!/usr/bin/env python3
"""
Flowchart Generator - Create flowcharts from definitions or DSL

Features:
- Python DSL for programmatic creation
- YAML/JSON file input
- Standard flowchart shapes
- Swimlanes for multi-actor processes
- Multiple style presets
- PNG/SVG/PDF/DOT export
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
from graphviz import Digraph


# Node type shapes
NODE_SHAPES = {
    'start': 'ellipse',
    'end': 'ellipse',
    'process': 'box',
    'decision': 'diamond',
    'io': 'parallelogram',
    'input_output': 'parallelogram',
    'connector': 'circle',
}

# Style presets
STYLE_PRESETS = {
    'business': {
        'start': {'fillcolor': '#27ae60', 'fontcolor': 'white'},
        'end': {'fillcolor': '#c0392b', 'fontcolor': 'white'},
        'process': {'fillcolor': '#3498db', 'fontcolor': 'white'},
        'decision': {'fillcolor': '#f39c12', 'fontcolor': 'white'},
        'io': {'fillcolor': '#9b59b6', 'fontcolor': 'white'},
        'connector': {'fillcolor': '#95a5a6', 'fontcolor': 'white'},
        'edge': {'color': '#2c3e50'},
        'graph': {'bgcolor': 'white'},
    },
    'technical': {
        'start': {'fillcolor': '#2d3436', 'fontcolor': 'white'},
        'end': {'fillcolor': '#2d3436', 'fontcolor': 'white'},
        'process': {'fillcolor': '#636e72', 'fontcolor': 'white'},
        'decision': {'fillcolor': '#b2bec3', 'fontcolor': '#2d3436'},
        'io': {'fillcolor': '#dfe6e9', 'fontcolor': '#2d3436'},
        'connector': {'fillcolor': '#74b9ff', 'fontcolor': 'white'},
        'edge': {'color': '#2d3436'},
        'graph': {'bgcolor': 'white'},
    },
    'minimal': {
        'start': {'fillcolor': 'white', 'fontcolor': 'black'},
        'end': {'fillcolor': 'white', 'fontcolor': 'black'},
        'process': {'fillcolor': 'white', 'fontcolor': 'black'},
        'decision': {'fillcolor': 'white', 'fontcolor': 'black'},
        'io': {'fillcolor': 'white', 'fontcolor': 'black'},
        'connector': {'fillcolor': 'white', 'fontcolor': 'black'},
        'edge': {'color': 'black'},
        'graph': {'bgcolor': 'white'},
    },
    'colorful': {
        'start': {'fillcolor': '#00b894', 'fontcolor': 'white'},
        'end': {'fillcolor': '#d63031', 'fontcolor': 'white'},
        'process': {'fillcolor': '#0984e3', 'fontcolor': 'white'},
        'decision': {'fillcolor': '#fdcb6e', 'fontcolor': '#2d3436'},
        'io': {'fillcolor': '#a29bfe', 'fontcolor': 'white'},
        'connector': {'fillcolor': '#fd79a8', 'fontcolor': 'white'},
        'edge': {'color': '#636e72'},
        'graph': {'bgcolor': '#ffeaa7'},
    },
}


@dataclass
class Node:
    """Represents a flowchart node."""
    id: str
    label: str
    node_type: str
    lane: Optional[str] = None


@dataclass
class Connection:
    """Represents a connection between nodes."""
    from_id: str
    to_id: str
    label: Optional[str] = None


class FlowchartGenerator:
    """
    Generate flowcharts from definitions or DSL.

    Example:
        flow = FlowchartGenerator()
        flow.start("Start")
        flow.process("Do Something", id="step1")
        flow.end("End")
        flow.connect("Start", "step1")
        flow.connect("step1", "End")
        flow.generate().save("chart.png")
    """

    def __init__(self):
        """Initialize flowchart generator."""
        self._nodes: Dict[str, Node] = {}
        self._connections: List[Connection] = []
        self._swimlanes: List[str] = []
        self._graph: Optional[Digraph] = None

        self._title: Optional[str] = None
        self._layout: str = 'TB'
        self._style_preset: str = 'business'
        self._custom_colors: Dict[str, Dict[str, str]] = {}

        # Auto-incrementing ID counter
        self._id_counter: int = 0

    def _generate_id(self, prefix: str = 'node') -> str:
        """Generate unique node ID."""
        self._id_counter += 1
        return f"{prefix}_{self._id_counter}"

    def _add_node(
        self,
        label: str,
        node_type: str,
        id: str = None,
        lane: str = None
    ) -> str:
        """Add a node to the flowchart."""
        node_id = id or self._generate_id(node_type)

        # Clean ID for graphviz
        safe_id = node_id.replace(' ', '_').replace('.', '_')

        node = Node(
            id=safe_id,
            label=label,
            node_type=node_type,
            lane=lane
        )
        self._nodes[safe_id] = node
        return safe_id

    def start(self, label: str = "Start", id: str = None, lane: str = None) -> 'FlowchartGenerator':
        """
        Add start node.

        Args:
            label: Node label
            id: Optional node ID
            lane: Optional swimlane

        Returns:
            Self for chaining
        """
        self._add_node(label, 'start', id or 'Start', lane)
        return self

    def end(self, label: str = "End", id: str = None, lane: str = None) -> 'FlowchartGenerator':
        """
        Add end node.

        Args:
            label: Node label
            id: Optional node ID
            lane: Optional swimlane

        Returns:
            Self for chaining
        """
        self._add_node(label, 'end', id or 'End', lane)
        return self

    def process(self, label: str, id: str = None, lane: str = None) -> 'FlowchartGenerator':
        """
        Add process node.

        Args:
            label: Node label
            id: Optional node ID
            lane: Optional swimlane

        Returns:
            Self for chaining
        """
        self._add_node(label, 'process', id, lane)
        return self

    def decision(self, label: str, id: str = None, lane: str = None) -> 'FlowchartGenerator':
        """
        Add decision node.

        Args:
            label: Node label
            id: Optional node ID
            lane: Optional swimlane

        Returns:
            Self for chaining
        """
        self._add_node(label, 'decision', id, lane)
        return self

    def input_output(self, label: str, id: str = None, lane: str = None) -> 'FlowchartGenerator':
        """
        Add input/output node.

        Args:
            label: Node label
            id: Optional node ID
            lane: Optional swimlane

        Returns:
            Self for chaining
        """
        self._add_node(label, 'io', id, lane)
        return self

    def connector(self, label: str, id: str = None, lane: str = None) -> 'FlowchartGenerator':
        """
        Add connector node.

        Args:
            label: Node label
            id: Optional node ID
            lane: Optional swimlane

        Returns:
            Self for chaining
        """
        self._add_node(label, 'connector', id, lane)
        return self

    def connect(
        self,
        from_id: str,
        to_id: str,
        label: str = None
    ) -> 'FlowchartGenerator':
        """
        Add connection between nodes.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            label: Optional edge label

        Returns:
            Self for chaining
        """
        # Clean IDs
        safe_from = from_id.replace(' ', '_').replace('.', '_')
        safe_to = to_id.replace(' ', '_').replace('.', '_')

        connection = Connection(safe_from, safe_to, label)
        self._connections.append(connection)
        return self

    def swimlanes(self, lanes: List[str]) -> 'FlowchartGenerator':
        """
        Define swimlanes.

        Args:
            lanes: List of lane names

        Returns:
            Self for chaining
        """
        self._swimlanes = lanes
        return self

    def from_yaml(self, filepath: str) -> 'FlowchartGenerator':
        """
        Load flowchart from YAML file.

        Args:
            filepath: Path to YAML file

        Returns:
            Self for chaining
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        return self._load_from_dict(data)

    def from_json(self, filepath: str) -> 'FlowchartGenerator':
        """
        Load flowchart from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Self for chaining
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return self._load_from_dict(data)

    def _load_from_dict(self, data: Dict[str, Any]) -> 'FlowchartGenerator':
        """Load flowchart from dictionary."""
        # Set properties
        if 'title' in data:
            self._title = data['title']
        if 'layout' in data:
            self._layout = data['layout']
        if 'style' in data:
            self._style_preset = data['style']
        if 'swimlanes' in data:
            self._swimlanes = data['swimlanes']

        # Add nodes
        for node_data in data.get('nodes', []):
            node_type = node_data.get('type', 'process')
            self._add_node(
                label=node_data.get('label', node_data.get('id', '')),
                node_type=node_type,
                id=node_data.get('id'),
                lane=node_data.get('lane')
            )

        # Add connections
        for conn_data in data.get('connections', []):
            self.connect(
                from_id=conn_data['from'],
                to_id=conn_data['to'],
                label=conn_data.get('label')
            )

        return self

    def title(self, text: str) -> 'FlowchartGenerator':
        """
        Set flowchart title.

        Args:
            text: Title text

        Returns:
            Self for chaining
        """
        self._title = text
        return self

    def layout(self, direction: str) -> 'FlowchartGenerator':
        """
        Set layout direction.

        Args:
            direction: TB, LR, BT, or RL

        Returns:
            Self for chaining
        """
        valid = ['TB', 'LR', 'BT', 'RL']
        if direction.upper() not in valid:
            raise ValueError(f"Invalid direction: {direction}. Use {valid}")
        self._layout = direction.upper()
        return self

    def style(self, preset: str) -> 'FlowchartGenerator':
        """
        Set style preset.

        Args:
            preset: business, technical, minimal, or colorful

        Returns:
            Self for chaining
        """
        if preset not in STYLE_PRESETS:
            raise ValueError(f"Invalid style: {preset}. Use {list(STYLE_PRESETS.keys())}")
        self._style_preset = preset
        return self

    def node_colors(self, colors: Dict[str, str]) -> 'FlowchartGenerator':
        """
        Set custom node colors.

        Args:
            colors: Dict mapping node types to colors

        Returns:
            Self for chaining
        """
        for node_type, color in colors.items():
            if node_type not in self._custom_colors:
                self._custom_colors[node_type] = {}
            self._custom_colors[node_type]['fillcolor'] = color
        return self

    def _get_node_style(self, node_type: str) -> Dict[str, str]:
        """Get style attributes for node type."""
        base_style = STYLE_PRESETS.get(self._style_preset, STYLE_PRESETS['business'])
        style = base_style.get(node_type, base_style.get('process', {})).copy()

        # Apply custom colors
        if node_type in self._custom_colors:
            style.update(self._custom_colors[node_type])

        return style

    def generate(self) -> 'FlowchartGenerator':
        """
        Generate the flowchart.

        Returns:
            Self for chaining
        """
        if not self._nodes:
            raise ValueError("No nodes defined")

        # Create main graph
        self._graph = Digraph(
            name='flowchart',
            format='png',
            engine='dot'
        )

        # Graph attributes
        self._graph.attr(
            rankdir=self._layout,
            splines='ortho',
            nodesep='0.5',
            ranksep='0.5'
        )

        if self._title:
            self._graph.attr(label=self._title, labelloc='t', fontsize='16')

        # Style attributes
        base_style = STYLE_PRESETS.get(self._style_preset, STYLE_PRESETS['business'])
        if 'graph' in base_style:
            self._graph.attr(**base_style['graph'])

        # Default node attributes
        self._graph.attr('node', style='filled', fontname='Arial')

        # Default edge attributes
        if 'edge' in base_style:
            self._graph.attr('edge', **base_style['edge'])

        # Handle swimlanes
        if self._swimlanes:
            self._generate_with_swimlanes()
        else:
            self._generate_simple()

        return self

    def _generate_simple(self) -> None:
        """Generate simple flowchart without swimlanes."""
        # Add nodes
        for node_id, node in self._nodes.items():
            shape = NODE_SHAPES.get(node.node_type, 'box')
            style = self._get_node_style(node.node_type)

            self._graph.node(
                node_id,
                label=node.label,
                shape=shape,
                **style
            )

        # Add edges
        base_style = STYLE_PRESETS.get(self._style_preset, STYLE_PRESETS['business'])
        edge_style = base_style.get('edge', {})

        for conn in self._connections:
            edge_attrs = edge_style.copy()
            if conn.label:
                edge_attrs['label'] = conn.label

            self._graph.edge(conn.from_id, conn.to_id, **edge_attrs)

    def _generate_with_swimlanes(self) -> None:
        """Generate flowchart with swimlanes."""
        # Create subgraphs for each lane
        for i, lane in enumerate(self._swimlanes):
            with self._graph.subgraph(name=f'cluster_{i}') as sub:
                sub.attr(label=lane, style='rounded', bgcolor='#f8f9fa')

                # Add nodes belonging to this lane
                for node_id, node in self._nodes.items():
                    if node.lane == lane:
                        shape = NODE_SHAPES.get(node.node_type, 'box')
                        style = self._get_node_style(node.node_type)

                        sub.node(
                            node_id,
                            label=node.label,
                            shape=shape,
                            **style
                        )

        # Add nodes without lane assignment to main graph
        for node_id, node in self._nodes.items():
            if not node.lane:
                shape = NODE_SHAPES.get(node.node_type, 'box')
                style = self._get_node_style(node.node_type)

                self._graph.node(
                    node_id,
                    label=node.label,
                    shape=shape,
                    **style
                )

        # Add edges
        base_style = STYLE_PRESETS.get(self._style_preset, STYLE_PRESETS['business'])
        edge_style = base_style.get('edge', {})

        for conn in self._connections:
            edge_attrs = edge_style.copy()
            if conn.label:
                edge_attrs['label'] = conn.label

            self._graph.edge(conn.from_id, conn.to_id, **edge_attrs)

    def to_dot(self) -> str:
        """
        Get DOT source code.

        Returns:
            DOT source string
        """
        if self._graph is None:
            self.generate()
        return self._graph.source

    def save(self, path: str, dpi: int = 96) -> str:
        """
        Save flowchart to file.

        Args:
            path: Output file path
            dpi: Image resolution

        Returns:
            Path to saved file
        """
        if self._graph is None:
            self.generate()

        output_path = Path(path)
        format_ext = output_path.suffix.lower().lstrip('.')

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

        output_base = str(output_path.with_suffix(''))
        self._graph.render(output_base, cleanup=True)

        return str(output_path)

    def show(self) -> None:
        """Display flowchart in system viewer."""
        if self._graph is None:
            self.generate()
        self._graph.view()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate flowcharts from YAML/JSON definitions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python flowchart_gen.py --input process.yaml --output flowchart.png
  python flowchart_gen.py --input workflow.json --layout LR --output workflow.svg
  python flowchart_gen.py --input process.yaml --style technical --output chart.png
        """
    )

    parser.add_argument('--input', '-i', required=True,
                        help='Input YAML or JSON file')
    parser.add_argument('--output', '-o', default='flowchart.png',
                        help='Output file path (default: flowchart.png)')
    parser.add_argument('--layout', '-l', choices=['TB', 'LR', 'BT', 'RL'],
                        default='TB', help='Layout direction (default: TB)')
    parser.add_argument('--style', '-s',
                        choices=['business', 'technical', 'minimal', 'colorful'],
                        default='business', help='Style preset (default: business)')
    parser.add_argument('--dpi', type=int, default=96,
                        help='Image resolution (default: 96)')

    args = parser.parse_args()

    try:
        flow = FlowchartGenerator()

        # Load from file
        input_path = Path(args.input)
        if input_path.suffix.lower() in ['.yaml', '.yml']:
            flow.from_yaml(args.input)
        elif input_path.suffix.lower() == '.json':
            flow.from_json(args.input)
        else:
            raise ValueError(f"Unsupported input format: {input_path.suffix}")

        # Apply overrides
        flow.layout(args.layout)
        flow.style(args.style)

        # Generate and save
        flow.generate()
        output_path = flow.save(args.output, dpi=args.dpi)
        print(f"Flowchart saved to: {output_path}")

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
