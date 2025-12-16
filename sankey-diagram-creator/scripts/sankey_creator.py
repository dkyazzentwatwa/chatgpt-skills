#!/usr/bin/env python3
"""
Sankey Diagram Creator - Create interactive Sankey diagrams for flow visualization

Features:
- Multiple input sources (dict, DataFrame, CSV)
- Node and link customization
- Interactive HTML output
- Static image export (PNG, SVG)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative


# Default configuration
DEFAULT_WIDTH = 1000
DEFAULT_HEIGHT = 600
DEFAULT_COLORMAP = 'Pastel1'
DEFAULT_LINK_OPACITY = 0.5
DEFAULT_PAD = 15


class SankeyCreator:
    """
    Create interactive Sankey diagrams from flow data.

    Example:
        sankey = SankeyCreator()
        sankey.from_dict({'source': ['A'], 'target': ['B'], 'value': [10]})
        sankey.generate().save_html("flow.html")
    """

    def __init__(self):
        """Initialize Sankey creator."""
        self._sources: List[str] = []
        self._targets: List[str] = []
        self._values: List[float] = []
        self._labels: List[str] = []

        self._figure: Optional[go.Figure] = None
        self._title: Optional[str] = None
        self._title_font_size: int = 16

        # Styling
        self._node_colors: Dict[str, str] = {}
        self._colormap: str = DEFAULT_COLORMAP
        self._link_opacity: float = DEFAULT_LINK_OPACITY
        self._link_color_by: str = 'source'  # 'source', 'target', or None
        self._custom_link_colors: Dict[Tuple[str, str], str] = {}

        # Layout
        self._orientation: str = 'h'
        self._pad: int = DEFAULT_PAD
        self._width: int = DEFAULT_WIDTH
        self._height: int = DEFAULT_HEIGHT

    def from_dict(self, data: Dict[str, List]) -> 'SankeyCreator':
        """
        Load flow data from dictionary.

        Args:
            data: Dict with 'source', 'target', 'value' keys (and optional 'label')

        Returns:
            Self for chaining
        """
        required = ['source', 'target', 'value']
        for key in required:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        self._sources = list(data['source'])
        self._targets = list(data['target'])
        self._values = list(data['value'])
        self._labels = list(data.get('label', [''] * len(self._sources)))

        return self

    def from_dataframe(
        self,
        df: pd.DataFrame,
        source: str,
        target: str,
        value: str,
        label: str = None
    ) -> 'SankeyCreator':
        """
        Load flow data from DataFrame.

        Args:
            df: Input DataFrame
            source: Source column name
            target: Target column name
            value: Value column name
            label: Optional label column name

        Returns:
            Self for chaining
        """
        self._sources = df[source].tolist()
        self._targets = df[target].tolist()
        self._values = df[value].tolist()

        if label and label in df.columns:
            self._labels = df[label].tolist()
        else:
            self._labels = [''] * len(self._sources)

        return self

    def from_csv(
        self,
        filepath: str,
        source: str,
        target: str,
        value: str,
        label: str = None
    ) -> 'SankeyCreator':
        """
        Load flow data from CSV file.

        Args:
            filepath: Path to CSV file
            source: Source column name
            target: Target column name
            value: Value column name
            label: Optional label column name

        Returns:
            Self for chaining
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)
        return self.from_dataframe(df, source, target, value, label)

    def add_flow(
        self,
        source: str,
        target: str,
        value: float,
        label: str = ''
    ) -> 'SankeyCreator':
        """
        Add a single flow.

        Args:
            source: Source node name
            target: Target node name
            value: Flow value
            label: Optional flow label

        Returns:
            Self for chaining
        """
        self._sources.append(source)
        self._targets.append(target)
        self._values.append(value)
        self._labels.append(label)
        return self

    def node_colors(
        self,
        colors: Dict[str, str] = None,
        colormap: str = None
    ) -> 'SankeyCreator':
        """
        Set node colors.

        Args:
            colors: Dict mapping node names to colors
            colormap: Matplotlib/Plotly colormap name for auto-coloring

        Returns:
            Self for chaining
        """
        if colors:
            self._node_colors = colors
        if colormap:
            self._colormap = colormap
        return self

    def link_colors(
        self,
        by: str = None,
        opacity: float = None,
        colors: Dict[Tuple[str, str], str] = None
    ) -> 'SankeyCreator':
        """
        Set link colors.

        Args:
            by: Color links by 'source', 'target', or None (uniform)
            opacity: Link opacity (0-1)
            colors: Dict mapping (source, target) tuples to colors

        Returns:
            Self for chaining
        """
        if by is not None:
            self._link_color_by = by
        if opacity is not None:
            self._link_opacity = opacity
        if colors:
            self._custom_link_colors = colors
        return self

    def title(self, text: str, font_size: int = 16) -> 'SankeyCreator':
        """
        Set diagram title.

        Args:
            text: Title text
            font_size: Font size

        Returns:
            Self for chaining
        """
        self._title = text
        self._title_font_size = font_size
        return self

    def layout(
        self,
        orientation: str = None,
        pad: int = None
    ) -> 'SankeyCreator':
        """
        Set layout options.

        Args:
            orientation: 'h' for horizontal, 'v' for vertical
            pad: Node padding

        Returns:
            Self for chaining
        """
        if orientation:
            self._orientation = orientation
        if pad is not None:
            self._pad = pad
        return self

    def _get_all_nodes(self) -> List[str]:
        """Get unique nodes in order."""
        seen = set()
        nodes = []
        for s, t in zip(self._sources, self._targets):
            if s not in seen:
                nodes.append(s)
                seen.add(s)
            if t not in seen:
                nodes.append(t)
                seen.add(t)
        return nodes

    def _get_node_indices(self) -> Tuple[List[int], List[int]]:
        """Get source and target indices."""
        nodes = self._get_all_nodes()
        node_map = {name: idx for idx, name in enumerate(nodes)}
        sources = [node_map[s] for s in self._sources]
        targets = [node_map[t] for t in self._targets]
        return sources, targets

    def _get_colormap_colors(self, n: int) -> List[str]:
        """Get colors from colormap."""
        # Try plotly qualitative colormaps
        colormap_mapping = {
            'Pastel1': qualitative.Pastel1,
            'Pastel2': qualitative.Pastel2,
            'Set1': qualitative.Set1,
            'Set2': qualitative.Set2,
            'Set3': qualitative.Set3,
            'Paired': qualitative.Pastel,
            'Dark2': qualitative.Dark2,
            'Alphabet': qualitative.Alphabet,
            'Bold': qualitative.Bold,
            'Safe': qualitative.Safe,
            'Vivid': qualitative.Vivid,
            'Prism': qualitative.Prism,
        }

        if self._colormap in colormap_mapping:
            colors = colormap_mapping[self._colormap]
        else:
            # Default to Pastel1
            colors = qualitative.Pastel1

        # Cycle through colors if not enough
        return [colors[i % len(colors)] for i in range(n)]

    def _get_node_colors(self) -> List[str]:
        """Get node colors."""
        nodes = self._get_all_nodes()
        auto_colors = self._get_colormap_colors(len(nodes))

        colors = []
        for i, node in enumerate(nodes):
            if node in self._node_colors:
                colors.append(self._node_colors[node])
            else:
                colors.append(auto_colors[i])

        return colors

    def _get_link_colors(self) -> List[str]:
        """Get link colors with opacity."""
        node_colors = self._get_node_colors()
        nodes = self._get_all_nodes()
        node_map = {name: idx for idx, name in enumerate(nodes)}

        colors = []
        for s, t in zip(self._sources, self._targets):
            # Check custom colors first
            if (s, t) in self._custom_link_colors:
                color = self._custom_link_colors[(s, t)]
            elif self._link_color_by == 'source':
                color = node_colors[node_map[s]]
            elif self._link_color_by == 'target':
                color = node_colors[node_map[t]]
            else:
                color = 'rgba(200, 200, 200, 0.5)'

            # Add opacity
            if color.startswith('#'):
                # Convert hex to rgba
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                color = f'rgba({r}, {g}, {b}, {self._link_opacity})'
            elif color.startswith('rgb('):
                # Convert rgb to rgba
                color = color.replace('rgb(', 'rgba(').replace(')', f', {self._link_opacity})')
            elif not color.startswith('rgba'):
                # Named color - wrap in rgba
                color = f'rgba(150, 150, 150, {self._link_opacity})'

            colors.append(color)

        return colors

    def generate(self) -> 'SankeyCreator':
        """
        Generate the Sankey diagram.

        Returns:
            Self for chaining
        """
        if not self._sources:
            raise ValueError("No data provided. Use from_dict, from_csv, or add_flow first.")

        nodes = self._get_all_nodes()
        source_indices, target_indices = self._get_node_indices()
        node_colors = self._get_node_colors()
        link_colors = self._get_link_colors()

        # Create hover templates
        link_customdata = list(zip(self._sources, self._targets, self._values, self._labels))
        link_hovertemplate = (
            '<b>%{customdata[0]}</b> → <b>%{customdata[1]}</b><br>'
            'Value: %{value:,.0f}<br>'
            '%{customdata[3]}'
            '<extra></extra>'
        )

        # Build Sankey diagram
        sankey = go.Sankey(
            orientation=self._orientation,
            node=dict(
                pad=self._pad,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=nodes,
                color=node_colors,
                hovertemplate='<b>%{label}</b><br>Total: %{value:,.0f}<extra></extra>'
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=self._values,
                color=link_colors,
                customdata=link_customdata,
                hovertemplate=link_hovertemplate
            )
        )

        # Create figure
        self._figure = go.Figure(data=[sankey])

        # Update layout
        layout_kwargs = {
            'width': self._width,
            'height': self._height,
            'font': dict(size=12),
        }

        if self._title:
            layout_kwargs['title'] = dict(
                text=self._title,
                font=dict(size=self._title_font_size)
            )

        self._figure.update_layout(**layout_kwargs)

        return self

    def get_figure(self) -> go.Figure:
        """
        Get the Plotly figure object.

        Returns:
            Plotly Figure object
        """
        if self._figure is None:
            self.generate()
        return self._figure

    def save_html(self, path: str, auto_open: bool = False) -> str:
        """
        Save as interactive HTML.

        Args:
            path: Output file path
            auto_open: Open in browser after saving

        Returns:
            Path to saved file
        """
        if self._figure is None:
            self.generate()

        self._figure.write_html(path, auto_open=auto_open)
        return path

    def save_image(
        self,
        path: str,
        format: str = None,
        width: int = None,
        height: int = None
    ) -> str:
        """
        Save as static image.

        Args:
            path: Output file path
            format: 'png', 'svg', 'pdf', or 'jpeg'
            width: Image width
            height: Image height

        Returns:
            Path to saved file
        """
        if self._figure is None:
            self.generate()

        output_path = Path(path)
        if format is None:
            format = output_path.suffix.lower().lstrip('.')

        self._figure.write_image(
            str(output_path),
            format=format,
            width=width or self._width,
            height=height or self._height
        )

        return str(output_path)

    def show(self) -> None:
        """Display diagram in browser/notebook."""
        if self._figure is None:
            self.generate()
        self._figure.show()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Create interactive Sankey diagrams',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sankey_creator.py --input flows.csv --source from --target to --value amount --output flow.html
  python sankey_creator.py --input budget.csv --source cat --target subcat --value val --title "Budget" --output budget.html
  python sankey_creator.py --input data.csv --source src --target dst --value val --output diagram.png
        """
    )

    # Input options
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--source', '-s', required=True, help='Source column name')
    parser.add_argument('--target', '-t', required=True, help='Target column name')
    parser.add_argument('--value', '-v', required=True, help='Value column name')
    parser.add_argument('--label', '-l', help='Optional label column name')

    # Output options
    parser.add_argument('--output', '-o', default='sankey.html',
                        help='Output file path (default: sankey.html)')

    # Styling options
    parser.add_argument('--title', help='Diagram title')
    parser.add_argument('--colormap', '-c', default='Pastel1',
                        help='Color scheme (default: Pastel1)')
    parser.add_argument('--link-opacity', type=float, default=0.5,
                        help='Link opacity 0-1 (default: 0.5)')
    parser.add_argument('--orientation', choices=['h', 'v'], default='h',
                        help='Orientation: h=horizontal, v=vertical')

    # Size options
    parser.add_argument('--width', '-W', type=int, default=1000,
                        help='Image width (default: 1000)')
    parser.add_argument('--height', '-H', type=int, default=600,
                        help='Image height (default: 600)')

    # Behavior
    parser.add_argument('--open', action='store_true',
                        help='Open HTML in browser after saving')

    args = parser.parse_args()

    try:
        # Create Sankey
        sankey = SankeyCreator()
        sankey.from_csv(
            args.input,
            source=args.source,
            target=args.target,
            value=args.value,
            label=args.label
        )

        # Configure
        sankey.node_colors(colormap=args.colormap)
        sankey.link_colors(opacity=args.link_opacity)
        sankey.layout(orientation=args.orientation)
        sankey._width = args.width
        sankey._height = args.height

        if args.title:
            sankey.title(args.title)

        # Generate
        sankey.generate()

        # Save
        output_path = Path(args.output)
        if output_path.suffix.lower() == '.html':
            sankey.save_html(args.output, auto_open=args.open)
        else:
            sankey.save_image(args.output, width=args.width, height=args.height)

        print(f"Sankey diagram saved to: {args.output}")

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
