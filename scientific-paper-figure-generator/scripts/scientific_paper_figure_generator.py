#!/usr/bin/env python3
"""Scientific Paper Figure Generator - Create publication-ready figures."""

import argparse
import os
from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams


class PaperFigureGenerator:
    """Generate publication-ready scientific figures."""

    JOURNAL_STYLES = {
        'nature': {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'font.size': 8,
            'axes.labelsize': 8,
            'axes.titlesize': 8,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'figure.dpi': 300
        },
        'science': {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'font.size': 9,
            'axes.labelsize': 9,
            'axes.titlesize': 9,
            'figure.dpi': 300
        },
        'ieee': {
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'figure.dpi': 300
        }
    }

    def __init__(self, style: str = 'nature', figsize: Tuple[float, float] = (7, 5)):
        """Initialize with journal style."""
        if style in self.JOURNAL_STYLES:
            rcParams.update(self.JOURNAL_STYLES[style])

        self.fig, self.axes = plt.subplots(figsize=figsize)
        self.panels = []

    def create_multi_panel(self, layout: Tuple[int, int], figsize: Tuple[float, float] = None):
        """Create multi-panel figure."""
        if figsize is None:
            figsize = (7 * layout[1], 5 * layout[0])

        self.fig, self.axes = plt.subplots(*layout, figsize=figsize)
        if layout[0] == 1 and layout[1] == 1:
            self.axes = np.array([self.axes])
        elif layout[0] == 1 or layout[1] == 1:
            self.axes = self.axes.flatten()
        else:
            self.axes = self.axes.flatten()

        return self

    def plot_bar(self, data: pd.DataFrame, x: str, y: str, ax_idx: int = 0,
                xlabel: str = None, ylabel: str = None, title: str = None):
        """Create bar plot."""
        ax = self.axes[ax_idx] if isinstance(self.axes, np.ndarray) else self.axes

        ax.bar(data[x], data[y], color='steelblue', edgecolor='black', linewidth=0.5)
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        if title:
            ax.set_title(title)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        return self

    def plot_line(self, data: pd.DataFrame, x: str, y: str, ax_idx: int = 0,
                 xlabel: str = None, ylabel: str = None, title: str = None,
                 error_column: str = None):
        """Create line plot with optional error bars."""
        ax = self.axes[ax_idx] if isinstance(self.axes, np.ndarray) else self.axes

        if error_column and error_column in data.columns:
            ax.errorbar(data[x], data[y], yerr=data[error_column],
                       marker='o', capsize=3, capthick=1)
        else:
            ax.plot(data[x], data[y], marker='o')

        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        if title:
            ax.set_title(title)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        return self

    def annotate_significance(self, ax_idx: int, x1: float, x2: float,
                             p_value: float, y: float = None):
        """Add significance annotation (p-value bars)."""
        ax = self.axes[ax_idx] if isinstance(self.axes, np.ndarray) else self.axes

        if y is None:
            y = ax.get_ylim()[1] * 0.95

        # Draw bar
        ax.plot([x1, x1, x2, x2], [y, y*1.02, y*1.02, y],
               'k-', linewidth=1)

        # Add p-value
        if p_value < 0.001:
            text = '***'
        elif p_value < 0.01:
            text = '**'
        elif p_value < 0.05:
            text = '*'
        else:
            text = 'ns'

        ax.text((x1 + x2) / 2, y*1.03, text, ha='center', va='bottom', fontsize=8)

        return self

    def add_panel_label(self, ax_idx: int, label: str, x: float = -0.1, y: float = 1.05):
        """Add panel label (a, b, c, etc.)."""
        ax = self.axes[ax_idx] if isinstance(self.axes, np.ndarray) else self.axes
        ax.text(x, y, label, transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top', ha='right')
        return self

    def save(self, output: str, dpi: int = 300, format: str = None):
        """Save figure."""
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

        if format is None:
            format = os.path.splitext(output)[1][1:] or 'pdf'

        self.fig.tight_layout()
        self.fig.savefig(output, dpi=dpi, format=format, bbox_inches='tight')
        plt.close(self.fig)

        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate publication-ready figures')
    parser.add_argument('--data', required=True, help='CSV data file')
    parser.add_argument('--x', required=True, help='X column')
    parser.add_argument('--y', required=True, help='Y column')
    parser.add_argument('--plot', choices=['bar', 'line'], default='bar', help='Plot type')
    parser.add_argument('--style', choices=['nature', 'science', 'ieee'], default='nature')
    parser.add_argument('--output', '-o', default='figure.pdf', help='Output file')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output')
    parser.add_argument('--xlabel', help='X-axis label')
    parser.add_argument('--ylabel', help='Y-axis label')
    parser.add_argument('--title', help='Figure title')

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data)

    # Create figure
    fig_gen = PaperFigureGenerator(style=args.style)

    # Plot
    if args.plot == 'bar':
        fig_gen.plot_bar(data, args.x, args.y,
                        xlabel=args.xlabel, ylabel=args.ylabel, title=args.title)
    else:
        fig_gen.plot_line(data, args.x, args.y,
                         xlabel=args.xlabel, ylabel=args.ylabel, title=args.title)

    # Save
    fig_gen.save(args.output, dpi=args.dpi)
    print(f"✓ Figure saved to: {args.output}")
