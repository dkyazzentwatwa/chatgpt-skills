#!/usr/bin/env python3
"""
Gantt Chart Creator - Create project timeline Gantt charts

Features:
- Multiple input sources (CSV, dict, programmatic)
- Task dependencies and milestones
- Progress tracking
- Color coding by category/assignee
- Static (matplotlib) and interactive (plotly) output
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

try:
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Default colors
DEFAULT_COLORS = [
    '#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12',
    '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b'
]

DEFAULT_PROGRESS_COLOR = '#27ae60'
DEFAULT_MILESTONE_COLOR = '#e74c3c'
DEFAULT_TODAY_COLOR = '#e74c3c'


@dataclass
class Task:
    """Represents a task in the Gantt chart."""
    name: str
    start: datetime
    end: datetime
    progress: float = 0
    category: Optional[str] = None
    assignee: Optional[str] = None
    color: Optional[str] = None


@dataclass
class Milestone:
    """Represents a milestone marker."""
    name: str
    date: datetime
    color: str = DEFAULT_MILESTONE_COLOR


@dataclass
class Dependency:
    """Represents a task dependency."""
    from_task: str
    to_task: str
    dep_type: str = 'FS'  # FS, SS, FF, SF


class GanttChartCreator:
    """
    Create Gantt charts for project timelines.

    Example:
        gantt = GanttChartCreator()
        gantt.add_task("Design", "2024-01-01", "2024-01-14")
        gantt.generate().save("gantt.png")
    """

    def __init__(self):
        """Initialize Gantt chart creator."""
        self._tasks: List[Task] = []
        self._milestones: List[Milestone] = []
        self._dependencies: List[Dependency] = []

        self._figure = None
        self._is_interactive = False

        # Display options
        self._title: Optional[str] = None
        self._title_font_size: int = 14
        self._show_progress: bool = False
        self._show_today_line: bool = False
        self._today_color: str = DEFAULT_TODAY_COLOR
        self._today_label: str = 'Today'
        self._grid_style: str = 'weekly'

        # Color options
        self._color_by: Optional[str] = None  # 'category', 'assignee'
        self._color_map: Dict[str, str] = {}

        # Date range (auto-calculated if not set)
        self._date_start: Optional[datetime] = None
        self._date_end: Optional[datetime] = None

        # Size
        self._width: float = 12
        self._height: float = 6

    def _parse_date(self, date_str: Union[str, datetime]) -> datetime:
        """Parse date string to datetime."""
        if isinstance(date_str, datetime):
            return date_str

        formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        raise ValueError(f"Could not parse date: {date_str}")

    def from_csv(
        self,
        filepath: str,
        task: str,
        start: str,
        end: str,
        progress: str = None,
        category: str = None,
        assignee: str = None
    ) -> 'GanttChartCreator':
        """
        Load tasks from CSV file.

        Args:
            filepath: Path to CSV file
            task: Task name column
            start: Start date column
            end: End date column
            progress: Optional progress % column
            category: Optional category column
            assignee: Optional assignee column

        Returns:
            Self for chaining
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)

        for _, row in df.iterrows():
            task_obj = Task(
                name=str(row[task]),
                start=self._parse_date(str(row[start])),
                end=self._parse_date(str(row[end])),
                progress=float(row[progress]) if progress and pd.notna(row.get(progress)) else 0,
                category=str(row[category]) if category and pd.notna(row.get(category)) else None,
                assignee=str(row[assignee]) if assignee and pd.notna(row.get(assignee)) else None
            )
            self._tasks.append(task_obj)

        return self

    def from_tasks(self, tasks: List[Dict[str, Any]]) -> 'GanttChartCreator':
        """
        Load tasks from list of dictionaries.

        Args:
            tasks: List of task dicts with 'name', 'start', 'end' keys

        Returns:
            Self for chaining
        """
        for t in tasks:
            task_obj = Task(
                name=t['name'],
                start=self._parse_date(t['start']),
                end=self._parse_date(t['end']),
                progress=t.get('progress', 0),
                category=t.get('category'),
                assignee=t.get('assignee')
            )
            self._tasks.append(task_obj)

        return self

    def add_task(
        self,
        name: str,
        start: str,
        end: str,
        progress: float = 0,
        category: str = None,
        assignee: str = None
    ) -> 'GanttChartCreator':
        """
        Add a single task.

        Args:
            name: Task name
            start: Start date
            end: End date
            progress: Completion percentage (0-100)
            category: Optional category
            assignee: Optional assignee

        Returns:
            Self for chaining
        """
        task = Task(
            name=name,
            start=self._parse_date(start),
            end=self._parse_date(end),
            progress=progress,
            category=category,
            assignee=assignee
        )
        self._tasks.append(task)
        return self

    def add_milestone(
        self,
        name: str,
        date: str,
        color: str = None
    ) -> 'GanttChartCreator':
        """
        Add a milestone marker.

        Args:
            name: Milestone name
            date: Milestone date
            color: Optional color

        Returns:
            Self for chaining
        """
        milestone = Milestone(
            name=name,
            date=self._parse_date(date),
            color=color or DEFAULT_MILESTONE_COLOR
        )
        self._milestones.append(milestone)
        return self

    def add_dependency(
        self,
        from_task: str,
        to_task: str,
        dep_type: str = 'FS'
    ) -> 'GanttChartCreator':
        """
        Add task dependency.

        Args:
            from_task: Predecessor task name
            to_task: Successor task name
            dep_type: Dependency type (FS, SS, FF, SF)

        Returns:
            Self for chaining
        """
        dependency = Dependency(from_task, to_task, dep_type)
        self._dependencies.append(dependency)
        return self

    def title(self, text: str, font_size: int = 14) -> 'GanttChartCreator':
        """
        Set chart title.

        Args:
            text: Title text
            font_size: Font size

        Returns:
            Self for chaining
        """
        self._title = text
        self._title_font_size = font_size
        return self

    def color_by(
        self,
        field: str,
        colors: Dict[str, str] = None
    ) -> 'GanttChartCreator':
        """
        Set color coding.

        Args:
            field: 'category' or 'assignee'
            colors: Optional dict mapping values to colors

        Returns:
            Self for chaining
        """
        self._color_by = field
        if colors:
            self._color_map = colors
        return self

    def show_progress(self, show: bool = True) -> 'GanttChartCreator':
        """
        Show progress bars.

        Args:
            show: Whether to show progress

        Returns:
            Self for chaining
        """
        self._show_progress = show
        return self

    def show_today(
        self,
        show: bool = True,
        color: str = None,
        label: str = None
    ) -> 'GanttChartCreator':
        """
        Show today marker line.

        Args:
            show: Whether to show today line
            color: Line color
            label: Line label

        Returns:
            Self for chaining
        """
        self._show_today_line = show
        if color:
            self._today_color = color
        if label:
            self._today_label = label
        return self

    def date_range(self, start: str, end: str) -> 'GanttChartCreator':
        """
        Set date range.

        Args:
            start: Start date
            end: End date

        Returns:
            Self for chaining
        """
        self._date_start = self._parse_date(start)
        self._date_end = self._parse_date(end)
        return self

    def grid(self, show: bool = True, style: str = 'weekly') -> 'GanttChartCreator':
        """
        Configure grid lines.

        Args:
            show: Whether to show grid
            style: 'daily', 'weekly', or 'monthly'

        Returns:
            Self for chaining
        """
        self._grid_style = style if show else None
        return self

    def size(self, width: float, height: float) -> 'GanttChartCreator':
        """
        Set chart size.

        Args:
            width: Width in inches
            height: Height in inches

        Returns:
            Self for chaining
        """
        self._width = width
        self._height = height
        return self

    def _get_task_color(self, task: Task, index: int) -> str:
        """Get color for a task."""
        if task.color:
            return task.color

        if self._color_by == 'category' and task.category:
            if task.category in self._color_map:
                return self._color_map[task.category]
            # Auto-assign color
            categories = list(set(t.category for t in self._tasks if t.category))
            idx = categories.index(task.category) if task.category in categories else 0
            return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

        if self._color_by == 'assignee' and task.assignee:
            if task.assignee in self._color_map:
                return self._color_map[task.assignee]
            assignees = list(set(t.assignee for t in self._tasks if t.assignee))
            idx = assignees.index(task.assignee) if task.assignee in assignees else 0
            return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

        return DEFAULT_COLORS[index % len(DEFAULT_COLORS)]

    def _get_date_range(self) -> Tuple[datetime, datetime]:
        """Calculate date range from tasks."""
        if self._date_start and self._date_end:
            return self._date_start, self._date_end

        all_dates = []
        for task in self._tasks:
            all_dates.extend([task.start, task.end])
        for milestone in self._milestones:
            all_dates.append(milestone.date)

        if not all_dates:
            today = datetime.now()
            return today, today + timedelta(days=30)

        min_date = min(all_dates) - timedelta(days=2)
        max_date = max(all_dates) + timedelta(days=2)

        return min_date, max_date

    def generate(self, interactive: bool = False) -> 'GanttChartCreator':
        """
        Generate the Gantt chart.

        Args:
            interactive: Generate interactive plotly chart

        Returns:
            Self for chaining
        """
        if not self._tasks:
            raise ValueError("No tasks provided")

        self._is_interactive = interactive

        if interactive and HAS_PLOTLY:
            self._generate_plotly()
        else:
            self._generate_matplotlib()

        return self

    def _generate_matplotlib(self) -> None:
        """Generate static matplotlib chart."""
        fig, ax = plt.subplots(figsize=(self._width, self._height))

        date_start, date_end = self._get_date_range()

        # Reverse tasks for top-to-bottom display
        tasks = list(reversed(self._tasks))
        y_positions = range(len(tasks))

        bar_height = 0.6

        for i, task in enumerate(tasks):
            color = self._get_task_color(task, len(tasks) - 1 - i)

            # Draw task bar
            duration = (task.end - task.start).days + 1
            ax.barh(
                i, duration,
                left=mdates.date2num(task.start),
                height=bar_height,
                color=color,
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )

            # Draw progress bar
            if self._show_progress and task.progress > 0:
                progress_duration = duration * (task.progress / 100)
                ax.barh(
                    i, progress_duration,
                    left=mdates.date2num(task.start),
                    height=bar_height,
                    color=DEFAULT_PROGRESS_COLOR,
                    alpha=0.9
                )

        # Add milestones
        for milestone in self._milestones:
            ax.scatter(
                mdates.date2num(milestone.date),
                -0.5,  # Below tasks
                marker='D',
                s=100,
                color=milestone.color,
                zorder=5
            )
            ax.annotate(
                milestone.name,
                (mdates.date2num(milestone.date), -0.5),
                xytext=(0, -15),
                textcoords='offset points',
                ha='center',
                fontsize=8
            )

        # Today line
        if self._show_today_line:
            today = datetime.now()
            if date_start <= today <= date_end:
                ax.axvline(
                    mdates.date2num(today),
                    color=self._today_color,
                    linestyle='--',
                    linewidth=1.5,
                    label=self._today_label
                )

        # Configure axes
        ax.set_yticks(y_positions)
        ax.set_yticklabels([t.name for t in tasks])
        ax.set_xlim(mdates.date2num(date_start), mdates.date2num(date_end))

        # Date formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

        plt.xticks(rotation=45, ha='right')

        # Grid
        if self._grid_style:
            ax.grid(True, axis='x', alpha=0.3)

        # Title
        if self._title:
            ax.set_title(self._title, fontsize=self._title_font_size, pad=10)

        # Legend for today line
        if self._show_today_line:
            ax.legend(loc='upper right')

        plt.tight_layout()
        self._figure = fig

    def _generate_plotly(self) -> None:
        """Generate interactive plotly chart."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for interactive charts")

        df_data = []
        for i, task in enumerate(self._tasks):
            color = self._get_task_color(task, i)
            df_data.append({
                'Task': task.name,
                'Start': task.start.strftime('%Y-%m-%d'),
                'Finish': task.end.strftime('%Y-%m-%d'),
                'Resource': task.category or task.assignee or 'Task',
                'Progress': task.progress,
                'Color': color
            })

        df = pd.DataFrame(df_data)

        # Create Gantt chart
        fig = ff.create_gantt(
            df,
            colors={row['Resource']: row['Color'] for _, row in df.iterrows()},
            index_col='Resource',
            show_colorbar=True,
            showgrid_x=True,
            showgrid_y=True
        )

        # Add today line
        if self._show_today_line:
            today = datetime.now().strftime('%Y-%m-%d')
            fig.add_vline(
                x=today,
                line_dash='dash',
                line_color=self._today_color,
                annotation_text=self._today_label
            )

        # Update layout
        fig.update_layout(
            title=self._title,
            xaxis_title='Date',
            height=max(400, len(self._tasks) * 40),
            font=dict(size=12)
        )

        self._figure = fig

    def get_figure(self):
        """Get the figure object."""
        if self._figure is None:
            self.generate()
        return self._figure

    def save(self, path: str, dpi: int = 150) -> str:
        """
        Save chart to file.

        Args:
            path: Output file path
            dpi: Image resolution (for PNG)

        Returns:
            Path to saved file
        """
        if self._figure is None:
            self.generate()

        output_path = Path(path)
        ext = output_path.suffix.lower()

        if self._is_interactive and HAS_PLOTLY:
            if ext == '.html':
                self._figure.write_html(str(output_path))
            else:
                self._figure.write_image(str(output_path))
        else:
            self._figure.savefig(str(output_path), dpi=dpi, bbox_inches='tight')
            plt.close(self._figure)

        return str(output_path)

    def show(self) -> None:
        """Display chart."""
        if self._figure is None:
            self.generate()

        if self._is_interactive and HAS_PLOTLY:
            self._figure.show()
        else:
            plt.show()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Create Gantt charts for project timelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gantt_creator.py --input tasks.csv --task name --start start --end end --output gantt.png
  python gantt_creator.py --input project.csv --task task --start begin --end finish --progress pct --title "Project" --output timeline.png
  python gantt_creator.py --input tasks.csv --task name --start start --end end --interactive --output gantt.html
        """
    )

    # Input options
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--task', '-t', required=True, help='Task name column')
    parser.add_argument('--start', '-s', required=True, help='Start date column')
    parser.add_argument('--end', '-e', required=True, help='End date column')
    parser.add_argument('--progress', '-p', help='Progress % column')
    parser.add_argument('--category', '-c', help='Category column')
    parser.add_argument('--assignee', '-a', help='Assignee column')

    # Output options
    parser.add_argument('--output', '-o', default='gantt.png',
                        help='Output file path (default: gantt.png)')

    # Display options
    parser.add_argument('--title', help='Chart title')
    parser.add_argument('--interactive', action='store_true',
                        help='Generate interactive HTML')
    parser.add_argument('--show-today', action='store_true',
                        help='Show today marker')
    parser.add_argument('--show-progress', action='store_true',
                        help='Show progress bars')

    # Size options
    parser.add_argument('--width', type=float, default=12,
                        help='Chart width (default: 12)')
    parser.add_argument('--height', type=float, default=6,
                        help='Chart height (default: 6)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Image DPI (default: 150)')

    args = parser.parse_args()

    try:
        gantt = GanttChartCreator()
        gantt.from_csv(
            args.input,
            task=args.task,
            start=args.start,
            end=args.end,
            progress=args.progress,
            category=args.category,
            assignee=args.assignee
        )

        if args.title:
            gantt.title(args.title)

        if args.category:
            gantt.color_by('category')
        elif args.assignee:
            gantt.color_by('assignee')

        if args.show_progress:
            gantt.show_progress(True)

        if args.show_today:
            gantt.show_today(True)

        gantt.size(args.width, args.height)
        gantt.generate(interactive=args.interactive)
        output_path = gantt.save(args.output, dpi=args.dpi)
        print(f"Gantt chart saved to: {output_path}")

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
