#!/usr/bin/env python3
"""
Geo Visualizer - Create interactive maps with markers, heatmaps, and more.

Features:
- Markers with custom icons and popups
- Heatmaps for density visualization
- Choropleth maps for regional data
- Routes and paths
- Marker clustering
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class GeoVisualizer:
    """Create interactive maps using Folium."""

    TILE_LAYERS = {
        'openstreetmap': 'OpenStreetMap',
        'cartodb positron': 'CartoDB positron',
        'cartodb dark_matter': 'CartoDB dark_matter',
        'stamen terrain': 'Stamen Terrain',
        'stamen toner': 'Stamen Toner',
    }

    ICON_COLORS = [
        'red', 'blue', 'green', 'purple', 'orange', 'darkred',
        'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
        'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
        'gray', 'black', 'lightgray'
    ]

    def __init__(
        self,
        center: Optional[Tuple[float, float]] = None,
        zoom: int = 10,
        tiles: str = "OpenStreetMap"
    ):
        """
        Initialize map.

        Args:
            center: (lat, lon) center point
            zoom: Initial zoom level (1-18)
            tiles: Base map tiles
        """
        self._folium = None
        self._load_folium()

        self.center = center or (0, 0)
        self.zoom = zoom
        self.tiles = tiles
        self.map = None
        self.data = None
        self._markers = []
        self._layers = []
        self._init_map()

    def _load_folium(self):
        """Load folium library."""
        try:
            import folium
            self._folium = folium
        except ImportError:
            raise ImportError("folium required. Install with: pip install folium")

    def _init_map(self):
        """Initialize the map object."""
        self.map = self._folium.Map(
            location=self.center,
            zoom_start=self.zoom,
            tiles=self.tiles
        )

    def from_csv(
        self,
        filepath: str,
        lat_col: str,
        lon_col: str,
        **kwargs
    ) -> 'GeoVisualizer':
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV
            lat_col: Latitude column name
            lon_col: Longitude column name
            **kwargs: Additional pandas read_csv args

        Returns:
            self for chaining
        """
        import pandas as pd
        self.data = pd.read_csv(filepath, **kwargs)
        self._lat_col = lat_col
        self._lon_col = lon_col

        # Auto-center on data
        if self.center == (0, 0):
            self.center = (
                self.data[lat_col].mean(),
                self.data[lon_col].mean()
            )
            self._init_map()

        return self

    def from_dataframe(
        self,
        df,
        lat_col: str,
        lon_col: str
    ) -> 'GeoVisualizer':
        """
        Load data from pandas DataFrame.

        Args:
            df: DataFrame with location data
            lat_col: Latitude column name
            lon_col: Longitude column name

        Returns:
            self for chaining
        """
        self.data = df.copy()
        self._lat_col = lat_col
        self._lon_col = lon_col

        # Auto-center
        if self.center == (0, 0):
            self.center = (df[lat_col].mean(), df[lon_col].mean())
            self._init_map()

        return self

    def from_geojson(self, filepath: str) -> 'GeoVisualizer':
        """
        Load GeoJSON file.

        Args:
            filepath: Path to GeoJSON file

        Returns:
            self for chaining
        """
        with open(filepath) as f:
            self._geojson = json.load(f)
        return self

    def add_marker(
        self,
        lat: float,
        lon: float,
        popup: Optional[str] = None,
        tooltip: Optional[str] = None,
        icon: Optional[str] = None,
        color: str = "blue"
    ) -> 'GeoVisualizer':
        """
        Add a single marker.

        Args:
            lat: Latitude
            lon: Longitude
            popup: Popup content (HTML)
            tooltip: Hover tooltip
            icon: FontAwesome icon name (e.g., 'fa-coffee')
            color: Marker color

        Returns:
            self for chaining
        """
        marker_icon = None
        if icon:
            marker_icon = self._folium.Icon(color=color, icon=icon, prefix='fa')
        else:
            marker_icon = self._folium.Icon(color=color)

        marker = self._folium.Marker(
            location=[lat, lon],
            popup=popup,
            tooltip=tooltip,
            icon=marker_icon
        )
        marker.add_to(self.map)
        self._markers.append((lat, lon))
        return self

    def add_markers(
        self,
        locations: Optional[List[Dict]] = None,
        name_col: Optional[str] = None,
        popup_cols: Optional[List[str]] = None,
        color: str = "blue"
    ) -> 'GeoVisualizer':
        """
        Add multiple markers.

        Args:
            locations: List of dicts with lat/lon keys, or use loaded data
            name_col: Column for tooltip
            popup_cols: Columns to include in popup
            color: Marker color

        Returns:
            self for chaining
        """
        if locations is None and self.data is not None:
            # Use loaded data
            for _, row in self.data.iterrows():
                lat = row[self._lat_col]
                lon = row[self._lon_col]

                # Build popup
                popup = None
                if popup_cols:
                    popup_lines = []
                    for col in popup_cols:
                        if col in row:
                            popup_lines.append(f"<b>{col}:</b> {row[col]}")
                    popup = "<br>".join(popup_lines)

                tooltip = row.get(name_col) if name_col else None

                self.add_marker(lat, lon, popup=popup, tooltip=tooltip, color=color)
        else:
            # Use provided locations
            for loc in locations:
                lat = loc.get('lat') or loc.get('latitude')
                lon = loc.get('lon') or loc.get('lng') or loc.get('longitude')
                popup = loc.get('popup') or loc.get('name')
                tooltip = loc.get('tooltip') or loc.get('name')
                self.add_marker(lat, lon, popup=popup, tooltip=tooltip, color=color)

        return self

    def cluster_markers(self, enabled: bool = True) -> 'GeoVisualizer':
        """
        Enable marker clustering for dense data.

        Args:
            enabled: Whether to cluster

        Returns:
            self for chaining
        """
        if enabled:
            try:
                from folium.plugins import MarkerCluster
                self._marker_cluster = MarkerCluster().add_to(self.map)
            except ImportError:
                print("Warning: MarkerCluster not available")
        return self

    def add_heatmap(
        self,
        points: Optional[List[Tuple[float, float]]] = None,
        weight_col: Optional[str] = None,
        radius: int = 15,
        blur: int = 10,
        max_zoom: int = 12
    ) -> 'GeoVisualizer':
        """
        Add heatmap layer.

        Args:
            points: List of (lat, lon) or (lat, lon, weight)
            weight_col: Column for weights if using loaded data
            radius: Heat point radius
            blur: Blur amount
            max_zoom: Max zoom for heatmap

        Returns:
            self for chaining
        """
        try:
            from folium.plugins import HeatMap
        except ImportError:
            raise ImportError("folium.plugins required for heatmap")

        if points is None and self.data is not None:
            if weight_col:
                heat_data = [
                    [row[self._lat_col], row[self._lon_col], row[weight_col]]
                    for _, row in self.data.iterrows()
                ]
            else:
                heat_data = [
                    [row[self._lat_col], row[self._lon_col]]
                    for _, row in self.data.iterrows()
                ]
        else:
            heat_data = points

        HeatMap(
            heat_data,
            radius=radius,
            blur=blur,
            max_zoom=max_zoom
        ).add_to(self.map)

        return self

    def add_choropleth(
        self,
        geojson: str,
        data,
        key_on: str,
        value_col: str,
        fill_color: str = "YlOrRd",
        fill_opacity: float = 0.7,
        line_opacity: float = 0.2,
        legend_name: Optional[str] = None
    ) -> 'GeoVisualizer':
        """
        Add choropleth layer.

        Args:
            geojson: Path to GeoJSON file
            data: DataFrame with values
            key_on: GeoJSON property to match
            value_col: Data column for coloring
            fill_color: Color scale
            fill_opacity: Fill opacity
            line_opacity: Border opacity
            legend_name: Legend title

        Returns:
            self for chaining
        """
        import pandas as pd

        if isinstance(data, str):
            data = pd.read_csv(data)

        self._folium.Choropleth(
            geo_data=geojson,
            data=data,
            columns=[data.columns[0], value_col],
            key_on=key_on,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            line_opacity=line_opacity,
            legend_name=legend_name or value_col
        ).add_to(self.map)

        return self

    def add_route(
        self,
        points: List[Tuple[float, float]],
        color: str = "blue",
        weight: int = 3,
        opacity: float = 1.0
    ) -> 'GeoVisualizer':
        """
        Add route/path line.

        Args:
            points: List of (lat, lon) points
            color: Line color
            weight: Line width
            opacity: Line opacity

        Returns:
            self for chaining
        """
        self._folium.PolyLine(
            points,
            color=color,
            weight=weight,
            opacity=opacity
        ).add_to(self.map)

        return self

    def add_circle(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        color: str = "blue",
        fill: bool = True,
        fill_opacity: float = 0.3,
        popup: Optional[str] = None
    ) -> 'GeoVisualizer':
        """
        Add circle area.

        Args:
            lat: Center latitude
            lon: Center longitude
            radius_m: Radius in meters
            color: Circle color
            fill: Whether to fill
            fill_opacity: Fill opacity
            popup: Popup text

        Returns:
            self for chaining
        """
        self._folium.Circle(
            location=[lat, lon],
            radius=radius_m,
            color=color,
            fill=fill,
            fill_opacity=fill_opacity,
            popup=popup
        ).add_to(self.map)

        return self

    def fit_bounds(self) -> 'GeoVisualizer':
        """
        Fit map to show all markers.

        Returns:
            self for chaining
        """
        if self._markers:
            self.map.fit_bounds(self._markers)
        elif self.data is not None:
            bounds = [
                [self.data[self._lat_col].min(), self.data[self._lon_col].min()],
                [self.data[self._lat_col].max(), self.data[self._lon_col].max()]
            ]
            self.map.fit_bounds(bounds)
        return self

    def add_layer_control(self) -> 'GeoVisualizer':
        """
        Add layer control widget.

        Returns:
            self for chaining
        """
        self._folium.LayerControl().add_to(self.map)
        return self

    def save(self, filepath: str) -> str:
        """
        Save map to HTML file.

        Args:
            filepath: Output path

        Returns:
            Path to saved file
        """
        self.map.save(filepath)
        return filepath

    def get_html(self) -> str:
        """Get map as HTML string."""
        return self.map._repr_html_()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Create interactive maps')
    parser.add_argument('--input', '-i', help='Input CSV file')
    parser.add_argument('--lat', default='lat', help='Latitude column')
    parser.add_argument('--lon', default='lon', help='Longitude column')
    parser.add_argument('--output', '-o', default='map.html', help='Output HTML file')
    parser.add_argument('--heatmap', action='store_true', help='Add heatmap layer')
    parser.add_argument('--weight', help='Weight column for heatmap')
    parser.add_argument('--cluster', action='store_true', help='Cluster markers')
    parser.add_argument('--popup', nargs='+', help='Columns for popup')
    parser.add_argument('--tooltip', help='Column for tooltip')
    parser.add_argument('--tiles', default='OpenStreetMap', help='Base map tiles')
    parser.add_argument('--geojson', help='GeoJSON file for choropleth')
    parser.add_argument('--data', help='Data CSV for choropleth')
    parser.add_argument('--key', help='GeoJSON key property')
    parser.add_argument('--value', help='Value column for choropleth')
    parser.add_argument('--color', default='blue', help='Marker color')

    args = parser.parse_args()

    viz = GeoVisualizer(tiles=args.tiles)

    if args.geojson and args.data:
        # Choropleth mode
        viz.add_choropleth(
            geojson=args.geojson,
            data=args.data,
            key_on=f"feature.properties.{args.key}",
            value_col=args.value
        )
    elif args.input:
        viz.from_csv(args.input, lat_col=args.lat, lon_col=args.lon)

        if args.heatmap:
            viz.add_heatmap(weight_col=args.weight)
        else:
            if args.cluster:
                viz.cluster_markers(True)
            viz.add_markers(
                popup_cols=args.popup,
                name_col=args.tooltip,
                color=args.color
            )

        viz.fit_bounds()
    else:
        parser.print_help()
        return

    viz.save(args.output)
    print(f"Map saved to: {args.output}")


if __name__ == "__main__":
    main()
