#!/usr/bin/env python3
"""Territory Mapper - Visualize territories on interactive maps."""

import argparse
import json
import os
from typing import List, Tuple, Dict, Any
import folium
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd


class TerritoryMapper:
    """Create interactive territory maps."""

    def __init__(self, center: Tuple[float, float] = None, zoom: int = 10):
        """Initialize map."""
        self.center = center or (37.7749, -122.4194)  # San Francisco default
        self.zoom = zoom
        self.map = folium.Map(location=self.center, zoom_start=self.zoom)
        self.territories = []

    def add_territory(self, name: str, coordinates: List[Tuple[float, float]],
                     color: str = 'blue', opacity: float = 0.5,
                     data: Dict[str, Any] = None):
        """
        Add territory polygon to map.

        Args:
            name: Territory name
            coordinates: List of (lat, lon) tuples defining polygon
            color: Fill color
            opacity: Fill opacity (0-1)
            data: Additional data for tooltip
        """
        # Create polygon
        polygon = Polygon([(lon, lat) for lat, lon in coordinates])

        # Create popup text
        popup_text = f"<b>{name}</b><br>"
        if data:
            for key, value in data.items():
                popup_text += f"{key}: {value}<br>"

        # Add to map
        folium.Polygon(
            locations=coordinates,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=opacity,
            popup=popup_text,
            tooltip=name
        ).add_to(self.map)

        self.territories.append({
            'name': name,
            'geometry': polygon,
            'color': color,
            'data': data or {}
        })

        return self

    def load_geojson(self, filepath: str, name_column: str = 'name',
                    color_column: str = None):
        """Load territories from GeoJSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GeoJSON not found: {filepath}")

        gdf = gpd.read_file(filepath)

        # Color mapping
        if color_column and color_column in gdf.columns:
            # Generate colors based on unique values
            unique_vals = gdf[color_column].unique()
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred',
                     'lightred', 'beige', 'darkblue', 'darkgreen']
            color_map = {val: colors[i % len(colors)]
                        for i, val in enumerate(unique_vals)}
        else:
            color_map = None

        # Add each territory
        for idx, row in gdf.iterrows():
            name = row[name_column] if name_column in gdf.columns else f"Territory {idx}"

            # Get coordinates
            if row.geometry.geom_type == 'Polygon':
                coords = [(y, x) for x, y in row.geometry.exterior.coords]
            else:
                continue  # Skip non-polygons

            # Determine color
            if color_map:
                color = color_map[row[color_column]]
            else:
                color = 'blue'

            # Get additional data
            data = {k: v for k, v in row.items()
                   if k not in ['geometry', name_column]}

            self.add_territory(name, coords, color=color, data=data)

        # Auto-fit bounds
        self.map.fit_bounds(gdf.total_bounds[[1, 0, 3, 2]].reshape(2, 2).tolist())

        return self

    def save_html(self, output: str):
        """Save map as HTML."""
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        self.map.save(output)
        return output

    def add_markers(self, locations: List[Tuple[float, float, str]]):
        """Add markers to map (lat, lon, label)."""
        for lat, lon, label in locations:
            folium.Marker(
                location=(lat, lon),
                popup=label,
                tooltip=label
            ).add_to(self.map)
        return self


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize territories on interactive maps')
    parser.add_argument('--geojson', help='GeoJSON file with territories')
    parser.add_argument('--name-column', default='name', help='Column for territory names')
    parser.add_argument('--color-by', help='Column to color territories by')
    parser.add_argument('--output', '-o', default='territories.html', help='Output HTML file')
    parser.add_argument('--center', help='Map center as "lat,lon"')
    parser.add_argument('--zoom', type=int, default=10, help='Initial zoom level')

    args = parser.parse_args()

    # Parse center
    if args.center:
        lat, lon = map(float, args.center.split(','))
        center = (lat, lon)
    else:
        center = None

    mapper = TerritoryMapper(center=center, zoom=args.zoom)

    if args.geojson:
        print(f"Loading territories from {args.geojson}...")
        mapper.load_geojson(
            args.geojson,
            name_column=args.name_column,
            color_column=args.color_by
        )
        print(f"Loaded {len(mapper.territories)} territories")

    mapper.save_html(args.output)
    print(f"✓ Map saved to: {args.output}")
