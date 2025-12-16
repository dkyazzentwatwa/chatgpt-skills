#!/usr/bin/env python3
"""
Distance Calculator - Calculate geographic distances and find nearby points.

Features:
- Point-to-point distance (Haversine, Vincenty)
- Distance matrix
- Nearest neighbor search
- Radius search
- Multiple units
"""

import argparse
import csv
import math
from typing import Dict, List, Optional, Tuple, Union


class DistanceCalculator:
    """Calculate distances between geographic coordinates."""

    # Earth radius in km
    EARTH_RADIUS_KM = 6371.0

    # Unit conversion factors (to km)
    UNITS = {
        'km': 1.0,
        'miles': 0.621371,
        'm': 1000.0,
        'meters': 1000.0,
        'nm': 0.539957,  # nautical miles
        'ft': 3280.84,
        'feet': 3280.84,
    }

    def __init__(self, unit: str = "km", method: str = "haversine"):
        """
        Initialize calculator.

        Args:
            unit: Output unit (km, miles, m, nm, ft)
            method: Distance method (haversine, vincenty)
        """
        self.unit = unit.lower()
        self.method = method.lower()

        if self.unit not in self.UNITS:
            raise ValueError(f"Unknown unit: {unit}. Use: {list(self.UNITS.keys())}")
        if self.method not in ('haversine', 'vincenty'):
            raise ValueError(f"Unknown method: {method}. Use: haversine, vincenty")

    def distance(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """
        Calculate distance between two points.

        Args:
            point1: (lat, lon) first point
            point2: (lat, lon) second point

        Returns:
            Distance in configured unit
        """
        if self.method == 'haversine':
            dist_km = self._haversine(point1, point2)
        else:
            dist_km = self._vincenty(point1, point2)

        return self._convert_from_km(dist_km)

    def _haversine(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """Calculate haversine distance in km."""
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return self.EARTH_RADIUS_KM * c

    def _vincenty(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """Calculate Vincenty distance in km (more accurate)."""
        try:
            from geopy.distance import geodesic
            return geodesic(point1, point2).kilometers
        except ImportError:
            # Fallback to haversine
            return self._haversine(point1, point2)

    def _convert_from_km(self, km: float) -> float:
        """Convert km to configured unit."""
        return km * self.UNITS[self.unit]

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert between units.

        Args:
            value: Distance value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value
        """
        # Convert to km first
        km = value / self.UNITS[from_unit.lower()]
        # Convert to target
        return km * self.UNITS[to_unit.lower()]

    def distance_with_details(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> Dict:
        """
        Get distance with full details.

        Args:
            point1: (lat, lon) first point
            point2: (lat, lon) second point

        Returns:
            Dict with distance and metadata
        """
        dist = self.distance(point1, point2)
        return {
            'distance': dist,
            'unit': self.unit,
            'from': {'lat': point1[0], 'lon': point1[1]},
            'to': {'lat': point2[0], 'lon': point2[1]},
            'method': self.method
        }

    def distance_matrix(
        self,
        points: List[Tuple[float, float]]
    ) -> List[List[float]]:
        """
        Calculate all pairwise distances.

        Args:
            points: List of (lat, lon) tuples

        Returns:
            NxN distance matrix
        """
        n = len(points)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                dist = self.distance(points[i], points[j])
                matrix[i][j] = dist
                matrix[j][i] = dist

        return matrix

    def distances_from_origin(
        self,
        origin: Tuple[float, float],
        points: List[Tuple[float, float]]
    ) -> List[float]:
        """
        Calculate distances from origin to all points.

        Args:
            origin: (lat, lon) origin point
            points: List of destination points

        Returns:
            List of distances
        """
        return [self.distance(origin, p) for p in points]

    def find_nearest(
        self,
        origin: Tuple[float, float],
        points: List[Union[Tuple[float, float], Dict]],
        n: int = 1
    ) -> List[Dict]:
        """
        Find nearest N points to origin.

        Args:
            origin: (lat, lon) origin point
            points: List of points or dicts with lat/lon
            n: Number of nearest to return

        Returns:
            List of nearest points with distances
        """
        results = []

        for p in points:
            if isinstance(p, dict):
                lat = p.get('lat') or p.get('latitude')
                lon = p.get('lon') or p.get('lng') or p.get('longitude')
                point = (lat, lon)
                data = p
            else:
                point = p
                data = {}

            dist = self.distance(origin, point)
            results.append({
                'point': point,
                'distance': dist,
                'data': data
            })

        # Sort by distance
        results.sort(key=lambda x: x['distance'])
        return results[:n]

    def find_within_radius(
        self,
        origin: Tuple[float, float],
        points: List[Union[Tuple[float, float], Dict]],
        radius: float
    ) -> List[Dict]:
        """
        Find all points within radius of origin.

        Args:
            origin: (lat, lon) origin point
            points: List of points
            radius: Search radius in configured unit

        Returns:
            List of points within radius
        """
        results = []

        for p in points:
            if isinstance(p, dict):
                lat = p.get('lat') or p.get('latitude')
                lon = p.get('lon') or p.get('lng') or p.get('longitude')
                point = (lat, lon)
                data = p
            else:
                point = p
                data = {}

            dist = self.distance(origin, point)
            if dist <= radius:
                results.append({
                    'point': point,
                    'distance': dist,
                    'data': data
                })

        results.sort(key=lambda x: x['distance'])
        return results

    def from_csv(
        self,
        filepath: str,
        lat_col: str = 'lat',
        lon_col: str = 'lon'
    ) -> List[Dict]:
        """
        Load points from CSV.

        Args:
            filepath: Path to CSV
            lat_col: Latitude column name
            lon_col: Longitude column name

        Returns:
            List of point dicts
        """
        points = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['lat'] = float(row[lat_col])
                row['lon'] = float(row[lon_col])
                points.append(row)
        return points

    def matrix_to_csv(
        self,
        matrix: List[List[float]],
        labels: List[str],
        output: str
    ) -> str:
        """
        Save distance matrix to CSV.

        Args:
            matrix: Distance matrix
            labels: Point labels
            output: Output file path

        Returns:
            Path to saved file
        """
        with open(output, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([''] + labels)
            # Rows
            for i, row in enumerate(matrix):
                writer.writerow([labels[i]] + [f"{d:.2f}" for d in row])

        return output


def parse_point(s: str) -> Tuple[float, float]:
    """Parse 'lat,lon' string to tuple."""
    parts = s.split(',')
    return (float(parts[0].strip()), float(parts[1].strip()))


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Calculate geographic distances')
    parser.add_argument('--from', dest='from_point', help='Origin point (lat,lon)')
    parser.add_argument('--to', help='Destination point (lat,lon)')
    parser.add_argument('--origin', help='Origin for search operations (lat,lon)')
    parser.add_argument('--input', '-i', help='Input CSV file')
    parser.add_argument('--lat', default='lat', help='Latitude column')
    parser.add_argument('--lon', default='lon', help='Longitude column')
    parser.add_argument('--nearest', type=int, help='Find N nearest points')
    parser.add_argument('--radius', type=float, help='Search radius')
    parser.add_argument('--matrix', action='store_true', help='Calculate distance matrix')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--unit', default='km', choices=['km', 'miles', 'm', 'nm', 'ft'])
    parser.add_argument('--method', default='haversine', choices=['haversine', 'vincenty'])

    args = parser.parse_args()
    calc = DistanceCalculator(unit=args.unit, method=args.method)

    # Point-to-point distance
    if args.from_point and args.to:
        p1 = parse_point(args.from_point)
        p2 = parse_point(args.to)
        result = calc.distance_with_details(p1, p2)
        print(f"Distance: {result['distance']:.2f} {result['unit']}")
        print(f"From: ({p1[0]}, {p1[1]})")
        print(f"To: ({p2[0]}, {p2[1]})")
        print(f"Method: {result['method']}")

    # Search operations
    elif args.origin and args.input:
        origin = parse_point(args.origin)
        points = calc.from_csv(args.input, args.lat, args.lon)

        if args.nearest:
            results = calc.find_nearest(origin, points, args.nearest)
            print(f"Nearest {args.nearest} points:")
            for i, r in enumerate(results, 1):
                name = r['data'].get('name', f"Point {i}")
                print(f"  {i}. {name}: {r['distance']:.2f} {args.unit}")

        elif args.radius:
            results = calc.find_within_radius(origin, points, args.radius)
            print(f"Points within {args.radius} {args.unit}:")
            for r in results:
                name = r['data'].get('name', str(r['point']))
                print(f"  {name}: {r['distance']:.2f} {args.unit}")
            print(f"\nTotal: {len(results)} points")

    # Distance matrix
    elif args.input and args.matrix:
        points_data = calc.from_csv(args.input, args.lat, args.lon)
        points = [(p['lat'], p['lon']) for p in points_data]
        labels = [p.get('name', f"P{i}") for i, p in enumerate(points_data)]

        matrix = calc.distance_matrix(points)

        if args.output:
            calc.matrix_to_csv(matrix, labels, args.output)
            print(f"Matrix saved to: {args.output}")
        else:
            # Print matrix
            print(f"Distance Matrix ({args.unit}):")
            print("\t" + "\t".join(labels))
            for i, row in enumerate(matrix):
                print(f"{labels[i]}\t" + "\t".join(f"{d:.1f}" for d in row))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
