#!/usr/bin/env python3
"""
Geocoder - Convert between addresses and coordinates.

Features:
- Geocoding (address to coordinates)
- Reverse geocoding (coordinates to address)
- Batch processing
- Multiple providers
- CSV file operations
"""

import argparse
import csv
import time
from typing import Dict, List, Optional, Tuple


class Geocoder:
    """Geocoding and reverse geocoding operations."""

    PROVIDERS = {
        'nominatim': 'Nominatim',
        'google': 'GoogleV3',
        'bing': 'Bing',
        'arcgis': 'ArcGIS',
    }

    def __init__(
        self,
        provider: str = "nominatim",
        api_key: Optional[str] = None,
        user_agent: str = "geocoder-skill"
    ):
        """
        Initialize geocoder.

        Args:
            provider: Geocoding provider
            api_key: API key for paid providers
            user_agent: User agent for Nominatim
        """
        self._geopy = None
        self._load_geopy()

        self.provider = provider.lower()
        self.api_key = api_key
        self.user_agent = user_agent
        self._geocoder = self._init_geocoder()

    def _load_geopy(self):
        """Load geopy library."""
        try:
            import geopy
            self._geopy = geopy
        except ImportError:
            raise ImportError("geopy required. Install with: pip install geopy")

    def _init_geocoder(self):
        """Initialize the geocoder for the selected provider."""
        from geopy.geocoders import Nominatim, GoogleV3, Bing, ArcGIS

        if self.provider == 'nominatim':
            return Nominatim(user_agent=self.user_agent)
        elif self.provider == 'google':
            if not self.api_key:
                raise ValueError("Google provider requires api_key")
            return GoogleV3(api_key=self.api_key)
        elif self.provider == 'bing':
            if not self.api_key:
                raise ValueError("Bing provider requires api_key")
            return Bing(api_key=self.api_key)
        elif self.provider == 'arcgis':
            return ArcGIS()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def geocode(self, address: str) -> Optional[Dict]:
        """
        Convert address to coordinates.

        Args:
            address: Address string

        Returns:
            Dict with lat, lon, components or None
        """
        try:
            location = self._geocoder.geocode(address, addressdetails=True)
            if location is None:
                return None

            result = {
                'address': location.address,
                'lat': location.latitude,
                'lon': location.longitude,
                'components': {},
                'raw': location.raw
            }

            # Extract address components (Nominatim format)
            if hasattr(location, 'raw') and 'address' in location.raw:
                addr = location.raw['address']
                result['components'] = {
                    'house_number': addr.get('house_number'),
                    'road': addr.get('road'),
                    'city': addr.get('city') or addr.get('town') or addr.get('village'),
                    'state': addr.get('state'),
                    'postcode': addr.get('postcode'),
                    'country': addr.get('country'),
                }

            return result

        except Exception as e:
            return {'error': str(e)}

    def reverse(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Convert coordinates to address.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Dict with address and components
        """
        try:
            location = self._geocoder.reverse(
                f"{lat}, {lon}",
                addressdetails=True
            )
            if location is None:
                return None

            result = {
                'lat': lat,
                'lon': lon,
                'address': location.address,
                'components': {},
                'raw': location.raw
            }

            # Extract components
            if hasattr(location, 'raw') and 'address' in location.raw:
                addr = location.raw['address']
                result['components'] = {
                    'house_number': addr.get('house_number'),
                    'road': addr.get('road'),
                    'city': addr.get('city') or addr.get('town') or addr.get('village'),
                    'state': addr.get('state'),
                    'postcode': addr.get('postcode'),
                    'country': addr.get('country'),
                }

            return result

        except Exception as e:
            return {'error': str(e)}

    def batch_geocode(
        self,
        addresses: List[str],
        delay: float = 1.0
    ) -> List[Optional[Dict]]:
        """
        Geocode multiple addresses.

        Args:
            addresses: List of address strings
            delay: Delay between requests (seconds)

        Returns:
            List of geocoding results
        """
        results = []
        for i, addr in enumerate(addresses):
            result = self.geocode(addr)
            results.append(result)

            # Rate limiting
            if i < len(addresses) - 1:
                time.sleep(delay)

        return results

    def batch_reverse(
        self,
        coordinates: List[Tuple[float, float]],
        delay: float = 1.0
    ) -> List[Optional[Dict]]:
        """
        Reverse geocode multiple coordinates.

        Args:
            coordinates: List of (lat, lon) tuples
            delay: Delay between requests

        Returns:
            List of reverse geocoding results
        """
        results = []
        for i, (lat, lon) in enumerate(coordinates):
            result = self.reverse(lat, lon)
            results.append(result)

            if i < len(coordinates) - 1:
                time.sleep(delay)

        return results

    def geocode_csv(
        self,
        input_path: str,
        column: str,
        output_path: str,
        delay: float = 1.0
    ) -> Dict:
        """
        Geocode addresses from CSV file.

        Args:
            input_path: Input CSV path
            column: Address column name
            output_path: Output CSV path
            delay: Delay between requests

        Returns:
            Statistics dict
        """
        import pandas as pd

        df = pd.read_csv(input_path)
        total = len(df)
        success = 0

        lats = []
        lons = []
        full_addresses = []

        for i, row in df.iterrows():
            address = row[column]
            result = self.geocode(str(address))

            if result and 'error' not in result:
                lats.append(result['lat'])
                lons.append(result['lon'])
                full_addresses.append(result['address'])
                success += 1
            else:
                lats.append(None)
                lons.append(None)
                full_addresses.append(None)

            if i < total - 1:
                time.sleep(delay)

            # Progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{total}")

        df['geocoded_lat'] = lats
        df['geocoded_lon'] = lons
        df['geocoded_address'] = full_addresses

        df.to_csv(output_path, index=False)

        return {
            'total': total,
            'success': success,
            'failed': total - success,
            'output': output_path
        }

    def reverse_csv(
        self,
        input_path: str,
        lat_col: str,
        lon_col: str,
        output_path: str,
        delay: float = 1.0
    ) -> Dict:
        """
        Reverse geocode coordinates from CSV.

        Args:
            input_path: Input CSV path
            lat_col: Latitude column name
            lon_col: Longitude column name
            output_path: Output CSV path
            delay: Delay between requests

        Returns:
            Statistics dict
        """
        import pandas as pd

        df = pd.read_csv(input_path)
        total = len(df)
        success = 0

        addresses = []

        for i, row in df.iterrows():
            lat = row[lat_col]
            lon = row[lon_col]
            result = self.reverse(lat, lon)

            if result and 'error' not in result:
                addresses.append(result['address'])
                success += 1
            else:
                addresses.append(None)

            if i < total - 1:
                time.sleep(delay)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{total}")

        df['reverse_geocoded_address'] = addresses
        df.to_csv(output_path, index=False)

        return {
            'total': total,
            'success': success,
            'failed': total - success,
            'output': output_path
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Geocoding operations')
    parser.add_argument('--geocode', '-g', help='Address to geocode')
    parser.add_argument('--reverse', '-r', help='Coordinates to reverse (lat,lon)')
    parser.add_argument('--input', '-i', help='Input CSV file')
    parser.add_argument('--column', '-c', help='Address column for geocoding')
    parser.add_argument('--lat', default='lat', help='Latitude column')
    parser.add_argument('--lon', default='lon', help='Longitude column')
    parser.add_argument('--output', '-o', help='Output CSV file')
    parser.add_argument('--reverse-batch', action='store_true', help='Batch reverse geocode')
    parser.add_argument('--provider', default='nominatim', help='Geocoding provider')
    parser.add_argument('--api-key', help='API key for provider')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests')

    args = parser.parse_args()
    geo = Geocoder(provider=args.provider, api_key=args.api_key)

    if args.geocode:
        result = geo.geocode(args.geocode)
        if result and 'error' not in result:
            print(f"Address: {result['address']}")
            print(f"Latitude: {result['lat']}")
            print(f"Longitude: {result['lon']}")
            if result['components']:
                print("\nComponents:")
                for k, v in result['components'].items():
                    if v:
                        print(f"  {k}: {v}")
        elif result:
            print(f"Error: {result['error']}")
        else:
            print("Address not found")

    elif args.reverse:
        parts = args.reverse.split(',')
        lat, lon = float(parts[0].strip()), float(parts[1].strip())
        result = geo.reverse(lat, lon)
        if result and 'error' not in result:
            print(f"Address: {result['address']}")
            if result['components']:
                print("\nComponents:")
                for k, v in result['components'].items():
                    if v:
                        print(f"  {k}: {v}")
        elif result:
            print(f"Error: {result['error']}")
        else:
            print("Location not found")

    elif args.input and args.output:
        if args.reverse_batch:
            stats = geo.reverse_csv(
                args.input, args.lat, args.lon,
                args.output, delay=args.delay
            )
        elif args.column:
            stats = geo.geocode_csv(
                args.input, args.column,
                args.output, delay=args.delay
            )
        else:
            parser.error("--column required for geocoding CSV")
            return

        print(f"\nCompleted:")
        print(f"  Total: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Output: {stats['output']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
