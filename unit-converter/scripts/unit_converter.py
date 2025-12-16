#!/usr/bin/env python3
"""
Unit Converter - Convert between physical units.

Features:
- Multiple categories (length, mass, temp, time, etc.)
- Precision control
- Batch conversion
- Formula display
"""

import argparse
from typing import Dict, List, Optional, Tuple, Union


class UnitConverter:
    """Convert between physical units."""

    # Base unit conversions (to SI base)
    CONVERSIONS = {
        'length': {
            'base': 'meter',
            'units': {
                'meter': 1.0,
                'm': 1.0,
                'kilometer': 1000.0,
                'km': 1000.0,
                'centimeter': 0.01,
                'cm': 0.01,
                'millimeter': 0.001,
                'mm': 0.001,
                'micrometer': 1e-6,
                'um': 1e-6,
                'nanometer': 1e-9,
                'nm_length': 1e-9,  # To avoid confusion with nautical mile
                'inch': 0.0254,
                'in': 0.0254,
                'foot': 0.3048,
                'ft': 0.3048,
                'yard': 0.9144,
                'yd': 0.9144,
                'mile': 1609.344,
                'mi': 1609.344,
                'nautical_mile': 1852.0,
                'nm': 1852.0,
            }
        },
        'mass': {
            'base': 'kilogram',
            'units': {
                'kilogram': 1.0,
                'kg': 1.0,
                'gram': 0.001,
                'g': 0.001,
                'milligram': 1e-6,
                'mg': 1e-6,
                'microgram': 1e-9,
                'ug': 1e-9,
                'pound': 0.453592,
                'lb': 0.453592,
                'lbs': 0.453592,
                'ounce': 0.0283495,
                'oz': 0.0283495,
                'ton': 907.185,  # US ton
                'metric_ton': 1000.0,
                'tonne': 1000.0,
                'stone': 6.35029,
            }
        },
        'time': {
            'base': 'second',
            'units': {
                'second': 1.0,
                's': 1.0,
                'sec': 1.0,
                'millisecond': 0.001,
                'ms': 0.001,
                'microsecond': 1e-6,
                'us': 1e-6,
                'minute': 60.0,
                'min': 60.0,
                'hour': 3600.0,
                'h': 3600.0,
                'hr': 3600.0,
                'day': 86400.0,
                'd': 86400.0,
                'week': 604800.0,
                'wk': 604800.0,
                'month': 2592000.0,  # 30 days
                'mo': 2592000.0,
                'year': 31536000.0,  # 365 days
                'yr': 31536000.0,
            }
        },
        'volume': {
            'base': 'liter',
            'units': {
                'liter': 1.0,
                'l': 1.0,
                'milliliter': 0.001,
                'ml': 0.001,
                'cubic_meter': 1000.0,
                'm3': 1000.0,
                'cubic_centimeter': 0.001,
                'cc': 0.001,
                'gallon': 3.78541,  # US gallon
                'gal': 3.78541,
                'quart': 0.946353,
                'qt': 0.946353,
                'pint': 0.473176,
                'pt': 0.473176,
                'cup': 0.236588,
                'fluid_ounce': 0.0295735,
                'fl_oz': 0.0295735,
                'tablespoon': 0.0147868,
                'tbsp': 0.0147868,
                'teaspoon': 0.00492892,
                'tsp': 0.00492892,
            }
        },
        'area': {
            'base': 'square_meter',
            'units': {
                'square_meter': 1.0,
                'm2': 1.0,
                'sqm': 1.0,
                'square_kilometer': 1e6,
                'km2': 1e6,
                'square_centimeter': 0.0001,
                'cm2': 0.0001,
                'square_foot': 0.092903,
                'sqft': 0.092903,
                'ft2': 0.092903,
                'square_inch': 0.00064516,
                'sqin': 0.00064516,
                'acre': 4046.86,
                'hectare': 10000.0,
                'ha': 10000.0,
            }
        },
        'speed': {
            'base': 'meters_per_second',
            'units': {
                'meters_per_second': 1.0,
                'm/s': 1.0,
                'mps': 1.0,
                'kilometers_per_hour': 0.277778,
                'km/h': 0.277778,
                'kph': 0.277778,
                'miles_per_hour': 0.44704,
                'mph': 0.44704,
                'feet_per_second': 0.3048,
                'ft/s': 0.3048,
                'knots': 0.514444,
                'kt': 0.514444,
            }
        },
        'digital': {
            'base': 'byte',
            'units': {
                'bit': 0.125,
                'byte': 1.0,
                'b': 1.0,
                'kilobyte': 1024.0,
                'kb': 1024.0,
                'megabyte': 1048576.0,
                'mb': 1048576.0,
                'gigabyte': 1073741824.0,
                'gb': 1073741824.0,
                'terabyte': 1099511627776.0,
                'tb': 1099511627776.0,
                'petabyte': 1125899906842624.0,
                'pb': 1125899906842624.0,
            }
        },
        'energy': {
            'base': 'joule',
            'units': {
                'joule': 1.0,
                'j': 1.0,
                'kilojoule': 1000.0,
                'kj': 1000.0,
                'calorie': 4.184,
                'cal': 4.184,
                'kilocalorie': 4184.0,
                'kcal': 4184.0,
                'watt_hour': 3600.0,
                'wh': 3600.0,
                'kilowatt_hour': 3600000.0,
                'kwh': 3600000.0,
                'btu': 1055.06,
                'electronvolt': 1.60218e-19,
                'ev': 1.60218e-19,
            }
        },
        'pressure': {
            'base': 'pascal',
            'units': {
                'pascal': 1.0,
                'pa': 1.0,
                'kilopascal': 1000.0,
                'kpa': 1000.0,
                'bar': 100000.0,
                'atmosphere': 101325.0,
                'atm': 101325.0,
                'psi': 6894.76,
                'torr': 133.322,
                'mmhg': 133.322,
            }
        },
        'angle': {
            'base': 'degree',
            'units': {
                'degree': 1.0,
                'deg': 1.0,
                'radian': 57.2958,
                'rad': 57.2958,
                'gradian': 0.9,
                'grad': 0.9,
                'arcminute': 1/60,
                'arcsecond': 1/3600,
            }
        },
    }

    # Temperature requires special handling
    TEMPERATURE = {
        'celsius': 'c',
        'c': 'c',
        'fahrenheit': 'f',
        'f': 'f',
        'kelvin': 'k',
        'k': 'k',
    }

    def __init__(self):
        """Initialize converter."""
        # Build reverse lookup
        self._unit_to_category = {}
        for category, data in self.CONVERSIONS.items():
            for unit in data['units']:
                self._unit_to_category[unit.lower()] = category

        for unit in self.TEMPERATURE:
            self._unit_to_category[unit.lower()] = 'temperature'

    def convert(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert value between units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        # Handle temperature separately
        if from_unit in self.TEMPERATURE:
            return self._convert_temperature(value, from_unit, to_unit)

        # Find category
        from_cat = self._unit_to_category.get(from_unit)
        to_cat = self._unit_to_category.get(to_unit)

        if from_cat is None:
            raise ValueError(f"Unknown unit: {from_unit}")
        if to_cat is None:
            raise ValueError(f"Unknown unit: {to_unit}")
        if from_cat != to_cat:
            raise ValueError(f"Cannot convert between {from_cat} and {to_cat}")

        # Convert via base unit
        units = self.CONVERSIONS[from_cat]['units']
        base_value = value * units[from_unit]
        result = base_value / units[to_unit]

        return result

    def _convert_temperature(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """Convert temperature values."""
        from_unit = self.TEMPERATURE.get(from_unit, from_unit)
        to_unit = self.TEMPERATURE.get(to_unit, to_unit)

        if to_unit not in self.TEMPERATURE.values():
            raise ValueError(f"Unknown temperature unit: {to_unit}")

        # Convert to Celsius first
        if from_unit == 'c':
            celsius = value
        elif from_unit == 'f':
            celsius = (value - 32) * 5/9
        elif from_unit == 'k':
            celsius = value - 273.15
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")

        # Convert from Celsius to target
        if to_unit == 'c':
            return celsius
        elif to_unit == 'f':
            return celsius * 9/5 + 32
        elif to_unit == 'k':
            return celsius + 273.15

    def convert_with_details(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> Dict:
        """
        Convert with full details.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Dict with result and metadata
        """
        result = self.convert(value, from_unit, to_unit)
        category = self._unit_to_category.get(from_unit.lower(), 'temperature')

        return {
            'value': value,
            'from_unit': from_unit,
            'to_unit': to_unit,
            'result': result,
            'formula': self.get_formula(from_unit, to_unit),
            'category': category
        }

    def batch_convert(
        self,
        values: List[float],
        from_unit: str,
        to_unit: str
    ) -> List[float]:
        """
        Convert multiple values.

        Args:
            values: List of values
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            List of converted values
        """
        return [self.convert(v, from_unit, to_unit) for v in values]

    def list_categories(self) -> List[str]:
        """List all unit categories."""
        return list(self.CONVERSIONS.keys()) + ['temperature']

    def list_units(self, category: Optional[str] = None) -> Dict:
        """
        List units, optionally filtered by category.

        Args:
            category: Category to filter by

        Returns:
            Dict of category -> units
        """
        if category:
            category = category.lower()
            if category == 'temperature':
                return {'temperature': list(set(self.TEMPERATURE.values()))}
            if category in self.CONVERSIONS:
                units = list(self.CONVERSIONS[category]['units'].keys())
                # Filter aliases
                main_units = [u for u in units if len(u) > 2 or u in ['m', 's', 'g', 'l', 'j']]
                return {category: main_units}
            raise ValueError(f"Unknown category: {category}")

        result = {}
        for cat, data in self.CONVERSIONS.items():
            units = [u for u in data['units'].keys() if len(u) > 2 or u in ['m', 's', 'g', 'l', 'j']]
            result[cat] = units
        result['temperature'] = ['celsius', 'fahrenheit', 'kelvin']
        return result

    def get_formula(self, from_unit: str, to_unit: str) -> str:
        """
        Get conversion formula string.

        Args:
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Formula string
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        # Temperature formulas
        if from_unit in self.TEMPERATURE:
            f = self.TEMPERATURE.get(from_unit)
            t = self.TEMPERATURE.get(to_unit)
            formulas = {
                ('c', 'f'): "F = (C × 9/5) + 32",
                ('c', 'k'): "K = C + 273.15",
                ('f', 'c'): "C = (F - 32) × 5/9",
                ('f', 'k'): "K = (F - 32) × 5/9 + 273.15",
                ('k', 'c'): "C = K - 273.15",
                ('k', 'f'): "F = (K - 273.15) × 9/5 + 32",
            }
            return formulas.get((f, t), f"{to_unit} = f({from_unit})")

        # Linear conversions
        category = self._unit_to_category.get(from_unit)
        if category:
            units = self.CONVERSIONS[category]['units']
            factor = units[from_unit] / units[to_unit]
            if factor == 1:
                return f"{to_unit} = {from_unit}"
            return f"{to_unit} = {from_unit} × {factor:.6g}"

        return ""

    def find_unit(self, query: str) -> List[str]:
        """
        Search for units matching query.

        Args:
            query: Search string

        Returns:
            List of matching unit names
        """
        query = query.lower()
        matches = []

        for category, data in self.CONVERSIONS.items():
            for unit in data['units']:
                if query in unit.lower():
                    matches.append(unit)

        for unit in self.TEMPERATURE:
            if query in unit.lower():
                matches.append(unit)

        return list(set(matches))


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Convert between units')
    parser.add_argument('value', nargs='?', type=float, help='Value to convert')
    parser.add_argument('from_unit', nargs='?', help='Source unit')
    parser.add_argument('to_unit', nargs='?', help='Target unit')
    parser.add_argument('--precision', '-p', type=int, default=6, help='Decimal places')
    parser.add_argument('--list', '-l', nargs='?', const='all', help='List units (optionally by category)')
    parser.add_argument('--formula', '-f', action='store_true', help='Show conversion formula')
    parser.add_argument('--search', '-s', help='Search for unit')

    args = parser.parse_args()
    converter = UnitConverter()

    if args.list:
        if args.list == 'all':
            units = converter.list_units()
            for cat, unit_list in units.items():
                print(f"\n{cat.upper()}:")
                print(f"  {', '.join(unit_list)}")
        else:
            try:
                units = converter.list_units(args.list)
                for cat, unit_list in units.items():
                    print(f"{cat.upper()}:")
                    print(f"  {', '.join(unit_list)}")
            except ValueError as e:
                print(f"Error: {e}")

    elif args.search:
        matches = converter.find_unit(args.search)
        if matches:
            print(f"Units matching '{args.search}':")
            for unit in matches:
                print(f"  {unit}")
        else:
            print(f"No units found matching '{args.search}'")

    elif args.value is not None and args.from_unit and args.to_unit:
        try:
            result = converter.convert(args.value, args.from_unit, args.to_unit)
            print(f"{args.value} {args.from_unit} = {result:.{args.precision}g} {args.to_unit}")

            if args.formula:
                formula = converter.get_formula(args.from_unit, args.to_unit)
                print(f"Formula: {formula}")
        except ValueError as e:
            print(f"Error: {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
