#!/usr/bin/env python3
"""
Generic Data Converter for Technical Analysis

Supports multiple data sources:
- CoinGecko API format
- Binance/Exchange OHLCV format
- Yahoo Finance format
- Generic price-only data
- Auto-detection of format

This module converts any supported format to standardized OHLCV DataFrame
ready for TechnicalAnalyzer.
"""

import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union


# =============================================================================
# Format Detection
# =============================================================================

def detect_data_format(data: Union[pd.DataFrame, List, Dict, str]) -> str:
    """
    Auto-detect the input data format.

    Returns one of:
    - "coingecko": CoinGecko chart format [[timestamp, price, market_cap, volume], ...]
    - "exchange_ohlcv": Standard exchange OHLCV (5-12 columns)
    - "yahoo": Yahoo Finance format (Date, Open, High, Low, Close, Adj Close, Volume)
    - "price_only": Simple price data (timestamp, price) or just prices
    - "ohlcv_dict": List of dicts with OHLCV keys
    - "unknown": Unrecognized format
    """
    # Parse JSON if string
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return "unknown"

    # Handle dict with 'prices' key (CoinGecko wrapper)
    if isinstance(data, dict):
        if "prices" in data:
            return "coingecko"
        if "chart" in data and "result" in data.get("chart", {}):
            return "yahoo"
        return "unknown"

    # Handle list
    if isinstance(data, list):
        if len(data) == 0:
            return "unknown"

        first = data[0]

        # List of lists (array format)
        if isinstance(first, (list, tuple)):
            cols = len(first)
            if cols == 4:
                return "coingecko"  # [timestamp, price, market_cap, volume]
            elif cols == 2:
                return "price_only"  # [timestamp, price]
            elif cols >= 5:
                return "exchange_ohlcv"  # [timestamp, o, h, l, c, v, ...]
            return "unknown"

        # List of dicts
        if isinstance(first, dict):
            keys = set(first.keys())
            lower_keys = {k.lower() for k in keys}

            # Check for OHLCV keys
            ohlcv_keys = {"open", "high", "low", "close"}
            if ohlcv_keys.issubset(lower_keys):
                return "ohlcv_dict"

            # Check for Yahoo Finance keys
            if "adj close" in lower_keys or "adjclose" in lower_keys:
                return "yahoo"

            # Check for price-only
            if "price" in lower_keys or "close" in lower_keys:
                return "price_only"

            return "unknown"

    # Handle DataFrame
    if isinstance(data, pd.DataFrame):
        cols = {c.lower() for c in data.columns}

        # Check for full OHLCV
        if {"open", "high", "low", "close"}.issubset(cols):
            if "adj close" in cols or "adjclose" in cols:
                return "yahoo"
            return "exchange_ohlcv"

        # Check for price-only
        if "close" in cols or "price" in cols:
            return "price_only"

        return "unknown"

    return "unknown"


# =============================================================================
# Format-Specific Converters
# =============================================================================

def coingecko_to_ohlcv(chart_data: List[List], enhanced: bool = True) -> pd.DataFrame:
    """
    Convert CoinGecko chart data to OHLCV DataFrame.

    CoinGecko returns: [[timestamp, price, market_cap, volume], ...]
    We approximate OHLC from price data.

    Args:
        chart_data: Raw chart data from CoinGecko
        enhanced: Use enhanced OHLC approximation (recommended)

    Returns:
        DataFrame with OHLCV data
    """
    if len(chart_data) == 0:
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

    # Handle different CoinGecko formats
    if len(chart_data[0]) == 4:
        df = pd.DataFrame(chart_data, columns=['timestamp', 'price', 'market_cap', 'volume'])
    elif len(chart_data[0]) == 2:
        df = pd.DataFrame(chart_data, columns=['timestamp', 'price'])
        df['volume'] = 0.0
    else:
        df = pd.DataFrame(chart_data)
        df.columns = ['timestamp', 'price'] + [f'col{i}' for i in range(2, len(df.columns))]
        if 'volume' not in df.columns:
            df['volume'] = 0.0

    prices = df['price'].values
    volumes = df['volume'].fillna(0).values if 'volume' in df.columns else np.zeros(len(df))

    if enhanced:
        ohlcv_df = _enhanced_ohlc_approximation(prices, volumes, df['timestamp'].values)
    else:
        ohlcv_df = _simple_ohlc_approximation(prices, volumes, df['timestamp'].values)

    return ohlcv_df


def _simple_ohlc_approximation(
    prices: np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray
) -> pd.DataFrame:
    """Simple OHLC approximation using adjacent prices."""
    ohlcv_data = []

    for i in range(len(prices)):
        price = prices[i]
        volume = volumes[i] if not np.isnan(volumes[i]) else 0

        if i > 0:
            prev_price = prices[i - 1]
            high = max(price, prev_price)
            low = min(price, prev_price)
            open_price = prev_price
        else:
            high = price
            low = price
            open_price = price

        ohlcv_data.append({
            'time': timestamps[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })

    return pd.DataFrame(ohlcv_data)


def _enhanced_ohlc_approximation(
    prices: np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray,
    window: int = 3
) -> pd.DataFrame:
    """
    Enhanced OHLC approximation using rolling window for better high/low estimates.

    This method considers nearby prices to better estimate the true range
    within each period.
    """
    n = len(prices)
    if n == 0:
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

    ohlcv_data = []

    for i in range(n):
        price = prices[i]
        volume = volumes[i] if not np.isnan(volumes[i]) else 0

        # Use rolling window for high/low estimation
        start = max(0, i - window + 1)
        end = min(n, i + 2)  # Include current and next price if available

        window_prices = prices[start:end]

        if i > 0:
            open_price = prices[i - 1]
        else:
            open_price = price

        # Estimate high/low from window
        high = float(np.max(window_prices))
        low = float(np.min(window_prices))

        # Ensure OHLC consistency
        high = max(high, open_price, price)
        low = min(low, open_price, price)

        ohlcv_data.append({
            'time': timestamps[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })

    return pd.DataFrame(ohlcv_data)


def exchange_ohlcv_to_df(data: Union[List, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert exchange OHLCV format to standardized DataFrame.

    Handles formats like:
    - [timestamp, open, high, low, close, volume, ...]
    - DataFrame with OHLCV columns
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, list):
        if len(data) == 0:
            return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

        if isinstance(data[0], dict):
            df = pd.DataFrame(data)
        else:
            # List of lists
            cols = len(data[0])
            if cols >= 6:
                col_names = ['time', 'open', 'high', 'low', 'close', 'volume'] + [f'col{i}' for i in range(6, cols)]
            elif cols == 5:
                col_names = ['time', 'open', 'high', 'low', 'close']
            else:
                col_names = [f'col{i}' for i in range(cols)]
            df = pd.DataFrame(data, columns=col_names)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    # Normalize column names
    df = _normalize_columns(df)

    return df


def yahoo_to_ohlcv(data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
    """Convert Yahoo Finance format to standardized OHLCV."""
    if isinstance(data, dict):
        # Yahoo API JSON format
        if "chart" in data and "result" in data["chart"]:
            result = data["chart"]["result"][0]
            timestamps = result.get("timestamp", [])
            quotes = result.get("indicators", {}).get("quote", [{}])[0]

            df = pd.DataFrame({
                'time': timestamps,
                'open': quotes.get('open', []),
                'high': quotes.get('high', []),
                'low': quotes.get('low', []),
                'close': quotes.get('close', []),
                'volume': quotes.get('volume', [])
            })
        else:
            df = pd.DataFrame(data)
    else:
        df = data.copy()

    # Normalize columns
    df = _normalize_columns(df)

    # Handle Adj Close if present
    if 'adj_close' in df.columns or 'adjclose' in df.columns:
        adj_col = 'adj_close' if 'adj_close' in df.columns else 'adjclose'
        # Use adjusted close as the close price
        if df[adj_col].notna().any():
            df['close'] = df[adj_col]
        df = df.drop(columns=[adj_col], errors='ignore')

    return df


def price_only_to_ohlcv(
    data: Union[List, pd.DataFrame, pd.Series],
    enhanced: bool = True
) -> pd.DataFrame:
    """
    Convert price-only data to OHLCV format.

    Args:
        data: Price data (list, DataFrame with 'price'/'close', or Series)
        enhanced: Use enhanced approximation
    """
    if isinstance(data, pd.Series):
        prices = data.values
        timestamps = data.index.values if hasattr(data.index, 'values') else np.arange(len(data))
        volumes = np.zeros(len(data))
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        df = _normalize_columns(df)

        if 'close' in df.columns:
            prices = df['close'].values
        elif 'price' in df.columns:
            prices = df['price'].values
        else:
            raise ValueError("No price/close column found")

        timestamps = df.get('time', pd.Series(np.arange(len(df)))).values
        volumes = df.get('volume', pd.Series(np.zeros(len(df)))).values
    elif isinstance(data, list):
        if len(data) == 0:
            return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

        if isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
            timestamps = np.array([x[0] for x in data])
            prices = np.array([x[1] for x in data])
            volumes = np.zeros(len(data))
        else:
            prices = np.array(data)
            timestamps = np.arange(len(data))
            volumes = np.zeros(len(data))
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    if enhanced:
        return _enhanced_ohlc_approximation(prices, volumes, timestamps)
    return _simple_ohlc_approximation(prices, volumes, timestamps)


# =============================================================================
# Normalization and Validation
# =============================================================================

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to standard format."""
    rename_map = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ('timestamp', 'datetime', 'date', 't'):
            rename_map[c] = 'time'
        elif cl in ('o', 'open_price', 'openprice'):
            rename_map[c] = 'open'
        elif cl in ('h', 'high_price', 'highprice', 'max'):
            rename_map[c] = 'high'
        elif cl in ('l', 'low_price', 'lowprice', 'min'):
            rename_map[c] = 'low'
        elif cl in ('c', 'close_price', 'closeprice', 'price', 'last'):
            rename_map[c] = 'close'
        elif cl in ('v', 'vol', 'volume_24h', 'volume24h', 'base_volume'):
            rename_map[c] = 'volume'
        elif cl in ('adj close', 'adj_close', 'adjusted_close', 'adjclose'):
            rename_map[c] = 'adj_close'

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def validate_and_repair(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate OHLCV data and auto-repair common issues.

    Returns:
        (cleaned_df, warnings_list)
    """
    warnings = []
    df = df.copy()

    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            warnings.append(f"Missing required column: {col}")
            return df, warnings

    # Ensure volume exists
    if 'volume' not in df.columns:
        df['volume'] = 0.0
        warnings.append("Volume column missing, set to 0")

    # Convert to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove duplicate timestamps
    if 'time' in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=['time'], keep='last')
        after = len(df)
        if after < before:
            warnings.append(f"Removed {before - after} duplicate timestamps")

    # Sort by time
    if 'time' in df.columns:
        df = df.sort_values('time').reset_index(drop=True)

    # Fix OHLC inconsistencies
    inconsistent_count = 0
    for i in range(len(df)):
        o, h, l, c = df.loc[i, 'open'], df.loc[i, 'high'], df.loc[i, 'low'], df.loc[i, 'close']

        if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
            continue

        fixed = False
        # High should be >= all others
        if h < max(o, l, c):
            df.loc[i, 'high'] = max(o, h, l, c)
            fixed = True

        # Low should be <= all others
        if l > min(o, h, c):
            df.loc[i, 'low'] = min(o, h, l, c)
            fixed = True

        if fixed:
            inconsistent_count += 1

    if inconsistent_count > 0:
        warnings.append(f"Fixed {inconsistent_count} OHLC inconsistencies (high < low, etc.)")

    # Fill small gaps with interpolation
    null_counts = df[['open', 'high', 'low', 'close']].isnull().sum()
    total_nulls = null_counts.sum()

    if total_nulls > 0 and total_nulls < len(df) * 0.1:  # Less than 10% missing
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].interpolate(method='linear', limit=3)
        warnings.append(f"Interpolated {total_nulls} missing values")

    # Drop rows with any remaining NaN in OHLC
    before = len(df)
    df = df.dropna(subset=['open', 'high', 'low', 'close']).reset_index(drop=True)
    after = len(df)
    if after < before:
        warnings.append(f"Dropped {before - after} rows with missing OHLC data")

    return df, warnings


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate OHLCV data quality and return comprehensive statistics.

    Args:
        df: OHLCV DataFrame

    Returns:
        Dictionary with data quality metrics
    """
    result = {
        'total_records': len(df),
        'missing_values': df.isnull().sum().to_dict() if len(df) > 0 else {},
        'data_quality': 'unknown',
        'issues': [],
    }

    if len(df) == 0:
        result['data_quality'] = 'empty'
        result['issues'].append('No data')
        return result

    # Price statistics
    if 'close' in df.columns and df['close'].notna().any():
        result['price_range'] = {
            'min': float(df['low'].min()) if 'low' in df.columns else float(df['close'].min()),
            'max': float(df['high'].max()) if 'high' in df.columns else float(df['close'].max()),
            'current': float(df['close'].iloc[-1]),
            'mean': float(df['close'].mean()),
        }

    # Volume statistics
    if 'volume' in df.columns and df['volume'].notna().any():
        result['volume_stats'] = {
            'avg': float(df['volume'].mean()),
            'total': float(df['volume'].sum()),
            'zero_volume_pct': float((df['volume'] == 0).sum() / len(df) * 100),
        }

    # Time coverage
    if 'time' in df.columns:
        try:
            times = pd.to_datetime(df['time'], errors='coerce')
            if times.notna().any():
                result['time_range'] = {
                    'start': str(times.min()),
                    'end': str(times.max()),
                    'duration_hours': float((times.max() - times.min()).total_seconds() / 3600),
                }
        except Exception:
            pass

    # Quality assessment
    issues = []

    if len(df) < 50:
        issues.append('insufficient_data')

    null_count = df[['open', 'high', 'low', 'close']].isnull().sum().sum()
    if null_count > 0:
        issues.append('missing_values')

    if 'volume' in df.columns:
        zero_vol_pct = (df['volume'] == 0).sum() / len(df)
        if zero_vol_pct > 0.5:
            issues.append('mostly_zero_volume')

    result['issues'] = issues

    if len(issues) == 0 and len(df) >= 100:
        result['data_quality'] = 'good'
    elif len(issues) == 0 and len(df) >= 50:
        result['data_quality'] = 'acceptable'
    elif 'insufficient_data' not in issues:
        result['data_quality'] = 'fair'
    else:
        result['data_quality'] = 'insufficient'

    return result


# =============================================================================
# Main Interface
# =============================================================================

def normalize_ohlcv(
    data: Union[pd.DataFrame, List, Dict, str],
    source: str = "auto",
    enhanced_approximation: bool = True,
    validate: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalize any data source to standard OHLCV format.

    This is the main entry point for data conversion.

    Args:
        data: Input data in any supported format
        source: Data source hint ("auto", "coingecko", "exchange", "yahoo", "price_only")
        enhanced_approximation: Use enhanced OHLC approximation for price-only data
        validate: Run validation and repair

    Returns:
        (ohlcv_dataframe, metadata_dict)

    Supported sources:
    - "coingecko": [[timestamp, price, market_cap, volume], ...]
    - "exchange": Standard OHLCV with 5-12 columns
    - "yahoo": Yahoo Finance format
    - "price_only": Any DataFrame/list with price data
    - "auto": Auto-detect format
    """
    metadata = {
        'detected_format': None,
        'original_rows': 0,
        'final_rows': 0,
        'warnings': [],
    }

    # Parse JSON if string
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    # Auto-detect format if needed
    if source == "auto":
        source = detect_data_format(data)
        metadata['detected_format'] = source

    if source == "unknown":
        raise ValueError("Could not auto-detect data format. Please specify source parameter.")

    # Handle dict with 'prices' key (CoinGecko wrapper)
    if isinstance(data, dict) and "prices" in data:
        data = data["prices"]
        source = "coingecko"

    # Convert based on source
    if source == "coingecko":
        df = coingecko_to_ohlcv(data, enhanced=enhanced_approximation)
    elif source in ("exchange_ohlcv", "exchange", "ohlcv_dict"):
        df = exchange_ohlcv_to_df(data)
    elif source == "yahoo":
        df = yahoo_to_ohlcv(data)
    elif source == "price_only":
        df = price_only_to_ohlcv(data, enhanced=enhanced_approximation)
    else:
        raise ValueError(f"Unsupported source: {source}")

    metadata['original_rows'] = len(df)

    # Normalize column names
    df = _normalize_columns(df)

    # Validate and repair
    if validate:
        df, warnings = validate_and_repair(df)
        metadata['warnings'] = warnings

    metadata['final_rows'] = len(df)
    metadata['quality'] = validate_data_quality(df)

    return df, metadata


# =============================================================================
# Backward Compatibility (keep old function names working)
# =============================================================================

def prepare_analysis_data(coingecko_json: str) -> pd.DataFrame:
    """
    Prepare CoinGecko JSON data for technical analysis.
    (Backward compatible with old coingecko_converter.py)

    Args:
        coingecko_json: JSON string from CoinGecko chart endpoint

    Returns:
        DataFrame ready for TechnicalAnalyzer
    """
    df, _ = normalize_ohlcv(coingecko_json, source="coingecko")
    return df


# Keep the old name as alias
coingecko_to_ohlcv_legacy = coingecko_to_ohlcv


if __name__ == "__main__":
    print("Generic Data Converter for Technical Analysis")
    print("=" * 50)
    print("\nSupported formats:")
    print("  - CoinGecko API charts")
    print("  - Exchange OHLCV (Binance, etc.)")
    print("  - Yahoo Finance")
    print("  - Price-only data")
    print("\nUsage:")
    print("  from data_converter import normalize_ohlcv")
    print("  df, metadata = normalize_ohlcv(data, source='auto')")
