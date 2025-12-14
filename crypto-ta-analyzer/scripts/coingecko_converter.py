#!/usr/bin/env python3
"""
Helper script to convert CoinGecko chart data to TA-compatible OHLCV format.
"""

import json
import pandas as pd
from typing import Dict, List


def coingecko_to_ohlcv(chart_data: List[List]) -> pd.DataFrame:
    """
    Convert CoinGecko chart data to OHLCV DataFrame.
    
    CoinGecko returns: [[timestamp, price, market_cap, volume], ...]
    We need: time, open, high, low, close, volume
    
    Args:
        chart_data: Raw chart data from CoinGecko
        
    Returns:
        DataFrame with OHLCV data
    """
    df = pd.DataFrame(chart_data, columns=['timestamp', 'price', 'market_cap', 'volume'])
    
    # Create OHLCV approximation from price data
    # Since CoinGecko doesn't provide OHLC, we approximate:
    ohlcv_data = []
    
    for i in range(len(df)):
        price = df.iloc[i]['price']
        volume = df.iloc[i]['volume'] if not pd.isna(df.iloc[i]['volume']) else 0
        
        # Approximate OHLC from price point
        # In real scenarios, adjacent prices give us range
        if i > 0:
            prev_price = df.iloc[i-1]['price']
            high = max(price, prev_price)
            low = min(price, prev_price)
            open_price = prev_price
        else:
            high = price
            low = price
            open_price = price
        
        ohlcv_data.append({
            'time': df.iloc[i]['timestamp'],
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    return pd.DataFrame(ohlcv_data)


def prepare_analysis_data(coingecko_json: str) -> pd.DataFrame:
    """
    Prepare CoinGecko JSON data for technical analysis.
    
    Args:
        coingecko_json: JSON string from CoinGecko chart endpoint
        
    Returns:
        DataFrame ready for TechnicalAnalyzer
    """
    data = json.loads(coingecko_json)
    
    # Handle different CoinGecko response formats
    if 'prices' in data:
        chart_data = data['prices']
    else:
        chart_data = data
    
    return coingecko_to_ohlcv(chart_data)


def validate_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate OHLCV data quality and return statistics.
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        Dictionary with data quality metrics
    """
    return {
        'total_records': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'price_range': {
            'min': float(df['low'].min()),
            'max': float(df['high'].max()),
            'current': float(df['close'].iloc[-1])
        },
        'volume_stats': {
            'avg': float(df['volume'].mean()),
            'total': float(df['volume'].sum())
        },
        'data_quality': 'good' if len(df) >= 50 and df.isnull().sum().sum() == 0 else 'insufficient'
    }


if __name__ == "__main__":
    print("CoinGecko Data Converter")
    print("=======================")
    print("Use this module to convert CoinGecko API data to OHLCV format.")
