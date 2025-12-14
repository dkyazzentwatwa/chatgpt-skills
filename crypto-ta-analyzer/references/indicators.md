# Technical Indicator Reference Guide

Comprehensive guide to all 24+ technical indicators used in the crypto-ta-analyzer skill.

## Table of Contents

1. [Trend Indicators](#trend-indicators)
2. [Momentum Indicators](#momentum-indicators)
3. [Volume Indicators](#volume-indicators)
4. [Volatility Indicators](#volatility-indicators)
5. [Scoring System](#scoring-system)

---

## Trend Indicators

### SMA (Simple Moving Average)
**Weight**: 1.0  
**Timeframes**: 20-period (short), 50-period (long)

Crossover strategy - bullish when short crosses above long, bearish when crosses below.

### EMA (Exponential Moving Average)
**Weight**: 1.0  
**Timeframes**: 12-period (short), 26-period (long)

More responsive than SMA due to exponential weighting of recent prices.

### DEMA (Double Exponential Moving Average)
**Weight**: 1.0  
**Timeframe**: 30-period

Reduces lag compared to SMA/EMA by double smoothing.

### TRIMA (Triangular Moving Average)
**Weight**: 0.5  
**Timeframe**: 30-period

Gives more weight to middle portion of data, smoother than SMA.

### WMA (Weighted Moving Average)
**Weight**: 0.5  
**Timeframe**: 30-period

Linear weighting with most recent data weighted highest.

### KAMA (Kaufman Adaptive Moving Average)
**Weight**: 0.5  
**Timeframe**: 30-period

Adapts to market volatility - fast in trending markets, slow in ranging.

### T3 (Tillson T3)
**Weight**: 0.5  
**Timeframe**: 5-period

Smooth moving average with reduced lag and overshooting.

### TRIX (Triple Exponential Moving Average)
**Weight**: 0.5  
**Timeframe**: 30-period

Triple smoothed to filter out insignificant price movements.

### MESA (MESA Adaptive Moving Average)
**Weight**: 1.0  
**Parameters**: fastlimit=0.5, slowlimit=0.05

Adaptive indicator that adjusts to current market cycle period.

### Parabolic SAR
**Weight**: 1.0  
**Parameters**: acceleration=0.02, maximum=0.2

Stop and Reverse system - dots above price suggest downtrend, below suggests uptrend.

---

## Momentum Indicators

### RSI (Relative Strength Index)
**Weight**: 1.0  
**Timeframe**: 14-period  
**Thresholds**: Oversold <30, Overbought >70

Measures speed and magnitude of price changes. Classic overbought/oversold indicator.

### MACD (Moving Average Convergence Divergence)
**Weight**: 1.0  
**Parameters**: fast=12, slow=26, signal=9

Trend-following momentum indicator showing relationship between two EMAs.

### MOM (Momentum)
**Weight**: 0.5  
**Timeframe**: 10-period

Simple rate of price change measurement.

### ROC (Rate of Change)
**Weight**: 0.5  
**Timeframe**: 10-period

Percentage change in price over specified period.

### CMO (Chande Momentum Oscillator)
**Weight**: 0.5  
**Timeframe**: 14-period  
**Thresholds**: Overbought >50, Oversold <-50

Modified RSI using sum of gains/losses rather than averages.

### PPO (Percentage Price Oscillator)
**Weight**: 0.5  
**Parameters**: fast=12, slow=26

MACD expressed in percentage terms for easier comparison across securities.

### APO (Absolute Price Oscillator)
**Weight**: 1.0  
**Parameters**: fast=12, slow=26

Difference between two moving averages expressed in absolute terms.

### CCI (Commodity Channel Index)
**Weight**: 1.0  
**Timeframe**: 14-period  
**Thresholds**: Overbought >100, Oversold <-100

Identifies cyclical trends - how far price deviates from average.

### AROON
**Weight**: 1.0  
**Timeframe**: 14-period  
**Thresholds**: Strong trend >70

Two lines (up/down) identify trend presence and direction.

### KDJ (Stochastic with J line)
**Weight**: 1.0  
**Parameters**: K=9, D=3, J=3K-2D  
**Thresholds**: Oversold <20, Overbought >80

Enhanced stochastic with J line for earlier signals.

---

## Volume Indicators

### MFI (Money Flow Index)
**Weight**: 1.0  
**Timeframe**: 14-period  
**Thresholds**: Oversold <20, Overbought >80

Volume-weighted RSI - incorporates both price and volume.

---

## Directional Indicators

### ADX (Average Directional Index)
**Weight**: 0.5  
**Timeframe**: 14-period  
**Threshold**: Strong trend >25

Measures trend strength regardless of direction.

### DMI (Directional Movement Index)
**Weight**: 0.5  
**Timeframe**: 14-period

Component of ADX - +DI and -DI lines show directional movement.

---

## Scoring System

### Score Weights

Each indicator contributes to the total score based on its signal:

- **Strong signals**: ±1.0 (RSI, MACD, EMA, SMA, DEMA, MESA, SAR, CCI, AROON, APO, MFI, KDJ)
- **Medium signals**: ±0.5 (ADX, CMO, DMI, KAMA, MOMI, PPO, ROC, TRIMA, TRIX, T3, WMA)

### Interpretation

**Total Score >= 7.0**: **STRONG UPTREND**
- Multiple indicators confirm bullish momentum
- High probability of continued upward movement
- Consider long positions

**Total Score 3.0 - 6.9**: **NEUTRAL**
- Mixed signals across indicators
- Market consolidation or transition phase
- Wait for clearer direction

**Total Score < 3.0**: **DOWNTREND**
- Multiple indicators confirm bearish momentum
- High probability of continued downward movement
- Avoid long positions, consider shorts

### Best Practices

1. **Never trade on single indicator** - the scoring system's strength is in consensus
2. **Consider market context** - indicators work differently in trending vs ranging markets
3. **Use multiple timeframes** - confirm signals across different periods
4. **Watch for divergences** - when price and indicators disagree, reversal may be coming
5. **Volume confirms price** - MFI adds crucial volume context to price-based indicators

### Indicator Combinations

**High Conviction Bullish**: RSI <30 + MACD crossover + Price above MESA + ADX >25  
**High Conviction Bearish**: RSI >70 + MACD crossunder + Price below MESA + ADX >25  
**Trend Exhaustion**: Score >7 but RSI >80 or MFI >90 (overbought warning)  
**False Breakout**: Strong price move but ADX <20 and volume declining

---

## Limitations and Considerations

- **Lagging nature**: Most indicators are calculated from past data
- **False signals**: Can generate whipsaws in choppy markets
- **Market regime changes**: What works in trends may fail in ranges
- **Overfitting**: Too many indicators can lead to analysis paralysis
- **CoinGecko data**: Price-only data means OHLC is approximated from adjacent points

### Recommended Minimum Data

- **Absolute minimum**: 50 data points
- **Recommended**: 100+ data points for reliable signals
- **Optimal**: 200+ data points for comprehensive analysis
