# Technical Indicator Reference Guide

Comprehensive guide to all 29 technical indicators used in the crypto-ta-analyzer skill.

## Table of Contents

1. [New Indicators (v2)](#new-indicators-v2)
2. [Trend Indicators](#trend-indicators)
3. [Momentum Indicators](#momentum-indicators)
4. [Volume Indicators](#volume-indicators)
5. [Volatility Indicators](#volatility-indicators)
6. [Divergence Detection](#divergence-detection)
7. [Scoring System](#scoring-system)

---

## New Indicators (v2)

### Bollinger Bands (BB)
**Weight**: 1.0
**Parameters**: period=20, std_dev=2.0, squeeze_threshold=5.0

Volatility-based bands around a moving average. Key components:
- **Middle Band**: 20-period SMA
- **Upper Band**: Middle + 2σ (standard deviations)
- **Lower Band**: Middle - 2σ
- **Bandwidth**: (Upper - Lower) / Middle * 100
- **%B**: (Price - Lower) / (Upper - Lower) — position within bands

**Signals**:
- In RANGING markets: Mean reversion (buy at lower band, sell at upper)
- In TRENDING markets: Breakout continuation (riding bands is bullish/bearish)
- **Squeeze Detection**: When bandwidth < threshold for 6+ bars, low volatility may precede breakout

### On-Balance Volume (OBV)
**Weight**: 1.0
**Parameters**: signal_period=20

Cumulative volume indicator that adds volume on up days, subtracts on down days.
**Formula**: OBV = prev_OBV ± volume (based on close vs prev_close)

**Signals**:
- OBV above signal line = bullish
- OBV crossing above signal = buy trigger
- **Divergence**: OBV diverging from price often precedes reversals (volume leads price)

### Ichimoku Cloud
**Weight**: 1.0
**Parameters**: tenkan=10, kijun=30, senkou_b=60 (crypto-optimized)

Multi-component Japanese trend system. Components:
- **Tenkan-sen (Conversion)**: (10-period high + low) / 2
- **Kijun-sen (Base)**: (30-period high + low) / 2
- **Senkou Span A**: (Tenkan + Kijun) / 2, shifted forward
- **Senkou Span B**: (60-period high + low) / 2, shifted forward
- **Cloud (Kumo)**: Area between Senkou A and B

**Signals**:
- Price above cloud = bullish, below = bearish, inside = neutral
- Tenkan above Kijun = bullish cross
- Cloud color (A > B = green/bullish)

### VWAP (Volume Weighted Average Price)
**Weight**: 0.5

Institutional benchmark price. **Formula**: Cumulative(TP * Volume) / Cumulative(Volume)

**Signals**:
- Price above VWAP = bullish bias
- Price below VWAP = bearish bias
- Used by institutions for execution benchmarks

### ATR Signal
**Weight**: 0.5
**Parameters**: period=14

ATR (Average True Range) exposed as a signal indicator for volatility-based trading.

**Signals**:
- Low volatility (ATR contraction) = potential breakout setup
- High volatility + trending = trend confirmation
- Combines with regime detection for context-aware signals

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

## Divergence Detection

Divergences occur when price and indicator move in opposite directions, often signaling reversals.

### Types of Divergence

**Regular Bullish Divergence**:
- Price makes lower low
- Indicator makes higher low
- Signal: Potential reversal upward

**Regular Bearish Divergence**:
- Price makes higher high
- Indicator makes lower high
- Signal: Potential reversal downward

**Hidden Bullish Divergence**:
- Price makes higher low
- Indicator makes lower low
- Signal: Trend continuation in uptrend

**Hidden Bearish Divergence**:
- Price makes lower high
- Indicator makes higher high
- Signal: Trend continuation in downtrend

### Indicators with Divergence Detection

1. **RSI**: Most common divergence indicator
2. **MACD Histogram**: Reliable for momentum divergences
3. **OBV**: Most reliable — volume precedes price

### Divergence Confidence Ranking

1. **OBV divergence**: Highest reliability (volume leads price)
2. **RSI divergence**: High reliability on longer timeframes
3. **MACD divergence**: Good for momentum shifts

---

## Scoring System

### Score Weights (Updated v2)

Each indicator contributes to the total score based on its signal:

**Core (1.0)**: RSI, MACD, BB, OBV, ICHIMOKU, EMA, SMA, MFI, KDJ, SAR
**Strong (0.75)**: DEMA, MESA, CCI, AROON, APO
**Supporting (0.5)**: ADX, DMI, CMO, KAMA, MOMI, PPO, ROC, TRIMA, TRIX, T3, WMA, VWAP, ATR_SIGNAL, CAD

### 7-Tier Signal System (NEW)

**STRONG_BUY**: Normalized score >= 0.5, confidence >= 0.7
**BUY**: Normalized score >= 0.35, confidence >= 0.5
**WEAK_BUY**: Normalized score >= 0.2
**NEUTRAL**: -0.2 < normalized < 0.2
**WEAK_SELL**: Normalized score <= -0.2
**SELL**: Normalized score <= -0.35, confidence >= 0.5
**STRONG_SELL**: Normalized score <= -0.5, confidence >= 0.7

### Legacy Interpretation (backward compatible)

**Total Score >= 7.0**: **STRONG UPTREND**
**Total Score 3.0 - 6.9**: **NEUTRAL**
**Total Score < 3.0**: **DOWNTREND**

### Confidence Calculation

Confidence is computed from multiple factors:
- **Alignment** (30%): How much indicators agree
- **ADX Score** (20%): Trend strength
- **Coverage** (15%): % of indicators computed
- **Volume Confirmation** (15%): OBV/MFI agreement
- **Divergence Penalty** (10%): Lower if divergences detected
- **Volatility Adjustment** (10%): Lower in extreme volatility

### Volume Confirmation

New factor that measures if volume supports price direction:
- OBV trend matching price trend = +1.0
- MFI agreeing with price direction = +0.5 to +1.0
- No OBV divergence = +0.8

### Best Practices

1. **Never trade on single indicator** - the scoring system's strength is in consensus
2. **Watch divergences** - especially OBV divergences (volume leads price)
3. **Monitor BB squeeze** - low volatility often precedes breakouts
4. **Use Ichimoku cloud** - price position vs cloud is strong trend filter
5. **Check volume confirmation** - ensure volume supports the signal
6. **Consider market regime** - trending vs ranging affects indicator reliability

### High Conviction Setups

**Strong Buy Setup**:
- 7-tier: STRONG_BUY or BUY
- Confidence >= 0.7
- Price above Ichimoku cloud
- OBV confirms (no bearish divergence)
- No BB squeeze (or just breaking out of squeeze)

**Breakout Setup**:
- BB squeeze detected
- ADX rising from < 20
- Watch for band expansion with volume

**Reversal Warning**:
- Bearish divergence on RSI/MACD/OBV
- Price at upper BB (%B > 1.0)
- RSI > 70 or MFI > 80

---

## Limitations and Considerations

- **Lagging nature**: Most indicators are calculated from past data
- **False signals**: Can generate whipsaws in choppy markets
- **Market regime changes**: What works in trends may fail in ranges
- **Divergences**: Not all divergences lead to reversals
- **OHLC approximation**: Price-only data sources approximate high/low

### Recommended Minimum Data

- **Absolute minimum**: 50 data points
- **Recommended**: 100+ data points for reliable signals
- **Optimal**: 200+ data points for comprehensive analysis
- **Ichimoku**: Requires 60+ bars for full cloud calculation
