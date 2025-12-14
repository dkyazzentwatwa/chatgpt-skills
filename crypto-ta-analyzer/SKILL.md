---
name: crypto-ta-analyzer
description: Comprehensive cryptocurrency and stock technical analysis using 24+ proven indicators. Use when users request price analysis, trading signals, trend identification, or multi-indicator technical assessments for crypto or stocks. Integrates with CoinGecko MCP tools for real-time market data. Generates scored trading signals (STRONG_UPTREND/NEUTRAL/DOWNTREND) based on consensus across RSI, MACD, moving averages, momentum oscillators, and volume indicators.
---

# Crypto & Stock Technical Analysis

Multi-indicator technical analysis system that generates high-confidence trading signals by combining 24+ proven algorithms. Ideal for cryptocurrency and stock market analysis.

## Core Workflow

### 1. Data Acquisition via CoinGecko

First, fetch historical price data using CoinGecko MCP tools:

```
Use coingecko_get_historical_chart tool with:
- coin_id: Target cryptocurrency (e.g., 'bitcoin', 'ethereum')
- days: Time range ('7', '30', '90', '365', 'max')
- vs_currency: Base currency (default 'usd')
```

**Minimum Requirements**:
- At least 100 data points for reliable analysis
- Price, market cap, and volume data
- Recent data preferred for active trading signals

### 2. Convert CoinGecko Data to OHLCV Format

Run the converter script to prepare data:

```python
python3 scripts/coingecko_converter.py
```

Or programmatically:
```python
from scripts.coingecko_converter import prepare_analysis_data, validate_data_quality
import json

# Convert CoinGecko JSON to OHLCV DataFrame
ohlcv_df = prepare_analysis_data(coingecko_json_data)

# Validate data quality
quality_report = validate_data_quality(ohlcv_df)
```

### 3. Run Technical Analysis

Execute the analyzer with prepared data:

```python
from scripts.ta_analyzer import TechnicalAnalyzer
import json

# Initialize analyzer with OHLCV data
analyzer = TechnicalAnalyzer(ohlcv_df)

# Run comprehensive analysis
results = analyzer.analyze_all()

# Display results
print(json.dumps(results, indent=2))
```

### 4. Interpret Results

Analysis returns:
```json
{
  "scoreTotal": 8.5,
  "tradeSignal": "STRONG_UPTREND",
  "tradeTrigger": true,
  "currentPrice": 45234.56,
  "priceChange24h": 3.45,
  "individualScores": {
    "RSI": 1.0,
    "MACD": 1.0,
    "EMA": 1.0,
    ...
  },
  "individualSignals": {
    "RSI": "BUY",
    "MACD": "BUY",
    "EMA": "BUY",
    ...
  }
}
```

**Signal Interpretation**:
- **scoreTotal >= 7.0**: STRONG_UPTREND - High confidence bullish signal
- **scoreTotal 3.0-6.9**: NEUTRAL - Mixed signals, wait for clarity
- **scoreTotal < 3.0**: DOWNTREND - Bearish signal, avoid longs

## Available Indicators

### Trend Indicators (8)
- **SMA** (Simple Moving Average) - Weight: 1.0
- **EMA** (Exponential Moving Average) - Weight: 1.0
- **DEMA** (Double Exponential MA) - Weight: 1.0
- **TRIMA** (Triangular MA) - Weight: 0.5
- **WMA** (Weighted MA) - Weight: 0.5
- **KAMA** (Kaufman Adaptive MA) - Weight: 0.5
- **T3** (Tillson T3) - Weight: 0.5
- **MESA** (MESA Adaptive MA) - Weight: 1.0

### Momentum Indicators (10)
- **RSI** (Relative Strength Index) - Weight: 1.0
- **MACD** (Moving Average Convergence Divergence) - Weight: 1.0
- **AROON** - Weight: 1.0
- **CCI** (Commodity Channel Index) - Weight: 1.0
- **CMO** (Chande Momentum Oscillator) - Weight: 0.5
- **MOM** (Momentum) - Weight: 0.5
- **PPO** (Percentage Price Oscillator) - Weight: 0.5
- **APO** (Absolute Price Oscillator) - Weight: 1.0
- **ROC** (Rate of Change) - Weight: 0.5
- **KDJ** (Stochastic with J line) - Weight: 1.0

### Directional Indicators (3)
- **ADX** (Average Directional Index) - Weight: 0.5
- **DMI** (Directional Movement Index) - Weight: 0.5
- **SAR** (Parabolic SAR) - Weight: 1.0

### Volume Indicators (1)
- **MFI** (Money Flow Index) - Weight: 1.0

### Other Indicators (2)
- **TRIX** (Triple Exponential MA) - Weight: 0.5
- **CAD** (Chande Momentum) - Weight: 1.0

See [references/indicators.md](references/indicators.md) for detailed indicator explanations.

## Usage Patterns

### Quick Analysis
For rapid assessment of a single cryptocurrency:

```
1. Call coingecko_get_historical_chart for target coin (7-30 days)
2. Convert data using coingecko_converter
3. Run ta_analyzer.analyze_all()
4. Present scoreTotal and tradeSignal to user
```

### Comparative Analysis
To compare multiple cryptocurrencies:

```
1. Call coingecko_compare_coins for target coins
2. For each coin:
   - Fetch historical chart data
   - Run technical analysis
   - Store results
3. Create comparison table with scores and signals
4. Identify strongest/weakest performers
```

### Deep Dive Analysis
For comprehensive assessment with context:

```
1. Fetch multiple timeframes (7d, 30d, 90d)
2. Run analysis on each timeframe
3. Check for signal agreement across timeframes
4. Review individual indicator signals for divergences
5. Cross-reference with market data (market cap, volume, dominance)
6. Provide detailed report with confidence levels
```

### Trend Monitoring
For ongoing market surveillance:

```
1. Fetch current data for watchlist
2. Run analysis on all coins
3. Filter for STRONG_UPTREND signals (score >= 7)
4. Rank by score descending
5. Present top opportunities with context
```

## Best Practices

### Data Quality
- **Always validate** data quality before analysis using validate_data_quality()
- Ensure minimum 100 data points (preferably 200+)
- Check for missing values or data gaps
- Use appropriate timeframe for user's trading strategy

### Interpretation Guidelines
- **Never rely on single indicator** - the power is in consensus
- **Consider market context** - indicators behave differently in trending vs ranging markets
- **Watch for divergences** - when price contradicts indicators, reversal may be coming
- **Volume confirms price** - MFI provides crucial validation
- **Multiple timeframes** - confirm signals across different periods

### Common Patterns

**High Conviction Bullish**:
- Score >= 7
- RSI between 30-70 (not overbought)
- MACD bullish crossover
- Price above key moving averages
- ADX > 25 (strong trend)
- MFI shows accumulation

**Trend Exhaustion Warning**:
- Score > 7 BUT RSI > 80 or MFI > 90
- Suggests overbought conditions despite bullish consensus
- Potential reversal or pullback incoming

**False Breakout**:
- Strong price move BUT ADX < 20
- Low volume (MFI neutral/bearish)
- Likely whipsaw or temporary spike

## Limitations

### CoinGecko Data Considerations
- CoinGecko provides price points, not true OHLC bars
- Converter approximates OHLC from adjacent prices
- Works well for trend analysis, less precise for intraday patterns

### Indicator Nature
- Most indicators are **lagging** - calculated from past data
- Can generate **whipsaws** in choppy, sideways markets
- **Overfitting risk** - too many indicators can cause analysis paralysis
- Market regime changes require adaptation

### Recommended Use Cases
✅ **Great for**: Trend identification, medium-term signals, portfolio screening  
✅ **Good for**: Entry/exit timing, risk assessment, comparative analysis  
⚠️ **Limited for**: High-frequency trading, precise intraday timing, ranging markets  
❌ **Avoid for**: News-driven moves, low-liquidity coins, extreme volatility events

## Advanced Techniques

### Custom Scoring Weights
Modify indicator weights based on market conditions:
- **Trending markets**: Increase weight of MACD, EMA, ADX
- **Ranging markets**: Increase weight of RSI, CCI, Stochastic
- **High volatility**: Increase weight of SAR, KAMA (adaptive indicators)

### Multi-Timeframe Confirmation
Analyze same coin across multiple timeframes:
```
- 7 days (short-term trend)
- 30 days (medium-term trend)  
- 90 days (long-term trend)
```

Strongest signals occur when all timeframes agree.

### Sector Analysis
Analyze multiple coins in same sector to identify:
- Sector-wide trends vs individual coin movements
- Relative strength leaders
- Laggard coins with catch-up potential

## Troubleshooting

### Issue: Score stuck at 0 or very low
**Cause**: Insufficient data or flat price action  
**Solution**: Fetch longer historical period or check data quality

### Issue: Conflicting signals across indicators
**Cause**: Market in transition or ranging  
**Solution**: Score will be neutral - wait for clearer direction

### Issue: High score but bearish user intuition
**Cause**: Indicators lag price, or news-driven move  
**Solution**: Cross-reference with market context, recent news, volume

### Issue: Analysis fails with NaN values
**Cause**: Insufficient data for indicator calculation  
**Solution**: Fetch minimum 100 data points, preferably 200+

## Integration with CoinGecko MCP

This skill is designed to work seamlessly with CoinGecko MCP tools:

**Primary Tools Used**:
- `coingecko_get_historical_chart` - Main data source
- `coingecko_get_price` - Quick current price checks
- `coingecko_compare_coins` - Multi-coin analysis
- `coingecko_get_market_data` - Context and validation

**Workflow Integration**:
1. User asks about a cryptocurrency
2. Use CoinGecko tools to fetch data
3. Convert to OHLCV format
4. Run technical analysis
5. Present results with context from market data

## Example Outputs

### Simple Analysis Response
```
Bitcoin Technical Analysis (7-day period)

📊 Overall Signal: STRONG_UPTREND
🎯 Confidence Score: 8.5/24.0
💰 Current Price: $45,234.56 (+3.45% 24h)

Key Indicators:
✅ RSI: BUY (38.2 - healthy level)
✅ MACD: BUY (bullish crossover)
✅ EMA: BUY (price above all EMAs)
✅ Volume: ACCUMULATION (MFI bullish)

Recommendation: Strong buy signal with healthy fundamentals. 
No overbought conditions detected.
```

### Comparative Analysis Response
```
Top 5 Cryptocurrencies by Technical Score (30-day analysis)

1. Solana (SOL): 9.0 - STRONG_UPTREND
   - All momentum indicators bullish
   - Strong volume confirmation
   
2. Ethereum (ETH): 7.5 - STRONG_UPTREND
   - Trending higher, minor overbought warning
   
3. Bitcoin (BTC): 5.0 - NEUTRAL
   - Consolidating after recent move
   
4. Cardano (ADA): 2.5 - DOWNTREND
   - Multiple bearish signals
   
5. XRP: 1.0 - DOWNTREND
   - Weak momentum and volume
```

## Related Resources

- **Indicator Details**: See [references/indicators.md](references/indicators.md)
- **Core Analysis Engine**: [scripts/ta_analyzer.py](scripts/ta_analyzer.py)
- **Data Converter**: [scripts/coingecko_converter.py](scripts/coingecko_converter.py)
