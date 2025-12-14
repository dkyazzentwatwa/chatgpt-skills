# CLAUDE.md - Crypto TA Analyzer

## Overview

This is a comprehensive cryptocurrency and stock technical analysis skill using 29 indicators. It generates 7-tier trading signals (STRONG_BUY to STRONG_SELL) with divergence detection, volume confirmation, and Bollinger Band squeeze alerts.

## Quick Start

```python
# Activate virtual environment
source .venv/bin/activate

# Run analysis
from scripts.data_converter import normalize_ohlcv
from scripts.ta_analyzer import TechnicalAnalyzer

ohlcv_df, metadata = normalize_ohlcv(raw_data, source="auto")
analyzer = TechnicalAnalyzer(ohlcv_df)
results = analyzer.analyze_all()
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/ta_analyzer.py` | Main analysis engine (29 indicators) |
| `scripts/data_converter.py` | Generic data converter (CoinGecko, exchanges, Yahoo) |
| `scripts/coingecko_converter.py` | Legacy CoinGecko converter (backward compatible) |
| `SKILL.md` | Full skill documentation |
| `references/indicators.md` | Detailed indicator reference |

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
```

## Output Keys

**Primary signals:**
- `tradeSignal7Tier`: STRONG_BUY, BUY, WEAK_BUY, NEUTRAL, WEAK_SELL, SELL, STRONG_SELL
- `tradeSignal`: Legacy (STRONG_UPTREND, NEUTRAL, DOWNTREND)
- `confidence`: 0-1 confidence score
- `tradeTrigger`: Boolean for high-conviction entries

**New features:**
- `divergences`: RSI/MACD/OBV divergence detection
- `squeezeDetected`: Bollinger Band squeeze alert
- `volumeConfirmation`: Volume agreement score

## Indicators (29)

**Core (1.0):** RSI, MACD, BB, OBV, ICHIMOKU, EMA, SMA, MFI, KDJ, SAR
**Strong (0.75):** DEMA, MESA, CCI, AROON, APO
**Supporting (0.5):** ADX, DMI, CMO, KAMA, MOMI, PPO, ROC, TRIMA, TRIX, T3, WMA, VWAP, ATR_SIGNAL, CAD

## Testing

```bash
# Fetch and analyze BTC
source .venv/bin/activate
python3 -c "
from scripts.data_converter import normalize_ohlcv
from scripts.ta_analyzer import TechnicalAnalyzer
import urllib.request, json, ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=7'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
data = json.loads(urllib.request.urlopen(req, timeout=30, context=ctx).read())

df, _ = normalize_ohlcv(data, source='coingecko')
results = TechnicalAnalyzer(df).analyze_all()
print(f'Signal: {results[\"tradeSignal7Tier\"]} ({results[\"confidence\"]*100:.0f}% confidence)')
"
```

## Development Notes

- Virtual environment in `.venv/` (not committed)
- Reports saved to `reports/` (not committed)
- All indicator algorithms are pure numpy/pandas (no TA-Lib dependency)
- Ichimoku uses crypto-optimized periods (10/30/60)
- Backward compatible with legacy output keys
