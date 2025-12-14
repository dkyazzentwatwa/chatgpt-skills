#!/usr/bin/env python3
"""
Ultimate Technical Analysis Engine (pure numpy/pandas, no TA-Lib)

What’s new vs the current ta_analyzer.py
----------------------------------------
1) Correct indicator math where it matters:
   - DMI/ADX: fixes +DM/-DM logic (common source of subtle bias)
   - RSI/MFI/Stoch: handles zero-division edge cases sanely (no “always 50” flat-spots)
2) Real MESA / MAMA+FAMA:
   - Implements Ehlers-style MAMA/FAMA (Hilbert Transform based) instead of a KAMA proxy.
3) Less “vague” scoring:
   - Separates *state* (trend bias) from *trigger* (fresh crossover/flip)
   - Adds regime awareness (trend vs range) to reduce oscillator whipsaw in strong trends
4) More useful output for ChatGPT:
   - Keeps legacy keys/shape (scoreTotal, tradeSignal, tradeTrigger, individualScores/Signals, currentPrice, priceChange24h)
   - Adds: indicatorValues, indicatorMeta, regime, warnings, confidence, normalizedScore, tradeSignalV2

Notes / guardrails
------------------
- Still a technical indicator consensus engine. No indicator is “truth”; it’s a probabilistic read.
- Designed to work with both crypto and stocks (any OHLCV).
- Assumes input is time-ordered ascending. If you pass a 'time' column, it will auto-sort.

License: You own your code; modify freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd


# =============================================================================
# Helpers
# =============================================================================

def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)


def _is_datetime_like(s: pd.Series) -> bool:
    return np.issubdtype(s.dtype, np.datetime64)


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Best-effort conversion of a 'time' series to pandas datetime."""
    s = series.copy()
    if _is_datetime_like(s):
        return s

    # Heuristic: CoinGecko uses ms timestamps; many sources use seconds.
    # If values are huge, assume ms.
    if pd.api.types.is_numeric_dtype(s):
        v = s.dropna()
        if len(v) == 0:
            return pd.to_datetime(s, errors="coerce")
        median = float(v.median())
        unit = "ms" if median > 1e12 else "s" if median > 1e9 else None
        if unit:
            return pd.to_datetime(s, unit=unit, errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def safe_last(s: pd.Series, default: float = np.nan) -> float:
    if s is None or len(s) == 0:
        return float(default)
    v = s.iloc[-1]
    if pd.isna(v) or np.isinf(v):
        return float(default)
    return float(v)


def safe_prev(s: pd.Series, default: float = np.nan) -> float:
    if s is None or len(s) < 2:
        return float(default)
    v = s.iloc[-2]
    if pd.isna(v) or np.isinf(v):
        return float(default)
    return float(v)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def crossed_above(fast: pd.Series, slow: pd.Series) -> bool:
    if len(fast) < 2 or len(slow) < 2:
        return False
    a0, a1 = fast.iloc[-2], fast.iloc[-1]
    b0, b1 = slow.iloc[-2], slow.iloc[-1]
    if pd.isna(a0) or pd.isna(a1) or pd.isna(b0) or pd.isna(b1):
        return False
    return (a0 <= b0) and (a1 > b1)


def crossed_below(fast: pd.Series, slow: pd.Series) -> bool:
    if len(fast) < 2 or len(slow) < 2:
        return False
    a0, a1 = fast.iloc[-2], fast.iloc[-1]
    b0, b1 = slow.iloc[-2], slow.iloc[-1]
    if pd.isna(a0) or pd.isna(a1) or pd.isna(b0) or pd.isna(b1):
        return False
    return (a0 >= b0) and (a1 < b1)


# =============================================================================
# Core smoothing
# =============================================================================

def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=period).mean()


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def wilder(s: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing ~= EMA with alpha=1/period."""
    return s.ewm(alpha=1 / period, adjust=False).mean()


def wma(s: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)

    def _w(x):
        return float(np.dot(x, weights) / weights.sum())

    return s.rolling(period, min_periods=period).apply(_w, raw=True)


# =============================================================================
# Indicators
# =============================================================================

def roc(close: pd.Series, period: int = 10) -> pd.Series:
    prev = close.shift(period)
    return (close / prev.replace(0, np.nan) - 1.0) * 100.0


def mom(close: pd.Series, period: int = 10) -> pd.Series:
    return close - close.shift(period)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI with correct handling of 0-loss / 0-gain edge cases."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = wilder(gain, period)
    avg_loss = wilder(loss, period)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    r = 100 - (100 / (1 + rs))

    # Edge cases:
    # - If avg_loss == 0 and avg_gain > 0 => RSI = 100
    # - If avg_gain == 0 and avg_loss > 0 => RSI = 0
    # - If both == 0 => RSI = 50
    both_zero = (avg_gain.abs() < 1e-12) & (avg_loss.abs() < 1e-12)
    loss_zero = (avg_loss.abs() < 1e-12) & (avg_gain.abs() >= 1e-12)
    gain_zero = (avg_gain.abs() < 1e-12) & (avg_loss.abs() >= 1e-12)

    r = r.where(~loss_zero, 100.0)
    r = r.where(~gain_zero, 0.0)
    r = r.where(~both_zero, 50.0)

    return r


def cmo(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    su = up.rolling(period, min_periods=period).sum()
    sd = dn.rolling(period, min_periods=period).sum()
    denom = (su + sd).replace(0, np.nan)
    out = 100 * (su - sd) / denom
    return out.fillna(0.0)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    ma = tp.rolling(period, min_periods=period).mean()

    def _mad(x):
        m = float(np.mean(x))
        return float(np.mean(np.abs(x - m)))

    mad = tp.rolling(period, min_periods=period).apply(_mad, raw=True)
    denom = (0.015 * mad).replace(0, np.nan)
    out = (tp - ma) / denom
    return out.fillna(0.0)


def aroon(high: pd.Series, low: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """Returns (aroon_down, aroon_up)."""
    def _pos_argmax(x):
        return int(np.argmax(x))  # 0 oldest, period-1 newest

    def _pos_argmin(x):
        return int(np.argmin(x))

    up_pos = high.rolling(period, min_periods=period).apply(_pos_argmax, raw=True)
    dn_pos = low.rolling(period, min_periods=period).apply(_pos_argmin, raw=True)

    aroon_up = 100.0 * (up_pos + 1.0) / period
    aroon_down = 100.0 * (dn_pos + 1.0) / period
    return aroon_down.fillna(0.0), aroon_up.fillna(0.0)


def apo(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return ema(close, fast) - ema(close, slow)


def ppo(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    slow_ema = ema(close, slow)
    out = (ema(close, fast) - slow_ema) / slow_ema.replace(0, np.nan) * 100.0
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """MFI with correct 0-flow edge cases."""
    tp = (high + low + close) / 3.0
    rmf = tp * volume

    tp_prev = tp.shift(1)
    pos = rmf.where(tp > tp_prev, 0.0)
    neg = rmf.where(tp < tp_prev, 0.0)

    pos_sum = pos.rolling(period, min_periods=period).sum()
    neg_sum = neg.rolling(period, min_periods=period).sum()

    # Money Flow Ratio
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + mfr))

    # Edge cases:
    both_zero = (pos_sum.abs() < 1e-12) & (neg_sum.abs() < 1e-12)
    neg_zero = (neg_sum.abs() < 1e-12) & (pos_sum.abs() >= 1e-12)
    pos_zero = (pos_sum.abs() < 1e-12) & (neg_sum.abs() >= 1e-12)

    out = out.where(~neg_zero, 100.0)
    out = out.where(~pos_zero, 0.0)
    out = out.where(~both_zero, 50.0)

    return out


def trima(close: pd.Series, period: int = 30) -> pd.Series:
    if period % 2 == 1:
        p1 = (period + 1) // 2
        p2 = p1
    else:
        p1 = period // 2
        p2 = p1 + 1
    return sma(sma(close, p1), p2)


def trix(close: pd.Series, period: int = 30) -> pd.Series:
    e1 = ema(close, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    out = (e3 / e3.shift(1).replace(0, np.nan) - 1.0) * 100.0
    return out


def t3(close: pd.Series, period: int = 5, vfactor: float = 0.7) -> pd.Series:
    """Tillson T3. vfactor typical ~0.7."""
    e1 = ema(close, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    e4 = ema(e3, period)
    e5 = ema(e4, period)
    e6 = ema(e5, period)

    v = float(vfactor)
    c1 = -v ** 3
    c2 = 3 * v ** 2 + 3 * v ** 3
    c3 = -6 * v ** 2 - 3 * v - 3 * v ** 3
    c4 = 1 + 3 * v + v ** 3 + 3 * v ** 2

    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def kama(close: pd.Series, period: int = 30, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    change = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period, min_periods=period).sum()
    er = (change / volatility.replace(0, np.nan)).fillna(0.0)

    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    out = close.copy().astype(float)
    if len(out) == 0:
        return out
    out.iloc[0] = close.iloc[0]
    for i in range(1, len(close)):
        a = sc.iloc[i]
        if pd.isna(a):
            out.iloc[i] = out.iloc[i - 1]
        else:
            out.iloc[i] = out.iloc[i - 1] + float(a) * (close.iloc[i] - out.iloc[i - 1])
    return out


def parabolic_sar(high: pd.Series, low: pd.Series, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
    """Classic Parabolic SAR (iterative)."""
    h = high.values.astype(float)
    l = low.values.astype(float)
    n = len(high)
    if n == 0:
        return pd.Series([], dtype=float)

    sar = np.zeros(n, dtype=float)

    # Initial trend guess from first two bars
    uptrend = True
    if n >= 2 and (h[1] + l[1]) < (h[0] + l[0]):
        uptrend = False

    af = float(acceleration)
    ep = h[0] if uptrend else l[0]
    sar[0] = l[0] if uptrend else h[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        sar_i = prev_sar + af * (ep - prev_sar)

        if uptrend:
            if i >= 2:
                sar_i = min(sar_i, l[i - 1], l[i - 2])
            else:
                sar_i = min(sar_i, l[i - 1])

            # flip?
            if l[i] < sar_i:
                uptrend = False
                sar_i = ep
                ep = l[i]
                af = float(acceleration)
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + float(acceleration), float(maximum))
        else:
            if i >= 2:
                sar_i = max(sar_i, h[i - 1], h[i - 2])
            else:
                sar_i = max(sar_i, h[i - 1])

            # flip?
            if h[i] > sar_i:
                uptrend = True
                sar_i = ep
                ep = h[i]
                af = float(acceleration)
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + float(acceleration), float(maximum))

        sar[i] = sar_i

    return pd.Series(sar, index=high.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return wilder(tr, period)


def dmi_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Standard DMI/ADX.

    Fix vs prior version:
    - Uses up_move vs down_move (not abs(low_diff)), preventing false suppression of +DM when lows rise sharply.
    """
    up_move = high.diff()
    down_move = low.shift(1) - low  # positive when today's low < yesterday's low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = true_range(high, low, close)

    atr_vals = wilder(tr, period)
    plus_dm_sm = wilder(plus_dm, period)
    minus_dm_sm = wilder(minus_dm, period)

    plus_di = 100.0 * (plus_dm_sm / atr_vals.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm_sm / atr_vals.replace(0, np.nan))

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    adx_vals = wilder(dx.fillna(0.0), period)

    return plus_di.fillna(0.0), minus_di.fillna(0.0), adx_vals.fillna(0.0)


def stochastic_kd(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 9,
    k_smooth: int = 3,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    ll = low.rolling(k_period, min_periods=k_period).min()
    hh = high.rolling(k_period, min_periods=k_period).max()
    denom = (hh - ll).replace(0, np.nan)
    k = 100.0 * (close - ll) / denom
    slowk = k.rolling(k_smooth, min_periods=k_smooth).mean()
    slowd = slowk.rolling(d_period, min_periods=d_period).mean()

    # When the range collapses (hh==ll), stochastic is conceptually “mid” (50), not 0.
    slowk = slowk.fillna(50.0)
    slowd = slowd.fillna(50.0)
    return slowk, slowd


# =============================================================================
# MESA (MAMA/FAMA) — Ehlers-style (Hilbert Transform based)
# =============================================================================

def mama_fama(close: pd.Series, fastlimit: float = 0.5, slowlimit: float = 0.05) -> Tuple[pd.Series, pd.Series]:
    """
    Mesa Adaptive Moving Average (MAMA) + Following Adaptive Moving Average (FAMA).

    This is an Ehlers-style implementation (same family as TA-Lib’s MAMA).
    It’s iterative because alpha depends on the instantaneous phase.

    Returns:
        (mama, fama)
    """
    price = close.values.astype(float)
    n = len(price)
    idx = close.index

    # Output arrays
    mama = np.full(n, np.nan, dtype=float)
    fama = np.full(n, np.nan, dtype=float)

    # Internal arrays (mostly for clarity)
    smooth = np.zeros(n, dtype=float)
    detrender = np.zeros(n, dtype=float)
    q1 = np.zeros(n, dtype=float)
    i1 = np.zeros(n, dtype=float)
    jI = np.zeros(n, dtype=float)
    jQ = np.zeros(n, dtype=float)
    i2 = np.zeros(n, dtype=float)
    q2 = np.zeros(n, dtype=float)
    re = np.zeros(n, dtype=float)
    im = np.zeros(n, dtype=float)
    period = np.zeros(n, dtype=float)
    phase = np.zeros(n, dtype=float)

    # Hilbert Transform constants (Ehlers)
    a = 0.0962
    b = 0.5769

    def ht(src: np.ndarray, i: int) -> float:
        # 0.0962*src[i] + 0.5769*src[i-2] - 0.5769*src[i-4] - 0.0962*src[i-6]
        if i < 6:
            return 0.0
        return a * src[i] + b * src[i - 2] - b * src[i - 4] - a * src[i - 6]

    # Initialize
    for i in range(n):
        # 4-3-2-1 WMA smoothing (Ehlers)
        if i >= 3:
            smooth[i] = (4 * price[i] + 3 * price[i - 1] + 2 * price[i - 2] + price[i - 3]) / 10.0
        else:
            smooth[i] = price[i]

        if i == 0:
            period[i] = 0.0
            mama[i] = price[i]
            fama[i] = price[i]
            continue

        adj = 0.075 * period[i - 1] + 0.54

        detrender[i] = ht(smooth, i) * adj
        q1[i] = ht(detrender, i) * adj
        i1[i] = detrender[i - 3] if i >= 3 else 0.0

        jI[i] = ht(i1, i) * adj
        jQ[i] = ht(q1, i) * adj

        i2_raw = i1[i] - jQ[i]
        q2_raw = q1[i] + jI[i]

        # Smooth I2/Q2
        i2[i] = 0.2 * i2_raw + 0.8 * i2[i - 1]
        q2[i] = 0.2 * q2_raw + 0.8 * q2[i - 1]

        # Homodyne discriminator
        re_raw = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im_raw = i2[i] * q2[i - 1] - q2[i] * i2[i - 1]
        re[i] = 0.2 * re_raw + 0.8 * re[i - 1]
        im[i] = 0.2 * im_raw + 0.8 * im[i - 1]

        # Compute period from Re/Im
        if abs(im[i]) > 1e-12 and abs(re[i]) > 1e-12:
            # angle in radians
            angle = np.arctan2(im[i], re[i])
            if abs(angle) > 1e-12:
                per = 2 * np.pi / angle  # radians -> period-ish
            else:
                per = period[i - 1]
        else:
            per = period[i - 1]

        # Clamp period to sane bounds and smooth
        per = _clamp(per, 6.0, 50.0)
        period[i] = 0.2 * per + 0.8 * period[i - 1]

        # Phase & delta-phase
        phase[i] = np.degrees(np.arctan2(q1[i], i1[i] if abs(i1[i]) > 1e-12 else 1e-12))
        delta_phase = phase[i - 1] - phase[i]
        if delta_phase < 1.0:
            delta_phase = 1.0

        alpha = fastlimit / delta_phase
        alpha = _clamp(alpha, slowlimit, fastlimit)

        # MAMA / FAMA recursion
        mama[i] = alpha * price[i] + (1 - alpha) * mama[i - 1]
        a2 = 0.5 * alpha
        fama[i] = a2 * mama[i] + (1 - a2) * fama[i - 1]

    return pd.Series(mama, index=idx), pd.Series(fama, index=idx)


# =============================================================================
# Scoring / configuration
# =============================================================================

@dataclass
class AnalyzerConfig:
    # Core periods / params
    rsi_period: int = 14
    cci_period: int = 14
    mfi_period: int = 14
    dmi_period: int = 14
    stoch_k: int = 9
    stoch_k_smooth: int = 3
    stoch_d: int = 3

    sma_short: int = 20
    sma_long: int = 50

    ema_short: int = 12
    ema_long: int = 26

    dema_period: int = 30
    trima_period: int = 30
    wma_period: int = 30
    kama_period: int = 30
    trix_period: int = 30
    t3_period: int = 5
    t3_vfactor: float = 0.7

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    mesa_fastlimit: float = 0.5
    mesa_slowlimit: float = 0.05

    sar_acceleration: float = 0.02
    sar_maximum: float = 0.2

    roc_period: int = 10
    mom_period: int = 10

    # Regime thresholds
    adx_trending: float = 25.0
    adx_ranging: float = 20.0

    # Dynamic weighting multipliers (mild; keep score scale familiar)
    trend_mult_trending: float = 1.15
    trend_mult_ranging: float = 0.85
    osc_mult_trending: float = 0.85
    osc_mult_ranging: float = 1.15

    # Volatility boost for adaptive indicators (SAR/KAMA/MESA)
    vol_window: int = 50
    vol_ratio_high: float = 1.5
    adaptive_vol_mult: float = 1.10

    # Trade trigger gating
    min_confidence_for_trigger: float = 0.60
    extreme_rsi: float = 80.0
    extreme_mfi: float = 90.0

    # Output
    round_score: int = 2


# Base weights aligned with your indicator guide (strong=1.0, medium=0.5)
WEIGHTS: Dict[str, float] = {
    # Strong (±1.0)
    "RSI": 1.0,
    "MACD": 1.0,
    "EMA": 1.0,
    "SMA": 1.0,
    "DEMA": 1.0,
    "MESA": 1.0,
    "SAR": 1.0,
    "CCI": 1.0,
    "AROON": 1.0,
    "APO": 1.0,
    "MFI": 1.0,
    "KDJ": 1.0,
    "CAD": 1.0,

    # Medium (±0.5)
    "ADX": 0.5,
    "DMI": 0.5,
    "CMO": 0.5,
    "KAMA": 0.5,
    "MOMI": 0.5,
    "PPO": 0.5,
    "ROC": 0.5,
    "TRIMA": 0.5,
    "TRIX": 0.5,
    "T3": 0.5,
    "WMA": 0.5,
}


CATEGORY: Dict[str, str] = {
    # Trend-following / moving average family
    "SMA": "trend",
    "EMA": "trend",
    "DEMA": "trend",
    "TRIMA": "trend",
    "WMA": "trend",
    "KAMA": "trend",
    "T3": "trend",
    "TRIX": "trend",
    "MESA": "trend",
    "SAR": "trend",
    "MACD": "trend",
    "APO": "trend",
    "PPO": "trend",
    "DMI": "trend",  # directional trend
    "ADX": "trend",  # trend strength (handled directionally in scoring)

    # Oscillators / mean reversion-ish
    "RSI": "osc",
    "CCI": "osc",
    "KDJ": "osc",
    "MFI": "osc",
    "CMO": "osc",

    # Momentum / speed
    "MOMI": "mom",
    "ROC": "mom",
    "CAD": "mom",
    "AROON": "mom",  # trend timing / momentum-ish
}


# =============================================================================
# Technical Analyzer
# =============================================================================

class TechnicalAnalyzer:
    """
    Multi-indicator technical analysis system.

    Backwards-compatible output keys:
      - scoreTotal, tradeSignal, tradeTrigger
      - currentPrice, priceChange24h
      - individualScores, individualSignals

    Added in "ultimate" version:
      - normalizedScore, confidence, regime, warnings
      - indicatorValues, indicatorMeta
      - tradeSignalV2 (a more symmetric / direction-aware signal)
    """

    INDICATOR_ORDER: List[str] = [
        "APO", "AROON", "CAD", "CMO", "CCI",
        "DEMA", "EMA", "MACD", "MFI", "MESA",
        "KAMA", "MOMI", "PPO", "RSI", "SAR",
        "SMA", "TRIMA", "TRIX", "T3", "WMA",
        "ADX", "DMI", "KDJ", "ROC",
    ]

    def __init__(self, ohlcv_data: pd.DataFrame, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()

        if not isinstance(ohlcv_data, pd.DataFrame):
            raise TypeError("ohlcv_data must be a pandas DataFrame")

        df = ohlcv_data.copy()

        # Normalize column names (common variants)
        rename_map = {}
        for c in df.columns:
            cl = str(c).lower()
            if cl in ("timestamp", "datetime", "date"):
                rename_map[c] = "time"
            elif cl in ("vol",):
                rename_map[c] = "volume"
        if rename_map:
            df = df.rename(columns=rename_map)

        # If 'time' exists, sort (robustness)
        if "time" in df.columns:
            t = _ensure_datetime(_to_series(df["time"]))
            df["time"] = t
            df = df.sort_values("time").reset_index(drop=True)

        # Ensure required columns exist
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Force numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with no close (can’t do TA)
        df = df.dropna(subset=["close"]).reset_index(drop=True)

        self.df = df

        self.open_s = _to_series(self.df["open"]).astype(float)
        self.high_s = _to_series(self.df["high"]).astype(float)
        self.low_s = _to_series(self.df["low"]).astype(float)
        self.close_s = _to_series(self.df["close"]).astype(float)
        self.volume_s = _to_series(self.df["volume"]).fillna(0.0).astype(float)

        # Back-compat numpy arrays
        self.open = self.open_s.values
        self.high = self.high_s.values
        self.low = self.low_s.values
        self.close = self.close_s.values
        self.volume = self.volume_s.values

        # Outputs
        self.score_total: float = 0.0
        self.signals: Dict[str, str] = {}
        self.scores: Dict[str, float] = {}
        self.values: Dict[str, Any] = {}
        self.meta: Dict[str, Any] = {}
        self.warnings: List[str] = []

        # Internal
        self._cache: Dict[str, Any] = {}
        self._regime: Dict[str, Any] = {}

    # ----------------------------
    # Public API
    # ----------------------------

    def analyze_all(self) -> Dict[str, Any]:
        """
        Run the full indicator suite and return a structured result.
        """
        self.score_total = 0.0
        self.signals = {}
        self.scores = {}
        self.values = {}
        self.meta = {}
        self.warnings = []
        self._cache = {}
        self._regime = {}

        if len(self.df) < 50:
            self.warnings.append(
                f"Only {len(self.df)} data points provided. Signals are less reliable (50 min; 100+ recommended)."
            )

        # Precompute regime (ADX/DMI + ATR% for volatility context)
        self._compute_regime()

        # Compute indicators
        for name in self.INDICATOR_ORDER:
            try:
                getattr(self, f"_ind_{name.lower()}")()
            except Exception as e:
                # Resilience: fail indicator -> HOLD/0
                self.signals[name] = "HOLD"
                self.scores[name] = 0.0
                self.values[name] = None
                self.meta[name] = {"error": str(e)}

        # Legacy-ish signal (kept for backward compatibility with your docs)
        trade_signal_legacy = self._trade_signal_legacy(self.score_total)

        # V2 signal (more symmetric; uses normalized score)
        norm = self._normalized_score(self.score_total)
        trade_signal_v2 = self._trade_signal_v2(norm)

        confidence = self._confidence(norm)

        # Trigger gating:
        # - require strong uptrend (legacy OR v2) + decent confidence
        # - avoid "late" overbought extremes per your guide
        rsi_v = self.values.get("RSI")
        mfi_v = self.values.get("MFI")
        overbought_extreme = (
            (isinstance(rsi_v, (int, float)) and not pd.isna(rsi_v) and rsi_v >= self.config.extreme_rsi)
            or (isinstance(mfi_v, (int, float)) and not pd.isna(mfi_v) and mfi_v >= self.config.extreme_mfi)
        )

        if overbought_extreme:
            self.warnings.append("Overbought extreme detected (RSI>=80 or MFI>=90): trend may be strong but risk of pullback is higher.")

        trade_trigger = bool(
            (trade_signal_v2 == "STRONG_UPTREND" or trade_signal_legacy == "STRONG_UPTREND")
            and (confidence >= self.config.min_confidence_for_trigger)
            and (not overbought_extreme)
        )

        # Price stats
        current_price = float(self.close_s.iloc[-1]) if len(self.close_s) else float("nan")
        price_change_24h = self._price_change_24h()

        out = {
            # legacy keys
            "scoreTotal": round(float(self.score_total), self.config.round_score),
            "tradeSignal": trade_signal_legacy,
            "tradeTrigger": trade_trigger,
            "currentPrice": current_price,
            "priceChange24h": round(float(price_change_24h), 2),

            "individualScores": {k: round(float(v), self.config.round_score) for k, v in self.scores.items()},
            "individualSignals": self.signals,

            # new keys (non-breaking adds)
            "normalizedScore": round(float(norm), 4),
            "confidence": round(float(confidence), 4),
            "tradeSignalV2": trade_signal_v2,
            "regime": self._regime,
            "warnings": self.warnings,
            "indicatorValues": self.values,
            "indicatorMeta": self.meta,
        }
        return out

    # ----------------------------
    # Regime helpers
    # ----------------------------

    def _get_cached(self, key: str, fn):
        if key in self._cache:
            return self._cache[key]
        v = fn()
        self._cache[key] = v
        return v

    def _compute_regime(self) -> None:
        plus_di, minus_di, adx_vals = self._get_cached(
            "dmi_adx",
            lambda: dmi_adx(self.high_s, self.low_s, self.close_s, period=self.config.dmi_period),
        )

        adx_cur = safe_last(adx_vals, np.nan)
        plus_cur = safe_last(plus_di, np.nan)
        minus_cur = safe_last(minus_di, np.nan)

        dmi_diff = plus_cur - minus_cur
        dmi_dir = _sign(dmi_diff)

        # ATR% for volatility
        atr_vals = self._get_cached(
            "atr",
            lambda: atr(self.high_s, self.low_s, self.close_s, period=self.config.dmi_period),
        )
        atr_cur = safe_last(atr_vals, np.nan)
        close_cur = safe_last(self.close_s, np.nan)
        atr_pct = float(atr_cur / close_cur) * 100.0 if close_cur and not pd.isna(atr_cur) and close_cur != 0 else np.nan

        # Rolling median ATR% to contextualize volatility
        atr_pct_series = (atr_vals / self.close_s.replace(0, np.nan)) * 100.0
        med = atr_pct_series.rolling(self.config.vol_window, min_periods=max(10, self.config.vol_window // 3)).median()
        atr_pct_med = safe_last(med, np.nan)
        vol_ratio = float(atr_pct / atr_pct_med) if (not pd.isna(atr_pct) and not pd.isna(atr_pct_med) and atr_pct_med != 0) else np.nan

        if not pd.isna(adx_cur) and adx_cur >= self.config.adx_trending:
            regime = "TRENDING"
        elif not pd.isna(adx_cur) and adx_cur <= self.config.adx_ranging:
            regime = "RANGING"
        else:
            regime = "TRANSITION"

        # Multipliers
        if regime == "TRENDING":
            trend_mult = self.config.trend_mult_trending
            osc_mult = self.config.osc_mult_trending
        elif regime == "RANGING":
            trend_mult = self.config.trend_mult_ranging
            osc_mult = self.config.osc_mult_ranging
        else:
            trend_mult = 1.0
            osc_mult = 1.0

        high_vol = bool((not pd.isna(vol_ratio)) and (vol_ratio >= self.config.vol_ratio_high))
        adaptive_mult = self.config.adaptive_vol_mult if high_vol else 1.0

        self._regime = {
            "regime": regime,
            "adx": round(float(adx_cur), 2) if not pd.isna(adx_cur) else None,
            "plusDI": round(float(plus_cur), 2) if not pd.isna(plus_cur) else None,
            "minusDI": round(float(minus_cur), 2) if not pd.isna(minus_cur) else None,
            "dmiDirection": "UP" if dmi_dir > 0 else "DOWN" if dmi_dir < 0 else "FLAT",
            "atrPct": round(float(atr_pct), 3) if not pd.isna(atr_pct) else None,
            "volRatio": round(float(vol_ratio), 3) if not pd.isna(vol_ratio) else None,
            "highVolatility": high_vol,
            "multipliers": {
                "trend": trend_mult,
                "osc": osc_mult,
                "adaptive": adaptive_mult,
            },
        }

    def _mult(self, indicator: str) -> float:
        cat = CATEGORY.get(indicator, "trend")
        m = float(self._regime.get("multipliers", {}).get(cat, 1.0))
        # Optional extra boost for adaptive indicators when vol is elevated
        if indicator in ("SAR", "KAMA", "MESA"):
            m *= float(self._regime.get("multipliers", {}).get("adaptive", 1.0))
        return m

    # ----------------------------
    # Score + signal utilities
    # ----------------------------

    def _add(self, name: str, raw_score: float, signal: str, value: Any = None, meta: Optional[Dict[str, Any]] = None) -> None:
        w = float(WEIGHTS.get(name, 0.0))
        m = float(self._mult(name))
        score = raw_score * w * m

        self.score_total += float(score)
        self.scores[name] = float(score)
        self.signals[name] = signal
        self.values[name] = value
        self.meta[name] = meta or {}

    def _score_state_trigger(self, state: int, trigger: int, state_weight: float = 0.6, trigger_weight: float = 0.4) -> float:
        """
        Combine ongoing state (-1/0/1) with a fresh trigger (-1/0/1) into [-1,1].
        """
        state = int(state)
        trigger = int(trigger)
        val = state_weight * state + trigger_weight * trigger
        return float(_clamp(val, -1.0, 1.0))

    def _signal_from_raw(self, raw: float, buy_th: float = 0.15, sell_th: float = -0.15) -> str:
        if raw >= buy_th:
            return "BUY"
        if raw <= sell_th:
            return "SELL"
        return "HOLD"

    # ----------------------------
    # Trade-signal mapping
    # ----------------------------

    def _trade_signal_legacy(self, score_total: float) -> str:
        # Matches your published thresholds
        if score_total >= 7:
            return "STRONG_UPTREND"
        if score_total >= 3:
            return "NEUTRAL"
        return "DOWNTREND"

    def _normalized_score(self, score_total: float) -> float:
        # Rough normalization based on base weight sum + mild multiplier headroom.
        base_max = float(sum(WEIGHTS.values()))
        # Allow for multipliers (but keep stable)
        headroom = 1.20
        denom = base_max * headroom if base_max > 0 else 1.0
        return float(_clamp(score_total / denom, -1.0, 1.0))

    def _trade_signal_v2(self, normalized: float) -> str:
        # Symmetric-ish: useful for shorts / bearish bias too.
        if normalized >= 0.35:
            return "STRONG_UPTREND"
        if normalized <= -0.15:
            return "DOWNTREND"
        return "NEUTRAL"

    def _confidence(self, normalized: float) -> float:
        """
        Confidence heuristic (0..1):
        - more agreement (|normalized|) -> higher
        - stronger trend (ADX) -> higher
        - too much volatility -> slightly lower
        - missing indicators -> lower
        """
        # coverage
        total = len(self.INDICATOR_ORDER)
        computed = sum(1 for k in self.INDICATOR_ORDER if k in self.signals and self.values.get(k, None) is not None)
        coverage = computed / total if total else 0.0

        adx = self._regime.get("adx")
        adx_score = _clamp((float(adx) if adx is not None else 0.0) / 50.0, 0.0, 1.0)

        atr_pct = self._regime.get("atrPct")
        # Gentle penalty past ~5% ATR (scale depends on timeframe; keep mild)
        vol_penalty = 0.0
        if atr_pct is not None:
            vol_penalty = _clamp((float(atr_pct) - 5.0) / 20.0, 0.0, 0.5)

        alignment = abs(float(normalized))

        conf = 0.40 * alignment + 0.25 * adx_score + 0.25 * coverage + 0.10 * (1.0 - vol_penalty)
        conf = _clamp(conf, 0.0, 1.0)
        return float(conf)

    # ----------------------------
    # Time-aware 24h change
    # ----------------------------

    def _price_change_24h(self) -> float:
        if len(self.close_s) < 2:
            return 0.0

        if "time" in self.df.columns:
            t = _ensure_datetime(_to_series(self.df["time"]))
            if t.isna().all():
                return self._price_change_by_bars(24)

            t_last = t.iloc[-1]
            target = t_last - pd.Timedelta(hours=24)
            # Find the latest index with time <= target
            # Using searchsorted on sorted times
            tt = t.values.astype("datetime64[ns]")
            pos = int(np.searchsorted(tt, np.datetime64(target), side="right") - 1)
            if pos >= 0:
                base = float(self.close_s.iloc[pos])
                if base != 0 and not pd.isna(base):
                    return (float(self.close_s.iloc[-1]) - base) / base * 100.0
            return 0.0

        # Fall back
        return self._price_change_by_bars(24)

    def _price_change_by_bars(self, bars: int) -> float:
        if len(self.close_s) <= bars:
            return 0.0
        base = float(self.close_s.iloc[-bars - 1])
        if base == 0 or pd.isna(base):
            return 0.0
        return (float(self.close_s.iloc[-1]) - base) / base * 100.0

    # =============================================================================
    # Individual indicator scorers
    # =============================================================================

    def _ind_apo(self) -> None:
        apo_vals = self._get_cached("apo", lambda: apo(self.close_s, fast=self.config.ema_short, slow=self.config.ema_long))
        cur = safe_last(apo_vals, np.nan)
        prev = safe_prev(apo_vals, np.nan)

        state = _sign(cur)
        trigger = _sign(cur - prev)  # acceleration proxy

        raw = self._score_state_trigger(state=state, trigger=trigger, state_weight=0.7, trigger_weight=0.3)
        sig = self._signal_from_raw(raw)

        self._add("APO", raw_score=raw, signal=sig, value=cur, meta={"prev": prev})

    def _ind_aroon(self) -> None:
        aroon_down, aroon_up = self._get_cached("aroon", lambda: aroon(self.high_s, self.low_s, period=self.config.cci_period))
        up = safe_last(aroon_up, np.nan)
        dn = safe_last(aroon_down, np.nan)

        # Strong thresholds still matter; otherwise use mild direction (difference)
        if up > 70 and up > dn:
            raw = 1.0
        elif dn > 70 and dn > up:
            raw = -1.0
        else:
            raw = _clamp((up - dn) / 100.0, -0.5, 0.5)

        sig = self._signal_from_raw(raw, buy_th=0.2, sell_th=-0.2)
        self._add("AROON", raw_score=raw, signal=sig, value={"up": up, "down": dn})

    def _ind_adx(self) -> None:
        plus_di, minus_di, adx_vals = self._cache.get("dmi_adx") or dmi_adx(self.high_s, self.low_s, self.close_s, period=self.config.dmi_period)
        # Ensure cache set even if accessed here first
        self._cache["dmi_adx"] = (plus_di, minus_di, adx_vals)

        adx_cur = safe_last(adx_vals, np.nan)
        plus_cur = safe_last(plus_di, np.nan)
        minus_cur = safe_last(minus_di, np.nan)
        dmi_dir = _sign(plus_cur - minus_cur)

        if pd.isna(adx_cur):
            raw = 0.0
        elif adx_cur >= self.config.adx_trending:
            raw = float(dmi_dir)  # trending *in* a direction
        elif adx_cur <= self.config.adx_ranging:
            raw = 0.0
        else:
            raw = 0.5 * float(dmi_dir)

        sig = self._signal_from_raw(raw, buy_th=0.4, sell_th=-0.4)
        self._add("ADX", raw_score=raw, signal=sig, value=adx_cur, meta={"plusDI": plus_cur, "minusDI": minus_cur})

    def _ind_cad(self) -> None:
        # CAD here = Chande Momentum “extremes” (implemented via CMO)
        cmo_vals = self._get_cached("cmo", lambda: cmo(self.close_s, period=self.config.rsi_period))
        cur = safe_last(cmo_vals, np.nan)

        # In ranging markets, extremes are mean-reversion signals.
        # In trending markets, treat sign as momentum confirmation.
        regime = self._regime.get("regime", "TRANSITION")
        if regime == "RANGING":
            if cur > 50:
                raw = -1.0
            elif cur < -50:
                raw = 1.0
            else:
                raw = 0.0
        else:
            raw = _clamp(cur / 100.0, -1.0, 1.0)

        sig = self._signal_from_raw(raw)
        self._add("CAD", raw_score=raw, signal=sig, value=cur)

    def _ind_cmo(self) -> None:
        cmo_vals = self._get_cached("cmo", lambda: cmo(self.close_s, period=self.config.rsi_period))
        cur = safe_last(cmo_vals, np.nan)
        prev = safe_prev(cmo_vals, np.nan)

        # Use sign + slope
        state = _sign(cur)
        trigger = _sign(cur - prev)
        raw = self._score_state_trigger(state, trigger, state_weight=0.6, trigger_weight=0.4)
        sig = self._signal_from_raw(raw)
        self._add("CMO", raw_score=raw, signal=sig, value=cur, meta={"prev": prev})

    def _ind_cci(self) -> None:
        cci_vals = self._get_cached("cci", lambda: cci(self.high_s, self.low_s, self.close_s, period=self.config.cci_period))
        cur = safe_last(cci_vals, np.nan)

        regime = self._regime.get("regime", "TRANSITION")
        if regime == "RANGING":
            if cur < -100:
                raw = 1.0
            elif cur > 100:
                raw = -1.0
            else:
                raw = 0.0
        else:
            # Momentum-style
            if cur > 100:
                raw = 0.75
            elif cur < -100:
                raw = -0.75
            else:
                raw = _clamp(cur / 200.0, -0.5, 0.5)

        sig = self._signal_from_raw(raw)
        self._add("CCI", raw_score=raw, signal=sig, value=cur)

    def _ind_dema(self) -> None:
        e1 = self._get_cached("ema_30", lambda: ema(self.close_s, self.config.dema_period))
        e2 = self._get_cached("ema_30_of_ema", lambda: ema(e1, self.config.dema_period))
        dema_vals = 2 * e1 - e2

        price = float(self.close_s.iloc[-1])
        prev_price = float(self.close_s.iloc[-2]) if len(self.close_s) >= 2 else price
        cur = safe_last(dema_vals, np.nan)
        prev = safe_prev(dema_vals, np.nan)

        state = _sign(price - cur)
        trigger = 0
        if (prev_price <= prev) and (price > cur):
            trigger = 1
        elif (prev_price >= prev) and (price < cur):
            trigger = -1

        raw = self._score_state_trigger(state, trigger, state_weight=0.7, trigger_weight=0.3)
        sig = self._signal_from_raw(raw)
        self._add("DEMA", raw_score=raw, signal=sig, value=cur, meta={"prev": prev})

    def _ind_dmi(self) -> None:
        plus_di, minus_di, adx_vals = self._cache.get("dmi_adx") or dmi_adx(self.high_s, self.low_s, self.close_s, period=self.config.dmi_period)
        self._cache["dmi_adx"] = (plus_di, minus_di, adx_vals)

        p = safe_last(plus_di, np.nan)
        m = safe_last(minus_di, np.nan)

        diff = p - m
        s = p + m
        strength = abs(diff) / s if s and not pd.isna(s) else 0.0
        raw = float(_sign(diff)) * float(_clamp(0.5 + 0.5 * strength, 0.0, 1.0))
        sig = self._signal_from_raw(raw, buy_th=0.3, sell_th=-0.3)
        self._add("DMI", raw_score=raw, signal=sig, value={"plusDI": p, "minusDI": m})

    def _ind_ema(self) -> None:
        ema_short = self._get_cached("ema_short", lambda: ema(self.close_s, self.config.ema_short))
        ema_long = self._get_cached("ema_long", lambda: ema(self.close_s, self.config.ema_long))

        s_cur, l_cur = safe_last(ema_short, np.nan), safe_last(ema_long, np.nan)
        state = _sign(s_cur - l_cur)

        trig = 1 if crossed_above(ema_short, ema_long) else -1 if crossed_below(ema_short, ema_long) else 0
        raw = self._score_state_trigger(state, trig)
        sig = self._signal_from_raw(raw, buy_th=0.25, sell_th=-0.25)
        self._add("EMA", raw_score=raw, signal=sig, value={"short": s_cur, "long": l_cur})

    def _ind_kama(self) -> None:
        kama_vals = self._get_cached("kama", lambda: kama(self.close_s, period=self.config.kama_period, fast=2, slow=30))
        cur = safe_last(kama_vals, np.nan)
        prev = safe_prev(kama_vals, np.nan)
        price = float(self.close_s.iloc[-1])

        state = _sign(price - cur)
        slope = _sign(cur - prev)

        # If price agrees with slope, stronger.
        raw = float(state) * (1.0 if state == slope and state != 0 else 0.5 if state != 0 else 0.0)
        sig = self._signal_from_raw(raw, buy_th=0.25, sell_th=-0.25)
        self._add("KAMA", raw_score=raw, signal=sig, value=cur, meta={"prev": prev})

    def _ind_kdj(self) -> None:
        slowk, slowd = self._get_cached(
            "stoch",
            lambda: stochastic_kd(self.high_s, self.low_s, self.close_s, k_period=self.config.stoch_k, k_smooth=self.config.stoch_k_smooth, d_period=self.config.stoch_d),
        )
        j = 3 * slowk - 2 * slowd
        j_cur = safe_last(j, np.nan)
        j_prev = safe_prev(j, np.nan)

        regime = self._regime.get("regime", "TRANSITION")
        if regime == "RANGING":
            # Mean reversion + confirmation by slope
            if j_cur < 20 and j_cur > j_prev:
                raw = 1.0
            elif j_cur > 80 and j_cur < j_prev:
                raw = -1.0
            else:
                raw = 0.0
        else:
            # Momentum: K>D bullish, K<D bearish
            k_cur, d_cur = safe_last(slowk, np.nan), safe_last(slowd, np.nan)
            state = _sign(k_cur - d_cur)
            trigger = _sign(j_cur - j_prev)
            raw = self._score_state_trigger(state, trigger, state_weight=0.7, trigger_weight=0.3)

        sig = self._signal_from_raw(raw)
        self._add("KDJ", raw_score=raw, signal=sig, value={"K": safe_last(slowk, np.nan), "D": safe_last(slowd, np.nan), "J": j_cur})

    def _ind_macd(self) -> None:
        macd_line, signal_line, hist = self._get_cached(
            "macd",
            lambda: macd(self.close_s, fast=self.config.macd_fast, slow=self.config.macd_slow, signal=self.config.macd_signal),
        )
        m_cur = safe_last(macd_line, np.nan)
        s_cur = safe_last(signal_line, np.nan)
        h_cur = safe_last(hist, np.nan)
        h_prev = safe_prev(hist, np.nan)

        state = _sign(m_cur - s_cur)
        trig = 1 if crossed_above(macd_line, signal_line) else -1 if crossed_below(macd_line, signal_line) else 0
        accel = _sign(h_cur - h_prev)

        # Stronger when histogram accelerates in same direction
        raw = self._score_state_trigger(state, trig, state_weight=0.6, trigger_weight=0.4)
        if state != 0 and accel == state:
            raw = _clamp(raw + 0.15 * state, -1.0, 1.0)

        sig = self._signal_from_raw(raw, buy_th=0.25, sell_th=-0.25)
        self._add("MACD", raw_score=raw, signal=sig, value={"macd": m_cur, "signal": s_cur, "hist": h_cur}, meta={"histPrev": h_prev})

    def _ind_mfi(self) -> None:
        mfi_vals = self._get_cached("mfi", lambda: mfi(self.high_s, self.low_s, self.close_s, self.volume_s, period=self.config.mfi_period))
        cur = safe_last(mfi_vals, np.nan)

        regime = self._regime.get("regime", "TRANSITION")
        if regime == "RANGING":
            if cur < 20:
                raw = 1.0
            elif cur > 80:
                raw = -1.0
            else:
                raw = 0.0
        else:
            # Momentum-style around 50, but treat extreme overbought as warning, not immediate sell.
            if cur >= 80:
                raw = 0.25
            elif cur >= 55:
                raw = 0.5
            elif cur <= 20:
                raw = -0.5
            elif cur <= 45:
                raw = -0.5
            else:
                raw = 0.0

        sig = self._signal_from_raw(raw)
        self._add("MFI", raw_score=raw, signal=sig, value=cur)

    def _ind_mesa(self) -> None:
        mama, fama = self._get_cached(
            "mama_fama",
            lambda: mama_fama(self.close_s, fastlimit=self.config.mesa_fastlimit, slowlimit=self.config.mesa_slowlimit),
        )
        m_cur, f_cur = safe_last(mama, np.nan), safe_last(fama, np.nan)
        state = _sign(m_cur - f_cur)
        trig = 1 if crossed_above(mama, fama) else -1 if crossed_below(mama, fama) else 0
        raw = self._score_state_trigger(state, trig)
        sig = self._signal_from_raw(raw, buy_th=0.25, sell_th=-0.25)
        self._add("MESA", raw_score=raw, signal=sig, value={"mama": m_cur, "fama": f_cur})

    def _ind_momi(self) -> None:
        mom_vals = self._get_cached("mom", lambda: mom(self.close_s, period=self.config.mom_period))
        cur = safe_last(mom_vals, np.nan)
        prev = safe_prev(mom_vals, np.nan)
        state = _sign(cur)
        trig = _sign(cur - prev)
        raw = self._score_state_trigger(state, trig, state_weight=0.65, trigger_weight=0.35)
        sig = self._signal_from_raw(raw)
        self._add("MOMI", raw_score=raw, signal=sig, value=cur, meta={"prev": prev})

    def _ind_ppo(self) -> None:
        ppo_vals = self._get_cached("ppo", lambda: ppo(self.close_s, fast=self.config.ema_short, slow=self.config.ema_long))
        cur = safe_last(ppo_vals, np.nan)
        prev = safe_prev(ppo_vals, np.nan)
        state = _sign(cur)
        trig = _sign(cur - prev)
        raw = self._score_state_trigger(state, trig, state_weight=0.7, trigger_weight=0.3)
        sig = self._signal_from_raw(raw)
        self._add("PPO", raw_score=raw, signal=sig, value=cur, meta={"prev": prev})

    def _ind_roc(self) -> None:
        roc_vals = self._get_cached("roc", lambda: roc(self.close_s, period=self.config.roc_period))
        cur = safe_last(roc_vals, np.nan)
        prev = safe_prev(roc_vals, np.nan)
        state = _sign(cur)
        trig = _sign(cur - prev)
        raw = self._score_state_trigger(state, trig, state_weight=0.65, trigger_weight=0.35)
        sig = self._signal_from_raw(raw)
        self._add("ROC", raw_score=raw, signal=sig, value=cur, meta={"prev": prev})

    def _ind_rsi(self) -> None:
        rsi_vals = self._get_cached("rsi", lambda: rsi(self.close_s, period=self.config.rsi_period))
        cur = safe_last(rsi_vals, np.nan)

        regime = self._regime.get("regime", "TRANSITION")
        if regime == "RANGING":
            if cur < 30:
                raw = 1.0
            elif cur > 70:
                raw = -1.0
            else:
                raw = 0.0
        else:
            # Momentum-style around 50. Treat >70 as “strong but stretched”.
            if cur >= 70:
                raw = 0.5
            elif cur >= 55:
                raw = 0.5
            elif cur <= 30:
                raw = -1.0
            elif cur <= 45:
                raw = -0.5
            else:
                raw = 0.0

        sig = self._signal_from_raw(raw)
        self._add("RSI", raw_score=raw, signal=sig, value=cur)

    def _ind_sar(self) -> None:
        sar_vals = self._get_cached(
            "sar",
            lambda: parabolic_sar(self.high_s, self.low_s, acceleration=self.config.sar_acceleration, maximum=self.config.sar_maximum),
        )
        cur = safe_last(sar_vals, np.nan)
        prev = safe_prev(sar_vals, np.nan)
        price = float(self.close_s.iloc[-1])
        prev_price = float(self.close_s.iloc[-2]) if len(self.close_s) >= 2 else price

        state = _sign(price - cur)
        trig = 0
        if (prev_price <= prev) and (price > cur):
            trig = 1
        elif (prev_price >= prev) and (price < cur):
            trig = -1

        raw = self._score_state_trigger(state, trig, state_weight=0.7, trigger_weight=0.3)
        sig = self._signal_from_raw(raw, buy_th=0.25, sell_th=-0.25)
        self._add("SAR", raw_score=raw, signal=sig, value=cur, meta={"prev": prev})

    def _ind_sma(self) -> None:
        sma_short = self._get_cached("sma_short", lambda: sma(self.close_s, self.config.sma_short))
        sma_long = self._get_cached("sma_long", lambda: sma(self.close_s, self.config.sma_long))

        s_cur, l_cur = safe_last(sma_short, np.nan), safe_last(sma_long, np.nan)
        state = _sign(s_cur - l_cur)
        trig = 1 if crossed_above(sma_short, sma_long) else -1 if crossed_below(sma_short, sma_long) else 0

        raw = self._score_state_trigger(state, trig)
        sig = self._signal_from_raw(raw, buy_th=0.25, sell_th=-0.25)
        self._add("SMA", raw_score=raw, signal=sig, value={"short": s_cur, "long": l_cur})

    def _ind_trima(self) -> None:
        trima_vals = self._get_cached("trima", lambda: trima(self.close_s, period=self.config.trima_period))
        cur = safe_last(trima_vals, np.nan)
        prev = safe_prev(trima_vals, np.nan)
        price = float(self.close_s.iloc[-1])

        state = _sign(price - cur)
        slope = _sign(cur - prev)
        raw = float(state) * (1.0 if state == slope and state != 0 else 0.5 if state != 0 else 0.0)
        sig = self._signal_from_raw(raw, buy_th=0.25, sell_th=-0.25)
        self._add("TRIMA", raw_score=raw, signal=sig, value=cur, meta={"prev": prev})

    def _ind_trix(self) -> None:
        trix_vals = self._get_cached("trix", lambda: trix(self.close_s, period=self.config.trix_period))
        cur = safe_last(trix_vals, np.nan)
        prev = safe_prev(trix_vals, np.nan)
        state = _sign(cur)
        trig = _sign(cur - prev)
        raw = self._score_state_trigger(state, trig)
        sig = self._signal_from_raw(raw)
        self._add("TRIX", raw_score=raw, signal=sig, value=cur, meta={"prev": prev})

    def _ind_t3(self) -> None:
        t3_vals = self._get_cached("t3", lambda: t3(self.close_s, period=self.config.t3_period, vfactor=self.config.t3_vfactor))
        cur = safe_last(t3_vals, np.nan)
        prev = safe_prev(t3_vals, np.nan)
        price = float(self.close_s.iloc[-1])
        prev_price = float(self.close_s.iloc[-2]) if len(self.close_s) >= 2 else price

        state = _sign(price - cur)
        trig = 0
        if (prev_price <= prev) and (price > cur):
            trig = 1
        elif (prev_price >= prev) and (price < cur):
            trig = -1

        raw = self._score_state_trigger(state, trig)
        sig = self._signal_from_raw(raw, buy_th=0.25, sell_th=-0.25)
        self._add("T3", raw_score=raw, signal=sig, value=cur, meta={"prev": prev})

    def _ind_wma(self) -> None:
        wma_vals = self._get_cached("wma", lambda: wma(self.close_s, period=self.config.wma_period))
        cur = safe_last(wma_vals, np.nan)
        prev = safe_prev(wma_vals, np.nan)
        price = float(self.close_s.iloc[-1])

        state = _sign(price - cur)
        slope = _sign(cur - prev)
        raw = float(state) * (1.0 if state == slope and state != 0 else 0.5 if state != 0 else 0.0)
        sig = self._signal_from_raw(raw, buy_th=0.25, sell_th=-0.25)
        self._add("WMA", raw_score=raw, signal=sig, value=cur, meta={"prev": prev})


# =============================================================================
# Convenience functions
# =============================================================================

def analyze_from_json(json_data: str, config: Optional[AnalyzerConfig] = None) -> Dict[str, Any]:
    """
    Analyze technical indicators from JSON OHLCV data.

    Expected JSON: list[dict] with keys open/high/low/close/volume (and optional time)
    """
    data = json.loads(json_data)
    df = pd.DataFrame(data)
    analyzer = TechnicalAnalyzer(df, config=config)
    return analyzer.analyze_all()


if __name__ == "__main__":
    print("Ultimate Technical Analysis Engine (pure numpy/pandas)")
    print("This module is meant to be imported and run against OHLCV data.")
