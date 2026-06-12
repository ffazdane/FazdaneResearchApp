"""
FazDane Analytics - Tier 3
Elliott Wave Analysis -- Ground-up implementation using mathematical models.

Mathematical foundations
========================
1.  SWING DETECTION -- Adaptive ATR-scaled local-extrema filter.
2.  REGIME CLASSIFICATION -- EMA, ADX, and RSI trend regime detection.
3.  PROBABILITY ENGINE -- Multi-candidate wave generation and ranking.
4.  WAVE RULES & MOMENTUM SCORING -- Inviolable rules, guidelines, Fibonacci
    ratio accuracy, and momentum peak/divergence checks.
5.  OPTIONS STRATEGY BIAS -- Wave-stage options trade recommendations.
6.  MULTI-TIMEFRAME ALIGNMENT -- Weekly, Daily, and 1H wave alignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from modules.base_module import FazDaneModule
from utils.universe_manager import (
    format_ticker_display,
    get_ticker_names,
    get_universe_names,
    render_universe_manager,
)

logger = logging.getLogger("Elliott_Wave_Analysis")

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS & ENUMS
# ═══════════════════════════════════════════════════════════════════════

TICKER_ALIASES = {"SPX": "^GSPC", "NDX": "^NDX", "RUT": "^RUT", "NASDAQ": "^IXIC"}

FIB = {
    "0.236": 0.236,
    "0.382": 0.382,
    "0.500": 0.500,
    "0.618": 0.618,
    "0.786": 0.786,
    "1.000": 1.000,
    "1.272": 1.272,
    "1.618": 1.618,
    "2.000": 2.000,
    "2.618": 2.618,
}


class WavePhase(Enum):
    """Current position within the Elliott Wave cycle."""
    BUILDING_WAVE_3 = "Building Wave 3"   # Waves 0-1-2 confirmed
    BUILDING_WAVE_4 = "Building Wave 4"   # Waves 0-1-2-3 confirmed
    BUILDING_WAVE_5 = "Building Wave 5"   # Waves 0-1-2-3-4 confirmed
    IMPULSE_COMPLETE = "Impulse Complete"  # Full 5-wave impulse done
    CORRECTIVE_ABC = "Corrective ABC"     # ABC correction in progress
    NONE = "No Pattern"


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SwingPoint:
    """A statistically significant price pivot."""
    bar_index: int        # Position in the DataFrame
    timestamp: datetime   # Date/time of the pivot
    price: float          # Price at the pivot
    kind: str             # "H" (swing high) or "L" (swing low)
    atr_at_pivot: float   # ATR value when this pivot was confirmed


@dataclass
class FibTarget:
    """A single Fibonacci projection target."""
    label: str
    price: float
    ratio: str        # e.g. "1.618x"
    description: str


@dataclass
class TargetZone:
    """A price zone defined by two Fibonacci levels."""
    label: str
    price_low: float
    price_high: float
    midpoint: float
    description: str


@dataclass
class RuleCheck:
    """Result of validating one of the three inviolable rules."""
    name: str
    passed: bool
    detail: str


@dataclass
class WaveAnalysis:
    """Complete result of an Elliott Wave scan."""
    phase: WavePhase
    direction: str                       # "Bullish" or "Bearish"
    swing_points: list[SwingPoint]       # The pivot sequence forming the wave
    wave_labels: list[str]               # ["0","1","2",...] matching swing_points
    rules: list[RuleCheck]               # Inviolable rule results
    guidelines: list[RuleCheck]          # Guideline (soft) checks
    quality_score: float                 # 0–100 composite quality
    target_zones: list[TargetZone]       # Projected price zones
    fib_targets: list[FibTarget]         # Individual Fib levels
    wave_ratios: dict                    # Measured Fib ratios between waves
    regime: str = "Consolidating/Rangebound"


# ═══════════════════════════════════════════════════════════════════════
# 1. DATA FETCHING & REGIME CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════

def _normalize_symbol(symbol: str) -> str:
    clean = symbol.strip().upper()
    return TICKER_ALIASES.get(clean, clean)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlc(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download OHLC data and compute ATR (Average True Range)."""
    symbol = _normalize_symbol(ticker)
    df = yf.download(
        symbol, period=period, interval=interval,
        auto_adjust=True, progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df[~df.index.duplicated(keep="last")].sort_index().dropna(how="any")
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # Ensure we have OHLC columns
    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            match = [c for c in df.columns if c.lower() == col.lower()]
            if match:
                df[col] = df[match[0]]
            else:
                return pd.DataFrame()

    # Compute ATR (14-period)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)
    close = df["Close"].values.astype(np.float64)

    tr = np.empty(len(df), dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    atr_period = 14
    atr = np.empty(len(df), dtype=np.float64)
    atr[:atr_period] = np.nan
    atr[atr_period - 1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(df)):
        atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period

    df["ATR"] = atr
    return df


def classify_regime(df: pd.DataFrame) -> dict:
    """Determine the trend strength and consolidation phase using EMA, ADX, and RSI."""
    if len(df) < 50:
        return {"regime": "Consolidating/Rangebound", "adx": 0.0, "rsi": 50.0}

    close = df["Close"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)
    n = len(df)

    # Compute EMAs
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    ema50 = df["Close"].ewm(span=50, adjust=False).mean()
    ema200 = df["Close"].ewm(span=200, adjust=False).mean()

    # Compute RSI (14)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))

    # Compute ADX (14)
    tr = np.zeros(n)
    p_dm = np.zeros(n)
    m_dm = np.zeros(n)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            p_dm[i] = up
        else:
            p_dm[i] = 0
        if down > up and down > 0:
            m_dm[i] = down
        else:
            m_dm[i] = 0

    atr_val = np.zeros(n)
    smoothed_p_dm = np.zeros(n)
    smoothed_m_dm = np.zeros(n)

    atr_val[13] = np.mean(tr[:14])
    smoothed_p_dm[13] = np.mean(p_dm[:14])
    smoothed_m_dm[13] = np.mean(m_dm[:14])

    for i in range(14, n):
        atr_val[i] = (atr_val[i - 1] * 13 + tr[i]) / 14
        smoothed_p_dm[i] = (smoothed_p_dm[i - 1] * 13 + p_dm[i]) / 14
        smoothed_m_dm[i] = (smoothed_m_dm[i - 1] * 13 + m_dm[i]) / 14

    p_di = 100 * smoothed_p_dm / np.where(atr_val == 0, 1e-9, atr_val)
    m_di = 100 * smoothed_m_dm / np.where(atr_val == 0, 1e-9, atr_val)

    dx = 100 * abs(p_di - m_di) / np.where((p_di + m_di) == 0, 1e-9, (p_di + m_di))

    adx = np.zeros(n)
    adx[27] = np.mean(dx[14:28])
    for i in range(28, n):
        adx[i] = (adx[i - 1] * 13 + dx[i]) / 14

    latest_close = close[-1]
    latest_ema200 = ema200.iloc[-1]
    latest_adx = adx[-1]
    latest_rsi = rsi.iloc[-1]

    # Classification logic
    if latest_adx > 22:
        if latest_close > latest_ema200:
            regime = "Trending Bullish"
        else:
            regime = "Trending Bearish"
    else:
        regime = "Consolidating/Rangebound"

    return {
        "regime": regime,
        "adx": latest_adx,
        "rsi": latest_rsi,
        "ema20": ema20.iloc[-1],
        "ema50": ema50.iloc[-1],
        "ema200": latest_ema200,
    }


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators (RSI and MACD) for momentum verification."""
    df = df.copy()
    if len(df) < 30:
        df["RSI"] = 50.0
        df["MACD_Hist"] = 0.0
        return df

    # RSI (14)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. SWING DETECTION — ATR-Scaled Adaptive Pivots
# ═══════════════════════════════════════════════════════════════════════

def detect_swings(
    df: pd.DataFrame,
    atr_multiplier: float = 2.0,
    min_bars_between: int = 5,
) -> list[SwingPoint]:
    """Detect significant swing highs and lows using ATR as the volatility normaliser."""
    close = df["Close"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)
    atr = df["ATR"].values.astype(np.float64)
    n = len(close)

    if n < 30:
        return []

    start = max(14, min_bars_between)
    swings: list[SwingPoint] = []

    extreme_idx = start
    extreme_price = high[start]
    trend = "up"

    look_ahead = min(start + 20, n)
    initial_high = np.max(high[start:look_ahead])
    initial_low = np.min(low[start:look_ahead])
    if initial_high - close[start] > close[start] - initial_low:
        trend = "up"
        extreme_price = high[start]
    else:
        trend = "down"
        extreme_price = low[start]

    for i in range(start + 1, n):
        current_atr = atr[i] if not np.isnan(atr[i]) else atr[i - 1]
        if np.isnan(current_atr) or current_atr <= 0:
            continue

        threshold = atr_multiplier * current_atr

        if trend == "up":
            if high[i] > extreme_price:
                extreme_idx = i
                extreme_price = high[i]
            elif extreme_price - low[i] >= threshold:
                if (not swings) or (extreme_idx - swings[-1].bar_index >= min_bars_between):
                    swings.append(SwingPoint(
                        bar_index=extreme_idx,
                        timestamp=df.index[extreme_idx],
                        price=extreme_price,
                        kind="H",
                        atr_at_pivot=float(atr[extreme_idx] if not np.isnan(atr[extreme_idx]) else current_atr),
                    ))
                trend = "down"
                extreme_idx = i
                extreme_price = low[i]
        else:
            if low[i] < extreme_price:
                extreme_idx = i
                extreme_price = low[i]
            elif high[i] - extreme_price >= threshold:
                if (not swings) or (extreme_idx - swings[-1].bar_index >= min_bars_between):
                    swings.append(SwingPoint(
                        bar_index=extreme_idx,
                        timestamp=df.index[extreme_idx],
                        price=extreme_price,
                        kind="L",
                        atr_at_pivot=float(atr[extreme_idx] if not np.isnan(atr[extreme_idx]) else current_atr),
                    ))
                trend = "up"
                extreme_idx = i
                extreme_price = high[i]

    # Add trailing pivot
    if swings:
        trailing_kind = "H" if swings[-1].kind == "L" else "L"
        trailing_price = high[extreme_idx] if trailing_kind == "H" else low[extreme_idx]
        if extreme_idx - swings[-1].bar_index >= min_bars_between:
            swings.append(SwingPoint(
                bar_index=extreme_idx,
                timestamp=df.index[extreme_idx],
                price=trailing_price,
                kind=trailing_kind,
                atr_at_pivot=float(atr[extreme_idx] if not np.isnan(atr[extreme_idx]) else atr[-1]),
            ))

    # Clean: ensure strict alternation
    cleaned: list[SwingPoint] = []
    for sw in swings:
        if not cleaned or cleaned[-1].kind != sw.kind:
            cleaned.append(sw)
        elif sw.kind == "H" and sw.price >= cleaned[-1].price:
            cleaned[-1] = sw
        elif sw.kind == "L" and sw.price <= cleaned[-1].price:
            cleaned[-1] = sw
    return cleaned


# ═══════════════════════════════════════════════════════════════════════
# 3. SCORING ENGINE & PROBABILITY ENGINE
# ═══════════════════════════════════════════════════════════════════════

def _abs_wave_length(p_start: SwingPoint, p_end: SwingPoint) -> float:
    return abs(p_end.price - p_start.price)


def score_wave_count(
    df: pd.DataFrame,
    seq: list[SwingPoint],
    phase: WavePhase,
    direction: str,
    strict_mode: bool = True,
    diagonal_mode: bool = False
) -> tuple[float, list[RuleCheck], list[RuleCheck], dict]:
    """Score a candidate wave count structure (0-100 pts) based on rules, guidelines, and momentum."""
    p0 = seq[0]
    p1 = seq[1]
    p2 = seq[2]

    w1_len = abs(p1.price - p0.price)
    w2_len = abs(p2.price - p1.price)

    rules: list[RuleCheck] = []

    # Rule 1: Wave 2 never retraces 100% of Wave 1
    if direction == "Bullish":
        r1_passed = p2.price > p0.price
        r1_detail = f"Wave 2 Low ({p2.price:.2f}) > Wave 0 Low ({p0.price:.2f})"
    else:
        r1_passed = p2.price < p0.price
        r1_detail = f"Wave 2 High ({p2.price:.2f}) < Wave 0 High ({p0.price:.2f})"
    rules.append(RuleCheck("Rule 1: Wave 2 does not retrace 100% of Wave 1", r1_passed, r1_detail))

    # Rule 2: Wave 3 is never the shortest
    w3_len = 0.0
    w5_len = 0.0
    if len(seq) >= 4:
        w3_len = abs(seq[3].price - seq[2].price)
    if len(seq) >= 6:
        w5_len = abs(seq[5].price - seq[4].price)

    if len(seq) >= 6:
        r2_passed = w3_len >= min(w1_len, w5_len)
        r2_detail = f"W3 ({w3_len:.2f}) vs min(W1: {w1_len:.2f}, W5: {w5_len:.2f})"
    elif len(seq) >= 4:
        r2_passed = w3_len >= w1_len * 0.8
        r2_detail = f"Wave 3 ({w3_len:.2f}) vs Wave 1 ({w1_len:.2f})"
    else:
        r2_passed = True
        r2_detail = "N/A (requires Wave 3)"
    rules.append(RuleCheck("Rule 2: Wave 3 is not the shortest impulse wave", r2_passed, r2_detail))

    # Rule 3: Wave 4 does not overlap Wave 1
    if len(seq) >= 5:
        p4 = seq[4]
        if direction == "Bullish":
            if diagonal_mode:
                overlap_limit = p1.price - 0.15 * w1_len
                r3_passed = p4.price >= overlap_limit
                r3_detail = f"Wave 4 Low ({p4.price:.2f}) >= Diagonal Limit ({overlap_limit:.2f})"
            else:
                r3_passed = p4.price >= p1.price
                r3_detail = f"Wave 4 Low ({p4.price:.2f}) >= Wave 1 High ({p1.price:.2f})"
        else:
            if diagonal_mode:
                overlap_limit = p1.price + 0.15 * w1_len
                r3_passed = p4.price <= overlap_limit
                r3_detail = f"Wave 4 High ({p4.price:.2f}) <= Diagonal Limit ({overlap_limit:.2f})"
            else:
                r3_passed = p4.price <= p1.price
                r3_detail = f"Wave 4 High ({p4.price:.2f}) <= Wave 1 Low ({p1.price:.2f})"
    else:
        r3_passed = True
        r3_detail = "N/A (requires Wave 4)"
    rules.append(RuleCheck("Rule 3: Wave 4 does not overlap Wave 1 territory", r3_passed, r3_detail))

    rule_score = 0.0
    if r1_passed: rule_score += 10.0
    if r2_passed: rule_score += 10.0
    if r3_passed: rule_score += 10.0

    # Guidelines (30 points total, 6 pts each)
    guidelines: list[RuleCheck] = []
    ratios = {}

    r_w2w1 = w2_len / max(w1_len, 1e-9)
    ratios["W2/W1 Retracement"] = r_w2w1
    g1_passed = 0.382 <= r_w2w1 <= 0.786
    guidelines.append(RuleCheck("Wave 2 retracement 38.2%-78.6% of Wave 1", g1_passed, f"Measured: {r_w2w1:.3f} (ideal: 0.500-0.618)"))

    if len(seq) >= 4:
        r_w3w1 = w3_len / max(w1_len, 1e-9)
        ratios["W3/W1 Extension"] = r_w3w1
        g2_passed = r_w3w1 >= 1.0
        g2_detail = f"Measured: {r_w3w1:.3f} (ideal: 1.618)"
    else:
        g2_passed = False
        g2_detail = "N/A"
    guidelines.append(RuleCheck("Wave 3 extends >= 1.0x Wave 1", g2_passed, g2_detail))

    if len(seq) >= 5:
        w4_len = abs(seq[4].price - seq[3].price)
        r_w4w3 = w4_len / max(w3_len, 1e-9)
        ratios["W4/W3 Retracement"] = r_w4w3
        g3_passed = 0.236 <= r_w4w3 <= 0.500
        g3_detail = f"Measured: {r_w4w3:.3f} (ideal: 0.382)"
    else:
        g3_passed = False
        g3_detail = "N/A"
    guidelines.append(RuleCheck("Wave 4 retracement 23.6%-50% of Wave 3", g3_passed, g3_detail))

    if len(seq) >= 6:
        r_w5w1 = w5_len / max(w1_len, 1e-9)
        ratios["W5/W1 Ratio"] = r_w5w1
        g4_passed = 0.618 <= r_w5w1 <= 1.618
        g4_detail = f"Measured: {r_w5w1:.3f} (ideal: 1.000)"
    else:
        g4_passed = False
        g4_detail = "N/A"
    guidelines.append(RuleCheck("Wave 5 ~ Wave 1 in length (0.618x-1.618x)", g4_passed, g4_detail))

    if len(seq) >= 5:
        depth_diff = abs(r_w2w1 - r_w4w3)
        ratios["Alternation Differential"] = depth_diff
        g5_passed = depth_diff >= 0.10
        g5_detail = f"Measured: {depth_diff:.3f} (ideal: >= 0.10)"
    else:
        g5_passed = False
        g5_detail = "N/A"
    guidelines.append(RuleCheck("Alternation: Waves 2 and 4 differ in character", g5_passed, g5_detail))

    guide_score = 0.0
    for g in guidelines:
        if g.passed:
            guide_score += 6.0

    # Momentum confirmation (25 points max)
    momentum_score = 0.0
    has_indicators = "RSI" in df.columns
    if has_indicators and len(seq) >= 4:
        w1_idx_start = p0.bar_index
        w1_idx_end = p1.bar_index
        w3_idx_start = p2.bar_index
        w3_idx_end = seq[3].bar_index

        # Wave 3 peak check
        if direction == "Bullish":
            w1_max_rsi = df["RSI"].iloc[w1_idx_start:w1_idx_end+1].max()
            w3_max_rsi = df["RSI"].iloc[w3_idx_start:w3_idx_end+1].max()
            w3_mom_ok = w3_max_rsi > w1_max_rsi or w3_max_rsi > 65.0
        else:
            w1_min_rsi = df["RSI"].iloc[w1_idx_start:w1_idx_end+1].min()
            w3_min_rsi = df["RSI"].iloc[w3_idx_start:w3_idx_end+1].min()
            w3_mom_ok = w3_min_rsi < w1_min_rsi or w3_min_rsi < 35.0

        if w3_mom_ok:
            momentum_score += 12.5

        # Wave 5 divergence check
        if len(seq) >= 6:
            w5_idx_start = seq[4].bar_index
            w5_idx_end = seq[5].bar_index
            if direction == "Bullish":
                w5_max_rsi = df["RSI"].iloc[w5_idx_start:w5_idx_end+1].max()
                div_ok = (seq[5].price > seq[3].price) and (w5_max_rsi < w3_max_rsi)
            else:
                w5_min_rsi = df["RSI"].iloc[w5_idx_start:w5_idx_end+1].min()
                div_ok = (seq[5].price < seq[3].price) and (w5_min_rsi > w3_min_rsi)

            if div_ok:
                momentum_score += 12.5
            else:
                if direction == "Bullish" and w5_max_rsi > 60.0:
                    momentum_score += 5.0
                elif direction == "Bearish" and w5_min_rsi < 40.0:
                    momentum_score += 5.0
    else:
        momentum_score = 15.0 if len(seq) >= 4 else 5.0

    # Fibonacci Confluence (15 points max)
    fib_score = 0.0
    ideals = {
        "W2/W1 Retracement": 0.618,
        "W3/W1 Extension": 1.618,
        "W4/W3 Retracement": 0.382,
        "W5/W1 Ratio": 1.000,
    }
    present_ratios = 0
    total_acc = 0.0
    for key, ideal_val in ideals.items():
        if key in ratios:
            present_ratios += 1
            error = abs(ratios[key] - ideal_val) / max(ideal_val, 1e-9)
            accuracy = max(0.0, 1.0 - error)
            total_acc += accuracy
    if present_ratios > 0:
        fib_score = (total_acc / present_ratios) * 15.0
    else:
        fib_score = 10.0

    total_score = rule_score + guide_score + momentum_score + fib_score
    total_score = round(min(100.0, max(0.0, total_score)), 1)

    return total_score, rules, guidelines, ratios


# ═══════════════════════════════════════════════════════════════════════
# 4. FIBONACCI PROJECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def project_targets(seq: list[SwingPoint], phase: WavePhase, direction: str) -> tuple[list[TargetZone], list[FibTarget]]:
    """Generate forward-looking Fibonacci target zones based on the current wave phase."""
    sign = 1.0 if direction == "Bullish" else -1.0
    zones: list[TargetZone] = []
    targets: list[FibTarget] = []

    if phase == WavePhase.NONE or len(seq) < 3:
        return zones, targets

    p0, p1, p2 = seq[0], seq[1], seq[2]
    w1_len = _abs_wave_length(p0, p1)

    if phase == WavePhase.BUILDING_WAVE_3:
        for ratio_name, ratio_val in [("1.000x", 1.0), ("1.618x", 1.618), ("2.618x", 2.618)]:
            price = p2.price + sign * ratio_val * w1_len
            targets.append(FibTarget(f"Wave 3 ({ratio_name})", price, ratio_name, f"Wave 1 × {ratio_name} from Wave 2"))

        zone_lo = p2.price + sign * 1.0 * w1_len
        zone_hi = p2.price + sign * 2.618 * w1_len
        zones.append(TargetZone("Wave 3 Target Zone", min(zone_lo, zone_hi), max(zone_lo, zone_hi),
                                p2.price + sign * 1.618 * w1_len, "Fibonacci extension zone for Wave 3"))

    if phase in (WavePhase.BUILDING_WAVE_3, WavePhase.BUILDING_WAVE_4) and len(seq) >= 4:
        p3 = seq[3]
        w3_len = _abs_wave_length(p2, p3)
        for ratio_name, ratio_val in [("0.236x", 0.236), ("0.382x", 0.382), ("0.500x", 0.500)]:
            price = p3.price - sign * ratio_val * w3_len
            targets.append(FibTarget(f"Wave 4 ({ratio_name})", price, ratio_name, f"Wave 3 × {ratio_name} retracement"))

        zone_lo = p3.price - sign * 0.500 * w3_len
        zone_hi = p3.price - sign * 0.236 * w3_len
        zones.append(TargetZone("Wave 4 Target Zone", min(zone_lo, zone_hi), max(zone_lo, zone_hi),
                                p3.price - sign * 0.382 * w3_len, "Fibonacci retracement zone for Wave 4"))

    if phase in (WavePhase.BUILDING_WAVE_4, WavePhase.BUILDING_WAVE_5) and len(seq) >= 5:
        p3 = seq[3]
        p4 = seq[4]
        w5_eq = p4.price + sign * w1_len
        w5_fib = p4.price + sign * 0.618 * abs(p3.price - p0.price)
        targets.append(FibTarget("Wave 5 (W1 equality)", w5_eq, "1.0x W1", "Wave 5 = Wave 1 in length"))
        targets.append(FibTarget("Wave 5 (0.618 × W0-W3)", w5_fib, "0.618x", "Fibonacci projection from Wave 4"))

        zone_lo = min(w5_eq, w5_fib)
        zone_hi = max(w5_eq, w5_fib)
        zones.append(TargetZone("Wave 5 Target Zone", zone_lo, zone_hi, (zone_lo + zone_hi) / 2.0,
                                "Projected completion zone for Wave 5"))

    if phase == WavePhase.IMPULSE_COMPLETE and len(seq) >= 6:
        p5 = seq[5]
        total = abs(p5.price - p0.price)
        for ratio_name, ratio_val in [("0.382x", 0.382), ("0.500x", 0.500), ("0.618x", 0.618)]:
            price = p5.price - sign * ratio_val * total
            targets.append(FibTarget(f"Wave A ({ratio_name})", price, ratio_name, f"Impulse retracement"))

        zone_lo = p5.price - sign * 0.618 * total
        zone_hi = p5.price - sign * 0.382 * total
        zones.append(TargetZone("Corrective Zone (A)", min(zone_lo, zone_hi), max(zone_lo, zone_hi),
                                p5.price - sign * 0.500 * total, "Expected Wave A target zone"))

    return zones, targets


# ═══════════════════════════════════════════════════════════════════════
# 5. WAVE SCANNER & PROBABILITY ENGINE
# ═══════════════════════════════════════════════════════════════════════

def _determine_direction(seq: list[SwingPoint]) -> Optional[str]:
    if len(seq) < 2:
        return None
    delta = seq[1].price - seq[0].price
    if abs(delta) < 1e-9:
        return None
    return "Bullish" if delta > 0 else "Bearish"


def _alternation_check(seq: list[SwingPoint]) -> bool:
    for i in range(1, len(seq)):
        if seq[i].kind == seq[i - 1].kind:
            return False
    return True


def _direction_check(seq: list[SwingPoint], direction: str) -> bool:
    n = len(seq)
    if direction == "Bullish":
        for i in range(n - 1):
            delta = seq[i + 1].price - seq[i].price
            expected_up = (i % 2 == 0)
            if expected_up and delta <= 0:
                return False
            if not expected_up and delta >= 0:
                return False
    else:
        for i in range(n - 1):
            delta = seq[i + 1].price - seq[i].price
            expected_down = (i % 2 == 0)
            if expected_down and delta >= 0:
                return False
            if not expected_down and delta <= 0:
                return False
    return True


def scan_waves_multi_count(
    df: pd.DataFrame,
    swings: list[SwingPoint],
    strict_mode: bool = True,
    diagonal_mode: bool = False,
    pinned_p0: Optional[SwingPoint] = None
) -> list[WaveAnalysis]:
    """Scan from newest swings backwards and generate ranked candidate wave counts."""
    n = len(swings)
    if n < 3:
        return []

    configs = [
        (6, WavePhase.IMPULSE_COMPLETE, ["0", "1", "2", "3", "4", "5"]),
        (5, WavePhase.BUILDING_WAVE_5, ["0", "1", "2", "3", "4"]),
        (4, WavePhase.BUILDING_WAVE_4, ["0", "1", "2", "3"]),
        (3, WavePhase.BUILDING_WAVE_3, ["0", "1", "2"]),
    ]

    candidates = []
    df_ind = compute_indicators(df)
    reg_dict = classify_regime(df)

    for num_pts, phase, labels in configs:
        for start in range(n - num_pts, -1, -1):
            seq = swings[start : start + num_pts]

            if pinned_p0 is not None:
                if seq[0].bar_index != pinned_p0.bar_index or seq[0].kind != pinned_p0.kind:
                    continue

            if not _alternation_check(seq):
                continue
            direction = _determine_direction(seq)
            if direction is None:
                continue
            if not _direction_check(seq, direction):
                continue

            score, rules, guidelines, ratios = score_wave_count(df_ind, seq, phase, direction, strict_mode, diagonal_mode)

            rules_passed = all(r.passed for r in rules)
            if strict_mode and not rules_passed:
                continue

            zones, fib_targets = project_targets(seq, phase, direction)

            candidates.append(WaveAnalysis(
                phase=phase,
                direction=direction,
                swing_points=list(seq),
                wave_labels=labels[:num_pts],
                rules=rules,
                guidelines=guidelines,
                quality_score=score,
                target_zones=zones,
                fib_targets=fib_targets,
                wave_ratios=ratios,
                regime=reg_dict["regime"]
            ))

    # Sort candidates by score descending
    candidates.sort(key=lambda x: x.quality_score, reverse=True)
    return candidates


def scan_waves(swings: list[SwingPoint], df: Optional[pd.DataFrame] = None) -> Optional[WaveAnalysis]:
    """Legacy compatibility wrapper that returns the top candidate."""
    if df is None:
        prices = [sw.price for sw in swings]
        dates = [sw.timestamp for sw in swings]
        df = pd.DataFrame({"Close": prices, "High": prices, "Low": prices, "Open": prices}, index=dates)
        # Add basic ATR
        df["ATR"] = 1.0

    candidates = scan_waves_multi_count(df, swings, strict_mode=False)
    return candidates[0] if candidates else None


# ═══════════════════════════════════════════════════════════════════════
# 6. LIFECYCLE & OPTIONS STRATEGY RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════

def classify_setup_and_options_strategy(analysis: Optional[WaveAnalysis]) -> dict:
    """Map the current active wave and regime to options strategy recommendations."""
    if analysis is None or analysis.phase == WavePhase.NONE:
        return {
            "setup": "No active setup detected",
            "rank": 5,
            "stage": "Neutral / Consolidation",
            "entry_zone": "N/A",
            "invalidation": "N/A",
            "target_zone": "N/A",
            "strategy": "Avoid selling premium or wait for pivot",
            "confidence": 0.0
        }

    phase = analysis.phase
    direction = analysis.direction
    ratios = analysis.wave_ratios
    pts = analysis.swing_points
    score = analysis.quality_score

    inval_price = 0.0
    if len(pts) > 0:
        if phase == WavePhase.BUILDING_WAVE_3:
            inval_price = pts[0].price
        elif phase == WavePhase.BUILDING_WAVE_4:
            inval_price = pts[1].price
        elif phase == WavePhase.BUILDING_WAVE_5:
            inval_price = pts[4].price
        elif phase == WavePhase.IMPULSE_COMPLETE:
            inval_price = pts[5].price

    target_str = "N/A"
    if analysis.target_zones:
        z = analysis.target_zones[0]
        target_str = f"${z.price_low:.2f} - ${z.price_high:.2f}"

    setup_name = "N/A"
    rank = 5
    strategy = "N/A"
    stage = "N/A"
    entry_str = "N/A"

    if phase == WavePhase.BUILDING_WAVE_3:
        w2_retr = ratios.get("W2/W1 Retracement", 0.0)
        if 0.500 <= w2_retr <= 0.786 and pts[-1].kind == "L" and direction == "Bullish":
            setup_name = "Wave 2 ending, possible Wave 3 start"
            rank = 1
            stage = "Accumulation / Early Stage"
            strategy = "Long Call Spread / Bull Put Spread"
            entry_str = f"${pts[-1].price:.2f} - ${pts[-1].price * 1.02:.2f}"
        elif 0.500 <= w2_retr <= 0.786 and pts[-1].kind == "H" and direction == "Bearish":
            setup_name = "Wave 2 ending, possible Wave 3 start"
            rank = 1
            stage = "Distribution / Early Stage"
            strategy = "Long Put Spread / Bear Call Spread"
            entry_str = f"${pts[-1].price * 0.98:.2f} - ${pts[-1].price:.2f}"
        else:
            setup_name = "Early Wave 3 breakout"
            rank = 2
            stage = "Expansion / Strong Trend"
            strategy = "Long Call Spread / Buy Calls" if direction == "Bullish" else "Long Put Spread / Buy Puts"
            entry_str = f"Breakout past ${pts[1].price:.2f}"

    elif phase == WavePhase.BUILDING_WAVE_4:
        setup_name = "Wave 4 active correction"
        rank = 3
        stage = "Consolidation / Pullback"
        strategy = "Iron Condor / Iron Butterfly"
        entry_str = "Delta neutral range entry"

    elif phase == WavePhase.BUILDING_WAVE_5:
        w4_retr = ratios.get("W4/W3 Retracement", 0.0)
        if 0.236 <= w4_retr <= 0.500 and pts[-1].kind == "L" and direction == "Bullish":
            setup_name = "Wave 4 ending, possible Wave 5 start"
            rank = 3
            stage = "Pullback Complete"
            strategy = "Bull Put Spread / Long Call Spread"
            entry_str = f"${pts[-1].price:.2f} - ${pts[-1].price * 1.02:.2f}"
        elif 0.236 <= w4_retr <= 0.500 and pts[-1].kind == "H" and direction == "Bearish":
            setup_name = "Wave 4 ending, possible Wave 5 start"
            rank = 3
            stage = "Pullback Complete"
            strategy = "Bear Call Spread / Long Put Spread"
            entry_str = f"${pts[-1].price * 0.98:.2f} - ${pts[-1].price:.2f}"
        else:
            setup_name = "Wave 5 late-stage continuation"
            rank = 5
            stage = "Late Stage / Exhaustion"
            strategy = "Bear Call Spread / Long Puts" if direction == "Bullish" else "Bull Put Spread / Long Calls"
            entry_str = "Fade late expansion"

    elif phase == WavePhase.IMPULSE_COMPLETE:
        setup_name = "Completed ABC correction with reversal"
        rank = 4
        stage = "Correction / Transition"
        strategy = "Long Stock / Bull Put Spread" if direction == "Bullish" else "Short Stock / Bear Call Spread"
        entry_str = "Reversal from ABC baseline"

    return {
        "setup": setup_name,
        "rank": rank,
        "stage": stage,
        "entry_zone": entry_str,
        "invalidation": f"${inval_price:.2f}" if inval_price > 0 else "N/A",
        "target_zone": target_str,
        "strategy": strategy,
        "confidence": score
    }


# ═══════════════════════════════════════════════════════════════════════
# 7. CHART BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_chart(
    df: pd.DataFrame,
    swings: list[SwingPoint],
    analysis: Optional[WaveAnalysis],
    symbol: str,
    show_minor_pivots: bool,
    show_fib_levels: bool,
    plot_last_n: int,
) -> go.Figure:
    """Build a clean, professional Plotly chart for Wave Count display."""
    tail = df if plot_last_n <= 0 else df.iloc[-plot_last_n:]
    start_pos = len(df) - len(tail)
    end_pos = len(df) - 1

    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=tail.index,
        open=tail["Open"],
        high=tail["High"],
        low=tail["Low"],
        close=tail["Close"],
        name="OHLC",
        increasing_line_color="rgba(34, 197, 94, 0.6)",
        decreasing_line_color="rgba(239, 68, 68, 0.5)",
        increasing_fillcolor="rgba(34, 197, 94, 0.25)",
        decreasing_fillcolor="rgba(239, 68, 68, 0.2)",
        showlegend=False,
    ))

    # Minor swing pivots
    if show_minor_pivots:
        visible = [sw for sw in swings if start_pos <= sw.bar_index <= end_pos]
        if visible:
            fig.add_trace(go.Scatter(
                x=[sw.timestamp for sw in visible],
                y=[sw.price for sw in visible],
                mode="markers",
                name="Swing Pivots",
                marker=dict(size=5, color="rgba(148, 163, 184, 0.6)", symbol="diamond"),
                hovertemplate="%{y:.2f}<extra>Pivot</extra>",
            ))

    bar_delta = pd.Series(df.index).diff().median()
    if pd.isna(bar_delta):
        bar_delta = timedelta(days=1)

    # Main wave structure
    if analysis is not None:
        pts = analysis.swing_points
        color = "#10b981" if analysis.direction == "Bullish" else "#ef4444"

        # Solid historical lines
        fig.add_trace(go.Scatter(
            x=[p.timestamp for p in pts],
            y=[p.price for p in pts],
            mode="lines+markers+text",
            name=f"Elliott Wave Path",
            text=analysis.wave_labels,
            textposition="top center" if analysis.direction == "Bullish" else "bottom center",
            textfont=dict(size=16, color="#ffffff", family="Inter, sans-serif"),
            line=dict(color=color, width=3.5),
            marker=dict(size=12, color=color, symbol="circle",
                        line=dict(color="#ffffff", width=1.5)),
        ))

        # Invalidation Level
        inval_price = 0.0
        if len(pts) > 0:
            if analysis.phase == WavePhase.BUILDING_WAVE_3:
                inval_price = pts[0].price
            elif analysis.phase == WavePhase.BUILDING_WAVE_4:
                inval_price = pts[1].price
            elif analysis.phase == WavePhase.BUILDING_WAVE_5:
                inval_price = pts[4].price
            elif analysis.phase == WavePhase.IMPULSE_COMPLETE:
                inval_price = pts[5].price

        if inval_price > 0:
            fig.add_hline(
                y=inval_price,
                line_width=1.5,
                line_dash="dash",
                line_color="rgba(249, 115, 22, 0.8)",
                annotation_text="Invalidation Level",
                annotation_position="left",
                annotation_font_color="rgba(249, 115, 22, 0.8)"
            )

        # Projections
        avg_bars = 15
        if len(pts) > 1:
            spans = [pts[i].bar_index - pts[i - 1].bar_index for i in range(1, len(pts))]
            avg_bars = max(5, int(np.median(spans)))

        last_ts = pts[-1].timestamp
        last_price = pts[-1].price

        if analysis.target_zones:
            for i, zone in enumerate(analysis.target_zones):
                proj_ts = last_ts + (i + 1) * avg_bars * bar_delta
                half_w = 0.3 * avg_bars * bar_delta

                # Shaded target zone
                fig.add_trace(go.Scatter(
                    x=[proj_ts - half_w, proj_ts + half_w, proj_ts + half_w, proj_ts - half_w, proj_ts - half_w],
                    y=[zone.price_low, zone.price_low, zone.price_high, zone.price_high, zone.price_low],
                    fill="toself",
                    fillcolor="rgba(16, 185, 129, 0.08)" if analysis.direction == "Bullish" else "rgba(239, 68, 68, 0.08)",
                    line=dict(color="rgba(16, 185, 129, 0.25)" if analysis.direction == "Bullish" else "rgba(239, 68, 68, 0.25)", width=1),
                    name=zone.label,
                    hoverinfo="text",
                    text=f"{zone.label}: ${zone.price_low:.2f} — ${zone.price_high:.2f}",
                    showlegend=False,
                ))

                # Dashed projection path line
                fig.add_trace(go.Scatter(
                    x=[last_ts, proj_ts],
                    y=[last_price, zone.midpoint],
                    mode="lines",
                    line=dict(color=color, width=2.0, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                ))

                fig.add_annotation(
                    x=proj_ts, y=zone.midpoint,
                    text=f"<b>{zone.label}</b><br>${zone.midpoint:.2f}",
                    showarrow=False,
                    font=dict(size=10, color="rgba(255,255,255,0.85)"),
                    bgcolor="rgba(15, 23, 42, 0.75)",
                    borderpad=3,
                )

                last_ts = proj_ts
                last_price = zone.midpoint

        # Fib target lines
        if show_fib_levels and analysis.fib_targets:
            for ft in analysis.fib_targets:
                fig.add_hline(
                    y=ft.price,
                    line_width=0.7,
                    line_dash="dot",
                    line_color="rgba(148, 163, 184, 0.3)",
                    annotation_text=f"{ft.label}",
                    annotation_position="right",
                    annotation_font_size=8,
                    annotation_font_color="rgba(203, 213, 225, 0.5)",
                )

    title = f"{symbol} Wave Count analysis"
    if analysis:
        title = f"{symbol} | {analysis.phase.value} ({analysis.direction}) | Confidence: {analysis.quality_score:.1f}%"

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f1f5f9")),
        xaxis_title="",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        height=600,
        margin=dict(l=15, r=15, t=50, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=10)),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.05)")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# 8. STREAMLIT MODULE
# ═══════════════════════════════════════════════════════════════════════

class ElliottWaveAnalysisModule(FazDaneModule):
    MODULE_NAME = "Elliott Wave Analysis"
    MODULE_ICON = "📊"
    MODULE_DESCRIPTION = (
        "Ground-Up Elliott Wave probability engine with multi-timeframe alignment, "
        "options strategy recommendations, and pivot manual pinning."
    )
    TIER = 3
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def render_sidebar(self):
        self._init_defaults()

        st.markdown("### 🌌 Universe Manager")
        self.universe_name, tickers, _ = render_universe_manager(
            key_prefix="ew", show_benchmark=False, label="Active Universe:",
        )
        ticker_names = get_ticker_names(self.universe_name)
        if "^GSPC" not in tickers:
            tickers = ["^GSPC"] + tickers
            ticker_names.setdefault("^GSPC", "S&P 500 Index")
        self.tickers = tickers

        default_idx = self.tickers.index("^GSPC") if "^GSPC" in self.tickers else 0
        if st.session_state.get("ew_ticker") not in self.tickers:
            st.session_state["ew_ticker"] = self.tickers[default_idx]
        self.ticker = st.selectbox(
            "Ticker Select:",
            self.tickers,
            index=default_idx,
            key="ew_ticker",
            format_func=lambda t: format_ticker_display(t, ticker_names),
        )

        st.markdown("### 📅 Time Horizon")
        self.period = st.selectbox("Period Selection:", ["6mo", "1y", "2y", "5y", "max"], index=2, key="ew_period")
        self.interval = st.selectbox("Interval Selection:", ["1d", "1h", "15m"], index=0, key="ew_interval")

        st.markdown("### 🎛️ Settings")
        self.atr_mult = float(st.slider(
            "ATR Swing Multiplier:", 1.0, 5.0, 2.0, step=0.25, key="ew_atr",
            help="Higher = macro counts. Lower = micro counts."
        ))
        self.min_bars = int(st.slider(
            "Min Bars separation:", 3, 20, 5, key="ew_minbars",
        ))
        self.plot_last_n = int(st.slider("Plot Last N candles:", 100, 1500, 500, step=50, key="ew_plot_n"))

        st.markdown("### 📐 Rules Configuration")
        self.strict_rules = st.checkbox("Enforce Inviolable Rules", value=True, key="ew_strict",
                                       help="If checked, invalid counts are discarded.")
        self.diagonal_mode = st.checkbox("Allow Diagonal Overlap", value=False, key="ew_diagonal",
                                        help="Allows minor Wave 4 overlap (up to 15% of Wave 1).")

        st.markdown("### 👁️ Display")
        self.show_minor = st.checkbox("Show all swing pivots", value=False, key="ew_show_minor")
        self.show_fib = st.checkbox("Show Fibonacci levels", value=True, key="ew_show_fib")

        if st.button("Clear Cache & Refresh", width="stretch", type="primary"):
            fetch_ohlc.clear()
            st.rerun()

    def render_main(self):
        self.render_section_header(
            "Elliott Wave Probability Engine",
            "Advanced multi-candidate scanning, technical regime detection, options strategy recommendations, and multi-timeframe alignment.",
        )

        symbol = _normalize_symbol(self.ticker)
        with st.spinner(f"Loading {symbol} ..."):
            df = fetch_ohlc(symbol, self.period, self.interval)

        if df.empty:
            st.warning(f"No data returned for {symbol}.")
            return
        if len(df) < 50:
            st.warning("Not enough bars for Elliott Wave analysis (need ≥ 50).")
            return

        # Core swing detection
        swings = detect_swings(df, atr_multiplier=self.atr_mult, min_bars_between=self.min_bars)

        # Tab Setup
        tab_summary, tab_chart, tab_forecast, tab_mtf = st.tabs([
            "🏆 Setup rankings",
            "📈 Wave Chart Analysis",
            "🔮 Forecast Projections",
            "🔗 MTF Alignment"
        ])

        # ---------------------------------------------------------
        # TAB 1: SUMMARY DASHBOARD & RANKING
        # ---------------------------------------------------------
        with tab_summary:
            st.markdown("### 🏆 Options Universe Setup Rankings")
            st.caption("Rank tickers in the active universe by setup quality and options strategy bias.")

            if st.button("Run Universe Scan", type="primary", width="stretch"):
                progress = st.progress(0.0)
                ranked_rows = []
                scanned_plots = []
                for i, t in enumerate(self.tickers):
                    progress.progress((i + 1) / len(self.tickers))
                    tsym = _normalize_symbol(t)
                    tdf = fetch_ohlc(tsym, self.period, self.interval)
                    if tdf.empty or len(tdf) < 40:
                        continue
                    tswings = detect_swings(tdf, atr_multiplier=self.atr_mult, min_bars_between=self.min_bars)
                    tcands = scan_waves_multi_count(tdf, tswings, strict_mode=self.strict_rules, diagonal_mode=self.diagonal_mode)
                    if tcands:
                        prim = tcands[0]
                        recomm = classify_setup_and_options_strategy(prim)
                        ranked_rows.append({
                            "Ticker": tsym,
                            "Wave Stage": prim.phase.value,
                            "Direction": prim.direction,
                            "Confidence Score": f"{prim.quality_score:.1f}%",
                            "Setup Quality": recomm["setup"],
                            "Entry Zone": recomm["entry_zone"],
                            "Invalidation": recomm["invalidation"],
                            "Target Zone": recomm["target_zone"],
                            "Option Strategy Bias": recomm["strategy"],
                            "_score": prim.quality_score
                        })
                        scanned_plots.append((tsym, tdf, tswings, prim))
                    else:
                        ranked_rows.append({
                            "Ticker": tsym,
                            "Wave Stage": "No Pattern",
                            "Direction": "Neutral",
                            "Confidence Score": "0.0%",
                            "Setup Quality": "No active setup",
                            "Entry Zone": "N/A",
                            "Invalidation": "N/A",
                            "Target Zone": "N/A",
                            "Option Strategy Bias": "Avoid active strategy",
                            "_score": 0.0
                        })

                # Sort by score descending
                ranked_df = pd.DataFrame(ranked_rows)
                if not ranked_df.empty:
                    ranked_df = ranked_df.sort_values(by="_score", ascending=False).drop(columns=["_score"])
                    st.dataframe(ranked_df, width="stretch", hide_index=True)

                    # Sort plots by score descending
                    scanned_plots.sort(key=lambda x: x[3].quality_score, reverse=True)

                    st.markdown("### 📈 Scanned Wave Charts")
                    for tsym, tdf, tswings, prim in scanned_plots:
                        with st.expander(f"📊 Chart for {tsym} — {prim.phase.value} ({prim.direction}) | Confidence: {prim.quality_score:.1f}%", expanded=False):
                            fig = build_chart(tdf, tswings, prim, tsym, self.show_minor, self.show_fib, self.plot_last_n)
                            st.plotly_chart(fig, width="stretch", key=f"universe_chart_{tsym}")
                else:
                    st.info("No tickers scanned successfully.")
            else:
                st.info("Click the 'Run Universe Scan' button to scan all tickers in the active universe.")

        # ---------------------------------------------------------
        # TAB 2: WAVE CHART ANALYSIS
        # ---------------------------------------------------------
        with tab_chart:
            # Let the user pin the Wave 0 origin manually if desired
            mode = st.radio("Origin Selection Mode:", ["Auto-Detect Latest", "Manual Pinning (Select Origin)"], horizontal=True)
            pinned_sw = None

            if mode == "Manual Pinning (Select Origin)":
                if len(swings) > 2:
                    pivot_options = [
                        (i, f"{sw.timestamp.strftime('%Y-%m-%d')} - {sw.kind} @ ${sw.price:.2f}")
                        for i, sw in enumerate(swings[-15:])
                    ]
                    sel_p = st.selectbox("Select Wave 0 Origin Pivot:", pivot_options, format_func=lambda x: x[1])
                    pinned_sw = swings[sel_p[0]]
                else:
                    st.warning("Not enough pivots found to pin.")

            # Scan waves based on current settings
            candidates = scan_waves_multi_count(
                df, swings,
                strict_mode=self.strict_rules,
                diagonal_mode=self.diagonal_mode,
                pinned_p0=pinned_sw
            )

            active_analysis = None
            if candidates:
                st.markdown("### 🎛️ Select Wave Count Candidate")
                count_options = [
                    (i, f"Count {i+1}: {c.phase.value} ({c.direction}) - Quality: {c.quality_score:.1f}%")
                    for i, c in enumerate(candidates[:5]) # Top 5
                ]
                sel_cand = st.selectbox("Available Counts:", count_options, format_func=lambda x: x[1])
                active_analysis = candidates[sel_cand[0]]
            else:
                st.warning("No candidate wave counts found matching settings.")

            # Metrics summary
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Asset", symbol)
            c2.metric("Pivots Located", len(swings))

            reg = classify_regime(df)
            c3.metric("Regime", reg["regime"])

            if active_analysis:
                c4.metric("Active Quality", f"{active_analysis.quality_score:.1f}%")
            else:
                c4.metric("Active Quality", "N/A")

            # Chart plot
            fig = build_chart(df, swings, active_analysis, symbol, self.show_minor, self.show_fib, self.plot_last_n)
            st.plotly_chart(fig, width="stretch")

            # Recommend options trade
            if active_analysis:
                recomm = classify_setup_and_options_strategy(active_analysis)
                st.markdown("### 💡 Recommended Options Strategy Setup")
                cols = st.columns(3)
                cols[0].markdown(f"**Setup Stage:** `{recomm['stage']}`")
                cols[1].markdown(f"**Trade Setup:** `{recomm['setup']}`")
                cols[2].markdown(f"**Options Strategy Bias:** :green[{recomm['strategy']}]" if "Bull" in recomm['strategy'] or "Long" in recomm['strategy'] else f"**Options Strategy Bias:** :red[{recomm['strategy']}]" if "Bear" in recomm['strategy'] else f"**Options Strategy Bias:** :orange[{recomm['strategy']}]")

                st.markdown(
                    f"**Entry Target Zone:** `{recomm['entry_zone']}` | "
                    f"**Invalidation Level:** `{recomm['invalidation']}` | "
                    f"**Projection Target:** `{recomm['target_zone']}`"
                )

        # ---------------------------------------------------------
        # TAB 3: FORECAST PROJECTIONS
        # ---------------------------------------------------------
        with tab_forecast:
            if active_analysis:
                st.markdown("### 📐 Fibonacci Projections & Rules Verification")
                col_l, col_r = st.columns(2)

                with col_l:
                    st.markdown("#### Inviolable Rules Check")
                    for r in active_analysis.rules:
                        icon = "✅" if r.passed else "❌"
                        st.markdown(f"**{icon} {r.name}**")
                        st.caption(f"  {r.detail}")

                    st.markdown("#### Fibonacci Guidelines")
                    for g in active_analysis.guidelines:
                        icon = "🟢" if g.passed else "🟡"
                        st.markdown(f"{icon} {g.name} -- {g.detail}")

                with col_r:
                    st.markdown("#### Price Targets")
                    if active_analysis.target_zones:
                        for zone in active_analysis.target_zones:
                            st.markdown(
                                f"🎯 **{zone.label}**: "
                                f"`${zone.price_low:.2f} — ${zone.price_high:.2f}` "
                                f"(Midpoint: `${zone.midpoint:.2f}`)"
                            )
                            st.caption(f"  {zone.description}")
                    else:
                        st.caption("No target zones generated.")

                    if active_analysis.wave_ratios:
                        st.markdown("#### Calculated Wave Ratios")
                        ratio_rows = [{"Wave Ratio": k, "Measured": f"{v:.4f}"} for k, v in active_analysis.wave_ratios.items()]
                        st.dataframe(pd.DataFrame(ratio_rows), width="stretch", hide_index=True)
            else:
                st.info("No active wave analysis selected. Choose a candidate in the Wave Chart tab.")

        # ---------------------------------------------------------
        # TAB 4: MULTI-TIMEFRAME ALIGNMENT
        # ---------------------------------------------------------
        with tab_mtf:
            st.markdown("### 🔗 Multi-Timeframe (MTF) Wave Alignment")
            st.caption("Verifying primary wave structure across Weekly, Daily, and Hourly horizons.")

            if st.button("Perform MTF Alignment Scan", key="ew_mtf_scan"):
                with st.spinner("Fetching alignment data..."):
                    mtf_results = []
                    timeframes = [
                        ("Weekly", "5y", "1wk"),
                        ("Daily", "2y", "1d"),
                        ("Hourly", "2y", "1h")
                    ]
                    for tf_name, tf_period, tf_interval in timeframes:
                        try:
                            tdf = fetch_ohlc(symbol, tf_period, tf_interval)
                            if tdf.empty and tf_interval == "1h":
                                tdf = fetch_ohlc(symbol, "60d", "1h") # hourly fallback
                            if not tdf.empty and len(tdf) >= 30:
                                tswings = detect_swings(tdf, atr_multiplier=self.atr_mult, min_bars_between=self.min_bars)
                                tcands = scan_waves_multi_count(tdf, tswings, strict_mode=False)
                                if tcands:
                                    prim = tcands[0]
                                    mtf_results.append({
                                        "Timeframe": tf_name,
                                        "Wave State": prim.phase.value,
                                        "Direction": prim.direction,
                                        "Quality Score": f"{prim.quality_score:.1f}%",
                                        "Regime": prim.regime
                                    })
                                else:
                                    treg = classify_regime(tdf)
                                    mtf_results.append({
                                        "Timeframe": tf_name,
                                        "Wave State": "No Pattern",
                                        "Direction": "Neutral",
                                        "Quality Score": "0.0%",
                                        "Regime": treg["regime"]
                                    })
                            else:
                                mtf_results.append({
                                    "Timeframe": tf_name,
                                    "Wave State": "Insufficient Data",
                                    "Direction": "N/A",
                                    "Quality Score": "N/A",
                                    "Regime": "N/A"
                                })
                        except Exception as ex:
                            logger.error(f"Error scanning MTF {tf_name}: {ex}")
                            mtf_results.append({
                                "Timeframe": tf_name,
                                "Wave State": "Error",
                                "Direction": "N/A",
                                "Quality Score": "N/A",
                                "Regime": "N/A"
                            })

                    st.dataframe(pd.DataFrame(mtf_results), width="stretch", hide_index=True)
            else:
                st.info("Click 'Perform MTF Alignment Scan' to align structures.")

        # Legacy diagnostic tabs footer
        with st.expander("🛠️ Diagnostics & Export Logs"):
            tab_waves, tab_pivots, tab_export = st.tabs(["Wave Detail", "All Pivots", "Export"])

            with tab_waves:
                if active_analysis:
                    pts = active_analysis.swing_points
                    rows = []
                    for i in range(len(pts) - 1):
                        rows.append({
                            "Segment": f"Wave {active_analysis.wave_labels[i]} -> {active_analysis.wave_labels[i+1]}",
                            "Start": pts[i].timestamp.strftime("%Y-%m-%d"),
                            "End": pts[i + 1].timestamp.strftime("%Y-%m-%d"),
                            "Bars": pts[i + 1].bar_index - pts[i].bar_index,
                            "Points ($)": round(pts[i + 1].price - pts[i].price, 2),
                            "Abs Length ($)": round(_abs_wave_length(pts[i], pts[i + 1]), 2),
                        })
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                else:
                    st.info("No wave segments available.")

            with tab_pivots:
                pivot_rows = [{
                    "#": i,
                    "Date": sw.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "Price": round(sw.price, 2),
                    "Type": "High" if sw.kind == "H" else "Low",
                    "ATR": round(sw.atr_at_pivot, 2),
                } for i, sw in enumerate(swings)]
                st.dataframe(pd.DataFrame(pivot_rows), width="stretch", hide_index=True)

            with tab_export:
                log_text = self._build_log(symbol, df, swings, active_analysis)
                c_dl1, c_dl2 = st.columns(2)
                c_dl1.download_button(
                    "Download Pivots CSV",
                    data=pd.DataFrame([{
                        "Date": sw.timestamp.strftime("%Y-%m-%d"),
                        "Price": sw.price, "Type": sw.kind, "ATR": sw.atr_at_pivot,
                    } for sw in swings]).to_csv(index=False),
                    file_name=f"ew_pivots_{symbol.replace('^', '').lower()}.csv",
                    mime="text/csv",
                    width="stretch",
                )
                c_dl2.download_button(
                    "Download Analysis Log",
                    data=log_text,
                    file_name=f"ew_log_{symbol.replace('^', '').lower()}.txt",
                    mime="text/plain",
                    width="stretch",
                )

    def _init_defaults(self):
        key = "ew_sel"
        target = "Index Universe"
        if target in get_universe_names() and key not in st.session_state:
            st.session_state[key] = target

    def _build_log(self, symbol: str, df: pd.DataFrame, swings: list[SwingPoint], analysis: Optional[WaveAnalysis]) -> str:
        lines = [
            "Elliott Wave Analysis — Run Log",
            "=" * 40,
            f"Symbol        : {symbol}",
            f"Period        : {self.period}",
            f"Interval      : {self.interval}",
            f"ATR Multiplier: {self.atr_mult}",
            f"Min Bar Gap   : {self.min_bars}",
            f"Date Range    : {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            f"Total Bars    : {len(df)}",
            f"Swing Pivots  : {len(swings)}",
            "",
        ]
        if analysis is None:
            lines.append("Result: No valid Elliott Wave pattern detected.")
        else:
            lines.extend([
                f"Phase         : {analysis.phase.value}",
                f"Direction     : {analysis.direction}",
                f"Quality Score : {analysis.quality_score}",
                "",
                "Rules:",
            ])
            for r in analysis.rules:
                lines.append(f"  {'PASS' if r.passed else 'FAIL'} — {r.name}: {r.detail}")
            if analysis.guidelines:
                lines.append("")
                lines.append("Guidelines:")
                for g in analysis.guidelines:
                    lines.append(f"  {'PASS' if g.passed else 'SOFT'} — {g.name}: {g.detail}")
            if analysis.wave_ratios:
                lines.append("")
                lines.append("Fibonacci Ratios:")
                for k, v in analysis.wave_ratios.items():
                    lines.append(f"  {k}: {v:.4f}")
            if analysis.target_zones:
                lines.append("")
                lines.append("Target Zones:")
                for z in analysis.target_zones:
                    lines.append(f"  {z.label}: ${z.price_low:.2f} — ${z.price_high:.2f} (mid: ${z.midpoint:.2f})")

        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)
