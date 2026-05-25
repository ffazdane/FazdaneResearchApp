"""
FazDane Analytics - Tier 3
Elliott Wave Analysis -- Ground-up implementation using mathematical models.

Mathematical foundations
========================
1.  SWING DETECTION -- Adaptive ATR-scaled local-extrema filter.
    Instead of a fixed-percentage ZigZag (which is arbitrary and scale-
    dependent), we detect statistically significant swing points using
    Average True Range (ATR) as the volatility normaliser.  A swing high
    at bar i is confirmed when subsequent price drops by >= k * ATR(i).

2.  WAVE GEOMETRY VALIDATION -- The three inviolable rules of R.N.
    Prechter / A.J. Frost "Elliott Wave Principle" (1978):
        Rule 1  Wave 2 never retraces beyond the origin of Wave 1.
        Rule 2  Wave 3 is never the shortest of Waves 1, 3, and 5.
        Rule 3  Wave 4 never enters the price territory of Wave 1.
    Plus guidelines (scored but not disqualifying):
        -  Wave 2 retracement typically 50%-78.6% of Wave 1.
        -  Wave 3 typically 1.618x Wave 1.
        -  Wave 4 retracement typically 38.2% of Wave 3.
        -  Wave 5 typically ~ Wave 1 in length.
        -  Alternation between Waves 2 and 4 (one sharp, one sideways).

3.  FIBONACCI PROJECTION ENGINE -- Forward-looking target zones derived
    from completed wave segments using standard Fib ratios (0.236, 0.382,
    0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618).

4.  WAVE-STATE MACHINE -- Classifies the current market position as one
    of: BUILDING_WAVE_3, BUILDING_WAVE_4, BUILDING_WAVE_5, IMPULSE_DONE,
    or CORRECTIVE_ABC, and generates the appropriate projection set.

5.  QUALITY SCORE -- Composite metric measuring how closely the detected
    impulse matches ideal Fibonacci proportions (0-100 scale).
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
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

TICKER_ALIASES = {"SPX": "^GSPC", "NDX": "^NDX", "RUT": "^RUT", "NASDAQ": "^IXIC"}

# Standard Fibonacci ratios used throughout Elliott Wave theory
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


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

class WavePhase(Enum):
    """Current position within the Elliott Wave cycle."""
    BUILDING_WAVE_3 = "Building Wave 3"   # Waves 0-1-2 confirmed
    BUILDING_WAVE_4 = "Building Wave 4"   # Waves 0-1-2-3 confirmed
    BUILDING_WAVE_5 = "Building Wave 5"   # Waves 0-1-2-3-4 confirmed
    IMPULSE_COMPLETE = "Impulse Complete"  # Full 5-wave impulse done
    CORRECTIVE_ABC = "Corrective ABC"     # ABC correction in progress
    NONE = "No Pattern"


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


# ═══════════════════════════════════════════════════════════════════════
# 1. DATA FETCHING
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
            # Try to find a case-insensitive match
            match = [c for c in df.columns if c.lower() == col.lower()]
            if match:
                df[col] = df[match[0]]
            else:
                return pd.DataFrame()

    # ── Compute ATR (14-period) ──────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════
# 2. SWING DETECTION — ATR-Scaled Adaptive Pivots
# ═══════════════════════════════════════════════════════════════════════

def detect_swings(
    df: pd.DataFrame,
    atr_multiplier: float = 2.0,
    min_bars_between: int = 5,
) -> list[SwingPoint]:
    """
    Detect significant swing highs and lows using ATR as the volatility
    normaliser.  A swing is confirmed when price reverses by at least
    `atr_multiplier × ATR` from the most recent extreme.

    This is mathematically superior to a fixed-percentage ZigZag because
    it adapts to the asset's own volatility regime — a 2% move in a low-
    vol stock is significant, but in a high-vol stock it is noise.
    """
    close = df["Close"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)
    atr = df["ATR"].values.astype(np.float64)
    n = len(close)

    if n < 30:
        return []

    # Start scanning after ATR is available
    start = max(14, min_bars_between)
    swings: list[SwingPoint] = []

    # Track the running extreme
    extreme_idx = start
    extreme_price = high[start]
    trend = "up"  # Assume initial up-trend; will self-correct

    # Find actual initial direction
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
                # Confirm the swing high
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
                # Confirm the swing low
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
# 3. WAVE GEOMETRY — Rules & Guidelines Validation
# ═══════════════════════════════════════════════════════════════════════

def _wave_length(p_start: SwingPoint, p_end: SwingPoint) -> float:
    """Signed price displacement of a wave segment."""
    return p_end.price - p_start.price


def _abs_wave_length(p_start: SwingPoint, p_end: SwingPoint) -> float:
    """Absolute price displacement of a wave segment."""
    return abs(p_end.price - p_start.price)


def validate_impulse_rules(seq: list[SwingPoint], direction: str) -> tuple[list[RuleCheck], bool]:
    """
    Validate the three inviolable rules of Elliott Wave theory.

    Parameters
    ----------
    seq : list[SwingPoint]
        Six points: Wave-0, Wave-1, Wave-2, Wave-3, Wave-4, Wave-5.
    direction : str
        "Bullish" or "Bearish".

    Returns
    -------
    rules : list[RuleCheck]
        Detailed result for each rule.
    all_valid : bool
        True only if all three rules pass.
    """
    p0, p1, p2, p3, p4, p5 = seq[0], seq[1], seq[2], seq[3], seq[4], seq[5]
    rules: list[RuleCheck] = []

    # ── Rule 1: Wave 2 never retraces beyond the origin of Wave 1 ────
    if direction == "Bullish":
        r1_pass = p2.price > p0.price
        r1_detail = f"Wave-2 low ({p2.price:.2f}) {'>' if r1_pass else '<='} Wave-0 ({p0.price:.2f})"
    else:
        r1_pass = p2.price < p0.price
        r1_detail = f"Wave-2 high ({p2.price:.2f}) {'<' if r1_pass else '>='} Wave-0 ({p0.price:.2f})"
    rules.append(RuleCheck("Rule 1: Wave 2 does not retrace 100% of Wave 1", r1_pass, r1_detail))

    # ── Rule 2: Wave 3 is never the shortest of 1, 3, 5 ──────────────
    w1_len = _abs_wave_length(p0, p1)
    w3_len = _abs_wave_length(p2, p3)
    w5_len = _abs_wave_length(p4, p5)
    r2_pass = w3_len >= min(w1_len, w5_len)
    r2_detail = f"|W1|={w1_len:.2f}, |W3|={w3_len:.2f}, |W5|={w5_len:.2f}"
    rules.append(RuleCheck("Rule 2: Wave 3 is not the shortest impulse wave", r2_pass, r2_detail))

    # ── Rule 3: Wave 4 does not enter the price territory of Wave 1 ──
    if direction == "Bullish":
        r3_pass = p4.price >= p1.price
        r3_detail = f"Wave-4 low ({p4.price:.2f}) {'>=' if r3_pass else '<'} Wave-1 high ({p1.price:.2f})"
    else:
        r3_pass = p4.price <= p1.price
        r3_detail = f"Wave-4 high ({p4.price:.2f}) {'<=' if r3_pass else '>'} Wave-1 low ({p1.price:.2f})"
    rules.append(RuleCheck("Rule 3: Wave 4 does not overlap Wave 1 territory", r3_pass, r3_detail))

    return rules, all(r.passed for r in rules)


def evaluate_guidelines(seq: list[SwingPoint], direction: str) -> tuple[list[RuleCheck], dict]:
    """
    Evaluate Elliott Wave guidelines (soft rules) and compute Fibonacci
    ratios between wave segments.

    Returns
    -------
    guidelines : list[RuleCheck]
        Guideline evaluations.
    ratios : dict
        Measured wave ratios.
    """
    p0, p1, p2, p3, p4, p5 = seq[0], seq[1], seq[2], seq[3], seq[4], seq[5]
    w1 = _abs_wave_length(p0, p1)
    w2 = _abs_wave_length(p1, p2)
    w3 = _abs_wave_length(p2, p3)
    w4 = _abs_wave_length(p3, p4)
    w5 = _abs_wave_length(p4, p5)

    guidelines: list[RuleCheck] = []
    ratios = {}

    # Wave 2 / Wave 1 retracement
    r_w2w1 = w2 / max(w1, 1e-9)
    ratios["W2/W1 Retracement"] = r_w2w1
    g_w2 = 0.382 <= r_w2w1 <= 0.786
    guidelines.append(RuleCheck(
        "Wave 2 retracement 38.2%-78.6% of Wave 1",
        g_w2,
        f"Measured: {r_w2w1:.3f} (ideal: 0.500-0.618)"
    ))

    # Wave 3 / Wave 1 extension
    r_w3w1 = w3 / max(w1, 1e-9)
    ratios["W3/W1 Extension"] = r_w3w1
    g_w3 = r_w3w1 >= 1.0
    guidelines.append(RuleCheck(
        "Wave 3 extends >= 1.0x Wave 1 (ideal 1.618x)",
        g_w3,
        f"Measured: {r_w3w1:.3f} (ideal: 1.618)"
    ))

    # Wave 4 / Wave 3 retracement
    r_w4w3 = w4 / max(w3, 1e-9)
    ratios["W4/W3 Retracement"] = r_w4w3
    g_w4 = 0.236 <= r_w4w3 <= 0.500
    guidelines.append(RuleCheck(
        "Wave 4 retracement 23.6%-50% of Wave 3",
        g_w4,
        f"Measured: {r_w4w3:.3f} (ideal: 0.382)"
    ))

    # Wave 5 / Wave 1 equality
    r_w5w1 = w5 / max(w1, 1e-9)
    ratios["W5/W1 Ratio"] = r_w5w1
    g_w5 = 0.618 <= r_w5w1 <= 1.618
    guidelines.append(RuleCheck(
        "Wave 5 ~ Wave 1 in length (0.618x-1.618x)",
        g_w5,
        f"Measured: {r_w5w1:.3f} (ideal: 1.000)"
    ))

    # Alternation: Waves 2 and 4 differ in depth
    depth_diff = abs(r_w2w1 - r_w4w3)
    g_alt = depth_diff >= 0.10
    ratios["Alternation Differential"] = depth_diff
    guidelines.append(RuleCheck(
        "Alternation: Waves 2 and 4 differ in character",
        g_alt,
        f"Retracement difference: {depth_diff:.3f} (want >= 0.10)"
    ))

    return guidelines, ratios


def compute_quality_score(rules: list[RuleCheck], guidelines: list[RuleCheck], ratios: dict) -> float:
    """
    Composite quality score 0-100.

    Breakdown:
    -  30 pts -- rules compliance (10 each)
    -  25 pts -- guidelines compliance (5 each)
    -  45 pts -- Fibonacci ratio accuracy (closeness to ideal)
    """
    score = 0.0

    # Rules (30 pts)
    for r in rules:
        score += 10.0 if r.passed else 0.0

    # Guidelines (25 pts)
    for g in guidelines:
        score += 5.0 if g.passed else 0.0

    # Fibonacci accuracy (45 pts)
    ideal = {
        "W2/W1 Retracement": 0.618,
        "W3/W1 Extension": 1.618,
        "W4/W3 Retracement": 0.382,
        "W5/W1 Ratio": 1.000,
    }
    max_fib_pts = 45.0
    fib_score = 0.0
    for key, ideal_val in ideal.items():
        measured = ratios.get(key, 0.0)
        error = abs(measured - ideal_val) / max(ideal_val, 1e-9)
        accuracy = max(0.0, 1.0 - error)  # 1.0 = perfect match
        fib_score += accuracy * (max_fib_pts / len(ideal))
    score += fib_score

    return round(min(100.0, max(0.0, score)), 1)


# ═══════════════════════════════════════════════════════════════════════
# 4. FIBONACCI PROJECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def project_targets(seq: list[SwingPoint], phase: WavePhase, direction: str) -> tuple[list[TargetZone], list[FibTarget]]:
    """
    Generate forward-looking Fibonacci target zones based on the
    current wave phase and completed segments.
    """
    sign = 1.0 if direction == "Bullish" else -1.0
    zones: list[TargetZone] = []
    targets: list[FibTarget] = []

    if phase == WavePhase.NONE or len(seq) < 3:
        return zones, targets

    p0, p1, p2 = seq[0], seq[1], seq[2]
    w1_len = _abs_wave_length(p0, p1)

    if phase == WavePhase.BUILDING_WAVE_3:
        # Project Wave 3 targets from end of Wave 2
        for ratio_name, ratio_val in [("1.000x", 1.0), ("1.618x", 1.618), ("2.618x", 2.618)]:
            price = p2.price + sign * ratio_val * w1_len
            targets.append(FibTarget(f"Wave 3 ({ratio_name})", price, ratio_name, f"Wave 1 × {ratio_name} from Wave 2"))

        zone_lo = p2.price + sign * 1.0 * w1_len
        zone_hi = p2.price + sign * 2.618 * w1_len
        zones.append(TargetZone("Wave 3 Zone", min(zone_lo, zone_hi), max(zone_lo, zone_hi),
                                p2.price + sign * 1.618 * w1_len, "Fibonacci extension zone for Wave 3"))

    if phase in (WavePhase.BUILDING_WAVE_3, WavePhase.BUILDING_WAVE_4) and len(seq) >= 4:
        p3 = seq[3]
        w3_len = _abs_wave_length(p2, p3)
        for ratio_name, ratio_val in [("0.236x", 0.236), ("0.382x", 0.382), ("0.500x", 0.500)]:
            price = p3.price - sign * ratio_val * w3_len
            targets.append(FibTarget(f"Wave 4 ({ratio_name})", price, ratio_name, f"Wave 3 × {ratio_name} retracement"))

        zone_lo = p3.price - sign * 0.500 * w3_len
        zone_hi = p3.price - sign * 0.236 * w3_len
        zones.append(TargetZone("Wave 4 Zone", min(zone_lo, zone_hi), max(zone_lo, zone_hi),
                                p3.price - sign * 0.382 * w3_len, "Fibonacci retracement zone for Wave 4"))

    if phase in (WavePhase.BUILDING_WAVE_4, WavePhase.BUILDING_WAVE_5) and len(seq) >= 5:
        p3 = seq[3]
        p4 = seq[4]
        # Wave 5 targets: equality with W1, or 0.618 × (W0-W3 distance)
        w5_eq = p4.price + sign * w1_len
        w5_fib = p4.price + sign * 0.618 * abs(p3.price - p0.price)
        targets.append(FibTarget("Wave 5 (W1 equality)", w5_eq, "1.0x W1", "Wave 5 = Wave 1 in length"))
        targets.append(FibTarget("Wave 5 (0.618 × W0-W3)", w5_fib, "0.618x", "Fibonacci projection from Wave 4"))

        zone_lo = min(w5_eq, w5_fib)
        zone_hi = max(w5_eq, w5_fib)
        zones.append(TargetZone("Wave 5 Zone", zone_lo, zone_hi, (zone_lo + zone_hi) / 2.0,
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
# 5. WAVE SCANNER — Right-to-Left Search
# ═══════════════════════════════════════════════════════════════════════

def _determine_direction(seq: list[SwingPoint]) -> Optional[str]:
    """Check if the first two pivots form a valid impulse direction."""
    if len(seq) < 2:
        return None
    delta = seq[1].price - seq[0].price
    if abs(delta) < 1e-9:
        return None
    return "Bullish" if delta > 0 else "Bearish"


def _alternation_check(seq: list[SwingPoint]) -> bool:
    """Verify strict high-low alternation in the swing sequence."""
    for i in range(1, len(seq)):
        if seq[i].kind == seq[i - 1].kind:
            return False
    return True


def _direction_check(seq: list[SwingPoint], direction: str) -> bool:
    """Verify that each wave segment moves in the correct direction."""
    n = len(seq)
    if direction == "Bullish":
        # Odd waves go up, even waves go down (0→1↑, 1→2↓, 2→3↑, 3→4↓, 4→5↑)
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


def scan_waves(swings: list[SwingPoint]) -> Optional[WaveAnalysis]:
    """
    Scan from the most recent swings backwards, looking for the best
    valid Elliott Wave impulse pattern.

    Search priority (most complete first):
        1.  6-point complete impulse (Waves 0–5)
        2.  5-point partial (Waves 0–4, building Wave 5)
        3.  4-point partial (Waves 0–3, building Wave 4)
        4.  3-point partial (Waves 0–2, building Wave 3)
    """
    n = len(swings)
    if n < 3:
        return None

    # Define search configurations: (num_points, phase, labels)
    configs = [
        (6, WavePhase.IMPULSE_COMPLETE, ["0", "1", "2", "3", "4", "5"]),
        (5, WavePhase.BUILDING_WAVE_5, ["0", "1", "2", "3", "4"]),
        (4, WavePhase.BUILDING_WAVE_4, ["0", "1", "2", "3"]),
        (3, WavePhase.BUILDING_WAVE_3, ["0", "1", "2"]),
    ]

    best_result: Optional[WaveAnalysis] = None
    best_score = -1.0

    for num_pts, phase, labels in configs:
        # Scan right-to-left
        for start in range(n - num_pts, -1, -1):
            seq = swings[start : start + num_pts]

            # Quick structural checks
            if not _alternation_check(seq):
                continue
            direction = _determine_direction(seq)
            if direction is None:
                continue
            if not _direction_check(seq, direction):
                continue

            # Rule 1 (Wave 2 check) — always applicable
            if direction == "Bullish" and seq[2].price <= seq[0].price:
                continue
            if direction == "Bearish" and seq[2].price >= seq[0].price:
                continue

            # For complete impulse, validate all three rules
            if num_pts == 6:
                rules, rules_valid = validate_impulse_rules(seq, direction)
                if not rules_valid:
                    continue
                guidelines, ratios = evaluate_guidelines(seq, direction)
                score = compute_quality_score(rules, guidelines, ratios)
            else:
                # Partial — check applicable rules only
                rules = []
                w1 = _abs_wave_length(seq[0], seq[1])

                # Rule 1 always checked (already passed above)
                rules.append(RuleCheck(
                    "Rule 1: Wave 2 does not retrace 100% of Wave 1",
                    True,
                    f"Wave-2 ({seq[2].price:.2f}) valid vs Wave-0 ({seq[0].price:.2f})"
                ))

                if num_pts >= 4:
                    w3 = _abs_wave_length(seq[2], seq[3])
                    # Rule 2 is only fully checkable with Wave 5
                    rules.append(RuleCheck(
                        "Rule 2: Wave 3 not shortest (partial check)",
                        w3 >= w1 * 0.8,
                        f"|W1|={w1:.2f}, |W3|={w3:.2f}"
                    ))

                if num_pts >= 5:
                    # Rule 3: Wave 4 overlap
                    if direction == "Bullish":
                        r3 = seq[4].price >= seq[1].price
                    else:
                        r3 = seq[4].price <= seq[1].price
                    rules.append(RuleCheck(
                        "Rule 3: Wave 4 does not overlap Wave 1",
                        r3,
                        f"Wave-4 ({seq[4].price:.2f}) vs Wave-1 ({seq[1].price:.2f})"
                    ))
                    if not r3:
                        continue

                # Score partial patterns
                guidelines = []
                ratios = {}
                if num_pts >= 3:
                    r_w2w1 = _abs_wave_length(seq[1], seq[2]) / max(w1, 1e-9)
                    ratios["W2/W1 Retracement"] = r_w2w1
                    guidelines.append(RuleCheck("W2 retracement", 0.382 <= r_w2w1 <= 0.786, f"{r_w2w1:.3f}"))

                if num_pts >= 4:
                    r_w3w1 = _abs_wave_length(seq[2], seq[3]) / max(w1, 1e-9)
                    ratios["W3/W1 Extension"] = r_w3w1
                    guidelines.append(RuleCheck("W3 extension", r_w3w1 >= 1.0, f"{r_w3w1:.3f}"))

                if num_pts >= 5:
                    w3 = _abs_wave_length(seq[2], seq[3])
                    r_w4w3 = _abs_wave_length(seq[3], seq[4]) / max(w3, 1e-9)
                    ratios["W4/W3 Retracement"] = r_w4w3
                    guidelines.append(RuleCheck("W4 retracement", 0.236 <= r_w4w3 <= 0.500, f"{r_w4w3:.3f}"))

                # Simplified score for partials
                base = sum(10.0 for r in rules if r.passed)
                guide = sum(5.0 for g in guidelines if g.passed)
                score = base + guide

            # Is this the best we've found?
            if score > best_score:
                zones, fib_targets = project_targets(seq, phase, direction)
                best_result = WaveAnalysis(
                    phase=phase,
                    direction=direction,
                    swing_points=list(seq),
                    wave_labels=labels[:num_pts],
                    rules=rules,
                    guidelines=guidelines,
                    quality_score=score if num_pts == 6 else round(score, 1),
                    target_zones=zones,
                    fib_targets=fib_targets,
                    wave_ratios=ratios,
                )
                best_score = score

    return best_result


# ═══════════════════════════════════════════════════════════════════════
# 6. CHART BUILDER
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
    """Build a clean, professional Plotly chart."""

    tail = df if plot_last_n <= 0 else df.iloc[-plot_last_n:]
    start_pos = len(df) - len(tail)
    end_pos = len(df) - 1

    fig = go.Figure()

    # ── Candlestick chart (subtle, low-noise) ─────────────────────────
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

    # ── Minor swing pivots (optional) ─────────────────────────────────
    if show_minor_pivots:
        visible = [sw for sw in swings if start_pos <= sw.bar_index <= end_pos]
        if visible:
            fig.add_trace(go.Scatter(
                x=[sw.timestamp for sw in visible],
                y=[sw.price for sw in visible],
                mode="markers",
                name="Swing Pivots",
                marker=dict(size=4, color="rgba(148, 163, 184, 0.5)", symbol="diamond"),
                hovertemplate="%{y:.2f}<extra>Pivot</extra>",
            ))

    # Estimate bar spacing for forecasts
    bar_delta = pd.Series(df.index).diff().median()
    if pd.isna(bar_delta):
        bar_delta = timedelta(days=1)

    # ── Main wave structure ───────────────────────────────────────────
    if analysis is not None:
        pts = analysis.swing_points
        color = "#10b981" if analysis.direction == "Bullish" else "#ef4444"

        # Historical wave path (solid)
        fig.add_trace(go.Scatter(
            x=[p.timestamp for p in pts],
            y=[p.price for p in pts],
            mode="lines+markers+text",
            name=f"Elliott Wave ({analysis.direction})",
            text=analysis.wave_labels,
            textposition="top center" if analysis.direction == "Bullish" else "bottom center",
            textfont=dict(size=16, color="#ffffff", family="Inter, sans-serif"),
            line=dict(color=color, width=3),
            marker=dict(size=12, color=color, symbol="circle",
                        line=dict(color="#ffffff", width=1.5)),
        ))

        # ── Forecast projections (dashed) ─────────────────────────────
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
                    fillcolor="rgba(16, 185, 129, 0.07)" if analysis.direction == "Bullish" else "rgba(239, 68, 68, 0.07)",
                    line=dict(color="rgba(16, 185, 129, 0.20)" if analysis.direction == "Bullish" else "rgba(239, 68, 68, 0.20)", width=1),
                    name=zone.label,
                    hoverinfo="text",
                    text=f"{zone.label}: ${zone.price_low:.2f} — ${zone.price_high:.2f}",
                    showlegend=False,
                ))

                # Dashed line from last point to zone midpoint
                fig.add_trace(go.Scatter(
                    x=[last_ts, proj_ts],
                    y=[last_price, zone.midpoint],
                    mode="lines",
                    line=dict(color=color, width=1.5, dash="dashdot"),
                    showlegend=False,
                    hoverinfo="skip",
                ))

                # Label
                fig.add_annotation(
                    x=proj_ts, y=zone.midpoint,
                    text=f"<b>{zone.label}</b><br>${zone.midpoint:.0f}",
                    showarrow=False,
                    font=dict(size=10, color="rgba(255,255,255,0.7)"),
                    bgcolor="rgba(0,0,0,0.4)",
                    borderpad=3,
                )

                last_ts = proj_ts
                last_price = zone.midpoint

        # ── Fibonacci level lines ─────────────────────────────────────
        if show_fib_levels and analysis.fib_targets:
            for ft in analysis.fib_targets:
                fig.add_hline(
                    y=ft.price,
                    line_width=0.6,
                    line_dash="dot",
                    line_color="rgba(148, 163, 184, 0.25)",
                    annotation_text=f"{ft.label}",
                    annotation_position="right",
                    annotation_font_size=8,
                    annotation_font_color="rgba(203, 213, 225, 0.6)",
                )

    # ── Layout ────────────────────────────────────────────────────────
    phase_str = analysis.phase.value if analysis else "No Pattern"
    dir_str = f" ({analysis.direction})" if analysis else ""
    score_str = f" | Quality: {analysis.quality_score:.0f}/100" if analysis else ""
    title = f"{symbol} | {phase_str}{dir_str}{score_str}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        xaxis_title="",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#0c1017",
        plot_bgcolor="#0c1017",
        height=620,
        margin=dict(l=16, r=16, t=56, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=10)),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.06)")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# 7. STREAMLIT MODULE
# ═══════════════════════════════════════════════════════════════════════

class ElliottWaveAnalysisModule(FazDaneModule):
    MODULE_NAME = "Elliott Wave Analysis"
    MODULE_ICON = "Wave"
    MODULE_DESCRIPTION = (
        "ATR-adaptive swing detection, rules-based wave validation, "
        "Fibonacci projection engine, and quality scoring"
    )
    TIER = 3
    SOURCE_NOTEBOOK = "Forecasting/Cycle Analysis/Elliott Wave Analysis"
    CACHE_TTL = 3600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    # ── Sidebar ───────────────────────────────────────────────────────

    def render_sidebar(self):
        self._init_defaults()

        st.markdown("**Universe**")
        self.universe_name, tickers, _ = render_universe_manager(
            key_prefix="ew", show_benchmark=False, label="Ticker Universe:",
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
            "Ticker / Index:",
            self.tickers,
            index=default_idx,
            key="ew_ticker",
            format_func=lambda t: format_ticker_display(t, ticker_names),
        )

        st.markdown("**Data**")
        self.period = st.selectbox("Period:", ["6mo", "1y", "2y", "5y", "10y", "max"], index=2, key="ew_period")
        self.interval = st.selectbox("Interval:", ["1d", "1h", "15m"], index=0, key="ew_interval")

        st.markdown("**Swing Detection**")
        self.atr_mult = float(st.slider(
            "ATR Multiplier:", 1.0, 5.0, 2.0, step=0.25, key="ew_atr",
            help="Higher = fewer, more significant swings.  Lower = more granular."
        ))
        self.min_bars = int(st.slider(
            "Min Bars Between Swings:", 3, 20, 5, key="ew_minbars",
            help="Minimum bar separation between consecutive pivots."
        ))
        self.plot_last_n = int(st.slider("Plot Last N Bars:", 100, 1500, 500, step=50, key="ew_plot_n"))

        st.markdown("**Display**")
        self.show_minor = st.checkbox("Show all swing pivots", value=False, key="ew_show_minor")
        self.show_fib = st.checkbox("Show Fibonacci levels", value=True, key="ew_show_fib")

        if st.button("Refresh", use_container_width=True, type="primary", key="ew_refresh"):
            fetch_ohlc.clear()
            st.rerun()

    # ── Main Content ──────────────────────────────────────────────────

    def render_main(self):
        self.render_section_header(
            "Elliott Wave Analysis",
            "ATR-adaptive swing detection with rules-based wave validation and Fibonacci projections",
        )

        symbol = _normalize_symbol(self.ticker)
        with st.spinner(f"Fetching {symbol} ..."):
            df = fetch_ohlc(symbol, self.period, self.interval)

        if df.empty:
            st.warning(f"No data returned for {symbol}.")
            return
        if len(df) < 50:
            st.warning("Not enough bars for Elliott Wave analysis (need ≥ 50).")
            return

        # Detect swings
        swings = detect_swings(df, atr_multiplier=self.atr_mult, min_bars_between=self.min_bars)
        # Scan for wave structures
        analysis = scan_waves(swings)

        # ── Metrics Row ───────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Symbol", symbol)
        c2.metric("Bars", f"{len(df):,}")
        c3.metric("Swing Pivots", f"{len(swings)}")
        if analysis:
            c4.metric("Quality Score", f"{analysis.quality_score:.0f} / 100")
        else:
            c4.metric("Wave Status", "No Pattern")

        # ── Chart ─────────────────────────────────────────────────────
        st.plotly_chart(
            build_chart(df, swings, analysis, symbol, self.show_minor, self.show_fib, self.plot_last_n),
            use_container_width=True,
        )

        # ── Rules & Targets Dashboard ─────────────────────────────────
        if analysis:
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("### Wave Theory Rules")
                for r in analysis.rules:
                    icon = "✅" if r.passed else "❌"
                    st.markdown(f"**{icon} {r.name}**")
                    st.caption(f"  {r.detail}")

                if analysis.guidelines:
                    st.markdown("### Fibonacci Guidelines")
                    for g in analysis.guidelines:
                        icon = "🟢" if g.passed else "🔶"
                        st.markdown(f"{icon} {g.name} — {g.detail}")

            with col_right:
                st.markdown("### Projected Target Zones")
                if analysis.target_zones:
                    for zone in analysis.target_zones:
                        st.markdown(
                            f"**{zone.label}**: "
                            f"`${zone.price_low:.2f}` — `${zone.price_high:.2f}` "
                            f"(mid: `${zone.midpoint:.2f}`)"
                        )
                        st.caption(f"  {zone.description}")
                else:
                    st.caption("Target zones require at least a 3-point wave structure.")

                if analysis.wave_ratios:
                    st.markdown("### Measured Fibonacci Ratios")
                    ratio_rows = [{"Ratio": k, "Value": f"{v:.3f}"} for k, v in analysis.wave_ratios.items()]
                    st.dataframe(pd.DataFrame(ratio_rows), use_container_width=True, hide_index=True)

        else:
            st.info(
                "No Elliott Wave pattern was identified with the current settings.  "
                "Try adjusting the **ATR Multiplier** (lower = more sensitive) or "
                "changing the **Period / Interval**."
            )

        # ── Detail Tabs ───────────────────────────────────────────────
        tab_waves, tab_pivots, tab_export = st.tabs(["Wave Detail", "All Pivots", "Export"])

        with tab_waves:
            if analysis:
                pts = analysis.swing_points
                rows = []
                for i in range(len(pts) - 1):
                    rows.append({
                        "Segment": f"Wave {analysis.wave_labels[i]} -> {analysis.wave_labels[i+1]}",
                        "Start": pts[i].timestamp.strftime("%Y-%m-%d"),
                        "End": pts[i + 1].timestamp.strftime("%Y-%m-%d"),
                        "Bars": pts[i + 1].bar_index - pts[i].bar_index,
                        "Points ($)": round(pts[i + 1].price - pts[i].price, 2),
                        "Abs Length ($)": round(_abs_wave_length(pts[i], pts[i + 1]), 2),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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
            st.dataframe(pd.DataFrame(pivot_rows), use_container_width=True, hide_index=True)

        with tab_export:
            # Run log
            log_text = self._build_log(symbol, df, swings, analysis)
            c_dl1, c_dl2 = st.columns(2)
            c_dl1.download_button(
                "Download Pivots CSV",
                data=pd.DataFrame([{
                    "Date": sw.timestamp.strftime("%Y-%m-%d"),
                    "Price": sw.price, "Type": sw.kind, "ATR": sw.atr_at_pivot,
                } for sw in swings]).to_csv(index=False),
                file_name=f"ew_pivots_{symbol.replace('^', '').lower()}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            c_dl2.download_button(
                "Download Analysis Log",
                data=log_text,
                file_name=f"ew_log_{symbol.replace('^', '').lower()}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # ── Helpers ────────────────────────────────────────────────────────

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
