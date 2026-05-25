"""
FazDane Analytics - Tier 3
Elliott Wave analysis using ZigZag pivots and rules-based impulse scoring.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from modules.base_module import FazDaneModule
from utils.universe_manager import format_ticker_display, get_ticker_names, get_universe_names, render_universe_manager


TICKER_ALIASES = {
    "SPX": "^GSPC",
    "NDX": "^NDX",
    "RUT": "^RUT",
    "NASDAQ": "^IXIC",
}


class WaveState:
    IN_WAVE_3 = "IN_WAVE_3"      # Completed 0,1,2 -> Wave 3 active
    IN_WAVE_4 = "IN_WAVE_4"      # Completed 0,1,2,3 -> Wave 4 active
    IN_WAVE_5 = "IN_WAVE_5"      # Completed 0,1,2,3,4 -> Wave 5 active
    COMPLETE_5W = "COMPLETE_5W"  # Completed 0,1,2,3,4,5 -> ABC correction active
    UNKNOWN = "UNKNOWN"


@dataclass
class Pivot:
    idx: int
    price: float
    kind: str


@dataclass
class ElliottWaveStructure:
    state: str
    direction: str  # "bull" or "bear"
    pivot_idxs: list[int]  # Indices in the pivots list
    pivots: list[Pivot]
    rules_check: dict
    score: float
    forecasts: dict


def normalize_symbol(symbol: str) -> str:
    clean = symbol.strip().upper()
    return TICKER_ALIASES.get(clean, clean)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_elliott_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    symbol = normalize_symbol(ticker)
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df[~df.index.duplicated(keep="last")].sort_index().dropna(how="any")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(part) for part in col if part).strip() for col in df.columns]

    close_col = pick_close_column(df)
    df["Close_PLOT"] = df[close_col].astype("float64")
    if all(col in df.columns for col in ["High", "Low"]):
        df["Mid"] = (df["High"] + df["Low"]) / 2.0
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def pick_close_column(df: pd.DataFrame) -> str:
    candidates = []
    if "Close" in df.columns:
        candidates.append("Close")
    if "Adj Close" in df.columns:
        candidates.append("Adj Close")
    candidates.extend([col for col in df.columns if "close" in str(col).lower() and col not in candidates])
    if not candidates:
        return df.columns[0]
    return candidates[0]


def zigzag_percent(prices: pd.Series, pct: float) -> list[Pivot]:
    arr = np.asarray(prices, dtype="float64").reshape(-1)
    n = int(arr.shape[0])
    if n < 3:
        return []

    threshold = float(pct) / 100.0
    pivots = []
    last_pivot_idx = 0
    last_pivot_price = float(arr[0])
    trend = None

    for i in range(1, n):
        price = float(arr[i])
        move = (price - last_pivot_price) / max(1e-12, last_pivot_price)

        if trend is None:
            if abs(move) >= threshold:
                trend = "up" if move > 0.0 else "down"
        elif trend == "up":
            if price > last_pivot_price:
                last_pivot_idx = i
                last_pivot_price = price
            else:
                drop = (last_pivot_price - price) / max(1e-12, last_pivot_price)
                if drop >= threshold:
                    pivots.append(Pivot(last_pivot_idx, last_pivot_price, "H"))
                    trend = "down"
                    last_pivot_idx = i
                    last_pivot_price = price
        else:
            if price < last_pivot_price:
                last_pivot_idx = i
                last_pivot_price = price
            else:
                rise = (price - last_pivot_price) / max(1e-12, last_pivot_price)
                if rise >= threshold:
                    pivots.append(Pivot(last_pivot_idx, last_pivot_price, "L"))
                    trend = "up"
                    last_pivot_idx = i
                    last_pivot_price = price

    if pivots:
        trailing_kind = "H" if pivots[-1].kind == "L" else "L"
        pivots.append(Pivot(last_pivot_idx, last_pivot_price, trailing_kind))
    else:
        trailing_kind = "H" if last_pivot_price >= float(arr[0]) else "L"
        pivots.append(Pivot(last_pivot_idx, last_pivot_price, trailing_kind))

    cleaned = []
    for pivot in pivots:
        if not cleaned or cleaned[-1].kind != pivot.kind:
            cleaned.append(pivot)
        elif pivot.kind == "H" and pivot.price >= cleaned[-1].price:
            cleaned[-1] = pivot
        elif pivot.kind == "L" and pivot.price <= cleaned[-1].price:
            cleaned[-1] = pivot
    return cleaned


def check_rules_for_sequence(seq: list[Pivot], state: str, strict: bool = True) -> tuple[str | None, dict, bool]:
    rules = {
        "Alternating Peaks/Valleys": True,
        "Wave 2 Retracement <= 100%": True,
        "Wave 3 Not Shortest": True,
        "Wave 4 No Overlap": True,
        "Direction Alignment": True
    }
    
    n = len(seq)
    if n < 3:
        return None, rules, False

    # 1. Alternating kinds (High and Low)
    for i in range(1, n):
        if seq[i].kind == seq[i-1].kind:
            rules["Alternating Peaks/Valleys"] = False
            return None, rules, False

    # Determine direction based on Wave 1 (0 -> 1)
    w1 = seq[1].price - seq[0].price
    direction = "bull" if w1 > 0 else "bear"
    
    # 2. Check Direction Alignment
    # Bullish: P1 > P0, P2 < P1, P3 > P2, P4 < P3, P5 > P4
    # Bearish: P1 < P0, P2 > P1, P3 < P2, P4 > P3, P5 < P4
    if direction == "bull":
        if seq[1].price <= seq[0].price or seq[2].price >= seq[1].price:
            rules["Direction Alignment"] = False
        if n > 3 and seq[3].price <= seq[2].price:
            rules["Direction Alignment"] = False
        if n > 4 and seq[4].price >= seq[3].price:
            rules["Direction Alignment"] = False
        if n > 5 and seq[5].price <= seq[4].price:
            rules["Direction Alignment"] = False
    else:
        if seq[1].price >= seq[0].price or seq[2].price <= seq[1].price:
            rules["Direction Alignment"] = False
        if n > 3 and seq[3].price >= seq[2].price:
            rules["Direction Alignment"] = False
        if n > 4 and seq[4].price <= seq[3].price:
            rules["Direction Alignment"] = False
        if n > 5 and seq[5].price >= seq[4].price:
            rules["Direction Alignment"] = False

    if not rules["Direction Alignment"]:
        return None, rules, False

    # 3. Rule 1: Wave 2 must not retrace more than 100% of Wave 1
    if direction == "bull":
        if seq[2].price <= seq[0].price:
            rules["Wave 2 Retracement <= 100%"] = False
    else:
        if seq[2].price >= seq[0].price:
            rules["Wave 2 Retracement <= 100%"] = False
            
    if not rules["Wave 2 Retracement <= 100%"]:
        return None, rules, False

    # 4. Rule 2: Wave 3 is not the shortest
    if n >= 4:
        w1_len = abs(seq[1].price - seq[0].price)
        w3_len = abs(seq[3].price - seq[2].price)
        if n >= 6:
            w5_len = abs(seq[5].price - seq[4].price)
            if w3_len < min(w1_len, w5_len):
                rules["Wave 3 Not Shortest"] = False
        else:
            # If Wave 5 is not complete yet, Wave 3 must not be significantly shorter than Wave 1
            if w3_len < w1_len * 0.9:
                rules["Wave 3 Not Shortest"] = False

    # 5. Rule 3: Wave 4 does not enter the price territory of Wave 1 (overlap check)
    if n >= 5:
        p1_price = seq[1].price
        p4_price = seq[4].price
        if direction == "bull":
            overlap = p4_price - p1_price
            if overlap < 0:
                if strict:
                    rules["Wave 4 No Overlap"] = False
                elif abs(overlap) / (seq[1].price - seq[0].price) > 0.15: # Allow up to 15% overlap in diagonal
                    rules["Wave 4 No Overlap"] = False
        else:
            overlap = p1_price - p4_price
            if overlap < 0:
                if strict:
                    rules["Wave 4 No Overlap"] = False
                elif abs(overlap) / (seq[0].price - seq[1].price) > 0.15:
                    rules["Wave 4 No Overlap"] = False

    is_valid = all(rules.values())
    return direction, rules, is_valid


def calculate_forecasts(seq: list[Pivot], state: str, direction: str) -> dict:
    forecasts = {}
    sign = 1 if direction == "bull" else -1
    
    p0 = seq[0].price
    p1 = seq[1].price
    p2 = seq[2].price
    
    w1_len = abs(p1 - p0)
    
    if state == WaveState.IN_WAVE_3:
        w3_target = p2 + sign * 1.618 * w1_len
        w3_len_proj = abs(w3_target - p2)
        w4_target = w3_target - sign * 0.382 * w3_len_proj
        w5_target = w4_target + sign * w1_len
        
        forecasts["Wave 3 Target"] = {
            "price": float(w3_target),
            "min": float(p2 + sign * 1.272 * w1_len),
            "max": float(p2 + sign * 2.618 * w1_len),
            "description": "FIB 1.618x Extension"
        }
        forecasts["Wave 4 Target"] = {
            "price": float(w4_target),
            "min": float(w3_target - sign * 0.5 * w3_len_proj),
            "max": float(w3_target - sign * 0.236 * w3_len_proj),
            "description": "FIB 38.2% Retracement"
        }
        forecasts["Wave 5 Target"] = {
            "price": float(w5_target),
            "min": float(w4_target + sign * 0.618 * w1_len),
            "max": float(w4_target + sign * 1.618 * w1_len),
            "description": "1.0x Wave 1 Extension"
        }
        
    elif state == WaveState.IN_WAVE_4:
        p3 = seq[3].price
        w3_len = abs(p3 - p2)
        
        w4_target = p3 - sign * 0.382 * w3_len
        w5_target = w4_target + sign * w1_len
        
        forecasts["Wave 4 Target"] = {
            "price": float(w4_target),
            "min": float(p3 - sign * 0.5 * w3_len),
            "max": float(p3 - sign * 0.236 * w3_len),
            "description": "FIB 38.2% Retracement"
        }
        forecasts["Wave 5 Target"] = {
            "price": float(w5_target),
            "min": float(w4_target + sign * 0.618 * w1_len),
            "max": float(w4_target + sign * 1.618 * w1_len),
            "description": "1.0x Wave 1 Extension"
        }
        
    elif state == WaveState.IN_WAVE_5:
        p3 = seq[3].price
        p4 = seq[4].price
        w3_len = abs(p3 - p2)
        
        w5_target_1 = p4 + sign * w1_len
        w5_target_2 = p4 + sign * 0.618 * abs(p3 - p0)
        w5_target = (w5_target_1 + w5_target_2) / 2.0
        
        forecasts["Wave 5 Target"] = {
            "price": float(w5_target),
            "min": float(min(w5_target_1, w5_target_2)),
            "max": float(max(w5_target_1, w5_target_2)),
            "description": "FIB 0.618x (0-3) Extension"
        }
        
        total_impulse_len = abs(w5_target - p0)
        a_target = w5_target - sign * 0.382 * total_impulse_len
        b_target = a_target + sign * 0.5 * abs(w5_target - a_target)
        c_target = b_target - sign * 1.0 * abs(w5_target - a_target)
        
        forecasts["Wave A Target"] = {"price": float(a_target), "description": "38.2% Retracement"}
        forecasts["Wave B Target"] = {"price": float(b_target), "description": "50% Retracement of Wave A"}
        forecasts["Wave C Target"] = {"price": float(c_target), "description": "1.0x Wave A Extension"}

    elif state == WaveState.COMPLETE_5W:
        p3 = seq[3].price
        p4 = seq[4].price
        p5 = seq[5].price
        
        total_impulse_len = abs(p5 - p0)
        
        a_target = p5 - sign * 0.382 * total_impulse_len
        b_target = a_target + sign * 0.5 * abs(p5 - a_target)
        c_target = b_target - sign * 1.0 * abs(p5 - a_target)
        
        forecasts["Wave A Target"] = {
            "price": float(a_target),
            "min": float(p5 - sign * 0.5 * total_impulse_len),
            "max": float(p5 - sign * 0.236 * total_impulse_len),
            "description": "FIB 38.2% Retracement"
        }
        forecasts["Wave B Target"] = {
            "price": float(b_target),
            "min": float(a_target + sign * 0.382 * abs(p5 - a_target)),
            "max": float(a_target + sign * 0.618 * abs(p5 - a_target)),
            "description": "FIB 50% Retracement of A"
        }
        forecasts["Wave C Target"] = {
            "price": float(c_target),
            "min": float(b_target - sign * 1.618 * abs(p5 - a_target)),
            "max": float(b_target - sign * 0.618 * abs(p5 - a_target)),
            "description": "1.0x Wave A Extension"
        }
        
    return forecasts


def score_structure(seq: list[Pivot], state: str, direction: str) -> float:
    n = len(seq)
    score = 10.0
    
    if n >= 4:
        w1 = abs(seq[1].price - seq[0].price)
        w3 = abs(seq[3].price - seq[2].price)
        ratio = w3 / max(w1, 1e-9)
        if ratio >= 1.618:
            score += 2.0
        elif ratio >= 1.0:
            score += 1.0
        else:
            score -= 2.0
            
    if n >= 3:
        w1 = abs(seq[1].price - seq[0].price)
        w2 = abs(seq[2].price - seq[1].price)
        retr2 = w2 / max(w1, 1e-9)
        if 0.5 <= retr2 <= 0.786:
            score += 1.0
        elif retr2 > 0.9:
            score -= 1.5
            
    if n >= 5:
        w3 = abs(seq[3].price - seq[2].price)
        w4 = abs(seq[4].price - seq[3].price)
        retr4 = w4 / max(w3, 1e-9)
        if 0.236 <= retr4 <= 0.382:
            score += 1.0
        elif retr4 > 0.5:
            score -= 1.5
            
    return float(score)


def scan_for_elliott_structures(pivots: list[Pivot], strict: bool = True, mode: str = "Auto-Detect", manual_wave_0: int = 0) -> ElliottWaveStructure | None:
    n = len(pivots)
    if n < 3:
        return None
        
    if mode == "Manual Pinning":
        # Fit structure starting exactly at the selected manual_wave_0 pivot
        if manual_wave_0 < 0 or manual_wave_0 >= n - 2:
            return None
        # Try length 6 down to 3
        for length, state in [(6, WaveState.COMPLETE_5W), (5, WaveState.IN_WAVE_5), (4, WaveState.IN_WAVE_4), (3, WaveState.IN_WAVE_3)]:
            if manual_wave_0 + length <= n:
                seq = pivots[manual_wave_0 : manual_wave_0 + length]
                direction, rules, is_valid = check_rules_for_sequence(seq, state, strict)
                if is_valid:
                    score = score_structure(seq, state, direction)
                    forecasts = calculate_forecasts(seq, state, direction)
                    return ElliottWaveStructure(
                        state=state,
                        direction=direction,
                        pivot_idxs=list(range(manual_wave_0, manual_wave_0 + length)),
                        pivots=seq,
                        rules_check=rules,
                        score=score,
                        forecasts=forecasts
                    )
        return None
        
    # Auto-Detect mode: scan right-to-left (newest first)
    for length, state in [(6, WaveState.COMPLETE_5W), (5, WaveState.IN_WAVE_5), (4, WaveState.IN_WAVE_4), (3, WaveState.IN_WAVE_3)]:
        for start in range(n - length, -1, -1):
            seq = pivots[start:start+length]
            direction, rules, is_valid = check_rules_for_sequence(seq, state, strict)
            if is_valid:
                score = score_structure(seq, state, direction)
                forecasts = calculate_forecasts(seq, state, direction)
                return ElliottWaveStructure(
                    state=state,
                    direction=direction,
                    pivot_idxs=list(range(start, start + length)),
                    pivots=seq,
                    rules_check=rules,
                    score=score,
                    forecasts=forecasts
                )
    return None


def build_levels(pivots: list[Pivot], best: ElliottWaveStructure | None, show_abc: bool) -> tuple[pd.DataFrame, dict | None]:
    rows = []
    abc = None
    if best is None:
        return pd.DataFrame(columns=["Group", "Level", "Price"]), abc

    seq = best.pivots
    w1_start = seq[0].price
    w1_end = seq[1].price
    w2_end = seq[2].price
    
    # Wave 2 Retracements
    length_w1 = w1_end - w1_start
    for r in [0.382, 0.5, 0.618, 0.786]:
        rows.append(["Wave2 retr", f"{int(r*100)}%", float(w1_end - r * length_w1)])
        
    if len(seq) >= 4:
        w2_end = seq[2].price
        w3_end = seq[3].price
        length_w3 = w3_end - w2_end
        
        # Wave 4 Retracements
        for r in [0.382, 0.5, 0.618]:
            rows.append(["Wave4 retr", f"{int(r*100)}%", float(w3_end - r * length_w3)])
            
        # Wave 3 Extensions
        for ext in [1.0, 1.272, 1.618, 2.0, 2.618]:
            rows.append(["Wave3 ext", f"{ext}x", float(w2_end + ext * length_w1)])
            
    if len(seq) >= 5:
        w4_end = seq[4].price
        # Wave 5 Extensions
        for ext in [0.618, 1.0, 1.618]:
            rows.append(["Wave5 ext", f"{ext}x", float(w4_end + ext * length_w1)])

    if best.state == WaveState.COMPLETE_5W and len(seq) >= 6:
        p5 = seq[5].price
        p0 = seq[0].price
        sign = 1 if best.direction == "bull" else -1
        total_len = abs(p5 - p0)
        
        a = p5 - sign * 0.382 * total_len
        b = a + sign * 0.5 * abs(p5 - a)
        c = b - sign * 1.0 * abs(p5 - a)
        abc = {"A": a, "B": b, "C_target": float(c)}
        rows.append(["ABC", "C_target", float(c)])

    return pd.DataFrame(rows, columns=["Group", "Level", "Price"]), abc


def pivots_to_frame(df: pd.DataFrame, pivots: list[Pivot]) -> pd.DataFrame:
    rows = []
    for i, pivot in enumerate(pivots):
        dt = df.index[pivot.idx] if 0 <= pivot.idx < len(df.index) else None
        rows.append(
            {
                "Pivot #": int(i),
                "df_index": int(pivot.idx),
                "Date": dt.strftime("%Y-%m-%d %H:%M") if hasattr(dt, "strftime") else str(dt),
                "Price": float(pivot.price),
                "Kind": "High" if pivot.kind == "H" else "Low",
            }
        )
    return pd.DataFrame(rows)


def build_elliott_chart(
    df: pd.DataFrame,
    pivots: list[Pivot],
    best: ElliottWaveStructure | None,
    levels: pd.DataFrame,
    abc: dict | None,
    symbol: str,
    zigzag_pct: float,
    plot_last_n: int,
    show_levels: bool,
    show_minor_pivots: bool,
) -> go.Figure:
    tail = df if plot_last_n <= 0 else df.iloc[-plot_last_n:]
    start_pos = len(df) - len(tail)
    end_pos = len(df) - 1

    fig = go.Figure()
    
    # Close price line (Subtle and clean)
    fig.add_trace(
        go.Scatter(
            x=tail.index,
            y=tail["Close_PLOT"],
            mode="lines",
            name="Close Price",
            line=dict(color="rgba(148, 163, 184, 0.45)", width=1.5),
        )
    )

    # Median duration of waves to project forecast dates
    bar_delta = pd.Series(df.index).diff().median()
    if pd.isna(bar_delta):
        bar_delta = timedelta(days=1)
        
    avg_bars = 15
    if best is not None and len(best.pivot_idxs) > 1:
        avg_bars = int(np.median([pivots[idx].idx - pivots[idx-1].idx for idx in best.pivot_idxs[1:]]))
        avg_bars = max(5, avg_bars)

    # 1. Plot minor ZigZag pivots if enabled
    if show_minor_pivots:
        visible_pivots = [pivot for pivot in pivots if start_pos <= pivot.idx <= end_pos]
        if visible_pivots:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[pivot.idx] for pivot in visible_pivots],
                    y=[pivot.price for pivot in visible_pivots],
                    mode="markers+text",
                    name="Minor Pivots",
                    text=[pivot.kind for pivot in visible_pivots],
                    textposition="top center",
                    marker=dict(size=5, color="rgba(100, 116, 139, 0.6)", line=dict(color="#0d1b2e", width=1)),
                )
            )

    # 2. Draw Historical Elliott Waves & Forecast lines
    if best is not None:
        seq = best.pivots
        direction_color = "#10b981" if best.direction == "bull" else "#ef4444"
        
        # Historical segments
        hx = [df.index[p.idx] for p in seq]
        hy = [p.price for p in seq]
        
        # Add labels
        labels = [str(i) for i in range(len(seq))]
        
        # Historical Wave paths (Solid bold line)
        fig.add_trace(
            go.Scatter(
                x=hx,
                y=hy,
                mode="lines+markers+text",
                name="Confirmed Waves",
                text=labels,
                textposition="top center" if best.direction == "bull" else "bottom center",
                textfont=dict(size=14, color="#ffffff", family="Courier New, monospace"),
                line=dict(color=direction_color, width=3.5),
                marker=dict(size=10, color=direction_color, symbol="circle-dot")
            )
        )
        
        # Forecasted segments
        last_dt = df.index[seq[-1].idx]
        forecast_pts = []
        
        if best.state == WaveState.IN_WAVE_3:
            w3_t = best.forecasts["Wave 3 Target"]
            w4_t = best.forecasts["Wave 4 Target"]
            w5_t = best.forecasts["Wave 5 Target"]
            forecast_pts = [
                ("3", last_dt + avg_bars * bar_delta, w3_t),
                ("4", last_dt + 2 * avg_bars * bar_delta, w4_t),
                ("5", last_dt + 3 * avg_bars * bar_delta, w5_t),
            ]
        elif best.state == WaveState.IN_WAVE_4:
            w4_t = best.forecasts["Wave 4 Target"]
            w5_t = best.forecasts["Wave 5 Target"]
            forecast_pts = [
                ("4", last_dt + avg_bars * bar_delta, w4_t),
                ("5", last_dt + 2 * avg_bars * bar_delta, w5_t),
            ]
        elif best.state == WaveState.IN_WAVE_5:
            w5_t = best.forecasts["Wave 5 Target"]
            forecast_pts = [
                ("5", last_dt + avg_bars * bar_delta, w5_t),
            ]
            if "Wave A Target" in best.forecasts:
                forecast_pts.extend([
                    ("A", last_dt + 2 * avg_bars * bar_delta, best.forecasts["Wave A Target"]),
                    ("B", last_dt + 3 * avg_bars * bar_delta, best.forecasts["Wave B Target"]),
                    ("C", last_dt + 4 * avg_bars * bar_delta, best.forecasts["Wave C Target"]),
                ])
        elif best.state == WaveState.COMPLETE_5W:
            if "Wave A Target" in best.forecasts:
                forecast_pts = [
                    ("A", last_dt + avg_bars * bar_delta, best.forecasts["Wave A Target"]),
                    ("B", last_dt + 2 * avg_bars * bar_delta, best.forecasts["Wave B Target"]),
                    ("C", last_dt + 3 * avg_bars * bar_delta, best.forecasts["Wave C Target"]),
                ]
                
        # Draw Forecast paths (Dashed line) and shaded target ranges
        if forecast_pts:
            fx = [hx[-1]] + [pt[1] for pt in forecast_pts]
            fy = [hy[-1]] + [pt[2]["price"] for pt in forecast_pts]
            flabels = [""] + [pt[0] for pt in forecast_pts]
            
            fig.add_trace(
                go.Scatter(
                    x=fx,
                    y=fy,
                    mode="lines+markers+text",
                    name="Forecasted Projections",
                    text=flabels,
                    textposition="top center" if best.direction == "bull" else "bottom center",
                    textfont=dict(size=14, color="rgba(255, 255, 255, 0.8)", family="Courier New, monospace"),
                    line=dict(color=direction_color, width=2, dash="dashdot"),
                    marker=dict(size=8, color=direction_color, symbol="star-triangle-up")
                )
            )
            
            # Shaded Fibonacci target zones
            for label, f_date, t_info in forecast_pts:
                if "min" in t_info and "max" in t_info:
                    t_min = t_info["min"]
                    t_max = t_info["max"]
                    half_width = 0.25 * avg_bars * bar_delta
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[f_date - half_width, f_date + half_width, f_date + half_width, f_date - half_width],
                            y=[t_min, t_min, t_max, t_max],
                            fill="toself",
                            fillcolor="rgba(16, 185, 129, 0.08)" if best.direction == "bull" else "rgba(239, 110, 110, 0.08)",
                            line=dict(color="rgba(16, 185, 129, 0.25)" if best.direction == "bull" else "rgba(239, 110, 110, 0.25)", width=1),
                            name=f"Wave {label} Target Zone",
                            hoverinfo="text",
                            text=f"Wave {label} target range: {t_min:.2f} - {t_max:.2f}",
                            hoveron="fills",
                            showlegend=False
                        )
                    )

    # 3. Fibonacci level lines if enabled
    if show_levels and not levels.empty:
        for _, row in levels.iterrows():
            if row["Group"] != "ABC":
                fig.add_hline(
                    y=row["Price"],
                    line_width=0.8,
                    line_dash="dash",
                    line_color="rgba(148, 163, 184, 0.3)",
                    annotation_text=f"{row['Group']} {row['Level']}",
                    annotation_position="right",
                    annotation_font_size=9,
                    annotation_font_color="rgba(203, 213, 225, 0.7)",
                )

    title = f"{symbol} | ZigZag {zigzag_pct:.1f}%"
    if best is not None:
        title += f" | {best.state.replace('_', ' ')} ({best.direction.upper()}) | Score {best.score:.1f}"
    else:
        title += " | No structure detected"

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#0c1017",
        plot_bgcolor="#0c1017",
        height=600,
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.05)")
    return fig


class ElliottWaveAnalysisModule(FazDaneModule):
    MODULE_NAME = "Elliott Wave Analysis"
    MODULE_ICON = "Wave"
    MODULE_DESCRIPTION = "ZigZag pivot detection, rules-based wave forecasting, Fibonacci target extensions, and rules checklist dashboards"
    TIER = 3
    SOURCE_NOTEBOOK = "Forecasting/Cycle Analysis/Elliott Wave Analysis"
    CACHE_TTL = 3600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def render_sidebar(self):
        self._default_universe()
        st.markdown("**Elliott Universe**")
        self.universe_name, tickers, _ = render_universe_manager(
            key_prefix="elliott",
            show_benchmark=False,
            label="Ticker Universe:",
        )
        ticker_names = get_ticker_names(self.universe_name)
        if "^GSPC" not in tickers:
            tickers = ["^GSPC"] + tickers
            ticker_names.setdefault("^GSPC", "S&P 500 Index")
        self.tickers = tickers
        default_idx = self.tickers.index("^GSPC") if "^GSPC" in self.tickers else 0
        if st.session_state.get("elliott_ticker") not in self.tickers:
            st.session_state["elliott_ticker"] = self.tickers[default_idx]
        self.ticker = st.selectbox(
            "Ticker / Index:",
            self.tickers,
            index=default_idx,
            key="elliott_ticker",
            format_func=lambda ticker: format_ticker_display(ticker, ticker_names),
        )

        st.markdown("**Data Window**")
        self.period = st.selectbox("Period:", ["6mo", "1y", "2y", "5y", "10y", "max"], index=2, key="elliott_period")
        self.interval = st.selectbox("Interval:", ["1d", "1h", "15m"], index=0, key="elliott_interval")

        st.markdown("**Wave Detection & Controls**")
        self.zigzag_pct = float(st.slider("ZigZag Reversal %:", 1.0, 12.0, 5.0, step=0.5, key="elliott_zigzag"))
        self.plot_last_n = int(st.slider("Plot Last N Bars:", 100, 1500, 600, step=50, key="elliott_plot_n"))
        self.show_levels = st.checkbox("Show Fibonacci levels", value=True, key="elliott_levels")
        
        st.markdown("**Forecasting Engine Settings**")
        self.ew_mode = st.selectbox("Analysis Mode:", ["Auto-Detect", "Manual Pinning"], index=0, key="elliott_ew_mode")
        self.strict_rules = st.checkbox("Strict Rules (Forbid Wave 4 overlap)", value=True, key="elliott_strict")
        self.show_minor_pivots = st.checkbox("Show minor ZigZag pivots", value=False, key="elliott_minor_pivots")

        if st.button("Refresh Elliott Wave", use_container_width=True, type="primary", key="elliott_refresh"):
            fetch_elliott_data.clear()
            st.rerun()

    def render_main(self):
        self.render_section_header(
            "Elliott Wave Analysis",
            "Modern rules-based wave forecasting from ZigZag pivots with Fibonacci projections and corrective targets",
        )

        symbol = normalize_symbol(self.ticker)
        with st.spinner(f"Fetching {symbol} and scanning Elliott Wave pivots..."):
            df = fetch_elliott_data(symbol, self.period, self.interval)

        if df.empty:
            st.warning(f"No data returned for {symbol}. Try another ticker, period, or interval.")
            return
        if len(df) < 30:
            st.warning("Not enough bars returned for Elliott Wave analysis.")
            return

        pivots = zigzag_percent(df["Close_PLOT"], self.zigzag_pct)
        
        # If Manual Pinning, render selector in main container/sidebar
        manual_wave_0 = 0
        if self.ew_mode == "Manual Pinning":
            if not pivots:
                st.warning("No pivots detected to pin.")
                return
            pivots_list = pivots_to_frame(df, pivots)
            
            # Select start pivot index
            manual_wave_0 = st.selectbox(
                "📍 Select Wave 0 Start Pivot:",
                options=range(len(pivots) - 2),
                format_func=lambda idx: f"Pivot #{idx}: {'High' if pivots[idx].kind == 'H' else 'Low'} @ ${pivots[idx].price:.2f} ({pivots_list.loc[idx, 'Date']})",
                key="elliott_manual_w0"
            )
            
        best = scan_for_elliott_structures(
            pivots=pivots,
            strict=self.strict_rules,
            mode=self.ew_mode,
            manual_wave_0=manual_wave_0
        )
        
        levels, abc = build_levels(pivots, best, show_abc=True)
        pivots_df = pivots_to_frame(df, pivots)

        # Overview Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", symbol)
        c2.metric("Total Bars Checked", f"{len(df):,}")
        c3.metric("ZigZag Pivots", f"{len(pivots):,}")
        
        if best:
            state_disp = best.state.replace("IN_", "In ").replace("_", " ").title()
            c4.metric("Active Wave State", state_disp, f"Score: {best.score:.1f}")
        else:
            c4.metric("Active Wave State", "No Wave Pattern", None)

        # Plotly chart
        st.plotly_chart(
            build_elliott_chart(
                df=df,
                pivots=pivots,
                best=best,
                levels=levels,
                abc=abc,
                symbol=symbol,
                zigzag_pct=self.zigzag_pct,
                plot_last_n=self.plot_last_n,
                show_levels=self.show_levels,
                show_minor_pivots=self.show_minor_pivots
            ),
            use_container_width=True,
        )

        # Rules Check Dashboard & Details Panel
        col_rules, col_info = st.columns([1, 1])
        
        with col_rules:
            st.markdown("### 📋 Elliott Wave Theory Rule Check")
            if best:
                for rule_name, passed in best.rules_check.items():
                    status_emoji = "✅" if passed else "❌"
                    status_text = "PASSED" if passed else "VIOLATED"
                    st.markdown(f"**{status_emoji} {rule_name}**: `{status_text}`")
            else:
                st.info("No active wave pattern matched. Adjust your ZigZag Reversal % or toggle 'Strict Rules' in the sidebar.")

        with col_info:
            st.markdown("### 🎯 Forecasted Target Ranges")
            if best and best.forecasts:
                for t_name, t_val in best.forecasts.items():
                    if "min" in t_val and "max" in t_val:
                        st.markdown(f"**{t_name}**: `${t_val['price']:.2f}` (Range: `${t_val['min']:.2f}` - `${t_val['max']:.2f}`) — *{t_val['description']}*")
                    else:
                        st.markdown(f"**{t_name}**: `${t_val['price']:.2f}` — *{t_val['description']}*")
            else:
                st.caption("Forecasts are generated automatically when a valid wave structure is found.")

        # Data Detail Tabs
        tab_summary, tab_pivots, tab_levels, tab_exports = st.tabs(["Summary", "Pivots List", "Fibonacci Levels", "Data Exports"])

        with tab_summary:
            if best is None:
                st.info("No clean Elliott Wave pattern was identified under the current parameter options.")
            else:
                seq = best.pivots
                waves = [abs(seq[i + 1].price - seq[i].price) for i in range(len(seq) - 1)]
                
                wave_rows = []
                for i in range(len(seq) - 1):
                    wave_rows.append({
                        "Wave segment": f"{i} -> {i+1}",
                        "Start Date": df.index[seq[i].idx].strftime("%Y-%m-%d"),
                        "End Date": df.index[seq[i+1].idx].strftime("%Y-%m-%d"),
                        "Points delta ($)": round(seq[i+1].price - seq[i].price, 2),
                        "Abs Length ($)": round(waves[i], 2)
                    })
                st.dataframe(pd.DataFrame(wave_rows), use_container_width=True, hide_index=True)
                
                if len(waves) >= 3:
                    col_r1, col_r2 = st.columns(2)
                    col_r1.metric("Wave 3 / Wave 1 Ratio", f"{waves[2] / max(waves[0], 1e-9):.3f}")
                    if len(waves) >= 5:
                        col_r2.metric("Wave 5 / Wave 3 Ratio", f"{waves[4] / max(waves[2], 1e-9):.3f}")

        with tab_pivots:
            st.dataframe(pivots_df.round(2), use_container_width=True, hide_index=True)

        with tab_levels:
            if levels.empty:
                st.info("Fibonacci levels will populate when an active wave pattern is detected.")
            else:
                st.dataframe(levels.round(2), use_container_width=True, hide_index=True)

        with tab_exports:
            run_log = self._run_log(symbol, df, pivots, best)
            c_dl1, c_dl2, c_dl3 = st.columns(3)
            
            c_dl1.download_button(
                "Download Pivots CSV",
                data=pivots_df.to_csv(index=False),
                file_name=f"elliott_pivots_{symbol.replace('^', '').lower()}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            c_dl2.download_button(
                "Download Wave Levels CSV",
                data=levels.to_csv(index=False),
                file_name=f"elliott_levels_{symbol.replace('^', '').lower()}.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=levels.empty,
            )
            c_dl3.download_button(
                "Download Process Log",
                data=run_log,
                file_name=f"elliott_process_log_{symbol.replace('^', '').lower()}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    def _default_universe(self):
        key = "elliott_sel"
        target = "Index Universe"
        if target in get_universe_names() and key not in st.session_state:
            st.session_state[key] = target

    def _run_log(self, symbol: str, df: pd.DataFrame, pivots: list[Pivot], best: ElliottWaveStructure | None) -> str:
        start = df.index.min().strftime("%Y-%m-%d")
        end = df.index.max().strftime("%Y-%m-%d")
        lines = [
            "Elliott Wave Run Log",
            "--------------------",
            f"Ticker        : {symbol}",
            f"Period        : {self.period}",
            f"Interval      : {self.interval}",
            f"ZigZag %      : {self.zigzag_pct}",
            f"Strict Rules  : {self.strict_rules}",
            f"Analysis Mode : {self.ew_mode}",
            f"Range         : {start} to {end}",
            f"Bars Count    : {len(df)}",
            f"Pivots Count  : {len(pivots)}",
        ]
        if best is None:
            lines.append("Impulse Pattern: None detected")
        else:
            lines.extend(
                [
                    f"Active Wave   : {best.state}",
                    f"Direction     : {best.direction.upper()}",
                    f"Score         : {best.score:.1f}",
                    f"Pivot Indices : {best.pivot_idxs}",
                ]
            )
        lines.append(f"Generated At  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)
