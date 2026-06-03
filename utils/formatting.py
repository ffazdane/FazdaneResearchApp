"""
Display Formatting Helpers for FazDane Research Application.
Contains reusable functions for calculating and formatting price Action indicators,
such as the 15-day price change (3-week) Strength indicator.
"""

import pandas as pd

def calculate_strength_pct(data) -> float | None:
    """
    Calculate the 15-day price change (3-week strength).
    Accepts either a pandas Series of close prices, or a DataFrame containing a 'Close' column.
    """
    if data is None or len(data) < 16:
        return None
    try:
        if isinstance(data, pd.DataFrame):
            if 'Close' not in data.columns:
                return None
            s = data['Close'].dropna()
        else:
            s = data.dropna()
            
        if len(s) < 16:
            return None
            
        current_val = float(s.iloc[-1])
        close_15d_ago = float(s.iloc[-16])
        if close_15d_ago > 0 and current_val > 0:
            return (current_val - close_15d_ago) / close_15d_ago
    except Exception:
        pass
    return None

def format_strength_meter(strength_pct: float | None) -> tuple[str, str]:
    """
    Calculate the thinkorswim (tos) style strength meter triangles and color.
    Uptrend (> +10%): '▲' in green (#00D4AA)
    Downtrend (< -10%): '▼' in red (#FF4B4B)
    Range-Bound (-10% to +10%): '▶' in yellow/orange (#FFA421)
    """
    if strength_pct is None:
        return "—", "#888888"
    try:
        val = float(strength_pct)
        if val > 0.10:
            return "▲", "#00D4AA"
        elif val < -0.10:
            return "▼", "#FF4B4B"
        else:
            return "▶", "#FFA421"
    except (ValueError, TypeError):
        return "—", "#888888"

def format_strength_meter_html(strength_pct: float | None) -> str:
    """Get the strength meter rendered as a styled HTML span."""
    bars, color = format_strength_meter(strength_pct)
    if bars == "—":
        return f'<span style="color: {color}; font-family: sans-serif;">{bars}</span>'
    return f'<span style="color: {color}; font-size: 14px; font-weight: bold;">{bars}</span>'
