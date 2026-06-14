import numpy as np
import pandas as pd
from modules.cycle_engine.dominant_cycle_detector import detrend_series

def calculate_cycle_phase(prices: pd.Series, dominant_cycle_days: float) -> dict:
    """
    Fits a sine wave of the dominant period to detrended price data
    and computes the current phase (0% to 100%).
    0% = cycle bottom
    50% = cycle peak
    100% = next cycle bottom
    """
    if len(prices) < 10:
        return {
            "cycle_phase_pct": 50.0,
            "cycle_direction": "rising",
            "phase_label": "Mid Expansion",
            "estimated_days_to_bottom": int(dominant_cycle_days / 2),
            "estimated_days_to_peak": 0
        }

    detrended = detrend_series(prices)
    n = len(detrended)
    t = np.arange(n)
    omega = 2.0 * np.pi / dominant_cycle_days

    # Fit y = A * cos(omega*t) + B * sin(omega*t) + C
    X = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones(n)])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, detrended, rcond=None)
        A, B, _ = coeffs
        
        # Express as R * sin(omega*t + phase_offset)
        # y = R * (cos(phase_offset)*sin(omega*t) + sin(phase_offset)*cos(omega*t))
        # Thus: R * sin(phase_offset) = A  and R * cos(phase_offset) = B
        phase_offset = np.arctan2(A, B)
        
        # Current phase angle at the latest data point (t = n - 1)
        theta = omega * (n - 1) + phase_offset
        
        # Normalize theta to [0, 2*pi)
        theta_norm = theta % (2.0 * np.pi)
        
        # Map theta_norm to cycle phase relative to bottom (3*pi/2 is bottom, 0.5*pi is peak)
        # We define bottom at 0% (theta = 1.5*pi) and peak at 50% (theta = 0.5*pi)
        alpha = (theta_norm - 1.5 * np.pi) % (2.0 * np.pi)
        phase_pct = (alpha / (2.0 * np.pi)) * 100.0
    except Exception:
        # Fallback to simple estimate based on last momentum indicator
        ma20 = prices.rolling(window=20).mean().iloc[-1]
        p = prices.iloc[-1]
        phase_pct = 25.0 if p > ma20 else 75.0

    # 1. Classify phase label
    if 0.0 <= phase_pct < 20.0:
        phase_label = "Early Expansion"
        direction = "rising"
    elif 20.0 <= phase_pct < 40.0:
        phase_label = "Mid Expansion"
        direction = "rising"
    elif 40.0 <= phase_pct < 55.0:
        phase_label = "Peak Zone"
        direction = "peaking"
    elif 55.0 <= phase_pct < 75.0:
        phase_label = "Early Contraction"
        direction = "falling"
    elif 75.0 <= phase_pct < 90.0:
        phase_label = "Late Contraction"
        direction = "falling"
    else:
        phase_label = "Bottom Zone"
        direction = "bottoming"

    # 2. Estimate days to turning points
    # 0% is bottom, 50% is peak, 100% is bottom
    if phase_pct < 50.0:
        # We are rising towards the peak (50%)
        days_to_peak = max(int(((50.0 - phase_pct) / 100.0) * dominant_cycle_days), 0)
        days_to_bottom = int(((100.0 - phase_pct) / 100.0) * dominant_cycle_days)
    else:
        # We are falling towards the bottom (100%)
        days_to_peak = int(((150.0 - phase_pct) / 100.0) * dominant_cycle_days)
        days_to_bottom = max(int(((100.0 - phase_pct) / 100.0) * dominant_cycle_days), 0)

    return {
        "cycle_phase_pct": round(phase_pct, 1),
        "cycle_direction": direction,
        "phase_label": phase_label,
        "estimated_days_to_bottom": days_to_bottom,
        "estimated_days_to_peak": days_to_peak
    }
