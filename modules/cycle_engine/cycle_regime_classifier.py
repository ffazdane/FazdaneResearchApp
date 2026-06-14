import numpy as np
import pandas as pd

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def classify_market_regime(
    df: pd.DataFrame,
    cycle_direction: str,
    cycle_phase_pct: float,
    vix_value: float = 15.0
) -> dict:
    """
    Integrates moving averages, ATR expansion, and cycle phase to classify
    the current market regime and assign scoring values.
    """
    if len(df) < 50:
        return {
            "regime": "No Edge",
            "regime_score": 50.0,
            "buy_the_dip_score": 30.0,
            "sell_the_rip_score": 30.0,
            "sideways_score": 50.0,
            "support": df["Close"].min() if not df.empty else 0.0,
            "resistance": df["Close"].max() if not df.empty else 0.0
        }

    close_series = df["Close"]
    close = close_series.iloc[-1]
    
    # 1. Compute Technical Indicators
    ma20 = close_series.rolling(window=20).mean().iloc[-1]
    ma50 = close_series.rolling(window=50).mean().iloc[-1]
    ma200 = close_series.rolling(window=200).mean().iloc[-1] if len(df) >= 200 else close_series.rolling(window=50).mean().iloc[-1]
    
    # Compute ATR
    atr_series = calculate_atr(df)
    atr = atr_series.iloc[-1] if not atr_series.empty else close * 0.015
    atr_norm = atr / close * 100.0 # ATR as % of price
    
    # Determine support & resistance levels (using rolling min/max)
    support = float(close_series.rolling(window=40).min().iloc[-1])
    resistance = float(close_series.rolling(window=40).max().iloc[-1])
    
    # Determine MA slopes
    ma50_prev = close_series.rolling(window=50).mean().iloc[-5] if len(df) >= 55 else ma50
    ma50_slope = (ma50 - ma50_prev) / ma50_prev * 100.0 # % change over 5 days
    
    # Proximity metrics
    dist_to_ma50 = (close - ma50) / ma50 * 100.0
    dist_to_ma200 = (close - ma200) / ma200 * 100.0

    # 2. Score sub-categories (0-100)
    
    # Buy the Dip Score
    # Prefers: MA 50/200 sloping up (bull market), price near/under MA 50, cycle phase at bottom (90-100%) or early expansion (0-15%)
    dip_score = 0.0
    if ma50_slope > 0: # Bull market
        dip_score += 40.0
        if dist_to_ma50 <= 1.0: # Close to MA50 or lower
            dip_score += 30.0
        if cycle_phase_pct >= 85.0 or cycle_phase_pct <= 20.0: # Cycle bottoming
            dip_score += 30.0
    buy_the_dip_score = max(min(dip_score, 100.0), 10.0)

    # Sell the Rip Score
    # Prefers: MA 50/200 sloping down (bear market), price near/above MA 50, cycle phase peaking (40-60%)
    rip_score = 0.0
    if ma50_slope < 0: # Bear market
        rip_score += 40.0
        if dist_to_ma50 >= -1.0: # Near or above MA50
            rip_score += 30.0
        if 35.0 <= cycle_phase_pct <= 65.0: # Cycle peaking
            rip_score += 30.0
    sell_the_rip_score = max(min(rip_score, 100.0), 10.0)

    # Sideways Score
    # Prefers: MA 50/200 flat, price trading inside support/resistance bounds, low VIX, low ATR
    flat_score = 100.0 - abs(ma50_slope) * 20.0 - abs(dist_to_ma50) * 5.0
    if vix_value > 22.0:
        flat_score -= 20.0
    if atr_norm > 2.5:
        flat_score -= 20.0
    sideways_score = max(min(flat_score, 100.0), 0.0)

    # 3. Rule-based Regime Classification
    if close > ma50 > ma200 and cycle_direction == "rising":
        regime = "Bull Trend"
        regime_score = 80.0 + (ma50_slope * 10.0)
    elif close < ma50 < ma200 and cycle_direction == "falling":
        regime = "Bear Trend"
        regime_score = 80.0 - (ma50_slope * 10.0)
    elif ma50_slope > 0 and dist_to_ma50 < 0 and (cycle_phase_pct >= 80.0 or cycle_phase_pct <= 20.0):
        regime = "Dip Buy Zone"
        regime_score = buy_the_dip_score
    elif ma50_slope < 0 and dist_to_ma50 > 0 and (35.0 <= cycle_phase_pct <= 65.0):
        regime = "Sell the Rip Zone"
        regime_score = sell_the_rip_score
    elif abs(ma50_slope) < 0.1 and sideways_score > 65.0:
        regime = "Sideways Range"
        regime_score = sideways_score
    elif atr_norm > 2.5 and vix_value > 25.0:
        regime = "Volatile Range"
        regime_score = 50.0 + (vix_value - 25.0) * 2.0
    elif close > ma20 * 1.05 and 40.0 <= cycle_phase_pct <= 60.0:
        regime = "Trend Exhaustion"
        regime_score = 75.0
    else:
        regime = "No Edge"
        regime_score = 50.0

    return {
        "regime": regime,
        "regime_score": round(min(regime_score, 100.0), 1),
        "buy_the_dip_score": round(buy_the_dip_score, 1),
        "sell_the_rip_score": round(sell_the_rip_score, 1),
        "sideways_score": round(sideways_score, 1),
        "support": round(support, 2),
        "resistance": round(resistance, 2)
    }
