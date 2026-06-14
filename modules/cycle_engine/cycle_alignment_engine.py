import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from modules.cycle_engine.dominant_cycle_detector import detect_dominant_cycle
from modules.cycle_engine.cycle_phase_engine import calculate_cycle_phase

def fetch_index_prices(start_date: date, end_date: date) -> dict[str, pd.Series]:
    """Fetch index close prices for alignment analysis."""
    tickers = {
        "SPY": "SPY",
        "QQQ": "QQQ",
        "IWM": "IWM",
        "DIA": "DIA",
        "VIX": "^VIX"
    }
    
    out = {}
    # Fetch in a single batch download for efficiency
    try:
        data = yf.download(list(tickers.values()), start=start_date, end=end_date + timedelta(days=1), auto_adjust=True, progress=False)
        if data.empty:
            return {}
            
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            
        close_data = data["Close"]
        for key, ticker in tickers.items():
            if ticker in close_data.columns:
                out[key] = close_data[ticker].dropna()
    except Exception:
        pass
    return out

def calculate_alignment(as_of_date: date) -> dict:
    """
    Computes cycle directions of SPY, QQQ, IWM, DIA, and VIX.
    Returns weighted alignment score (0-100) and state.
    """
    start_date = as_of_date - timedelta(days=180)
    prices_dict = fetch_index_prices(start_date, as_of_date)
    
    if not prices_dict or len(prices_dict) < 3:
        # Fallback if downloads fail
        return {
            "SPY": "No Edge", "QQQ": "No Edge", "IWM": "No Edge", "DIA": "No Edge", "VIX": "No Edge",
            "alignment_score": 50.0,
            "alignment_state": "No Edge",
            "directions": {}
        }
        
    directions = {}
    phases = {}
    
    for key, prices in prices_dict.items():
        if len(prices) < 15:
            directions[key] = "Sideways"
            phases[key] = 50.0
            continue
        try:
            cycle_info = detect_dominant_cycle(prices)
            phase_info = calculate_cycle_phase(prices, cycle_info["dominant_cycle_days"])
            
            phases[key] = phase_info["cycle_phase_pct"]
            if phase_info["cycle_direction"] == "rising":
                directions[key] = "Rising"
            elif phase_info["cycle_direction"] == "falling":
                directions[key] = "Falling"
            else:
                directions[key] = "Sideways"
        except Exception:
            directions[key] = "Sideways"
            phases[key] = 50.0
            
    # Calculate weighted directional sum
    # Rising is +1, Falling is -1. Sideways is 0.
    # VIX is inverse (falling VIX is bullish).
    dir_vals = {}
    for key in ["SPY", "QQQ", "IWM", "DIA", "VIX"]:
        d = directions.get(key, "Sideways")
        val = 1.0 if d == "Rising" else -1.0 if d == "Falling" else 0.0
        if key == "VIX":
            val = -val # VIX inverse
        dir_vals[key] = val
        
    # Weights: SPY (30%), QQQ (25%), IWM (15%), DIA (10%), VIX (20%)
    weighted_sum = (
        0.30 * dir_vals.get("SPY", 0.0) +
        0.25 * dir_vals.get("QQQ", 0.0) +
        0.15 * dir_vals.get("IWM", 0.0) +
        0.10 * dir_vals.get("DIA", 0.0) +
        0.20 * dir_vals.get("VIX", 0.0)
    )
    
    # Map from [-1.0, 1.0] to [0.0, 100.0]
    alignment_score = (weighted_sum + 1.0) / 2.0 * 100.0
    
    # Determine alignment state
    vix_dir = directions.get("VIX", "Sideways")
    vix_phase = phases.get("VIX", 50.0)
    
    # Volatility Warning if VIX is rising and active
    if vix_dir == "Rising" and 20.0 < vix_phase < 80.0:
        alignment_state = "Volatility Warning"
    elif alignment_score >= 70.0:
        alignment_state = "Bullish Alignment"
    elif alignment_score <= 30.0:
        alignment_state = "Bearish Alignment"
    elif 30.0 < alignment_score < 70.0:
        alignment_state = "Mixed Alignment"
    else:
        alignment_state = "No Edge"
        
    return {
        "SPY": directions.get("SPY", "Sideways"),
        "QQQ": directions.get("QQQ", "Sideways"),
        "IWM": directions.get("IWM", "Sideways"),
        "DIA": directions.get("DIA", "Sideways"),
        "VIX": directions.get("VIX", "Sideways"),
        "alignment_score": round(alignment_score, 1),
        "alignment_state": alignment_state,
        "directions": directions
    }
