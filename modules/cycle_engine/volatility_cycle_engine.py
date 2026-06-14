import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

def fetch_vix_metrics(as_of_date: date) -> dict:
    """Fetch indices for VIX, VVIX, VIX9D, and VIX3M."""
    tickers = {
        "VIX": "^VIX",
        "VVIX": "^VVIX",
        "VIX9D": "^VIX9D",
        "VIX3M": "^VIX3M"
    }
    
    start_date = as_of_date - timedelta(days=365)
    out = {}
    try:
        data = yf.download(list(tickers.values()), start=start_date, end=as_of_date + timedelta(days=1), auto_adjust=True, progress=False)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            
            close_data = data["Close"]
            for key, ticker in tickers.items():
                if ticker in close_data.columns:
                    out[key] = close_data[ticker].dropna()
    except Exception:
        pass
    return out

def analyze_volatility(ticker: str, prices: pd.Series, as_of_date: date) -> dict:
    """
    Evaluates volatility indicators (VIX, VVIX, term structure, realized vol)
    and computes the volatility cycle status and calendar suitability.
    """
    vix_data = fetch_vix_metrics(as_of_date)
    
    # Defaults
    vix_val = 15.0
    vvix_val = 90.0
    vix9d_val = 14.5
    vix3m_val = 16.0
    vix_pct = 50.0
    vix_z = 0.0
    
    vix_series = vix_data.get("VIX")
    if vix_series is not None and not vix_series.empty:
        vix_val = float(vix_series.iloc[-1])
        # Percentile over past year
        vix_pct = float((vix_series < vix_val).mean() * 100.0)
        # Z-score
        mean_vix = vix_series.mean()
        std_vix = vix_series.std()
        vix_z = float((vix_val - mean_vix) / (std_vix + 1e-6))
        
    vvix_series = vix_data.get("VVIX")
    if vvix_series is not None and not vvix_series.empty:
        vvix_val = float(vvix_series.iloc[-1])
        
    vix9d_series = vix_data.get("VIX9D")
    if vix9d_series is not None and not vix9d_series.empty:
        vix9d_val = float(vix9d_series.iloc[-1])
        
    vix3m_series = vix_data.get("VIX3M")
    if vix3m_series is not None and not vix3m_series.empty:
        vix3m_val = float(vix3m_series.iloc[-1])

    # 1. Realized Volatility of the asset (rolling 20-day returns standard deviation)
    realized_vol = 20.0
    if len(prices) >= 21:
        returns = prices.pct_change().dropna()
        realized_vol = float(returns.rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100.0)

    # 2. Term Structure Class
    # normal: VIX9D < VIX < VIX3M (Contango)
    # inverted: VIX9D > VIX > VIX3M (Backwardation)
    if vix9d_val < vix_val < vix3m_val:
        term_structure = "contango"
    elif vix9d_val > vix_val > vix3m_val:
        term_structure = "backwardation"
    else:
        term_structure = "flat"

    # 3. VVIX warning (elevated tail risk)
    vvix_warning = vvix_val > 110.0

    # 4. Volatility Status
    if vix_val >= 25.0 and vix_z < 0.0:
        vol_status = "Vol Mean Reversion"
    elif vix_val < 12.0 and abs(vix_z) < 0.5:
        vol_status = "Vol Compressed"
    elif vix_z > 1.5 or vvix_val > 115.0:
        vol_status = "Vol Spike Risk"
    elif vix_z > 0.5:
        vol_status = "Vol Expanding"
    elif vix_z < -0.5:
        vol_status = "Vol Contracting"
    else:
        vol_status = "Vol Unstable"

    # 5. Calendar suitability scoring
    # Calendar spreads benefit from Contango and stable/low vol (long vega benefits from rising vol,
    # but short front-month leg requires stability to decay theta).
    suitability = 100.0
    
    if term_structure == "backwardation":
        suitability -= 30.0
    elif term_structure == "flat":
        suitability -= 10.0
        
    if vol_status in ["Vol Spike Risk", "Vol Unstable"]:
        suitability -= 40.0
    elif vol_status == "Vol Expanding":
        suitability -= 15.0
    elif vol_status == "Vol Contracting":
        suitability += 10.0 # Good for calendars (short-month decays, long-month stable)
        
    if vix_pct > 75.0:
        # High VIX has contraction risk, which cuts Vega value of the calendar
        suitability -= 20.0
        
    if vvix_warning:
        suitability -= 25.0

    calendar_suitability = max(min(suitability, 100.0), 0.0)

    return {
        "volatility_cycle_status": vol_status,
        "vix_percentile": round(vix_pct, 1),
        "vvix_warning": vvix_warning,
        "term_structure": term_structure,
        "calendar_suitability": round(calendar_suitability, 1),
        "vix_value": round(vix_val, 2),
        "vvix_value": round(vvix_val, 2),
        "realized_volatility": round(realized_vol, 1)
    }
