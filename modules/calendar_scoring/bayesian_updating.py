import logging
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("CalendarBayesianUpdating")

def fetch_intraday_data(ticker_symbol: str) -> dict:
    """
    Fetch intraday 15-minute bars for the current day to calculate
    VWAP, volume velocity, and intraday volatility expansion signals.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Fetch 15-minute data for the last 5 days (to ensure we have a full day's data during off-hours)
        df = ticker.history(period="5d", interval="15m")
        
        if df.empty:
            raise ValueError(f"No intraday data returned for {ticker_symbol}")
            
        # Get the most recent day's data from the dataframe
        last_date = df.index[-1].date()
        df_today = df[df.index.date == last_date].copy()
        
        if len(df_today) < 2:
            # Fallback to the previous trading day if today just started or is empty
            last_date = df.index[-2].date() if len(df.index.unique()) > 1 else last_date
            df_today = df[df.index.date == last_date].copy()
            
        if df_today.empty:
            raise ValueError("No matching intraday subset found")
            
        close_price = float(df_today['Close'].iloc[-1])
        
        # 1. Calculate Intraday VWAP
        typical_price = (df_today['High'] + df_today['Low'] + df_today['Close']) / 3.0
        pv = typical_price * df_today['Volume']
        cum_pv = pv.cumsum()
        cum_vol = df_today['Volume'].cumsum()
        df_today['VWAP'] = cum_pv / (cum_vol + 1e-10)
        current_vwap = float(df_today['VWAP'].iloc[-1])
        above_vwap = close_price > current_vwap
        
        # 2. Volume Velocity (Last bar volume compared to rolling average volume)
        last_volume = float(df_today['Volume'].iloc[-1])
        avg_volume = float(df_today['Volume'].mean())
        high_vol = last_volume > (1.2 * avg_volume) if avg_volume > 0 else False
        
        # 3. Volatility Expansion (latest rolling std of returns vs average of the day)
        df_today['Returns'] = df_today['Close'].pct_change()
        rolling_std = df_today['Returns'].rolling(5).std()
        
        latest_vol = float(rolling_std.iloc[-1]) if not np.isnan(rolling_std.iloc[-1]) else 0.0
        avg_vol = float(rolling_std.mean()) if not np.isnan(rolling_std.mean()) else 0.0
        iv_spike = latest_vol > (1.5 * avg_vol) if avg_vol > 0 else False
        
        return {
            "success": True,
            "ticker": ticker_symbol,
            "last_price": close_price,
            "vwap": current_vwap,
            "above_vwap": above_vwap,
            "last_volume": last_volume,
            "avg_volume": avg_volume,
            "high_vol": high_vol,
            "latest_vol": latest_vol,
            "avg_vol_std": avg_vol,
            "iv_spike": iv_spike
        }
        
    except Exception as e:
        logger.warning(f"Failed to fetch intraday data for {ticker_symbol}: {e}. Returning default neutral signals.")
        return {
            "success": False,
            "ticker": ticker_symbol,
            "last_price": 0.0,
            "vwap": 0.0,
            "above_vwap": True,
            "last_volume": 0.0,
            "avg_volume": 0.0,
            "high_vol": False,
            "latest_vol": 0.0,
            "avg_vol_std": 0.0,
            "iv_spike": False
        }

def calculate_bayesian_posterior(base_score: float, above_vwap: bool, high_vol: bool, iv_spike: bool) -> tuple[float, str]:
    """
    Apply Bayes' Theorem to update the base scoring probability (prior) with intraday signals.
    Returns:
        tuple: (posterior_probability, entry_signal_label)
    """
    # 1. Map Prior Probability (Clamp base score to a logical probability range)
    prior_prob = float(np.clip(base_score / 100.0, 0.1, 0.95))
    prior_fail = 1.0 - prior_prob
    
    # 2. Define Conditional Likelihoods based on historical setups
    # P(Signal | Success)
    L_succ_above_vwap = 0.65
    L_succ_high_vol = 0.60
    L_succ_iv_spike = 0.30 # calendar debit entry is worse during sudden intraday IV spike
    
    # P(Signal | Failure)
    L_fail_above_vwap = 0.45
    L_fail_high_vol = 0.45
    L_fail_iv_spike = 0.55
    
    # 3. Calculate likelihoods for observed signals
    L_succ = 1.0
    L_fail = 1.0
    
    # Above VWAP signal
    if above_vwap:
        L_succ *= L_succ_above_vwap
        L_fail *= L_fail_above_vwap
    else:
        L_succ *= (1.0 - L_succ_above_vwap)
        L_fail *= (1.0 - L_fail_above_vwap)
        
    # High Volume velocity signal
    if high_vol:
        L_succ *= L_succ_high_vol
        L_fail *= L_fail_high_vol
    else:
        L_succ *= (1.0 - L_succ_high_vol)
        L_fail *= (1.0 - L_fail_high_vol)
        
    # IV Spike signal
    if iv_spike:
        L_succ *= L_succ_iv_spike
        L_fail *= L_fail_iv_spike
    else:
        L_succ *= (1.0 - L_succ_iv_spike)
        L_fail *= (1.0 - L_fail_iv_spike)
        
    # 4. Calculate posterior probability of success using Bayes' Theorem
    numerator = prior_prob * L_succ
    denominator = numerator + (prior_fail * L_fail)
    
    if denominator <= 0:
        posterior = prior_prob
    else:
        posterior = float(numerator / denominator)
        
    # Convert to percentage
    posterior_pct = round(posterior * 100.0, 1)
    
    # 5. Classify the Bayesian Intraday Entry Signal
    if posterior_pct >= 80.0:
        signal = "Green Light (VWAP/Volume supportive)"
    elif iv_spike:
        signal = "Hold (IV expansion unfavorable)"
    elif not above_vwap:
        signal = "Avoid (Intraday breakdown)"
    else:
        signal = "Monitor (Neutral intraday)"
        
    return posterior_pct, signal
