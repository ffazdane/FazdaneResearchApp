import logging
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import streamlit as st
from datetime import datetime, timedelta
from modules.tier2.market_regime_db import save_daily_regime, save_regime_history, get_latest_regime

logger = logging.getLogger("MarketRegimeCalculations")

# A diversified representative sample of ~55 tickers for breadth
BREADTH_CONSTITUENTS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "UNH", "LLY",
    "JNJ", "ABBV", "MRK", "PFE", "XOM", "CVX", "COP", "CAT", "DE", "GE",
    "HON", "UPS", "RTX", "BA", "LMT", "WMT", "COST", "HD", "LOW", "MCD",
    "SBUX", "DIS", "XLE", "XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLB",
    "XLU", "XLRE", "XLC"
]

@pd.api.extensions.register_dataframe_accessor("regime_indicators")
class RegimeIndicatorsAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def ema(self, period=20):
        return self._obj['close'].ewm(span=period, adjust=False).mean()

    def sma(self, period=50):
        return self._obj['close'].rolling(period).mean()

    def atr(self, period=14):
        high_low = self._obj['high'] - self._obj['low']
        high_close = (self._obj['high'] - self._obj['close'].shift()).abs()
        low_close = (self._obj['low'] - self._obj['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()

    def ichimoku(self):
        high_9 = self._obj['high'].rolling(9).max()
        low_9 = self._obj['low'].rolling(9).min()
        tenkan = (high_9 + low_9) / 2

        high_26 = self._obj['high'].rolling(26).max()
        low_26 = self._obj['low'].rolling(26).min()
        kijun = (high_26 + low_26) / 2

        senkou_a = ((tenkan + kijun) / 2).shift(26)

        high_52 = self._obj['high'].rolling(52).max()
        low_52 = self._obj['low'].rolling(52).min()
        senkou_b = ((high_52 + low_52) / 2).shift(26)

        return senkou_a, senkou_b

@st.cache_data(ttl=3600, show_spinner=False)
def download_data_cached(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical close, high, low, volume for given tickers."""
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=True
        )
        return data
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return pd.DataFrame()

def extract_ticker_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Safely extract single ticker data from multi-ticker DataFrame."""
    ticker_df = pd.DataFrame(index=df.index)
    if df.empty:
        return ticker_df
    
    if isinstance(df.columns, pd.MultiIndex):
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns.levels[0] and ticker in df[col].columns:
                ticker_df[col.lower()] = df[col][ticker]
    else:
        # Single ticker case
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            col_match = [c for c in df.columns if c.lower() == col.lower()]
            if col_match:
                ticker_df[col.lower()] = df[col_match[0]]
    
    # Fill close if missing or empty
    if 'close' not in ticker_df.columns and not ticker_df.empty:
        ticker_df['close'] = df.iloc[:, 0]
        
    return ticker_df.dropna()

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_market_regime(as_of_date_str: str) -> dict:
    """
    Main function to compute all components, apply overrides, and output the regime result.
    Calculates as of a specific date to prevent forward-looking bias.
    """
    as_of_dt = datetime.strptime(as_of_date_str, "%Y-%m-%d")
    
    # We need about 350 calendar days of data to get 252 trading days for 200 SMA + Ichimoku shift
    start_dt = as_of_dt - timedelta(days=500)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = (as_of_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    
    indices = ["SPY", "QQQ", "IWM", "DIA", "SMH"]
    vol_tickers = ["^VIX", "^VVIX", "^VIX9D", "^VIX3M"]
    sentiment_tickers = ["^CPC", "XLK", "XLF", "XLI", "XLY", "XLU", "XLP", "XLV", "HYG", "LQD"]
    
    all_query_tickers = list(set(indices + vol_tickers + sentiment_tickers + BREADTH_CONSTITUENTS))
    
    logger.info(f"Downloading data for Market Regime Engine as of {as_of_date_str}...")
    raw_data = download_data_cached(all_query_tickers, start_str, end_str)
    
    if raw_data.empty:
        raise ValueError("No market data returned from yfinance. Calculation cancelled.")
        
    # Extract historical indices and indices close
    prices = {}
    for ticker in indices:
        prices[ticker] = extract_ticker_data(raw_data, ticker)
        
    spy_df = prices["SPY"]
    if spy_df.empty or len(spy_df) < 50:
        raise ValueError(f"Insufficient historical data for SPY (needs >= 50 days, got {len(spy_df)}).")
        
    # Filter prices data up to as_of_date_str
    for ticker in indices:
        prices[ticker] = prices[ticker][prices[ticker].index <= as_of_date_str]
    spy_df = prices["SPY"]
    
    # ----------------------------------------------------
    # 1. TREND SCORE (30%)
    # ----------------------------------------------------
    trend_score_val, trend_status_val, trend_details = _calculate_trend_score(prices)
    
    # ----------------------------------------------------
    # 2. BREADTH SCORE (25%)
    # ----------------------------------------------------
    breadth_score_val, breadth_status_val, breadth_details = _calculate_breadth_score(raw_data, as_of_date_str)
    
    # ----------------------------------------------------
    # 3. VOLATILITY SCORE (20%)
    # ----------------------------------------------------
    vix_df = extract_ticker_data(raw_data, "^VIX")
    vix_df = vix_df[vix_df.index <= as_of_date_str]
    
    vvix_df = extract_ticker_data(raw_data, "^VVIX")
    vvix_df = vvix_df[vvix_df.index <= as_of_date_str]
    
    vix9d_df = extract_ticker_data(raw_data, "^VIX9D")
    vix9d_df = vix9d_df[vix9d_df.index <= as_of_date_str]
    
    vix3m_df = extract_ticker_data(raw_data, "^VIX3M")
    vix3m_df = vix3m_df[vix3m_df.index <= as_of_date_str]
    
    vol_score_val, vix_status_val, vol_inversion_val, vol_details = _calculate_volatility_score(
        vix_df, vvix_df, vix9d_df, vix3m_df, spy_df
    )
    
    # ----------------------------------------------------
    # 4. MOMENTUM SCORE (15%)
    # ----------------------------------------------------
    mom_score_val, mom_status_val, mom_details = _calculate_momentum_score(spy_df)
    
    # ----------------------------------------------------
    # 5. RISK SENTIMENT SCORE (10%)
    # ----------------------------------------------------
    qqq_df = prices["QQQ"]
    iwm_df = prices["IWM"]
    smh_df = prices["SMH"]
    
    xlk_df = extract_ticker_data(raw_data, "XLK")
    xlk_df = xlk_df[xlk_df.index <= as_of_date_str]
    
    xlf_df = extract_ticker_data(raw_data, "XLF")
    xlf_df = xlf_df[xlf_df.index <= as_of_date_str]
    
    xli_df = extract_ticker_data(raw_data, "XLI")
    xli_df = xli_df[xli_df.index <= as_of_date_str]
    
    xly_df = extract_ticker_data(raw_data, "XLY")
    xly_df = xly_df[xly_df.index <= as_of_date_str]
    
    xlu_df = extract_ticker_data(raw_data, "XLU")
    xlu_df = xlu_df[xlu_df.index <= as_of_date_str]
    
    xlp_df = extract_ticker_data(raw_data, "XLP")
    xlp_df = xlp_df[xlp_df.index <= as_of_date_str]
    
    xlv_df = extract_ticker_data(raw_data, "XLV")
    xlv_df = xlv_df[xlv_df.index <= as_of_date_str]
    
    hyg_df = extract_ticker_data(raw_data, "HYG")
    hyg_df = hyg_df[hyg_df.index <= as_of_date_str]
    
    lqd_df = extract_ticker_data(raw_data, "LQD")
    lqd_df = lqd_df[lqd_df.index <= as_of_date_str]
    
    cpc_df = extract_ticker_data(raw_data, "^CPC")
    cpc_df = cpc_df[cpc_df.index <= as_of_date_str]
    
    risk_score_val, risk_status_val, risk_details = _calculate_risk_sentiment_score(
        spy_df, qqq_df, iwm_df, smh_df,
        xlk_df, xlf_df, xli_df, xly_df, xlu_df, xlp_df, xlv_df,
        hyg_df, lqd_df, cpc_df
    )
    
    # ----------------------------------------------------
    # REGIME FORMULA
    # ----------------------------------------------------
    raw_final_score = (
        trend_score_val * 0.30 +
        breadth_score_val * 0.25 +
        vol_score_val * 0.20 +
        mom_score_val * 0.15 +
        risk_score_val * 0.10
    )
    
    final_score = round(raw_final_score, 1)
    
    # Initial classification
    regime_name = _classify_score_regime(final_score)
    original_regime = regime_name
    
    # ----------------------------------------------------
    # OVERRIDE RULES
    # ----------------------------------------------------
    override_notes = []
    
    # A. Volatility Override
    vix_val = float(vix_df['close'].iloc[-1]) if not vix_df.empty else 16.0
    if vix_val > 30.0:
        regime_name = "Risk Off / Volatility Shock"
        override_notes.append("Volatility Override: VIX > 30 forces Risk Off / Volatility Shock.")
        
    # B. Moving Average Override
    spy_close = float(spy_df['close'].iloc[-1]) if not spy_df.empty else 0.0
    qqq_close = float(qqq_df['close'].iloc[-1]) if not qqq_df.empty else 0.0
    
    spy_sma50_s = spy_df.regime_indicators.sma(50)
    qqq_sma50_s = qqq_df.regime_indicators.sma(50)
    spy_sma50 = float(spy_sma50_s.iloc[-1]) if len(spy_sma50_s) >= 50 and not pd.isna(spy_sma50_s.iloc[-1]) else spy_close
    qqq_sma50 = float(qqq_sma50_s.iloc[-1]) if len(qqq_sma50_s) >= 50 and not pd.isna(qqq_sma50_s.iloc[-1]) else qqq_close
    
    spy_sma200_s = spy_df.regime_indicators.sma(200)
    qqq_sma200_s = qqq_df.regime_indicators.sma(200)
    spy_sma200 = float(spy_sma200_s.iloc[-1]) if len(spy_sma200_s) >= 200 and not pd.isna(spy_sma200_s.iloc[-1]) else spy_close
    qqq_sma200 = float(qqq_sma200_s.iloc[-1]) if len(qqq_sma200_s) >= 200 and not pd.isna(qqq_sma200_s.iloc[-1]) else qqq_close
    
    if spy_close < spy_sma200 and qqq_close < qqq_sma200:
        if regime_name in ["Strong Buy The Dip", "Buy Dips Selectively", "Range Bound"]:
            regime_name = "Sell The Rip"
            override_notes.append("Moving Average Override: SPY & QQQ below 200 SMA caps regime at Sell The Rip.")
    elif spy_close < spy_sma50 and qqq_close < qqq_sma50:
        if regime_name in ["Strong Buy The Dip", "Buy Dips Selectively"]:
            regime_name = "Range Bound"
            override_notes.append("Moving Average Override: SPY & QQQ below 50 SMA caps regime at Range Bound.")
            
    # C. Breadth Override
    vix_slope = vix_df['close'].iloc[-1] - vix_df['close'].iloc[-5:].mean() if len(vix_df) >= 5 else 0.0
    if breadth_score_val < 35.0 and vix_slope > 0:
        prev_reg = regime_name
        regime_name = _downgrade_regime(regime_name)
        if prev_reg != regime_name:
            override_notes.append(f"Breadth Override: Breadth < 35% and VIX rising downgrades regime from {prev_reg} to {regime_name}.")
            
    # D. Narrow Leadership Override
    spy_ema20_s = spy_df.regime_indicators.ema(20)
    spy_ema20 = float(spy_ema20_s.iloc[-1]) if len(spy_ema20_s) > 0 and not pd.isna(spy_ema20_s.iloc[-1]) else spy_close
    if spy_close > spy_ema20 and breadth_score_val < 40.0:
        if regime_name == "Strong Buy The Dip":
            regime_name = "Buy Dips Selectively"
            override_notes.append("Narrow Leadership Override: SPY > 20 EMA but breadth is weak (< 40%). Capped at Buy Dips Selectively.")
            
    # E. Volatility Term Structure Override
    block_calendars = False
    if vol_inversion_val:
        block_calendars = True
        override_notes.append("Volatility Inversion: Inversion detected (VIX9D > VIX or VIX > VIX3M). Calendar strategies blocked.")

    # Determine Market Bias
    market_bias = {
        "Strong Buy The Dip": "Bullish",
        "Buy Dips Selectively": "Bullish but selective",
        "Range Bound": "Neutral",
        "Sell The Rip": "Bearish",
        "Risk Off / Volatility Shock": "Risk Off"
    }.get(regime_name, "Neutral")

    # Get strategy mapping from DB
    from modules.tier2.market_regime_db import load_strategy_rules
    rules = load_strategy_rules(regime_name)
    
    preferred_strategies = []
    restricted_strategies = []
    blocked_strategies = []
    
    for r in rules:
        strat = r["strategy_name"]
        status = r["strategy_status"]
        reason = r["reason"]
        
        if status == "Preferred":
            preferred_strategies.append(f"{strat}: {reason}")
        elif status == "Avoid":
            restricted_strategies.append(f"{strat}: {reason}")
        elif status == "Blocked":
            blocked_strategies.append(f"{strat}: {reason}")

    # Add custom blocked calendars if term structure inverted
    if block_calendars:
        blocked_strategies.append("Calendar Spreads: Blocked due to volatility term structure inversion.")
        
    # Exposure guidance mapping
    exposure_guidance = _get_exposure_guidance(regime_name)
    
    # Save daily record
    record = {
        "regime_date": as_of_date_str,
        "spy_close": round(spy_close, 2),
        "qqq_close": round(qqq_close, 2),
        "iwm_close": round(float(prices["IWM"]['close'].iloc[-1]), 2) if not prices["IWM"].empty else 0.0,
        "dia_close": round(float(prices["DIA"]['close'].iloc[-1]), 2) if not prices["DIA"].empty else 0.0,
        "smh_close": round(float(prices["SMH"]['close'].iloc[-1]), 2) if not prices["SMH"].empty else 0.0,
        "vix_close": round(vix_val, 2),
        "trend_score": round(trend_score_val, 1),
        "breadth_score": round(breadth_score_val, 1),
        "volatility_score": round(vol_score_val, 1),
        "momentum_score": round(mom_score_val, 1),
        "risk_sentiment_score": round(risk_score_val, 1),
        "final_regime_score": final_score,
        "regime_name": regime_name,
        "confidence_score": 75.0,
        "market_bias": market_bias
    }
    save_daily_regime(record)
    
    # Save History and Trigger Alert if changed
    latest_saved = get_latest_regime()
    prev_regime_name = latest_saved.get("regime_name", "Unknown") if latest_saved else "Unknown"
    
    trigger_reason_str = ", ".join(override_notes) if override_notes else "Calculated score regime shift."
    if prev_regime_name != "Unknown" and prev_regime_name != regime_name:
        save_regime_history(as_of_date_str, prev_regime_name, regime_name, trigger_reason_str)
        
    # Output JSON representation
    return {
        "as_of_date": as_of_date_str,
        "regime_name": regime_name,
        "final_score": final_score,
        "confidence": 78,
        "trend_score": trend_score_val,
        "breadth_score": breadth_score_val,
        "volatility_score": vol_score_val,
        "momentum_score": mom_score_val,
        "risk_sentiment_score": risk_score_val,
        "market_bias": market_bias,
        "preferred_strategies": preferred_strategies,
        "restricted_strategies": restricted_strategies,
        "blocked_strategies": blocked_strategies,
        "risk_notes": override_notes + [
            f"Trend status is {trend_status_val}.",
            f"Breadth status is {breadth_status_val}.",
            f"VIX level is {vix_status_val}.",
            f"Momentum status is {mom_status_val}."
        ],
        "exposure_guidance": exposure_guidance,
        "raw_indices": {
            "SPY": round(spy_close, 2),
            "QQQ": round(qqq_close, 2),
            "IWM": round(float(prices["IWM"]['close'].iloc[-1]), 2) if not prices["IWM"].empty else 0.0,
            "DIA": round(float(prices["DIA"]['close'].iloc[-1]), 2) if not prices["DIA"].empty else 0.0,
            "SMH": round(float(prices["SMH"]['close'].iloc[-1]), 2) if not prices["SMH"].empty else 0.0,
            "VIX": round(vix_val, 2)
        },
        "trend_details": trend_details,
        "volatility_details": vol_details,
        "momentum_details": mom_details,
        "risk_details": risk_details
    }

def _calculate_trend_score(prices: dict) -> tuple[float, str, dict]:
    """Calculates Trend Score (0 to 100) based on 5 indices."""
    total_conditions = 0
    bullish_met = 0
    bearish_met = 0
    
    details = {}
    for ticker, df in prices.items():
        if df.empty or len(df) < 20:
            details[ticker] = {
                "price": 0.0,
                "above_20ema": False, "above_50sma": False, "above_200sma": False,
                "ema20_above_50sma": False, "sma50_above_200sma": False,
                "above_ichimoku": False, "higher_high_low": False,
                "atr_pct": 1.5, "ret_5d": 0.0, "ret_20d": 0.0,
                "trend": "Neutral"
            }
            continue
            
        close = df['close']
        c_price = float(close.iloc[-1])
        
        ema20 = df.regime_indicators.ema(20)
        c_ema20 = float(ema20.iloc[-1]) if len(ema20) > 0 and not pd.isna(ema20.iloc[-1]) else c_price
        
        sma50 = df.regime_indicators.sma(50)
        c_sma50 = float(sma50.iloc[-1]) if len(sma50) >= 50 and not pd.isna(sma50.iloc[-1]) else c_price
        
        sma200 = df.regime_indicators.sma(200)
        c_sma200 = float(sma200.iloc[-1]) if len(sma200) >= 200 and not pd.isna(sma200.iloc[-1]) else c_price
        
        # Ichimoku cloud
        span_a, span_b = df.regime_indicators.ichimoku()
        c_span_a = float(span_a.iloc[-1]) if len(span_a) > 0 and not pd.isna(span_a.iloc[-1]) else c_price
        c_span_b = float(span_b.iloc[-1]) if len(span_b) > 0 and not pd.isna(span_b.iloc[-1]) else c_price
        cloud_max = max(c_span_a, c_span_b)
        cloud_min = min(c_span_a, c_span_b)
        
        # Higher Highs / Higher Lows
        higher_hl = False
        lower_hl = False
        if len(df) >= 40:
            max_curr = float(df['high'].iloc[-20:].max())
            max_prev = float(df['high'].iloc[-40:-20].max())
            min_curr = float(df['low'].iloc[-20:].min())
            min_prev = float(df['low'].iloc[-40:-20].min())
            higher_hl = (max_curr > max_prev) and (min_curr > min_prev)
            lower_hl = (max_curr < max_prev) and (min_curr < min_prev)
            
        # Conditions list
        bull_conds = [
            c_price > c_ema20,
            c_price > c_sma50,
            c_price > c_sma200,
            c_ema20 > c_sma50,
            c_sma50 > c_sma200,
            c_price > cloud_max,
            higher_hl
        ]
        
        bear_conds = [
            c_price < c_ema20,
            c_price < c_sma50,
            c_price < c_sma200,
            c_ema20 < c_sma50,
            c_sma50 < c_sma200,
            c_price < cloud_min,
            lower_hl
        ]
        
        bullish_met += sum(bull_conds)
        bearish_met += sum(bear_conds)
        total_conditions += len(bull_conds)
        
        atr_val = df.regime_indicators.atr(14)
        c_atr = float(atr_val.iloc[-1]) if len(atr_val) > 0 and not pd.isna(atr_val.iloc[-1]) else 0.01 * c_price
        
        ret_5 = float((c_price / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0.0
        ret_20 = float((c_price / close.iloc[-20] - 1) * 100) if len(close) >= 20 else 0.0
        
        details[ticker] = {
            "price": round(c_price, 2),
            "above_20ema": c_price > c_ema20,
            "above_50sma": c_price > c_sma50,
            "above_200sma": c_price > c_sma200,
            "ema20_above_50sma": c_ema20 > c_sma50,
            "sma50_above_200sma": c_sma50 > c_sma200,
            "above_ichimoku": c_price > cloud_max,
            "higher_high_low": higher_hl,
            "atr_pct": round(float(c_atr / c_price * 100), 2) if c_price > 0 else 0.0,
            "ret_5d": round(ret_5, 2),
            "ret_20d": round(ret_20, 2),
            "trend": "Bullish" if sum(bull_conds) >= 4 else "Bearish" if sum(bear_conds) >= 4 else "Neutral"
        }
        
    if total_conditions == 0:
        return 50.0, "Neutral", details
        
    trend_score = 50.0 + (bullish_met - bearish_met) * (50.0 / total_conditions)
    trend_score = max(0.0, min(100.0, trend_score))
    
    if trend_score >= 70.0:
        status = "Bullish"
    elif trend_score <= 30.0:
        status = "Bearish"
    else:
        status = "Neutral"
        
    return round(trend_score, 1), status, details

def _calculate_breadth_score(raw_data: pd.DataFrame, as_of_date_str: str) -> tuple[float, str, dict]:
    """Calculates Breadth Score (0 to 100) using BREADTH_CONSTITUENTS."""
    above_50_count = 0
    new_highs_count = 0
    new_lows_count = 0
    advances = 0
    declines = 0
    valid_symbols = 0
    
    for sym in BREADTH_CONSTITUENTS:
        df = extract_ticker_data(raw_data, sym)
        if df.empty or len(df) < 5:
            continue
            
        # Filter up to as_of_date_str
        df = df[df.index <= as_of_date_str]
        if df.empty:
            continue
            
        close = df['close']
        c_price = float(close.iloc[-1])
        
        valid_symbols += 1
        
        # 1. Above 50 DMA
        sma50 = df.regime_indicators.sma(50)
        c_sma50 = float(sma50.iloc[-1]) if len(sma50) >= 50 and not pd.isna(sma50.iloc[-1]) else c_price
        if c_price > c_sma50:
            above_50_count += 1
            
        # 2. 252-day Highs vs Lows
        lookback_window = min(252, len(df))
        high_252 = float(df['high'].iloc[-lookback_window:].max())
        low_252 = float(df['low'].iloc[-lookback_window:].min())
        if c_price >= high_252 * 0.985:
            new_highs_count += 1
        if c_price <= low_252 * 1.015:
            new_lows_count += 1
            
        # 3. Advance/Decline
        if len(close) >= 2:
            daily_ret = float(close.iloc[-1] / close.iloc[-2] - 1)
            if daily_ret > 0:
                advances += 1
            elif daily_ret < 0:
                declines += 1
            
    if valid_symbols == 0:
        return 50.0, "Mixed", {"pct_above_50sma": 50.0, "net_highs_lows": 0, "ad_ratio": 1.0}
        
    pct_above_50 = (above_50_count / valid_symbols) * 100.0
    
    score = 0.0
    if pct_above_50 > 60.0:
        score += 35.0
    elif pct_above_50 >= 40.0:
        score += 15.0
        
    if new_highs_count > new_lows_count:
        score += 35.0
    elif new_highs_count < new_lows_count:
        score += 0.0
    else:
        score += 15.0
        
    if advances > declines:
        score += 30.0
    elif advances < declines:
        score += 0.0
    else:
        score += 15.0
        
    if score >= 70.0:
        status = "Strong"
    elif score >= 40.0:
        status = "Mixed"
    else:
        status = "Weak"
        
    details = {
        "pct_above_50sma": round(pct_above_50, 1),
        "new_highs": new_highs_count,
        "new_lows": new_lows_count,
        "net_highs_lows": new_highs_count - new_lows_count,
        "advances": advances,
        "declines": declines,
        "ad_ratio": round(advances / max(1, declines), 2)
    }
    return round(score, 1), status, details

def _calculate_volatility_score(vix_df, vvix_df, vix9d_df, vix3m_df, spy_df) -> tuple[float, str, bool, dict]:
    """Calculates Volatility Score (0 to 100)."""
    vix_val = float(vix_df['close'].iloc[-1]) if not vix_df.empty else 16.0
    
    if vix_val < 16.0:
        score = 90.0
        status = "Low"
    elif vix_val <= 22.0:
        score = 70.0
        status = "Normal"
    elif vix_val <= 30.0:
        score = 40.0
        status = "Elevated"
    else:
        score = 10.0
        status = "Shock"
        
    vix_slope = 0.0
    spy_slope = 0.0
    if len(vix_df) >= 5:
        vix_slope = vix_df['close'].iloc[-1] - vix_df['close'].iloc[-5:].mean()
    if len(spy_df) >= 5:
        spy_slope = spy_df['close'].iloc[-1] - spy_df['close'].iloc[-5:].mean()
        
    if vix_slope > 0 and spy_slope > 0:
        score -= 15.0
    elif vix_slope < 0 and spy_slope > 0:
        score += 10.0
        
    vix9d_val = float(vix9d_df['close'].iloc[-1]) if not vix9d_df.empty else vix_val
    vix3m_val = float(vix3m_df['close'].iloc[-1]) if not vix3m_df.empty else vix_val * 1.05
    
    vol_inversion = (vix9d_val > vix_val) or (vix_val > vix3m_val)
    if vol_inversion:
        score -= 20.0
        
    score = max(0.0, min(100.0, score))
    
    vvix_val = float(vvix_df['close'].iloc[-1]) if not vvix_df.empty else 85.0
    details = {
        "vix": round(vix_val, 2),
        "vvix": round(vvix_val, 2),
        "vix9d": round(vix9d_val, 2),
        "vix3m": round(vix3m_val, 2),
        "vix_slope": round(vix_slope, 2),
        "term_structure": "Inverted" if vol_inversion else "Normal"
    }
    return round(score, 1), status, vol_inversion, details

def _calculate_momentum_score(spy_df) -> tuple[float, str, dict]:
    """Calculates Momentum Score (0 to 100)."""
    close = spy_df['close']
    c_price = float(close.iloc[-1])
    
    ret_5 = float((c_price / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0.0
    ret_10 = float((c_price / close.iloc[-10] - 1) * 100) if len(close) >= 10 else 0.0
    ret_20 = float((c_price / close.iloc[-20] - 1) * 100) if len(close) >= 20 else 0.0
    
    score = 50.0
    if len(close) >= 20:
        if ret_5 > 0 and ret_10 > 0 and ret_20 > 0:
            score += 20.0
        elif ret_5 < 0 and ret_10 < 0 and ret_20 < 0:
            score -= 20.0
        
    ema20 = spy_df.regime_indicators.ema(20)
    sma50 = spy_df.regime_indicators.sma(50)
    c_ema20 = float(ema20.iloc[-1]) if len(ema20) > 0 and not pd.isna(ema20.iloc[-1]) else c_price
    c_sma50 = float(sma50.iloc[-1]) if len(sma50) >= 50 and not pd.isna(sma50.iloc[-1]) else c_price
    
    holds_20ema = (c_price > c_ema20) and (c_price <= 1.015 * c_ema20)
    holds_50sma = (c_price > c_sma50) and (c_price <= 1.015 * c_sma50)
    if holds_20ema or holds_50sma:
        score += 15.0
        
    is_extended = (c_price > 1.05 * c_ema20) or (c_price > 1.08 * c_sma50)
    if is_extended:
        score -= 15.0
        
    reversal = False
    if len(spy_df) >= 12:
        yesterday_high = float(spy_df['high'].iloc[-2])
        high_10d_prev = float(spy_df['high'].iloc[-11:-1].max())
        reversal = (c_price < float(close.iloc[-2])) and (yesterday_high >= high_10d_prev)
        if reversal:
            score -= 10.0
        
    atr_14 = spy_df.regime_indicators.atr(14)
    c_atr = float(atr_14.iloc[-1]) if len(atr_14) > 0 and not pd.isna(atr_14.iloc[-1]) else 0.01 * c_price
    atr_sma10 = float(atr_14.rolling(10).mean().iloc[-1]) if len(atr_14) >= 10 and not pd.isna(atr_14.rolling(10).mean().iloc[-1]) else c_atr
    atr_expanding = c_atr > atr_sma10
    
    if atr_expanding:
        if ret_5 < 0:
            score -= 10.0
        else:
            score += 10.0
            
    score = max(0.0, min(100.0, score))
    
    if is_extended:
        status = "Extended"
    elif score >= 75.0:
        status = "Positive"
    elif score >= 50.0:
        status = "Neutral"
    else:
        status = "Negative"
        
    details = {
        "ret_5d": round(ret_5, 2),
        "ret_10d": round(ret_10, 2),
        "ret_20d": round(ret_20, 2),
        "dist_from_20ema_pct": round(float((c_price / c_ema20 - 1) * 100), 2) if c_ema20 > 0 else 0.0,
        "dist_from_50sma_pct": round(float((c_price / c_sma50 - 1) * 100), 2) if c_sma50 > 0 else 0.0,
        "atr_expanding": atr_expanding,
        "is_extended": is_extended
    }
    return round(score, 1), status, details

def _calculate_risk_sentiment_score(
    spy_df, qqq_df, iwm_df, smh_df,
    xlk_df, xlf_df, xli_df, xly_df, xlu_df, xlp_df, xlv_df,
    hyg_df, lqd_df, cpc_df
) -> tuple[float, str, dict]:
    """Calculates Risk Sentiment Score (0 to 100)."""
    score = 50.0
    
    def safe_20d_ret(df, default_val=0.0):
        if df.empty or len(df) < 20:
            return default_val
        return float((df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100)
        
    spy_ret = safe_20d_ret(spy_df)
    qqq_ret = safe_20d_ret(qqq_df)
    iwm_ret = safe_20d_ret(iwm_df)
    smh_ret = safe_20d_ret(smh_df)
    
    if qqq_ret > spy_ret:
        score += 15.0
    else:
        score -= 5.0
        
    if smh_ret > spy_ret:
        score += 15.0
    else:
        score -= 5.0
        
    if iwm_ret > spy_ret:
        score += 15.0
    else:
        score -= 5.0
        
    xlk_r = safe_20d_ret(xlk_df)
    xlf_r = safe_20d_ret(xlf_df)
    xli_r = safe_20d_ret(xli_df)
    xly_r = safe_20d_ret(xly_df)
    xlu_r = safe_20d_ret(xlu_df)
    xlp_r = safe_20d_ret(xlp_df)
    xlv_r = safe_20d_ret(xlv_df)
    
    cyc_ret = np.mean([xlk_r, xlf_r, xli_r, xly_r])
    def_ret = np.mean([xlu_r, xlp_r, xlv_r])
    
    if cyc_ret > def_ret:
        score += 25.0
    else:
        score -= 15.0
        
    cpc_val = 0.85
    cpc_sma10 = 0.85
    if not cpc_df.empty:
        cpc_val = float(cpc_df['close'].iloc[-1])
        if len(cpc_df) >= 10:
            cpc_sma10 = float(cpc_df['close'].rolling(10).mean().iloc[-1])
            
    if cpc_val < cpc_sma10 and cpc_sma10 > 0.80:
        score += 15.0
    elif cpc_val > 1.05 * cpc_sma10:
        score -= 15.0
        
    hyg_ret = safe_20d_ret(hyg_df)
    lqd_ret = safe_20d_ret(lqd_df)
    
    if hyg_ret > lqd_ret:
        score += 15.0
    else:
        score -= 15.0
        
    score = max(0.0, min(100.0, score))
    
    if score >= 65.0:
        status = "Risk On"
    elif score >= 40.0:
        status = "Mixed"
    else:
        status = "Defensive"
        
    details = {
        "qqq_vs_spy_20d": round(qqq_ret - spy_ret, 2),
        "smh_vs_spy_20d": round(smh_ret - spy_ret, 2),
        "iwm_vs_spy_20d": round(iwm_ret - spy_ret, 2),
        "cyclical_vs_defensive_20d": round(cyc_ret - def_ret, 2),
        "put_call_ratio": round(cpc_val, 2),
        "hyg_vs_lqd_20d": round(hyg_ret - lqd_ret, 2)
    }
    return round(score, 1), status, details

def _classify_score_regime(score: float) -> str:
    if score >= 80.0:
        return "Strong Buy The Dip"
    elif score >= 60.0:
        return "Buy Dips Selectively"
    elif score >= 40.0:
        return "Range Bound"
    elif score >= 20.0:
        return "Sell The Rip"
    else:
        return "Risk Off / Volatility Shock"

def _downgrade_regime(regime_name: str) -> str:
    downgrades = {
        "Strong Buy The Dip": "Buy Dips Selectively",
        "Buy Dips Selectively": "Range Bound",
        "Range Bound": "Sell The Rip",
        "Sell The Rip": "Risk Off / Volatility Shock",
        "Risk Off / Volatility Shock": "Risk Off / Volatility Shock"
    }
    return downgrades.get(regime_name, regime_name)

def _get_exposure_guidance(regime_name: str) -> dict:
    guidance = {
        "Strong Buy The Dip": {
            "target_delta": "+70% to +100%",
            "cash": "0% to 15%",
            "hedge": "0% to 10%",
            "preferred_risk": "Bullish trend continuation"
        },
        "Buy Dips Selectively": {
            "target_delta": "+40% to +70%",
            "cash": "15% to 30%",
            "hedge": "5% to 15%",
            "preferred_risk": "Selective bullish exposure"
        },
        "Range Bound": {
            "target_delta": "-10% to +30%",
            "cash": "20% to 40%",
            "hedge": "10% to 20%",
            "preferred_risk": "Neutral theta income"
        },
        "Sell The Rip": {
            "target_delta": "-30% to +10%",
            "cash": "30% to 50%",
            "hedge": "20% to 40%",
            "preferred_risk": "Bearish or neutral exposure"
        },
        "Risk Off / Volatility Shock": {
            "target_delta": "-50% to 0%",
            "cash": "50% or higher",
            "hedge": "30% or higher",
            "preferred_risk": "Capital preservation"
        }
    }
    return guidance.get(regime_name, guidance["Range Bound"])

def calculate_regime_forecasts(as_of_date_str: str, current_regime: str) -> dict:
    """
    Builds the 5x5 regime transition probability matrix from historical records
    and projects the dominant market regime over 5, 10, 20, 40, and 60 trading days.
    """
    regimes_ordered = [
        "Strong Buy The Dip",
        "Buy Dips Selectively",
        "Range Bound",
        "Sell The Rip",
        "Risk Off / Volatility Shock"
    ]
    regime_to_idx = {r: i for i, r in enumerate(regimes_ordered)}
    
    from modules.tier2.market_regime_db import get_historical_regimes
    history = get_historical_regimes(limit=500)
    
    counts = np.zeros((5, 5))
    for i in range(len(history) - 1):
        r_curr = history[i].get("regime_name")
        r_next = history[i+1].get("regime_name")
        if r_curr in regime_to_idx and r_next in regime_to_idx:
            idx_curr = regime_to_idx[r_curr]
            idx_next = regime_to_idx[r_next]
            counts[idx_curr, idx_next] += 1
            
    smoothed_counts = counts + 0.1
    P = smoothed_counts / smoothed_counts.sum(axis=1, keepdims=True)
    
    forecasts = {}
    if current_regime in regime_to_idx:
        curr_idx = regime_to_idx[current_regime]
        v = np.zeros(5)
        v[curr_idx] = 1.0
        
        horizons = [5, 10, 20, 40, 60]
        for H in horizons:
            P_H = np.linalg.matrix_power(P, H)
            v_H = np.dot(v, P_H)
            
            dom_idx = np.argmax(v_H)
            dom_regime = regimes_ordered[dom_idx]
            prob = v_H[dom_idx]
            
            forecasts[f"{H}d"] = {
                "regime_name": dom_regime,
                "probability": round(float(prob), 3),
                "probabilities": {regimes_ordered[i]: round(float(v_H[i]), 3) for i in range(5)}
            }
    else:
        for H in [5, 10, 20, 40, 60]:
            forecasts[f"{H}d"] = {
                "regime_name": current_regime,
                "probability": 1.0,
                "probabilities": {r: (1.0 if r == current_regime else 0.0) for r in regimes_ordered}
            }
            
    return {
        "transition_matrix": P.tolist(),
        "state_labels": regimes_ordered,
        "forecasts": forecasts
    }

def backfill_regime_history(days=252):
    """
    Backfills daily regime calculations for the past year to populate the database.
    Calculates day by day using historical pricing.
    """
    import pandas as pd
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days * 1.5)
    
    spy_data = yf.download("SPY", start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), progress=False)
    trading_dates = [d.strftime("%Y-%m-%d") for d in spy_data.index][-days:]
    
    logger.info(f"Backfilling {len(trading_dates)} historical regime records...")
    completed = 0
    for d_str in trading_dates:
        try:
            calculate_market_regime(d_str)
            completed += 1
        except Exception as e:
            logger.warning(f"Failed to backfill {d_str}: {e}")
            
    logger.info(f"Completed historical backfill for {completed}/{len(trading_dates)} dates.")
