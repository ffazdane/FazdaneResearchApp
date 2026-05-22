"""
FazDane Analytics - Tier 4
Volatility Strategy Engine
Ported from the standalone Volatility Dashboard into the FazDane module system.
"""

import io
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from modules.base_module import FazDaneModule
from utils.universe_manager import format_ticker_display, get_ticker_names, render_universe_manager
from utils.volatility_pdf_generator import generate_pdf_report
INDEX_PROXIES = {"SPX": "SPY", "^GSPC": "SPY", "NDX": "QQQ", "^NDX": "QQQ", "RUT": "IWM", "^RUT": "IWM"}
ASSET_TYPES = {
    "SPX": "Index", "^GSPC": "Index", "NDX": "Index", "^NDX": "Index", "RUT": "Index", "^RUT": "Index",
    "AAPL": "Stock", "MSFT": "Stock", "NVDA": "Stock", "AMZN": "Stock",
    "GOOGL": "Stock", "META": "Stock", "TSLA": "Stock",
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "XLE": "ETF", "XLF": "ETF", "XLK": "ETF", "SMH": "ETF", "TLT": "ETF", "GLD": "ETF",
    "AMD": "Stock", "NFLX": "Stock", "COIN": "Stock", "BA": "Stock", "JPM": "Stock",
}
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TICKER SHORT NAMES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TICKER_NAMES = {
    "SPX":  "S&P 500 Index",
    "NDX":  "Nasdaq-100 Index",
    "RUT":  "Russell 2000 Index",
    "AAPL": "Apple Inc. (AAPL)",
    "MSFT": "Microsoft Corporation (MSFT)",
    "NVDA": "NVIDIA Corporation (NVDA)",
    "AMZN": "Amazon.com Inc. (AMZN)",
    "GOOGL": "Alphabet Inc. (GOOGL)",
    "META": "Meta Platforms Inc. (META)",
    "TSLA": "Tesla Inc. (TSLA)",
    "SPY":  "SPDR S&P 500 ETF Trust (SPY)",
    "QQQ":  "Invesco QQQ Trust (QQQ)",
    "IWM":  "iShares Russell 2000 ETF (IWM)",
    "DIA":  "SPDR Dow Jones Industrial Average ETF (DIA)",
    "XLE":  "Energy Select Sector SPDR Fund (XLE)",
    "XLF":  "Financial Select Sector SPDR Fund (XLF)",
    "XLK":  "Technology Select Sector SPDR Fund (XLK)",
    "SMH":  "VanEck Semiconductor ETF (SMH)",
    "TLT":  "iShares 20+ Year Treasury Bond ETF (TLT)",
    "GLD":  "SPDR Gold Shares ETF (GLD)",
    "AMD":  "Advanced Micro Devices Inc. (AMD)",
    "NFLX": "Netflix Inc. (NFLX)",
    "COIN": "Coinbase Global Inc. (COIN)",
    "BA":   "The Boeing Company (BA)",
    "JPM":  "JPMorgan Chase & Co. (JPM)",
}

DROPDOWN_OPTIONS = {
    "Indices": ["SPX", "NDX", "RUT"],
    "Mag 7": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"],
    "Premium Selling Favorites": ["SPY", "QQQ", "IWM", "DIA", "XLE", "XLF", "XLK", "SMH", "TLT", "GLD"],
    "High IV / Active Options": ["AMD", "NFLX", "COIN", "BA", "JPM"],
}
FLAT_OPTIONS = []
for _g, _ts in DROPDOWN_OPTIONS.items():
    FLAT_OPTIONS.append(_g)
    for _t in _ts:
        FLAT_OPTIONS.append(_t)
QUICK_BUTTONS = ["SPY", "QQQ", "TSLA", "AAPL", "NVDA", "GLD"]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SESSION STATE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HELPER: BADGE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def make_badge(text, style="gray"):
    colors = {
        "green":  ("rgba(50,200,100,0.18)",  "#52D68A"),
        "yellow": ("rgba(255,210,50,0.18)",   "#FFD700"),
        "red":    ("rgba(220,50,50,0.18)",    "#FF6B6B"),
        "orange": ("rgba(255,150,50,0.18)",   "#FFB347"),
        "blue":   ("rgba(0,173,181,0.18)",    "#00ADB5"),
        "gray":   ("rgba(180,180,180,0.12)",  "#9B9B9B"),
    }
    bg, fg = colors.get(style, colors["gray"])
    return (f'<span style="background:{bg};color:{fg};padding:3px 12px;border-radius:20px;'
            f'font-size:0.75rem;font-weight:700;letter-spacing:.5px;border:1px solid {fg}55;">'
            f'{text}</span>')

def panel_intro(ticker, heading, description):
    """Render a compact, self-contained intro box above each panel section."""
    return (
        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:8px;padding:7px 14px;margin-bottom:4px;width:fit-content;max-width:90%;">'
        f'<p style="font-size:0.76rem;color:#8B9CB6;margin:0;line-height:1.5;">'
        f'<span style="color:#CDD5E0;font-weight:600;">{ticker}</span>'
        f'<span style="color:rgba(0,173,181,0.8);"> - {heading}</span>'
        f' - {description}</p>'
        f'</div>'
    )

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DATA FUNCTIONS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_data(show_spinner=False)
def get_price_data(ticker, start, end):
    try:
        data = yf.Ticker(ticker).history(start=start, end=end)
        return data if (data is not None and not data.empty) else None
    except:
        return None

@st.cache_data(ttl=86400, show_spinner=False)
def get_company_name(ticker):
    """Fetch the official company/ETF short name from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ""
        if name:
            # Format: "Avis Budget Group, Inc. (CAR)"
            return f"{name} ({ticker})"
        return ""
    except:
        return ""

@st.cache_data(show_spinner=False)
def get_vix_data(start, end):
    try:
        d = yf.Ticker("^VIX").history(start=start, end=end)["Close"]
        return d if (d is not None and not d.empty) else None
    except:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_vvix():
    try:
        d = yf.Ticker("^VVIX").history(period="5d")["Close"]
        return float(d.iloc[-1]) if len(d) > 0 else None
    except:
        return None

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CALCULATION FUNCTIONS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def calculate_hv(close_series, window=20):
    return close_series.pct_change().rolling(window=window).std() * np.sqrt(252) * 100

def calculate_hv_rank(hv_series, window=252):
    hi  = hv_series.rolling(window=window, min_periods=1).max()
    lo  = hv_series.rolling(window=window, min_periods=1).min()
    den = (hi - lo).replace(0, np.nan)
    return ((hv_series - lo) / den) * 100

def calculate_expected_move(price, iv_pct, dte):
    return price * (iv_pct / 100) * np.sqrt(dte / 365)

def classify_regime(hvr):
    if hvr >= 80:   return "EXTREME", "red"
    elif hvr >= 60: return "HIGH",    "orange"
    elif hvr >= 30: return "NORMAL",  "yellow"
    else:           return "LOW",     "green"

def get_trend_label(df, fast=20, slow=50):
    if len(df) < slow:
        return "INSUFFICIENT DATA", "gray"
    sma_f = df["Close"].rolling(fast).mean().iloc[-1]
    sma_s = df["Close"].rolling(slow).mean().iloc[-1]
    p = df["Close"].iloc[-1]
    if p > sma_f > sma_s:   return "UPTREND",     "green"
    elif p < sma_f < sma_s: return "DOWNTREND",   "red"
    else:                   return "RANGE-BOUND",  "yellow"

def get_vix_percentile(vix_series):
    if vix_series is None or len(vix_series) < 5:
        return None
    return round((vix_series < vix_series.iloc[-1]).mean() * 100, 1)

@st.cache_data(ttl=3600, show_spinner=False)
def get_options_chain(ticker, dte_target=30):
    try:
        stock = yf.Ticker(ticker)
        exps  = stock.options
        if not exps:
            return None, None, None, None, None
        today = date.today()
        best  = min(exps, key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d").date() - today).days - dte_target))
        a_dte = (datetime.strptime(best, "%Y-%m-%d").date() - today).days
        chain = stock.option_chain(best)
        calls, puts = chain.calls.copy(), chain.puts.copy()
        price = stock.fast_info["lastPrice"]
        valid = calls[calls["impliedVolatility"] > 0.01]
        if valid.empty:
            return calls, puts, None, best, a_dte
        idx    = (valid["strike"] - price).abs().idxmin()
        atm_iv = float(valid.loc[idx, "impliedVolatility"]) * 100
        return calls, puts, atm_iv, best, a_dte
    except:
        return None, None, None, None, None

@st.cache_data(ttl=3600, show_spinner=False)
def get_term_structure(ticker):
    try:
        stock = yf.Ticker(ticker)
        exps  = stock.options
        if not exps:
            return []
        price = stock.fast_info["lastPrice"]
        today = date.today()
        out   = []
        for exp in exps[:12]:
            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
            if dte < 3 or dte > 130:
                continue
            try:
                calls = stock.option_chain(exp).calls
                valid = calls[calls["impliedVolatility"] > 0.01]
                if valid.empty:
                    continue
                idx = (valid["strike"] - price).abs().idxmin()
                iv  = float(valid.loc[idx, "impliedVolatility"]) * 100
                out.append({"dte": dte, "iv": iv, "expiry": exp})
            except:
                continue
        return sorted(out, key=lambda x: x["dte"])
    except:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_skew_data(ticker, dte_target=30):
    try:
        stock = yf.Ticker(ticker)
        exps  = stock.options
        if not exps:
            return None, None, None
        today = date.today()
        best  = min(exps, key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d").date() - today).days - dte_target))
        chain = stock.option_chain(best)
        price = stock.fast_info["lastPrice"]
        calls = chain.calls[chain.calls["impliedVolatility"] > 0.01].copy()
        puts  = chain.puts[chain.puts["impliedVolatility"]  > 0.01].copy()
        if calls.empty:
            return None, None, None
        idx    = (calls["strike"] - price).abs().idxmin()
        atm_iv = float(calls.loc[idx, "impliedVolatility"]) * 100
        otm_put_iv  = float(puts.loc[(puts["strike"] - price * 0.95).abs().idxmin(), "impliedVolatility"]) * 100 if not puts.empty else None
        otm_call_iv = float(calls.loc[(calls["strike"] - price * 1.05).abs().idxmin(), "impliedVolatility"]) * 100
        return otm_put_iv, atm_iv, otm_call_iv
    except:
        return None, None, None

def get_liquidity_score(calls, puts, price):
    try:
        if calls is None or calls.empty:
            return "N/A", "gray", {}
        valid = calls[calls["impliedVolatility"] > 0.01].copy()
        if valid.empty:
            return "N/A", "gray", {}
        row  = valid.loc[(valid["strike"] - price).abs().idxmin()]
        bid, ask = float(row.get("bid", 0)), float(row.get("ask", 0))
        mid  = (bid + ask) / 2 if (bid + ask) > 0 else 1
        sprd = round((ask - bid) / mid * 100, 1) if mid > 0 else 99
        oi   = int(row.get("openInterest", 0) or 0)
        vol  = int(row.get("volume", 0) or 0)
        pts  = (2 if sprd < 5 else 1 if sprd < 10 else 0) + (2 if oi > 1000 else 1 if oi > 200 else 0) + (2 if vol > 500 else 1 if vol > 100 else 0)
        lbl  = "GOOD" if pts >= 5 else "MODERATE" if pts >= 3 else "POOR"
        sty  = "green"  if pts >= 5 else "yellow"  if pts >= 3 else "red"
        return lbl, sty, {"Bid-Ask Spread": f"{sprd}%", "Open Interest": f"{oi:,}", "Volume": f"{vol:,}"}
    except:
        return "N/A", "gray", {}

@st.cache_data(ttl=3600, show_spinner=False)
def get_earnings_date(ticker):
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            return None
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed and len(ed) > 0:
                return ed[0].date() if hasattr(ed[0], "date") else ed[0]
        elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.columns:
            return cal["Earnings Date"].iloc[0]
        return None
    except:
        return None

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STRATEGY ENGINE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def strategy_engine(hvr, atm_iv, hv20, trend_label, skew_label,
                    term_shape, vix_pct, days_to_earnings, liquidity_label, iv_hv_diff):
    warnings_list = []

    if days_to_earnings is not None and days_to_earnings < 14:
        warnings_list.append(f"Earnings in **{days_to_earnings} days** - elevated event risk.")
    if liquidity_label == "POOR":
        warnings_list.append("Poor liquidity - wide bid-ask spreads may erode edge.")
    if vix_pct is not None and vix_pct > 80:
        warnings_list.append("VIX in panic zone - prefer defined-risk spreads only.")

    if hvr is None or atm_iv is None:
        return {"strategy": "INSUFFICIENT DATA", "confidence": "N/A", "dte_rec": "-",
                "strike_note": "-", "reason": "Key data unavailable.", "warnings": warnings_list, "badge_style": "gray"}

    diff = iv_hv_diff if iv_hv_diff else 0

    if hvr < 15:
        return {"strategy": "AVOID SELLING OPTIONS", "confidence": "High", "dte_rec": "-",
                "strike_note": "Wait for volatility expansion",
                "reason": f"HV Rank is extremely low ({hvr:.1f}). Premium is historically cheap. Symmetrical risk is skewed against sellers.",
                "warnings": warnings_list, "badge_style": "red"}

    if hvr >= 50 and diff > 1 and term_shape == "Contango" and trend_label == "RANGE-BOUND":
        return {"strategy": "SELL IRON CONDOR", "confidence": "High" if hvr >= 75 else "Medium",
                "dte_rec": "30-45 days",
                "strike_note": "Place short strikes at +/-1 Expected Move (~1 std dev)",
                "reason": f"High HV Rank ({hvr:.1f}) + IV>HV + Contango + Range-bound. Ideal iron condor setup.",
                "warnings": warnings_list, "badge_style": "green"}

    if hvr >= 40 and trend_label == "RANGE-BOUND" and term_shape == "Contango":
        return {"strategy": "SELL STRANGLE / CONDOR", "confidence": "Medium",
                "dte_rec": "30-45 days",
                "strike_note": "Place short strikes at +/-1 SD (expected move)",
                "reason": f"Elevated HV Rank ({hvr:.1f}) + range-bound + contango. Strangle or Condor captures premium on both sides.",
                "warnings": warnings_list, "badge_style": "green"}

    if hvr >= 30 and trend_label == "UPTREND" and skew_label in ["Put Skew High", "Flat Skew"]:
        return {"strategy": "SELL BULL PUT SPREAD", "confidence": "High" if hvr >= 50 else "Medium",
                "dte_rec": "21-35 days",
                "strike_note": "Sell put at -1 SD, buy put 1-2 strikes lower",
                "reason": f"Uptrend + decent HV Rank ({hvr:.1f}) + favorable skew. Selling downside premium aligns with directional bias.",
                "warnings": warnings_list, "badge_style": "green"}

    if hvr >= 30 and trend_label == "DOWNTREND" and skew_label in ["Call Skew High", "Flat Skew"]:
        return {"strategy": "SELL BEAR CALL SPREAD", "confidence": "Medium",
                "dte_rec": "21-35 days",
                "strike_note": "Sell call at +1 SD, buy call 1-2 strikes higher",
                "reason": f"Downtrend + decent HV Rank ({hvr:.1f}). Selling upside call spread aligns with directional bias.",
                "warnings": warnings_list, "badge_style": "yellow"}

    if hvr >= 70 and term_shape == "Backwardation":
        return {"strategy": "SELL NEAR-TERM DEFINED RISK", "confidence": "Medium",
                "dte_rec": "7-21 days",
                "strike_note": "Use spreads - avoid naked short premium in backwardation",
                "reason": f"Backwardation + Extreme HV Rank ({hvr:.1f}). Near-term IV spike. Sell spreads that expire quickly.",
                "warnings": warnings_list, "badge_style": "yellow"}

    if hvr >= 25:
        return {"strategy": "SELL CREDIT SPREAD (Directional)", "confidence": "Low",
                "dte_rec": "30-45 days",
                "strike_note": "Direction + skew analysis required to choose put vs call",
                "reason": f"Moderate HV Rank ({hvr:.1f}). Some opportunity exists - use directional bias to choose side.",
                "warnings": warnings_list, "badge_style": "yellow"}

    return {"strategy": "HOLD / WAIT", "confidence": "Medium", "dte_rec": "-",
            "strike_note": "-",
            "reason": f"HV Rank is very low ({hvr:.1f}). Premium is not elevated enough to justify forcing a trade.",
            "warnings": warnings_list, "badge_style": "red"}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TICKER RESOLUTION FUNCTION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def get_selected_ticker(dropdown_val, custom_input, use_proxy):
    raw = custom_input.strip().upper() if custom_input.strip() else str(dropdown_val).strip().upper()
    if raw in DROPDOWN_OPTIONS or raw.startswith("──"):
        raw = "SPY"
    display, note = raw, ""
    if use_proxy and raw in INDEX_PROXIES:
        return display, INDEX_PROXIES[raw], f"Using {INDEX_PROXIES[raw]} as ETF proxy for {raw}"
    if raw == "SPX": return display, "^GSPC", "Using ^GSPC (S&P 500 Index)"
    if raw == "NDX": return display, "^NDX",  "Using ^NDX (Nasdaq-100)"
    if raw == "RUT": return display, "^RUT",  "Using ^RUT (Russell 2000)"
    return display, raw, ""

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SIDEBAR
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def render_volatility_css():
    st.markdown("""
    <style>
    .ve-panel-card, .panel-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 18px 22px; margin-bottom: 16px; }
    .panel-title { font-size: 0.72rem; font-weight: 700; color: #3ab54a; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 14px; border-bottom: 1px solid rgba(58,181,74,0.18); padding-bottom: 6px; }
    .sidebar-section { font-size: 0.68rem; font-weight: 700; color: #3ab54a !important; text-transform: uppercase; letter-spacing: 1.5px; margin: 18px 0 4px 0; border-bottom: 1px solid rgba(58,181,74,0.2); padding-bottom: 4px; }
    div[data-testid="stMetric"] { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; padding: 12px 14px; }
    div[data-testid="stMetricValue"] { color: #3ab54a; font-size: 1.2rem !important; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.72rem !important; font-weight: 600; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

class VolatilityEngineModule(FazDaneModule):
    MODULE_NAME = "Volatility Strategy Engine"
    MODULE_ICON = "Vol"
    MODULE_DESCRIPTION = "Premium-selling volatility regime, IV/HV, term structure, skew, and strategy dashboard."
    TIER = 4
    SOURCE_NOTEBOOK = "Volatility Dashboard"
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def run(self):
        with st.sidebar:
            self.render_sidebar()
        self.render_main()

    def render_sidebar(self):
        st.markdown("**Volatility Strategy Engine**")
        st.caption("Premium selling decision platform")
        st.divider()
        if "ve_custom_ticker" not in st.session_state:
            st.session_state.ve_custom_ticker = ""

        st.markdown('<p class="sidebar-section">Instrument Selection</p>', unsafe_allow_html=True)
        universe_name, tickers_list, _ = render_universe_manager(
            key_prefix="ve_universe",
            show_benchmark=False,
            label="Ticker Universe:",
        )
        ticker_names = get_ticker_names(universe_name)
        tradeable = tickers_list or ["SPY"]
        previous = st.session_state.get("ve_selected_ticker", "SPY")
        index = tradeable.index(previous) if previous in tradeable else 0
        st.selectbox(
            "Instrument:",
            options=tradeable,
            index=index,
            key="ve_selected_ticker",
            format_func=lambda ticker: format_ticker_display(ticker, ticker_names),
        )
        custom_input = st.text_input(
            "Or Enter Custom Ticker:",
            value=st.session_state.ve_custom_ticker,
            placeholder="e.g. SHOP, MSTR...",
            key="ve_custom_input_field",
        )
        st.session_state.ve_custom_ticker = custom_input
        st.session_state["ve_use_proxy"] = st.toggle(
            "Use ETF Proxy for Indices",
            value=st.session_state.get("ve_use_proxy", True),
            help="SPX -> SPY, NDX -> QQQ, RUT -> IWM",
            key="ve_use_proxy_toggle",
        )

        st.divider()
        st.markdown('<p class="sidebar-section">Date Range</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.session_state["ve_start_date"] = st.date_input(
                "From", value=st.session_state.get("ve_start_date", datetime.today() - timedelta(days=365)), key="ve_start_date_input"
            )
        with c2:
            st.session_state["ve_end_date"] = st.date_input(
                "To", value=st.session_state.get("ve_end_date", datetime.today()), key="ve_end_date_input"
            )

        st.divider()
        st.markdown('<p class="sidebar-section">Parameters</p>', unsafe_allow_html=True)
        st.session_state["ve_hv_window"] = st.slider("HV Window (Days)", 5, 60, st.session_state.get("ve_hv_window", 20), key="ve_hv_window_slider")
        st.session_state["ve_dte_target"] = st.slider("IV Target DTE (Days)", 7, 90, st.session_state.get("ve_dte_target", 30), key="ve_dte_target_slider")

        st.divider()
        st.markdown('<p class="sidebar-section">Event Risk Override</p>', unsafe_allow_html=True)
        st.session_state["ve_macro_event"] = st.checkbox(
            "Flag Active Macro Event (Fed, CPI, etc.)", value=st.session_state.get("ve_macro_event", False), key="ve_macro_event_checkbox"
        )

    def render_main(self):
        render_volatility_css()
        selected_dte = st.session_state.get("ve_dte_target", 30)
        raw_dd = st.session_state.get("ve_selected_ticker", "SPY") or "SPY"
        if raw_dd in DROPDOWN_OPTIONS or str(raw_dd).startswith("──"):
            raw_dd = "SPY"
        display_ticker, data_ticker, proxy_note = get_selected_ticker(
            raw_dd,
            st.session_state.ve_custom_ticker,
            st.session_state.get("ve_use_proxy", True),
        )
        asset_type  = ASSET_TYPES.get(display_ticker, "Stock")
        type_icon   = {"Index": "Index", "ETF": "ETF", "Stock": "Stock"}.get(asset_type, "Asset")

        # HEADER
        st.title("FazDane Analytics: Volatility Engine")
        st.markdown("*Professional options premium selling decision platform*")
        b1, b2 = st.columns([1, 6])
        _ticker_name = TICKER_NAMES.get(display_ticker, "")
        if not _ticker_name:
            _ticker_name = get_company_name(data_ticker)
        _name_suffix  = f" - {_ticker_name}" if _ticker_name else ""
        with b1: st.markdown(f"### {display_ticker}")
        with b2: st.info(f"**{type_icon}** | {proxy_note if proxy_note else f'Ticker: `{data_ticker}`'}{_name_suffix}")

        st.markdown("---")

        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # LOAD ALL DATA
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with st.spinner(f"Loading price data for {data_ticker}..."):
            df = get_price_data(data_ticker, st.session_state.get("ve_start_date"), st.session_state.get("ve_end_date"))

        if df is None or df.empty:
            st.warning(f"No price data found for **{data_ticker}**. Check ticker or date range.")
            st.stop()

        # Price calculations
        current_price = float(df["Close"].iloc[-1])
        hv20_s  = calculate_hv(df["Close"], 20)
        hv30_s  = calculate_hv(df["Close"], 30)
        hvs     = calculate_hv(df["Close"], st.session_state.get("ve_hv_window", 20))
        hvr_s   = calculate_hv_rank(hvs)
        hv20    = float(hv20_s.iloc[-1])
        hv30    = float(hv30_s.iloc[-1])
        hvr     = float(hvr_s.iloc[-1]) if not pd.isna(hvr_s.iloc[-1]) else 0.0
        sma20   = df["Close"].rolling(20).mean()
        sma50   = df["Close"].rolling(50).mean()
        regime_label, regime_style = classify_regime(hvr)
        trend_label,  trend_style  = get_trend_label(df)

        # Options + market data (with individual spinners)
        with st.spinner("Fetching options chain..."):
            calls, puts, atm_iv, best_expiry, actual_dte = get_options_chain(data_ticker, selected_dte)

        with st.spinner("Building term structure..."):
            term_structure = get_term_structure(data_ticker)

        with st.spinner("Fetching skew data..."):
            otm_put_iv, atm_iv_skew, otm_call_iv = get_skew_data(data_ticker, selected_dte)

        with st.spinner("Fetching VIX/VVIX..."):
            vix_data = get_vix_data(st.session_state.get("ve_start_date"), st.session_state.get("ve_end_date"))
            vvix_val = get_vvix()

        earnings_date    = get_earnings_date(data_ticker)
        today_d          = date.today()
        days_to_earnings = (earnings_date - today_d).days if earnings_date else None

        liq_label, liq_style, liq_detail = (get_liquidity_score(calls, puts, current_price)
                                             if calls is not None and not calls.empty
                                             else ("N/A", "gray", {}))

        vix_current  = float(vix_data.iloc[-1]) if vix_data is not None else None
        vix_pct      = get_vix_percentile(vix_data)

        iv_hv_diff   = (atm_iv - hv20) if atm_iv else None

        if iv_hv_diff is not None:
            if iv_hv_diff > 5:    premium_label, premium_style = "PREMIUM RICH",  "green"
            elif iv_hv_diff > -3: premium_label, premium_style = "FAIR VALUE",    "yellow"
            else:                 premium_label, premium_style = "PREMIUM CHEAP", "red"
        else:
            premium_label, premium_style = "N/A", "gray"

        if len(term_structure) >= 2:
            term_shape = "Contango" if term_structure[-1]["iv"] > term_structure[0]["iv"] else "Backwardation"
            term_style = "green" if term_shape == "Contango" else "red"
        else:
            term_shape, term_style = "N/A", "gray"

        if otm_put_iv and atm_iv:
            if otm_put_iv / atm_iv > 1.15:                        skew_label = "Put Skew High"
            elif otm_call_iv and otm_call_iv / atm_iv > 1.15:     skew_label = "Call Skew High"
            else:                                                  skew_label = "Flat Skew"
        else:
            skew_label = "N/A"

        exp_move    = calculate_expected_move(current_price, atm_iv if atm_iv else hv20, selected_dte)
        upper_range = current_price + exp_move
        lower_range = current_price - exp_move

        result = strategy_engine(
            hvr=hvr, atm_iv=atm_iv, hv20=hv20, trend_label=trend_label,
            skew_label=skew_label, term_shape=term_shape, vix_pct=vix_pct,
            days_to_earnings=days_to_earnings, liquidity_label=liq_label,
            iv_hv_diff=iv_hv_diff if iv_hv_diff else 0
        )
        if st.session_state.get("ve_macro_event", False):
            result["warnings"].append("Manual macro event flagged (Fed/CPI/etc.) - reduce size or avoid.")

        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # TABS
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        tab1, tab2, tab3, tab4 = st.tabs(["Snapshot", "Volatility Structure", "Strategy Engine", "User Guide"])
        fig = None
        fig_ts = None # Pre-initialize for PDF export

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # TAB 1: SNAPSHOT
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        with tab1:
            m = st.columns(6)
            m[0].metric("Last Price",               f"${current_price:,.2f}")
            m[1].metric(f"ATM IV (~{selected_dte}DTE)", f"{atm_iv:.1f}%" if atm_iv else "N/A")
            m[2].metric("20D Historical Vol",       f"{hv20:.1f}%")
            m[3].metric("30D Historical Vol",       f"{hv30:.1f}%")
            m[4].metric(f"HV Rank",                 f"{hvr:.1f}")
            m[5].metric(f"Expected Move ({selected_dte}D)", f"${exp_move:.2f}")

            st.markdown(
                f'<div style="margin:12px 0 20px 0;">'
                f'{make_badge(f"REGIME: {regime_label}", regime_style)}&nbsp;&nbsp;'
                f'{make_badge(trend_label, trend_style)}&nbsp;&nbsp;'
                f'{make_badge(f"VIX {vix_current:.1f}" if vix_current else "VIX N/A", "blue")}'
                f'</div>', unsafe_allow_html=True
            )

            st.markdown("---")
            ch1, ch2 = st.columns([3, 2])

            with ch1:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                    name="Price",
                    increasing_line_color="#52D68A", decreasing_line_color="#FF6B6B",
                    increasing_fillcolor="rgba(82,214,138,0.7)", decreasing_fillcolor="rgba(255,107,107,0.7)"
                ))
                fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA 20",
                    line=dict(color="#00ADB5", width=1.5)))
                fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA 50",
                    line=dict(color="#F8B195", width=1.5, dash="dash")))
                fig.add_hline(y=upper_range, line_dash="dot", line_color="rgba(82,214,138,0.7)",
                              annotation_text=f"+EM ${upper_range:.0f}", annotation_position="right")
                fig.add_hline(y=lower_range, line_dash="dot", line_color="rgba(255,107,107,0.7)",
                              annotation_text=f"-EM ${lower_range:.0f}", annotation_position="right")
                fig.add_hline(y=current_price, line_dash="solid", line_color="rgba(255,255,255,0.2)")
                fig.update_layout(
                    title=f"{display_ticker} - Candlestick with SMAs & Expected Move",
                    xaxis_rangeslider_visible=False,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, color="#8B9CB6"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#8B9CB6"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CDD5E0")),
                    margin=dict(l=0, r=80, t=40, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Expected move bands: **${lower_range:.2f}** to **${upper_range:.2f}** over {selected_dte} days "
                           f"using {'ATM IV' if atm_iv else 'HV20'} {(atm_iv or hv20):.1f}%")

            with ch2:
                fig2 = go.Figure()
                if atm_iv:
                    fig2.add_hline(y=atm_iv, line_dash="dot", line_color="#00ADB5",
                                   annotation_text=f"ATM IV {atm_iv:.1f}%", annotation_position="right")
                fig2.add_trace(go.Scatter(x=df.index, y=hv20_s, name="20D HV",
                    line=dict(color="#F8B195", width=2)))
                fig2.add_trace(go.Scatter(x=df.index, y=hv30_s, name="30D HV",
                    line=dict(color="#B195F8", width=1.5, dash="dash")))
                fig2.update_layout(
                    title="IV vs Historical Volatility",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, color="#8B9CB6"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#8B9CB6", title="Vol (%)"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CDD5E0")),
                    margin=dict(l=0, r=80, t=40, b=0)
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown("**For selling options:**")
                if iv_hv_diff is not None:
                    if iv_hv_diff > 5:    st.success(f"ATM IV is **{iv_hv_diff:.1f}%** above 20D HV - **Premium Rich**. Favorable for selling.")
                    elif iv_hv_diff > -3: st.warning(f"ATM IV is near 20D HV (diff: {iv_hv_diff:+.1f}%) - **Fair Value**. Be selective.")
                    else:                 st.error(f"ATM IV is {abs(iv_hv_diff):.1f}% below HV - **Premium Cheap**. Avoid selling.")
                else:
                    st.info("Options data unavailable for comparison.")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # TAB 2: VOLATILITY STRUCTURE
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        with tab2:
            row1a, row1b = st.columns(2)

            with row1a:
                st.markdown(panel_intro(display_ticker, "IV vs HV Spread", "Compares at-the-money implied volatility against 20-day realized historical volatility. A positive spread indicates options are overpriced relative to actual price movement - the primary edge for premium sellers."), unsafe_allow_html=True)
                st.markdown('<p class="panel-title">IV vs HV Spread</p>', unsafe_allow_html=True)
                sc = st.columns(3)
                sc[0].metric("ATM IV",  f"{atm_iv:.1f}%"      if atm_iv      else "N/A")
                sc[1].metric("20D HV",  f"{hv20:.1f}%")
                sc[2].metric("IV - HV", f"{iv_hv_diff:+.1f}%" if iv_hv_diff  else "N/A")
                st.markdown(f"<br>{make_badge(premium_label, premium_style)}", unsafe_allow_html=True)
                st.markdown("**For selling options:**")
                if premium_style == "green":  st.success("IV elevated vs realized vol. Ideal for premium selling.")
                elif premium_style == "yellow": st.warning("Fair value. Moderate opportunity - be selective.")
                else: st.error("Premium cheap vs realized vol. Selling offers poor edge.")
                st.markdown('</div>', unsafe_allow_html=True)

            with row1b:
                st.markdown(panel_intro(display_ticker, "Market Volatility Risk", "Monitors the CBOE VIX (fear gauge) and VVIX (volatility-of-volatility). Elevated VIX percentile signals broad market stress, increasing tail risk for short premium positions."), unsafe_allow_html=True)
                st.markdown('<p class="panel-title">Volatility Risk - VIX / VVIX</p>', unsafe_allow_html=True)
                vc = st.columns(3)
                vc[0].metric("VIX",             f"{vix_current:.2f}" if vix_current else "N/A")
                vc[1].metric("VIX 52W Pct",    f"{vix_pct:.0f}%"    if vix_pct    else "N/A")
                vc[2].metric("VVIX",            f"{vvix_val:.1f}"    if vvix_val   else "N/A")
                if vix_pct is not None:
                    vr, vs = ("PANIC ZONE", "red") if vix_pct > 80 else ("RISING RISK", "orange") if vix_pct > 55 else ("STABLE", "green")
                    st.markdown(f"<br>{make_badge(vr, vs)}", unsafe_allow_html=True)
                st.markdown("**For selling options:**")
                if vix_pct is not None:
                    if vix_pct > 80:   st.error("VIX panic zone. Use defined-risk strategies only.")
                    elif vix_pct > 55: st.warning("VIX rising. Caution with undefined-risk positions.")
                    else:              st.success("VIX stable. Favorable backdrop for premium selling.")
                else:
                    st.info("VIX data unavailable.")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")

            # Term Structure
            st.markdown(panel_intro(display_ticker, "IV Term Structure", "Plots implied volatility across all available expiration dates. A normal contango curve (rising IV over time) supports time-decay selling; backwardation signals near-term stress and warrants reduced position sizing."), unsafe_allow_html=True)
            st.markdown('<p class="panel-title">IV Term Structure</p>', unsafe_allow_html=True)
            if term_structure:
                ts_df  = pd.DataFrame(term_structure)
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(
                    x=ts_df["dte"], y=ts_df["iv"], mode="lines+markers", name="IV",
                    line=dict(color="#00ADB5", width=2.5), marker=dict(size=8, color="#00ADB5")
                ))
                fig_ts.update_layout(
                    xaxis=dict(title="Days to Expiration (DTE)", showgrid=False, color="#8B9CB6"),
                    yaxis=dict(title="Implied Volatility (%)", showgrid=True,
                               gridcolor="rgba(255,255,255,0.05)", color="#8B9CB6"),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CDD5E0")),
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_ts, use_container_width=True)
                st.markdown(f"**Shape:** {make_badge(term_shape, term_style)}", unsafe_allow_html=True)
                st.markdown("**For selling options:**")
                if term_shape == "Contango": st.success("Contango: Normal. Near-term options decay faster - favorable for time-decay sellers.")
                elif term_shape == "Backwardation": st.warning("Backwardation: Stress signal. Near-term IV elevated - use caution with short premium.")
            else:
                st.info("Term structure unavailable for this ticker.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")
            skew_col, liq_col = st.columns(2)

            with skew_col:
                st.markdown(panel_intro(display_ticker, "Volatility Skew", "Compares OTM put IV, ATM IV, and OTM call IV to detect directional demand imbalances. Elevated put skew means the market is pricing downside protection at a premium, which informs which side of the market offers better edge."), unsafe_allow_html=True)
                st.markdown('<p class="panel-title">Skew Analysis</p>', unsafe_allow_html=True)
                sk = st.columns(3)
                sk[0].metric("OTM Put IV (-5%)", f"{otm_put_iv:.1f}%"  if otm_put_iv  else "N/A")
                sk[1].metric("ATM IV",           f"{atm_iv_skew:.1f}%" if atm_iv_skew else "N/A")
                sk[2].metric("OTM Call IV (+5%)",f"{otm_call_iv:.1f}%" if otm_call_iv else "N/A")
                if skew_label != "N/A":
                    sk_s = "red" if "Put" in skew_label else ("orange" if "Call" in skew_label else "green")
                    st.markdown(f"<br>{make_badge(skew_label, sk_s)}", unsafe_allow_html=True)
                st.markdown("**For selling options:**")
                if skew_label == "Put Skew High":   st.warning("Puts expensive - market pricing downside risk. Favor put spreads with caution.")
                elif skew_label == "Call Skew High": st.success("Call premium elevated - sell call spreads.")
                elif skew_label == "Flat Skew":     st.success("Flat skew - neutral. Iron condors and strangles well-positioned.")
                else: st.info("Skew data unavailable.")
                st.markdown('</div>', unsafe_allow_html=True)

            with liq_col:
                st.markdown(panel_intro(display_ticker, "Options Liquidity", "Evaluates execution quality based on ATM bid-ask spread tightness, open interest depth, and daily volume. Poor liquidity erodes theoretical edge through slippage and wide fills."), unsafe_allow_html=True)
                st.markdown('<p class="panel-title">Liquidity Score (ATM Options)</p>', unsafe_allow_html=True)
                st.markdown(f"<br>{make_badge(liq_label, liq_style)}<br><br>", unsafe_allow_html=True)
                if liq_detail:
                    for k, v in liq_detail.items():
                        st.markdown(f"**{k}:** `{v}`")
                st.markdown("**For selling options:**")
                if liq_label == "GOOD":     st.success("Tight spreads and high volume. Easy to enter/exit positions efficiently.")
                elif liq_label == "MODERATE": st.warning("Moderate liquidity. Use limit orders to avoid slippage.")
                elif liq_label == "POOR":   st.error("Poor liquidity. Wide bid-ask spreads will significantly erode edge.")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")
            # Event Risk
            st.markdown(panel_intro(display_ticker, "Event Risk Calendar", "Identifies upcoming earnings dates and active macro events (Fed decisions, CPI, etc.). Implied volatility typically inflates ahead of known events and collapses after - timing entries around event risk is critical for premium sellers."), unsafe_allow_html=True)
            st.markdown('<p class="panel-title">Event Risk</p>', unsafe_allow_html=True)
            ev1, ev2 = st.columns(2)
            with ev1:
                if earnings_date:
                    dte_earn = (earnings_date - today_d).days
                    earn_style = "red" if dte_earn < 14 else "yellow" if dte_earn < 30 else "green"
                    earn_label = "HIGH" if dte_earn < 14 else "MODERATE" if dte_earn < 30 else "LOW"
                    st.metric("Next Earnings", str(earnings_date))
                    st.markdown(f"{make_badge(f'EARNINGS RISK: {earn_label} ({dte_earn}d away)', earn_style)}", unsafe_allow_html=True)
                else:
                    st.metric("Next Earnings", "N/A")
            with ev2:
                macro_style = "red" if st.session_state.get("ve_macro_event", False) else "green"
                macro_text  = "MACRO EVENT ACTIVE" if st.session_state.get("ve_macro_event", False) else "NO MACRO OVERRIDE"
                st.markdown(f"<br>{make_badge(macro_text, macro_style)}", unsafe_allow_html=True)
            st.markdown("**For selling options:**")
            if earnings_date and (earnings_date - today_d).days < 14:
                st.error("Earnings within 14 days. IV typically spikes then collapses post-earnings. Do not sell premium into earnings unless it is your specific strategy.")
            else:
                st.success("No near-term earnings risk detected. Clear to evaluate selling strategies.")
            st.markdown('</div>', unsafe_allow_html=True)

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # TAB 3: STRATEGY ENGINE
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        with tab3:
            # Directional Filter
            st.markdown(panel_intro(display_ticker, "Directional Trend Filter", "Classifies the current price trend using the 20-day and 50-day simple moving averages. Trend alignment is used to select the appropriate directional side for credit spreads and to assess suitability for non-directional strategies."), unsafe_allow_html=True)
            st.markdown('<p class="panel-title">Directional Filter</p>', unsafe_allow_html=True)
            df1, df2, df3, df4 = st.columns(4)
            df1.metric("Current Price", f"${current_price:,.2f}")
            df2.metric("SMA 20",        f"${sma20.iloc[-1]:,.2f}" if not pd.isna(sma20.iloc[-1]) else "N/A")
            df3.metric("SMA 50",        f"${sma50.iloc[-1]:,.2f}" if not pd.isna(sma50.iloc[-1]) else "N/A")
            df4.metric("Trend",         trend_label)
            st.markdown(f"<br>{make_badge(trend_label, trend_style)}", unsafe_allow_html=True)
            st.markdown("**For selling options:**")
            if trend_label == "RANGE-BOUND":   st.success("Range-bound: Ideal for non-directional strategies (Iron Condor, Strangle).")
            elif trend_label == "UPTREND":     st.info("Uptrend: Favor selling put spreads (Bull Put Spread). Avoid selling covered calls aggressively.")
            elif trend_label == "DOWNTREND":   st.warning("Downtrend: Favor selling call spreads (Bear Call Spread). Avoid undefined downside risk.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")

            # Full Decision Table
            st.markdown(panel_intro(display_ticker, "Comprehensive Metrics Breakdown", "A transparent log of every volatility, options, and market structure input used by the Strategy Engine - including current values and their interpretation. Use this table to understand the full rationale behind any strategy recommendation."), unsafe_allow_html=True)
            st.markdown('<p class="panel-title">Full Metrics Decision Table</p>', unsafe_allow_html=True)
            table_rows = [
                ("Last Price",          f"${current_price:,.2f}",                      "Current market price"),
                ("ATM Implied Vol",     f"{atm_iv:.1f}%" if atm_iv else "N/A",        "Market's forward vol expectation"),
                ("20D Historical Vol",  f"{hv20:.1f}%",                                "Recent realized price volatility"),
                ("30D Historical Vol",  f"{hv30:.1f}%",                                "Longer-term realized volatility"),
                ("HV Rank (HVR)",       f"{hvr:.1f} / 100",                            "Where current vol sits vs 52W range"),
                ("Volatility Regime",   regime_label,                                  "LOW < 30 | NORMAL 30-60 | HIGH 60-80 | EXTREME 80+"),
                ("IV vs HV Spread",     f"{iv_hv_diff:+.1f}%" if iv_hv_diff else "N/A", premium_label),
                ("Term Structure",      term_shape,                                    "Contango=normal, Backwardation=stress"),
                ("Market Trend",        trend_label,                                   "Based on 20/50 SMA relationship"),
                ("Skew",                skew_label,                                    "OTM put vs ATM vs OTM call IVs"),
                ("VIX",                 f"{vix_current:.2f}" if vix_current else "N/A", "CBOE Volatility Index"),
                ("VIX 52W Percentile",  f"{vix_pct:.0f}%" if vix_pct else "N/A",      ">80%=Panic | 55-80%=Rising | <55%=Stable"),
                ("Liquidity",           liq_label,                                     "ATM options bid-ask, OI, volume"),
                ("Expected Move",       f"${exp_move:.2f} (+/-{exp_move/current_price*100:.1f}%)", f"${lower_range:.2f} - ${upper_range:.2f}"),
                ("Options Expiry Used", best_expiry if best_expiry else "N/A",        f"~{actual_dte}DTE" if actual_dte else ""),
                ("Earnings Date", f"{days_to_earnings} Days" if days_to_earnings is not None else "N/A", "Upcoming earnings risk"),
            ]
            table_html = """
            <table style="width:100%;border-collapse:collapse;">
            <thead><tr>
              <th style="background:rgba(0,173,181,0.12);color:#00ADB5;font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;padding:10px 14px;text-align:left;width:22%">Metric</th>
              <th style="background:rgba(0,173,181,0.12);color:#00ADB5;font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;padding:10px 14px;text-align:left;width:20%">Value</th>
              <th style="background:rgba(0,173,181,0.12);color:#00ADB5;font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;padding:10px 14px;text-align:left;">Interpretation</th>
            </tr></thead><tbody>"""
            for i, (metric, value, interp) in enumerate(table_rows):
                bg = "rgba(255,255,255,0.02)" if i % 2 == 0 else "transparent"
                table_html += f'<tr style="background:{bg};"><td style="color:#CDD5E0;font-size:.85rem;padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.04);font-weight:500;">{metric}</td><td style="color:#00ADB5;font-size:.85rem;padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.04);font-weight:700;">{value}</td><td style="color:#8B9CB6;font-size:.82rem;padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.04);">{interp}</td></tr>'
            table_html += "</tbody></table>"
            st.markdown(table_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")

            # Final Decision Box
            st.markdown(panel_intro(display_ticker, "Algorithmic Strategy Output", "Synthesizes HV Rank, IV premium, term structure shape, price trend, skew, liquidity, and event risk into a single actionable recommendation. Includes strategy type, confidence level, target DTE, and strike placement guidance."), unsafe_allow_html=True)
            st.markdown('<p class="panel-title">Strategy Decision Engine</p>', unsafe_allow_html=True)

            conf_color = {"High": "#52D68A", "Medium": "#FFD700", "Low": "#FF6B6B", "N/A": "#9B9B9B"}.get(result["confidence"], "#9B9B9B")
            box_style  = {"green": "rgba(82,214,138,0.08)", "yellow": "rgba(255,215,0,0.08)", "red": "rgba(255,107,107,0.08)", "gray": "rgba(180,180,180,0.05)"}.get(result["badge_style"], "rgba(0,0,0,0)")
            border_c   = {"green": "#52D68A55", "yellow": "#FFD70055", "red": "#FF6B6B55", "gray": "#9B9B9B33"}.get(result["badge_style"], "#33333355")

            st.markdown(f"""
            <div style="background:{box_style};border:1px solid {border_c};border-radius:16px;padding:30px 36px;text-align:center;margin:8px 0 20px 0;">
                <div style="font-size:1.7rem;font-weight:800;color:#E0E6F0;margin-bottom:6px;">{result['strategy']}</div>
                <div style="font-size:0.9rem;color:{conf_color};font-weight:600;margin-bottom:18px;">Confidence: {result['confidence']}</div>
                <div style="display:flex;justify-content:center;gap:40px;flex-wrap:wrap;margin-bottom:20px;">
                    <div><div style="font-size:.7rem;color:#8B9CB6;text-transform:uppercase;letter-spacing:.5px;">Suggested DTE</div><div style="font-size:1rem;color:#CDD5E0;font-weight:600;">{result['dte_rec']}</div></div>
                    <div><div style="font-size:.7rem;color:#8B9CB6;text-transform:uppercase;letter-spacing:.5px;">Strike Guidance</div><div style="font-size:1rem;color:#CDD5E0;font-weight:600;">{result['strike_note']}</div></div>
                </div>
                <div style="font-size:.85rem;color:#8B9CB6;max-width:600px;margin:0 auto;line-height:1.6;">{result['reason']}</div>
            </div>
            """, unsafe_allow_html=True)

            if result["warnings"]:
                for w in result["warnings"]:
                    st.warning(w)

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")

            # Excel Export
            st.markdown(panel_intro(display_ticker, "Data Export", "Downloads the full analysis as a structured Excel workbook containing the volatility summary, price history, and strategy output."), unsafe_allow_html=True)
            st.markdown('<p class="panel-title">Export to Excel</p>', unsafe_allow_html=True)

            _excel_bytes = None
            _excel_error = None
            try:
                summary_data = {
                    "Metric": [r[0] for r in table_rows],
                    "Value":  [r[1] for r in table_rows],
                    "Interpretation": [r[2] for r in table_rows],
                }
                summary_df = pd.DataFrame(summary_data)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    summary_df.to_excel(writer, sheet_name="Volatility Summary", index=False)
                    price_hist = df.tail(252).reset_index()
                    if "Date" in price_hist.columns and pd.api.types.is_datetime64_any_dtype(price_hist["Date"]):
                        price_hist["Date"] = price_hist["Date"].dt.tz_localize(None)
                    price_hist.to_excel(writer, sheet_name="Price History", index=False)
                    strategy_df = pd.DataFrame([{
                        "Strategy": result["strategy"],
                        "Confidence": result["confidence"],
                        "Suggested DTE": result["dte_rec"],
                        "Strike Note": result["strike_note"],
                        "Reasoning": result["reason"],
                        "Warnings": " | ".join(result["warnings"]) if result["warnings"] else "None"
                    }])
                    strategy_df.to_excel(writer, sheet_name="Strategy Output", index=False)
                _excel_bytes = output.getvalue()
            except Exception as e:
                _excel_error = str(e)

            if _excel_error:
                st.error(f"Error generating Excel: {_excel_error}")

            # Always render unconditionally â€” conditional rendering causes Streamlit UUID filename bug
            st.download_button(
                label="Download Full Analysis (.xlsx)",
                data=_excel_bytes if _excel_bytes else b"",
                file_name=f"FazDane_{display_ticker}_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                disabled=(_excel_bytes is None)
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")

            # PDF Export
            st.markdown(panel_intro(display_ticker, "1-Pager PDF Summary", "Generates a fully branded, professionally formatted PDF report answering the critical question: 'Can I sell options on this ticker today?' - perfect for sending to clients or saving to your risk journal."), unsafe_allow_html=True)
            st.markdown('<p class="panel-title">Download 1-Pager Report</p>', unsafe_allow_html=True)

            gen_col, dl_col = st.columns([1, 4])
            with gen_col:
                if st.button("Generate PDF Report", use_container_width=True, key="generate_pdf_btn"):
                    with st.spinner("Compiling PDF and rendering graphs..."):
                        try:
                            pdf_bytes = generate_pdf_report(
                                display_ticker, _ticker_name, current_price,
                                result, fig, fig_ts, table_rows
                            )
                            st.session_state.ve_pdf_bytes = pdf_bytes
                            st.session_state.ve_pdf_ticker = display_ticker
                        except Exception as e:
                            st.error(f"Failed to generate PDF: {e}")

            # Determine if we have a valid PDF ready for this ticker
            _pdf_ready = (
                "ve_pdf_bytes" in st.session_state
                and st.session_state.get("ve_pdf_ticker") == display_ticker
                and st.session_state.ve_pdf_bytes
            )
            _pdf_data = st.session_state.ve_pdf_bytes if _pdf_ready else b""
            _pdf_name = f"FazDane_Report_{display_ticker}_{datetime.now().strftime('%Y-%m-%d')}.pdf"

            with dl_col:
                # Always render unconditionally â€” conditional rendering causes Streamlit UUID filename bug
                st.download_button(
                    label="Download 1-Pager PDF" if _pdf_ready else "Generate PDF first",
                    data=_pdf_data,
                    file_name=_pdf_name,
                    mime="application/pdf",
                    type="primary",
                    disabled=(not _pdf_ready)
                )

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # TAB 4: USER GUIDE
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        with tab4:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown('<p class="panel-title">Volatility Engine User Guide</p>', unsafe_allow_html=True)

            st.markdown("""
            Welcome to the **FazDane Analytics Volatility Engine**. This platform is designed to answer a single critical question for options premium sellers: **"Is the current market environment favorable for selling options, and if so, what strategy should I use?"**

            Below is a breakdown of every component, how to read it, and why it is important.

            ---

            ### Left Panel: Parameters & Inputs
            - **Ticker Selection**: Choose from indices, Mag 7 stocks, or high-liquidity premium-selling favorites. You can also input any custom ticker. Indices are proxy-mapped to their highly liquid ETF equivalents (e.g., SPX mapped to SPY since SPX free options data is sparse).
            - **HV Window (Days)**: The rolling window used to calculate Historical Volatility (default 20 days).
            - **IV Target DTE**: Days To Expiration. The engine searches the live options chain for expirations closest to this target (default 30 days).
            - **Event Risk Override**: Check this if there's a looming macro event (like a Fed decision or CPI release) to force the Strategy Engine to output high-risk warnings, regardless of how safe the data looks.

            ---

            ### Tab 1: Snapshot
            The Snapshot gives you the immediate pulse of the asset.
            - **HV Rank (HVR)**: *[CRITICAL]* Since free data sources do not provide true historical Implied Volatility arrays, we use Historical Volatility Rank. This measures where the current 20-day volatility sits relative to its 52-week high and low. 
              - *How to read:* `> 50` means volatility is historically elevated (good for sellers). `< 25` means volatility is completely crushed (bad for sellers).
            - **Expected Move**: Mathematically derived from the Current Price, ATM IV, and DTE. Shows the 1 Standard Deviation expected range by expiration.
            - **Regime & Trend Badges**: Shows if volatility is `LOW, NORMAL, HIGH, EXTREME` and whether the stock's 20-day/50-day moving averages indicate an `UPTREND, DOWNTREND, or RANGE-BOUND` market.

            ---

            ### Tab 2: Volatility Structure
            This tab digs into the specific underlying options data to look for mispricings.
            - **IV vs HV Spread**: Compares the live ATM Implied Volatility against actual realized Historical Volatility. 
              - *Why it's important:* If IV > HV, the market is overestimating risk. This is known as "Premium Rich" and is the exact edge option sellers look for.
            - **VIX / VVIX Risk**: Tracks the broad market volatility index. If VIX is above its 80th percentile, it is flagged as a "Panic Zone" where undefined risk (like naked puts/calls) should be avoided.
            - **IV Term Structure**: Plots the IV across different expiration dates (7D, 14D, 30D, 60D). 
              - *How to read:* **Contango** (upward sloping) is normal. **Backwardation** (downward sloping) means near-term panic is spiking.
            - **Skew Analysis**: Compares OTM Put IV vs ATM IV vs OTM Call IV. 
              - *Why it's important:* A high Put Skew means downside protection is expensive (favoring put sellers). A flat skew means Iron Condors are well-priced.
            - **Liquidity Score**: Grades the Bid-Ask spread, Open Interest, and Volume. Poor liquidity will erase edge via slippage.

            ---

            ### Tab 3: Strategy Decision Engine
            The brain of the platform. It aggregates all data to provide an algorithmic recommendation.
            - **How it suggests:**
              1. **Is Volatility High Enough?** It first checks the `HV Rank`. If it's below 15, it forces an `AVOID` or `HOLD`. Selling options when premium is dirt-cheap is a losing long-term game.
              2. **Is there Directional Bias?** It reads the `Trend Label`. If in an `UPTREND`, it will favor selling `Bull Put Spreads`.
              3. **Is the Market Choppy?** If the asset is `RANGE-BOUND`, IV > HV, and we are in Contango, it triggers the most profitable premium strategy: `SELL IRON CONDOR` or `SELL STRANGLE`.
            - **Full Decision Table**: A transparent log of every metric and the engine's internal interpretation of it.

            ---

            ***"Trade the math, not the emotion. FazDane Analytics is designed to quantify your edge."***
            """)
            st.markdown('</div>', unsafe_allow_html=True)

