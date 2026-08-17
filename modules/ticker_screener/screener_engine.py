import yfinance as yf
import pandas as pd
from datetime import datetime, date
import logging
from modules.tier4.gamma_flip.data_loader import (
    get_available_expirations_tastytrade,
    get_available_expirations,
    load_option_chain_tastytrade,
    load_option_chain,
    tastytrade_configured
)
from modules.tier4.volatility_engine import get_liquidity_score

logger = logging.getLogger(__name__)

def check_volume_and_options(ticker):
    """
    Checks if a ticker matches the liquidity screening criteria:
    - Average daily volume > 5,000,000
    - Options are highly liquid ("GOOD" rating from get_liquidity_score)
    - ATM Option premium is >= $1.50
    """
    try:
        hist = yf.Ticker(ticker).history(period="1mo")
        if hist.empty or len(hist) < 5:
            return None
            
        avg_vol = hist["Volume"].mean()
        if avg_vol < 5_000_000:
            return None
            
        last_price = hist["Close"].iloc[-1]
        
        use_tasty = tastytrade_configured()
        source = "tastytrade" if use_tasty else "yfinance"
        
        try:
            expirations = get_available_expirations_tastytrade(ticker) if use_tasty else get_available_expirations(ticker)
            if not expirations and use_tasty:
                expirations = get_available_expirations(ticker)
                source = "yfinance (fallback)"
        except Exception:
            expirations = get_available_expirations(ticker)
            source = "yfinance (fallback)"

        if not expirations:
            return None
            
        # Get nearest monthly expiration (or just the very next expiration)
        # The prompt didn't specify DTE, so we just use the nearest expiration to check general option liquidity
        nearest_exp = expirations[0]
        
        try:
            if "tastytrade" in source:
                chain_result = load_option_chain_tastytrade(ticker, (nearest_exp,))
            else:
                chain_result = load_option_chain(ticker, (nearest_exp,))
        except Exception:
            try:
                chain_result = load_option_chain(ticker, (nearest_exp,))
                source = "yfinance (fallback)"
            except Exception as e:
                logger.debug(f"Failed to load options for {ticker}: {e}")
                return None
                
        chain = chain_result.chain
        if chain.empty:
            return None
            
        calls = chain[chain["option_type"] == "call"].copy()
        puts = chain[chain["option_type"] == "put"].copy()
        
        lbl, sty, detail = get_liquidity_score(calls, puts, last_price)
        
        if lbl not in ["GOOD", "MODERATE"]:
            return None
            
        calls["dist"] = (calls["strike"] - last_price).abs()
        atm_call = calls.loc[calls["dist"].idxmin()]
        
        atm_call_mid = (float(atm_call.get("bid", 0)) + float(atm_call.get("ask", 0))) / 2
        
        if atm_call_mid < 1.50:
            return None
            
        return {
            "Ticker": ticker,
            "Last Price": round(last_price, 2),
            "Avg Volume (20d)": int(avg_vol),
            "ATM Call Premium": round(atm_call_mid, 2),
            "Liquidity Score": lbl,
            "Option Source": source
        }
    except Exception as e:
        logger.debug(f"Error screening {ticker}: {e}")
        return None
