import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from scipy.stats import norm

logger = logging.getLogger("CalendarDataLoader")

# ══════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    
    # DM filters
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    # True Range
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Smooth wilder style via rolling sum proxy
    tr_sum = tr.rolling(window=period).sum()
    plus_dm_sum = pd.Series(plus_dm, index=df.index).rolling(window=period).sum()
    minus_dm_sum = pd.Series(minus_dm, index=df.index).rolling(window=period).sum()
    
    plus_di = 100 * (plus_dm_sum / (tr_sum + 1e-8))
    minus_di = 100 * (minus_dm_sum / (tr_sum + 1e-8))
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    adx = dx.rolling(window=period).mean()
    return adx

# ══════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES GREEKS ENGINE
# ══════════════════════════════════════════════════════════════════════

def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes Call Price and Greeks."""
    if T <= 0:
        return max(0.0, S - K), 1.0, 0.0, 0.0, 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    # Theta calendar day (divide by 365)
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    vega = (S * np.sqrt(T) * norm.pdf(d1)) / 100.0 # standardized vega for 1% change
    
    return price, delta, gamma, theta, vega

# ══════════════════════════════════════════════════════════════════════
# CORE DATA LOADER
# ══════════════════════════════════════════════════════════════════════

def fetch_technical_data(ticker_symbol: str) -> dict:
    """Fetch 1 year of price history from yfinance and calculate indicators."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="1y")
        if df.empty:
            raise ValueError(f"No price history found for {ticker_symbol}")
            
        close_series = df['Close']
        ema_20 = close_series.ewm(span=20, adjust=False).mean()
        ema_50 = close_series.ewm(span=50, adjust=False).mean()
        ema_200 = close_series.ewm(span=200, adjust=False).mean()
        
        rsi = calculate_rsi(df)
        atr = calculate_atr(df)
        adx = calculate_adx(df)
        
        spot = close_series.iloc[-1]
        
        # Calculate Implied Volatility parameters (fallback placeholder or actual info if present)
        # We will estimate a baseline historical volatility as proxy if IV not found
        hist_vol_30 = df['Close'].pct_change().rolling(30).std().iloc[-1] * np.sqrt(252)
        if np.isnan(hist_vol_30):
            hist_vol_30 = 0.30
            
        return {
            "spot_price": float(spot),
            "ema_20": float(ema_20.iloc[-1]),
            "ema_50": float(ema_50.iloc[-1]),
            "ema_200": float(ema_200.iloc[-1]),
            "rsi_14": float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0,
            "atr_14": float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else spot * 0.02,
            "adx_14": float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 20.0,
            "hv_30": float(hist_vol_30),
            "df_history": df
        }
    except Exception as e:
        logger.error(f"Error fetching technical data for {ticker_symbol}: {e}")
        # Default mock technical data if fetch fails
        mock_spot = 150.0
        return {
            "spot_price": mock_spot,
            "ema_20": mock_spot * 0.98,
            "ema_50": mock_spot * 0.95,
            "ema_200": mock_spot * 0.90,
            "rsi_14": 55.0,
            "atr_14": mock_spot * 0.025,
            "adx_14": 22.0,
            "hv_30": 0.32,
            "df_history": pd.DataFrame()
        }

def fetch_option_chain_data(ticker_symbol: str, spot_price: float, use_synthetic: bool = False) -> dict:
    """Fetch option chains. Generates synthetic option chains as fallback/testing mode."""
    if use_synthetic:
        return generate_synthetic_chain(ticker_symbol, spot_price)
        
    try:
        ticker = yf.Ticker(ticker_symbol)
        expirations = ticker.options
        if not expirations or len(expirations) < 2:
            raise ValueError(f"No option expirations available for {ticker_symbol}")
            
        # Target Short Leg ~20 DTE, Long Leg ~40 DTE
        today = datetime.now()
        parsed_expirations = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            dte = (exp_date - today).days
            parsed_expirations.append((exp, dte))
            
        # Find closest DTEs
        short_candidates = sorted(parsed_expirations, key=lambda x: abs(x[1] - 20))
        long_candidates = sorted(parsed_expirations, key=lambda x: abs(x[1] - 40))
        
        short_exp, short_dte = short_candidates[0]
        # Ensure long expiration is further out than short expiration
        long_candidates = [x for x in long_candidates if x[1] > short_dte]
        if not long_candidates:
            long_exp, long_dte = short_exp, short_dte + 20
        else:
            long_exp, long_dte = long_candidates[0]
            
        # Fetch actual chains
        short_chain = ticker.option_chain(short_exp)
        long_chain = ticker.option_chain(long_exp)
        
        # Format columns to be uniform
        short_calls = short_chain.calls
        long_calls = long_chain.calls
        
        if short_calls.empty or long_calls.empty:
            raise ValueError(f"Empty option chain received for {ticker_symbol}")
            
        return {
            "short_expiry": short_exp,
            "short_dte": short_dte,
            "short_calls": short_calls,
            "long_expiry": long_exp,
            "long_dte": long_dte,
            "long_calls": long_calls,
            "is_synthetic": False
        }
    except Exception as e:
        logger.warning(f"Failed to fetch real option chain for {ticker_symbol}: {e}. Falling back to synthetic option chain.")
        return generate_synthetic_chain(ticker_symbol, spot_price)

def generate_synthetic_chain(ticker_symbol: str, spot_price: float) -> dict:
    """Generate a clean, mathematically correct synthetic option chain using Black-Scholes."""
    today = datetime.today()
    short_dte = 20
    long_dte = 40
    short_exp = (today + timedelta(days=short_dte)).strftime("%Y-%m-%d")
    long_exp = (today + timedelta(days=long_dte)).strftime("%Y-%m-%d")
    
    # Establish baseline implied vol
    base_iv = 0.28
    term_structure_premium = 0.02 # Long month slightly higher IV
    
    # Create strike ranges around spot
    strikes = []
    # Strike interval based on spot size
    if spot_price < 25:
        interval = 0.5
    elif spot_price < 100:
        interval = 1.0
    elif spot_price < 250:
        interval = 2.5
    elif spot_price < 500:
        interval = 5.0
    else:
        interval = 10.0
        
    start_strike = int((spot_price * 0.8) / interval) * interval
    end_strike = int((spot_price * 1.2) / interval) * interval
    strikes = np.arange(start_strike, end_strike + interval, interval)
    
    # Risk-free rate
    r = 0.045 # 4.5%
    
    short_rows = []
    long_rows = []
    
    for strike in strikes:
        # Front leg option pricing
        T_short = short_dte / 365.0
        price_short, delta_short, gamma_short, theta_short, vega_short = black_scholes_call(spot_price, strike, T_short, r, base_iv)
        
        # Back leg option pricing (contango IV curve)
        T_long = long_dte / 365.0
        price_long, delta_long, gamma_long, theta_long, vega_long = black_scholes_call(spot_price, strike, T_long, r, base_iv + term_structure_premium)
        
        # Add bid-ask spread
        spread_pct = 0.015 # 1.5%
        short_spread = max(0.05, price_short * spread_pct)
        long_spread = max(0.05, price_long * spread_pct)
        
        short_bid = max(0.01, price_short - short_spread / 2)
        short_ask = price_short + short_spread / 2
        
        long_bid = max(0.01, price_long - long_spread / 2)
        long_ask = price_long + long_spread / 2
        
        # Volume & Open Interest mock distributions
        dist_from_spot = abs(strike - spot_price) / spot_price
        vol_short = int(max(10, 5000 * np.exp(-15 * dist_from_spot)))
        oi_short = int(max(50, 15000 * np.exp(-12 * dist_from_spot)))
        vol_long = int(max(5, 2000 * np.exp(-15 * dist_from_spot)))
        oi_long = int(max(20, 8000 * np.exp(-12 * dist_from_spot)))
        
        short_rows.append({
            "strike": strike,
            "bid": short_bid,
            "ask": short_ask,
            "impliedVolatility": base_iv,
            "volume": vol_short,
            "openInterest": oi_short,
            "delta": delta_short,
            "gamma": gamma_short,
            "theta": theta_short,
            "vega": vega_short
        })
        
        long_rows.append({
            "strike": strike,
            "bid": long_bid,
            "ask": long_ask,
            "impliedVolatility": base_iv + term_structure_premium,
            "volume": vol_long,
            "openInterest": oi_long,
            "delta": delta_long,
            "gamma": gamma_long,
            "theta": theta_long,
            "vega": vega_long
        })
        
    return {
        "short_expiry": short_exp,
        "short_dte": short_dte,
        "short_calls": pd.DataFrame(short_rows),
        "long_expiry": long_exp,
        "long_dte": long_dte,
        "long_calls": pd.DataFrame(long_rows),
        "is_synthetic": True
    }


# ══════════════════════════════════════════════════════════════════════
# BENCHMARK DATA LOADER (pre-fetch once per scan run)
# ══════════════════════════════════════════════════════════════════════

def fetch_benchmark_data() -> tuple:
    """
    Fetch SPY and QQQ market data for PCA and relative-strength calculations.

    Should be called ONCE before the ticker loop in execute_engine_scan() and
    the results passed down to calculate_pca_score() and
    calculate_leading_lagging_score() to avoid redundant API calls per ticker.

    Returns:
        tuple: (spy_df, benchmark_returns_df)
            - spy_df: Raw SPY history DataFrame (columns: Open, High, Low, Close, Volume, …)
            - benchmark_returns_df: DataFrame with columns ['SPY', 'QQQ'] of daily pct returns.
              Empty DataFrame on failure.
    """
    spy_df = pd.DataFrame()
    benchmark_returns = pd.DataFrame()

    try:
        spy_raw = yf.Ticker("SPY").history(period="1y")
        qqq_raw = yf.Ticker("QQQ").history(period="1y")

        if not spy_raw.empty:
            spy_df = spy_raw

        if not spy_raw.empty and not qqq_raw.empty:
            benchmark_returns = pd.concat(
                [
                    spy_raw['Close'].pct_change().rename("SPY"),
                    qqq_raw['Close'].pct_change().rename("QQQ"),
                ],
                axis=1,
            ).dropna()

        logger.info("Benchmark data (SPY, QQQ) fetched successfully.")
    except Exception as e:
        logger.warning(f"Could not fetch benchmark data: {e}. PCA and leading/lagging will use fallback.")

    return spy_df, benchmark_returns
