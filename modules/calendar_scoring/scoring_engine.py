import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("CalendarScoringEngine")

# ══════════════════════════════════════════════════════════════════════
# COMPONENT SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def calculate_trend_score(spot_price: float, ema_20: float, ema_50: float, ema_200: float, adx_14: float) -> float:
    """Evaluate trend strength and alignment (Preferred: Uptrend)."""
    score = 50.0
    
    # Check EMA alignment
    if spot_price > ema_20 > ema_50 > ema_200:
        score = 85.0
        # ADX trend strength bonus
        if adx_14 > 25.0:
            score += 10.0
        elif adx_14 < 18.0:
            score -= 5.0
    elif spot_price > ema_50 > ema_200:
        score = 75.0
        if adx_14 > 20.0:
            score += 5.0
    elif spot_price > ema_200:
        score = 60.0
    else: # below 200 EMA
        score = 30.0
        
    return min(100.0, max(0.0, score))

def calculate_option_structure_score(front_iv: float, back_iv: float) -> float:
    """Evaluate volatility structure (Preferred: back_iv > front_iv - stable term structure)."""
    iv_diff = back_iv - front_iv
    
    # Contango is good for calendars: we buy back_iv, sell front_iv.
    # Normal contango term structure (back_iv > front_iv) gets high score
    if iv_diff > 0.05:
        score = 95.0
    elif iv_diff > 0.0:
        score = 85.0
    elif iv_diff > -0.03:
        score = 65.0
    else: # Backwardation (front_iv significantly higher than back_iv) is risky
        score = 35.0
        
    return score

def calculate_volatility_score(iv_rank: float, iv_percentile: float) -> float:
    """Evaluate IV levels (Preferred: low to moderate IV rank to avoid collapse)."""
    # Ideal IV Rank is low-to-mid range (e.g. 15 to 55) for stable/expanding IV potential.
    # High IV rank (>70-80) carries crush risk.
    if 15.0 <= iv_rank <= 55.0:
        score = 90.0
    elif iv_rank < 15.0:
        score = 75.0  # low but stable
    elif iv_rank <= 75.0:
        score = 60.0  # elevated
    else:
        score = 30.0  # very high crush risk
        
    # blend in percentile
    final_score = (score * 0.7) + (iv_percentile * 0.3)
    return min(100.0, max(0.0, final_score))

def calculate_fdts_score(fdts_signal_score: float) -> float:
    """Score matching the custom FDTS signal value."""
    return fdts_signal_score

def calculate_pca_score(ticker: str, df_history: pd.DataFrame) -> float:
    """Calculate PCA relative strength score proxy (Phase 2 model placeholder)."""
    # Standard proxy: check ticker beta/momentum compared to first principal component
    # For MVP: Return a technical score proxy using rolling price momentum relative to benchmark volatility
    if df_history.empty:
        return 70.0
        
    try:
        returns = df_history['Close'].pct_change().dropna()
        # Mock PCA score: calculate momentum stability
        momentum = returns.rolling(20).mean().iloc[-1]
        vol = returns.rolling(20).std().iloc[-1]
        sharpe_proxy = momentum / (vol + 1e-8)
        
        # Scale to 0-100
        score = 50.0 + (sharpe_proxy * 150.0)
        return min(95.0, max(40.0, score))
    except Exception:
        return 72.0

def calculate_cluster_score(ticker: str, df_history: pd.DataFrame) -> tuple[float, str]:
    """Calculate cluster classification score and label (Phase 2 model placeholder)."""
    # Categorizes tickers into: "Early Trend", "Mid Trend", "Overextended", "Consolidating"
    if df_history.empty:
        return 70.0, "Early Trend"
        
    try:
        close = df_history['Close']
        ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        
        # Distance metrics for clustering
        dist_20 = (close.iloc[-1] - ema_20) / ema_20
        dist_50 = (close.iloc[-1] - ema_50) / ema_50
        
        if dist_50 > 0.08:
            label = "Overextended"
            score = 60.0
        elif dist_20 > 0.01 and dist_50 > 0.02:
            label = "Mid Trend"
            score = 90.0
        elif dist_20 > -0.01 and dist_50 > 0.01:
            label = "Early Trend"
            score = 95.0
        else:
            label = "Consolidating"
            score = 75.0
            
        return score, label
    except Exception:
        return 70.0, "Early Trend"

def calculate_leading_lagging_score(ticker: str, df_history: pd.DataFrame) -> tuple[float, str]:
    """Determine leading/lagging state relative to market benchmark (Phase 2 placeholder)."""
    if df_history.empty:
        return 75.0, "Leading"
        
    try:
        # Standard proxy: Compare ticker 20-day returns vs benchmark SPY 20-day returns
        ticker_ret = (df_history['Close'].iloc[-1] / df_history['Close'].iloc[-20]) - 1
        
        # Check SPY return
        spy_ret = 0.02 # default positive return
        try:
            spy = yf.Ticker("SPY").history(period="1m")
            spy_ret = (spy['Close'].iloc[-1] / spy['Close'].iloc[-20]) - 1
        except Exception:
            pass
            
        diff = ticker_ret - spy_ret
        if diff > 0.03:
            state = "Strong Leader"
            score = 95.0
        elif diff > 0.0:
            state = "Leading"
            score = 85.0
        elif diff > -0.03:
            state = "Lagging"
            score = 65.0
        else:
            state = "Strong Lagger"
            score = 45.0
            
        return score, state
    except Exception:
        return 75.0, "Leading"

def calculate_liquidity_score(bid_ask_spread_pct: float, avg_option_volume: float) -> float:
    """Evaluate option chain liquidity (Preferred: tight spreads and high volume)."""
    score = 50.0
    
    # Spread evaluation
    if bid_ask_spread_pct <= 0.01:
        score += 30.0
    elif bid_ask_spread_pct <= 0.03:
        score += 20.0
    elif bid_ask_spread_pct <= 0.07:
        score += 10.0
    else:
        score -= 10.0
        
    # Volume/OI evaluation
    if avg_option_volume > 1000:
        score += 20.0
    elif avg_option_volume > 200:
        score += 10.0
        
    return min(100.0, max(0.0, score))

def calculate_event_risk_score(earnings_date_str: str, short_dte: int) -> tuple[float, int]:
    """Calculate event risk score based on earnings proximity (Preferred: earnings outside DTE)."""
    if not earnings_date_str:
        return 95.0, 0 # No earnings date found, low risk
        
    try:
        earn_date = datetime.strptime(earnings_date_str, "%Y-%m-%d")
        today = datetime.now()
        days_to_earn = (earn_date - today).days
        
        # If earnings occur during our option trade (especially before short leg expiry)
        if 0 <= days_to_earn <= (short_dte + 5):
            # Extremely high event risk (implied vol crush, stock gap risk)
            return 20.0, 1
        elif 0 <= days_to_earn <= 45:
            # Moderate event risk (occurs between short and long expiry)
            return 60.0, 0
        else:
            return 95.0, 0
    except Exception:
        return 90.0, 0

# ══════════════════════════════════════════════════════════════════════
# HARD FILTERS
# ══════════════════════════════════════════════════════════════════════

def apply_hard_filters(ticker: str, tech_data: dict, option_setup: dict, fdts_signal: str) -> list[str]:
    """Verify ticker eligibility against hard criteria. Returns reasons for exclusion, if any."""
    exclusions = []
    
    # 1. FDTS == Sell
    if fdts_signal == "Sell":
        exclusions.append("FDTS Signal is Sell")
        
    # 2. Bid/Ask spread > 7%
    spread = option_setup.get("bid_ask_spread_pct", 1.0)
    if spread > 0.07:
        exclusions.append(f"Bid/Ask Spread too wide ({spread*100:.1f}%)")
        
    # 3. Average option volume too low
    vol = option_setup.get("avg_option_volume", 0)
    if vol < 5:
        exclusions.append(f"Low Option Volume ({vol:.0f})")
        
    # 4. Open interest too low
    oi = option_setup.get("avg_open_interest", 0)
    if oi < 50:
        exclusions.append(f"Low Open Interest ({oi:.0f})")
        
    # 5. Earnings inside trade window (before short-leg expiration)
    event_risk_flag = tech_data.get("event_risk_flag", 0)
    # Check earnings proximity
    earn_date_str = tech_data.get("earnings_date")
    if earn_date_str:
        try:
            earn_date = datetime.strptime(earn_date_str, "%Y-%m-%d")
            days_to_earn = (earn_date - datetime.now()).days
            if 0 <= days_to_earn <= option_setup.get("short_dte", 20):
                exclusions.append(f"Earnings Inside Trade Window ({days_to_earn} days to earnings)")
        except Exception:
            pass
            
    # 6. IV Rank > 80
    iv_rank = tech_data.get("iv_rank", 0.0)
    if iv_rank > 80.0:
        exclusions.append(f"IV Rank too high ({iv_rank:.1f})")
        
    # 7. ADX too weak (< 14 or 15)
    adx = tech_data.get("adx_14", 20.0)
    if adx < 12.0:
        exclusions.append(f"Trend ADX too weak ({adx:.1f})")
        
    # 8. Price below 50 EMA and 200 EMA
    spot = tech_data.get("spot_price", 0.0)
    ema_50 = tech_data.get("ema_50", 0.0)
    ema_200 = tech_data.get("ema_200", 0.0)
    if spot < ema_50 and spot < ema_200:
        exclusions.append("Price below both 50 EMA and 200 EMA")
        
    # 9. Missing back/front expiry or legs
    if not option_setup.get("short_expiry") or not option_setup.get("long_expiry"):
        exclusions.append("Missing front or back month option legs")
        
    return exclusions
