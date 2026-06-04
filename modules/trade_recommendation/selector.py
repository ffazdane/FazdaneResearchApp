import logging

logger = logging.getLogger("StrategySelector")

def calculate_component_scores(daily_ind: dict, vix_value: float, option_liq: dict = None, earnings_date: str = None) -> dict:
    """Calculate the 9 sub-scores out of 100 based on indicators and option data."""
    price = daily_ind["price"]
    ma20 = daily_ind["ma20"]
    ma50 = daily_ind["ma50"]
    ma200 = daily_ind["ma200"]
    
    # 1. Trend Structure (20 pts)
    if price > ma20 > ma50 > ma200:
        trend_score = 100.0
    elif price > ma50 > ma200:
        trend_score = 80.0
    elif price > ma200:
        trend_score = 60.0
    elif price < ma20 < ma50 < ma200:
        trend_score = 10.0
    else:
        trend_score = 40.0
        
    # 2. FDTS Signal (15 pts)
    fdts = daily_ind["fdts_signal"]
    if fdts == "Buy":
        fdts_score = 100.0
    elif fdts == "Sell":
        fdts_score = 10.0
    else:
        fdts_score = 50.0
        
    # 3. MACD Momentum (15 pts)
    macd_hist = daily_ind["macd_hist"]
    macd_sig = daily_ind["macd_signal"]
    if macd_sig == "Bullish" and macd_hist > 0:
        macd_score = 100.0
    elif macd_sig == "Bullish":
        macd_score = 75.0
    elif macd_sig == "Bearish" and macd_hist < 0:
        macd_score = 10.0
    else:
        macd_score = 40.0
        
    # 4. Darvas breakout/range structure (15 pts)
    darvas_up = daily_ind["darvas_upper"]
    darvas_lo = daily_ind["darvas_lower"]
    darvas_sig = daily_ind["darvas_signal"]
    
    if darvas_sig == "Breakout":
        darvas_score = 100.0
    elif darvas_sig == "Breakdown":
        darvas_score = 10.0
    elif abs(price - darvas_up) / darvas_up < 0.02:
        darvas_score = 80.0
    elif abs(price - darvas_lo) / darvas_lo < 0.02:
        darvas_score = 30.0
    else:
        darvas_score = 50.0
        
    # 5. Regression Location (10 pts)
    reg_up = daily_ind["regression_upper"]
    reg_lo = daily_ind["regression_lower"]
    reg_range = reg_up - reg_lo
    if reg_range > 0:
        pos_pct = (price - reg_lo) / reg_range
        # 100 is near bottom band (support / mean reversion buy), 0 is near top band (overextended)
        regression_score = max(0.0, min(100.0, (1.0 - pos_pct) * 100.0))
    else:
        regression_score = 50.0
        
    # 6. Volatility / IV Condition (10 pts)
    iv_rank = daily_ind.get("iv_rank", 30.0)
    if 15.0 <= iv_rank <= 55.0:
        volatility_score = 100.0
    elif iv_rank < 15.0:
        volatility_score = 80.0
    elif iv_rank <= 75.0:
        volatility_score = 50.0
    else:
        volatility_score = 20.0
        
    # 7. WPR / Market Forecast Timing (5 pts)
    wpr_s = daily_ind["wpr_signal"]
    if wpr_s == "Oversold":
        timing_score = 100.0
    elif wpr_s == "Overbought":
        timing_score = 20.0
    else:
        timing_score = 60.0
        
    # 8. Liquidity / Option Quality (5 pts)
    # Default to 80.0 if not provided
    liquidity_score = 80.0
    if option_liq:
        spread = option_liq.get("bid_ask_spread_pct", 0.02)
        vol = option_liq.get("avg_option_volume", 500)
        
        liq_val = 50.0
        if spread <= 0.01:
            liq_val += 30.0
        elif spread <= 0.03:
            liq_val += 20.0
        elif spread <= 0.07:
            liq_val += 10.0
            
        if vol > 1000:
            liq_val += 20.0
        elif vol > 200:
            liq_val += 10.0
        liquidity_score = min(100.0, liq_val)
        
    # 9. Event Risk (5 pts)
    # Default to 90.0 if not provided
    event_risk_score = 90.0
    if earnings_date:
        try:
            from datetime import datetime
            earn_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
            days_to_earn = (earn_dt - datetime.now()).days
            if 0 <= days_to_earn <= 20:
                event_risk_score = 20.0
            elif 20 < days_to_earn <= 40:
                event_risk_score = 60.0
            else:
                event_risk_score = 95.0
        except Exception:
            pass
            
    return {
        "trend_score": trend_score,
        "fdts_score": fdts_score,
        "momentum_score": macd_score,
        "range_score": darvas_score,
        "regression_score": regression_score,
        "volatility_score": volatility_score,
        "timing_score": timing_score,
        "liquidity_score": liquidity_score,
        "event_risk_score": event_risk_score
    }

def calculate_composite_score(scores: dict) -> float:
    """Calculate the final weighted trade score out of 100."""
    return float(
        scores["trend_score"] * 0.20 +
        scores["fdts_score"] * 0.15 +
        scores["momentum_score"] * 0.15 +
        scores["range_score"] * 0.15 +
        scores["regression_score"] * 0.10 +
        scores["volatility_score"] * 0.10 +
        scores["timing_score"] * 0.05 +
        scores["liquidity_score"] * 0.05 +
        scores["event_risk_score"] * 0.05
    )

def select_trade_strategy(scores: dict, daily_ind: dict, expected_40d_path: str) -> tuple[str, str]:
    """Select the best options strategy and provide the rationale."""
    trend_score = scores["trend_score"]
    momentum_score = scores["momentum_score"]
    range_score = scores["range_score"]
    volatility_score = scores["volatility_score"]
    liquidity_score = scores["liquidity_score"]
    event_risk_score = scores["event_risk_score"]
    
    # Check hard filters
    if liquidity_score < 70:
        return "Reject", "Poor option liquidity (score: {:.1f} < 70)".format(liquidity_score)
        
    if event_risk_score < 60:
        return "Reject", "Event risk too high / earnings too close (score: {:.1f} < 60)".format(event_risk_score)
        
    iv_rank = daily_ind.get("iv_rank", 30.0)
    
    if expected_40d_path == "sideways_range":
        if iv_rank >= 40 and range_score >= 75:
            return "Iron Condor", "Sideways 40-day range with elevated implied volatility"
        else:
            return "Calendar", "Sideways range but premium selling edge is weak"
            
    elif expected_40d_path == "sideways_bullish":
        if trend_score >= 75 and momentum_score < 70:
            return "Bullish Call Calendar", "Bullish trend with controlled sideways-up drift"
        elif trend_score >= 80 and momentum_score >= 70:
            return "Bullish Call Diagonal", "Bullish drift with momentum improving"
            
    elif expected_40d_path == "directional_bullish":
        if momentum_score >= 80:
            return "Call Debit Spread", "Confirmed bullish breakout with momentum expansion"
        else:
            return "Call Diagonal", "Bullish trend, but use defined risk diagonal spread"
            
    elif expected_40d_path == "mean_reversion_up":
        if iv_rank >= 40:
            return "Short Put Spread", "Mean reversion bounce setup at support with premium"
        else:
            return "Bullish Call Calendar", "Pullback bounce setup near support"
            
    elif expected_40d_path == "directional_bearish":
        if momentum_score <= 35:
            return "Put Debit Spread", "Bearish breakdown confirmed with momentum expansion"
        else:
            return "Put Diagonal", "Bearish drift but not aggressive breakdown"
            
    elif expected_40d_path == "sideways_bearish":
        if iv_rank >= 40:
            return "Bear Call Spread", "Resistance rejection with premium"
        else:
            return "Put Calendar", "Slow bearish drift"
            
    return "Reject", "No clean trade edge identified based on indicators"

def adjust_decision_for_alignment(decision: str, strategy: str, daily_ind: dict) -> tuple[str, str]:
    """Downgrade Deploy to Watch if there is no state alignment (e.g. price directly under resistance)."""
    if decision != "Deploy":
        return decision, ""
        
    price = daily_ind["price"]
    darvas_up = daily_ind["darvas_upper"]
    reg_up = daily_ind["regression_upper"]
    darvas_lo = daily_ind["darvas_lower"]
    reg_lo = daily_ind["regression_lower"]
    
    # Bullish strategies check
    if strategy in ("Bullish Call Calendar", "Bullish Call Diagonal", "Call Debit Spread", "Short Put Spread"):
        # Directly under resistance (within 1.5%)
        if price >= darvas_up * 0.985 or price >= reg_up * 0.985:
            return "Watch", "Downgraded to Watch: Price is directly under resistance levels."
            
    # Bearish strategies check
    elif strategy in ("Put Debit Spread", "Put Calendar", "Put Diagonal", "Bear Call Spread", "Short Call Spread"):
        # Directly above support (within 1.5%)
        if price <= darvas_lo * 1.015 or price <= reg_lo * 1.015:
            return "Watch", "Downgraded to Watch: Price is directly above support levels."
            
    return decision, ""
