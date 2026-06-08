import logging

logger = logging.getLogger("VolatilityScoreEngine")

def _score_vix_regime(vix_current, vix_percentile, vvix_current, term_shape, triggers) -> float:
    """Calculate VIX regime sub-score (0-100), weighted 20%."""
    score = 0.0
    
    # 1. Base on VIX level
    if vix_current is not None:
        if vix_current < 12.0:
            score += 15.0
            triggers.append(f"VIX is very low ({vix_current:.1f}) - showing complacency")
        elif vix_current < 15.0:
            score += 30.0
        elif vix_current < 20.0:
            score += 55.0
        elif vix_current < 30.0:
            score += 80.0
            triggers.append(f"VIX is elevated ({vix_current:.1f})")
        else:
            score += 100.0
            triggers.append(f"VIX is in stress zone ({vix_current:.1f})")
            
    # 2. Base on VIX percentile
    if vix_percentile is not None:
        if vix_percentile < 20.0:
            score += 15.0
            triggers.append(f"VIX Percentile is very low ({vix_percentile:.1f}%) - high complacency")
        elif vix_percentile > 80.0:
            score += 10.0
            triggers.append(f"VIX Percentile is high ({vix_percentile:.1f}%)")
            
    # 3. Base on VVIX
    if vvix_current is not None:
        if vvix_current > 110.0:
            score += 15.0
            triggers.append(f"VVIX is high ({vvix_current:.1f}) - tail risk pricing is elevated")
        elif vvix_current > 90.0:
            score += 5.0
            
    # 4. Base on Term Structure
    if term_shape == "Backwardation":
        score += 20.0
        triggers.append("VIX Term Structure is in Backwardation")
        
    return min(100.0, max(0.0, score))


def _score_put_call(put_call_ratio, triggers) -> float:
    """Calculate Put/Call ratio sub-score (0-100), weighted 15%."""
    if put_call_ratio is None:
        return 30.0
        
    if put_call_ratio < 0.65:
        triggers.append(f"Options Put/Call Ratio is extremely low ({put_call_ratio:.2f}) - high complacency")
        return 100.0
    elif put_call_ratio < 0.8:
        triggers.append(f"Options Put/Call Ratio is low ({put_call_ratio:.2f}) - complacency")
        return 80.0
    elif put_call_ratio > 1.3:
        triggers.append(f"Options Put/Call Ratio is high ({put_call_ratio:.2f}) - defensive hedging active")
        return 60.0
    else:
        return 30.0


def _score_price_action(spy_vs_20ema_pct, spy_vs_50ema_pct, trend_label, consecutive_down_days, triggers) -> float:
    """Calculate Price Action sub-score (0-100), weighted 20%."""
    score = 0.0
    
    # 1. Trend Label
    if trend_label in ["DOWNTREND", "DOWN", "BEARISH"]:
        score += 40.0
        triggers.append("Index is in a confirmed downtrend")
    
    # 2. 20 EMA
    if spy_vs_20ema_pct is not None:
        if spy_vs_20ema_pct < 0.0:
            score += 30.0
            triggers.append(f"Price is below 20 EMA ({spy_vs_20ema_pct:.2f}%)")
        elif spy_vs_20ema_pct > 3.0:
            score += 20.0
            triggers.append(f"Price is overextended above 20 EMA ({spy_vs_20ema_pct:.2f}%)")
            
    # 3. 50 EMA
    if spy_vs_50ema_pct is not None:
        if spy_vs_50ema_pct < 0.0:
            score += 20.0
            triggers.append(f"Price is below 50 EMA ({spy_vs_50ema_pct:.2f}%)")
        elif spy_vs_50ema_pct > 5.0:
            score += 25.0
            triggers.append(f"Price is extremely overextended above 50 EMA ({spy_vs_50ema_pct:.2f}%)")
            
    # 4. Consecutive down days
    if consecutive_down_days is not None:
        if consecutive_down_days >= 5:
            score += 25.0
            triggers.append(f"Market has {consecutive_down_days} consecutive down days")
        elif consecutive_down_days >= 3:
            score += 15.0
            triggers.append(f"Market has {consecutive_down_days} consecutive down days")
        elif consecutive_down_days >= 1:
            score += 5.0
            
    return min(100.0, max(0.0, score))


def _score_breadth(spy_5d_return, qqq_spy_divergence, triggers) -> float:
    """Calculate Breadth sub-score (0-100), weighted 15%."""
    score = 30.0 # Baseline
    
    # 1. SPY 5d return
    if spy_5d_return is not None:
        if spy_5d_return < -0.02: # -2.0%
            score = 90.0
            triggers.append(f"SPY 5-day return is negative ({spy_5d_return*100:.1f}%) - selling pressure")
        elif spy_5d_return < -0.005: # -0.5%
            score = 60.0
            triggers.append(f"SPY 5-day return is weak ({spy_5d_return*100:.1f}%)")
        elif spy_5d_return > 0.03: # +3.0%
            score = 75.0
            triggers.append(f"SPY 5-day return is overextended ({spy_5d_return*100:.1f}%)")
            
    # 2. QQQ/SPY divergence
    if qqq_spy_divergence is not None:
        if qqq_spy_divergence > 0.015: # 1.5%
            score += 25.0
            triggers.append(f"QQQ is diverging significantly above SPY (+{qqq_spy_divergence*100:.1f}%) - narrow tech rally")
        elif qqq_spy_divergence < -0.015: # -1.5%
            score += 20.0
            triggers.append(f"QQQ is diverging significantly below SPY ({qqq_spy_divergence*100:.1f}%) - tech leading downside")
            
    return min(100.0, max(0.0, score))


def _score_liquidity_gamma(gamma_regime, liquidity_label, hvr, triggers) -> float:
    """Calculate Liquidity and Gamma sub-score (0-100), weighted 15%."""
    score = 0.0
    
    # 1. Gamma regime
    if gamma_regime is not None:
        gamma_str = str(gamma_regime).lower()
        if "negative" in gamma_str:
            score += 80.0
            triggers.append(f"Gamma Regime is Negative GEX")
        elif "transition" in gamma_str:
            score += 50.0
            triggers.append(f"Gamma Regime is Transition Zone")
        else:
            score += 20.0
    else:
        score += 30.0 # Default if unknown
        
    # 2. Liquidity label
    if liquidity_label is not None:
        liq_str = str(liquidity_label).upper()
        if liq_str in ["POOR", "LOW", "VERY POOR"]:
            score += 20.0
            triggers.append(f"Option book liquidity is poor ({liq_str})")
        elif liq_str in ["FAIR", "MEDIUM"]:
            score += 10.0
            
    # 3. HVR (Historical Volatility Ratio)
    if hvr is not None:
        # Check scale of hvr (could be ratio like 1.8 or percentage like 180)
        hvr_val = float(hvr)
        if hvr_val > 1.5 or hvr_val > 150.0:
            score += 15.0
            triggers.append(f"Historical Volatility Ratio is high ({hvr_val:.1f})")
            
    return min(100.0, max(0.0, score))


def _score_macro_event(days_to_earnings, macro_event_flagged, term_shape, triggers) -> float:
    """Calculate Macro Event sub-score (0-100), weighted 15%."""
    score = 10.0 # Baseline
    
    # 1. Manual macro event flagged
    if macro_event_flagged:
        score += 40.0
        triggers.append("Macro / Event risk flagged in sidebar")
        
    # 2. Days to earnings
    if days_to_earnings is not None:
        if days_to_earnings <= 7:
            score += 30.0
            triggers.append(f"Corporate earnings in {days_to_earnings} days")
            
    # 3. VIX Backwardation also impacts macro/event readiness
    if term_shape == "Backwardation":
        score += 20.0
        
    return min(100.0, max(0.0, score))


def calculate_volatility_risk_score(
    # VIX inputs
    vix_current, vix_percentile, vvix_current, term_shape,
    # P/C ratio
    put_call_ratio,
    # Price action
    spy_vs_20ema_pct, spy_vs_50ema_pct, trend_label, consecutive_down_days,
    # Breadth
    spy_5d_return, qqq_spy_divergence,
    # Liquidity / Gamma
    gamma_regime, liquidity_label, hvr,
    # Macro / Event
    days_to_earnings, macro_event_flagged,
) -> dict:
    """
    Returns a composite score and action recommendation based on 6 categories of inputs.
    
    Returns:
    {
        'volatility_risk_score': float,     # 0-100
        'risk_regime': str,                 # LOW/ELEVATED/HIGH/EXTREME
        'delta_action': str,                # HOLD/REDUCE_25/REDUCE_50/NEUTRAL
        'sub_scores': { category: score },  # For transparency
        'triggers': [str, ...],             # Human-readable trigger list
    }
    """
    triggers = []
    
    # Sub-scores
    vix_regime_score = _score_vix_regime(vix_current, vix_percentile, vvix_current, term_shape, triggers)
    put_call_score = _score_put_call(put_call_ratio, triggers)
    price_action_score = _score_price_action(spy_vs_20ema_pct, spy_vs_50ema_pct, trend_label, consecutive_down_days, triggers)
    breadth_score = _score_breadth(spy_5d_return, qqq_spy_divergence, triggers)
    liquidity_gamma_score = _score_liquidity_gamma(gamma_regime, liquidity_label, hvr, triggers)
    macro_event_score = _score_macro_event(days_to_earnings, macro_event_flagged, term_shape, triggers)
    
    # Composite Score calculation (weights sum to 100%)
    volatility_risk_score = (
        0.20 * vix_regime_score +
        0.15 * put_call_score +
        0.20 * price_action_score +
        0.15 * breadth_score +
        0.15 * liquidity_gamma_score +
        0.15 * macro_event_score
    )
    
    volatility_risk_score = min(100.0, max(0.0, volatility_risk_score))
    
    # Regime and Delta Action classification
    if volatility_risk_score <= 25.0:
        risk_regime = "LOW"
        delta_action = "HOLD"
    elif volatility_risk_score <= 50.0:
        risk_regime = "ELEVATED"
        delta_action = "REDUCE_25"
    elif volatility_risk_score <= 75.0:
        risk_regime = "HIGH"
        delta_action = "REDUCE_50"
    else:
        risk_regime = "EXTREME"
        delta_action = "NEUTRAL"
        
    return {
        "volatility_risk_score": round(volatility_risk_score, 2),
        "risk_regime": risk_regime,
        "delta_action": delta_action,
        "sub_scores": {
            "vix_regime_score": round(vix_regime_score, 2),
            "put_call_score": round(put_call_score, 2),
            "price_action_score": round(price_action_score, 2),
            "breadth_score": round(breadth_score, 2),
            "liquidity_gamma_score": round(liquidity_gamma_score, 2),
            "macro_event_score": round(macro_event_score, 2)
        },
        "triggers": triggers
    }
