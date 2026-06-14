def map_cycle_strategy(
    regime: str,
    cycle_direction: str,
    cycle_phase_pct: float,
    vol_status: str,
    calendar_suitability: float,
    event_risk_score: float,
    trade_size_modifier: float,
    vix_pct: float = 50.0
) -> dict:
    """
    Translates market regimes and cycle phase values into an option strategy recommendation,
    along with entry quality, size modifiers, and reason explanations.
    """
    
    # 1. Critical Overrides (Spike Risk / High Event Risk)
    if event_risk_score >= 80.0 or vol_status == "Vol Spike Risk":
        return {
            "recommended_strategy": "REJECT / HOLD CASH",
            "strategy_confidence": 10.0,
            "entry_quality": "Poor",
            "position_size_modifier": 0.0,
            "reason_code": "Volatility spike risk or extreme event risk (score: {:.1f}). Stand aside.".format(event_risk_score)
        }

    # 2. Bullish Early Expansion (Phase 0% to 35%, Rising, Bullish Regime)
    if regime in ["Bull Trend", "Dip Buy Zone"] and cycle_direction == "rising" and cycle_phase_pct <= 35.0:
        if calendar_suitability >= 75.0:
            strategy = "Bull Call Calendar"
            conf = 85.0
            reason = "Bull trend, early cycle expansion, VIX stable, low event risk"
        else:
            strategy = "Bull Call Diagonal"
            conf = 75.0
            reason = "Bull trend, early cycle expansion, IV elevated/expanding. Diagonal captures skew."
        return {
            "recommended_strategy": strategy,
            "strategy_confidence": conf,
            "entry_quality": "Excellent",
            "position_size_modifier": trade_size_modifier,
            "reason_code": reason
        }

    # 3. Bullish Late Expansion / Peak Zone (Phase 40% to 60%)
    if regime in ["Bull Trend", "Trend Exhaustion"] and 40.0 <= cycle_phase_pct <= 60.0:
        return {
            "recommended_strategy": "TAKE PROFIT / REDUCE GAMMA",
            "strategy_confidence": 80.0,
            "entry_quality": "Fair",
            "position_size_modifier": max(trade_size_modifier * 0.5, 0.25),
            "reason_code": "Cycle peaking (phase: {:.1f}%). Avoid adding new long delta. Hedge or take profit.".format(cycle_phase_pct)
        }

    # 4. Bearish Contraction (Phase 55% to 90%, Falling, Bearish Regime)
    if regime in ["Bear Trend", "Sell the Rip Zone"] and cycle_direction == "falling" and 55.0 <= cycle_phase_pct <= 90.0:
        if vol_status in ["Vol Expanding", "Vol Contracting"] and vix_pct > 60.0:
            strategy = "Bear Put Spread"
            conf = 70.0
            reason = "Bearish mature cycle contraction, VIX elevated. Debit spread limits risk."
        else:
            strategy = "Put Diagonal"
            conf = 65.0
            reason = "Bearish cycle contraction. Put diagonal captures downside drift and IV skew."
            
        return {
            "recommended_strategy": strategy,
            "strategy_confidence": conf,
            "entry_quality": "Good",
            "position_size_modifier": trade_size_modifier * 0.75,
            "reason_code": reason
        }

    # 5. Sideways Cycle (Regime Flat/Sideways)
    if regime in ["Sideways Range", "No Edge"] or (30.0 < calendar_suitability < 75.0 and abs(cycle_phase_pct - 50.0) > 20.0):
        # IV Rank or VIX decides between condors and calendars
        if vix_pct >= 50.0:
            strategy = "Iron Condor"
            conf = 80.0
            reason = "Sideways range with elevated IV percentile. Sell premium on both sides."
        else:
            strategy = "Double Calendar (At Center Strike)"
            conf = 75.0
            reason = "Sideways price action with low IV percentile. Double calendar captures volatility compression."
            
        return {
            "recommended_strategy": strategy,
            "strategy_confidence": conf,
            "entry_quality": "Good",
            "position_size_modifier": trade_size_modifier,
            "reason_code": reason
        }

    # 6. Default Fallback / Neutral watch
    return {
        "recommended_strategy": "WATCH / NO EDGE",
        "strategy_confidence": 40.0,
        "entry_quality": "Fair",
        "position_size_modifier": 0.25,
        "reason_code": "Mixed cycle signals. Wait for cycle alignment or volatility expansion."
    }
