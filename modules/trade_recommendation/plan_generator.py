import logging

logger = logging.getLogger("TradePlanGenerator")

def generate_trade_plan(ticker: str, spot: float, decision: str, strategy: str, trend_state: str, 
                        expected_low: float, expected_high: float, daily_ind: dict) -> dict:
    """Generate trade plan details including triggers, invalidations, stops, and rationales."""
    
    # 1. Entry Trigger
    if strategy == "Call Debit Spread":
        entry_trigger = f"Close above Darvas upper band of ${daily_ind['darvas_upper']:.2f} with MACD expanding."
    elif strategy in ("Bullish Call Calendar", "Bullish Call Diagonal"):
        entry_trigger = f"Pullback confirmation near 20MA (${daily_ind['ma20']:.2f}) or 1H reversion to mean."
    elif strategy == "Iron Condor":
        entry_trigger = f"Entry within current range boundaries (${daily_ind['darvas_lower']:.2f} - ${daily_ind['darvas_upper']:.2f})."
    elif strategy == "Short Put Spread":
        entry_trigger = f"Reclaim of MA support near ${daily_ind['ma50']:.2f} with high IV rank."
    elif strategy == "Bear Call Spread":
        entry_trigger = f"Rejection of regression channel resistance at ${daily_ind['regression_upper']:.2f}."
    elif strategy == "Put Debit Spread":
        entry_trigger = f"Close below Darvas lower band of ${daily_ind['darvas_lower']:.2f} with MACD falling."
    elif strategy in ("Put Calendar", "Put Diagonal"):
        entry_trigger = f"Breakdown below 20MA (${daily_ind['ma20']:.2f}) or slow distribution rollover."
    else:
        entry_trigger = "No trade setup or triggers confirmed."

    # 2. Target Zone
    if "Bull" in trend_state or strategy in ("Call Debit Spread", "Bullish Call Calendar", "Bullish Call Diagonal", "Short Put Spread"):
        target_zone = f"${expected_high * 0.98:.2f} - ${expected_high * 1.02:.2f} (near 40D upper expected path)."
    elif "Bear" in trend_state or strategy in ("Put Debit Spread", "Put Calendar", "Put Diagonal", "Bear Call Spread"):
        target_zone = f"${expected_low * 0.98:.2f} - ${expected_low * 1.02:.2f} (near 40D lower expected path)."
    else:
        target_zone = f"${spot * 0.98:.2f} - ${spot * 1.02:.2f} (mean-reversion range)."

    # 3. Invalidation Rule
    if strategy in ("Call Debit Spread", "Bullish Call Calendar", "Bullish Call Diagonal", "Short Put Spread"):
        invalidation_rule = f"Close below regression lower band (${daily_ind['regression_lower']:.2f}) or below 50 EMA (${daily_ind['ma50']:.2f})."
    elif strategy in ("Put Debit Spread", "Put Calendar", "Put Diagonal", "Bear Call Spread"):
        invalidation_rule = f"Close above regression upper band (${daily_ind['regression_upper']:.2f}) or above 50 EMA (${daily_ind['ma50']:.2f})."
    else:
        invalidation_rule = f"Price breakout/breakdown beyond range boundaries (${daily_ind['darvas_lower']:.2f} - ${daily_ind['darvas_upper']:.2f})."

    # 4. Adjustment Rule
    if strategy == "Iron Condor":
        adjustment_rule = "Roll untested side towards price to collect credit if one leg is tested (delta > 30)."
    elif "Calendar" in strategy or "Diagonal" in strategy:
        adjustment_rule = "Roll front-month option to next available weekly expiration if trend remains active."
    elif "Spread" in strategy:
        adjustment_rule = "No adjustments. Allow defined-risk parameters to execute to expiration."
    else:
        adjustment_rule = "N/A"

    # 5. Profit Target
    if "Calendar" in strategy or "Diagonal" in strategy:
        profit_target = "Take profit at 30% - 50% of debit paid."
    elif "Condor" in strategy or "Spread" in strategy:
        if "Debit" in strategy:
            profit_target = "Take profit at 50% - 75% of maximum potential value."
        else:
            profit_target = "Take profit at 50% - 60% of credit collected."
    else:
        profit_target = "N/A"

    # 6. Max Loss Rule
    if "Calendar" in strategy or "Diagonal" in strategy or "Debit" in strategy:
        max_loss_rule = "Defined risk. Exit if spread loses 35% - 50% of the debit paid."
    elif "Credit" in strategy or "Condor" in strategy:
        max_loss_rule = "Defined risk. Exit if spread value exceeds 2x credit collected or if technical level is breached."
    else:
        max_loss_rule = "N/A"

    # 7. Rationale
    rationale = (
        f"The 3-month trend is classified as '{trend_state}' with a spot price of ${spot:.2f}. "
        f"The selected strategy '{strategy}' was chosen because the expected 40-day path is "
        f"pointing towards '{expected_low:.2f} to {expected_high:.2f}'. "
        f"Indicators show MACD is '{daily_ind['macd_signal']}' and FDTS signal is '{daily_ind['fdts_signal']}'. "
        f"This setup provides a highly defined risk/reward ratio."
    )

    # Option Structure recommendation details
    if "Calendar" in strategy:
        option_structure = "20/40 DTE Calendar Spread (Sell 20 DTE / Buy 40 DTE) at or near ATM strike."
    elif "Diagonal" in strategy:
        option_structure = "20/40 DTE Diagonal Spread (Sell 20 DTE ATM / Buy 40 DTE OTM) to capture directional bias."
    elif "Spread" in strategy:
        if "Debit" in strategy:
            option_structure = "30 DTE vertical debit spread (Buy ATM / Sell OTM)."
        else:
            option_structure = "30 DTE vertical credit spread (Sell OTM / Buy further OTM)."
    elif strategy == "Iron Condor":
        option_structure = "30 DTE Iron Condor (Sell OTM Put & Call, buy further OTM protection) outside expected ranges."
    else:
        option_structure = "No option structure recommended."

    return {
        "ticker": ticker,
        "decision": decision,
        "strategy": strategy,
        "option_structure": option_structure,
        "entry_trigger": entry_trigger,
        "target_zone": target_zone,
        "invalidation_rule": invalidation_rule,
        "adjustment_rule": adjustment_rule,
        "profit_target": profit_target,
        "max_loss_rule": max_loss_rule,
        "rationale": rationale
    }
