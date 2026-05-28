import logging
from modules.calendar_scoring.option_chain_engine import select_optimal_calendar_legs

logger = logging.getLogger("CalendarTradeSetup")

def select_calendar_setup(ticker: str, option_chain: dict, spot_price: float, hv_30: float, target_delta: float = 0.25) -> dict:
    """Formulate the bullish option calendar setup parameters for logging and display."""
    legs = select_optimal_calendar_legs(option_chain, spot_price, hv_30, target_delta)
    if not legs:
        logger.warning(f"No optimal legs found for {ticker}")
        return {}
        
    # Calculate breakevens (approximation using volatility range standard deviation)
    # Lower: strike - debit, Upper: strike + debit + (implied volatility move)
    strike = legs["selected_strike"]
    debit = legs["net_debit"]
    
    breakeven_low = strike - debit * 0.85
    breakeven_high = strike + debit * 1.5
    
    return {
        "ticker": ticker,
        "strategy_type": "Bullish Calendar Spread",
        "short_dte": legs["short_dte"],
        "long_dte": legs["long_dte"],
        "target_delta": target_delta,
        "short_expiry": legs["short_expiry"],
        "long_expiry": legs["long_expiry"],
        "selected_strike": strike,
        "short_bid": legs["short_bid"],
        "short_ask": legs["short_ask"],
        "short_mid": legs["short_mid"],
        "long_bid": legs["long_bid"],
        "long_ask": legs["long_ask"],
        "long_mid": legs["long_mid"],
        "net_debit": debit,
        "max_risk": debit,
        "setup_delta": legs["setup_delta"],
        "setup_gamma": legs["setup_gamma"],
        "setup_theta": legs["setup_theta"],
        "setup_vega": legs["setup_vega"],
        "breakeven_low": float(breakeven_low),
        "breakeven_high": float(breakeven_high),
        "avg_option_volume": legs["avg_option_volume"],
        "avg_open_interest": legs["avg_open_interest"],
        "bid_ask_spread_pct": legs["bid_ask_spread_pct"],
        "front_iv": legs["front_iv"],
        "back_iv": legs["back_iv"]
    }
