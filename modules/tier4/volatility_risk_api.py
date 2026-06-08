import logging
from modules.tier4.volatility_run_database import get_latest_run, get_latest_run_any

logger = logging.getLogger("VolatilityRiskAPI")

def get_current_volatility_risk() -> dict | None:
    """
    Returns the latest volatility run for any symbol, or None if no runs exist.
    
    Returns:
    {
        'volatility_risk_score': float,
        'risk_regime': str,
        'delta_action': str,
        'symbol': str,
        'run_datetime': str,
        'sub_scores': dict,
        'raw_inputs': dict,
    }
    """
    try:
        return get_latest_run_any()
    except Exception as e:
        logger.error(f"Error getting current volatility risk from API: {e}", exc_info=True)
        return None

def get_volatility_risk_for_symbol(symbol: str) -> dict | None:
    """Returns the latest volatility run for a specific symbol."""
    if not symbol:
        return None
    try:
        return get_latest_run(symbol)
    except Exception as e:
        logger.error(f"Error getting volatility risk for symbol {symbol}: {e}", exc_info=True)
        return None

def should_reduce_delta() -> bool:
    """Quick check: True if current broad regime is ELEVATED, HIGH, or EXTREME."""
    risk = get_current_volatility_risk()
    if not risk:
        return False
    return risk.get("risk_regime", "LOW") in ["ELEVATED", "HIGH", "EXTREME"]

def get_delta_reduction_factor() -> float:
    """
    Returns the exposure factor to apply based on the risk regime:
    - LOW: 1.0 (no reduction)
    - ELEVATED: 0.75 (trim 25%)
    - HIGH: 0.50 (trim 50%)
    - EXTREME: 0.0 (flatten / go neutral)
    """
    risk = get_current_volatility_risk()
    if not risk:
        return 1.0
    
    regime = risk.get("risk_regime", "LOW")
    if regime == "LOW":
        return 1.0
    elif regime == "ELEVATED":
        return 0.75
    elif regime == "HIGH":
        return 0.50
    elif regime == "EXTREME":
        return 0.0
    return 1.0
