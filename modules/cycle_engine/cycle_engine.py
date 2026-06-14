import os
import json
import logging
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta

from modules.cycle_engine.cycle_database import save_signal
from modules.cycle_engine.dominant_cycle_detector import detect_dominant_cycle
from modules.cycle_engine.cycle_phase_engine import calculate_cycle_phase
from modules.cycle_engine.turning_point_engine import calculate_turning_points
from modules.cycle_engine.cycle_alignment_engine import calculate_alignment
from modules.cycle_engine.volatility_cycle_engine import analyze_volatility
from modules.cycle_engine.liquidity_cycle_engine import calculate_liquidity_event_risk
from modules.cycle_engine.cycle_regime_classifier import classify_market_regime
from modules.cycle_engine.strategy_mapper import map_cycle_strategy

logger = logging.getLogger("CycleEngineAPI")

# Load configuration if available
def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "cycle_engine_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # Fallback default values
    return {
        "final_score_weights": {
            "dominant_cycle_strength": 0.20,
            "phase_quality_score": 0.15,
            "turning_point_confidence": 0.15,
            "alignment_score": 0.15,
            "volatility_suitability": 0.15,
            "liquidity_score": 0.10,
            "regime_score": 0.10
        }
    }

def run_cycle_analysis(ticker: str, as_of_date: str = None, timeframe: str = "daily") -> dict:
    """
    Runs full cycle analysis for a ticker.
    Returns cycle, regime, volatility, liquidity, and strategy recommendation.
    Saves the output signal to the SQLite database.
    """
    # 1. Parse dates
    if as_of_date:
        today_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    else:
        today_date = datetime.today().date()
        
    start_date = today_date - timedelta(days=365) # past year of data
    
    # 2. Download Daily Stock Prices
    try:
        # Standard yfinance fetch
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=today_date + timedelta(days=1), auto_adjust=True)
        if df.empty:
            raise ValueError(f"No pricing data found for ticker {ticker}")
        df.index = pd.to_datetime(df.index).tz_localize(None)
    except Exception as e:
        logger.error(f"Failed to fetch market data for {ticker}: {e}")
        raise e

    prices = df["Close"]
    current_price = float(prices.iloc[-1])

    # 3. Dominant Cycle Detection
    cycle_info = detect_dominant_cycle(prices)
    dom_cycle = cycle_info["dominant_cycle_days"]
    cycle_strength = cycle_info["cycle_strength"]

    # 4. Cycle Phase Analysis
    phase_info = calculate_cycle_phase(prices, dom_cycle)
    phase_pct = phase_info["cycle_phase_pct"]
    direction = phase_info["cycle_direction"]
    phase_label = phase_info["phase_label"]
    days_to_peak = phase_info["estimated_days_to_peak"]
    days_to_bottom = phase_info["estimated_days_to_bottom"]

    # 5. Volatility Cycle Analysis
    vol_info = analyze_volatility(ticker, prices, today_date)
    vol_status = vol_info["volatility_cycle_status"]
    vix_pct = vol_info["vix_percentile"]
    vix_stable = vol_status not in ["Vol Spike Risk", "Vol Unstable"]
    cal_suitability = vol_info["calendar_suitability"]

    # 6. Liquidity and Event Risk Analysis
    liq_info = calculate_liquidity_event_risk(ticker, today_date)
    event_risk = liq_info["event_risk_score"]
    trade_size_mod = liq_info["trade_size_modifier"]
    days_to_earn = liq_info.get("days_to_earnings")
    # Parse days to earnings if returned
    d_earn = None
    if days_to_earn:
        d_earn = (datetime.strptime(days_to_earn, "%Y-%m-%d").date() - today_date).days

    # 7. Regime Classifier
    regime_info = classify_market_regime(df, direction, phase_pct, vol_info["vix_value"])
    regime = regime_info["regime"]
    regime_score = regime_info["regime_score"]
    support = regime_info["support"]
    resistance = regime_info["resistance"]

    # 8. Turning Point Window Projections
    turn_info = calculate_turning_points(
        today_date,
        days_to_peak,
        days_to_bottom,
        cycle_strength,
        cycle_info["method_agreement_score"],
        current_price,
        support,
        resistance,
        vix_stable,
        event_risk,
        d_earn
    )

    # 9. Multi-Asset Cycle Alignment
    # Run alignment across SPY, QQQ, IWM, DIA, VIX
    alignment_info = calculate_alignment(today_date)
    alignment_score = alignment_info["alignment_score"]

    # 10. Option Strategy Mapper
    strategy_info = map_cycle_strategy(
        regime,
        direction,
        phase_pct,
        vol_status,
        cal_suitability,
        event_risk,
        trade_size_mod,
        vix_pct
    )

    # 11. Final Cycle Opportunity Score Calculation
    # Final Cycle Score = 0.20*strength + 0.15*phase + 0.15*turning + 0.15*alignment + 0.15*vol + 0.10*liq + 0.10*regime
    config = load_config()
    weights = config["final_score_weights"]
    
    # Calculate Phase Quality
    if 0.0 <= phase_pct <= 35.0 or 55.0 <= phase_pct <= 75.0:
        phase_quality = 100.0
    elif 40.0 <= phase_pct <= 55.0 or phase_pct >= 90.0:
        phase_quality = 60.0
    else:
        phase_quality = 30.0

    turn_confidence = (turn_info["peak_confidence"] + turn_info["bottom_confidence"]) / 2.0
    liq_score = 100.0 - event_risk

    final_score = (
        weights["dominant_cycle_strength"] * cycle_strength +
        weights["phase_quality_score"] * phase_quality +
        weights["turning_point_confidence"] * turn_confidence +
        weights["alignment_score"] * alignment_score +
        weights["volatility_suitability"] * cal_suitability +
        weights["liquidity_score"] * liq_score +
        weights["regime_score"] * regime_score
    )

    # Classification
    # 90 to 100 = Deploy
    # 80 to 89 = Strong Watch / Small Deploy
    # 70 to 79 = Watch
    # 60 to 69 = Weak Watch
    # Below 60 = Reject
    if final_score >= 90.0:
        decision = "Deploy"
    elif final_score >= 80.0:
        decision = "Strong Watch / Small Deploy"
    elif final_score >= 70.0:
        decision = "Watch"
    elif final_score >= 60.0:
        decision = "Weak Watch"
    else:
        decision = "Reject"

    # Save to database record
    db_record = {
        "signal_date": today_date.strftime("%Y-%m-%d"),
        "ticker": ticker,
        "timeframe": timeframe,
        "dominant_cycle_days": dom_cycle,
        "cycle_strength": cycle_strength,
        "cycle_phase_pct": phase_pct,
        "cycle_direction": direction,
        "next_peak_date": turn_info["next_peak_date"],
        "next_bottom_date": turn_info["next_bottom_date"],
        "peak_confidence": turn_info["peak_confidence"],
        "bottom_confidence": turn_info["bottom_confidence"],
        "alignment_score": alignment_score,
        "volatility_cycle_status": vol_status,
        "liquidity_cycle_status": liq_info["liquidity_cycle_status"],
        "regime": regime,
        "recommended_strategy": strategy_info["recommended_strategy"],
        "confidence_score": round(final_score, 1),
        "reason_code": strategy_info["reason_code"]
    }
    
    # Save database record and attach database primary key ID
    signal_id = save_signal(db_record)
    
    # Pack result dictionary for API output
    result = {
        **db_record,
        "id": signal_id,
        "as_of_date": today_date.strftime("%Y-%m-%d"),
        "phase_label": phase_label,
        "decision": decision,
        "current_price": current_price,
        "estimated_days_to_bottom": days_to_bottom,
        "estimated_days_to_peak": days_to_peak,
        "next_peak_window": turn_info["next_peak_window"],
        "next_bottom_window": turn_info["next_bottom_window"],
        "alignment_state": alignment_info["alignment_state"],
        "vix_value": vol_info["vix_value"],
        "vix_percentile": vix_pct,
        "vvix_value": vol_info["vvix_value"],
        "vvix_warning": vol_info["vvix_warning"],
        "term_structure": vol_info["term_structure"],
        "calendar_suitability": cal_suitability,
        "event_risk_score": event_risk,
        "nearest_event": liq_info["nearest_event"],
        "days_to_event": liq_info["days_to_event"],
        "trade_size_modifier": trade_size_mod,
        "support": support,
        "resistance": resistance,
        "historical_prices_df": df,
        "methods_breakdown": cycle_info["methods"]
    }
    
    return result
