import sys
import os

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import logging
from modules.calendar_scoring.database import create_tables, get_active_model_weights, save_model_weights
from modules.calendar_scoring.data_loader import fetch_technical_data, fetch_option_chain_data
from modules.calendar_scoring.market_regime import detect_market_regime
from modules.calendar_scoring.fdts_engine import calculate_fdts_signal
from modules.calendar_scoring.scoring_engine import (
    calculate_trend_score, calculate_option_structure_score, calculate_volatility_score,
    calculate_fdts_score, calculate_pca_score, calculate_cluster_score,
    calculate_leading_lagging_score, calculate_liquidity_score, calculate_event_risk_score,
    apply_hard_filters
)
from modules.calendar_scoring.trade_setup_engine import select_calendar_setup
from modules.calendar_scoring.decision_logger import log_daily_run
from modules.calendar_scoring.outcome_tracker import update_decision_outcomes
from modules.calendar_scoring.backtest_engine import run_backtest_analysis
from modules.calendar_scoring.optimization_engine import run_weights_optimization

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CalendarEngineVerifier")

def run_verification_tests():
    logger.info("Starting Calendar Opportunity Scoring Engine Verification...")
    
    # 1. Database table creation
    logger.info("Step 1: Initializing database and tables...")
    create_tables()
    
    # 2. Get active model weights
    logger.info("Step 2: Testing weights manager...")
    weights = get_active_model_weights()
    logger.info(f"Loaded weights: {weights}")
    save_model_weights(weights)
    logger.info("Saved weights successfully.")
    
    # 3. Regime Detector
    logger.info("Step 3: Testing Market Regime Detector...")
    regime = detect_market_regime("SPY")
    logger.info(f"Current detected regime: {regime['regime']} (VIX: {regime['vix_value']})")
    
    # 4. Data loading & Indicator checks
    logger.info("Step 4: Fetching technicals and option chains...")
    tech = fetch_technical_data("AAPL")
    logger.info(f"AAPL Spot Price: ${tech['spot_price']:.2f}, RSI: {tech['rsi_14']:.1f}, ADX: {tech['adx_14']:.1f}")
    
    options = fetch_option_chain_data("AAPL", tech["spot_price"], use_synthetic=True)
    logger.info(f"AAPL Expirations: Front {options['short_expiry']} (DTE {options['short_dte']}), Back {options['long_expiry']} (DTE {options['long_dte']})")
    
    # 5. Setup selections
    logger.info("Step 5: Formulating calendar legs...")
    setup = select_calendar_setup("AAPL", options, tech["spot_price"], tech["hv_30"])
    logger.info(f"Optimal AAPL setup: Strike ${setup['selected_strike']:.2f}, Net Debit: ${setup['net_debit']:.2f}")
    
    # 6. Scoring Calculations
    logger.info("Step 6: Executing indicator scoring math...")
    fdts = calculate_fdts_signal(tech["spot_price"], tech["ema_20"], tech["ema_50"], tech["ema_200"], tech["rsi_14"])
    trend_s = calculate_trend_score(tech["spot_price"], tech["ema_20"], tech["ema_50"], tech["ema_200"], tech["adx_14"])
    opt_s = calculate_option_structure_score(setup["front_iv"], setup["back_iv"])
    vol_s = calculate_volatility_score(45.0, 42.0)
    fdts_s = calculate_fdts_score(fdts["score"])
    pca_s = calculate_pca_score("AAPL", tech["df_history"])
    clus_s, clus_lbl = calculate_cluster_score("AAPL", tech["df_history"])
    lead_s, lead_state = calculate_leading_lagging_score("AAPL", tech["df_history"])
    liq_s = calculate_liquidity_score(setup["bid_ask_spread_pct"], setup["avg_option_volume"])
    evt_s, _ = calculate_event_risk_score("2026-06-15", setup["short_dte"])
    
    logger.info(f"Sub-scores calculated successfully. Cluster: {clus_lbl}, Lead/Lag: {lead_state}")
    
    # 7. Apply hard filters
    logger.info("Step 7: Evaluating hard filters...")
    exclusions = apply_hard_filters("AAPL", tech, setup, fdts["signal"])
    logger.info(f"Hard filter exclusions: {exclusions}")
    
    # 8. Logger
    logger.info("Step 8: Testing decision logger...")
    candidate = {
        "ticker": "AAPL",
        "final_score": 88.5,
        "recommendation": "Deploy" if not exclusions else "Filtered",
        "fdts_signal": fdts["signal"],
        "fdts_score": fdts["score"],
        "trend_score": trend_s,
        "option_structure_score": opt_s,
        "volatility_score": vol_s,
        "pca_score": pca_s,
        "cluster_score": clus_s,
        "cluster_label": clus_lbl,
        "leading_lagging_score": lead_s,
        "leading_lagging_state": lead_state,
        "liquidity_score": liq_s,
        "event_risk_score": evt_s,
        "spot_price": tech["spot_price"],
        "ema_20": tech["ema_20"],
        "ema_50": tech["ema_50"],
        "ema_200": tech["ema_200"],
        "rsi_14": tech["rsi_14"],
        "adx_14": tech["adx_14"],
        "atr_14": tech["atr_14"],
        "iv_rank": 45.0,
        "iv_percentile": 42.0,
        "avg_option_volume": setup["avg_option_volume"],
        "avg_open_interest": setup["avg_open_interest"],
        "bid_ask_spread_pct": setup["bid_ask_spread_pct"],
        "front_iv": setup["front_iv"],
        "back_iv": setup["back_iv"],
        "earnings_date": "2026-06-15",
        "event_risk_flag": 0,
        "market_regime": regime["regime"],
        "reason_summary": ", ".join(exclusions) if exclusions else "Meets all criteria",
        "option_setup": setup
    }
    
    count = log_daily_run("2026-05-28", [candidate], "MVP v2.02")
    logger.info(f"Decision logger completed. Logged {count} entries.")
    
    # 9. Outcomes
    logger.info("Step 9: Running outcomes tracking update...")
    outcomes = update_decision_outcomes()
    logger.info(f"Outcomes checked. Processed: {outcomes}")
    
    # 10. Backtests & Optimizations
    logger.info("Step 10: Executing analytics backtesting & weights grid search...")
    bt_res = run_backtest_analysis()
    logger.info("Backtest engine run complete.")
    
    opt_res = run_weights_optimization()
    logger.info("Optimization engine run complete.")
    
    logger.info("🎉 All verification tests completed SUCCESSFULY without errors!")

if __name__ == "__main__":
    run_verification_tests()
