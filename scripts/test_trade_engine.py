import sys
import os
import logging
from datetime import datetime

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TRADE_RECOMMENDATION_DB_PATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "trade_recommendation_test.sqlite"))

from modules.trade_recommendation.database import (
    create_tables, save_signal_snapshot, log_trade_outcome,
    fetch_historical_snapshots, fetch_active_trade_plans, fetch_trade_outcomes
)
from modules.trade_recommendation.indicators import (
    run_indicators_scan, run_trend_engine, run_regime_engine, run_forecast_engine
)
from modules.trade_recommendation.selector import (
    calculate_component_scores, calculate_composite_score,
    select_trade_strategy, adjust_decision_for_alignment
)
from modules.trade_recommendation.plan_generator import generate_trade_plan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestTradeEngine")

def run_tests():
    logger.info("Starting Ticker Trade Intelligence Engine Automated Verification Tests...")
    
    # 1. Database Table Creation
    logger.info("Step 1: Initializing database and tables...")
    create_tables()
    
    # 2. Mock indicators scan
    logger.info("Step 2: Testing indicators calculations with mock data...")
    import pandas as pd
    import numpy as np
    
    # Create 250 days of mock prices
    dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
    np.random.seed(42)
    closes = 100.0 + np.cumsum(np.random.normal(0.5, 1.5, 250)) # upward trend
    highs = closes + np.random.uniform(0.5, 3.0, 250)
    lows = closes - np.random.uniform(0.5, 3.0, 250)
    opens = closes + np.random.uniform(-1.0, 1.0, 250)
    volumes = np.random.randint(10000, 50000, 250)
    
    df_mock = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes
    }, index=dates)
    
    daily_ind = run_indicators_scan(df_mock, "Daily")
    assert daily_ind is not None, "Daily indicator scan returned None"
    logger.info(f"Daily indicators computed successfully. Spot Price: ${daily_ind['price']:.2f}")
    
    # 3. Trend and Regime Engines
    logger.info("Step 3: Evaluating Trend and Regime Engines...")
    trend = run_trend_engine(daily_ind)
    regime = run_regime_engine(df_mock, 15.0)
    logger.info(f"Trend state: {trend} | Regime state: {regime}")
    
    # 4. Forecast Engine
    logger.info("Step 4: Running 40-Day Forecast Engine...")
    path, low, high = run_forecast_engine(daily_ind["price"], 0.25, daily_ind, trend, regime)
    logger.info(f"Forecast Path: {path} | Expected Low: ${low:.2f} | Expected High: ${high:.2f}")
    
    # 5. Scoring and Strategy Matrix
    logger.info("Step 5: Running scoring model & strategy selector matrix...")
    scores = calculate_component_scores(daily_ind, 15.0)
    comp_score = calculate_composite_score(scores)
    strategy, reason = select_trade_strategy(scores, daily_ind, path)
    decision, adj_reason = adjust_decision_for_alignment("Deploy", strategy, daily_ind)
    logger.info(f"Composite Score: {comp_score:.1f} | Decision: {decision} | Strategy: {strategy}")
    
    # 6. Trade Plan Generator
    logger.info("Step 6: Generating trade plan...")
    plan = generate_trade_plan(
        "MOCK", daily_ind["price"], "Deploy", strategy, trend, low, high, daily_ind
    )
    assert plan is not None, "Trade plan generation failed"
    logger.info(f"Trade Plan recommended structure: {plan['option_structure']}")
    
    # 7. Database Logging
    logger.info("Step 7: Testing database logging of snapshots and plans...")
    snapshot = {
        "ticker": "MOCK",
        "snapshot_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "price": daily_ind["price"],
        "trend_state": trend,
        "market_state": regime,
        "trade_decision": "Deploy",
        "recommended_strategy": strategy,
        "trade_score": comp_score,
        "trend_score": scores["trend_score"],
        "momentum_score": scores["momentum_score"],
        "range_score": scores["range_score"],
        "volatility_score": scores["volatility_score"],
        "liquidity_score": scores["liquidity_score"],
        "event_risk_score": scores["event_risk_score"],
        "expected_40d_path": path,
        "expected_40d_low": low,
        "expected_40d_high": high,
        "support_level": daily_ind["darvas_lower"],
        "resistance_level": daily_ind["darvas_upper"],
        "trigger_level": daily_ind["darvas_upper"],
        "invalidation_level": daily_ind["ma50"],
        "notes": reason
    }
    
    indicators = [
        {
            "ticker": "MOCK",
            "timeframe": "Daily",
            "price": daily_ind["price"],
            **daily_ind
        }
    ]
    
    snapshot_id = save_signal_snapshot(snapshot, indicators, plan)
    logger.info(f"Signal snapshot saved. Snapshot ID: {snapshot_id}")
    
    # Verify retrieval
    snaps = fetch_historical_snapshots()
    assert len(snaps) > 0, "No snapshots retrieved from database"
    logger.info("Snapshot verified on read-back.")
    
    # 8. Outcomes Logging
    logger.info("Step 8: Testing trade outcome logging...")
    active_plans = fetch_active_trade_plans()
    assert len(active_plans) > 0, "No active trade plans retrieved"
    
    plan_id = active_plans[0]["trade_plan_id"]
    outcome = {
        "trade_plan_id": plan_id,
        "ticker": "MOCK",
        "entry_date": datetime.now().strftime("%Y-%m-%d"),
        "review_date": datetime.now().strftime("%Y-%m-%d"),
        "price_at_signal": daily_ind["price"],
        "price_after_5d": daily_ind["price"] * 1.02,
        "price_after_10d": daily_ind["price"] * 1.04,
        "price_after_20d": daily_ind["price"] * 1.05,
        "price_after_40d": daily_ind["price"] * 1.07,
        "max_favorable_move": daily_ind["price"] * 0.08,
        "max_adverse_move": daily_ind["price"] * 0.01,
        "strategy_result": "Profit",
        "estimated_pnl": 500.0,
        "notes": "Verified mock outcome"
    }
    
    outcome_id = log_trade_outcome(outcome)
    logger.info(f"Outcome logged. Outcome ID: {outcome_id}")
    
    outcomes = fetch_trade_outcomes()
    assert len(outcomes) > 0, "No outcomes retrieved from database"
    logger.info("Outcome log verified on read-back.")
    
    logger.info("🎉 All verification tests completed SUCCESSFULY without errors!")

if __name__ == "__main__":
    run_tests()
