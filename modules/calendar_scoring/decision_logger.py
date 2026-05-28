from datetime import datetime
import logging
from modules.calendar_scoring.database import insert_decision_log, insert_option_setup, get_connection

logger = logging.getLogger("CalendarDecisionLogger")

def log_daily_run(run_date: str, candidates: list, model_version: str) -> int:
    """Save all candidates (including filtered-out items) into the SQLite database.
    
    This preserves the complete daily state required to run backtests later.
    Clears any existing runs for the same date to prevent duplicate records.
    """
    try:
        # Clear existing logs for this run_date to maintain unique tickers per date
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT decision_id FROM ticker_decision_log WHERE decision_date = ?", (run_date,))
        decision_ids = [row[0] for row in cursor.fetchall()]
        
        if decision_ids:
            placeholders = ",".join(["?"] * len(decision_ids))
            cursor.execute(f"DELETE FROM option_trade_setup_log WHERE decision_id IN ({placeholders})", decision_ids)
            cursor.execute(f"DELETE FROM decision_outcome_log WHERE decision_id IN ({placeholders})", decision_ids)
            cursor.execute("DELETE FROM ticker_decision_log WHERE decision_date = ?", (run_date,))
            conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed clearing existing records for date {run_date}: {e}")

    logged_count = 0
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for rank_idx, cand in enumerate(candidates):
        try:
            # Map recommendation values
            rec = cand.get("recommendation", "Avoid")
            
            decision_data = {
                "decision_datetime": now_str,
                "decision_date": run_date,
                "ticker": cand["ticker"],
                "strategy_type": cand.get("strategy_type", "Bullish Calendar Spread"),
                "recommendation": rec,
                "rank_today": rank_idx + 1 if rec != "Filtered" else 999,
                "final_score": float(cand.get("final_score", 0.0)) if rec != "Filtered" else 0.0,
                
                "market_regime": cand.get("market_regime", "Bull Trend"),
                "fdts_signal": cand.get("fdts_signal", "Neutral"),
                "fdts_score": float(cand.get("fdts_score", 50.0)),
                
                "trend_score": float(cand.get("trend_score", 0.0)),
                "option_structure_score": float(cand.get("option_structure_score", 0.0)),
                "volatility_score": float(cand.get("volatility_score", 0.0)),
                "pca_score": float(cand.get("pca_score", 0.0)),
                "cluster_score": float(cand.get("cluster_score", 0.0)),
                "leading_lagging_score": float(cand.get("leading_lagging_score", 0.0)),
                "liquidity_score": float(cand.get("liquidity_score", 0.0)),
                "event_risk_score": float(cand.get("event_risk_score", 0.0)),
                "institutional_flow_score": float(cand.get("institutional_flow_score", 0.0)),
                
                "cluster_label": cand.get("cluster_label", "Early Trend"),
                "leading_lagging_state": cand.get("leading_lagging_state", "Leading"),
                
                "price_at_decision": float(cand.get("spot_price", 0.0)),
                "atr_14": float(cand.get("atr_14", 0.0)),
                "rsi_14": float(cand.get("rsi_14", 0.0)),
                "adx_14": float(cand.get("adx_14", 0.0)),
                "ema_20": float(cand.get("ema_20", 0.0)),
                "ema_50": float(cand.get("ema_50", 0.0)),
                "ema_200": float(cand.get("ema_200", 0.0)),
                
                "iv_rank": float(cand.get("iv_rank", 0.0)),
                "iv_percentile": float(cand.get("iv_percentile", 0.0)),
                "front_iv": float(cand.get("front_iv", 0.0)),
                "back_iv": float(cand.get("back_iv", 0.0)),
                "iv_term_structure": float(cand.get("back_iv", 0.0) - cand.get("front_iv", 0.0)),
                
                "avg_option_volume": float(cand.get("avg_option_volume", 0.0)),
                "avg_open_interest": float(cand.get("avg_open_interest", 0.0)),
                "bid_ask_spread_pct": float(cand.get("bid_ask_spread_pct", 0.0)),
                
                "earnings_date": cand.get("earnings_date"),
                "event_risk_flag": int(cand.get("event_risk_flag", 0)),
                
                "reason_summary": cand.get("reason_summary", ""),
                "model_version": model_version,
                "ml_predicted_return": cand.get("ml_predicted_return")
            }
            
            decision_id = insert_decision_log(decision_data)
            
            # Save trade setup snapshot if present
            setup = cand.get("option_setup")
            if setup and decision_id:
                setup_data = {
                    "decision_id": decision_id,
                    "ticker": cand["ticker"],
                    "strategy_type": cand.get("strategy_type", "Bullish Calendar Spread"),
                    "short_dte": int(setup.get("short_dte", 20)),
                    "long_dte": int(setup.get("long_dte", 40)),
                    "target_delta": float(setup.get("target_delta", 0.25)),
                    "short_expiry": setup.get("short_expiry"),
                    "long_expiry": setup.get("long_expiry"),
                    "selected_strike": float(setup.get("selected_strike", 0.0)),
                    "short_bid": float(setup.get("short_bid", 0.0)),
                    "short_ask": float(setup.get("short_ask", 0.0)),
                    "short_mid": float(setup.get("short_mid", 0.0)),
                    "long_bid": float(setup.get("long_bid", 0.0)),
                    "long_ask": float(setup.get("long_ask", 0.0)),
                    "long_mid": float(setup.get("long_mid", 0.0)),
                    "net_debit": float(setup.get("net_debit", 0.0)),
                    "max_risk": float(setup.get("max_risk", 0.0)),
                    "setup_delta": float(setup.get("setup_delta", 0.0)),
                    "setup_gamma": float(setup.get("setup_gamma", 0.0)),
                    "setup_theta": float(setup.get("setup_theta", 0.0)),
                    "setup_vega": float(setup.get("setup_vega", 0.0)),
                    "breakeven_low": float(setup.get("breakeven_low", 0.0)),
                    "breakeven_high": float(setup.get("breakeven_high", 0.0))
                }
                insert_option_setup(setup_data)
                
            logged_count += 1
        except Exception as e:
            logger.error(f"Error logging candidate {cand.get('ticker')}: {e}")
            
    return logged_count
