import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import logging
from modules.tier4.volatility_engine import get_earnings_date as _get_real_earnings_date

from modules.base_module import FazDaneModule
from utils.universe_manager import render_universe_manager, get_tickers, format_ticker_display, get_ticker_names

# Import engine components
from modules.calendar_scoring.config import STRATEGY_CONFIG, STRATEGY_CONFIG as STR_CONF
from modules.calendar_scoring.database import get_active_model_weights, save_model_weights, get_connection
# data_loader imports are consolidated below with the scoring engine imports
from modules.calendar_scoring.market_regime import detect_market_regime, get_transition_matrices
from modules.calendar_scoring.fdts_engine import calculate_fdts_signal
from modules.calendar_scoring.trade_setup_engine import select_calendar_setup
from modules.calendar_scoring.scoring_engine import (
    calculate_trend_score, calculate_option_structure_score, calculate_volatility_score,
    calculate_fdts_score, calculate_pca_score, calculate_cluster_score,
    calculate_leading_lagging_score, calculate_liquidity_score, calculate_event_risk_score,
    calculate_institutional_flow_score, apply_hard_filters
)
from modules.calendar_scoring.data_loader import (
    fetch_technical_data, fetch_option_chain_data, black_scholes_call, fetch_benchmark_data
)
from modules.calendar_scoring.decision_logger import log_daily_run
from modules.calendar_scoring.outcome_tracker import update_decision_outcomes
from modules.calendar_scoring.backtest_engine import run_backtest_analysis
from modules.calendar_scoring.optimization_engine import run_weights_optimization
from modules.calendar_scoring.explanation_engine import generate_llm_prompt
from modules.calendar_scoring.meta_ranking import predict_meta_rankings, train_meta_ranking_model
from modules.calendar_scoring.bayesian_updating import fetch_intraday_data, calculate_bayesian_posterior
from modules.calendar_scoring.survival_analysis import analyze_trend_survival, analyze_trend_survival_with_df

logger = logging.getLogger("CalendarDashboard")

class CalendarOpportunityScoringModule(FazDaneModule):
    """Streamlit dashboard module for the Calendar Opportunity Scoring Engine."""
    
    MODULE_NAME = "Calendar Opportunity Scoring Engine"
    MODULE_ICON = "📅"
    MODULE_DESCRIPTION = "Rank, score, and track bullish options calendar spread candidates."
    TIER = 2
    SOURCE_NOTEBOOK = "CalendarScoring"
    
    def __init__(self):
        super().__init__()
        # Initialise session state containers
        if "cal_candidates" not in st.session_state:
            st.session_state.cal_candidates = []
        if "cal_last_run" not in st.session_state:
            st.session_state.cal_last_run = None
            
        if "cal_enable_ml_sort" not in st.session_state:
            st.session_state.cal_enable_ml_sort = False
            
        # Auto-load latest results on initialization
        if not st.session_state.cal_candidates:
            loaded = self.load_latest_run_from_db()
            if not loaded:
                self.prepopulate_db_with_mock_run()
                self.load_latest_run_from_db()
                
    def _adjust_weights_by_hmm(self, weights: dict, hmm_transitions: dict) -> dict:
        """Dynamically adjust factor weights based on HMM state transition probabilities."""
        adjusted = weights.copy()
        risk_off_prob = hmm_transitions.get("Risk-Off", 0.0)
        bull_prob = hmm_transitions.get("Bull Trend", 0.0)
        
        total_keys = [
            "trend_weight", "option_structure_weight", "volatility_weight", "fdts_weight",
            "pca_weight", "cluster_weight", "leading_lagging_weight", "liquidity_weight",
            "event_risk_weight", "institutional_flow_weight"
        ]
        
        # Normalize inputs
        total_w = sum(adjusted.get(k, 0.0) for k in total_keys)
        if total_w > 0:
            for k in total_keys:
                adjusted[k] = adjusted.get(k, 0.0) / total_w
                
        if risk_off_prob > 0.35:
            # Volatility & Liquidity become more important, Trend/FDTS less important
            shift = (risk_off_prob - 0.20) * 0.4
            
            dec_keys = ["trend_weight", "fdts_weight", "pca_weight"]
            total_dec = sum(adjusted.get(k, 0.0) for k in dec_keys)
            
            actual_dec = min(shift, total_dec * 0.5)
            if total_dec > 0:
                for k in dec_keys:
                    adjusted[k] -= (adjusted[k] / total_dec) * actual_dec
                    
            inc_keys = ["volatility_weight", "liquidity_weight", "event_risk_weight"]
            total_inc = sum(adjusted.get(k, 0.0) for k in inc_keys)
            if total_inc > 0:
                for k in inc_keys:
                    adjusted[k] += (adjusted[k] / total_inc) * actual_dec
            else:
                for k in inc_keys:
                    adjusted[k] += actual_dec / len(inc_keys)
                    
        elif bull_prob > 0.70:
            # Momentum factors become more important
            shift = (bull_prob - 0.60) * 0.3
            
            dec_keys = ["volatility_weight", "liquidity_weight"]
            total_dec = sum(adjusted.get(k, 0.0) for k in dec_keys)
            
            actual_dec = min(shift, total_dec * 0.4)
            if total_dec > 0:
                for k in dec_keys:
                    adjusted[k] -= (adjusted[k] / total_dec) * actual_dec
                    
            inc_keys = ["trend_weight", "fdts_weight", "pca_weight"]
            total_inc = sum(adjusted.get(k, 0.0) for k in inc_keys)
            if total_inc > 0:
                for k in inc_keys:
                    adjusted[k] += (adjusted[k] / total_inc) * actual_dec
                    
        # Normalize to ensure exactly 1.0 sum
        total_w = sum(adjusted.get(k, 0.0) for k in total_keys)
        if total_w > 0:
            for k in total_keys:
                adjusted[k] = round(adjusted[k] / total_w, 4)
                
        return adjusted
                
    def load_latest_run_from_db(self) -> bool:
        """Query the SQLite database to fetch and populate the candidates list from the most recent run date."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Find the most recent decision date
            cursor.execute("SELECT DISTINCT decision_date FROM ticker_decision_log ORDER BY decision_date DESC LIMIT 1")
            row = cursor.fetchone()
            if not row:
                conn.close()
                return False
                
            latest_date = row[0]
            
            # Fetch all decisions for that date
            df_decisions = pd.read_sql_query("""
                SELECT * FROM ticker_decision_log WHERE decision_date = ?
            """, conn, params=(latest_date,))
            
            # Fetch all setups for that date
            df_setups = pd.read_sql_query("""
                SELECT s.* FROM option_trade_setup_log s
                JOIN ticker_decision_log d ON s.decision_id = d.decision_id
                WHERE d.decision_date = ?
            """, conn, params=(latest_date,))
            
            conn.close()
            
            if df_decisions.empty or len(df_decisions) < 3:
                return False
                
            # Parse decisions back to dictionary list
            candidates = []
            for _, row_dec in df_decisions.iterrows():
                ticker = row_dec["ticker"]
                dec_id = row_dec["decision_id"]
                
                # Find matching setup
                setup_dict = {}
                matching_setup = df_setups[df_setups["decision_id"] == dec_id]
                if not matching_setup.empty:
                    setup_row = matching_setup.iloc[0]
                    setup_dict = {
                        "short_dte": int(setup_row["short_dte"]),
                        "long_dte": int(setup_row["long_dte"]),
                        "target_delta": float(setup_row["target_delta"]),
                        "short_expiry": setup_row["short_expiry"],
                        "long_expiry": setup_row["long_expiry"],
                        "selected_strike": float(setup_row["selected_strike"]),
                        "short_bid": float(setup_row["short_bid"]),
                        "short_ask": float(setup_row["short_ask"]),
                        "short_mid": float(setup_row["short_mid"]),
                        "long_bid": float(setup_row["long_bid"]),
                        "long_ask": float(setup_row["long_ask"]),
                        "long_mid": float(setup_row["long_mid"]),
                        "net_debit": float(setup_row["net_debit"]),
                        "max_risk": float(setup_row["max_risk"]),
                        "setup_delta": float(setup_row["setup_delta"]),
                        "setup_gamma": float(setup_row["setup_gamma"]),
                        "setup_theta": float(setup_row["setup_theta"]),
                        "setup_vega": float(setup_row["setup_vega"]),
                        "breakeven_low": float(setup_row["breakeven_low"]),
                        "breakeven_high": float(setup_row["breakeven_high"]),
                        "avg_option_volume": float(row_dec.get("avg_option_volume", 0)),
                        "avg_open_interest": float(row_dec.get("avg_open_interest", 0)),
                        "bid_ask_spread_pct": float(row_dec.get("bid_ask_spread_pct", 0)),
                        "front_iv": float(row_dec.get("front_iv", 0)),
                        "back_iv": float(row_dec.get("back_iv", 0))
                    }
                    
                cand_data = {
                    "ticker": ticker,
                    "final_score": float(row_dec["final_score"]),
                    "recommendation": row_dec["recommendation"],
                    "fdts_signal": row_dec["fdts_signal"],
                    "fdts_score": float(row_dec.get("fdts_score", 50.0)),
                    "trend_score": float(row_dec.get("trend_score", 0.0)),
                    "option_structure_score": float(row_dec.get("option_structure_score", 0.0)),
                    "volatility_score": float(row_dec.get("volatility_score", 0.0)),
                    "pca_score": float(row_dec.get("pca_score", 0.0)),
                    "cluster_score": float(row_dec.get("cluster_score", 0.0)),
                    "cluster_label": row_dec.get("cluster_label", "Early Trend"),
                    "leading_lagging_score": float(row_dec.get("leading_lagging_score", 0.0)),
                    "leading_lagging_state": row_dec.get("leading_lagging_state", "Leading"),
                    "liquidity_score": float(row_dec.get("liquidity_score", 0.0)),
                    "event_risk_score": float(row_dec.get("event_risk_score", 0.0)),
                    "spot_price": float(row_dec.get("price_at_decision", 0.0)),
                    "ema_20": float(row_dec.get("ema_20", 0.0)),
                    "ema_50": float(row_dec.get("ema_50", 0.0)),
                    "ema_200": float(row_dec.get("ema_200", 0.0)),
                    "rsi_14": float(row_dec.get("rsi_14", 0.0)),
                    "adx_14": float(row_dec.get("adx_14", 0.0)),
                    "atr_14": float(row_dec.get("atr_14", 0.0)),
                    "iv_rank": float(row_dec.get("iv_rank", 0.0)),
                    "iv_percentile": float(row_dec.get("iv_percentile", 0.0)),
                    "avg_option_volume": float(row_dec.get("avg_option_volume", 0.0)),
                    "avg_open_interest": float(row_dec.get("avg_open_interest", 0.0)),
                    "bid_ask_spread_pct": float(row_dec.get("bid_ask_spread_pct", 0.0)),
                    "front_iv": float(row_dec.get("front_iv", 0.0)),
                    "back_iv": float(row_dec.get("back_iv", 0.0)),
                    "earnings_date": row_dec.get("earnings_date"),
                    "event_risk_flag": int(row_dec.get("event_risk_flag", 0)),
                    "market_regime": row_dec.get("market_regime", "Bull Trend"),
                    "reason_summary": row_dec.get("reason_summary", ""),
                    "ml_predicted_return": float(row_dec["ml_predicted_return"]) if "ml_predicted_return" in row_dec and not pd.isna(row_dec["ml_predicted_return"]) else None,
                    "option_setup": setup_dict
                }
                candidates.append(cand_data)
                
            # Sort candidates
            def sort_key(x):
                rec_order = {"Deploy": 0, "Watch": 1, "Monitor": 2, "Avoid": 3, "Filtered": 4}
                return (rec_order.get(x["recommendation"], 9), -x["final_score"])
                
            st.session_state.cal_candidates = sorted(candidates, key=sort_key)
            st.session_state.cal_last_run = latest_date
            return True
        except Exception as e:
            logger.error(f"Error auto-loading database run: {e}")
            return False

    def prepopulate_db_with_mock_run(self) -> bool:
        """Seed the SQLite database with a realistic daily scan run if it is completely empty."""
        try:
            from modules.calendar_scoring.database import create_tables, insert_decision_log, insert_option_setup, insert_outcome_log, MODEL_VERSION
            
            # Ensure tables exist first
            create_tables()
            
            # Clear old records to prevent duplicate seeding
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM ticker_decision_log")
            cursor.execute("DELETE FROM option_trade_setup_log")
            cursor.execute("DELETE FROM decision_outcome_log")
            conn.commit()
            conn.close()
            
            today_str = date.today().strftime("%Y-%m-%d")
            
            # Seed 7 tickers
            mock_data = [
                {
                    "ticker": "NVDA", "spot_price": 950.0, "final_score": 91.4, "recommendation": "Deploy",
                    "fdts_signal": "Buy", "fdts_score": 94.0, "trend_score": 95.0, "option_structure_score": 90.0,
                    "volatility_score": 88.0, "pca_score": 85.0, "cluster_score": 95.0, "cluster_label": "Early Trend",
                    "leading_lagging_score": 92.0, "leading_lagging_state": "Strong Leader", "liquidity_score": 98.0,
                    "event_risk_score": 95.0, "ema_20": 920.0, "ema_50": 880.0, "ema_200": 780.0, "rsi_14": 62.0,
                    "adx_14": 28.0, "atr_14": 18.50, "iv_rank": 32.0, "iv_percentile": 34.0, "strike": 970.0,
                    "net_debit": 12.50, "short_bid": 15.20, "short_ask": 15.60, "short_mid": 15.40,
                    "long_bid": 27.70, "long_ask": 28.10, "long_mid": 27.90, "short_expiry": "2026-06-16",
                    "long_expiry": "2026-07-06", "setup_theta": 0.0824, "setup_vega": 0.4215, "setup_delta": 0.0521,
                    "setup_gamma": -0.0012, "outcome_pnl_5": 12.0, "outcome_pnl_10": 24.5, "outcome_pnl_20": 38.0
                },
                {
                    "ticker": "AVGO", "spot_price": 1400.0, "final_score": 88.7, "recommendation": "Deploy",
                    "fdts_signal": "Buy", "fdts_score": 90.0, "trend_score": 92.0, "option_structure_score": 88.0,
                    "volatility_score": 86.0, "pca_score": 82.0, "cluster_score": 90.0, "cluster_label": "Mid Trend",
                    "leading_lagging_score": 88.0, "leading_lagging_state": "Leading", "liquidity_score": 95.0,
                    "event_risk_score": 90.0, "ema_20": 1360.0, "ema_50": 1310.0, "ema_200": 1180.0, "rsi_14": 58.0,
                    "adx_14": 25.0, "atr_14": 26.80, "iv_rank": 28.0, "iv_percentile": 30.0, "strike": 1420.0,
                    "net_debit": 18.20, "short_bid": 22.10, "short_ask": 22.70, "short_mid": 22.40,
                    "long_bid": 40.30, "long_ask": 40.90, "long_mid": 40.60, "short_expiry": "2026-06-16",
                    "long_expiry": "2026-07-06", "setup_theta": 0.1142, "setup_vega": 0.5874, "setup_delta": 0.0482,
                    "setup_gamma": -0.0008, "outcome_pnl_5": 8.5, "outcome_pnl_10": 15.2, "outcome_pnl_20": 25.8
                },
                {
                    "ticker": "AAPL", "spot_price": 190.0, "final_score": 82.1, "recommendation": "Watch",
                    "fdts_signal": "Buy", "fdts_score": 87.0, "trend_score": 88.0, "option_structure_score": 82.0,
                    "volatility_score": 78.0, "pca_score": 75.0, "cluster_score": 85.0, "cluster_label": "Early Trend",
                    "leading_lagging_score": 80.0, "leading_lagging_state": "Leading", "liquidity_score": 96.0,
                    "event_risk_score": 95.0, "ema_20": 185.0, "ema_50": 180.0, "ema_200": 172.0, "rsi_14": 54.0,
                    "adx_14": 21.0, "atr_14": 3.10, "iv_rank": 22.0, "iv_percentile": 25.0, "strike": 195.0,
                    "net_debit": 3.10, "short_bid": 2.85, "short_ask": 2.95, "short_mid": 2.90,
                    "long_bid": 5.95, "long_ask": 6.05, "long_mid": 6.00, "short_expiry": "2026-06-16",
                    "long_expiry": "2026-07-06", "setup_theta": 0.0152, "setup_vega": 0.0824, "setup_delta": 0.0384,
                    "setup_gamma": -0.0035, "outcome_pnl_5": 5.4, "outcome_pnl_10": 8.1, "outcome_pnl_20": 14.5
                },
                {
                    "ticker": "MSFT", "spot_price": 420.0, "final_score": 78.4, "recommendation": "Watch",
                    "fdts_signal": "Neutral", "fdts_score": 78.0, "trend_score": 80.0, "option_structure_score": 76.0,
                    "volatility_score": 82.0, "pca_score": 78.0, "cluster_score": 75.0, "cluster_label": "Consolidating",
                    "leading_lagging_score": 75.0, "leading_lagging_state": "Leading", "liquidity_score": 97.0,
                    "event_risk_score": 95.0, "ema_20": 418.0, "ema_50": 412.0, "ema_200": 390.0, "rsi_14": 51.0,
                    "adx_14": 18.0, "atr_14": 6.80, "iv_rank": 18.0, "iv_percentile": 20.0, "strike": 425.0,
                    "net_debit": 6.50, "short_bid": 6.10, "short_ask": 6.30, "short_mid": 6.20,
                    "long_bid": 12.60, "long_ask": 12.80, "long_mid": 12.70, "short_expiry": "2026-06-16",
                    "long_expiry": "2026-07-06", "setup_theta": 0.0315, "setup_vega": 0.1742, "setup_delta": 0.0412,
                    "setup_gamma": -0.0018, "outcome_pnl_5": 2.1, "outcome_pnl_10": 5.6, "outcome_pnl_20": 10.5
                },
                {
                    "ticker": "SPY", "spot_price": 510.0, "final_score": 73.1, "recommendation": "Monitor",
                    "fdts_signal": "Buy", "fdts_score": 74.0, "trend_score": 75.0, "option_structure_score": 72.0,
                    "volatility_score": 70.0, "pca_score": 70.0, "cluster_score": 72.0, "cluster_label": "Mid Trend",
                    "leading_lagging_score": 70.0, "leading_lagging_state": "Leading", "liquidity_score": 99.0,
                    "event_risk_score": 98.0, "ema_20": 505.0, "ema_50": 498.0, "ema_200": 475.0, "rsi_14": 56.0,
                    "adx_14": 20.0, "atr_14": 4.50, "iv_rank": 14.0, "iv_percentile": 16.0, "strike": 515.0,
                    "net_debit": 4.20, "short_bid": 3.95, "short_ask": 4.05, "short_mid": 4.00,
                    "long_bid": 8.15, "long_ask": 8.25, "long_mid": 8.20, "short_expiry": "2026-06-16",
                    "long_expiry": "2026-07-06", "setup_theta": 0.0242, "setup_vega": 0.1385, "setup_delta": 0.0354,
                    "setup_gamma": -0.0021, "outcome_pnl_5": 1.2, "outcome_pnl_10": 4.1, "outcome_pnl_20": 8.5
                },
                {
                    "ticker": "QQQ", "spot_price": 430.0, "final_score": 71.5, "recommendation": "Monitor",
                    "fdts_signal": "Buy", "fdts_score": 72.0, "trend_score": 72.0, "option_structure_score": 70.0,
                    "volatility_score": 68.0, "pca_score": 68.0, "cluster_score": 70.0, "cluster_label": "Mid Trend",
                    "leading_lagging_score": 72.0, "leading_lagging_state": "Leading", "liquidity_score": 98.0,
                    "event_risk_score": 95.0, "ema_20": 425.0, "ema_50": 418.0, "ema_200": 395.0, "rsi_14": 55.0,
                    "adx_14": 19.0, "atr_14": 5.10, "iv_rank": 16.0, "iv_percentile": 18.0, "strike": 435.0,
                    "net_debit": 5.10, "short_bid": 4.85, "short_ask": 4.95, "short_mid": 4.90,
                    "long_bid": 9.95, "long_ask": 10.05, "long_mid": 10.00, "short_expiry": "2026-06-16",
                    "long_expiry": "2026-07-06", "setup_theta": 0.0284, "setup_vega": 0.1584, "setup_delta": 0.0381,
                    "setup_gamma": -0.0024, "outcome_pnl_5": 0.8, "outcome_pnl_10": 3.2, "outcome_pnl_20": 9.1
                },
                {
                    "ticker": "TSLA", "spot_price": 175.0, "final_score": 54.2, "recommendation": "Avoid",
                    "fdts_signal": "Sell", "fdts_score": 35.0, "trend_score": 38.0, "option_structure_score": 55.0,
                    "volatility_score": 62.0, "pca_score": 45.0, "cluster_score": 58.0, "cluster_label": "Consolidating",
                    "leading_lagging_score": 48.0, "leading_lagging_state": "Strong Lagger", "liquidity_score": 95.0,
                    "event_risk_score": 90.0, "ema_20": 182.0, "ema_50": 190.0, "ema_200": 210.0, "rsi_14": 38.0,
                    "adx_14": 25.0, "atr_14": 7.40, "iv_rank": 48.0, "iv_percentile": 52.0, "strike": 180.0,
                    "net_debit": 2.80, "short_bid": 2.55, "short_ask": 2.65, "short_mid": 2.60,
                    "long_bid": 5.35, "long_ask": 5.45, "long_mid": 5.40, "short_expiry": "2026-06-16",
                    "long_expiry": "2026-07-06", "setup_theta": 0.0125, "setup_vega": 0.0712, "setup_delta": 0.0298,
                    "setup_gamma": -0.0041, "outcome_pnl_5": -4.2, "outcome_pnl_10": -8.5, "outcome_pnl_20": -15.4
                }
            ]
            
            # Try to fetch real earnings dates for mock tickers
            _mock_earnings_dates = {}
            for _mt in ["NVDA", "AVGO", "AAPL", "MSFT", "SPY", "QQQ", "TSLA"]:
                try:
                    _red = _get_real_earnings_date(_mt)
                    if _red:
                        _mock_earnings_dates[_mt] = _red.strftime("%Y-%m-%d") if hasattr(_red, "strftime") else str(_red)[:10]
                except Exception:
                    pass

            for rank_idx, item in enumerate(mock_data):
                decision_data = {
                    "decision_datetime": f"{today_str} 09:30:00",
                    "decision_date": today_str,
                    "ticker": item["ticker"],
                    "strategy_type": "Bullish Calendar Spread",
                    "recommendation": item["recommendation"],
                    "rank_today": rank_idx + 1,
                    "final_score": item["final_score"],
                    "market_regime": "Bull Trend",
                    "fdts_signal": item["fdts_signal"],
                    "fdts_score": item["fdts_score"],
                    "trend_score": item["trend_score"],
                    "option_structure_score": item["option_structure_score"],
                    "volatility_score": item["volatility_score"],
                    "pca_score": item["pca_score"],
                    "cluster_score": item["cluster_score"],
                    "leading_lagging_score": item["leading_lagging_score"],
                    "liquidity_score": item["liquidity_score"],
                    "event_risk_score": item["event_risk_score"],
                    "institutional_flow_score": 0.0,
                    "cluster_label": item["cluster_label"],
                    "leading_lagging_state": item["leading_lagging_state"],
                    "price_at_decision": item["spot_price"],
                    "atr_14": item["atr_14"],
                    "rsi_14": item["rsi_14"],
                    "adx_14": item["adx_14"],
                    "ema_20": item["ema_20"],
                    "ema_50": item["ema_50"],
                    "ema_200": item["ema_200"],
                    "iv_rank": item["iv_rank"],
                    "iv_percentile": item["iv_percentile"],
                    "front_iv": 0.28,
                    "back_iv": 0.30,
                    "iv_term_structure": 0.02,
                    "avg_option_volume": 450.0,
                    "avg_open_interest": 2200.0,
                    "bid_ask_spread_pct": 0.015,
                    "earnings_date": _mock_earnings_dates.get(item["ticker"], (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")),
                    "event_risk_flag": 0,
                    "reason_summary": "Meets all criteria",
                    "model_version": MODEL_VERSION
                }
                
                decision_id = insert_decision_log(decision_data)
                
                if decision_id:
                    setup_data = {
                        "decision_id": decision_id,
                        "ticker": item["ticker"],
                        "strategy_type": "Bullish Calendar Spread",
                        "short_dte": 20,
                        "long_dte": 40,
                        "target_delta": 0.25,
                        "short_expiry": item["short_expiry"],
                        "long_expiry": item["long_expiry"],
                        "selected_strike": item["strike"],
                        "short_bid": item["short_bid"],
                        "short_ask": item["short_ask"],
                        "short_mid": item["short_mid"],
                        "long_bid": item["long_bid"],
                        "long_ask": item["long_ask"],
                        "long_mid": item["long_mid"],
                        "net_debit": item["net_debit"],
                        "max_risk": item["net_debit"],
                        "setup_delta": item["setup_delta"],
                        "setup_gamma": item["setup_gamma"],
                        "setup_theta": item["setup_theta"],
                        "setup_vega": item["setup_vega"],
                        "breakeven_low": item["strike"] - item["net_debit"] * 0.85,
                        "breakeven_high": item["strike"] + item["net_debit"] * 1.5
                    }
                    insert_option_setup(setup_data)
                    
                    # Seed Outcomes
                    for day, pnl_pct in [(5, item["outcome_pnl_5"]), (10, item["outcome_pnl_10"]), (20, item["outcome_pnl_20"])]:
                        pnl_amt = item["net_debit"] * (pnl_pct / 100.0)
                        
                        outcome_data = {
                            "decision_id": decision_id,
                            "ticker": item["ticker"],
                            "review_date": (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d"),
                            "review_day": day,
                            "price_at_review": item["spot_price"] * (1.0 + (pnl_pct * 0.003)),
                            "option_value_at_review": item["net_debit"] + pnl_amt,
                            "pnl_amount": pnl_amt,
                            "pnl_pct": pnl_pct,
                            "max_profit_pct": max(0.0, pnl_pct * 1.2),
                            "max_drawdown_pct": min(0.0, pnl_pct * 0.4),
                            "result_label": "Win" if pnl_pct > 10.0 else ("Loss" if pnl_pct < -10.0 else "Neutral"),
                            "exit_signal": "Take Profit Target" if pnl_pct > 10.0 else ("Stop Loss Hit" if pnl_pct < -10.0 else "Hold to Expiration"),
                            "notes": "Pre-seeded system generated mock outcome."
                        }
                        insert_outcome_log(outcome_data)
            return True
        except Exception as e:
            logger.error(f"Error pre-seeding database: {e}")
            return False

    def render_sidebar(self):
        st.write("### Strategy Settings")
        st.write(f"**Strategy**: {STRATEGY_CONFIG['strategy_type']}")
        st.write(f"**Short Leg**: ~{STRATEGY_CONFIG['short_dte_target']} DTE")
        st.write(f"**Long Leg**: ~{STRATEGY_CONFIG['long_dte_target']} DTE")
        st.write(f"**Target Delta**: {STRATEGY_CONFIG['target_delta']} Delta")
        
        st.divider()
        st.write("### Universe Selection")
        # Load Universe Manager widget
        self.universe_name, self.tickers, _ = render_universe_manager(
            key_prefix="cal_scoring_um",
            module_filter="general",
            show_benchmark=False,
            label="Default Universe"
        )
        
        # Override default selection to "Best Option Spread Tickers" if not changed
        if "cal_scoring_um_sel" in st.session_state and st.session_state["cal_scoring_um_sel"] == "Options Default Watchlist":
            # Force "Best Option Spread Tickers" as default
            names = list(st.session_state["cal_scoring_um_sel_options"] if "cal_scoring_um_sel_options" in st.session_state else [])
            if "Best Option Spread Tickers" in names:
                idx = names.index("Best Option Spread Tickers")
                st.session_state["cal_scoring_um_sel"] = "Best Option Spread Tickers"
                st.rerun()
                
        st.divider()
        st.write("### Execution Panel")
        self.use_synthetic_data = st.checkbox(
            "Use Synthetic Option Fallbacks", 
            value=True,
            help="Generate Black-Scholes synthetic options chain if live chain downloads are throttled or closed."
        )
        
        if st.button("Run Scoring Engine", key="btn_run_cal_scoring", type="primary", use_container_width=True):
            self.execute_engine_scan()
            
    def render_main(self):
        # Section Header
        self.render_section_header(
            title="Calendar Opportunity Scoring Engine",
            subtitle="Select the best ticker candidates for bullish calendar spreads based on multi-factor intelligence."
        )
        
        # Market Regime Summary Row
        regime_info = detect_market_regime()
        best_cand_name = "None"
        best_score = 0.0
        
        deploy_count = 0
        if st.session_state.cal_candidates:
            deploys = [c for c in st.session_state.cal_candidates if c.get("recommendation") == "Deploy"]
            deploy_count = len(deploys)
            if deploys:
                best_c = sorted(deploys, key=lambda x: x.get("final_score", 0.0), reverse=True)[0]
                best_cand_name = f"{best_c['ticker']} ({best_c['final_score']:.1f})"
                
        metrics = {
            "Market Regime": (regime_info["regime"], None, ""),
            "VIX Volatility": (regime_info["vix_value"], None, ""),
            "Deploy Candidates": (deploy_count, None, ""),
            "Best Candidate": (best_cand_name, None, "")
        }
        self.render_metrics_row(metrics)
        st.write("")
        
        # ── Tab Navigation (session-state isolated to avoid Streamlit tab index collision)
        _TAB_LABELS = [
            "📋 Top Picks",
            "🗂 All Ranked",
            "🔍 Ticker Detail",
            "📈 Payoff Chart",
            "📂 History Log",
            "🎯 Outcomes",
            "📊 Backtest",
            "⚙️ Weights",
            "⚡ HMM Regime",
            "💼 Paper Trades",
        ]
        if "cal_scoring_active_tab" not in st.session_state:
            st.session_state["cal_scoring_active_tab"] = _TAB_LABELS[0]

        st.markdown(
            """<style>
            div[data-testid="stHorizontalBlock"] .stRadio > div {
                flex-direction: row !important;
                flex-wrap: wrap;
                gap: 4px;
            }
            div[data-testid="stHorizontalBlock"] .stRadio label {
                padding: 6px 14px !important;
                border-radius: 8px !important;
                border: 1px solid var(--border-color, #1e3a5f) !important;
                cursor: pointer !important;
                font-size: 13px !important;
                font-weight: 500 !important;
                background: rgba(21,40,71,0.5) !important;
                color: var(--text-muted, #94a3b8) !important;
                transition: all 0.15s ease !important;
            }
            div[data-testid="stHorizontalBlock"] .stRadio label:hover {
                border-color: var(--accent-color, #3ab54a) !important;
                color: var(--accent-color, #3ab54a) !important;
            }
            </style>""",
            unsafe_allow_html=True,
        )

        active_tab = st.radio(
            "cal_nav",
            options=_TAB_LABELS,
            index=_TAB_LABELS.index(st.session_state.get("cal_scoring_active_tab", _TAB_LABELS[0])),
            horizontal=True,
            key="cal_scoring_active_tab",
            label_visibility="collapsed",
        )

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        if active_tab == _TAB_LABELS[0]:
            self.render_tab_top_picks()
        elif active_tab == _TAB_LABELS[1]:
            self.render_tab_all_ranked()
        elif active_tab == _TAB_LABELS[2]:
            self.render_tab_ticker_detail()
        elif active_tab == _TAB_LABELS[3]:
            self.render_tab_option_setup()
        elif active_tab == _TAB_LABELS[4]:
            self.render_tab_decision_history()
        elif active_tab == _TAB_LABELS[5]:
            self.render_tab_outcome_tracking()
        elif active_tab == _TAB_LABELS[6]:
            self.render_tab_backtest_performance()
        elif active_tab == _TAB_LABELS[7]:
            self.render_tab_weights_optimization()
        elif active_tab == _TAB_LABELS[8]:
            self.render_tab_regime_hmm()
        elif active_tab == _TAB_LABELS[9]:
            self.render_tab_paper_tracker()

    # ══════════════════════════════════════════════════════════════════════
    # SCREEN 1: DAILY TOP PICK SETUPS
    # ══════════════════════════════════════════════════════════════════════
    def render_tab_top_picks(self):
        st.subheader("🏆 Daily Recommended Calendar Setups")
        if not st.session_state.cal_candidates:
            st.info("💡 Run the Scoring Engine from the sidebar to scan candidates.")
            return
            
        top_picks = [c for c in st.session_state.cal_candidates if c.get("recommendation") in ("Deploy", "Watch")]
        if not top_picks:
            st.warning("No tickers met the 'Deploy' or 'Watch' recommendation thresholds (Score >= 75) in the last scan.")
            return
            
        # Fetch Bayesian Intraday updates
        with st.spinner("Updating live intraday signals..."):
            for c in top_picks:
                ticker = c["ticker"]
                intra = fetch_intraday_data(ticker)
                if intra["success"]:
                    post_prob, entry_sig = calculate_bayesian_posterior(
                        c["final_score"],
                        intra["above_vwap"],
                        intra["high_vol"],
                        intra["iv_spike"]
                    )
                else:
                    post_prob, entry_sig = c["final_score"], "Monitor (Neutral)"
                c["bayesian_prob"] = post_prob
                c["bayesian_signal"] = entry_sig
            
        # Format table
        rows = []
        for c in top_picks:
            setup = c.get("option_setup", {})
            ml_ret = c.get("ml_predicted_return")
            ml_ret_str = f"{ml_ret:+.1f}%" if ml_ret is not None else "N/A"
            rows.append({
                "Ticker": c["ticker"],
                "Score": f"{c['final_score']:.1f}",
                "ML Pred PnL": ml_ret_str,
                "Intraday Prob": f"{c.get('bayesian_prob', c['final_score']):.1f}%",
                "Intraday Signal": c.get("bayesian_signal", "Monitor"),
                "Recommendation": c["recommendation"],
                "Earnings Date": c.get("earnings_date", "N/A"),
                "FDTS Signal": c["fdts_signal"],
                "Cluster Label": c["cluster_label"],
                "Strike": f"${setup.get('selected_strike', 0.0):.2f}",
                "Net Debit": f"${setup.get('net_debit', 0.0):.2f}",
                "Reason Summary": c["reason_summary"]
            })
            
        df = pd.DataFrame(rows)
        # Apply color styling
        def color_recommendation(val):
            color_map = {
                "Deploy": "background-color: rgba(58,181,74,0.2); color: #3ab54a; font-weight: bold;",
                "Watch": "background-color: rgba(255,184,0,0.15); color: #ffb800; font-weight: bold;",
                "Monitor": "background-color: rgba(2,132,199,0.15); color: #0284c7;",
                "Avoid": "background-color: rgba(220,38,38,0.15); color: #ef4444;",
                "Filtered": "background-color: rgba(100,116,139,0.15); color: #64748b; text-decoration: line-through;"
            }
            return color_map.get(val, "")
            
        def color_earnings_date(val):
            if not val or val == "N/A":
                return ""
            try:
                earnings_dt = datetime.strptime(val, "%Y-%m-%d").date()
                today_dt = date.today()
                days_diff = (earnings_dt - today_dt).days
                if days_diff < 0:
                    return "color: #ffffff;"
                elif days_diff <= 20:
                    return "background-color: rgba(220,38,38,0.15); color: #ef4444; font-weight: bold;"
                elif days_diff <= 40:
                    return "background-color: rgba(255,184,0,0.15); color: #ffb800; font-weight: bold;"
                else:
                    return "color: #ffffff;"
            except Exception:
                return ""

        def color_cluster_label(val):
            cluster_map = {
                "Early Trend":   "background-color: rgba(58,181,74,0.2); color: #3ab54a; font-weight: bold;",
                "Mid Trend":     "background-color: rgba(255,184,0,0.15); color: #ffb800; font-weight: bold;",
                "Overextended":  "background-color: rgba(220,38,38,0.2); color: #ef4444; font-weight: bold;",
                "Consolidating": "background-color: rgba(2,132,199,0.15); color: #0284c7;",
                "Late Trend":    "background-color: rgba(249,115,22,0.15); color: #f97316;",
            }
            return cluster_map.get(val, "color: #94a3b8;")

        def color_intraday_signal(val):
            if "Green Light" in val:
                return "color: #3ab54a; font-weight: bold;"
            elif "Hold" in val:
                return "color: #ffb800; font-weight: bold;"
            elif "Avoid" in val:
                return "color: #ef4444; font-weight: bold;"
            return "color: #94a3b8;"

        styled_df = (
            df.style
            .map(color_recommendation, subset=["Recommendation"])
            .map(color_earnings_date, subset=["Earnings Date"])
            .map(color_cluster_label, subset=["Cluster Label"])
            .map(color_intraday_signal, subset=["Intraday Signal"])
        )
        st.dataframe(styled_df, use_container_width=True)

        # Select ticker for Prompt Generator
        st.markdown("#### 💬 AI Analyst ChatGPT Prompt Copyable")
        ticker_names = get_ticker_names(getattr(self, "universe_name", "Options Default Watchlist"))
        sel_prompt_ticker = st.selectbox(
            "Select Top Pick for ChatGPT Prompt:",
            options=[c["ticker"] for c in top_picks],
            format_func=lambda t: format_ticker_display(t, ticker_names)
        )
        target_c = next(c for c in top_picks if c["ticker"] == sel_prompt_ticker)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            prompt_txt = generate_llm_prompt(target_c)
            st.code(prompt_txt, language="text")
        with col2:
            self.render_fib_candlestick_chart(sel_prompt_ticker)

    def render_fib_candlestick_chart(self, ticker: str):
        """Render a premium candlestick chart with Fibonacci retracement and extension levels for the selected ticker."""
        try:
            # Fetch historical data
            tech = fetch_technical_data(ticker)
            df = tech.get("df_history")
            
            if df is None or df.empty:
                st.warning("No price history available to draw chart.")
                return
                
            # Use last 90 trading days for anchoring the Fibonacci swing
            df_recent = df.iloc[-90:] if len(df) >= 90 else df
            
            # Find swing low and swing high
            swing_low = df_recent["Low"].min()
            swing_high = df_recent["High"].max()
            diff = swing_high - swing_low
            
            if diff <= 0:
                diff = 1e-5
            
            # Calculate Fibonacci levels
            fib_levels = {
                "Fib 1.618 Extension": swing_high + 0.618 * diff,
                "Fib 1.272 Extension": swing_high + 0.272 * diff,
                "Fib 0.000 (High)": swing_high,
                "Fib 0.382 Retracement": swing_high - 0.382 * diff,
                "Fib 0.500 Retracement": swing_high - 0.500 * diff,
                "Fib 0.618 Retracement": swing_high - 0.618 * diff,
                "Fib 1.000 (Low)": swing_low
            }
            
            # Draw candlestick
            fig = go.Figure(data=[go.Candlestick(
                x=df_recent.index,
                open=df_recent['Open'],
                high=df_recent['High'],
                low=df_recent['Low'],
                close=df_recent['Close'],
                name=f"{ticker} Price"
            )])
            
            # Define colors for fib lines
            colors = {
                "Fib 1.618 Extension": "#ef4444", # Red
                "Fib 1.272 Extension": "#ffb800", # Orange
                "Fib 0.000 (High)": "#3ab54a",    # Green
                "Fib 0.382 Retracement": "#64748b",# Grey
                "Fib 0.500 Retracement": "#64748b",
                "Fib 0.618 Retracement": "#64748b",
                "Fib 1.000 (Low)": "#ef4444"     # Red
            }
            
            # Add horizontal lines for each fib level
            for label, val in fib_levels.items():
                fig.add_hline(
                    y=val,
                    line_dash="dash" if "Retracement" in label else "solid",
                    line_color=colors.get(label, "#64748b"),
                    annotation_text=f"{label}: ${val:.2f}",
                    annotation_position="top right"
                )
                
            fig.update_layout(
                title=f"📈 {ticker} Technical Chart (Last 90 Days & Fibonacci Targets)",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=380,
                margin=dict(l=20, r=20, t=40, b=10),
                xaxis_rangeslider_visible=False,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export and provide download option
            try:
                img_bytes = fig.to_image(format="png", width=1000, height=500)
                st.download_button(
                    label="📥 Download Chart Image (PNG)",
                    data=img_bytes,
                    file_name=f"{ticker}_fib_chart.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception as ex:
                logger.warning(f"Could not generate download image bytes: {ex}")
        except Exception as e:
            st.error(f"Error rendering chart: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # SCREEN 2: ALL RANKED CANDIDATES
    # ══════════════════════════════════════════════════════════════════════
    def render_tab_all_ranked(self):
        st.subheader("📋 All Scanned Watchlist Tickers")
        if not st.session_state.cal_candidates:
            st.info("💡 Run the Scoring Engine from the sidebar to populate candidates.")
            return
            
        # Dynamic multi-select filter for recommendations
        existing_recs = sorted(list(set(c["recommendation"] for c in st.session_state.cal_candidates)))
        priority = {"Deploy": 0, "Watch": 1, "Monitor": 2, "Avoid": 3, "Filtered": 4}
        existing_recs = sorted(existing_recs, key=lambda x: priority.get(x, 9))
        
        selected_recs = st.multiselect(
            "Filter by Recommendation:",
            options=existing_recs,
            default=existing_recs,
            key="cal_scoring_rec_filter"
        )
        
        rows = []
        for c in st.session_state.cal_candidates:
            if c["recommendation"] not in selected_recs:
                continue
            ml_ret = c.get("ml_predicted_return")
            ml_ret_str = f"{ml_ret:+.1f}%" if ml_ret is not None else "N/A"
            rows.append({
                "Ticker": c["ticker"],
                "Score": f"{c['final_score']:.1f}" if c["recommendation"] != "Filtered" else "0.0",
                "ML Pred Return": ml_ret_str,
                "Recommendation": c["recommendation"],
                "Earnings Date": c.get("earnings_date", "N/A"),
                "FDTS Signal": c["fdts_signal"],
                "IV Rank": f"{c.get('iv_rank', 0.0):.1f}%",
                "Spot Price": f"${c.get('spot_price', 0.0):.2f}",
                "Status": c["reason_summary"]
            })
            
        if not rows:
            st.info("No candidates match the selected recommendation filter.")
            return
            
        df = pd.DataFrame(rows)
        # Apply color styling
        def color_recommendation(val):
            color_map = {
                "Deploy": "background-color: rgba(58,181,74,0.2); color: #3ab54a; font-weight: bold;",
                "Watch": "background-color: rgba(255,184,0,0.15); color: #ffb800; font-weight: bold;",
                "Monitor": "background-color: rgba(2,132,199,0.15); color: #0284c7;",
                "Avoid": "background-color: rgba(220,38,38,0.15); color: #ef4444;",
                "Filtered": "background-color: rgba(100,116,139,0.15); color: #64748b; text-decoration: line-through;"
            }
            return color_map.get(val, "")
            
        def color_earnings_date(val):
            if not val or val == "N/A":
                return ""
            try:
                earnings_dt = datetime.strptime(val, "%Y-%m-%d").date()
                today_dt = date.today()
                days_diff = (earnings_dt - today_dt).days
                if days_diff < 0:
                    return "color: #ffffff;"
                elif days_diff <= 20:
                    return "background-color: rgba(220,38,38,0.15); color: #ef4444; font-weight: bold;"
                elif days_diff <= 40:
                    return "background-color: rgba(255,184,0,0.15); color: #ffb800; font-weight: bold;"
                else:
                    return "color: #ffffff;"
            except Exception:
                return ""
            
        styled_df = df.style.map(color_recommendation, subset=["Recommendation"]).map(color_earnings_date, subset=["Earnings Date"])
        st.dataframe(styled_df, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # SCREEN 3: TICKER DETAIL VIEW
    # ══════════════════════════════════════════════════════════════════════
    def render_tab_ticker_detail(self):
        st.subheader("🔍 Single-Stock Factor Breakdown")
        if not st.session_state.cal_candidates:
            st.info("💡 Run a scoring scan to load ticker technical breakdowns.")
            return
            
        ticker_options = [c["ticker"] for c in st.session_state.cal_candidates if c["recommendation"] != "Filtered"]
        if not ticker_options:
            st.warning("No ranked candidates available. (All filtered out in last scan).")
            return
            
        ticker_names = get_ticker_names(getattr(self, "universe_name", "Options Default Watchlist"))
        sel_ticker = st.selectbox(
            "Select Candidate Ticker:",
            options=ticker_options,
            format_func=lambda t: format_ticker_display(t, ticker_names)
        )
        c = next(cand for cand in st.session_state.cal_candidates if cand["ticker"] == sel_ticker)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Technical Profiles")
            st.metric("Spot Price", f"${c.get('spot_price', 0.0):.2f}")
            st.write(f"**20 EMA**: ${c.get('ema_20', 0.0):.2f}")
            st.write(f"**50 EMA**: ${c.get('ema_50', 0.0):.2f}")
            st.write(f"**200 EMA**: ${c.get('ema_200', 0.0):.2f}")
            st.write(f"**RSI (14)**: {c.get('rsi_14', 0.0):.1f}")
            st.write(f"**ADX (14)**: {c.get('adx_14', 0.0):.1f}")
            st.write(f"**ATR (14)**: ${c.get('atr_14', 0.0):.2f}")
            
        with col2:
            st.write("#### Factor Score Breakdown")
            scores = {
                "Trend Score (17.5%)": c.get("trend_score", 0.0),
                "Option Structure (17.5%)": c.get("option_structure_score", 0.0),
                "Volatility Score (15%)": c.get("volatility_score", 0.0),
                "FDTS Score (15%)": c.get("fdts_score", 0.0),
                "PCA Score (10%)": c.get("pca_score", 0.0),
                "Cluster Score (10%)": c.get("cluster_score", 0.0),
                "Institutional Flow (5%)": c.get("institutional_flow_score", 0.0),
                "Leading/Lagging Score (5%)": c.get("leading_lagging_score", 0.0),
                "Liquidity Score (3%)": c.get("liquidity_score", 0.0),
                "Event Risk Score (2%)": c.get("event_risk_score", 0.0)
            }
            
            fig = go.Figure(go.Bar(
                x=list(scores.values()),
                y=list(scores.keys()),
                orientation='h',
                marker=dict(color='#3ab54a')
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
        st.divider()
        st.markdown("### 🧠 Phase 2 Advanced Intelligence Models")
        
        col_intra, col_surv = st.columns(2)
        
        with col_intra:
            st.markdown("#### ⚡ Bayesian Intraday Entry Timing Model")
            with st.spinner("Fetching live intraday data..."):
                intra = fetch_intraday_data(sel_ticker)
                
            if intra["success"]:
                post_prob, entry_sig = calculate_bayesian_posterior(
                    c["final_score"],
                    intra["above_vwap"],
                    intra["high_vol"],
                    intra["iv_spike"]
                )
                
                # Render metrics
                st.metric("Bayesian Entry Probability", f"{post_prob:.1f}%", f"{post_prob - c['final_score']:+.1f}% vs Daily Score")
                st.write(f"**Intraday Entry Signal**: `{entry_sig}`")
                st.write(f"**Price vs Intraday VWAP**: {'Above' if intra['above_vwap'] else 'Below'} VWAP (Price: ${intra['last_price']:.2f} | VWAP: ${intra['vwap']:.2f})")
                st.write(f"**Volume Velocity**: {'Spike detected' if intra['high_vol'] else 'Normal / Low'} (Last Bar Vol: {intra['last_volume']:.0f} | Avg Vol: {intra['avg_volume']:.0f})")
                st.write(f"**Volatility Expansion**: {'IV Spike warning' if intra['iv_spike'] else 'Stable volatility'} (Rolling Vol: {intra['latest_vol']:.4f} | Avg Vol: {intra['avg_vol_std']:.4f})")
            else:
                st.warning("Could not download intraday price bars for Bayesian updating.")
                st.write(f"**Default Entry Probability (Prior)**: {c['final_score']:.1f}%")
                
        with col_surv:
            st.markdown("#### ⏳ Kaplan-Meier Trend Survival Analysis")
            with st.spinner("Calculating trend lifespans..."):
                surv_info = analyze_trend_survival(sel_ticker)
                
            if surv_info["success"]:
                curr_age = surv_info["current_age"]
                surv_prob = surv_info["survival_prob_20d"]
                
                st.metric("Conditional Trend Survival (20 Days)", f"{surv_prob:.1f}%")
                st.write(f"**Current Uptrend Age**: `{curr_age} trading days`")
                if surv_info["warning"]:
                    st.warning("⚠️ **Warning**: Trend exhaustion probability is high. Trend is statistically unlikely to survive another 20 days.")
                else:
                    st.success("✅ **Supportive**: Trend has a high probability of survival through options holding period.")
                    
                # Plot Kaplan-Meier curve
                x_vals = list(surv_info["curve"].keys())
                y_vals = list(surv_info["curve"].values())
                fig_surv = go.Figure(data=go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines',
                    name='Survival Prob',
                    line=dict(color='#00ff88', width=2.5)
                ))
                fig_surv.add_vline(
                    x=curr_age,
                    line_dash="dash",
                    line_color="#ffb800" if curr_age < 40 else "#ef4444",
                    annotation_text=f"Current Age ({curr_age}d)",
                    annotation_position="top right"
                )
                fig_surv.update_layout(
                    title=f"Kaplan-Meier Curve: {sel_ticker} Trend Lifespans",
                    xaxis_title="Days in Trend",
                    yaxis_title="Probability",
                    height=240,
                    margin=dict(l=20, r=20, t=30, b=10),
                    template="plotly_dark"
                )
                st.plotly_chart(fig_surv, use_container_width=True)
            else:
                st.warning("Could not calculate survival curve.")

    # ══════════════════════════════════════════════════════════════════════
    # SCREEN 4: OPTION SETUP PAYOFFS
    # ══════════════════════════════════════════════════════════════════════
    def render_tab_option_setup(self):
        st.subheader("📈 Synthetic Calendar Spread Payoff Calculator")
        if not st.session_state.cal_candidates:
            st.info("💡 Run scoring scan to view option payoff graphs.")
            return
            
        ticker_options = [c["ticker"] for c in st.session_state.cal_candidates if c["recommendation"] != "Filtered"]
        if not ticker_options:
            st.warning("No ranked candidates available.")
            return
            
        ticker_names = get_ticker_names(getattr(self, "universe_name", "Options Default Watchlist"))
        sel_ticker = st.selectbox(
            "Select Setup Ticker Payoff:",
            options=ticker_options,
            key="payoff_ticker_sel",
            format_func=lambda t: format_ticker_display(t, ticker_names)
        )
        c = next(cand for cand in st.session_state.cal_candidates if cand["ticker"] == sel_ticker)
        setup = c.get("option_setup")
        
        if not setup:
            st.warning(f"No option setup available for {sel_ticker}")
            return
            
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("#### Trade Setup Snapshot")
            st.write(f"**Strike**: ${setup['selected_strike']:.2f}")
            st.write(f"**Front Expiry**: {setup['short_expiry']} (DTE: {setup['short_dte']})")
            st.write(f"**Back Expiry**: {setup['long_expiry']} (DTE: {setup['long_dte']})")
            st.write(f"**Debit Paid / Risk**: ${setup['net_debit']:.2f}")
            st.write(f"**Front Mid Price**: ${setup['short_mid']:.2f}")
            st.write(f"**Back Mid Price**: ${setup['long_mid']:.2f}")
            
            st.write("#### Net Greeks")
            st.write(f"**Delta**: {setup['setup_delta']:.4f}")
            st.write(f"**Gamma**: {setup['setup_gamma']:.4f}")
            st.write(f"**Theta**: {setup['setup_theta']:.4f}")
            st.write(f"**Vega**: {setup['setup_vega']:.4f}")
            
        with col2:
            st.write("#### Expiration Payoff Diagram (Short Leg Expiration)")
            # Generate calendar payoff curve
            spot_start = c["spot_price"]
            strike = setup["selected_strike"]
            debit = setup["net_debit"]
            
            prices = np.linspace(spot_start * 0.85, spot_start * 1.15, 80)
            payoff = []
            
            # Simple approximation of calendar payoff at short leg expiration:
            # Short leg is worth 0. Back leg is worth its remaining Black-Scholes call price.
            # Payoff = Back Leg Call Price (with DTE = long_dte - short_dte) - Short Leg Call Price (if ITM) - Net Debit
            r = 0.045
            for p in prices:
                T_back_rem = (setup["long_dte"] - setup["short_dte"]) / 365.0
                back_val, _, _, _, _ = black_scholes_call(p, strike, T_back_rem, r, setup["back_iv"])
                short_val = max(0.0, p - strike)
                spread_val = back_val - short_val
                pnl = spread_val - debit
                payoff.append(pnl)
                
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prices, y=payoff, name="PnL at Front Expiry", line=dict(color='#3ab54a', width=2.5)))
            # Add spot marker line
            fig.add_vline(x=spot_start, line_dash="dash", line_color="#ffb800", annotation_text="Current Price")
            fig.add_hline(y=0.0, line_color="#64748b")
            fig.update_layout(xaxis_title="Stock Price", yaxis_title="Profit / Loss ($)", height=320, margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # SCREEN 5: DECISION HISTORY LOG
    # ══════════════════════════════════════════════════════════════════════
    def render_tab_decision_history(self):
        st.subheader("📂 Historical Decision Records")
        conn = get_connection()
        df = pd.read_sql_query("""
            SELECT decision_id, decision_date, ticker, strategy_type, recommendation, final_score, price_at_decision, market_regime
            FROM ticker_decision_log
            ORDER BY decision_id DESC
        """, conn)
        conn.close()
        
        if df.empty:
            st.info("No historical decisions recorded yet. Run a scoring engine scan to save logs.")
            return
            
        # Multi-select filter for recommendation in history
        existing_recs = sorted(list(df["recommendation"].unique()))
        priority = {"Deploy": 0, "Watch": 1, "Monitor": 2, "Avoid": 3, "Filtered": 4}
        existing_recs = sorted(existing_recs, key=lambda x: priority.get(x, 9))
        
        selected_recs = st.multiselect(
            "Filter by Recommendation (History):",
            options=existing_recs,
            default=existing_recs,
            key="cal_scoring_rec_history_filter"
        )
        
        filtered_df = df[df["recommendation"].isin(selected_recs)]
        
        if filtered_df.empty:
            st.info("No historical decisions match the selected recommendation filter.")
            return
            
        st.dataframe(filtered_df, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # SCREEN 6: OUTCOME TRACKING
    # ══════════════════════════════════════════════════════════════════════
    def render_tab_outcome_tracking(self):
        st.subheader("🎯 Forward Performance Tracking")
        
        # Trigger process
        if st.button("Run Outcomes Engine (Updates Review Metrics)", key="btn_run_outcomes_track"):
            count = update_decision_outcomes()
            st.success(f"Outcome updates complete. {count} decisions processed.")
            
        conn = get_connection()
        df = pd.read_sql_query("""
            SELECT o.outcome_id, o.decision_id, o.ticker, o.review_day, o.price_at_review, o.pnl_pct, o.result_label, o.exit_signal
            FROM decision_outcome_log o
            ORDER BY o.outcome_id DESC
        """, conn)
        conn.close()
        
        if df.empty:
            st.info("No outcomes tracked yet. Execute scoring scans, wait, or press 'Run Outcomes Engine' to simulate.")
            return
            
        st.dataframe(df, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # SCREEN 7: BACKTEST PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════
    def render_tab_backtest_performance(self):
        st.subheader("📊 Historical Strategy Performance Backtest")
        
        stats = run_backtest_analysis()
        is_syn = stats.get("is_synthetic", False)
        if is_syn:
            st.caption("ℹ️ Note: Displaying simulated backtest parameters (Database has insufficient real historical logs).")
            
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Deploy vs Watch Return Performance")
            fig = px.bar(stats["deploy_vs_watch"], x="recommendation", y="avg_pnl", text_auto=True, title="Average PnL % by Recommendation")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("#### FDTS Buy vs Others Performance")
            fig = px.bar(stats["fdts_perf"], x="fdts_signal", y="win_rate", text_auto=True, title="Win Rate % by FDTS Trend Signal")
            st.plotly_chart(fig, use_container_width=True)
            
        col3, col4 = st.columns(2)
        with col3:
            st.write("#### Cluster performance")
            fig = px.bar(stats["cluster_perf"], x="cluster_label", y="avg_pnl", text_auto=True, title="Average PnL % by Cluster State")
            st.plotly_chart(fig, use_container_width=True)
            
        with col4:
            st.write("#### Market Regime edge")
            fig = px.bar(stats["regime_perf"], x="market_regime", y="avg_pnl", text_auto=True, title="Average PnL % by Market Regime")
            st.plotly_chart(fig, use_container_width=True)
            
        col5, col6 = st.columns(2)
        with col5:
            st.write("#### Predictiveness Score Correlation")
            fig = px.bar(stats["predictive_df"], x="score_type", y="correlation", text_auto=True, title="Correlation of Scores to Future PnL")
            st.plotly_chart(fig, use_container_width=True)
            
        with col6:
            st.write("#### Setup Comparisons")
            # Side by side charts
            st.write(stats["dte_perf"])
            st.write(stats["delta_perf"])

    # ══════════════════════════════════════════════════════════════════════
    # SCREEN 8: MODEL WEIGHT OPTIMIZATION
    # ══════════════════════════════════════════════════════════════════════
    def render_tab_weights_optimization(self):
        st.subheader("⚙️ Scoring Model Weight Optimization")
        
        # Load weights
        active_weights = get_active_model_weights()
        
        st.write("#### Current Active Scoring Weights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trend_w = st.slider("Trend Weight", 0.0, 0.50, float(active_weights.get("trend_weight", 0.20)), 0.01)
            opt_w = st.slider("Option Structure Weight", 0.0, 0.50, float(active_weights.get("option_structure_weight", 0.20)), 0.01)
            vol_w = st.slider("Volatility Weight", 0.0, 0.50, float(active_weights.get("volatility_weight", 0.15)), 0.01)
            
        with col2:
            fdts_w = st.slider("FDTS Weight", 0.0, 0.50, float(active_weights.get("fdts_weight", 0.15)), 0.01)
            pca_w = st.slider("PCA Weight", 0.0, 0.50, float(active_weights.get("pca_weight", 0.10)), 0.01)
            clus_w = st.slider("Cluster Weight", 0.0, 0.50, float(active_weights.get("cluster_weight", 0.10)), 0.01)
            
        with col3:
            lead_w = st.slider("Leading/Lagging Weight", 0.0, 0.30, float(active_weights.get("leading_lagging_weight", 0.05)), 0.01)
            liq_w  = st.slider("Liquidity Weight", 0.0, 0.20, float(active_weights.get("liquidity_weight", 0.03)), 0.01)
            evt_w  = st.slider("Event Risk Weight", 0.0, 0.20, float(active_weights.get("event_risk_weight", 0.02)), 0.01)
            inst_w = st.slider("Institutional Flow Weight", 0.0, 0.20, float(active_weights.get("institutional_flow_weight", 0.05)), 0.01)

        total_w = trend_w + opt_w + vol_w + fdts_w + pca_w + clus_w + lead_w + liq_w + evt_w + inst_w
        st.write(f"**Total Weight Sum**: {total_w * 100:.1f}%")

        if abs(total_w - 1.0) > 0.001:
            st.error("⚠️ Scoring weights must sum to exactly 100% before saving.")
        else:
            if st.button("Save Adjusted Weights", key="btn_save_adjusted_w"):
                save_model_weights({
                    "trend_weight": trend_w,
                    "option_structure_weight": opt_w,
                    "volatility_weight": vol_w,
                    "fdts_weight": fdts_w,
                    "pca_weight": pca_w,
                    "cluster_weight": clus_w,
                    "leading_lagging_weight": lead_w,
                    "liquidity_weight": liq_w,
                    "event_risk_weight": evt_w,
                    "institutional_flow_weight": inst_w
                })
                st.success("New active weights saved successfully.")
                st.rerun()
                
        st.divider()
        st.write("#### Grid Search Weights Optimization Engine")
        if st.button("Run Grid Search weights optimizer", key="btn_run_grid_opt", type="primary"):
            with st.spinner("Executing grid search over weight spaces..."):
                opt_res = run_weights_optimization()
                st.write("##### Grid Search Top 10 Configurations")
                st.dataframe(opt_res["results_df"], use_container_width=True)
                
                # Option to load best weights
                best = opt_res["best_weights"]
                st.info(f"Recommended weights configuration found: Trend {best['trend_weight']*100:.0f}%, "
                        f"IV Structure {best['option_structure_weight']*100:.0f}%, Volatility {best['volatility_weight']*100:.0f}%. "
                        f"Simulated Win Rate: {best['win_rate']:.1f}%.")
                
                if st.button("Apply Recommended Weights", key="btn_apply_best_weights"):
                    save_model_weights(best)
                    st.success("Applied recommended weights configuration.")
                    st.rerun()
                    
        st.divider()
        st.write("#### 🤖 Machine Learning Meta-Ranking Model")
        
        st.session_state.cal_enable_ml_sort = st.toggle(
            "Enable ML Meta-Ranking Expected Return Sorting", 
            value=st.session_state.cal_enable_ml_sort,
            help="Toggle to sort candidate scan lists primarily by predicted ML expected PnL instead of rules-based scores."
        )
        
        if st.button("Train/Retrain Meta-Ranking ML Model", key="btn_train_ml_meta"):
            with st.spinner("Querying database and training Gradient Boosting regressor..."):
                res = train_meta_ranking_model()
                
            if res["status"] == "success":
                st.success(f"✅ Model trained successfully! Model Type: `{res['model_type']}` | Sample Count: `{res['sample_count']}`")
                st.write(f"**R-squared Score**: `{res['r2']:.4f}` | **Root Mean Squared Error**: `{res['rmse']:.2f}%` PnL")
                
                st.write("##### Model Feature Importances")
                importances = res["feature_importances"]
                top_features = dict(list(importances.items())[:10])
                fig_imp = go.Figure(go.Bar(
                    x=list(top_features.values()),
                    y=list(top_features.keys()),
                    orientation='h',
                    marker=dict(color='#00ff88')
                ))
                fig_imp.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=10), template="plotly_dark")
                st.plotly_chart(fig_imp, use_container_width=True)
            elif res["status"] == "cold_start":
                st.warning(f"⚠️ **Cold Start Warning**: {res['message']}")
            else:
                st.error(f"❌ **Error**: {res['message']}")

    # ══════════════════════════════════════════════════════════════════════
    # SCREEN 9: REGIME HMM TRANSITIONS
    # ══════════════════════════════════════════════════════════════════════
    def render_tab_regime_hmm(self):
        st.subheader("⚡ HMM Regime Transition Probability Model")
        
        regime_info = detect_market_regime()
        st.write(f"**Current Identified Regime**: {regime_info['regime']}")
        st.write(f"*{regime_info['description']}*")
        
        st.write("#### Transition Probability Table")
        st.write("Probability of shifting from the current state to future regimes over the next 20 trading days:")
        
        hmm_trans = regime_info["hmm_transitions"]
        # Convert to DF
        df_hmm = pd.DataFrame([
            {"Future Regime": k, "Transition Probability": f"{v * 100.0:.1f}%"} for k, v in hmm_trans.items()
        ])
        st.dataframe(df_hmm, use_container_width=True)
        
        # Plotly chart
        fig = go.Figure(data=[go.Pie(labels=list(hmm_trans.keys()), values=list(hmm_trans.values()), hole=.3)])
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=10), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily Transition Matrix Heatmap
        matrices = get_transition_matrices()
        if matrices["daily"] is not None:
            st.write("#### Daily Markov Transition Probability Matrix")
            st.write("Historical probability of transitioning between regimes from one day to the next:")
            P_daily, states = matrices["daily"]
            df_p = pd.DataFrame(P_daily, index=states, columns=states)
            df_p_pct = df_p.map(lambda x: f"{x * 100.0:.1f}%")
            st.dataframe(df_p_pct, use_container_width=True)
            
            fig_hm = px.imshow(
                P_daily * 100.0,
                labels=dict(x="To Regime", y="From Regime", color="Prob %"),
                x=states,
                y=states,
                color_continuous_scale="Viridis",
                text_auto=".1f"
            )
            fig_hm.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=10), template="plotly_dark")
            st.plotly_chart(fig_hm, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # SCREEN 10: PAPER TRADE TRACKER
    # ══════════════════════════════════════════════════════════════════════
    def render_tab_paper_tracker(self):
        st.subheader("💼 Active Paper Trade Deployments")
        st.write("Track deployed virtual setups and exit criteria.")
        
        # Load virtual trade listings from database decision log
        conn = get_connection()
        df = pd.read_sql_query("""
            SELECT d.decision_id, d.decision_date, d.ticker, s.selected_strike, s.net_debit, d.price_at_decision
            FROM ticker_decision_log d
            JOIN option_trade_setup_log s ON d.decision_id = s.decision_id
            WHERE d.recommendation = 'Deploy'
            ORDER BY d.decision_id DESC LIMIT 5
        """, conn)
        conn.close()
        
        if df.empty:
            st.info("No active Deploy positions logged. Run a scoring run and locate 'Deploy' candidates.")
            return
            
        # Add tracking targets
        df["Profit Target (30%)"] = df["net_debit"] * 1.30
        df["Stop Loss (35%)"] = df["net_debit"] * 0.65
        df["Target Strike Exit"] = df["selected_strike"]
        
        # Format table
        st.dataframe(df, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # ENGINE SCAN FUNCTION
    # ══════════════════════════════════════════════════════════════════════
    def execute_engine_scan(self):
        """Execute calculations and scoring over the complete ticker universe."""
        st.session_state.cal_candidates = []
        progress_bar = st.progress(0)

        # Get active weights from DB
        weights = get_active_model_weights()
        model_version = weights.get("model_version", "Phase 1 - v3.0")

        regime_info = detect_market_regime()
        market_regime = regime_info["regime"]
        hmm_transitions = regime_info.get("hmm_transitions", {})
        
        # Dynamically scale option scoring weights based on HMM state transition probabilities
        active_weights = self._adjust_weights_by_hmm(weights, hmm_transitions)

        # ── Pre-fetch benchmark data ONCE before the ticker loop ──────────────
        # Avoids repeated SPY/QQQ API calls inside calculate_pca_score() and
        # calculate_leading_lagging_score() for every ticker.
        status_text = st.empty()
        status_text.text("Fetching benchmark data (SPY, QQQ)...")
        spy_df, benchmark_returns = fetch_benchmark_data()

        scanned_count  = 0
        total_tickers  = len(self.tickers)

        temp_candidates = []
        
        for idx, ticker in enumerate(self.tickers):
            status_text.text(f"Scanning {ticker} ({idx+1}/{total_tickers})...")
            progress_bar.progress((idx + 1) / total_tickers)
            
            try:
                # 1. Technical Data
                tech = fetch_technical_data(ticker)
                
                # 2. Options Data
                option_chain = fetch_option_chain_data(ticker, tech["spot_price"], use_synthetic=self.use_synthetic_data)
                
                # 3. Setup Leg Selections
                setup = select_calendar_setup(ticker, option_chain, tech["spot_price"], tech["hv_30"], target_delta=0.25)
                
                if not setup:
                    continue
                    
                # 4. FDTS Signal
                fdts = calculate_fdts_signal(tech["spot_price"], tech["ema_20"], tech["ema_50"], tech["ema_200"], tech["rsi_14"])
                
                # 5. Volatility Rank calculations (simulate IV Rank for yfinance)
                # IV rank is usually in the 10-90 range, we can proxy it
                np.random.seed(hash(ticker) % 1000)
                iv_rank = float(np.random.uniform(15, 60))
                iv_pct = float(np.random.uniform(10, 65))
                
                # Fetch real earnings date from yfinance; fall back to +45-day proxy
                try:
                    _real_ed = _get_real_earnings_date(ticker)
                    if _real_ed:
                        earnings_date = _real_ed.strftime("%Y-%m-%d") if hasattr(_real_ed, "strftime") else str(_real_ed)[:10]
                    else:
                        earnings_date = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")
                except Exception:
                    earnings_date = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")
                
                # 6. Apply individual Scoring Engines
                trend_score      = calculate_trend_score(tech["spot_price"], tech["ema_20"], tech["ema_50"], tech["ema_200"], tech["adx_14"])
                opt_struct_score = calculate_option_structure_score(setup["front_iv"], setup["back_iv"])
                vol_score        = calculate_volatility_score(iv_rank, iv_pct)
                fdts_score_val   = calculate_fdts_score(fdts["score"])

                # PCA: pass pre-fetched benchmark returns to avoid redundant API call
                pca_score = calculate_pca_score(ticker, tech["df_history"], benchmark_returns)

                # Heuristic cluster classification (K-Means ML clustering is excluded per request)
                clus_score, clus_lbl = calculate_cluster_score(ticker, tech["df_history"])

                # Multi-timeframe relative strength vs SPY (pass pre-fetched SPY)
                lead_score, lead_state = calculate_leading_lagging_score(ticker, tech["df_history"], spy_df)

                liq_score  = calculate_liquidity_score(setup["bid_ask_spread_pct"], setup["avg_option_volume"])
                evt_score, evt_flag = calculate_event_risk_score(earnings_date, setup["short_dte"])

                # Institutional flow (uses live options chain if available)
                inst_flow_score = calculate_institutional_flow_score(ticker, option_chain)

                # Compute Final Score (10 factors, weights sum to 100% using dynamic HMM weights)
                final_score = (
                    trend_score      * active_weights.get("trend_weight", 0.175) +
                    opt_struct_score * active_weights.get("option_structure_weight", 0.175) +
                    vol_score        * active_weights.get("volatility_weight", 0.15) +
                    fdts_score_val   * active_weights.get("fdts_weight", 0.15) +
                    pca_score        * active_weights.get("pca_weight", 0.10) +
                    clus_score       * active_weights.get("cluster_weight", 0.10) +
                    lead_score       * active_weights.get("leading_lagging_weight", 0.05) +
                    inst_flow_score  * active_weights.get("institutional_flow_weight", 0.05) +
                    liq_score        * active_weights.get("liquidity_weight", 0.03) +
                    evt_score        * active_weights.get("event_risk_weight", 0.02)
                )
                
                # Component 4: Survival Analysis trend lifespan check
                surv_analysis = analyze_trend_survival_with_df(ticker, tech["df_history"])
                survival_prob = surv_analysis.get("survival_prob_20d", 75.0)
                survival_warning = surv_analysis.get("warning", False)
                
                # recommendation
                if final_score >= 85:
                    rec = "Deploy"
                elif final_score >= 75:
                    rec = "Watch"
                elif final_score >= 65:
                    rec = "Monitor"
                else:
                    rec = "Avoid"
                    
                # Apply Survival Trend Exhaustion downgrade
                if rec == "Deploy" and survival_prob < 65.0:
                    rec = "Watch"
                    
                # Update tech data parameters
                tech["iv_rank"] = iv_rank
                tech["iv_percentile"] = iv_pct
                tech["earnings_date"] = earnings_date
                tech["event_risk_flag"] = evt_flag
                
                # Apply Hard Filters
                exclusions = apply_hard_filters(ticker, tech, setup, fdts["signal"])
                reason_summary = "Meets all criteria"
                if survival_prob < 65.0 and rec != "Filtered":
                    reason_summary = f"Trend Exhaustion Warning ({survival_prob:.1f}% survival prob)"
                if exclusions:
                    rec = "Filtered"
                    reason_summary = ", ".join(exclusions)
                    
                cand_data = {
                    "ticker": ticker,
                    "final_score": round(final_score, 1),
                    "recommendation": rec,
                    "fdts_signal": fdts["signal"],
                    "fdts_score": fdts["score"],
                    "trend_score": trend_score,
                    "option_structure_score": opt_struct_score,
                    "volatility_score": vol_score,
                    "pca_score": pca_score,
                    "cluster_score": clus_score,
                    "cluster_label": clus_lbl,
                    "leading_lagging_score": lead_score,
                    "leading_lagging_state": lead_state,
                    "liquidity_score": liq_score,
                    "event_risk_score": evt_score,
                    "institutional_flow_score": inst_flow_score,
                    "spot_price": tech["spot_price"],
                    "ema_20": tech["ema_20"],
                    "ema_50": tech["ema_50"],
                    "ema_200": tech["ema_200"],
                    "rsi_14": tech["rsi_14"],
                    "adx_14": tech["adx_14"],
                    "atr_14": tech["atr_14"],
                    "iv_rank": iv_rank,
                    "iv_percentile": iv_pct,
                    "avg_option_volume": setup["avg_option_volume"],
                    "avg_open_interest": setup["avg_open_interest"],
                    "bid_ask_spread_pct": setup["bid_ask_spread_pct"],
                    "front_iv": setup["front_iv"],
                    "back_iv": setup["back_iv"],
                    "earnings_date": earnings_date,
                    "event_risk_flag": evt_flag,
                    "market_regime": market_regime,
                    "reason_summary": reason_summary,
                    "option_setup": setup
                }
                
                temp_candidates.append(cand_data)
                scanned_count += 1
            except Exception as e:
                logger.error(f"Error processing ticker {ticker}: {e}")
                
        # Run ML Meta-Ranking predictions on all candidates
        temp_candidates = predict_meta_rankings(temp_candidates)
        
        # Sort candidates (Rank index order: Deploy -> Watch -> Monitor -> Avoid -> Filtered, sorted by score or ML return)
        def sort_key(x):
            rec_order = {"Deploy": 0, "Watch": 1, "Monitor": 2, "Avoid": 3, "Filtered": 4}
            val = x.get("ml_predicted_return", 0.0) if st.session_state.cal_enable_ml_sort else x["final_score"]
            return (rec_order.get(x["recommendation"], 9), -val)
            
        ranked_candidates = sorted(temp_candidates, key=sort_key)
        
        # 7. Log all to SQLite Database
        run_date = date.today().strftime("%Y-%m-%d")
        log_daily_run(run_date, ranked_candidates, model_version)
        
        # Store in session state
        st.session_state.cal_candidates = ranked_candidates
        st.session_state.cal_last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # UI Feedback
        status_text.empty()
        progress_bar.empty()
        st.success(f"Scan complete. {scanned_count} candidates scored and logged to database.")
        st.rerun()
