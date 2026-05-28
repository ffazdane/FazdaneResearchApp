import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import logging

from modules.base_module import FazDaneModule
from utils.universe_manager import render_universe_manager, get_tickers

# Import engine components
from modules.calendar_scoring.config import STRATEGY_CONFIG, STRATEGY_CONFIG as STR_CONF
from modules.calendar_scoring.database import get_active_model_weights, save_model_weights, get_connection
from modules.calendar_scoring.data_loader import fetch_technical_data, fetch_option_chain_data
from modules.calendar_scoring.market_regime import detect_market_regime
from modules.calendar_scoring.fdts_engine import calculate_fdts_signal
from modules.calendar_scoring.trade_setup_engine import select_calendar_setup
from modules.calendar_scoring.scoring_engine import (
    calculate_trend_score, calculate_option_structure_score, calculate_volatility_score,
    calculate_fdts_score, calculate_pca_score, calculate_cluster_score,
    calculate_leading_lagging_score, calculate_liquidity_score, calculate_event_risk_score,
    apply_hard_filters
)
from modules.calendar_scoring.decision_logger import log_daily_run
from modules.calendar_scoring.outcome_tracker import update_decision_outcomes
from modules.calendar_scoring.backtest_engine import run_backtest_analysis
from modules.calendar_scoring.optimization_engine import run_weights_optimization
from modules.calendar_scoring.explanation_engine import generate_llm_prompt

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
            
    def render_sidebar(self):
        st.write("### Strategy Settings")
        st.write(f"**Strategy**: {STRATEGY_CONFIG['strategy_type']}")
        st.write(f"**Short Leg**: ~{STRATEGY_CONFIG['short_dte_target']} DTE")
        st.write(f"**Long Leg**: ~{STRATEGY_CONFIG['long_dte_target']} DTE")
        st.write(f"**Target Delta**: {STRATEGY_CONFIG['target_delta']} Delta")
        
        st.divider()
        st.write("### Universe Selection")
        # Load Universe Manager widget
        universe_name, self.tickers, _ = render_universe_manager(
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
        
        # Primary Multipage Dashboard Tab Nav
        tabs = st.tabs([
            "1. Daily Top Pick Setups",
            "2. All Ranked Candidates",
            "3. Ticker Detail View",
            "4. Option Setup Payoffs",
            "5. Decision History Log",
            "6. Outcome Tracking",
            "7. Backtest Performance",
            "8. Weights Optimization",
            "9. Regime HMM Transitions",
            "10. Paper Trade Tracker"
        ])
        
        with tabs[0]:
            self.render_tab_top_picks()
        with tabs[1]:
            self.render_tab_all_ranked()
        with tabs[2]:
            self.render_tab_ticker_detail()
        with tabs[3]:
            self.render_tab_option_setup()
        with tabs[4]:
            self.render_tab_decision_history()
        with tabs[5]:
            self.render_tab_outcome_tracking()
        with tabs[6]:
            self.render_tab_backtest_performance()
        with tabs[7]:
            self.render_tab_weights_optimization()
        with tabs[8]:
            self.render_tab_regime_hmm()
        with tabs[9]:
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
            
        # Format table
        rows = []
        for c in top_picks:
            setup = c.get("option_setup", {})
            rows.append({
                "Ticker": c["ticker"],
                "Score": f"{c['final_score']:.1f}",
                "Recommendation": c["recommendation"],
                "FDTS Signal": c["fdts_signal"],
                "Cluster Label": c["cluster_label"],
                "Strike": f"${setup.get('selected_strike', 0.0):.2f}",
                "Net Debit": f"${setup.get('net_debit', 0.0):.2f}",
                "Comb. Theta": f"+{setup.get('setup_theta', 0.0):.4f}",
                "Comb. Vega": f"+{setup.get('setup_vega', 0.0):.4f}",
                "Reason Summary": c["reason_summary"]
            })
            
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        
        # Select ticker for Prompt Generator
        st.markdown("#### 💬 AI Analyst ChatGPT Prompt Copyable")
        sel_prompt_ticker = st.selectbox("Select Top Pick for ChatGPT Prompt:", options=[c["ticker"] for c in top_picks])
        target_c = next(c for c in top_picks if c["ticker"] == sel_prompt_ticker)
        prompt_txt = generate_llm_prompt(target_c)
        st.code(prompt_txt, language="text")

    # ══════════════════════════════════════════════════════════════════════
    # SCREEN 2: ALL RANKED CANDIDATES
    # ══════════════════════════════════════════════════════════════════════
    def render_tab_all_ranked(self):
        st.subheader("📋 All Scanned Watchlist Tickers")
        if not st.session_state.cal_candidates:
            st.info("💡 Run the Scoring Engine from the sidebar to populate candidates.")
            return
            
        rows = []
        for c in st.session_state.cal_candidates:
            rows.append({
                "Ticker": c["ticker"],
                "Score": f"{c['final_score']:.1f}" if c["recommendation"] != "Filtered" else "0.0",
                "Recommendation": c["recommendation"],
                "FDTS Signal": c["fdts_signal"],
                "IV Rank": f"{c.get('iv_rank', 0.0):.1f}%",
                "Spot Price": f"${c.get('spot_price', 0.0):.2f}",
                "Status": c["reason_summary"]
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
            
        styled_df = df.style.map(color_recommendation, subset=["Recommendation"])
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
            
        sel_ticker = st.selectbox("Select Candidate Ticker:", options=ticker_options)
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
                "Trend Score (20%)": c.get("trend_score", 0.0),
                "Option Structure (20%)": c.get("option_structure_score", 0.0),
                "Volatility Score (15%)": c.get("volatility_score", 0.0),
                "FDTS Score (15%)": c.get("fdts_score", 0.0),
                "PCA Score (10%)": c.get("pca_score", 0.0),
                "Cluster Score (10%)": c.get("cluster_score", 0.0),
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
            
        sel_ticker = st.selectbox("Select Setup Ticker Payoff:", options=ticker_options, key="payoff_ticker_sel")
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
            
        st.dataframe(df, use_container_width=True)

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
            liq_w = st.slider("Liquidity Weight", 0.0, 0.20, float(active_weights.get("liquidity_weight", 0.03)), 0.01)
            evt_w = st.slider("Event Risk Weight", 0.0, 0.20, float(active_weights.get("event_risk_weight", 0.02)), 0.01)
            
        total_w = trend_w + opt_w + vol_w + fdts_w + pca_w + clus_w + lead_w + liq_w + evt_w
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
                    "institutional_flow_weight": 0.0
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
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

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
        model_version = weights.get("model_version", "MVP v2.02")
        
        regime_info = detect_market_regime()
        market_regime = regime_info["regime"]
        
        scanned_count = 0
        total_tickers = len(self.tickers)
        
        status_text = st.empty()
        
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
                
                # Add earnings date proxy
                earnings_date = (datetime.now() + timedelta(days=int(np.random.randint(10, 90)))).strftime("%Y-%m-%d")
                
                # 6. Apply individual Scoring Engines
                trend_score = calculate_trend_score(tech["spot_price"], tech["ema_20"], tech["ema_50"], tech["ema_200"], tech["adx_14"])
                opt_struct_score = calculate_option_structure_score(setup["front_iv"], setup["back_iv"])
                vol_score = calculate_volatility_score(iv_rank, iv_pct)
                fdts_score_val = calculate_fdts_score(fdts["score"])
                pca_score = calculate_pca_score(ticker, tech["df_history"])
                clus_score, clus_lbl = calculate_cluster_score(ticker, tech["df_history"])
                lead_score, lead_state = calculate_leading_lagging_score(ticker, tech["df_history"])
                liq_score = calculate_liquidity_score(setup["bid_ask_spread_pct"], setup["avg_option_volume"])
                evt_score, evt_flag = calculate_event_risk_score(earnings_date, setup["short_dte"])
                
                # Compute Final Score
                final_score = (
                    trend_score * weights.get("trend_weight", 0.20) +
                    opt_struct_score * weights.get("option_structure_weight", 0.20) +
                    vol_score * weights.get("volatility_weight", 0.15) +
                    fdts_score_val * weights.get("fdts_weight", 0.15) +
                    pca_score * weights.get("pca_weight", 0.10) +
                    clus_score * weights.get("cluster_weight", 0.10) +
                    lead_score * weights.get("leading_lagging_weight", 0.05) +
                    liq_score * weights.get("liquidity_weight", 0.03) +
                    evt_score * weights.get("event_risk_weight", 0.02)
                )
                
                # recommendation
                if final_score >= 85:
                    rec = "Deploy"
                elif final_score >= 75:
                    rec = "Watch"
                elif final_score >= 65:
                    rec = "Monitor"
                else:
                    rec = "Avoid"
                    
                # Update tech data parameters
                tech["iv_rank"] = iv_rank
                tech["iv_percentile"] = iv_pct
                tech["earnings_date"] = earnings_date
                tech["event_risk_flag"] = evt_flag
                
                # Apply Hard Filters
                exclusions = apply_hard_filters(ticker, tech, setup, fdts["signal"])
                reason_summary = "Meets all criteria"
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
                
        # Sort candidates (Rank index order: Deploy -> Watch -> Monitor -> Avoid -> Filtered, sorted by score)
        def sort_key(x):
            rec_order = {"Deploy": 0, "Watch": 1, "Monitor": 2, "Avoid": 3, "Filtered": 4}
            return (rec_order.get(x["recommendation"], 9), -x["final_score"])
            
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
