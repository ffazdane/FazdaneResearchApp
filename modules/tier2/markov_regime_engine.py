import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.base_module import FazDaneModule
from utils.universe_manager import render_universe_manager, get_ticker_names, format_ticker_display
from utils.persistence import get_db_path

# Import sub-modules
from modules.tier2.markov_database import (
    create_markov_tables, save_daily_states, save_transition_matrix,
    save_forecast, save_backtest_results, get_latest_forecast,
    get_latest_backtest, get_historical_states, get_latest_transition_matrix,
    get_connection as get_mre_conn
)
from modules.tier2.hmm_regime_model import train_hmm_model
from modules.tier2.markov_backtester import run_walk_forward_backtest
from modules.tier2.markov_visuals import (
    generate_transition_heatmap, generate_regime_timeline,
    generate_probability_trend, generate_backtest_equity_curve
)
from modules.calendar_scoring.technical_indicators import format_fdts_signal

logger = logging.getLogger("MarkovRegimeEngine")

class MarkovRegimeEngineModule(FazDaneModule):
    MODULE_NAME = "Regime Intelligence Dashboard"
    MODULE_ICON = "🔄"
    MODULE_DESCRIPTION = "Hidden Markov Models & empirical Markov state transitions for multi-day regime forecasting."
    TIER = 2
    SOURCE_NOTEBOOK = "Markov Regime Engine Integration"
    CACHE_TTL = 300
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance", "daily_prices cache"]

    def __init__(self):
        super().__init__()
        create_markov_tables()
        self.db_lock = threading.Lock()

    def render_sidebar(self):
        st.markdown("**Watchlist**")
        self.universe_name, self.symbols, _ = render_universe_manager(
            key_prefix="mre_universe",
            show_benchmark=False,
            label="Select Universe:",
        )
        st.caption(f"{len(self.symbols)} symbols selected.")
        
        st.markdown("**Engine Settings**")
        self.lookback_years = st.slider("Lookback Period (Years)", 1, 10, 5, key="mre_lookback")
        self.n_states = st.selectbox("HMM Hidden States", [3, 4, 5], index=0, key="mre_n_states")
        
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        self.run_engine = st.button("Execute Regime Scan", width="stretch", type="primary", key="mre_run_btn")

    def execute_regime_scan(self, universe_name=None, symbols=None, lookback_years=None, n_states=None, rerun=True, progress_bar=None, status_text=None):
        """Execute empirical and hidden Markov regime scan and save results to SQLite."""
        u_name = universe_name if universe_name is not None else self.universe_name
        u_symbols = symbols if symbols is not None else self.symbols
        u_lookback = lookback_years if lookback_years is not None else self.lookback_years
        u_n_states = n_states if n_states is not None else self.n_states

        if progress_bar is None:
            progress_bar = st.progress(0.0)
        else:
            progress_bar.progress(0.0)
        if status_text is None:
            status_text = st.empty()

        status_text.text(f"Starting scan for Universe '{u_name}' with {len(u_symbols)} tickers...")
        results = {}
        
        # Fetch all fdts signals once to avoid multiple database queries
        fdts_data = self._get_latest_fdts_signals()
        
        # Run parallel processing
        max_workers = min(len(u_symbols), 8)
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_single_ticker, symbol, fdts_data, u_lookback, u_n_states): symbol
                for symbol in u_symbols
            }
            
            for future in as_completed(futures):
                symbol, ticker_res = future.result()
                if ticker_res is not None:
                    results[symbol] = ticker_res
                completed_count += 1
                status_text.text(
                    f"Regime Scan Universe: '{u_name}' | Tickers: {len(u_symbols)} | "
                    f"Progress: {completed_count}/{len(u_symbols)} completed (last: {symbol} — saved to SQLite database)"
                )
                progress_bar.progress(completed_count / len(u_symbols))
        
        progress_bar.empty()
        status_text.text(f"Saving regime intelligence scan results for {len(results)} tickers to calendar_scoring database...")
        # Database writes are performed individually inside _process_single_ticker
        status_text.text(f"✅ Saved regime intelligence scan for {len(results)} tickers to calendar_scoring SQLite database.")
        
        # Trigger backup to cloud
        try:
            from utils.persistence import backup_database
            status_text.text("Cloud Database Backup: Uploading databases to repository releases...")
            ok_cs, msg_cs = backup_database("calendar_scoring", reason=f"Markov regime scan: {u_name}")
            ok_ol, msg_ol = backup_database("options_liquidity", reason=f"Markov regime pricing cache: {u_name}")
            if ok_cs and ok_ol:
                status_text.text(f"✅ Saved scan and backed up databases: {msg_cs}")
            else:
                logger.warning(f"Backup warning: calendar_scoring ({msg_cs}), options_liquidity ({msg_ol})")
        except Exception as e:
            logger.error(f"Failed to backup databases: {e}")
        
        st.session_state["mre_results"] = results
        st.session_state["mre_scan_completed"] = True
        st.session_state["mre_needs_reprocess"] = False
        
        # Fetch created_at timestamp from the database for consistency
        latest_ts = None
        if results:
            first_sym = list(results.keys())[0]
            with self.db_lock:
                first_forecast = get_latest_forecast(first_sym)
            latest_ts = first_forecast.get("created_at")
        
        if not latest_ts:
            latest_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            
        st.session_state["mre_last_saved_time"] = latest_ts
        
        if rerun:
            st.session_state["mre_show_success_banner"] = True
            st.rerun()

    def render_main(self):
        self.render_section_header(
            "Markov & HMM Regime Intelligence Dashboard",
            "State transition forecasting and FDTS confirmation scaling layer"
        )
        
        if not self.symbols:
            st.warning("⚠️ Please select a ticker universe in the sidebar to begin.")
            return

        if st.session_state.pop("mre_show_success_banner", False):
            st.success("✅ **Universe Reprocessed & Saved!** All empirical regime states, HMM matrices, and forecasts have been successfully written to the database.")

        # Setup state to store calculations
        if "mre_scan_completed" not in st.session_state:
            st.session_state["mre_scan_completed"] = False
            st.session_state["mre_results"] = {}
            st.session_state["mre_last_saved_time"] = None
            st.session_state["mre_needs_reprocess"] = False

        # Check if the loaded results match the currently selected watchlist
        loaded_symbols = set(st.session_state["mre_results"].keys())
        selected_symbols = set(s.strip().upper() for s in self.symbols)
        
        if loaded_symbols != selected_symbols:
            st.session_state["mre_scan_completed"] = False
            st.session_state["mre_results"] = {}
            st.session_state["mre_last_saved_time"] = None

        # Determine if we should perform calculations
        should_reprocess = self.run_engine or st.session_state.get("mre_needs_reprocess", False)

        # Try to load existing data from database if not loaded and we are not reprocessing
        if not should_reprocess and not st.session_state["mre_scan_completed"]:
            db_results = {}
            newest_timestamp = None
            all_found = True
            
            for symbol in self.symbols:
                sym = symbol.strip().upper()
                forecast = get_latest_forecast(sym)
                backtest = get_latest_backtest(sym)
                hist_states = get_historical_states(sym)
                trans_matrix, state_list = get_latest_transition_matrix(sym)
                
                if not forecast or not backtest or not hist_states or trans_matrix is None:
                    all_found = False
                    break
                    
                df_db = pd.DataFrame(hist_states)
                df_db = df_db.rename(columns={"close_price": "close"})
                
                db_results[sym] = {
                    "df": df_db,
                    "hmm_model": None,
                    "state_labels": ["BULL", "SIDEWAYS", "BEAR"],
                    "trans_matrix": trans_matrix,
                    "state_list": state_list,
                    "forecast": forecast,
                    "backtest": backtest
                }
                
                ts = forecast.get("created_at")
                if ts:
                    if not newest_timestamp or ts > newest_timestamp:
                        newest_timestamp = ts
                        
            if all_found and db_results:
                st.session_state["mre_results"] = db_results
                st.session_state["mre_scan_completed"] = True
                st.session_state["mre_last_saved_time"] = newest_timestamp
                st.session_state["mre_needs_reprocess"] = False

        # Render status banner and reprocess button
        if st.session_state["mre_scan_completed"] and st.session_state.get("mre_last_saved_time"):
            c_info, c_action = st.columns([3, 1], vertical_alignment="center")
            with c_info:
                st.info(f"💾 **Using Processed Markov Models** | Data Saved: `{st.session_state['mre_last_saved_time']}` (UTC)")
            with c_action:
                if st.button("🔄 Reprocess Universe", key="mre_force_reprocess", width="stretch", type="primary"):
                    st.session_state["mre_scan_completed"] = False
                    st.session_state["mre_results"] = {}
                    st.session_state["mre_last_saved_time"] = None
                    st.session_state["mre_needs_reprocess"] = True
                    st.rerun()

        # Prompt for scan if no data exists and no scan has run
        if not st.session_state["mre_scan_completed"] and not should_reprocess:
            st.warning("⚠️ No processed regime model data found in database for this universe. You must run a fresh scan.")
            if st.button("🚀 Execute Markov Regime Scan Now", key="mre_first_run_btn", width="stretch", type="primary"):
                st.session_state["mre_needs_reprocess"] = True
                st.rerun()
            return

        if should_reprocess:
            with st.spinner(f"Processing regime engine models across '{self.universe_name}'..."):
                self.execute_regime_scan(
                    universe_name=self.universe_name,
                    symbols=self.symbols,
                    lookback_years=self.lookback_years,
                    n_states=self.n_states,
                    rerun=False
                )
                st.session_state["mre_show_success_banner"] = True
                st.rerun()
                
        results = st.session_state["mre_results"]
        if not results:
            st.error("No symbols could be processed successfully. Please adjust your lookback or symbols list.")
            return

        # Select ticker to display details
        ticker_names = get_ticker_names(self.universe_name)
        ticker_list = sorted(list(results.keys()))
        
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        with c1:
            selected_ticker = st.selectbox(
                "Select Ticker to Analyze",
                ticker_list,
                format_func=lambda ticker: format_ticker_display(ticker, ticker_names),
                key="mre_selected_ticker"
            )
        with c2:
            if st.button("Re-run Models", key="mre_rerun", width="stretch"):
                st.session_state["mre_scan_completed"] = False
                st.session_state["mre_needs_reprocess"] = True
                st.rerun()

        res = results[selected_ticker]
        df = res["df"]
        forecast = res["forecast"].copy()
        if "final_action" not in forecast:
            forecast["final_action"] = "Hold"
        bt = res["backtest"]
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Regime Summary", 
            "Transition Heatmap & Timeline", 
            "Probability Trends", 
            "FDTS Confirmation", 
            "Options Strategy Filter",
            "Help / Engine Guide"
        ])
        
        # Latest data point for metrics
        latest = df.iloc[-1]
        
        with tab1:
            st.markdown("### Regime Summary Analysis")
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            
            # State color
            state_color = "#3ab54a" if latest["price_state"] == "BULL" else "#ef4444" if latest["price_state"] == "BEAR" else "#94a3b8"
            m1.metric("Empirical Price State", latest["price_state"], delta=None, delta_color="normal")
            m2.metric("Volatility State", latest["volatility_state"])
            m3.metric("Markov Signal Score", f"{forecast['markov_signal']:.2f}")
            m4.metric("Stickiness / Exp. Duration", f"{forecast['stickiness_score']:.1%} / {forecast['expected_duration']:.1f} days")
            
            # Action Banner
            action_color = {
                "Deploy": "rgba(58,181,74,0.15)",
                "Watch": "rgba(245,158,11,0.15)",
                "Hold": "rgba(148,163,184,0.15)",
                "Reduce": "rgba(239,68,68,0.15)",
                "Exit or Hedge": "rgba(239,68,68,0.25)"
            }.get(forecast["final_action"], "rgba(21,40,71,0.8)")
            
            text_color = {
                "Deploy": "#fdba74" if "Watch" in forecast["final_action"] else "#3ab54a",
                "Watch": "#f59e0b",
                "Hold": "#94a3b8",
                "Reduce": "#f87171",
                "Exit or Hedge": "#f87171"
            }.get(forecast["final_action"], "#e2e8f0")
            
            st.markdown(
                f"""
                <div style="
                    background: {action_color};
                    border: 1px solid {text_color};
                    border-radius: 8px;
                    padding: 16px 20px;
                    margin: 10px 0 20px 0;
                    text-align: center;
                ">
                    <span style="color:#94a3b8;font-size:12px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Recommended Regime Action</span>
                    <h2 style="color:{text_color};margin:6px 0 0 0;font-size:28px;font-family:'Courier Prime',monospace;">{forecast['final_action']}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Forecast Probabilities Table
            st.markdown("#### Multi-Day Forecast Probabilities")
            f_df = pd.DataFrame([
                {"Horizon": "1-Day Forecast", "Bull (Long)": f"{forecast['bull_prob_1d']:.1%}", "Sideways (Neutral)": f"{forecast['sideways_prob_1d']:.1%}", "Bear (Short)": f"{forecast['bear_prob_1d']:.1%}"},
                {"Horizon": "5-Day Forecast", "Bull (Long)": f"{forecast['bull_prob_5d']:.1%}", "Sideways (Neutral)": f"{forecast['sideways_prob_5d']:.1%}", "Bear (Short)": f"{forecast['bear_prob_5d']:.1%}"},
                {"Horizon": "20-Day Forecast", "Bull (Long)": f"{forecast['bull_prob_20d']:.1%}", "Sideways (Neutral)": f"{forecast['sideways_prob_20d']:.1%}", "Bear (Short)": f"{forecast['bear_prob_20d']:.1%}"}
            ])
            st.dataframe(f_df, width="stretch", hide_index=True)
            
            # Backtest Strategy Summary
            st.markdown("#### Backtest Validation Metrics (Walk-Forward)")
            bm1, bm2, bm3, bm4 = st.columns(4)
            bm1.metric("Walk-Forward Accuracy", f"{bt['prediction_accuracy']:.1f}%")
            bm2.metric("Sharpe Ratio", f"{bt['sharpe_ratio']}")
            bm3.metric("Max Drawdown", f"-{bt['max_drawdown']:.1f}%")
            bm4.metric("Strategy Total Return", f"{bt['total_return']:.1f}%")

        with tab2:
            st.markdown("### Regime Transition Probability & Timeline")
            col1, col2 = st.columns([1, 2])
            with col1:
                # Heatmap
                fig_hm = generate_transition_heatmap(res["trans_matrix"], res["state_list"])
                st.plotly_chart(fig_hm, width="stretch", key="mre_hm_plot")
            with col2:
                # Timeline
                fig_tl = generate_regime_timeline(df, selected_ticker)
                st.plotly_chart(fig_tl, width="stretch", key="mre_tl_plot")
                
        with tab3:
            st.markdown("### Historical Probability Trend Analysis")
            st.caption("Continuous 1-day transition probabilities and Markov signal score calculated dynamically over the lookback history.")
            hist_forecasts = self._calculate_historical_rolling_probabilities(df)
            if not hist_forecasts.empty:
                fig_prob = generate_probability_trend(hist_forecasts, selected_ticker)
                st.plotly_chart(fig_prob, width="stretch", key="mre_prob_trend_plot")
            else:
                st.info("Insufficient historical price history to calculate rolling probability trends.")

        with tab4:
            st.markdown("### FDTS + HMM Confirmation Layer")
            
            # Retrieve latest FDTS decision from current forecast (calculated live during scan or loaded from DB)
            fdts_info = {
                "signal": forecast.get("fdts_signal", "Neutral"),
                "score": forecast.get("fdts_score", 50.0)
            }
            
            # Combined Regime Confirmation Score calculation
            fdts_score = fdts_info["score"]
            markov_sig_score = float((forecast["markov_signal"] + 1.0) * 50.0) # Map [-1, 1] to [0, 100]
            stickiness_score = float(forecast["stickiness_score"] * 100.0)
            
            regime_confirmation_score = (
                fdts_score * 0.50 +
                markov_sig_score * 0.30 +
                stickiness_score * 0.20
            )
            
            # Decision mapping based on confirmation
            if fdts_info["signal"] == "Buy" and latest["price_state"] == "BULL":
                deploy_action = "Strong Deploy (Double Size)"
                deploy_color = "#3ab54a"
            elif fdts_info["signal"] == "Buy" and latest["price_state"] == "SIDEWAYS":
                deploy_action = "Watch / Half Size"
                deploy_color = "#f59e0b"
            elif fdts_info["signal"] == "Buy" and latest["price_state"] == "BEAR":
                deploy_action = "Avoid Longs"
                deploy_color = "#ef4444"
            elif fdts_info["signal"] == "Sell" and latest["price_state"] == "BEAR":
                deploy_action = "Exit / Deploy Hedge"
                deploy_color = "#ef4444"
            else:
                deploy_action = "Hold Positions"
                deploy_color = "#94a3b8"
                
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("FDTS Signal Status", f"{format_fdts_signal(fdts_info['signal'])} (Score: {fdts_score:.1f})")
            fc2.metric("Regime Confirmation Score", f"{regime_confirmation_score:.1f}")
            
            with fc3:
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(21, 40, 71, 0.8);
                        border: 1px solid {deploy_color};
                        border-radius: 8px;
                        padding: 10px 14px;
                        text-align: center;
                    ">
                        <span style="color:#94a3b8;font-size:11px;text-transform:uppercase;">Deploy Action</span>
                        <div style="color:{deploy_color};font-weight:700;font-size:15px;margin-top:2px;">{deploy_action}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            # Benchmark/All symbols confirmation table
            st.markdown("#### Universe Confirmation Matrix")
            universe_matrix = []
            for symbol in results.keys():
                sym_forecast = results[symbol]["forecast"]
                sym_latest = results[symbol]["df"].iloc[-1]
                sym_fdts = {
                    "signal": sym_forecast.get("fdts_signal", "Neutral"),
                    "score": sym_forecast.get("fdts_score", 50.0)
                }
                
                sym_markov_sig = float((sym_forecast["markov_signal"] + 1.0) * 50.0)
                sym_stickiness = float(sym_forecast["stickiness_score"] * 100.0)
                sym_confirm = (
                    sym_fdts["score"] * 0.50 +
                    sym_markov_sig * 0.30 +
                    sym_stickiness * 0.20
                )
                
                universe_matrix.append({
                    "Ticker": symbol,
                    "FDTS Signal": format_fdts_signal(sym_fdts["signal"]),
                    "FDTS Score": round(sym_fdts["score"], 1),
                    "Markov State": sym_latest["price_state"],
                    "Markov Signal": round(sym_forecast["markov_signal"], 2),
                    "Stickiness": f"{sym_forecast['stickiness_score']:.1%}",
                    "Confirmation Score": round(sym_confirm, 1)
                })
            st.dataframe(pd.DataFrame(universe_matrix), width="stretch", hide_index=True)

        with tab5:
            st.markdown("### Calendar Options Deployment Filter")
            st.caption("Scan tickers for long Calendar Spread setups based on high stickiness and bull/sideways regime characteristics.")
            
            # Generate the candidates from the entire universe first (to compute aggregate universe metrics)
            candidates = []
            for symbol in results.keys():
                sym_forecast = results[symbol]["forecast"]
                sym_latest = results[symbol]["df"].iloc[-1]
                sym_fdts = {
                    "signal": sym_forecast.get("fdts_signal", "Neutral"),
                    "score": sym_forecast.get("fdts_score", 50.0)
                }
                
                # Rule: Deploy if:
                # - FDTS == Buy
                # - Markov State == Bull or Sideways
                # - Bear Prob < 30%
                # - Stickiness > 60%
                # - Volatility State != Crash (we mock this as realized vol < 45%)
                
                bear_prob_1d = sym_forecast["bear_prob_1d"]
                stickiness = sym_forecast["stickiness_score"]
                realized_vol = sym_latest["realized_vol_20d"]
                
                is_calendar_candidate = (
                    sym_fdts["signal"] == "Buy" and
                    sym_latest["price_state"] in ["BULL", "SIDEWAYS"] and
                    bear_prob_1d < 0.30 and
                    stickiness > 0.60 and
                    realized_vol < 0.45
                )
                
                deploy_status = "Deploy Setup" if is_calendar_candidate else "Watch" if (bear_prob_1d < 0.40 and stickiness > 0.50) else "Avoid"
                
                candidates.append({
                    "Ticker": symbol,
                    "Calendar Setup": "✅ Yes" if is_calendar_candidate else "❌ No",
                    "FDTS Signal": format_fdts_signal(sym_fdts["signal"]),
                    "Markov State": sym_latest["price_state"],
                    "Bear Probability (1D)": f"{bear_prob_1d:.1%}",
                    "Stickiness": f"{stickiness:.1%}",
                    "Realized Volatility": f"{realized_vol:.1%}",
                    "Action Status": deploy_status
                })
            
            # Calculate KPIs from the full candidate list (universe statistics)
            total_tickers = len(candidates)
            calendar_setup_yes = sum(1 for c in candidates if c["Calendar Setup"] == "✅ Yes")
            deploy_setup_cnt = sum(1 for c in candidates if c["Action Status"] == "Deploy Setup")
            watch_cnt = sum(1 for c in candidates if c["Action Status"] == "Watch")
            avoid_cnt = sum(1 for c in candidates if c["Action Status"] == "Avoid")
            
            # Display KPI Cards
            kpi_cols = st.columns(5)
            kpi_cols[0].metric("Universe Tickers", total_tickers)
            kpi_cols[1].metric("Calendar Setup (Yes)", calendar_setup_yes)
            kpi_cols[2].metric("Deploy Setup", deploy_setup_cnt)
            kpi_cols[3].metric("Watch", watch_cnt)
            kpi_cols[4].metric("Avoid", avoid_cnt)
            
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            
            # Multi-select filter for Action Status
            filter_options = st.multiselect(
                "Filter Tickers by Action Status",
                options=["Deploy Setup", "Watch", "Avoid"],
                default=["Deploy Setup", "Watch"],
                key="mre_tab5_filter"
            )
            
            # Filter candidate rows based on chosen option
            if filter_options:
                filtered_candidates = [c for c in candidates if c["Action Status"] in filter_options]
            else:
                filtered_candidates = []
                
            if not filtered_candidates:
                st.info("No tickers match the active filter criteria.")
            else:
                c_df = pd.DataFrame(filtered_candidates)
                styled_c_df = c_df.style.apply(
                    lambda row: [
                        "background-color: rgba(58, 181, 74, 0.15); color: #e2e8f0;"
                        if row["Action Status"] == "Deploy Setup"
                        else "background-color: rgba(239, 68, 68, 0.1); color: #e2e8f0;"
                        if row["Action Status"] == "Avoid"
                        else ""
                        for _ in row
                    ],
                    axis=1
                )
                st.dataframe(styled_c_df, width="stretch", hide_index=True)

        with tab6:
            st.markdown("### 📘 Hidden Markov Model & Empirical Regime Engine Guide")
            st.markdown("""
            This intelligence layer operates as a quantitative state classifier overlay across portfolio modules. 
            It models transitions dynamically to confirm trends and optimize calendars.
            
            ---
            
            #### 🧠 Core Engine Methodology
            The engine runs two complementary quantitative modules:
            1. **Gaussian Hidden Markov Model (HMM)**: An unsupervised machine learning algorithm that classifies price-action volatility clusters into $N$ states. The model infers the hidden state (BULL, SIDEWAYS, or BEAR) based on the statistical properties of daily returns and rolling 20-day realized volatility.
            2. **Empirical Markov Chains**: Analyzes the historical transition probability matrix $P_{i,j}$ (the likelihood of moving from state $i$ to state $j$ next-day) to project future trends over a multi-day horizon.
            
            ---
            
            #### 🗂️ Dashboard Tab Breakdown
            
            ##### 📈 1. Regime Summary
            * **Metrics**: Price state, Volatility state, Markov signal score ($P(\\text{BULL next}) - P(\\text{BEAR next})$), and **Stickiness** (probability of remaining in the current state).
            * **Forecast Table**: Projected state probabilities for the 1-Day, 5-Day, and 20-Day horizons using matrix power calculations ($P^1$, $P^5$, $P^{20}$).
            * **Backtest**: Walk-forward simulation validation detailing prediction accuracy and historical portfolio Sharpe metrics.
            
            ##### 🗺️ 2. Transition Heatmap & Timeline
            * **Heatmap Matrix**: Displays empirical transition probability frequencies. Higher values along the diagonal indicate strong state stickiness.
            * **Timeline Overlay**: Interactive price chart shaded by the historical regimes, with **orange dots** flagging high-volatility events.
            
            ##### 📊 3. Probability Trends
            * Displays a continuous timeline of rolling 1-Day transition probabilities alongside the Markov Signal Score. This helps identify trend exhaustion and regime shifts.
            
            ##### 🔒 4. FDTS Confirmation Layer
            * Combines **FDTS Opportunity Score** (from the Calendar Scoring module) and the HMM **Markov Signal** to formulate a **Regime Confirmation Score** ($50\\% \\text{ FDTS} + 30\\% \\text{ Markov} + 20\\% \\text{ Stickiness}$).
            * Outlines scaled sizing actions (e.g., *Double Size* when in a Bull Regime, *Avoid Longs* in Bear Regimes).
            
            ##### 🔍 5. Options Strategy Filter
            * Implements a strict scanner designed for long **Calendar Spread** setups. 
            * Evaluates:
              * **FDTS Signal**: Must be `Buy`.
              * **Regime**: Must be `BULL` or `SIDEWAYS`.
              * **Bear Probability**: Must be $< 30\\%$.
              * **Stickiness**: Must be $> 60\\%$ (to ensure the regime holds during the spread's duration).
              * **Realized Volatility**: Must be $< 45\\%$ (no crash regime).
            """)

    def _process_single_ticker(self, symbol: str, fdts_data: dict = None, lookback_years: int = None, n_states: int = None) -> tuple[str, dict | None]:
        symbol = symbol.strip().upper()
        l_years = lookback_years if lookback_years is not None else self.lookback_years
        n_st = n_states if n_states is not None else self.n_states
        try:
            # 1. Load historical price data
            df = self._load_price_data(symbol, l_years)
            if df.empty or len(df) < 50:
                logger.warning(f"Insufficient historical price data for {symbol}.")
                return symbol, None
                
            # 2. Calculate returns and rolling volatility
            df = self._calculate_returns_and_vol(df)
            
            # 3. Assign states (3-state price logic)
            df = self._assign_empirical_states(df)
            
            # 4. Train Gaussian HMM and assign HMM states
            df, hmm_model, state_labels = train_hmm_model(df, n_states=n_st)
            
            # 5. Build Transition Probability Matrix
            trans_matrix, state_list, trans_counts = self._build_transition_matrix(df)
            
            # 6. Save daily states & transitions to SQLite
            with self.db_lock:
                self._save_states_and_transitions(symbol, df, trans_matrix, state_list, trans_counts)
            
            # 7. Generate Multi-day forecasts & Signal score
            from modules.calendar_scoring.fdts_engine import calculate_fdts_signal
            live_fdts = calculate_fdts_signal(df)
            with self.db_lock:
                forecast = self._generate_forecast(symbol, df, trans_matrix, state_list, live_fdts=live_fdts)
            
            # 8. Run walk-forward backtest
            backtest = run_walk_forward_backtest(df)
            backtest["ticker"] = symbol
            backtest["run_date"] = datetime.now().strftime("%Y-%m-%d")
            backtest["strategy_name"] = "Regime Strategy"
            with self.db_lock:
                save_backtest_results(backtest)
            
            ticker_res = {
                "df": df,
                "hmm_model": hmm_model,
                "state_labels": state_labels,
                "trans_matrix": trans_matrix,
                "state_list": state_list,
                "forecast": forecast,
                "backtest": backtest
            }
            return symbol, ticker_res
        except Exception as e:
            logger.error(f"Error processing MRE engine for {symbol}: {e}", exc_info=True)
            return symbol, None

    def _calculate_historical_rolling_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dynamically calculate historical 1-day transition probabilities and Markov signal
        for each day in the historical DataFrame using a vectorized rolling lookback window.
        This provides a rich historical time-series of probabilities.
        """
        df = df.copy().sort_values("trade_date").reset_index(drop=True)
        n = len(df)
        window = 252  # 1 year lookback
        
        if n <= window:
            return pd.DataFrame()
            
        state_list = ["BULL", "SIDEWAYS", "BEAR"]
        state_to_idx = {s: i for i, s in enumerate(state_list)}
        
        # Pre-convert states to index
        state_idxs = np.array([state_to_idx.get(s, -1) for s in df["price_state"]])
        
        # Calculate transition indicators for index 0 to n-2
        # indicator_matrix of shape (n-1, 9)
        indicators = np.zeros((n - 1, 9))
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                indicators[:, idx] = (state_idxs[:-1] == i) & (state_idxs[1:] == j)
                
        # Calculate rolling sum of transitions using pandas
        # Window size of rolling transitions count is window - 1
        indicators_df = pd.DataFrame(indicators)
        rolling_sums = indicators_df.rolling(window - 1).sum().fillna(0).values # shape (n-1, 9)
        
        # Add Laplace smoothing (0.1 to each count)
        counts = rolling_sums + 0.1 # shape (n-1, 9)
        
        # Reshape to (n-1, 3, 3)
        counts_3d = counts.reshape(-1, 3, 3)
        
        # We need counts ending at t-1 (rolling sum at t-2) and state at t-1
        # t goes from window to n-1 (inclusive)
        # So t-2 goes from window-2 to n-3
        # t-1 goes from window-1 to n-2
        t_2_indices = np.arange(window - 2, n - 2)
        t_1_indices = np.arange(window - 1, n - 1)
        
        curr_state_idxs = state_idxs[t_1_indices]
        
        # Filter for valid states (should be all of them)
        valid_mask = (curr_state_idxs >= 0) & (curr_state_idxs < 3)
        
        if not np.any(valid_mask):
            return pd.DataFrame()
            
        t_2_indices = t_2_indices[valid_mask]
        t_1_indices = t_1_indices[valid_mask]
        curr_state_idxs = curr_state_idxs[valid_mask]
        
        selected_counts = counts_3d[t_2_indices, curr_state_idxs, :] # shape (num_valid, 3)
        row_sums = selected_counts.sum(axis=1, keepdims=True)
        probs = selected_counts / row_sums
        
        bull_probs = probs[:, 0]
        sideways_probs = probs[:, 1]
        bear_probs = probs[:, 2]
        signals = bull_probs - bear_probs
        
        dates = df.loc[t_1_indices + 1, "trade_date"].values
        
        return pd.DataFrame({
            "as_of_date": dates,
            "bull_prob_1d": bull_probs,
            "sideways_prob_1d": sideways_probs,
            "bear_prob_1d": bear_probs,
            "markov_signal": signals
        })

    def _load_price_data(self, symbol: str, lookback_years: int) -> pd.DataFrame:
        """Load daily price bars from daily_prices SQLite cache or download fallback."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years * 365)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Check SQLite cache
        import sqlite3
        conn = sqlite3.connect(get_db_path("options_liquidity"), timeout=15)
        try:
            sql = """
            SELECT date, open, high, low, close, volume 
            FROM daily_prices 
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
            """
            df_cache = pd.read_sql_query(sql, conn, params=(symbol, start_str, end_str))
            if not df_cache.empty and len(df_cache) > 200:
                logger.info(f"Loaded {len(df_cache)} price rows for {symbol} from cache.")
                df_cache = df_cache.rename(columns={"date": "trade_date"})
                return df_cache
        except Exception as e:
            logger.warning(f"Failed to query price cache: {e}")
        finally:
            conn.close()
            
        # Fallback to yfinance download
        try:
            logger.info(f"Downloading yfinance fallback data for {symbol}...")
            hist = self._fetch_yfinance(symbol, period=f"{lookback_years}y", interval="1d")
            if hist is not None and not hist.empty:
                df = hist.reset_index()
                df = df.rename(columns={
                    "Date": "trade_date", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume"
                })
                # Ensure date string
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
                
                # Cache to daily_prices asynchronously
                self._cache_price_data(symbol, df)
                return df[["trade_date", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.error(f"Failed to download price data for {symbol}: {e}")
            
        return pd.DataFrame()

    def _cache_price_data(self, symbol: str, df: pd.DataFrame):
        """Asynchronously cache downloaded pricing data into daily_prices SQLite."""
        with self.db_lock:
            try:
                import sqlite3
                conn = sqlite3.connect(get_db_path("options_liquidity"), timeout=15)
                cursor = conn.cursor()
                records = [
                    (
                        row["trade_date"], symbol, row["open"], row["high"],
                        row["low"], row["close"], row["volume"], 0.0
                    )
                    for _, row in df.iterrows()
                ]
                cursor.executemany("""
                    INSERT OR REPLACE INTO daily_prices (date, symbol, open, high, low, close, volume, open_interest)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                conn.commit()
                conn.close()
                logger.info(f"Cached {len(df)} price rows for {symbol} to daily_prices.")
            except Exception as e:
                logger.warning(f"Error caching pricing data: {e}")

    def _calculate_returns_and_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        df["daily_return"] = df["close"].pct_change()
        df["rolling_20d_return"] = df["close"] / df["close"].shift(20) - 1.0
        df["realized_vol_20d"] = df["daily_return"].rolling(20).std() * np.sqrt(252)
        return df.dropna(subset=["rolling_20d_return"]).reset_index(drop=True)

    def _assign_empirical_states(self, df: pd.DataFrame) -> pd.DataFrame:
        # Price State
        price_states = []
        for r20 in df["rolling_20d_return"]:
            if r20 >= 0.05:
                price_states.append("BULL")
            elif r20 <= -0.05:
                price_states.append("BEAR")
            else:
                price_states.append("SIDEWAYS")
        df["price_state"] = price_states
        
        # Volatility State (70th percentile of rolling vol)
        vol_70pct = df["realized_vol_20d"].quantile(0.70)
        df["volatility_state"] = df["realized_vol_20d"].apply(
            lambda v: "HIGH_VOL" if v > vol_70pct else "LOW_VOL"
        )
        df["combined_state"] = df["price_state"] + "_" + df["volatility_state"]
        return df

    def _build_transition_matrix(self, df: pd.DataFrame) -> tuple:
        state_list = ["BULL", "SIDEWAYS", "BEAR"]
        state_to_idx = {s: i for i, s in enumerate(state_list)}
        
        counts = np.ones((3, 3)) * 0.1  # Laplace smoothing
        
        for t in range(len(df) - 1):
            s_curr = df.loc[t, "price_state"]
            s_next = df.loc[t+1, "price_state"]
            if s_curr in state_to_idx and s_next in state_to_idx:
                counts[state_to_idx[s_curr], state_to_idx[s_next]] += 1.0
                
        row_sums = counts.sum(axis=1, keepdims=True)
        P = counts / row_sums
        return P, state_list, counts

    def _save_states_and_transitions(self, symbol: str, df: pd.DataFrame, P: np.ndarray, state_list: list, counts: np.ndarray):
        trade_date = df.iloc[-1]["trade_date"]
        
        # 1. Save daily state records
        daily_records = []
        for _, row in df.iterrows():
            daily_records.append({
                "trade_date": row["trade_date"],
                "ticker": symbol,
                "close_price": row["close"],
                "daily_return": row["daily_return"],
                "rolling_20d_return": row["rolling_20d_return"],
                "realized_vol_20d": row["realized_vol_20d"],
                "price_state": row["price_state"],
                "volatility_state": row["volatility_state"],
                "combined_state": row["combined_state"]
            })
        save_daily_states(daily_records)
        
        # 2. Save transitions
        trans_records = []
        for i, from_state in enumerate(state_list):
            for j, to_state in enumerate(state_list):
                trans_records.append({
                    "as_of_date": trade_date,
                    "ticker": symbol,
                    "from_state": from_state,
                    "to_state": to_state,
                    "transition_count": int(counts[i, j]),
                    "transition_probability": float(P[i, j]),
                    "lookback_days": len(df)
                })
        save_transition_matrix(trans_records)

    def _generate_forecast(self, symbol: str, df: pd.DataFrame, P: np.ndarray, state_list: list, live_fdts: dict = None) -> dict:
        latest = df.iloc[-1]
        curr_state = latest["price_state"]
        curr_idx = state_list.index(curr_state)
        
        # Matrix Powers
        P1 = P
        P5 = np.linalg.matrix_power(P, 5)
        P20 = np.linalg.matrix_power(P, 20)
        
        # Signal Score: P(BULL next) - P(BEAR next)
        markov_sig = float(P1[curr_idx, state_list.index("BULL")] - P1[curr_idx, state_list.index("BEAR")])
        
        # Stickiness
        stickiness = float(P1[curr_idx, curr_idx])
        expected_duration = float(1.0 / (1.0 - stickiness)) if stickiness < 1.0 else 999.0
        
        # Action classification
        # Pull latest FDTS signal status
        if live_fdts is None:
            from modules.calendar_scoring.fdts_engine import calculate_fdts_signal
            live_fdts = calculate_fdts_signal(df)
        fdts_val = live_fdts.get("signal", "Neutral")
        fdts_score = live_fdts.get("score", 50.0)
        
        bear_prob_1d = float(P1[curr_idx, state_list.index("BEAR")])
        
        if fdts_val == "Buy" and markov_sig > 0.30 and bear_prob_1d < 0.25:
            action = "Deploy"
        elif fdts_val == "Buy" and markov_sig > 0.0 and bear_prob_1d < 0.40:
            action = "Watch"
        elif markov_sig < -0.30 or bear_prob_1d > 0.50 or fdts_val == "Sell":
            action = "Exit or Hedge"
        else:
            action = "Hold"
            
        forecast = {
            "as_of_date": latest["trade_date"],
            "ticker": symbol,
            "current_state": curr_state,
            "bull_prob_1d": float(P1[curr_idx, state_list.index("BULL")]),
            "sideways_prob_1d": float(P1[curr_idx, state_list.index("SIDEWAYS")]),
            "bear_prob_1d": bear_prob_1d,
            "bull_prob_5d": float(P5[curr_idx, state_list.index("BULL")]),
            "sideways_prob_5d": float(P5[curr_idx, state_list.index("SIDEWAYS")]),
            "bear_prob_5d": float(P5[curr_idx, state_list.index("BEAR")]),
            "bull_prob_20d": float(P20[curr_idx, state_list.index("BULL")]),
            "sideways_prob_20d": float(P20[curr_idx, state_list.index("SIDEWAYS")]),
            "bear_prob_20d": float(P20[curr_idx, state_list.index("BEAR")]),
            "markov_signal": markov_sig,
            "stickiness_score": stickiness,
            "expected_duration": expected_duration,
            "final_regime_label": curr_state,
            "final_action": action,
            "fdts_signal": fdts_val,
            "fdts_score": fdts_score
        }
        save_forecast(forecast)
        return forecast

    def _load_historical_forecasts(self, ticker: str) -> pd.DataFrame:
        """Load all historical forecast logs for a specific ticker to plot trends."""
        conn = get_mre_conn()
        try:
            sql = """
            SELECT as_of_date, bull_prob_1d, bear_prob_1d, sideways_prob_1d, markov_signal
            FROM markov_forecast
            WHERE ticker = ?
            ORDER BY as_of_date ASC
            """
            return pd.read_sql_query(sql, conn, params=(ticker,))
        except Exception as e:
            logger.warning(f"Failed to load historical forecast logs: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def _get_latest_fdts_signals(self) -> dict:
        """Fetch the latest FDTS signal status for each ticker from SQL database."""
        from modules.calendar_scoring.database import get_connection as get_cs_conn
        conn = get_cs_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ticker, fdts_signal, fdts_score 
                FROM ticker_decision_log t1
                WHERE decision_id = (
                    SELECT MAX(decision_id) 
                    FROM ticker_decision_log t2 
                    WHERE t2.ticker = t1.ticker
                )
            """)
            rows = cursor.fetchall()
            return {row[0]: {"signal": row[1], "score": float(row[2])} for row in rows}
        except Exception as e:
            logger.warning(f"Could not load FDTS decision logs: {e}. Using baseline proxies.")
            return {}
        finally:
            conn.close()
