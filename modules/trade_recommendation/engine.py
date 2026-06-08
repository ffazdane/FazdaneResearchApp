import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
import yfinance as yf
from datetime import datetime, timedelta

from modules.base_module import FazDaneModule
from utils.universe_manager import load_universes, get_universe_names, get_tickers, format_ticker_display
from modules.trade_recommendation.database import (
    create_tables, save_signal_snapshot, log_trade_outcome,
    fetch_historical_snapshots, fetch_indicator_snapshots,
    fetch_active_trade_plans, fetch_trade_outcomes,
    fetch_latest_ticker_snapshot, deserialize_analysis_data
)
from modules.trade_recommendation.indicators import (
    run_indicators_scan, run_trend_engine, run_regime_engine, run_forecast_engine
)
from modules.trade_recommendation.selector import (
    calculate_component_scores, calculate_composite_score,
    select_trade_strategy, adjust_decision_for_alignment
)
from modules.trade_recommendation.plan_generator import generate_trade_plan
from modules.calendar_scoring.data_loader import fetch_option_chain_data
from modules.calendar_scoring.trade_setup_engine import select_calendar_setup
from modules.tier1.calendar_rotation import load_consolidated_recommendations

logger = logging.getLogger("TradeRecommendationEngine")

class TradeRecommendationEngineModule(FazDaneModule):
    MODULE_NAME = "Trade Intelligence Engine"
    MODULE_ICON = "🎯"
    MODULE_DESCRIPTION = "Standardized trade decision classifier, options strategy selector, and professional trade plan generator."
    TIER = 1

    def __init__(self):
        super().__init__()
        create_tables()

    def render_sidebar(self):
        st.markdown("### Universe Selector")
        names = get_universe_names("general")
        
        if not names:
            st.warning("No ticker universes are available.")
            return

        # Default to "Options Default Watchlist" (index 0)
        default_universe_idx = 0
        if "re_universe_sel" not in st.session_state:
            st.session_state["re_universe_sel"] = names[0]
            
        selected_universe = st.selectbox(
            "Select Watchlist / Universe",
            options=names,
            key="re_universe_sel"
        )
        tickers = get_tickers(selected_universe)
        
        if not tickers:
            st.warning("Selected universe is empty.")
            return
            
        # Persist tickers list
        prev_tickers = st.session_state.get("re_tickers", [])
        st.session_state["re_tickers"] = tickers
        
        # Auto-select first ticker when: none selected, or previous ticker not in new list
        current = st.session_state.get("re_ticker")
        if not current or current not in tickers:
            st.session_state["re_ticker"] = tickers[0]
        
        st.divider()
        st.markdown("### Settings")
        st.slider("Probability Cone Confidence Interval (Z)", min_value=1.0, max_value=3.0, value=1.64, step=0.1, key="re_cone_z")
        st.checkbox("Use Synthetic Options Chain (Fast Mode)", value=False, key="re_use_synthetic")
        
        # Show active ticker as read-only info (selection is via table row click in main area)
        active = st.session_state.get("re_ticker")
        if active:
            st.divider()
            st.caption("🟡 Active Ticker")
            st.markdown(f"### {active}")

    def render_main(self):
        tickers = st.session_state.get("re_tickers", [])
        active_ticker = st.session_state.get("re_ticker")
        
        # Safety: if still no ticker (first ever load before sidebar ran), show a spinner
        # and wait — sidebar will set it on the same rerun.
        if not tickers:
            st.info("🟡 Select a universe in the sidebar to begin.")
            return
        if not active_ticker or active_ticker not in tickers:
            # Auto-set to first ticker and rerun so the table loads immediately
            st.session_state["re_ticker"] = tickers[0]
            st.rerun()
            
        st.markdown(f"## 🎯 Trade Intelligence Dashboard")
        
        # 1. Universe Overview Table
        st.markdown("### 📋 Watchlist Summary Overview")
        
        col_gen, col_mode, col_spacing = st.columns([1.2, 1.5, 1.8])
        with col_gen:
            if st.button("🔄 Generate / Update Watchlist Data", use_container_width=True):
                self._generate_watchlist_data(tickers)
        with col_mode:
            view_mode = st.selectbox(
                "Display Columns View",
                options=["Standard Watchlist", "Consolidated Matrix View"],
                key="re_view_mode_sel",
                label_visibility="collapsed"
            )
                
        df_universe = self._get_universe_summary_table(tickers)
        
        if not df_universe.empty:
            # Filter columns based on selected view mode
            standard_cols = [
                "Ticker", "Company Name", "Ticker Price", "Net Change", "ATR", "FDTS", "Strength", 
                "Earnings Date", "Score", "Decision", "State", "Regime", "Strategy", "40D Range", "Last Updated"
            ]
            consolidated_cols = [
                "Ticker", "Company Name", "Ticker Price", "Net Change", "Alignment", "Cal Score Rec", 
                "Price Action Rec", "Markov Setup", "Option Bias", "Score", "Decision", "Strategy", "Earnings Date"
            ]
            
            if view_mode == "Consolidated Matrix View":
                cols_to_use = [c for c in consolidated_cols if c in df_universe.columns]
            else:
                cols_to_use = [c for c in standard_cols if c in df_universe.columns]
                
            display_df = df_universe[cols_to_use]
            
            # ── Watchlist Filtering ───────────────────────────────────────────
            # Dynamically filter by multiple columns and values
            if "re_active_filters" not in st.session_state:
                st.session_state["re_active_filters"] = {}
                
            # Automatically clean up active filters that are not in the selected cols_to_use
            st.session_state["re_active_filters"] = {
                k: v for k, v in st.session_state["re_active_filters"].items()
                if k in cols_to_use
            }

            def clear_filters():
                st.session_state["re_active_filters"] = {}
                st.session_state["re_filter_col"] = "-- Select Column --"
                st.session_state["re_filter_val"] = "-- Select Value --"

            def add_filter():
                col = st.session_state.get("re_filter_col")
                val = st.session_state.get("re_filter_val")
                if col and col != "-- Select Column --" and val and val != "-- Select Value --":
                    st.session_state["re_active_filters"][col] = val
                    st.session_state["re_filter_col"] = "-- Select Column --"
                    st.session_state["re_filter_val"] = "-- Select Value --"

            filter_cols_to_use = [c for c in cols_to_use if c not in {"Ticker Price", "Net Change", "ATR", "40D Range", "Last Updated", "Company Name"}]
            filter_options = ["-- Select Column --"] + filter_cols_to_use
            
            current_filter_col = st.session_state.get("re_filter_col", "-- Select Column --")
            if current_filter_col not in filter_options:
                st.session_state["re_filter_col"] = "-- Select Column --"
                if "re_filter_val" in st.session_state:
                    st.session_state["re_filter_val"] = "-- Select Value --"
            
            filt_col, filt_val, filt_btn = st.columns([1.5, 1.5, 1.0])
            with filt_col:
                selected_col = st.selectbox(
                    "🔍 Filter by Column Name",
                    options=filter_options,
                    key="re_filter_col"
                )
                
            if selected_col != "-- Select Column --":
                # Get unique values of this column in the universe table and sort them
                unique_vals = sorted(df_universe[selected_col].dropna().unique().astype(str).tolist())
                val_options = ["-- Select Value --"] + unique_vals
                
                current_filter_val = st.session_state.get("re_filter_val", "-- Select Value --")
                if current_filter_val not in val_options:
                    st.session_state["re_filter_val"] = "-- Select Value --"
                    
                with filt_val:
                    selected_val = st.selectbox(
                        "🎯 Select Value",
                        options=val_options,
                        key="re_filter_val"
                    )
                with filt_btn:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    is_val_selected = (selected_val != "-- Select Value --")
                    st.button(
                        "➕ Add Filter", 
                        use_container_width=True, 
                        on_click=add_filter,
                        disabled=not is_val_selected,
                        type="primary"
                    )
            else:
                with filt_val:
                    st.selectbox(
                        "🎯 Select Value",
                        options=["-- Select a Column First --"],
                        disabled=True,
                        key="re_filter_val_disabled"
                    )
                with filt_btn:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    st.button("➕ Add Filter", disabled=True, use_container_width=True)
            
            # Apply all active filters to the dataframe
            for filter_col, filter_val in st.session_state["re_active_filters"].items():
                if filter_col in display_df.columns:
                    display_df = display_df[display_df[filter_col].astype(str) == filter_val]
            
            # Display active filters as tags/buttons
            active_filters = st.session_state["re_active_filters"]
            if active_filters:
                st.markdown("**Active Filters (Refined):**")
                filter_items = list(active_filters.items())
                tag_cols = st.columns(len(filter_items) + 1)
                
                for idx, (f_col, f_val) in enumerate(filter_items):
                    with tag_cols[idx]:
                        # Closure builder to capture the key to delete
                        def make_delete_callback(col_to_del):
                            return lambda: st.session_state["re_active_filters"].pop(col_to_del, None)
                        
                        st.button(
                            f"❌ {f_col}: {f_val}",
                            key=f"rm_filt_{f_col}",
                            on_click=make_delete_callback(f_col),
                            use_container_width=True
                        )
                with tag_cols[-1]:
                    st.button("❌ Clear All", on_click=clear_filters, use_container_width=True, type="secondary")
                st.write("")
            
            if display_df.empty:
                st.warning("⚠️ No tickers match the current filter criteria.")
            else:
                tickers_list = sorted(list(display_df["Ticker"].unique()))
                tickers_str = ", ".join(tickers_list)
                st.markdown(
                    f"<div style='font-size:14px; font-weight:600; color:#94a3b8; margin-top:8px; margin-bottom:4px;'>"
                    f"📋 Copy Filtered Tickers ({len(tickers_list)} symbols)</div>", 
                    unsafe_allow_html=True
                )
                st.code(tickers_str, language="text")
                st.write("")
                
            # ── Single-row selection using native st.dataframe on_select ────────
            # This replaces the multi-select data_editor approach and gives clean
            # single-row click-to-select behaviour with a highlighted active row.
            styled_df = display_df.style.apply(self._apply_watchlist_styles, axis=None)
            event = st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="universe_table"
            )
            selected_rows = event.selection.rows if event.selection else []
            if selected_rows:
                chosen = display_df.iloc[selected_rows[0]]["Ticker"]
                if chosen != active_ticker:
                    st.session_state["re_ticker"] = chosen
                    st.rerun()

        st.markdown("---")
            
        # Fetch data for active ticker
        with st.spinner(f"Loading analysis for {active_ticker}..."):
            data = self._get_ticker_analysis_data(active_ticker)
            
        if not data:
            st.warning(f"⚠️ No cached analysis found for ticker {active_ticker}.")
            # Provide button to scan it now
            if st.button(f"🔍 Generate / Scan {active_ticker} Data Now", type="primary", use_container_width=True):
                with st.spinner(f"Scanning {active_ticker}..."):
                    import time
                    data = self._get_ticker_analysis_data(active_ticker, refresh_token=time.time())
                if data:
                    st.success(f"Successfully generated data for {active_ticker}!")
                    st.rerun()
                else:
                    st.error(f"Failed to generate data for {active_ticker}.")
            return
            
        # Header with timestamp of generation
        gen_time = data.get("snapshot_datetime")
        is_cached = data.get("from_db_cache", False)
        
        col_header, col_ref = st.columns([3, 1])
        with col_header:
            st.markdown(f"### 🔍 Detailed Analysis: {active_ticker}")
            if gen_time:
                source_lbl = "Database Cache" if is_cached else "Live Scan"
                st.caption(f"📅 **Data Version**: {gen_time} ({source_lbl})")
        with col_ref:
            if st.button(f"🔄 Regenerate {active_ticker} Data", use_container_width=True, type="secondary"):
                with st.spinner(f"Scanning {active_ticker}..."):
                    import time
                    self._get_ticker_analysis_data(active_ticker, refresh_token=time.time())
                st.success(f"Data regenerated for {active_ticker}!")
                st.rerun()
                
        # Navigation tabs for 8 screens
        tab_cc, tab_ss, tab_fm, tab_sm, tab_oc, tab_pm, tab_bo, tab_rt = st.tabs([
            "Command Center",
            "Strategy Selector",
            "40D Forecast Map",
            "Signal Matrix",
            "Option Chain",
            "Portfolio Monitor",
            "Backtest Outcomes",
            "Rejected Log"
        ])
        
        # 1. COMMAND CENTER SCREEN
        with tab_cc:
            self._render_command_center(data)
            
        # 2. STRATEGY SELECTOR SCREEN
        with tab_ss:
            self._render_strategy_selector(data)
            
        # 3. 40-DAY FORECAST MAP SCREEN
        with tab_fm:
            self._render_forecast_map(data)
            
        # 4. SIGNAL MATRIX SCREEN
        with tab_sm:
            self._render_signal_matrix(data)
            
        # 5. OPTION CHAIN SELECTOR SCREEN
        with tab_oc:
            self._render_option_chain_selector(data)
            
        # 6. PORTFOLIO MONITOR SCREEN
        with tab_pm:
            self._render_portfolio_monitor()
            
        # 7. BACKTEST OUTCOME SCREEN
        with tab_bo:
            self._render_backtest_outcomes()
            
        # 8. REJECTED TRADES LOG SCREEN
        with tab_rt:
            self._render_rejected_log()

    # ══════════════════════════════════════════════════════════════════════
    # ANALYTICAL DATA FETCHING & PIPELINE
    # ══════════════════════════════════════════════════════════════════════

    def _get_universe_summary_table(self, tickers: list) -> pd.DataFrame:
        """Compile a summary table for all tickers in the universe from database snapshots."""
        rows = []
        from utils.universe_manager import get_company_name
        from modules.trade_recommendation.database import fetch_latest_ticker_snapshot, deserialize_analysis_data
        from utils.formatting import calculate_strength_pct, format_strength_meter
        
        # Load consolidated rotation recommendations
        cal_dict = {}
        try:
            df_cal = load_consolidated_recommendations(tickers)
            if not df_cal.empty:
                for _, r in df_cal.iterrows():
                    cal_dict[r["ticker"]] = r.to_dict()
        except Exception as e:
            logger.error(f"Failed to load consolidated rotation recommendations: {e}")

        for t in tickers:
            snap = fetch_latest_ticker_snapshot(t)
            
            # Lookup cal recommendations
            cal_data = cal_dict.get(t, {})
            cs_rec = cal_data.get("cs_rec", "N/A")
            pa_display_rec = cal_data.get("pa_display_rec", "N/A")
            mre_display_rec = cal_data.get("mre_display_rec", "N/A")
            ol_bias = cal_data.get("ol_bias", "N/A")

            # Calculate alignment index (0-5)
            score_count = 0
            if cs_rec in ["Deploy", "Watch"]:
                score_count += 1
            if isinstance(pa_display_rec, str) and pa_display_rec.startswith("🟢"):
                score_count += 1
            if mre_display_rec == "✅ Yes":
                score_count += 1
            if ol_bias in ["🟢 Call Heavy", "⚪ Balanced"]:
                score_count += 1

            if snap:
                raw_json = snap.get("raw_analysis_json")
                daily_data = {}
                if raw_json:
                    try:
                        daily_data = deserialize_analysis_data(raw_json)
                    except Exception as e:
                        logger.error(f"Failed to deserialize raw analysis json for {t}: {e}")
                
                df_daily = daily_data.get("df_daily")
                
                # 1. Ticker Price
                price_val = snap.get("price") or daily_data.get("spot_price")
                price_str = f"${price_val:.2f}" if price_val else "N/A"
                
                # 2. Net Change
                net_change_str = "N/A"
                if df_daily is not None and not df_daily.empty and len(df_daily) >= 2:
                    try:
                        close_today = float(df_daily["Close"].iloc[-1])
                        close_yesterday = float(df_daily["Close"].iloc[-2])
                        nc_val = close_today - close_yesterday
                        nc_pct = (nc_val / close_yesterday) * 100
                        if nc_val > 0:
                            net_change_str = f"+${abs(nc_val):.2f} (+{abs(nc_pct):.2f}%)"
                        elif nc_val < 0:
                            net_change_str = f"-${abs(nc_val):.2f} (-{abs(nc_pct):.2f}%)"
                        else:
                            net_change_str = "$0.00 (0.00%)"
                    except Exception:
                        pass
                
                # 3. ATR
                daily_ind = daily_data.get("daily_indicators", {})
                atr_val = daily_ind.get("atr14")
                atr_str = f"${atr_val:.2f}" if atr_val else "N/A"
                
                # 4. FDTS
                fdts_signal = daily_ind.get("fdts_signal", "Neutral")
                fdts_emoji = {"Buy": "🟢 Buy", "Sell": "🔴 Sell", "Neutral": "⚪ Neutral", "No Trade": "⚪ No Trade"}.get(fdts_signal, f"⚪ {fdts_signal}")
                
                # 5. Strength
                strength_pct = calculate_strength_pct(df_daily)
                strength_icon, _ = format_strength_meter(strength_pct)
                
                # 6. Earnings Date
                ed_val = daily_data.get("earnings_date")
                if not ed_val or ed_val == "N/A":
                    try:
                        from utils.persistence import get_db_path
                        from utils.earnings_calendar_store import DB_PATH as ec_db_path
                        import sqlite3
                        
                        # 1. Try calendar_scoring first
                        cs_db = get_db_path("calendar_scoring")
                        if cs_db.exists():
                            with sqlite3.connect(cs_db) as conn:
                                today_str = datetime.today().strftime("%Y-%m-%d")
                                query = "SELECT earnings_date FROM ticker_decision_log WHERE ticker = ? AND earnings_date >= ? ORDER BY decision_id DESC LIMIT 1"
                                row = conn.execute(query, (t, today_str)).fetchone()
                                if row and row[0]:
                                    ed_val = row[0]
                                    
                        # 2. Try earnings_calendar fallback
                        if (not ed_val or ed_val == "N/A") and ec_db_path.exists():
                            with sqlite3.connect(ec_db_path) as conn:
                                today_str = datetime.today().strftime("%Y-%m-%d")
                                query = "SELECT MIN(date) FROM ec_earnings_events WHERE ticker = ? AND date >= ?"
                                row = conn.execute(query, (t, today_str)).fetchone()
                                if row and row[0]:
                                    ed_val = row[0]
                    except Exception:
                        pass
                if not ed_val:
                    ed_val = "N/A"
                    
                if ed_val != "N/A":
                    try:
                        days_diff = (datetime.strptime(ed_val, "%Y-%m-%d").date() - datetime.now().date()).days
                        if 0 <= days_diff <= 20:
                            ed_val = f"🔴 {ed_val}"
                        elif days_diff <= 40:
                            ed_val = f"🟡 {ed_val}"
                    except Exception:
                        pass
                
                # 5. TI Decision
                ti_decision = snap["trade_decision"] or "N/A"
                if ti_decision in ["Deploy", "Watch"]:
                    score_count += 1
                
                emoji_map = {
                    5: "🟢 5/5 Aligned",
                    4: "🟢 4/5 Aligned",
                    3: "🟡 3/5 Aligned",
                    2: "🔵 2/5 Aligned",
                    1: "🔴 1/5 Aligned",
                    0: "🔴 0/5 Aligned"
                }
                alignment_str = emoji_map.get(score_count, "🔴 0/5 Aligned")

                rows.append({
                    "Ticker": t,
                    "Company Name": get_company_name(t),
                    "Ticker Price": price_str,
                    "Net Change": net_change_str,
                    "ATR": atr_str,
                    "FDTS": fdts_emoji,
                    "Strength": strength_icon,
                    "Earnings Date": ed_val,
                    "Score": f"{snap['trade_score']:.1f}" if snap['trade_score'] else "N/A",
                    "Decision": ti_decision,
                    "State": snap["trend_state"] or "N/A",
                    "Regime": snap["market_state"] or "N/A",
                    "Strategy": snap["recommended_strategy"] or "N/A",
                    "40D Range": f"${snap['expected_40d_low']:.2f} - ${snap['expected_40d_high']:.2f}" if (snap['expected_40d_low'] and snap['expected_40d_high']) else "N/A",
                    "Last Updated": snap["snapshot_datetime"],
                    
                    "Alignment": alignment_str,
                    "Cal Score Rec": cs_rec,
                    "Price Action Rec": pa_display_rec,
                    "Markov Setup": mre_display_rec,
                    "Option Bias": ol_bias
                })
            else:
                emoji_map = {
                    4: "🟢 4/5 Aligned",
                    3: "🟡 3/5 Aligned",
                    2: "🔵 2/5 Aligned",
                    1: "🔴 1/5 Aligned",
                    0: "🔴 0/5 Aligned"
                }
                alignment_str = emoji_map.get(score_count, "🔴 0/5 Aligned")

                rows.append({
                    "Ticker": t,
                    "Company Name": get_company_name(t),
                    "Ticker Price": "No Data",
                    "Net Change": "N/A",
                    "ATR": "N/A",
                    "FDTS": "⚪ Pending",
                    "Strength": "—",
                    "Earnings Date": "N/A",
                    "Score": "-",
                    "Decision": "Pending Scan",
                    "State": "-",
                    "Regime": "-",
                    "Strategy": "-",
                    "40D Range": "-",
                    "Last Updated": "-",
                    
                    "Alignment": alignment_str,
                    "Cal Score Rec": cs_rec,
                    "Price Action Rec": pa_display_rec,
                    "Markov Setup": mre_display_rec,
                    "Option Bias": ol_bias
                })
        return pd.DataFrame(rows)

    def _apply_watchlist_styles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a same-shape DataFrame of CSS style strings for each cell of the watchlist."""
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        
        for i in df.index:
            row = df.loc[i]
            
            # Ticker
            if "Ticker" in df.columns:
                styles.at[i, "Ticker"] = "font-weight: 700; color: #60a5fa"
            
            # Ticker Price
            if "Ticker Price" in df.columns:
                styles.at[i, "Ticker Price"] = "font-weight: 600;"
            
            # Net Change
            if "Net Change" in df.columns:
                nc = str(row.get("Net Change", ""))
                if nc.startswith("+"):
                    styles.at[i, "Net Change"] = "color: #22c55e; font-weight: 700"
                elif nc.startswith("-"):
                    styles.at[i, "Net Change"] = "color: #ef4444; font-weight: 700"
                
            # FDTS
            if "FDTS" in df.columns:
                fdts = str(row.get("FDTS", ""))
                if "Buy" in fdts:
                    styles.at[i, "FDTS"] = "color: #22c55e; font-weight: 700"
                elif "Sell" in fdts:
                    styles.at[i, "FDTS"] = "color: #ef4444; font-weight: 700"
                else:
                    styles.at[i, "FDTS"] = "color: #94a3b8"
                
            # Strength
            if "Strength" in df.columns:
                st_val = str(row.get("Strength", ""))
                if "▲" in st_val:
                    styles.at[i, "Strength"] = "color: #00D4AA; font-weight: 700; font-size: 18px; text-align: center"
                elif "▼" in st_val:
                    styles.at[i, "Strength"] = "color: #FF4B4B; font-weight: 700; font-size: 18px; text-align: center"
                elif "▶" in st_val:
                    styles.at[i, "Strength"] = "color: #FFA421; font-weight: 700; font-size: 18px; text-align: center"
                else:
                    styles.at[i, "Strength"] = "color: #888888; text-align: center"
                
            # Earnings Date
            if "Earnings Date" in df.columns:
                ed = str(row.get("Earnings Date", ""))
                if "🔴" in ed:
                    styles.at[i, "Earnings Date"] = "background-color: rgba(220,38,38,0.28); color: #ef4444; font-weight: 700"
                elif "🟡" in ed:
                    styles.at[i, "Earnings Date"] = "background-color: rgba(255,184,0,0.22); color: #ffb800; font-weight: 700"
                
            # Decision
            if "Decision" in df.columns:
                dec = str(row.get("Decision", ""))
                if "Deploy" in dec:
                    styles.at[i, "Decision"] = "background-color: rgba(58, 181, 74, 0.22); color: #3ab54a; font-weight: 700"
                elif "Watch" in dec:
                    styles.at[i, "Decision"] = "background-color: rgba(255, 184, 0, 0.18); color: #ffb800; font-weight: 700"
                elif "Wait" in dec:
                    styles.at[i, "Decision"] = "background-color: rgba(2, 132, 199, 0.18); color: #0284c7; font-weight: 700"
                elif "Reject" in dec or "Avoid" in dec:
                    styles.at[i, "Decision"] = "background-color: rgba(220, 38, 38, 0.18); color: #ef4444; font-weight: 700"
                
            # State
            if "State" in df.columns:
                state = str(row.get("State", ""))
                if "Bull" in state:
                    styles.at[i, "State"] = "background-color: rgba(34, 197, 94, 0.15); color: #22c55e; font-weight: 700"
                elif "Bear" in state or "Breakdown" in state:
                    styles.at[i, "State"] = "background-color: rgba(220, 38, 38, 0.15); color: #ef4444; font-weight: 700"
                elif "Late" in state or "Transition" in state or "Mixed" in state or "Overextended" in state:
                    styles.at[i, "State"] = "background-color: rgba(255, 184, 0, 0.15); color: #ffb800; font-weight: 700"
                elif "Sideways" in state or "Range" in state:
                    styles.at[i, "State"] = "background-color: rgba(2, 132, 199, 0.15); color: #0284c7; font-weight: 700"
                
            # Regime
            if "Regime" in df.columns:
                reg = str(row.get("Regime", ""))
                if "Trending" in reg:
                    styles.at[i, "Regime"] = "background-color: rgba(34, 197, 94, 0.15); color: #22c55e; font-weight: 700"
                elif "Volatile" in reg:
                    styles.at[i, "Regime"] = "background-color: rgba(220, 38, 38, 0.15); color: #ef4444; font-weight: 700"
                elif "Compressed" in reg:
                    styles.at[i, "Regime"] = "background-color: rgba(139, 92, 246, 0.15); color: #a78bfa; font-weight: 700"
                elif "Mean Reverting" in reg:
                    styles.at[i, "Regime"] = "background-color: rgba(2, 132, 199, 0.15); color: #0284c7; font-weight: 700"
                
            # Strategy
            if "Strategy" in df.columns:
                strat = str(row.get("Strategy", ""))
                if "Reject" in strat:
                    styles.at[i, "Strategy"] = "background-color: rgba(220, 38, 38, 0.15); color: #ef4444; font-weight: 700"
                elif strat != "-" and strat != "N/A" and strat != "Pending Scan":
                    styles.at[i, "Strategy"] = "background-color: rgba(58, 181, 74, 0.15); color: #3ab54a; font-weight: 700"

            # Alignment
            if "Alignment" in df.columns:
                align = str(row.get("Alignment", ""))
                if "5/5" in align or "4/5" in align:
                    styles.at[i, "Alignment"] = "background-color: rgba(58, 181, 74, 0.22); color: #3ab54a; font-weight: 700"
                elif "3/5" in align:
                    styles.at[i, "Alignment"] = "background-color: rgba(255, 184, 0, 0.18); color: #ffb800; font-weight: 700"
                elif "2/5" in align:
                    styles.at[i, "Alignment"] = "background-color: rgba(2, 132, 199, 0.18); color: #0284c7; font-weight: 700"
                elif "1/5" in align or "0/5" in align:
                    styles.at[i, "Alignment"] = "background-color: rgba(220, 38, 38, 0.18); color: #ef4444; font-weight: 700"

            # Cal Score Rec
            if "Cal Score Rec" in df.columns:
                csr = str(row.get("Cal Score Rec", ""))
                if "Deploy" in csr:
                    styles.at[i, "Cal Score Rec"] = "background-color: rgba(58, 181, 74, 0.22); color: #3ab54a; font-weight: 700"
                elif "Watch" in csr:
                    styles.at[i, "Cal Score Rec"] = "background-color: rgba(255, 184, 0, 0.18); color: #ffb800; font-weight: 700"
                elif "Wait" in csr:
                    styles.at[i, "Cal Score Rec"] = "background-color: rgba(2, 132, 199, 0.18); color: #0284c7; font-weight: 700"
                elif "Avoid" in csr or "Reject" in csr:
                    styles.at[i, "Cal Score Rec"] = "background-color: rgba(220, 38, 38, 0.18); color: #ef4444; font-weight: 700"

            # Price Action Rec
            if "Price Action Rec" in df.columns:
                par = str(row.get("Price Action Rec", ""))
                if "Deploy" in par:
                    styles.at[i, "Price Action Rec"] = "background-color: rgba(58, 181, 74, 0.15); color: #3ab54a; font-weight: 700"
                elif "Watch" in par:
                    styles.at[i, "Price Action Rec"] = "background-color: rgba(255, 184, 0, 0.15); color: #ffb800; font-weight: 700"
                elif "Avoid" in par or "Reject" in par:
                    styles.at[i, "Price Action Rec"] = "background-color: rgba(220, 38, 38, 0.15); color: #ef4444; font-weight: 700"

            # Markov Setup
            if "Markov Setup" in df.columns:
                mks = str(row.get("Markov Setup", ""))
                if "Yes" in mks or "✅" in mks:
                    styles.at[i, "Markov Setup"] = "background-color: rgba(58, 181, 74, 0.22); color: #3ab54a; font-weight: 700; text-align: center"
                elif "No" in mks or "❌" in mks:
                    styles.at[i, "Markov Setup"] = "background-color: rgba(220, 38, 38, 0.18); color: #ef4444; font-weight: 700; text-align: center"

            # Option Bias
            if "Option Bias" in df.columns:
                opb = str(row.get("Option Bias", ""))
                if "Call Heavy" in opb:
                    styles.at[i, "Option Bias"] = "background-color: rgba(58, 181, 74, 0.15); color: #3ab54a; font-weight: 700"
                elif "Balanced" in opb:
                    styles.at[i, "Option Bias"] = "background-color: rgba(2, 132, 199, 0.15); color: #0284c7; font-weight: 700"
                elif "Put Heavy" in opb:
                    styles.at[i, "Option Bias"] = "background-color: rgba(220, 38, 38, 0.15); color: #ef4444; font-weight: 700"
                elif "Slight Put" in opb:
                    styles.at[i, "Option Bias"] = "background-color: rgba(255, 184, 0, 0.15); color: #ffb800; font-weight: 700"
                    
        return styles

    def _generate_watchlist_data(self, tickers: list):
        """Scan all tickers in the watchlist, save full analysis JSON to database."""
        if not tickers:
            st.warning("No tickers in watchlist to scan.")
            return

        import time
        from modules.trade_recommendation.database import serialize_analysis_data, save_signal_snapshot

        total = len(tickers)
        progress_bar = st.progress(0.0, text="Initialising scan...")
        log_placeholder = st.empty()
        log_rows = []   # accumulate per-ticker save status
        success = 0
        failed = []

        for i, t in enumerate(tickers):
            pct = i / total
            progress_bar.progress(pct, text=f"🔄 Scanning **{t}** ({i + 1}/{total}) — fetching market data...")
            try:
                data = self._get_ticker_analysis_data_live(t)
                if data:
                    progress_bar.progress(pct, text=f"💾 Saving **{t}** to database ({i + 1}/{total})...")
                    data["from_db_cache"] = False
                    self._save_active_snapshot(data)
                    success += 1
                    log_rows.append({"Ticker": t, "Status": "✅ Saved", "Price": f"${data.get('spot_price', 0):.2f}", "Decision": data.get('decision', '—')})
                else:
                    failed.append(t)
                    log_rows.append({"Ticker": t, "Status": "⚠️ No Data", "Price": "—", "Decision": "—"})
            except Exception as e:
                logger.error(f"Failed to scan {t}: {e}")
                failed.append(t)
                log_rows.append({"Ticker": t, "Status": f"❌ Error: {str(e)[:60]}", "Price": "—", "Decision": "—"})

            progress_bar.progress((i + 1) / total, text=f"Progress: {i + 1}/{total} tickers processed")
            # Update the live log table after every ticker
            log_placeholder.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

        # Final summary
        summary_parts = [f"✅ **{success}/{total} tickers saved to database**"]
        if failed:
            summary_parts.append(f"⚠️ Failed: {', '.join(failed)}")
        progress_bar.progress(1.0, text="Scan complete!")
        log_placeholder.success("  \n".join(summary_parts))
        time.sleep(2)
        log_placeholder.empty()
        progress_bar.empty()
        st.rerun()

    def _get_ticker_analysis_data(self, ticker: str, refresh_token: float = 0.0) -> dict:
        """Fetch analysis data for a ticker — DB-only when not forcing a refresh.
        
        When refresh_token == 0.0 (normal page load / ticker switch), this method
        ONLY reads from the database and never calls yfinance. This prevents
        flickering caused by slow live scans on every Streamlit rerun.
        
        Live scans are only triggered when:
          - The user clicks 'Generate / Update Watchlist Data'  (calls _get_ticker_analysis_data_live directly)
          - The user clicks 'Regenerate [TICKER] Data'          (passes refresh_token != 0.0)
        """
        from modules.trade_recommendation.database import fetch_latest_ticker_snapshot, deserialize_analysis_data
        
        # ── Fast path: load from DB (no yfinance) ──────────────────────────
        if refresh_token == 0.0:
            snap = fetch_latest_ticker_snapshot(ticker)
            if snap and snap.get("raw_analysis_json"):
                try:
                    data = deserialize_analysis_data(snap["raw_analysis_json"])
                    if data:
                        if not data.get("earnings_date") or data.get("earnings_date") == "N/A":
                            try:
                                from utils.persistence import get_db_path
                                from utils.earnings_calendar_store import DB_PATH as ec_db_path
                                import sqlite3
                                today_str = datetime.today().strftime("%Y-%m-%d")
                                
                                # 1. Try calendar_scoring first
                                cs_db = get_db_path("calendar_scoring")
                                if cs_db.exists():
                                    with sqlite3.connect(cs_db) as conn:
                                        query = "SELECT earnings_date FROM ticker_decision_log WHERE ticker = ? AND earnings_date >= ? ORDER BY decision_id DESC LIMIT 1"
                                        row = conn.execute(query, (ticker, today_str)).fetchone()
                                        if row and row[0]:
                                            data["earnings_date"] = row[0]
                                            
                                # 2. Try earnings_calendar fallback
                                if (not data.get("earnings_date") or data.get("earnings_date") == "N/A") and ec_db_path.exists():
                                    with sqlite3.connect(ec_db_path) as conn:
                                        query = "SELECT MIN(date) FROM ec_earnings_events WHERE ticker = ? AND date >= ?"
                                        row = conn.execute(query, (ticker, today_str)).fetchone()
                                        if row and row[0]:
                                            data["earnings_date"] = row[0]
                            except Exception:
                                pass
                        data["spot_price"]       = float(snap["price"]) if snap["price"] else data.get("spot_price", 0.0)
                        data["composite_score"]  = float(snap["trade_score"]) if snap["trade_score"] else data.get("composite_score", 0.0)
                        data["decision"]         = snap["trade_decision"] or data.get("decision", "Wait")
                        data["strategy"]         = snap["recommended_strategy"] or data.get("strategy", "Reject")
                        data["expected_low"]     = float(snap["expected_40d_low"]) if snap["expected_40d_low"] else data.get("expected_low", 0.0)
                        data["expected_high"]    = float(snap["expected_40d_high"]) if snap["expected_40d_high"] else data.get("expected_high", 0.0)
                        data["snapshot_datetime"]= snap["snapshot_datetime"]
                        data["from_db_cache"]    = True
                        return data
                except Exception as e:
                    logger.error(f"Failed to deserialize DB cache for {ticker}: {e}")
            # No DB data — return empty; render_main will show a 'no data' card
            return {}

        # ── Forced refresh (Regenerate button only) ─────────────────────────
        data = self._get_ticker_analysis_data_live(ticker)
        if data:
            data["from_db_cache"] = False
            self._save_active_snapshot(data)
        return data

    def _get_ticker_analysis_data_live(self, ticker: str) -> dict:
        """Execute Steps 1-8 of the trade recommendation engine workflow."""
        try:
            ticker_obj = yf.Ticker(ticker)
            
            # Fetch daily data (1 year)
            df_daily = ticker_obj.history(period="1y")
            if df_daily.empty:
                return {}
                
            # Fetch 1H data (60 days)
            df_1h = ticker_obj.history(period="60d", interval="1h")
            if df_1h.empty:
                # Fallback: copy daily if hourly fails
                df_1h = df_daily.copy()
                
            # Fetch VIX
            vix_val = 16.5
            try:
                vix_df = yf.Ticker("^VIX").history(period="5d")
                if not vix_df.empty:
                    vix_val = float(vix_df['Close'].iloc[-1])
            except Exception:
                pass
                
            # Run indicator scans
            daily_ind = run_indicators_scan(df_daily, "Daily")
            hourly_ind = run_indicators_scan(df_1h, "1H")
            
            if not daily_ind or not hourly_ind:
                return {}
                
            # Trend and Regime Engines
            trend_state = run_trend_engine(daily_ind)
            regime_state = run_regime_engine(df_daily, vix_val)
            
            # IV and Option metrics
            spot_price = daily_ind["price"]
            use_synthetic = st.session_state.get("re_use_synthetic", False)
            options_data = fetch_option_chain_data(ticker, spot_price, use_synthetic=use_synthetic)
            
            iv_rank = 30.0
            if "short_calls" in options_data and not options_data["short_calls"].empty:
                iv_rank = float(options_data["short_calls"]["impliedVolatility"].mean() * 100.0)
            daily_ind["iv_rank"] = iv_rank
            
            # 40-Day Forecast Engine
            expected_path, expected_low, expected_high = run_forecast_engine(
                spot_price, iv_rank / 100.0, daily_ind, trend_state, regime_state
            )
            
            # Scoring model
            option_liq = {
                "bid_ask_spread_pct": 0.02,
                "avg_option_volume": 500
            }
            if "short_calls" in options_data and not options_data["short_calls"].empty:
                option_liq["bid_ask_spread_pct"] = float(options_data.get("bid_ask_spread_pct", 0.02))
                option_liq["avg_option_volume"] = int(options_data["short_calls"].get("volume", pd.Series([500])).mean())
                
            earnings_date_str = None
            try:
                from utils.persistence import get_db_path
                from utils.earnings_calendar_store import DB_PATH as ec_db_path
                import sqlite3
                today_str = datetime.today().strftime("%Y-%m-%d")
                
                # 1. Try calendar_scoring first
                cs_db = get_db_path("calendar_scoring")
                if cs_db.exists():
                    with sqlite3.connect(cs_db) as conn:
                        query = "SELECT earnings_date FROM ticker_decision_log WHERE ticker = ? AND earnings_date >= ? ORDER BY decision_id DESC LIMIT 1"
                        row = conn.execute(query, (ticker, today_str)).fetchone()
                        if row and row[0]:
                            earnings_date_str = row[0]
                            
                # 2. Try earnings_calendar fallback
                if (not earnings_date_str) and ec_db_path.exists():
                    with sqlite3.connect(ec_db_path) as conn:
                        query = "SELECT MIN(date) FROM ec_earnings_events WHERE ticker = ? AND date >= ?"
                        row = conn.execute(query, (ticker, today_str)).fetchone()
                        if row and row[0]:
                            earnings_date_str = row[0]
            except Exception:
                pass
                
            if not earnings_date_str:
                try:
                    calendar = ticker_obj.calendar
                    if calendar is not None and not calendar.empty:
                        earnings_date_str = calendar.iloc[0, 0].strftime("%Y-%m-%d")
                except Exception:
                    pass
                
            scores = calculate_component_scores(daily_ind, vix_val, option_liq, earnings_date_str)
            composite_score = calculate_composite_score(scores)
            
            # Decision Classification
            raw_decision = "Reject"
            if composite_score >= 85.0:
                raw_decision = "Deploy"
            elif composite_score >= 75.0:
                raw_decision = "Watch"
            elif composite_score >= 65.0:
                raw_decision = "Wait"
            
            # Strategy Selector Matrix
            strategy, strategy_reason = select_trade_strategy(scores, daily_ind, expected_path)
            
            # State alignment correction
            decision, alignment_reason = adjust_decision_for_alignment(raw_decision, strategy, daily_ind)
            
            # Generate Trade Plan
            trade_plan = generate_trade_plan(
                ticker, spot_price, decision, strategy, trend_state, expected_low, expected_high, daily_ind
            )
            
            return {
                "ticker": ticker,
                "spot_price": spot_price,
                "daily_indicators": daily_ind,
                "hourly_indicators": hourly_ind,
                "trend_state": trend_state,
                "regime_state": regime_state,
                "expected_path": expected_path,
                "expected_low": expected_low,
                "expected_high": expected_high,
                "scores": scores,
                "composite_score": composite_score,
                "decision": decision,
                "strategy": strategy,
                "strategy_reason": strategy_reason,
                "alignment_reason": alignment_reason,
                "trade_plan": trade_plan,
                "df_daily": df_daily,
                "options_data": options_data,
                "vix_value": vix_val,
                "earnings_date": earnings_date_str
            }
        except Exception as e:
            logger.error(f"Error fetching ticker analysis data: {e}", exc_info=True)
            return {}

    # ═════════════════════════════════════════════════════════════════════
    # DATA PERSISTENCE
    # ═════════════════════════════════════════════════════════════════════

    def _save_active_snapshot(self, data: dict):
        """Serialize the full analysis data dict and persist it to the database."""
        from modules.trade_recommendation.database import save_signal_snapshot, serialize_analysis_data

        try:
            ticker = data.get("ticker", "")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            daily_ind = data.get("daily_indicators", {})
            scores    = data.get("scores", {})
            trade_plan = data.get("trade_plan", {})

            # Stamp the datetime so it can be displayed as the 'data version'
            data["snapshot_datetime"] = now

            # ─ Snapshot row ───────────────────────────────────────
            snapshot = {
                "ticker":                ticker,
                "snapshot_datetime":    now,
                "price":               data.get("spot_price"),
                "trend_state":         data.get("trend_state"),
                "market_state":        data.get("regime_state"),
                "trade_decision":      data.get("decision"),
                "recommended_strategy":data.get("strategy"),
                "trade_score":         data.get("composite_score"),
                "trend_score":         scores.get("trend_score"),
                "momentum_score":      scores.get("momentum_score"),
                "range_score":         scores.get("range_score"),
                "volatility_score":    scores.get("volatility_score"),
                "liquidity_score":     scores.get("liquidity_score"),
                "event_risk_score":    scores.get("event_risk_score"),
                "expected_40d_path":   None,   # omit — large list; stored in raw_analysis_json
                "expected_40d_low":    data.get("expected_low"),
                "expected_40d_high":   data.get("expected_high"),
                "support_level":       daily_ind.get("support"),
                "resistance_level":    daily_ind.get("resistance"),
                "trigger_level":       daily_ind.get("trigger"),
                "invalidation_level":  daily_ind.get("invalidation"),
                "notes":               data.get("strategy_reason", ""),
                "raw_analysis_json":   serialize_analysis_data(data),
            }

            # ─ Indicator rows (Daily + 1H) ────────────────────────
            indicators = []
            for tf_key, tf_label in [("daily_indicators", "Daily"), ("hourly_indicators", "1H")]:
                ind = data.get(tf_key, {})
                if ind:
                    indicators.append({
                        "ticker":           ticker,
                        "timeframe":        tf_label,
                        "price":            ind.get("price"),
                        "ma20":             ind.get("ma20"),
                        "ma50":             ind.get("ma50"),
                        "ma200":            ind.get("ma200"),
                        "vwap":             ind.get("vwap"),
                        "fdts_delta":       ind.get("fdts_delta"),
                        "fdts_signal":      ind.get("fdts_signal"),
                        "macd_value":       ind.get("macd"),
                        "macd_avg":         ind.get("macd_avg"),
                        "macd_hist":        ind.get("macd_hist"),
                        "macd_signal":      ind.get("macd_signal"),
                        "wpr_value":        ind.get("wpr"),
                        "wpr_signal":       ind.get("wpr_signal"),
                        "darvas_upper":     ind.get("darvas_upper"),
                        "darvas_lower":     ind.get("darvas_lower"),
                        "darvas_signal":    ind.get("darvas_signal"),
                        "regression_upper": ind.get("regression_upper"),
                        "regression_middle":ind.get("regression_middle"),
                        "regression_lower": ind.get("regression_lower"),
                        "ichimoku_span_a":  ind.get("ichimoku_span_a"),
                        "ichimoku_span_b":  ind.get("ichimoku_span_b"),
                        "cloud_signal":     ind.get("cloud_signal"),
                        "atr14":            ind.get("atr14"),
                        "iv_rank":          ind.get("iv_rank"),
                    })

            # ─ Trade plan row ──────────────────────────────────
            plan = None
            if trade_plan:
                plan = {
                    "ticker":           ticker,
                    "decision":         data.get("decision"),
                    "strategy":         data.get("strategy"),
                    "option_structure": trade_plan.get("option_structure"),
                    "entry_trigger":    trade_plan.get("entry_trigger"),
                    "target_zone":      trade_plan.get("target_zone"),
                    "invalidation_rule":trade_plan.get("invalidation_rule"),
                    "adjustment_rule":  trade_plan.get("adjustment_rule"),
                    "profit_target":    trade_plan.get("profit_target"),
                    "max_loss_rule":    trade_plan.get("max_loss_rule"),
                    "rationale":        trade_plan.get("rationale"),
                }

            save_signal_snapshot(snapshot, indicators, plan)
            logger.info(f"✅ Snapshot saved to database for {ticker} at {now}")

        except Exception as e:
            logger.error(f"Failed to save snapshot for {data.get('ticker', '?')}: {e}", exc_info=True)
            raise  # Re-raise so _generate_watchlist_data can log a real ❌ Error status

    # ═════════════════════════════════════════════════════════════════════
    # SCREEN RENDER METHODS
    # ═════════════════════════════════════════════════════════════════════

    def _render_command_center(self, data: dict):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Decision Summary")
            
            # Visual badge for decision
            decision = data["decision"]
            colors = {
                "Deploy": "#3ab54a",
                "Watch": "#ffb800",
                "Wait": "#0284c7",
                "Reject": "#ef4444",
                "Avoid": "#ef4444"
            }
            bg_color = colors.get(decision, "#64748b")
            
            st.markdown(
                f"""
                <div style='
                    background: {bg_color};
                    color: white;
                    padding: 16px;
                    border-radius: 10px;
                    text-align: center;
                    font-size: 24px;
                    font-weight: 800;
                    letter-spacing: 1.5px;
                    text-transform: uppercase;
                    margin-bottom: 20px;
                '>
                    {decision}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Metrics — guard against None values from stale DB cache
            composite = data.get('composite_score')
            spot_p = data.get('spot_price')
            st.metric("Trade Quality Score", f"{composite:.1f} / 100" if composite is not None else "N/A")
            st.metric("Spot Price", f"${spot_p:.2f}" if spot_p is not None else "N/A")
            st.metric("Suggested Strategy", data.get("strategy") or "N/A")
            
            # Log button
            if st.button("🚀 Log & Deploy Trade Setup", use_container_width=True, type="primary"):
                self._save_active_snapshot(data)
                st.success("Trade setup logged to database successfully!")
                
        with col2:
            st.markdown("### Professional Trade Plan")
            plan = data["trade_plan"]
            
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, rgba(21, 40, 71, 0.4) 0%, rgba(13, 27, 46, 0.4) 100%);
                    border: 1px solid #1e3a5f;
                    border-radius: 12px;
                    padding: 22px;
                ">
                    <h3 style="margin-top:0;color:#3ab54a;">Trade Plan: {data['ticker']}</h3>
                    <p><b>Decision Bias</b>: {data['trend_state']}</p>
                    <p><b>Recommended Structure</b>: {plan['option_structure']}</p>
                    <hr style="border-color:#1e3a5f;">
                    <p><b>Trigger</b>: {plan['entry_trigger']}</p>
                    <p><b>Target Zone</b>: {plan['target_zone']}</p>
                    <p><b>Invalidation Support</b>: {plan['invalidation_rule']}</p>
                    <p><b>Adjustment Rules</b>: {plan['adjustment_rule']}</p>
                    <hr style="border-color:#1e3a5f;">
                    <p><b>Profit Taking Target</b>: {plan['profit_target']}</p>
                    <p><b>Stop / Max Loss</b>: {plan['max_loss_rule']}</p>
                    <p><b>Strategic Rationale</b>: {plan['rationale']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    def _render_strategy_selector(self, data: dict):
        st.markdown("### 📊 Scoring Weights & Strategy Selection Matrix")
        
        # Display sub-scores
        scores = data.get("scores") or {}
        if not scores:
            st.info("⚠️ No scoring data available. Please regenerate the analysis for this ticker.")
            return

        df_scores = pd.DataFrame([
            {"Component": "Trend Structure",       "Weight %": 20, "Score": scores.get("trend_score",      "N/A")},
            {"Component": "FDTS Signal",           "Weight %": 15, "Score": scores.get("fdts_score",       "N/A")},
            {"Component": "MACD Momentum",         "Weight %": 15, "Score": scores.get("momentum_score",   "N/A")},
            {"Component": "Darvas Box Structure",  "Weight %": 15, "Score": scores.get("range_score",      "N/A")},
            {"Component": "Regression location",   "Weight %": 10, "Score": scores.get("regression_score", "N/A")},
            {"Component": "Volatility / IV condition","Weight %":10,"Score": scores.get("volatility_score", "N/A")},
            {"Component": "WPR / Timing",          "Weight %": 5,  "Score": scores.get("timing_score",     "N/A")},
            {"Component": "Option Liquidity",      "Weight %": 5,  "Score": scores.get("liquidity_score",  "N/A")},
            {"Component": "Event Risk",            "Weight %": 5,  "Score": scores.get("event_risk_score", "N/A")},
        ])
        
        st.dataframe(df_scores, use_container_width=True, hide_index=True)
        
        st.markdown("#### Strategy Decision Logic")
        st.info(f"**Strategy Selected**: {data.get('strategy', 'N/A')}\n\n**Reasoning**: {data.get('strategy_reason', 'N/A')}")
        
        if data.get("alignment_reason"):
            st.warning(f"**State Alignment Filter Action**: {data['alignment_reason']}")
            
        st.markdown("#### Options to Avoid")
        if data.get("decision") in ("Watch", "Wait"):
            st.markdown("- ⚠️ **Avoid** straight long call option structure until momentum trigger validates above resistance.")
        else:
            st.markdown("- ✔️ Standard risk parameters align. Avoid buying high IV premium legs if IV Rank drops.")

    def _render_forecast_map(self, data: dict):
        st.markdown("### 40-Day Blended Expected Range & Probability Cone")

        df_daily  = data.get("df_daily")
        spot      = data.get("spot_price")
        high_val  = data.get("expected_high")
        low_val   = data.get("expected_low")
        ticker    = data.get("ticker", "")
        path      = data.get("expected_path", "")

        # ── Auto-recover missing chart data from a lightweight yfinance call ──
        if (df_daily is None or (hasattr(df_daily, "empty") and df_daily.empty) or spot is None) and ticker:
            try:
                import yfinance as yf
                _df = yf.Ticker(ticker).history(period="1y")
                if not _df.empty:
                    if isinstance(_df.columns, pd.MultiIndex):
                        _df.columns = _df.columns.droplevel(1)
                    df_daily = _df
                    if spot is None:
                        spot = float(_df["Close"].iloc[-1])
            except Exception:
                pass

        if spot is None or df_daily is None or (hasattr(df_daily, "empty") and df_daily.empty):
            st.info("⚠️ No forecast data available. Please regenerate the analysis for this ticker.")
            return

        daily_ind = data.get("daily_indicators") or {}
        # iv_rank stores annualized IV (mean options-chain IV × 100); divide to get decimal
        iv = (daily_ind.get("iv_rank") or 30.0) / 100.0
        z  = st.session_state.get("re_cone_z", 1.64)

        # ── Expected Path classification badge ───────────────────────────────
        _path_meta = {
            "directional_bullish":  ("🟢", "Directional Bullish",     "#22c55e", "rgba(34,197,94,0.12)"),
            "mean_reversion_up":    ("🔵", "Mean Reversion Up",        "#60a5fa", "rgba(96,165,250,0.12)"),
            "sideways_bullish":     ("🟡", "Sideways / Bullish Bias",  "#facc15", "rgba(250,204,21,0.12)"),
            "directional_bearish":  ("🔴", "Directional Bearish",      "#ef4444", "rgba(239,68,68,0.12)"),
            "sideways_bearish":     ("🟠", "Sideways / Bearish Bias",  "#f97316", "rgba(249,115,22,0.12)"),
            "volatile_expansion":   ("⚡", "Volatile Expansion",        "#a78bfa", "rgba(167,139,250,0.12)"),
            "sideways_range":       ("⚪", "Sideways / Range Bound",   "#94a3b8", "rgba(148,163,184,0.12)"),
            "unclear":              ("❓", "Unclear / Mixed Signals",   "#94a3b8", "rgba(148,163,184,0.12)"),
        }
        if path in _path_meta:
            icon, label, color, bg = _path_meta[path]
            st.markdown(
                f"<div style='display:inline-flex;align-items:center;gap:12px;"
                f"background:{bg};border:1px solid {color}44;border-radius:8px;"
                f"padding:9px 18px;margin-bottom:14px'>"
                f"<span style='font-size:22px'>{icon}</span>"
                f"<div><span style='color:#94a3b8;font-size:10px;text-transform:uppercase;"
                f"letter-spacing:1.2px'>Expected 40D Path</span>"
                f"<br><span style='color:{color};font-size:15px;font-weight:700'>{label}</span></div></div>",
                unsafe_allow_html=True
            )

        # ── Trading-day calendar helper ───────────────────────────────────────
        def _trading_dates(from_date, n):
            """Return the next n Mon-Fri trading dates after from_date."""
            dates, cur = [], from_date
            while len(dates) < n:
                cur = cur + timedelta(days=1)
                if cur.weekday() < 5:
                    dates.append(cur)
            return dates

        hist_days = df_daily.tail(60)
        last_date = hist_days.index[-1]

        # 201-point grid (0 … 40 trading days) → smooth curves
        N  = 201
        t  = np.linspace(0, 40, N)          # trading days (0 = today)

        # Map to calendar dates by proportional interpolation over ~56 cal days
        td40       = _trading_dates(last_date, 40)
        cal_span   = (td40[-1] - last_date).days          # ≈ 56
        x_dates    = [last_date + timedelta(days=float(ti * cal_span / 40)) for ti in t]

        # σ bands — σ₀ = 0 at t=0 (cone always starts at spot)
        sigma = np.where(t > 0, spot * iv * np.sqrt(t / 252.0), 0.0)

        s3_up   = spot + 3.0 * sigma;  s3_dn  = spot - 3.0 * sigma
        s2_up   = spot + 2.0 * sigma;  s2_dn  = spot - 2.0 * sigma
        s1_up   = spot + 1.0 * sigma;  s1_dn  = spot - 1.0 * sigma
        s05_up  = spot + 0.5 * sigma;  s05_dn = spot - 0.5 * sigma
        # z-scaled edge (user-controlled from sidebar slider)
        z_up    = spot + z   * sigma;  z_dn   = spot - z   * sigma

        # Blended model targets — z-scaled if falling outside cone
        if high_val is None:  high_val = float(z_up[-1])
        if low_val  is None:  low_val  = float(z_dn[-1])
        # Already clamped to spot in run_forecast_engine, but guard again
        high_val = max(high_val, spot)
        low_val  = min(low_val,  spot)

        blend_up = np.linspace(spot, high_val, N)
        blend_dn = np.linspace(spot, low_val,  N)

        # End-point label helpers
        def _lbl(arr):
            return [""] * (N - 1) + [f"${arr[-1]:.2f}"]

        # ── Plotly figure ─────────────────────────────────────────────────────
        fig = go.Figure()

        # 1. Historical close (60 days)
        fig.add_trace(go.Scatter(
            x=hist_days.index, y=hist_days["Close"],
            name="Historical Close",
            line=dict(color="#3ab54a", width=2.5),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Close: $%{y:.2f}<extra>Historical</extra>"
        ))

        # 2. Layered cone fills — outer → inner, progressively more opaque
        #    Each fill pair: anchor (invisible bottom) + top (filled to previous)
        def _fill_band(fig, x, y_lo, y_hi, fill_color, legend_name=None):
            fig.add_trace(go.Scatter(
                x=x, y=y_lo, showlegend=False, hoverinfo="skip",
                mode="lines", line=dict(width=0)
            ))
            fig.add_trace(go.Scatter(
                x=x, y=y_hi,
                name=legend_name or "", showlegend=bool(legend_name),
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=fill_color, hoverinfo="skip"
            ))

        _fill_band(fig, x_dates, s3_dn,  s3_up,  "rgba(99,102,241,0.05)", "3σ Shadow")
        _fill_band(fig, x_dates, s2_dn,  s2_up,  "rgba(139,92,246,0.10)", "2σ  (95%)")
        _fill_band(fig, x_dates, s1_dn,  s1_up,  "rgba(139,92,246,0.20)", "1σ  (68%)")
        _fill_band(fig, x_dates, s05_dn, s05_up, "rgba(167,139,250,0.28)", "0.5σ Core")

        # 3. σ boundary edge lines (smooth — 200+ pts makes them naturally curved)
        for y_arr, color, dash, name in [
            (s2_up,  "rgba(139,92,246,0.75)", "dash",  "+2σ"),
            (s2_dn,  "rgba(139,92,246,0.75)", "dash",  "-2σ"),
            (s1_up,  "rgba(167,139,250,0.90)", "dot",   "+1σ"),
            (s1_dn,  "rgba(167,139,250,0.90)", "dot",   "-1σ"),
        ]:
            fig.add_trace(go.Scatter(
                x=x_dates, y=y_arr, name=name,
                mode="lines+text",
                text=_lbl(y_arr), textposition="middle right",
                textfont=dict(color=color, size=10, family="monospace"),
                line=dict(color=color, width=1.3, dash=dash),
                hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>$%{{y:.2f}}<extra></extra>"
            ))

        # 4. z-scaled confidence boundary (driven by sidebar slider)
        for y_arr, name in [(z_up, f"+{z:.2f}σ"), (z_dn, f"-{z:.2f}σ")]:
            fig.add_trace(go.Scatter(
                x=x_dates, y=y_arr, name=name,
                mode="lines+text",
                text=_lbl(y_arr), textposition="middle right",
                textfont=dict(color="#c4b5fd", size=10, family="monospace"),
                line=dict(color="#c4b5fd", width=2.0),
                hovertemplate=f"<b>{name} Conf.</b><br>%{{x|%Y-%m-%d}}<br>$%{{y:.2f}}<extra></extra>"
            ))

        # 5. Blended model upper / lower
        fig.add_trace(go.Scatter(
            x=x_dates, y=blend_up, name=f"Blended Target ${high_val:.2f}",
            mode="lines+text",
            text=[""] * (N - 1) + [f"Target  ${high_val:.2f}"],
            textposition="middle right",
            textfont=dict(color="#10b981", size=11, family="monospace"),
            line=dict(color="#10b981", width=2.2),
            hovertemplate="<b>Blended Upper</b><br>%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=x_dates, y=blend_dn, name=f"Blended Support ${low_val:.2f}",
            mode="lines+text",
            text=[""] * (N - 1) + [f"Support  ${low_val:.2f}"],
            textposition="middle right",
            textfont=dict(color="#ef4444", size=11, family="monospace"),
            line=dict(color="#ef4444", width=2.2),
            hovertemplate="<b>Blended Lower</b><br>%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>"
        ))

        # 6. Spot apex marker
        fig.add_trace(go.Scatter(
            x=[last_date], y=[spot],
            name=f"Spot ${spot:.2f}",
            mode="markers+text",
            text=[f"  ${spot:.2f}"],
            textposition="middle right",
            textfont=dict(color="#facc15", size=12, family="monospace"),
            marker=dict(size=12, color="#facc15", symbol="diamond",
                        line=dict(color="#0d1b2e", width=2)),
            hovertemplate=f"<b>Today Spot</b><br>${spot:.2f}<extra></extra>"
        ))

        # 7. "Today" vertical
        fig.add_vline(
            x=last_date.timestamp() * 1000,
            line_color="#facc15", line_width=1.5, line_dash="dot",
            annotation_text="<b>Today</b>",
            annotation_position="top right",
            annotation_font=dict(color="#facc15", size=11)
        )

        # 8. Earnings date overlay (if within chart window)
        earnings_str = data.get("earnings_date")
        if earnings_str and earnings_str != "N/A":
            try:
                clean_ed = earnings_str.strip().lstrip("🔴🟡 ").strip()
                ed_dt    = datetime.strptime(clean_ed, "%Y-%m-%d")
                days_to  = (ed_dt.date() - last_date.date()).days
                if 0 < days_to <= int(cal_span * 1.2):
                    ed_color = "#ef4444" if days_to <= 20 else "#facc15"
                    fig.add_vline(
                        x=ed_dt.timestamp() * 1000,
                        line_color=ed_color, line_width=1.5, line_dash="dashdot",
                        annotation_text=f"<b>⚡ Earnings {clean_ed}</b>",
                        annotation_position="top left",
                        annotation_font=dict(color=ed_color, size=10)
                    )
            except Exception:
                pass

        # ── Layout ────────────────────────────────────────────────────────────
        fig.update_layout(
            title=dict(
                text=(f"40-Day Probability Cone: {ticker}  |  "
                      f"Annualized IV={iv*100:.1f}%  |  Confidence Z={z:.2f}σ"),
                font=dict(color="#e2e8f0", size=14)
            ),
            xaxis=dict(
                title="Date", showgrid=True, gridcolor="#1e3a5f",
                zeroline=False, color="#94a3b8"
            ),
            yaxis=dict(
                title="Stock Price ($)", tickprefix="$",
                showgrid=True, gridcolor="#1e3a5f",
                zeroline=False, color="#94a3b8"
            ),
            plot_bgcolor="#0d1b2e",
            paper_bgcolor="#0d1b2e",
            font=dict(color="#94a3b8", family="Inter, sans-serif"),
            legend=dict(
                orientation="v",
                yanchor="top", y=0.98,
                xanchor="left", x=1.01,
                font=dict(size=10, color="#cbd5e1"),
                bgcolor="rgba(13,27,46,0.80)",
                bordercolor="#1e3a5f",
                borderwidth=1
            ),
            hovermode="x unified",
            margin=dict(l=20, r=175, t=60, b=30)

        )

        st.plotly_chart(fig, use_container_width=True)

        # ── Probability Cone Range Table ──────────────────────────────────────
        st.markdown("#### 📊 Probability Cone Ranges")
        intervals = [5, 10, 15, 20, 25, 30, 35, 40]
        rows = []
        for d in intervals:
            m1 = spot * iv * np.sqrt(d / 252.0)
            rows.append({
                "Horizon":          f"{d}D",
                "1σ Up  (+68%)":    f"${spot + m1:.2f}",
                "1σ Down (-68%)":   f"${spot - m1:.2f}",
                "2σ Up  (+95%)":    f"${spot + 2*m1:.2f}",
                "2σ Down (-95%)":   f"${spot - 2*m1:.2f}",
                f"±{z:.2f}σ Move":  f"${z * m1:.2f}",
            })
        df_ranges = pd.DataFrame(rows)

        def _style_ranges(df):
            """Highlight the 20-day row (standard options cycle)."""
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for i in df.index:
                if str(df.loc[i, "Horizon"]) == "20D":
                    for c in df.columns:
                        styles.at[i, c] = (
                            "background-color:rgba(139,92,246,0.20);"
                            "font-weight:700"
                        )
            return styles

        st.dataframe(
            df_ranges.style.apply(_style_ranges, axis=None),
            use_container_width=True, hide_index=True
        )

    def _render_signal_matrix(self, data: dict):

        st.markdown("### Timeframe Indicator Agreement (3M vs Daily vs 1H)")
        
        daily = data.get("daily_indicators")
        hourly = data.get("hourly_indicators")

        if not daily or not hourly:
            st.info("⚠️ No indicator data available. Please regenerate the analysis for this ticker.")
            return
        
        def _safe_cmp(a, b):
            """Return True if both a and b are non-None numbers and a > b, else False."""
            try:
                return (a is not None and b is not None) and (a > b)
            except Exception:
                return False

        d_price  = daily.get("price")
        d_ma200  = daily.get("ma200")
        d_ma50   = daily.get("ma50")
        d_ma20   = daily.get("ma20")
        h_price  = hourly.get("price")
        h_ma20   = hourly.get("ma20")

        df_matrix = pd.DataFrame([
            {
                "Indicator": "MA Trend Structure",
                "3-Month": "Price above 200 EMA" if _safe_cmp(d_price, d_ma200) else ("Price below 200 EMA" if d_price is not None else "N/A"),
                "Daily":   "Price above 50 EMA"  if _safe_cmp(d_price, d_ma50)  else ("Price below 50 EMA"  if d_price is not None else "N/A"),
                "1-Hour":  "Price above 20 EMA"  if _safe_cmp(h_price, h_ma20)  else ("Price below 20 EMA"  if h_price is not None else "N/A"),
            },
            {
                "Indicator": "MACD Momentum",
                "3-Month": "N/A",
                "Daily":   daily.get("macd_signal") or "N/A",
                "1-Hour":  hourly.get("macd_signal") or "N/A",
            },
            {
                "Indicator": "FDTS Acceleration",
                "3-Month": "N/A",
                "Daily":   daily.get("fdts_signal") or "N/A",
                "1-Hour":  hourly.get("fdts_signal") or "N/A",
            }
        ])
        
        st.dataframe(df_matrix, use_container_width=True, hide_index=True)
        
        # Verify agreement — safe against None
        d_macd = daily.get("macd_signal")
        h_macd = hourly.get("macd_signal")
        d_fdts = daily.get("fdts_signal")

        bull_agree = d_macd == "Bullish" and h_macd == "Bullish" and d_fdts == "Buy"
        bear_agree = d_macd == "Bearish" and h_macd == "Bearish" and d_fdts == "Sell"
        
        if bull_agree:
            st.success("🟢 Strong bullish agreement confirmed across all timeframes.")
        elif bear_agree:
            st.error("🔴 Strong bearish agreement confirmed across all timeframes.")
        else:
            st.info("⚪ Timeframe conflict present. Recommended posture: Watch or Wait.")

    def _render_option_chain_selector(self, data: dict):
        st.markdown("### Options Chain Structuring")
        
        options = data["options_data"]
        if not options or "short_calls" not in options:
            st.warning("No options chain details available.")
            return
            
        st.markdown(f"**Short Expiration (Front Leg)**: {options['short_expiry']} (DTE {options['short_dte']})")
        st.markdown(f"**Long Expiration (Back Leg)**: {options['long_expiry']} (DTE {options['long_dte']})")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Front Leg Options Chain")
            st.dataframe(options["short_calls"][["strike", "bid", "ask", "impliedVolatility", "volume", "openInterest"]].head(10), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("#### Back Leg Options Chain")
            st.dataframe(options["long_calls"][["strike", "bid", "ask", "impliedVolatility", "volume", "openInterest"]].head(10), use_container_width=True, hide_index=True)

    def _render_portfolio_monitor(self):
        st.markdown("### Deployed Portfolio Positions Monitor")
        
        active_plans = fetch_active_trade_plans()
        if not active_plans:
            st.info("No active deployed positions found in the log.")
            return
            
        df_plans = pd.DataFrame(active_plans)
        st.dataframe(df_plans[["trade_plan_id", "ticker", "strategy", "option_structure", "entry_trigger", "created_datetime"]], use_container_width=True, hide_index=True)
        
        # Real-time position check
        st.markdown("#### Live Status Evaluations")
        for plan in active_plans:
            ticker = plan["ticker"]
            price_at_sig = plan["current_price_at_signal"]
            
            with st.spinner(f"Verifying live status of {ticker}..."):
                try:
                    current_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
                except Exception:
                    current_price = price_at_sig
                    
            pnl_est = current_price - price_at_sig
            pnl_pct = (pnl_est / price_at_sig) * 100.0
            
            # Simple boundary check
            status = "Hold"
            if pnl_pct < -5.0:
                status = "Exit / Stop Loss"
            elif pnl_pct > 10.0:
                status = "Hedge / Take Profit"
                
            st.markdown(f"- **{ticker}**: entry ${price_at_sig:.2f} | current ${current_price:.2f} | P&L: **{pnl_pct:+.2f}%** | Suggested action: **{status}**")

    def _render_backtest_outcomes(self):
        st.markdown("### Trade Performance & Backtest Logs")
        
        # Outcome update trigger button
        if st.button("🔄 Execute Outcomes Review & Logging Update", use_container_width=True, type="secondary"):
            self._update_all_outcomes()
            
        outcomes = fetch_trade_outcomes()
        if not outcomes:
            st.info("No trade outcomes logged yet. Deploy setups and run outcomes review after 5+ days.")
            return
            
        df_outcomes = pd.DataFrame(outcomes)
        st.dataframe(df_outcomes, use_container_width=True, hide_index=True)

    def _render_rejected_log(self):
        st.markdown("### Rejected Trades Audit Log")
        
        snapshots = fetch_historical_snapshots()
        rejections = [s for s in snapshots if s["trade_decision"] in ("Reject", "Avoid")]
        
        if not rejections:
            st.info("No rejected tickers found in the scan histories.")
            return
            
        df_rej = pd.DataFrame(rejections)
        st.dataframe(df_rej[["snapshot_datetime", "ticker", "price", "trade_score", "recommended_strategy", "notes"]], use_container_width=True, hide_index=True)

    def _update_all_outcomes(self):
        active_plans = fetch_active_trade_plans()
        if not active_plans:
            st.info("No active deployed trade plans to review.")
            return
            
        updated = 0
        for plan in active_plans:
            created_dt = datetime.strptime(plan["created_datetime"][:19], "%Y-%m-%d %H:%M:%S")
            days_passed = (datetime.now() - created_dt).days
            
            # Review immediately for testing/demo, or check after days
            ticker = plan["ticker"]
            price_at_sig = plan["current_price_at_signal"]
            
            try:
                df = yf.Ticker(ticker).history(start=created_dt.strftime("%Y-%m-%d"))
                if df.empty or len(df) < 2:
                    continue
                    
                closes = df['Close']
                highs = df['High']
                lows = df['Low']
                
                p5 = float(closes.iloc[5]) if len(closes) > 5 else None
                p10 = float(closes.iloc[10]) if len(closes) > 10 else None
                p20 = float(closes.iloc[20]) if len(closes) > 20 else None
                p40 = float(closes.iloc[40]) if len(closes) > 40 else None
                
                max_fav = float(highs.max() - price_at_sig)
                max_adv = float(price_at_sig - lows.min())
                
                curr_price = float(closes.iloc[-1])
                pnl = curr_price - price_at_sig
                result = "Profit" if pnl > 0 else "Loss"
                
                outcome = {
                    "trade_plan_id": plan["trade_plan_id"],
                    "ticker": ticker,
                    "entry_date": created_dt.strftime("%Y-%m-%d"),
                    "review_date": datetime.now().strftime("%Y-%m-%d"),
                    "price_at_signal": price_at_sig,
                    "price_after_5d": p5,
                    "price_after_10d": p10,
                    "price_after_20d": p20,
                    "price_after_40d": p40,
                    "max_favorable_move": max_fav,
                    "max_adverse_move": max_adv,
                    "strategy_result": result,
                    "estimated_pnl": pnl,
                    "notes": f"Reviewed after {days_passed} days. Live Close: ${curr_price:.2f}"
                }
                log_trade_outcome(outcome)
                updated += 1
            except Exception as e:
                logger.error(f"Failed outcome updates for {ticker}: {e}")
                
        st.success(f"Outcome update complete! Processed {updated} entries.")
