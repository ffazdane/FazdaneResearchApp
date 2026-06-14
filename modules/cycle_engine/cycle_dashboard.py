import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from modules.base_module import FazDaneModule
from modules.cycle_engine.cycle_engine import run_cycle_analysis
from modules.cycle_engine.cycle_backtester import run_historical_backtest, compute_backtest_summary
from modules.cycle_engine.cycle_database import save_backtest_result, load_signal_history
from modules.cycle_engine.cycle_alignment_engine import calculate_alignment
from utils.universe_manager import format_ticker_display, get_ticker_names, render_universe_manager

class CycleAnalysisDashboardModule(FazDaneModule):
    MODULE_NAME = "Cycle Analysis Engine"
    MODULE_ICON = "⏳"
    MODULE_DESCRIPTION = "Dominant market cycle detector, phase turning windows, volatility structures, and option strategy selector."
    TIER = 3
    SOURCE_NOTEBOOK = "Cycle Analysis Engine"
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance", "SQLite"]

    def render_sidebar(self):
        st.markdown("**Instrument Selection**")
        if "ce_custom_ticker" not in st.session_state:
            st.session_state.ce_custom_ticker = ""

        # Universe manager integration
        universe_name, tickers_list, _ = render_universe_manager(
            key_prefix="ce_universe",
            show_benchmark=False,
            label="Ticker Universe:",
        )
        ticker_names = get_ticker_names(universe_name)
        tradeable = tickers_list or ["SPY"]
        
        previous = st.session_state.get("ce_selected_ticker", "SPY")
        index = tradeable.index(previous) if previous in tradeable else 0
        
        st.selectbox(
            "Select Ticker:",
            options=tradeable,
            index=index,
            key="ce_selected_ticker",
            format_func=lambda t: format_ticker_display(t, ticker_names),
        )
        
        custom_input = st.text_input(
            "Or Enter Ticker:",
            value=st.session_state.ce_custom_ticker,
            placeholder="e.g. SPY, QQQ, VIX...",
            key="ce_custom_input_field",
        ).strip().upper()
        st.session_state.ce_custom_ticker = custom_input

        st.divider()
        st.markdown("**Analysis Reference**")
        st.session_state["ce_as_of_date"] = st.date_input(
            "As of Date (Historical or Today):",
            value=st.session_state.get("ce_as_of_date", datetime.today().date()),
            key="ce_as_of_date_input"
        )
        
        st.divider()
        st.markdown("**Backtest Settings**")
        st.session_state["ce_backtest_periods"] = st.slider(
            "Backtest Historical Period (Days):",
            min_value=30,
            max_value=120,
            value=st.session_state.get("ce_backtest_periods", 60),
            key="ce_backtest_periods_slider"
        )

        if st.button("Run Full Cycle Analysis", width="stretch", type="primary", key="ce_run_analysis_btn"):
            st.cache_data.clear()
            st.rerun()

    def render_main(self):
        # Apply premium styling
        st.markdown("""
        <style>
        .cycle-card {
            background: linear-gradient(135deg, rgba(21, 40, 71, 0.4) 0%, rgba(13, 27, 46, 0.6) 100%);
            border: 1px solid #1e3a5f;
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 16px;
        }
        .cycle-header {
            font-size: 0.75rem;
            font-weight: 700;
            color: #3ab54a;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 12px;
            border-bottom: 1px solid rgba(58,181,74,0.18);
            padding-bottom: 4px;
        }
        .badge-positive { background: rgba(50,200,100,0.15); color: #52D68A; border: 1px solid #52D68A; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 700; }
        .badge-negative { background: rgba(220,50,50,0.15); color: #FF6B6B; border: 1px solid #FF6B6B; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 700; }
        .badge-warning { background: rgba(255,150,50,0.15); color: #FFB347; border: 1px solid #FFB347; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 700; }
        .badge-info { background: rgba(0,173,181,0.15); color: #00ADB5; border: 1px solid #00ADB5; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 700; }
        </style>
        """, unsafe_allow_html=True)

        ticker = st.session_state.ce_custom_ticker if st.session_state.ce_custom_ticker else st.session_state.get("ce_selected_ticker", "SPY")
        as_of_date = st.session_state.get("ce_as_of_date", datetime.today().date())

        self.render_section_header(
            f"Cycle Analysis Dashboard: {ticker}",
            f"As-of-Date: {as_of_date.strftime('%Y-%m-%d')} | Real-time cycle estimation and structural strategy maps"
        )

        with st.spinner("Executing cycle signal extraction engines..."):
            try:
                res = run_cycle_analysis(ticker, as_of_date.strftime("%Y-%m-%d"))
            except Exception as e:
                st.error(f"Failed to process cycle analysis: {e}")
                return

        # ── 1. TOP SUMMARY METRIC CARDS ──
        c1, c2, c3, c4 = st.columns(4)
        
        # Decide deploy badge color
        decision_style = "badge-positive" if res["decision"] == "Deploy" else "badge-warning" if "Watch" in res["decision"] else "badge-negative"
        
        with c1:
            st.markdown(f"""
            <div class="cycle-card">
                <div class="cycle-header">Asset Decision</div>
                <div style="font-size: 20px; font-weight: 700; margin-bottom: 8px;">{res['recommended_strategy']}</div>
                <span class="{decision_style}">{res['decision']}</span>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="cycle-card">
                <div class="cycle-header">Dominant Rhythm</div>
                <div style="font-size: 24px; font-weight: 700; color: #3ab54a;">{res['dominant_cycle_days']} Days</div>
                <div style="font-size: 12px; color: #94a3b8;">Strength: {res['cycle_strength']}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="cycle-card">
                <div class="cycle-header">Cycle Location</div>
                <div style="font-size: 24px; font-weight: 700; color: #3ab54a;">{res['cycle_phase_pct']}%</div>
                <div style="font-size: 12px; color: #94a3b8;">Phase: {res['phase_label']} ({res['cycle_direction']})</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="cycle-card">
                <div class="cycle-header">Opportunity Score</div>
                <div style="font-size: 24px; font-weight: 700; color: #3ab54a;">{res['confidence_score']}/100</div>
                <div style="font-size: 12px; color: #94a3b8;">Alignment: {res['alignment_score']}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Tab navigation for panels
        tab_chart, tab_alignment, tab_vol, tab_backtest, tab_db = st.tabs([
            "📈 Cycle Chart & Windows", 
            "🔄 Cross-Market Alignment", 
            "⚡ Volatility Dynamics",
            "📊 Backtest & Win Rates",
            "🗄️ Database Audit Logs"
        ])

        # ── TAB 1: CYCLE CHART & WINDOWS ──
        with tab_chart:
            # Let's construct a dual-axis chart (close price + cycle overlay)
            df = res["historical_prices_df"].copy()
            dom_days = res["dominant_cycle_days"]
            
            # Reconstruct the cycle sine wave over the historical period
            n = len(df)
            t = np.arange(n)
            omega = 2.0 * np.pi / dom_days
            
            # Detrend prices
            rolling_mean = df["Close"].rolling(window=20, min_periods=1).mean().values
            detrended = df["Close"].values - rolling_mean
            
            # Simple Sine wave fit
            X = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones(n)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, detrended, rcond=None)
                A, B, _ = coeffs
                phase_offset = np.arctan2(A, B)
                # Compute normalized cycle wave (0% to 100%)
                theta = omega * t + phase_offset
                cycle_wave = ((theta - 1.5 * np.pi) % (2.0 * np.pi)) / (2.0 * np.pi) * 100.0
                
                # Project 30 days into the future
                future_t = np.arange(n, n + 30)
                future_dates = [df.index[-1] + timedelta(days=int(i)) for i in range(1, 31)]
                # Keep dates as business days (approximate)
                future_bus_dates = []
                curr_d = df.index[-1]
                for _ in range(30):
                    curr_d += timedelta(days=1)
                    while curr_d.weekday() >= 5:
                        curr_d += timedelta(days=1)
                    future_bus_dates.append(curr_d)
                
                future_theta = omega * future_t + phase_offset
                future_cycle_wave = ((future_theta - 1.5 * np.pi) % (2.0 * np.pi)) / (2.0 * np.pi) * 100.0
            except Exception:
                cycle_wave = np.sin(omega * t) * 50.0 + 50.0
                future_bus_dates = [df.index[-1] + timedelta(days=int(i)) for i in range(1, 31)]
                future_cycle_wave = np.sin(omega * np.arange(n, n + 30)) * 50.0 + 50.0

            # ── CALCULATE HISTORICAL FORECAST ACCURACY ──
            acc_dates, acc_prices = [], []
            inacc_dates, inacc_prices = [], []
            evaluated_count = 0
            correct_count = 0
            
            for idx in range(5, n - 10, 5):
                c_val = cycle_wave[idx]
                close_t = df["Close"].iloc[idx]
                close_future = df["Close"].iloc[idx + 10]
                price_dir = "Rising" if close_future > close_t else "Falling"
                cycle_dir_t = "Rising" if c_val < 50.0 else "Falling"
                
                is_correct = (price_dir == cycle_dir_t)
                evaluated_count += 1
                if is_correct:
                    correct_count += 1
                    acc_dates.append(df.index[idx])
                    acc_prices.append(close_t)
                else:
                    inacc_dates.append(df.index[idx])
                    inacc_prices.append(close_t)
                    
            accuracy_pct = (correct_count / evaluated_count * 100.0) if evaluated_count > 0 else 0.0

            # Subplots
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Ticker close line
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Close"],
                    name=f"{ticker} Close",
                    line=dict(color="#3ab54a", width=2),
                ),
                secondary_y=False
            )

            # Accurate Forecast Markers
            if acc_dates:
                fig.add_trace(
                    go.Scatter(
                        x=acc_dates,
                        y=acc_prices,
                        mode="markers",
                        name=f"Accurate Forecast ({accuracy_pct:.1f}%)",
                        marker=dict(color="#52D68A", size=8, line=dict(color="black", width=1)),
                    ),
                    secondary_y=False
                )

            # Inaccurate Forecast Markers
            if inacc_dates:
                fig.add_trace(
                    go.Scatter(
                        x=inacc_dates,
                        y=inacc_prices,
                        mode="markers",
                        name="Inaccurate Forecast",
                        marker=dict(color="#FF6B6B", size=8, line=dict(color="black", width=1)),
                    ),
                    secondary_y=False
                )
            
            # Cycle Wave Overlay (Historical)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=cycle_wave,
                    name="Cycle Wave %",
                    line=dict(color="#00ADB5", width=1.5, dash="dash"),
                    opacity=0.6
                ),
                secondary_y=True
            )
            
            # Cycle Wave Overlay (Projected Future)
            fig.add_trace(
                go.Scatter(
                    x=future_bus_dates,
                    y=future_cycle_wave,
                    name="Future Cycle Projection",
                    line=dict(color="#ffb800", width=2.5),
                ),
                secondary_y=True
            )
            
            # Horizontal Support & Resistance
            fig.add_trace(
                go.Scatter(
                    x=[df.index[0], future_bus_dates[-1]],
                    y=[res["support"], res["support"]],
                    name="Support",
                    line=dict(color="#FF6B6B", width=1, dash="dot"),
                ),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(
                    x=[df.index[0], future_bus_dates[-1]],
                    y=[res["resistance"], res["resistance"]],
                    name="Resistance",
                    line=dict(color="#00ADB5", width=1, dash="dot"),
                ),
                secondary_y=False
            )
            
            fig.update_layout(
                title=f"{ticker} Price Action vs Dominant Cycle Wave Overlay &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style='color: #ffb800;'>(Backtested Directional Accuracy: {accuracy_pct:.1f}%)</span>",
                xaxis_title="Date",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=500,
                legend=dict(orientation="h", y=1.08, x=0),
                margin=dict(l=20, r=20, t=60, b=20),
                hoverlabel=dict(
                    bgcolor="#152847",
                    font=dict(
                        color="#ffffff",
                        size=13,
                        family="Outfit, Inter, sans-serif"
                    )
                )
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
            fig.update_yaxes(title_text="Stock Price ($)", secondary_y=False, showgrid=True, gridcolor="rgba(255,255,255,0.05)")
            fig.update_yaxes(title_text="Cycle Phase (%)", secondary_y=True, range=[-10, 110], showgrid=False)
            
            st.plotly_chart(fig, use_container_width=True)

            # Projections windows panel
            w1, w2 = st.columns(2)
            with w1:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding:16px; border-radius:8px;">
                    <div style="font-size:11px; text-transform:uppercase; color:#94a3b8; font-weight:700;">Projected Peak Window</div>
                    <div style="font-size:20px; font-weight:700; color:#ffb800; margin:6px 0;">{res['next_peak_date']}</div>
                    <div style="font-size:12px; color:#e2e8f0;">Window: {res['next_peak_window'][0]} to {res['next_peak_window'][1]}<br>Confidence: {res['peak_confidence']}%</div>
                </div>
                """, unsafe_allow_html=True)
            with w2:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding:16px; border-radius:8px;">
                    <div style="font-size:11px; text-transform:uppercase; color:#94a3b8; font-weight:700;">Projected Bottom Window</div>
                    <div style="font-size:20px; font-weight:700; color:#52D68A; margin:6px 0;">{res['next_bottom_date']}</div>
                    <div style="font-size:12px; color:#e2e8f0;">Window: {res['next_bottom_window'][0]} to {res['next_bottom_window'][1]}<br>Confidence: {res['bottom_confidence']}%</div>
                </div>
                """, unsafe_allow_html=True)

        # ── TAB 2: CROSS-MARKET ALIGNMENT ──
        with tab_alignment:
            st.markdown("### Index Cycle Phase Direction")
            st.caption("Weighted scoring: SPY (30%), QQQ (25%), IWM (15%), DIA (10%), VIX (20% inverse)")
            
            # Fetch the actual alignment statuses
            align_data = calculate_alignment(as_of_date)
            
            indexes = ["SPY", "QQQ", "IWM", "DIA", "VIX"]
            cols = st.columns(len(indexes))
            for i, idx_name in enumerate(indexes):
                with cols[i]:
                    state = align_data[idx_name]
                    badge = "badge-positive" if ((state == "Rising" and idx_name != "VIX") or (state == "Falling" and idx_name == "VIX")) else "badge-negative"
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:14px; text-align:center;">
                        <div style="font-weight:700; font-size:14px; margin-bottom:6px;">{idx_name}</div>
                        <span class="{badge}">{state}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
            st.divider()
            st.markdown(f"**Index Alignment Score**: `{align_data['alignment_score']}/100`  |  **State**: `{align_data['alignment_state']}`")

        # ── TAB 3: VOLATILITY DYNAMICS ──
        with tab_vol:
            st.markdown("### Option Volatility Regime & Suitability")
            
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("VIX Index Value", f"{res['vix_value']}", f"{res['vix_percentile']}% Percentile")
            v2.metric("VVIX Index Value", f"{res['vvix_value']}")
            v3.metric("Term Structure", f"{res['term_structure'].upper()}")
            v4.metric("Calendar Suitability", f"{res['calendar_suitability']}%")

            st.divider()
            st.markdown(f"**Volatility Cycle Status**: `{res['volatility_cycle_status']}`")
            if res["vvix_warning"]:
                st.warning("⚠️ VVIX Warning: Tail risk pricing is elevated. Symmetrical calendar structures are highly vulnerable.")
            else:
                st.success("✅ VVIX is within standard bounds. Tail risk is stabilized.")
                
            st.info("💡 **Calendar spreads** perform best in **Contango** slope shapes under stable or gently rising volatility regimes.")

        # ── TAB 4: BACKTEST & WIN RATES ──
        with tab_backtest:
            st.markdown("### Systematic Signal Backtest Simulation")
            st.caption(f"Simulating historical signals generated in the past {st.session_state.ce_backtest_periods} days.")
            
            bt_results = run_historical_backtest(df, dom_days, st.session_state.ce_backtest_periods)
            if not bt_results:
                st.info("Insufficient price history to complete backtest simulations.")
            else:
                summary = compute_backtest_summary(bt_results)
                
                # Save first backtest record to database for persistence
                for result in bt_results[:10]:
                    save_backtest_result({
                        "signal_id": res["id"],
                        "ticker": ticker,
                        "signal_date": result["signal_date"],
                        "forecast_horizon_days": result["forecast_horizon_days"],
                        "expected_direction": result["expected_direction"],
                        "actual_return": result["actual_return"],
                        "max_favorable_excursion": result["max_favorable_excursion"],
                        "max_adverse_excursion": result["max_adverse_excursion"],
                        "realized_vol_change": 0.0,
                        "strategy": res["recommended_strategy"],
                        "strategy_outcome": "Win" if result["win_flag"] == 1 else "Loss",
                        "pnl_estimate": result["pnl_estimate"],
                        "win_flag": result["win_flag"],
                        "notes": result["phase_label"]
                    })
                
                # Render aggregate stats
                b1, b2, b3 = st.columns(3)
                b1.metric("Historical Win Rate", f"{summary['win_rate']}%")
                b2.metric("Average Forward Return", f"{summary['avg_return']}%")
                b3.metric("Average Max Drawdown (MAE)", f"{summary['avg_drawdown']}%")
                
                st.markdown(f"**Best Performing Phase**: `{summary['best_phase']}`  |  **Worst Performing Phase**: `{summary['worst_phase']}`")
                
                st.divider()
                st.markdown("#### Performance Metrics by Horizon Window")
                # Format into clean table
                rows = []
                for h, stats in summary["horizon_stats"].items():
                    rows.append({
                        "Horizon (Days)": f"{h}d",
                        "Win Rate": f"{stats['win_rate']}%",
                        "Avg Return": f"{stats['avg_return']}%",
                        "Avg Drawdown (MAE)": f"{stats['avg_mae']}%",
                        "Avg Peak Profit (MFE)": f"{stats['avg_mfe']}%",
                        "Signal Count": stats["count"]
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── TAB 5: DATABASE AUDIT LOGS ──
        with tab_db:
            st.markdown("### Saved Signal History (SQLite Database)")
            st.caption("Logging signals generated across active research queries.")
            
            history = load_signal_history(30)
            if not history:
                st.info("No signal records saved in the cycle database yet.")
            else:
                hist_df = pd.DataFrame(history)
                # Keep display columns readable
                display_cols = [
                    "signal_date", "ticker", "timeframe", "dominant_cycle_days", "cycle_strength",
                    "cycle_phase_pct", "cycle_direction", "regime", "recommended_strategy", "confidence_score"
                ]
                st.dataframe(hist_df[display_cols], use_container_width=True, hide_index=True)
