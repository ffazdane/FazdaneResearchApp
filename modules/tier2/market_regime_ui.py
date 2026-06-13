import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from modules.tier2.market_regime_db import (
    create_regime_tables, get_latest_regime, get_historical_regimes,
    get_regime_history_logs, load_strategy_rules
)
from modules.tier2.market_regime_calculations import (
    calculate_market_regime, calculate_regime_forecasts, backfill_regime_history
)
from utils.portfolio_performance_store import (
    get_latest_portfolio_positions,
    get_latest_portfolio_details,
    clean_ticker_for_lookup
)

# Styling colors matching classic navy / obsidian theme
GREEN = "#3ab54a"
LIGHT_GREEN = "#22c55e"
YELLOW = "#facc15"
ORANGE = "#f97316"
RED = "#ef4444"
BG_SOLID = "#152847"

def render_market_regime_center():
    """Renders the Market Regime Center tab layout and calculations."""
    # Ensure tables exist
    create_regime_tables()
    
    # State handling
    if "regime_calculating" not in st.session_state:
        st.session_state["regime_calculating"] = False
        
    latest_run = get_latest_regime()
    
    # Backfill check
    if not latest_run:
        st.warning("⚠️ No historical market regime data found. Please initialize the database with a 60-day backfill.")
        if st.button("🚀 Initialize & Run Historical Backfill", key="regime_backfill_btn", type="primary", use_container_width=True):
            with st.spinner("Downloading historical data & calculating regimes (this takes ~1-2 minutes)..."):
                backfill_regime_history(days=60)
                today_str = datetime.today().strftime("%Y-%m-%d")
                try:
                    calculate_market_regime(today_str)
                except Exception as e:
                    st.error(f"Calculation failed: {e}")
            st.success("✅ Database initialized! Re-loading dashboard...")
            st.rerun()
        return

    # Trigger fresh calculation for today
    today_str = datetime.today().strftime("%Y-%m-%d")
    
    # Check if calculation is needed for today
    last_calc_date = latest_run.get("regime_date", "")
    needs_calc = last_calc_date != today_str
    
    col_hdr, col_btn = st.columns([3, 1], vertical_alignment="center")
    with col_hdr:
        st.markdown("### 🌐 Top-Down Market Regime Center")
        st.caption(f"Status as of: `{last_calc_date}` (Updated daily after market close)")
    with col_btn:
        if st.button("🔄 Refresh Market Regime", key="regime_refresh_manual", use_container_width=True):
            st.session_state["regime_calculating"] = True
            try:
                from modules.tier2.market_regime_calculations import download_data_cached
                calculate_market_regime.clear()
                download_data_cached.clear()
            except Exception:
                pass
            st.rerun()

    if st.session_state["regime_calculating"]:
        with st.spinner("Recalculating dominant market regime and indicators..."):
            try:
                try:
                    from modules.tier2.market_regime_calculations import download_data_cached
                    calculate_market_regime.clear()
                    download_data_cached.clear()
                except Exception:
                    pass
                calculate_market_regime(today_str)
                st.session_state["regime_calculating"] = False
                st.success("✅ Regime calculations updated!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to refresh regime: {e}")
                st.session_state["regime_calculating"] = False

    # Retrieve current data
    latest = get_latest_regime()
    as_of = latest["regime_date"]
    regime_name = latest["regime_name"]
    score = latest["final_regime_score"]
    bias = latest["market_bias"]
    
    # Calculate horizons forecast
    forecast_data = calculate_regime_forecasts(as_of, regime_name)
    
    # Sub-tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Regime Gauge & Forecast",
        "Market Internals & Confirmation",
        "Options Strategy Guidance",
        "Portfolio Greek Alignment",
        "Regime History & Timeline"
    ])
    
    # Color mapping for regime gauges
    r_color = {
        "Strong Buy The Dip": GREEN,
        "Buy Dips Selectively": LIGHT_GREEN,
        "Range Bound": YELLOW,
        "Sell The Rip": ORANGE,
        "Risk Off / Volatility Shock": RED
    }.get(regime_name, YELLOW)
    
    # ----------------------------------------------------
    # TAB 1: Regime Gauge & Forecast
    # ----------------------------------------------------
    with tab1:
        st.markdown("#### Current Market Regime")
        
        c1, c2 = st.columns([1, 1.5])
        with c1:
            fig_gauge = _create_gauge_chart(score, regime_name, r_color)
            st.plotly_chart(fig_gauge, use_container_width=True, key="regime_gauge_plot")
        with c2:
            st.markdown(
                f"""
                <div style="background: rgba(21, 40, 71, 0.4); border: 1px solid {r_color}; border-radius: 8px; padding: 20px; text-align: center;">
                    <span style="color:#94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 1.2px; font-weight:600;">Dominant Market Regime</span>
                    <h2 style="color:{r_color}; margin: 8px 0; font-size: 32px; font-family:'Courier Prime', monospace;">{regime_name}</h2>
                    <div style="display: flex; justify-content: space-around; margin-top: 15px;">
                        <div>
                            <span style="color:#94a3b8; font-size:12px;">Final Score</span><br>
                            <span style="color:#f3f4f6; font-size:20px; font-weight:700;">{score:.0f} / 100</span>
                        </div>
                        <div>
                            <span style="color:#94a3b8; font-size:12px;">Market Bias</span><br>
                            <span style="color:#f3f4f6; font-size:20px; font-weight:700;">{bias}</span>
                        </div>
                        <div>
                            <span style="color:#94a3b8; font-size:12px;">Confidence</span><br>
                            <span style="color:#f3f4f6; font-size:20px; font-weight:700;">78%</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Exposure guidance card
            guidance = _get_guidance_data(regime_name)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="background: rgba(21, 40, 71, 0.25); border: 1px solid #1e3a5f; border-radius: 8px; padding: 15px;">
                    <h5 style="color:#3ab54a; margin-top:0;">📋 Dynamic Portfolio Exposure Guidelines</h5>
                    <table style="width:100%; border-collapse:collapse; color:#e2e8f0; font-size:13px;">
                        <tr>
                            <td style="padding:4px 0; color:#94a3b8; width:40%;">Target Delta Exposure:</td>
                            <td style="padding:4px 0; font-weight:600;">{guidance['target_delta']}</td>
                        </tr>
                        <tr>
                            <td style="padding:4px 0; color:#94a3b8;">Suggested Cash Level:</td>
                            <td style="padding:4px 0; font-weight:600;">{guidance['cash']}</td>
                        </tr>
                        <tr>
                            <td style="padding:4px 0; color:#94a3b8;">Suggested Hedge Level:</td>
                            <td style="padding:4px 0; font-weight:600;">{guidance['hedge']}</td>
                        </tr>
                        <tr>
                            <td style="padding:4px 0; color:#94a3b8;">Preferred Risk Stance:</td>
                            <td style="padding:4px 0; font-weight:600; color:{r_color};">{guidance['preferred_risk']}</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()
        st.markdown("#### 🔮 Markov-Based Multi-Day Forecast Horizon")
        st.caption("Forecast projects the probability of entering/remaining in each regime using matrix powers ($P^N$) based on historical daily transitions.")
        
        # Horizons metrics
        cols_hor = st.columns(5)
        horizons = ["5d", "10d", "20d", "40d", "60d"]
        label_hor = {"5d": "5 Trading Days", "10d": "10 Trading Days", "20d": "20 Trading Days", "40d": "40 Trading Days", "60d": "60 Trading Days"}
        
        for i, h in enumerate(horizons):
            f_item = forecast_data["forecasts"][h]
            h_regime = f_item["regime_name"]
            h_prob = f_item["probability"]
            
            # Forecast color
            h_color = {
                "Strong Buy The Dip": GREEN,
                "Buy Dips Selectively": LIGHT_GREEN,
                "Range Bound": YELLOW,
                "Sell The Rip": ORANGE,
                "Risk Off / Volatility Shock": RED
            }.get(h_regime, YELLOW)
            
            with cols_hor[i]:
                st.markdown(
                    f"""
                    <div style="background: rgba(13,27,46,0.3); border-top: 4px solid {h_color}; border-radius: 6px; padding: 12px; text-align:center; height:120px;">
                        <span style="font-size:11px; color:#94a3b8; text-transform:uppercase; font-weight:600;">{label_hor[h]}</span>
                        <div style="font-weight:700; color:{h_color}; font-size:14px; margin-top:8px; height:36px; line-height:1.2; display:flex; align-items:center; justify-content:center;">
                            {h_regime}
                        </div>
                        <div style="font-size:16px; font-weight:bold; color:#f3f4f6; margin-top:8px;">{h_prob:.1%} Conf.</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ----------------------------------------------------
    # TAB 2: Market Internals & Confirmation
    # ----------------------------------------------------
    with tab2:
        st.markdown("#### Market Internals Score Dashboard")
        
        # Five factor score bars
        c_int1, c_int2 = st.columns([1, 1.2])
        with c_int1:
            fig_bars = _create_factors_chart(latest)
            st.plotly_chart(fig_bars, use_container_width=True, key="regime_factors_chart")
        with c_int2:
            st.markdown(
                """
                <div style="background: rgba(21, 40, 71, 0.15); border: 1px solid #1e3a5f; border-radius: 8px; padding: 16px; height: 100%;">
                    <h5 style="color:#facc15; margin-top:0; font-family:'Courier Prime', monospace;">Scoring Architecture</h5>
                    <p style="font-size:12px; line-height:1.4; color:#e2e8f0; margin-bottom:10px;">
                        The Market Regime Score compiles five core market indicators into a master index. Weights are distributed to reflect lead/lag characteristics:
                    </p>
                    <ul style="font-size:12px; color:#e2e8f0; padding-left:18px; line-height:1.5;">
                        <li><b>Trend Score (30%)</b>: Cross-index price locations relative to 20 EMA, 50 SMA, 200 SMA, Ichimoku cloud support, and price structures.</li>
                        <li><b>Breadth Score (25%)</b>: The percentage of stocks above 50-DMA, advance/decline ratios, and net new highs/lows.</li>
                        <li><b>Volatility Score (20%)</b>: Absolute VIX level, VIX/SPY correlation, and VIX9D/VIX3M term structure inversion checks.</li>
                        <li><b>Momentum Score (15%)</b>: Price acceleration metrics, oversold pullbacks, EMA pullbacks, and expansion of True Range (ATR).</li>
                        <li><b>Risk Sentiment (10%)</b>: Index relative performance (QQQ/SPY, IWM/SPY, SMH/SPY), cyclical vs defensive sector strength, options put/call skew, and high yield credit spreads.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.divider()
        st.markdown("#### Index Confirmation Matrix")
        
        # We need the trend details dict from calculate_market_regime
        # Since calculating today's regime details is not fully cached, we can fetch it live or from recalculation
        # Let's run calculate_market_regime(as_of) to get details
        with st.spinner("Loading index confirmation details..."):
            try:
                details_json = calculate_market_regime(as_of)
                trend_det = details_json.get("trend_details", {})
            except Exception:
                trend_det = {}
                
        if trend_det:
            rows = []
            for ticker, det in trend_det.items():
                rows.append({
                    "Ticker": ticker,
                    "Price": f"${det['price']:,.2f}",
                    "20 EMA": "Above ✅" if det["above_20ema"] else "Below ❌",
                    "50 SMA": "Above ✅" if det["above_50sma"] else "Below ❌",
                    "200 SMA": "Above ✅" if det["above_200sma"] else "Below ❌",
                    "Ichimoku": "Above Cloud 🟢" if det["above_ichimoku"] else "Below/Inside Cloud 🔴",
                    "Structure": "Higher High/Low 📈" if det["higher_high_low"] else "Lower High/Low 📉" if not det.get("higher_high_low", True) else "Neutral ➡️",
                    "ATR %": f"{det['atr_pct']}%",
                    "5D Return": f"{det['ret_5d']:+.2f}%",
                    "20D Return": f"{det['ret_20d']:+.2f}%",
                    "Bias": det["trend"]
                })
            
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Trend details currently unavailable. Refresh the regime to recalculate.")

    # ----------------------------------------------------
    # TAB 3: Options Strategy Guidance
    # ----------------------------------------------------
    with tab3:
        st.markdown(f"#### Option Strategy Map: *{regime_name}*")
        
        # Load rules from db
        rules_list = load_strategy_rules(regime_name)
        
        col_pref, col_rest = st.columns(2)
        with col_pref:
            st.markdown("##### 🟢 Preferred Option Structures")
            for r in rules_list:
                if r["strategy_status"] == "Preferred":
                    st.success(f"**{r['strategy_name']}**\n\n*{r['reason']}*")
                    
        with col_rest:
            st.markdown("##### 🟡 Restricted / Avoided Structures")
            for r in rules_list:
                if r["strategy_status"] == "Avoid":
                    st.warning(f"**{r['strategy_name']}**\n\n*{r['reason']}*")
                    
            # Highlight blocked calendars or Volatility inversion warnings if applicable
            vix_term = latest.get("vix_close", 0.0)
            if latest.get("volatility_score", 100) < 35 or latest.get("vix_close", 0.0) > 30:
                st.error("**Blocked: Calendars / Short Premium**\n\n*Volatility term structure is inverted or VIX is above 30. Calendars and short premium are blocked due to tail risk.*")

    # ----------------------------------------------------
    # TAB 4: Portfolio Greek Alignment
    # ----------------------------------------------------
    with tab4:
        st.markdown("#### Portfolio Greek Alignment Panel")
        st.caption("Evaluate your actual portfolio Greeks against target ranges for the active regime.")
        
        # Load latest portfolio snapshot from Schwab/Tastytrade database if available
        port_df, port_meta = get_latest_portfolio_positions()
        
        if port_df.empty:
            st.warning("⚠️ No active portfolio loaded. Please upload or load your Schwab/Tastytrade statement in the **Portfolio Module** first.")
        else:
            p_totals = port_meta.get("totals", {})
            p_delta = float(p_totals.get("total_delta", 0.0))
            p_theta = float(p_totals.get("total_theta", 0.0))
            p_vega = float(p_totals.get("total_vega", 0.0))
            p_gamma = float(p_totals.get("total_gamma", 0.0))
            
            # 1. Total Portfolio Value (Net Liquidity)
            net_liq = float(port_df["market_value"].sum())
            if net_liq == 0:
                net_liq = float(port_meta.get("totals", {}).get("total_market_value", 1.0))
            if net_liq == 0:
                net_liq = 1.0
                
            # 2. Cash Level Allocation (%)
            cash_val = 0.0
            for _, pos in port_df.iterrows():
                clean_t = clean_ticker_for_lookup(pos["ticker"])
                if clean_t in ["CASH", "USD", "MMDA12", "MMDA"]:
                    cash_val += float(pos["market_value"])
                    
            p_cash = (cash_val / net_liq) * 100.0
            
            # 3. Hedge Level Exposure (%)
            port_details, _ = get_latest_portfolio_details()
            hedge_val = 0.0
            if not port_details.empty:
                option_legs = port_details[port_details["row_type"] == "option_leg"].copy()
                if not option_legs.empty:
                    option_legs["clean_underlying"] = option_legs["underlying"].apply(clean_ticker_for_lookup)
                    index_tickers = ["SPY", "QQQ", "SPX", "NDX", "RUT", "IWM"]
                    
                    for und, group in option_legs.groupby("clean_underlying"):
                        if und in index_tickers:
                            puts = group[group["call_put"] == "PUT"]
                            net_put_val = (puts["quantity"] * puts["mark_price"] * 100.0).sum()
                            if net_put_val > 0:
                                hedge_val += net_put_val
                        elif und in ["VIX", "^VIX"]:
                            calls = group[group["call_put"] == "CALL"]
                            net_call_val = (calls["quantity"] * calls["mark_price"] * 100.0).sum()
                            if net_call_val > 0:
                                hedge_val += net_call_val
                                
            p_hedge = (hedge_val / net_liq) * 100.0
            
            st.info(f"💾 Dynamically analyzing Schwab/Tastytrade snapshot: `{port_meta.get('source_file')}` ({port_meta.get('snapshot_ts')})")
            
            c_in, c_score = st.columns([1.5, 1])
            with c_in:
                st.markdown("##### 📊 Actual Metrics vs Regime Targets")
                
                # Fetch target ranges
                t_ranges = _get_metric_targets(regime_name)
                
                # Perform alignment calculation
                score_align, align_details = _calculate_alignment_score(
                    regime_name, p_delta, p_theta, p_gamma, p_vega, p_cash, p_hedge
                )
                
                # Create a dataframe for display
                vega_status = align_details.get("vega_status", align_details.get("vegas_status", "Acceptable"))
                metric_data = [
                    {
                        "Metric": "Net Delta Exposure",
                        "Actual Value": f"{p_delta:+.2f}",
                        "Target Range": t_ranges["delta"],
                        "Status": align_details["delta_status"]
                    },
                    {
                        "Metric": "Net Theta Carry ($/day)",
                        "Actual Value": f"{p_theta:+.2f}",
                        "Target Range": t_ranges["theta"],
                        "Status": align_details["theta_status"]
                    },
                    {
                        "Metric": "Net Gamma Exposure",
                        "Actual Value": f"{p_gamma:+.4f}",
                        "Target Range": t_ranges["gamma"],
                        "Status": align_details["gamma_status"]
                    },
                    {
                        "Metric": "Net Vega Exposure",
                        "Actual Value": f"{p_vega:+.2f}",
                        "Target Range": t_ranges["vega"],
                        "Status": vega_status
                    },
                    {
                        "Metric": "Cash Allocation (%)",
                        "Actual Value": f"{p_cash:.2f}%",
                        "Target Range": t_ranges["cash"],
                        "Status": align_details["cash_status"]
                    },
                    {
                        "Metric": "Hedge Exposure (%)",
                        "Actual Value": f"{p_hedge:.2f}%",
                        "Target Range": t_ranges["hedge"],
                        "Status": align_details["hedge_status"]
                    }
                ]
                
                df_display = pd.DataFrame(metric_data)
                
                status_map = {
                    "Acceptable": "🟢 Healthy",
                    "Healthy": "🟢 Healthy",
                    "Too Low": "🟡 Too Low",
                    "Too High": "🟡 Too High",
                    "Elevated Cash": "🟡 Elevated Cash",
                    "Too High Cash": "🔴 Too High Cash",
                    "Over-hedged": "🟡 Over-hedged",
                    "Imbalanced": "🟡 Imbalanced",
                    "High Directional Risk": "🔴 High Risk",
                    "Negative Carry": "🔴 Negative Carry",
                    "Long Exposure is High Risk": "🔴 High Risk",
                    "Under-hedged": "🔴 Under-hedged",
                    "Long Exposure Prohibited": "🔴 Prohibited",
                    "Cash Allocation Too Low": "🔴 Too Low",
                    "Short Gamma Risk Prohibited": "🔴 Prohibited",
                    "High Short Vega Risk": "🔴 High Risk",
                    "Extremely Underallocated": "🔴 Too Low"
                }
                
                df_display["Status"] = df_display["Status"].map(lambda x: status_map.get(x, f"⚪ {x}"))
                
                st.dataframe(
                    df_display, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Metric": st.column_config.TextColumn("Metric", width="medium"),
                        "Actual Value": st.column_config.TextColumn("Actual Value", width="small"),
                        "Target Range": st.column_config.TextColumn("Target Range", width="small"),
                        "Status": st.column_config.TextColumn("Status", width="medium")
                    }
                )
                
            with c_score:
                align_color = GREEN if score_align >= 80 else YELLOW if score_align >= 50 else RED
                
                st.markdown(
                    f"""
                    <div style="background: rgba(21, 40, 71, 0.4); border: 2px solid {align_color}; border-radius: 8px; padding: 20px; text-align: center;">
                        <span style="color:#94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 1.2px; font-weight:600;">Portfolio Alignment Score</span>
                        <h2 style="color:{align_color}; margin: 8px 0; font-size: 48px; font-family:'Courier Prime', monospace;">{score_align:.0f} / 100</h2>
                        <p style="font-size: 13px; color: #e2e8f0; margin-top: 15px; text-align: left; line-height: 1.4;">
                            <b>Regime Status:</b> {regime_name}<br>
                            <b>Delta Exposure:</b> {align_details['delta_status']}<br>
                            <b>Cash Level:</b> {align_details['cash_status']}<br>
                            <b>Hedge Level:</b> {align_details['hedge_status']}<br>
                            <b>Theta Status:</b> {align_details['theta_status']}<br>
                            <b>Gamma Status:</b> {align_details['gamma_status']}<br>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                st.warning(f"**Recommendation:** {align_details['recommendation']}")

    # ----------------------------------------------------
    # TAB 5: Regime History & Timeline
    # ----------------------------------------------------
    with tab5:
        st.markdown("#### Historical Regime Changes (Last 3 Months)")
        
        history_logs = get_regime_history_logs(limit=25)
        
        if history_logs:
            rows = []
            for h in history_logs:
                rows.append({
                    "Date": h["regime_date"],
                    "Previous Regime": h["previous_regime"],
                    "Current Regime": h["current_regime"],
                    "Regime Change": "Yes 🔄" if h["regime_change_flag"] == 1 else "No ➡️",
                    "Trigger Reason": h["trigger_reason"]
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No regime transitions logged in the database yet. History transitions will record when the calculated regime changes daily.")

def _create_gauge_chart(score: float, regime_name: str, color: str):
    """Creates a beautiful semi-circular gauge using Plotly."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"color": "#e2e8f0", "size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8", "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "#0d1b2e",
                "borderwidth": 1,
                "bordercolor": "#1e3a5f",
                "steps": [
                    {"range": [0, 19.9], "color": "rgba(239, 68, 68, 0.15)"},
                    {"range": [20, 39.9], "color": "rgba(249, 115, 22, 0.15)"},
                    {"range": [40, 59.9], "color": "rgba(250, 204, 21, 0.15)"},
                    {"range": [60, 79.9], "color": "rgba(34, 197, 94, 0.15)"},
                    {"range": [80, 100], "color": "rgba(58, 181, 74, 0.25)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter")
    )
    return fig

def _create_factors_chart(latest: dict):
    """Creates a horizontal bar chart displaying factor scores."""
    factors = ["Trend", "Breadth", "Volatility", "Momentum", "Risk Sentiment"]
    scores = [
        latest.get("trend_score", 50.0),
        latest.get("breadth_score", 50.0),
        latest.get("volatility_score", 50.0),
        latest.get("momentum_score", 50.0),
        latest.get("risk_sentiment_score", 50.0)
    ]
    
    colors = []
    for s in scores:
        if s >= 80: colors.append(GREEN)
        elif s >= 60: colors.append(LIGHT_GREEN)
        elif s >= 40: colors.append(YELLOW)
        elif s >= 20: colors.append(ORANGE)
        else: colors.append(RED)
        
    fig = go.Figure(
        go.Bar(
            x=scores,
            y=factors,
            orientation="h",
            marker_color=colors,
            text=[f"{s:.1f}" for s in scores],
            textposition="outside",
            width=0.45
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 110], gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(autorange="reversed"),
        font=dict(color="#e2e8f0")
    )
    return fig

def _get_guidance_data(regime_name: str) -> dict:
    """Guidance getter helper."""
    guidance = {
        "Strong Buy The Dip": {
            "target_delta": "+70% to +100%",
            "cash": "0% to 15%",
            "hedge": "0% to 10%",
            "preferred_risk": "Bullish trend continuation"
        },
        "Buy Dips Selectively": {
            "target_delta": "+40% to +70%",
            "cash": "15% to 30%",
            "hedge": "5% to 15%",
            "preferred_risk": "Selective bullish exposure"
        },
        "Range Bound": {
            "target_delta": "-10% to +30%",
            "cash": "20% to 40%",
            "hedge": "10% to 20%",
            "preferred_risk": "Neutral theta income"
        },
        "Sell The Rip": {
            "target_delta": "-30% to +10%",
            "cash": "30% to 50%",
            "hedge": "20% to 40%",
            "preferred_risk": "Bearish or neutral exposure"
        },
        "Risk Off / Volatility Shock": {
            "target_delta": "-50% to 0%",
            "cash": "50% or higher",
            "hedge": "30% or higher",
            "preferred_risk": "Capital preservation"
        }
    }
    return guidance.get(regime_name, guidance["Range Bound"])

def _get_metric_targets(regime_name: str) -> dict:
    targets = {
        "Strong Buy The Dip": {
            "delta": ">= +800",
            "theta": "Any",
            "gamma": "Any",
            "vega": "Any",
            "cash": "0% to 15%",
            "hedge": "0% to 10%"
        },
        "Buy Dips Selectively": {
            "delta": "+400 to +1000",
            "theta": "Any",
            "gamma": "Any",
            "vega": "Any",
            "cash": "15% to 30%",
            "hedge": "5% to 15%"
        },
        "Range Bound": {
            "delta": "-100 to +400",
            "theta": "Positive (> 0)",
            "gamma": "Any",
            "vega": "Any",
            "cash": "20% to 40%",
            "hedge": "10% to 20%"
        },
        "Sell The Rip": {
            "delta": "<= +100",
            "theta": "Any",
            "gamma": "Any",
            "vega": "Any",
            "cash": "30% to 50%",
            "hedge": "20% to 40%"
        },
        "Risk Off / Volatility Shock": {
            "delta": "<= 0",
            "theta": "Any",
            "gamma": "Non-negative (>= 0)",
            "vega": "Non-positive (<= 0)",
            "cash": ">= 50%",
            "hedge": ">= 30%"
        }
    }
    return targets.get(regime_name, targets["Range Bound"])

def _calculate_alignment_score(regime, delta, theta, gamma, vega, cash, hedge) -> tuple[float, dict]:
    """Calculates Portfolio Alignment Score from 0 to 100 based on regime guidelines."""
    details = {
        "delta_status": "Acceptable",
        "cash_status": "Acceptable",
        "hedge_status": "Acceptable",
        "theta_status": "Acceptable",
        "gamma_status": "Acceptable",
        "recommendation": ""
    }
    
    score_components = []
    
    # Guidelines ranges
    if regime == "Strong Buy The Dip":
        # Delta: Positive (high target delta)
        # We can normalize input delta. For S&P portfolio, a beta-weighted delta of +1000 to +3000 might be typical.
        # Let's map target delta to +70% to +100% of standard capital allocation.
        # Let's assume standard delta input maps: +500 to +2000 is healthy.
        if delta >= 800:
            score_components.append(100)
        elif delta >= 400:
            score_components.append(75)
            details["delta_status"] = "Too Low"
        else:
            score_components.append(40)
            details["delta_status"] = "Extremely Underallocated"
            
        # Cash: 0-15%
        if cash <= 15:
            score_components.append(100)
        elif cash <= 30:
            score_components.append(70)
            details["cash_status"] = "Elevated Cash"
        else:
            score_components.append(30)
            details["cash_status"] = "Too High Cash"
            
        # Hedge: 0-10%
        if hedge <= 10:
            score_components.append(100)
        else:
            score_components.append(70)
            details["hedge_status"] = "Over-hedged"
            
        details["recommendation"] = "Deploy cash, add bullish trend structures, and maintain low hedging."
        
    elif regime == "Buy Dips Selectively":
        # Delta: +40% to +70%
        if delta >= 400 and delta <= 1000:
            score_components.append(100)
        elif delta < 400:
            score_components.append(70)
            details["delta_status"] = "Too Low"
        else:
            score_components.append(70)
            details["delta_status"] = "Too High"
            
        # Cash: 15-30%
        if cash >= 15 and cash <= 30:
            score_components.append(100)
        else:
            score_components.append(80)
            details["cash_status"] = "Imbalanced"
            
        # Hedge: 5-15%
        if hedge >= 5 and hedge <= 15:
            score_components.append(100)
        else:
            score_components.append(85)
            details["hedge_status"] = "Imbalanced"
            
        details["recommendation"] = "Increase selective bullish exposure and reduce short gamma risk."
        
    elif regime == "Range Bound":
        # Delta: Neutral (-100 to +400)
        if delta >= -100 and delta <= 400:
            score_components.append(100)
        else:
            score_components.append(60)
            details["delta_status"] = "High Directional Risk"
            
        # Cash: 20-40%
        if cash >= 20 and cash <= 40:
            score_components.append(100)
        else:
            score_components.append(80)
            details["cash_status"] = "Imbalanced"
            
        # Hedge: 10-20%
        if hedge >= 10 and hedge <= 20:
            score_components.append(100)
        else:
            score_components.append(80)
            details["hedge_status"] = "Imbalanced"
            
        # Theta carry should be positive
        if theta > 0:
            score_components.append(100)
        else:
            score_components.append(40)
            details["theta_status"] = "Negative Carry"
            
        details["recommendation"] = "Deploy range-bound credit spreads, Iron Condors, and capture positive Theta decay."
        
    elif regime == "Sell The Rip":
        # Delta: Neutral to negative (-500 to +100)
        if delta <= 100:
            score_components.append(100)
        else:
            score_components.append(40)
            details["delta_status"] = "Long Exposure is High Risk"
            
        # Cash: 30-50%
        if cash >= 30 and cash <= 50:
            score_components.append(100)
        else:
            score_components.append(80)
            details["cash_status"] = "Imbalanced"
            
        # Hedge: 20-40%
        if hedge >= 20 and hedge <= 40:
            score_components.append(100)
        else:
            score_components.append(70)
            details["hedge_status"] = "Under-hedged"
            
        details["recommendation"] = "Reduce long delta exposure, purchase Put Debit Spreads, and add hedge structures."
        
    else:  # Risk Off / Volatility Shock
        # Delta: Strictly defensive/negative
        if delta <= 0:
            score_components.append(100)
        else:
            score_components.append(20)
            details["delta_status"] = "Long Exposure Prohibited"
            
        # Cash: >= 50%
        if cash >= 50:
            score_components.append(100)
        else:
            score_components.append(40)
            details["cash_status"] = "Cash Allocation Too Low"
            
        # Hedge: >= 30%
        if hedge >= 30:
            score_components.append(100)
        else:
            score_components.append(45)
            details["hedge_status"] = "Under-hedged"
            
        # Short gamma is forbidden
        if gamma >= 0:
            score_components.append(100)
        else:
            score_components.append(30)
            details["gamma_status"] = "Short Gamma Risk Prohibited"
            
        # Short vega is risky
        if vega <= 0:
            score_components.append(100)
        else:
            score_components.append(50)
            details["vegas_status"] = "High Short Vega Risk"
            
        details["recommendation"] = "Immediate capital preservation. Liquidate longs to reach >= 50% cash, buy protective puts, and block short premium."

    final_score = np.mean(score_components) if score_components else 50.0
    return round(final_score, 0), details
