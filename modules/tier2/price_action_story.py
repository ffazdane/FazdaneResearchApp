"""
FazDane Analytics — Tier 2
Price Action Story Engine
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import yfinance as yf
import logging
from datetime import datetime, timedelta
from pathlib import Path
from modules.base_module import FazDaneModule
from utils.universe_manager import render_universe_manager, format_ticker_display, get_ticker_names
from utils.persistence import get_db_path, backup_database
from modules.calendar_scoring.technical_indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_adx,
    calculate_atr,
    compute_rrg_ratio_ema as calculate_rrg_values,
    calculate_fdts_ha_signal,
    format_fdts_signal,
    evaluate_price_action_lifecycle as evaluate_ticker_price_action
)

logger = logging.getLogger("PriceActionStory")

# Color palette for lifecycle stages
STAGE_DETAILS = {
    "Early Bull / Expansion": {"color": "#22c55e", "bg": "rgba(34,197,94,0.12)", "desc": "Bullish breakout, strong accumulation, volume confirming takeoff."},
    "Strong Bull": {"color": "#10b981", "bg": "rgba(16,185,129,0.12)", "desc": "Healthy momentum expansion, trend leadership intact."},
    "Mature Bull": {"color": "#eab308", "bg": "rgba(234,179,8,0.12)", "desc": "Price making new highs, volume participation starting to plateau."},
    "Fading Bull": {"color": "#f97316", "bg": "rgba(249,115,22,0.12)", "desc": "Price rising on declining volume, momentum divergence appearing."},
    "Distribution": {"color": "#ef4444", "bg": "rgba(239,68,68,0.12)", "desc": "Institutional selling, failed breakouts, multiple distribution days."},
    "Breakdown": {"color": "#991b1b", "bg": "rgba(153,27,27,0.15)", "desc": "Support failure, trend breakdown, volume expanding downwards."}
}

COLORS = ["#06b6d4", "#3b82f6", "#10b981", "#f59e0b", "#6366f1", "#ec4899", "#8b5cf6", "#14b8a6", "#f97316", "#84cc16", "#ef4444", "#a855f7", "#fbbf24", "#34d399", "#f87171"]

def color_stage_cell(val):
    if val in STAGE_DETAILS:
        color = STAGE_DETAILS[val]["color"]
        bg = STAGE_DETAILS[val]["bg"]
        return f"background-color: {bg}; color: {color}; font-weight: bold;"
    return ""

def color_fdts_cell(val):
    if "Buy" in str(val):
        return "color: #22c55e; font-weight: bold;"
    elif "Sell" in str(val):
        return "color: #ef4444; font-weight: bold;"
    return "color: #94a3b8;"

# =====================================================================
# Database Persistence
# =====================================================================

def _ensure_story_schema(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS price_action_runs (
                run_id TEXT PRIMARY KEY,
                scan_ts TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                universe_name TEXT NOT NULL,
                ticker_count INTEGER NOT NULL
            );
            DROP TABLE IF EXISTS ticker_stage_history;
            CREATE TABLE IF NOT EXISTS ticker_stage_history (
                run_id TEXT NOT NULL,
                scan_ts TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                health_score REAL NOT NULL,
                stage TEXT NOT NULL,
                fdts TEXT,
                close REAL,
                volume REAL,
                vpr REAL,
                rs_ratio REAL,
                rs_momentum REAL,
                rsi REAL,
                macd_line REAL,
                macd_signal REAL,
                adx REAL,
                atr REAL,
                cvd REAL,
                distribution_days INTEGER,
                PRIMARY KEY (run_id, ticker),
                FOREIGN KEY (run_id) REFERENCES price_action_runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_ticker_stage_history_ticker_ts
                ON ticker_stage_history(ticker, scan_ts);
        """)

def save_scan_run(universe_name: str, data_df: pd.DataFrame) -> str:
    """Save a scanner run to SQLite and trigger cloud backup."""
    try:
        db_path = get_db_path("price_action_story")
        _ensure_story_schema(db_path)
        
        scan_ts = datetime.now().replace(microsecond=0)
        trade_date = scan_ts.date().isoformat()
        run_id = f"pa_{scan_ts.strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO price_action_runs (run_id, scan_ts, trade_date, universe_name, ticker_count) VALUES (?, ?, ?, ?, ?)",
                (run_id, scan_ts.isoformat(sep=" "), trade_date, universe_name, len(data_df))
            )
            
            records = []
            for _, r in data_df.iterrows():
                records.append((
                    run_id,
                    scan_ts.isoformat(sep=" "),
                    trade_date,
                    r["Ticker"],
                    float(r["Health Score"]),
                    r["Stage"],
                    r["FDTS"],
                    float(r["Close"]),
                    float(r["Volume"]),
                    float(r["VPR"]),
                    float(r["RS Ratio"]),
                    float(r["RS Momentum"]),
                    float(r["RSI"]),
                    float(r["MACD Line"]),
                    float(r["MACD Signal"]),
                    float(r["ADX"]),
                    float(r["ATR"]),
                    float(r["CVD"]),
                    int(r["Distribution Days"])
                ))
                
            conn.executemany(
                """
                INSERT INTO ticker_stage_history (
                    run_id, scan_ts, trade_date, ticker, health_score, stage, fdts,
                    close, volume, vpr, rs_ratio, rs_momentum, rsi,
                    macd_line, macd_signal, adx, atr, cvd, distribution_days
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records
            )
            
        backup_database("price_action_story", reason=f"Scan: {universe_name}")
        return run_id
    except Exception as e:
        logger.error(f"Failed to save price action scan: {e}")
        return ""

def get_historical_stages(ticker: str) -> pd.DataFrame:
    """Retrieve historical stage records for a specific ticker."""
    try:
        db_path = get_db_path("price_action_story")
        if not db_path.exists():
            return pd.DataFrame()
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT trade_date, health_score, stage, close, volume, vpr, rs_ratio, rs_momentum
                FROM ticker_stage_history
                WHERE ticker = ?
                ORDER BY scan_ts ASC
                """,
                conn,
                params=(ticker.upper(),)
            )
            return df
    except Exception as e:
        logger.error(f"Failed to fetch historical stages for {ticker}: {e}")
        return pd.DataFrame()

# =====================================================================
# Math & Indicators Calculations
# =====================================================================

def get_fdts_signal(ticker_df: pd.DataFrame, period: int = 20) -> str:
    raw_sig = calculate_fdts_ha_signal(ticker_df, period)
    return format_fdts_signal(raw_sig)

# =====================================================================
# Options & Earnings Integrations
# =====================================================================

def query_options_liquidity_store(tickers: list[str]) -> dict[str, dict]:
    """Fetch stored options scan metrics if they exist."""
    summary = {}
    try:
        from utils.options_liquidity_store import DB_PATH as ol_db_path
        if not ol_db_path.exists():
            return {}
        with sqlite3.connect(ol_db_path) as conn:
            # Query the latest run summary for each symbol
            placeholders = ",".join("?" for _ in tickers)
            query = f"""
                SELECT symbol, total_volume, total_open_interest, avg_iv_pct, median_spread_pct, contract_count
                FROM ol_symbol_snapshot_summary
                WHERE symbol IN ({placeholders})
                GROUP BY symbol
                HAVING scan_ts = MAX(scan_ts)
            """
            rows = conn.execute(query, tickers).fetchall()
            for r in rows:
                summary[r[0]] = {
                    "total_volume": r[1],
                    "total_oi": r[2],
                    "avg_iv": r[3],
                    "median_spread_pct": r[4],
                    "contract_count": r[5]
                }
    except Exception as e:
        logger.warning(f"Could not read options liquidity database: {e}")
    return summary

def query_earnings_calendar_store(tickers: list[str]) -> dict[str, str]:
    """Fetch next earnings date for a set of symbols."""
    dates = {}
    try:
        from utils.earnings_calendar_store import DB_PATH as ec_db_path
        if not ec_db_path.exists():
            return {}
        with sqlite3.connect(ec_db_path) as conn:
            placeholders = ",".join("?" for _ in tickers)
            today_str = datetime.today().strftime("%Y-%m-%d")
            query = f"""
                SELECT ticker, MIN(date)
                FROM ec_earnings_events
                WHERE ticker IN ({placeholders}) AND date >= ?
                GROUP BY ticker
            """
            rows = conn.execute(query, [*tickers, today_str]).fetchall()
            for r in rows:
                dates[r[0]] = r[1]
    except Exception as e:
        logger.warning(f"Could not read earnings calendar database: {e}")
    return dates

# =====================================================================
# Main Module Class
# =====================================================================

class PriceActionStoryModule(FazDaneModule):
    MODULE_NAME = "Price Action Story Engine"
    MODULE_ICON = "📈"
    MODULE_DESCRIPTION = "Analyzes underlying trend health beyond RSI to classify market lifecycle, find option candidates, and review statistics."
    TIER = 2
    SOURCE_NOTEBOOK = "FazDane Price Action Lifecycle"
    CACHE_TTL = 3600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance", "price_action_story_sqlite", "options_liquidity_sqlite"]

    def render_sidebar(self):
        st.markdown("**Scanned Scope**")
        universe_name, tickers_list, benchmark = render_universe_manager(
            key_prefix="pa",
            show_benchmark=True,
            label="Target Universe:"
        )
        
        # Save sidebar selections
        st.session_state["pa_universe_name"] = universe_name
        st.session_state["pa_tickers"] = list(tickers_list)
        st.session_state["pa_benchmark"] = benchmark
        
        st.markdown("**Parameters**")
        lookback = st.slider("Lookback Days:", 120, 500, 252, step=20, key="pa_lookback")
        st.session_state["pa_lookback_days"] = lookback

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        scan_clicked = st.button("📊 Analyze Price Action", use_container_width=True, type="primary")
        
        if scan_clicked or "pa_scanned_data" not in st.session_state:
            st.session_state["pa_trigger_scan"] = True

    def render_main(self):
        # 1. Gather variables from session state
        universe_name = st.session_state.get("pa_universe_name", "SPX Sectors")
        tickers = st.session_state.get("pa_tickers", ["XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLRE","XLK","XLU"])
        benchmark = st.session_state.get("pa_benchmark", "SPY")
        lookback_days = st.session_state.get("pa_lookback_days", 252)

        # Batch download data
        if st.session_state.pop("pa_trigger_scan", False) or "pa_scanned_data" not in st.session_state:
            with st.spinner("Fetching data and evaluating lifecycle stages..."):
                try:
                    all_tickers = sorted(list(set(tickers + [benchmark, "QQQ", "IWM"])))
                    start_date = datetime.today() - timedelta(days=lookback_days + 150)
                    
                    raw_data = yf.download(
                        all_tickers,
                        start=start_date.strftime("%Y-%m-%d"),
                        end=(datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
                        group_by="column",
                        progress=False
                    )
                    
                    if raw_data.empty:
                        st.error("No historical data returned. Check symbols/benchmark.")
                        return

                    # Extract benchmark dataframes
                    def extract_single(df, sym):
                        res = pd.DataFrame(index=df.index)
                        if isinstance(df.columns, pd.MultiIndex):
                            for col in ["Open", "High", "Low", "Close", "Volume"]:
                                if col in df and sym in df[col].columns:
                                    res[col] = df[col][sym]
                        else:
                            res = df.copy()
                        return res.dropna(how="all")

                    bench_df = extract_single(raw_data, benchmark)
                    qqq_df = extract_single(raw_data, "QQQ")
                    iwm_df = extract_single(raw_data, "IWM")

                    results = []
                    # Keep raw ticker dataframes cached locally for detail pages
                    ticker_data_cache = {}
                    
                    for ticker in tickers:
                        if ticker == benchmark:
                            continue
                        ticker_raw = extract_single(raw_data, ticker)
                        if ticker_raw.empty or len(ticker_raw) < 50:
                            continue
                        
                        ticker_data_cache[ticker] = ticker_raw
                        eval_res = evaluate_ticker_price_action(ticker_raw, bench_df)
                        if eval_res:
                            eval_res["Ticker"] = ticker
                            # Fetch name if available
                            eval_res["Name"] = ticker
                            results.append(eval_res)
                    
                    # Also evaluate benchmarks for dashboard
                    bench_res = evaluate_ticker_price_action(bench_df, bench_df)
                    if bench_res: bench_res["Ticker"] = benchmark; bench_res["Name"] = "S&P 500 ETF"
                    
                    qqq_res = evaluate_ticker_price_action(qqq_df, bench_df)
                    if qqq_res: qqq_res["Ticker"] = "QQQ"; qqq_res["Name"] = "Nasdaq 100 ETF"
                    
                    iwm_res = evaluate_ticker_price_action(iwm_df, bench_df)
                    if iwm_res: iwm_res["Ticker"] = "IWM"; iwm_res["Name"] = "Russell 2000 ETF"
                    
                    scanned_df = pd.DataFrame(results)
                    st.session_state["pa_scanned_data"] = scanned_df
                    st.session_state["pa_ticker_cache"] = ticker_data_cache
                    st.session_state["pa_benchmarks"] = {"SPY": bench_res, "QQQ": qqq_res, "IWM": iwm_res}
                    st.session_state["pa_bench_df"] = bench_df
                    
                    # Save scan run to sqlite database
                    if not scanned_df.empty:
                        save_scan_run(universe_name, scanned_df)
                        
                except Exception as e:
                    st.error(f"Failed to compile price action scans: {e}")
                    logger.error("Scan compilation error", exc_info=True)
                    return

        scanned_df = st.session_state.get("pa_scanned_data", pd.DataFrame())
        ticker_cache = st.session_state.get("pa_ticker_cache", {})
        benchmarks = st.session_state.get("pa_benchmarks", {})
        bench_df = st.session_state.get("pa_bench_df", pd.DataFrame())

        if scanned_df.empty:
            st.warning("Universe scan results are empty. Adjust tickers or click 'Analyze Price Action' in the sidebar.")
            return

        # Main Layout Header
        st.markdown(
            f"""
            <div style="background:linear-gradient(90deg, #1e293b 0%, #0f172a 100%); padding: 16px 20px; border-radius:12px; border-left: 6px solid #22c55e; margin-bottom: 24px;">
                <div style="font-size: 22px; font-weight:700; color: #f8fafc;">Price Action Story Engine</div>
                <div style="font-size: 13px; color: #94a3b8; margin-top:2px;">Advanced Lifecycle Analysis for: <b>{universe_name}</b> | Benchmark: {benchmark}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # RENDER TABS
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Market Story Dashboard",
            "🔄 Stage Rotation & Matrix",
            "📊 Universe Stage Map",
            "⚡ Options Calendar Candidate Funnel",
            "🧬 Ticker Story Detail",
            "🧪 Backtest Lab"
        ])

        # =================================================================
        # TAB 1: Market Story Dashboard
        # =================================================================
        with tab1:
            st.markdown("### Market Lifecycle Distribution")
            
            c1, c2, c3 = st.columns([1, 1, 2])
            
            # Metrics
            total_scanned = len(scanned_df)
            stage_counts = scanned_df["Stage"].value_counts()
            
            with c1:
                st.metric("Scanned Universe", total_scanned)
                # Benchmark status
                spy_b = benchmarks.get("SPY", {})
                if spy_b:
                    st.metric("SPY Score", f"{spy_b['Health Score']:.1f}/100", spy_b["Stage"])
            with c2:
                early_bulls = len(scanned_df[scanned_df["Stage"] == "Early Bull / Expansion"])
                st.metric("Early Bull Setup", early_bulls)
                
                qqq_b = benchmarks.get("QQQ", {})
                if qqq_b:
                    st.metric("QQQ Score", f"{qqq_b['Health Score']:.1f}/100", qqq_b["Stage"])
                    
            with c3:
                # Donut Chart
                stage_data = pd.DataFrame({
                    "Stage": list(STAGE_DETAILS.keys()),
                    "Count": [stage_counts.get(stage, 0) for stage in STAGE_DETAILS.keys()]
                })
                fig_donut = px.pie(
                    stage_data, 
                    values="Count", 
                    names="Stage", 
                    hole=0.4,
                    color="Stage",
                    color_discrete_map={stage: STAGE_DETAILS[stage]["color"] for stage in STAGE_DETAILS.keys()}
                )
                fig_donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc", family="Inter"),
                    margin=dict(l=0, r=0, t=10, b=10),
                    height=200,
                    showlegend=True
                )
                st.plotly_chart(fig_donut, use_container_width=True, key="market_donut")

            st.divider()
            
            # Benchmark scoring cards
            st.markdown("### Benchmark Index Scoring & Regime")
            bc1, bc2, bc3 = st.columns(3)
            
            def render_bench_card(col, name, data):
                if not data: return
                color = STAGE_DETAILS[data["Stage"]]["color"]
                with col:
                    st.markdown(
                        f"""
                        <div style="background:rgba(30,41,59,0.5); border:1px solid #334155; border-top:3.5px solid {color}; border-radius:10px; padding:15px; min-height:190px;">
                            <h4 style="color:#f8fafc; margin-top:0; margin-bottom:8px;">{name} ({data['Ticker']})</h4>
                            <div style="font-size:24px; font-weight:700; color:{color};">{data['Health Score']:.1f} <span style="font-size:12px; color:#94a3b8;">Health Score</span></div>
                            <div style="font-size:14px; font-weight:600; color:#e2e8f0; margin-top:5px;">Stage: {data['Stage']}</div>
                            <div style="font-size:12px; color:#94a3b8; margin-top:8px; line-height:1.4;">
                                VPR: {data['VPR']:.2f} | RSI: {data['RSI']:.1f} <br>
                                ADX: {data['ADX']:.1f} | Dist Days: {data['Distribution Days']}/20
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            render_bench_card(bc1, "S&P 500 Index", spy_b)
            render_bench_card(bc2, "Nasdaq 100 Index", qqq_b)
            render_bench_card(bc3, "Russell 2000 Index", benchmarks.get("IWM", {}))

            # Regime Analysis Card
            pct_above_50 = (scanned_df["Close"] > scanned_df["Ticker"].apply(lambda t: ticker_cache[t]["Close"].rolling(50).mean().iloc[-1] if t in ticker_cache else 0)).mean() * 100
            
            if pct_above_50 > 65:
                regime = "Risk-On (Uptrend Extension)"
                regime_color = "#22c55e"
                regime_desc = "Participation is high. The majority of universe assets reside above their medium-term moving averages. Favors deployment of aggressive calendar spreads and leverage."
            elif pct_above_50 > 40:
                regime = "Neutral (Rotation / Consolidation)"
                regime_color = "#eab308"
                regime_desc = "Selective rotation is under way. Indices are consolidating or churning. Stick to early accumulation setups and highly liquid options."
            else:
                regime = "Defensive (De-risking Mode)"
                regime_color = "#ef4444"
                regime_desc = "Breadth has broken down. Less than 40% of the universe is in an active uptrend. Avoid new option deployments. Retain cash."
                
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="background:rgba(21, 40, 71, 0.4); border: 1px solid #1e3a5f; border-left: 5px solid {regime_color}; border-radius:10px; padding:18px;">
                    <span style="color:{regime_color}; font-weight:700; text-transform:uppercase; font-size:12px; letter-spacing:1px;">Core Regime State</span>
                    <h3 style="color:#f8fafc; margin-top:4px; margin-bottom:8px;">{regime}</h3>
                    <p style="color:#cbd5e1; font-size:14px; margin: 0 0 8px 0; line-height:1.5;">{regime_desc}</p>
                    <span style="color:#94a3b8; font-size:12px;">Breadth Index (% Above 50 SMA): <b>{pct_above_50:.1f}%</b></span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # =================================================================
        # TAB 2: Stage Rotation & Matrix
        # =================================================================
        with tab2:
            st.markdown("### Lifecycle Stage Rotation Matrix")
            st.caption("Plots Health Score (0-100) vs. Volume Participation (VPR) to track transitions across the 6 price action lifecycle stages.")
            
            # Interactive Controls
            c1, c2 = st.columns([1, 1])
            with c1:
                show_trails = st.checkbox("Show Stage Rotation Trails (Past path history)", value=True, key="pa_show_rot_trails")
            with c2:
                all_tickers = sorted(list(scanned_df["Ticker"].values))
                highlight_ticker = st.selectbox("🎯 Highlight Ticker in Chart:", options=["None"] + all_tickers, key="pa_highlight_ticker")
            
            try:
                # Calculate trails for Health Score vs VPR
                stage_trails = {}
                for ticker in scanned_df["Ticker"].values:
                    ticker_df = ticker_cache.get(ticker)
                    if ticker_df is not None and len(ticker_df) >= 30:
                        x_vals = []
                        y_vals = []
                        if show_trails:
                            for offset in range(5, -1, -1):
                                idx = len(ticker_df) - 1 - offset
                                if idx >= 0:
                                    hist_slice = ticker_df.iloc[:idx + 1]
                                    hist_bench = bench_df.iloc[:idx + 1]
                                    res = evaluate_ticker_price_action(hist_slice, hist_bench)
                                    if res:
                                        x_vals.append(res["Health Score"])
                                        y_vals.append(res["VPR"])
                        else:
                            t_row = scanned_df[scanned_df["Ticker"] == ticker]
                            if not t_row.empty:
                                x_vals.append(t_row["Health Score"].values[0])
                                y_vals.append(t_row["VPR"].values[0])
                                
                        if x_vals:
                            stage_trails[ticker] = {
                                "x": x_vals,
                                "y": y_vals,
                                "name": scanned_df.loc[scanned_df["Ticker"] == ticker, "Name"].values[0],
                                "fdts": scanned_df.loc[scanned_df["Ticker"] == ticker, "FDTS"].values[0]
                            }
                
                if stage_trails:
                    fig = go.Figure()
                    
                    # Determine background rect fill opacity multiplier when highlighting is active
                    bg_opacity_mult = 0.4 if (highlight_ticker != "None") else 1.0
                    
                    # Vertical background rects for stage bands
                    fig.add_vrect(x0=0, x1=25, fillcolor=f"rgba(153,27,27,{0.05 * bg_opacity_mult})", line_width=0, layer="below")
                    fig.add_vrect(x0=25, x1=40, fillcolor=f"rgba(239,68,68,{0.05 * bg_opacity_mult})", line_width=0, layer="below")
                    fig.add_vrect(x0=40, x1=55, fillcolor=f"rgba(249,115,22,{0.05 * bg_opacity_mult})", line_width=0, layer="below")
                    fig.add_vrect(x0=55, x1=70, fillcolor=f"rgba(234,179,8,{0.05 * bg_opacity_mult})", line_width=0, layer="below")
                    fig.add_vrect(x0=70, x1=85, fillcolor=f"rgba(16,185,129,{0.05 * bg_opacity_mult})", line_width=0, layer="below")
                    fig.add_vrect(x0=85, x1=100, fillcolor=f"rgba(34,197,94,{0.05 * bg_opacity_mult})", line_width=0, layer="below")
                    
                    # Boundary lines
                    boundary_color = "#1e293b" if (highlight_ticker != "None") else "#334155"
                    for val in [25, 40, 55, 70, 85]:
                        fig.add_vline(x=val, line_color=boundary_color, line_width=1, line_dash="dash")
                        
                    # Shading stage names at the top of bands
                    ann_opacity = 0.3 if (highlight_ticker != "None") else 1.0
                    fig.add_annotation(x=12.5, y=1.05, text="Breakdown", showarrow=False, font=dict(color=f"rgba(153,27,27,{ann_opacity})", size=10, weight="bold"))
                    fig.add_annotation(x=32.5, y=1.05, text="Distribution", showarrow=False, font=dict(color=f"rgba(239,68,68,{ann_opacity})", size=10, weight="bold"))
                    fig.add_annotation(x=47.5, y=1.05, text="Fading Bull", showarrow=False, font=dict(color=f"rgba(249,115,22,{ann_opacity})", size=10, weight="bold"))
                    fig.add_annotation(x=62.5, y=1.05, text="Mature Bull", showarrow=False, font=dict(color=f"rgba(234,179,8,{ann_opacity})", size=10, weight="bold"))
                    fig.add_annotation(x=77.5, y=1.05, text="Strong Bull", showarrow=False, font=dict(color=f"rgba(16,185,129,{ann_opacity})", size=10, weight="bold"))
                    fig.add_annotation(x=92.5, y=1.05, text="Early Bull", showarrow=False, font=dict(color=f"rgba(34,197,94,{ann_opacity})", size=10, weight="bold"))
                    
                    color_idx = 0
                    for ticker, data in stage_trails.items():
                        x = data["x"]
                        y = data["y"]
                        latest_x, latest_y = x[-1], y[-1]
                        
                        color = COLORS[color_idx % len(COLORS)]
                        color_idx += 1
                        
                        is_highlighted = (highlight_ticker == ticker)
                        is_any_highlighted = (highlight_ticker != "None")
                        
                        # Determine dynamic opacity, thickness, and style parameters
                        if is_any_highlighted:
                            if is_highlighted:
                                line_opacity = 1.0
                                line_width = 3.5
                                line_dash = "solid"
                                marker_size_trail = 7
                                marker_size_head = 14
                                text_font_size = 13
                                text_font_color = color
                                head_opacity = 1.0
                                trace_hover = "all"
                            else:
                                line_opacity = 0.10
                                line_width = 1.0
                                line_dash = "dot"
                                marker_size_trail = 3
                                marker_size_head = 7
                                text_font_size = 8
                                text_font_color = "rgba(148,163,184,0.15)"
                                head_opacity = 0.10
                                trace_hover = "skip"
                        else:
                            line_opacity = 1.0
                            line_width = 1.5
                            line_dash = "dot"
                            marker_size_trail = 4
                            marker_size_head = 11
                            text_font_size = 11
                            text_font_color = color
                            head_opacity = 1.0
                            trace_hover = "all"
                            
                        # Plot dotted trailing path line (only if show_trails is enabled and we have multiple points)
                        if show_trails and len(x) > 1:
                            fig.add_trace(go.Scatter(
                                x=x, y=y,
                                mode="lines",
                                name=f"{data['name']} ({ticker})",
                                line=dict(color=color, width=line_width, dash=line_dash),
                                opacity=line_opacity,
                                hoverinfo='skip',
                                showlegend=False
                            ))
                            
                            # Plot trailing path points
                            fig.add_trace(go.Scatter(
                                x=x, y=y,
                                mode="markers",
                                marker=dict(size=marker_size_trail, color=color),
                                opacity=line_opacity * 0.75 if is_any_highlighted else 0.75,
                                hoverinfo='skip',
                                showlegend=False
                            ))
                        
                        # Head point (latest position)
                        fig.add_trace(go.Scatter(
                            x=[latest_x], y=[latest_y],
                            mode="markers+text",
                            name=f"{data['name']} ({ticker})",
                            text=[f"<b>{ticker}</b>"],
                            textposition="top right",
                            textfont=dict(color=text_font_color, size=text_font_size, family="Inter"),
                            marker=dict(size=marker_size_head, color=color, line=dict(color="black", width=1.5 if is_highlighted else 1)),
                            opacity=head_opacity,
                            hoverinfo=trace_hover,
                            hovertemplate=f"<b>{data['name']} ({ticker})</b><br>Health Score: %{{x:.1f}}<br>VPR: %{{y:.2f}}<br>FDTS: {data['fdts']}<extra></extra>"
                        ))
                        
                    fig.update_layout(
                        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                        font=dict(color="#f8fafc", family="Inter"),
                        xaxis=dict(title="Price Action Health Score", range=[100, 0], gridcolor="#1e293b"),
                        yaxis=dict(title="Volume Participation Ratio (VPR)", range=[0.2, 1.15], gridcolor="#1e293b"),
                        margin=dict(l=10, r=10, t=30, b=10),
                        height=500,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True, key="stage_rotational_matrix")
                else:
                    st.warning("No stage rotation data to plot.")
                    
                # Stage-based Matrix Status Summary
                st.markdown("### 📊 Stage Matrix Status Summary")
                
                # Group tickers by Stage
                stage_groups = {stage: [] for stage in STAGE_DETAILS.keys()}
                for _, r in scanned_df.iterrows():
                    ticker = r["Ticker"]
                    name = r["Name"]
                    stage = r["Stage"]
                    fdts = r["FDTS"]
                    
                    ticker_idx = list(scanned_df["Ticker"].values).index(ticker)
                    color = COLORS[ticker_idx % len(COLORS)]
                    
                    item = f"<span style='color:{color};font-weight:bold;'>■</span> <b>{ticker}</b> ({fdts}) <span style='color:#94a3b8;font-size:11px;'>({name})</span>"
                    if stage in stage_groups:
                        stage_groups[stage].append(item)
                        
                # Row 1
                c1, c2, c3 = st.columns(3)
                
                def render_stage_card(col, title, stage_key):
                    details = STAGE_DETAILS[stage_key]
                    color = details["color"]
                    items = stage_groups.get(stage_key, [])
                    with col:
                        st.markdown(
                            f"""
                            <div style="background:rgba(30,41,59,0.4); border-top:3.5px solid {color}; padding:15px; border-radius:10px; min-height:220px; border-left:1px solid #334155; border-right:1px solid #334155; border-bottom:1px solid #334155; margin-bottom:15px;">
                                <h5 style="color:{color}; margin-top:0; margin-bottom:6px; font-family:'Inter',sans-serif; text-transform:uppercase; font-size:12px; letter-spacing:0.5px;">{title}</h5>
                                <div style="font-size:11px; color:#94a3b8; line-height:1.3; margin-bottom:12px;">{details['desc']}</div>
                                <div style="color:#e2e8f0; font-size:13px; line-height:1.9;">
                                    {'<br>'.join(items) if items else '<i style="color:#64748b;">None</i>'}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                render_stage_card(c1, "Early Bull / Expansion", "Early Bull / Expansion")
                render_stage_card(c2, "Strong Bull", "Strong Bull")
                render_stage_card(c3, "Mature Bull", "Mature Bull")
                
                # Row 2
                c4, c5, c6 = st.columns(3)
                render_stage_card(c4, "Fading Bull", "Fading Bull")
                render_stage_card(c5, "Distribution", "Distribution")
                render_stage_card(c6, "Breakdown", "Breakdown")
                
            except Exception as e:
                st.error(f"Failed to render Stage Rotation Matrix: {e}")
                logger.error("Stage matrix rendering error", exc_info=True)

        # =================================================================
        # TAB 3: Universe Stage Map
        # =================================================================
        with tab3:
            st.markdown("### Scanned Universe Stage Map")
            
            # Interactive Filters
            fc1, fc2 = st.columns(2)
            with fc1:
                search_txt = st.text_input("Search Ticker:", key="universe_search").upper()
            with fc2:
                selected_stages = st.multiselect("Filter by Stage:", options=list(STAGE_DETAILS.keys()), default=None)
                
            # Filter Dataframe
            df_display = scanned_df.copy()
            if search_txt:
                df_display = df_display[df_display["Ticker"].str.contains(search_txt)]
            if selected_stages:
                df_display = df_display[df_display["Stage"].isin(selected_stages)]
                
            df_display = df_display.sort_values("Health Score", ascending=False)
            
            # Formatting Columns
            table_df = df_display[[
                "Ticker", "Health Score", "Stage", "FDTS", "Close", "VPR", 
                "RS Ratio", "RS Momentum", "RSI", "ADX", "ATR Percentile", 
                "Distribution Days", "Divergence"
            ]].copy()
            
            table_df["Health Score"] = table_df["Health Score"].map('{:.1f}'.format)
            table_df["VPR"] = table_df["VPR"].map('{:.2f}'.format)
            table_df["RS Ratio"] = table_df["RS Ratio"].map('{:.2f}'.format)
            table_df["RS Momentum"] = table_df["RS Momentum"].map('{:.2f}'.format)
            table_df["RSI"] = table_df["RSI"].map('{:.1f}'.format)
            table_df["ADX"] = table_df["ADX"].map('{:.1f}'.format)
            table_df["ATR Percentile"] = table_df["ATR Percentile"].map('{:.1f}%'.format)
            table_df["Divergence"] = table_df["Divergence"].apply(lambda d: "⚠️ Yes" if d else "No")
            
            try:
                styled_table_df = table_df.style.map(color_stage_cell, subset=["Stage"]).map(color_fdts_cell, subset=["FDTS"])
            except AttributeError:
                styled_table_df = table_df.style.applymap(color_stage_cell, subset=["Stage"]).applymap(color_fdts_cell, subset=["FDTS"])
                
            st.dataframe(styled_table_df, use_container_width=True, hide_index=True)

        # =================================================================
        # TAB 4: Option Calendar Candidate Funnel
        # =================================================================
        with tab4:
            st.markdown("### Calendar Spread candidates (Early Bull & Strong Bull)")
            st.caption("Filters for assets transitioning from Accumulation to Expansion where IV remains cheap and range expansion is imminent.")
            
            # Interactive Filters
            fc1, fc2 = st.columns(2)
            with fc1:
                search_txt_funnel = st.text_input("Search Ticker:", key="funnel_search").upper()
            with fc2:
                selected_actions = st.multiselect("Filter by Action:", options=["🟢 Deploy Calendar", "🟢 Deploy Calendar (Watch Spread)", "🟡 Watch (Earnings Risk)", "🔴 Avoid (Extended Vol)"], default=None, key="funnel_actions")
            
            # Filter candidates: Stage is Early Bull / Expansion or Strong Bull
            candidates = scanned_df[scanned_df["Stage"].isin(["Early Bull / Expansion", "Strong Bull"])].copy()
            if search_txt_funnel:
                candidates = candidates[candidates["Ticker"].str.contains(search_txt_funnel)]
            
            if candidates.empty:
                st.info("No candidates currently in the sweet spot (Early Bull or Strong Bull) matching your search criteria.")
            else:
                # Query option metrics & earnings
                candidate_tickers = candidates["Ticker"].tolist()
                opt_summary = query_options_liquidity_store(candidate_tickers)
                earnings_dates = query_earnings_calendar_store(candidate_tickers)
                
                rows = []
                for _, r in candidates.iterrows():
                    sym = r["Ticker"]
                    
                    # Option Metrics Check
                    opt_data = opt_summary.get(sym) or {}
                    
                    spread_pct = opt_data.get("median_spread_pct")
                    if spread_pct is None:
                        spread_pct = 2.5
                        
                    oi = opt_data.get("total_oi")
                    if oi is None:
                        oi = 0
                        
                    total_volume = opt_data.get("total_volume")
                    if total_volume is None:
                        total_volume = 0
                        
                    # Option Liquidity proxy
                    if opt_data:
                        liq = "High" if total_volume > 1000 else "Medium"
                    else:
                        # Proxy by stock volume
                        liq = "High" if r["Volume"] > 2000000 else "Medium" if r["Volume"] > 500000 else "Low"
                        
                    # Earnings Check
                    earn_date_str = earnings_dates.get(sym, "None")
                    days_to_earnings = 999
                    if earn_date_str != "None":
                        try:
                            earn_date = datetime.strptime(earn_date_str, "%Y-%m-%d")
                            days_to_earnings = (earn_date - datetime.today()).days
                        except Exception:
                            pass
                            
                    # Volatility percentile check
                    atr_pct = r["ATR Percentile"]
                    
                    # Recommendations Action
                    # Ideal: VPR >= 0.70, ATR percentile < 50%, spread < 2.0%, earnings > 15 days
                    if atr_pct < 50 and days_to_earnings > 15 and liq != "Low" and spread_pct <= 2.5:
                        action = "🟢 Deploy Calendar"
                        reason = "Sweet Spot: Low Vol, Wide Earnings Window, Liquid."
                    elif days_to_earnings <= 15:
                        action = "🟡 Watch (Earnings Risk)"
                        reason = f"Earnings date {earn_date_str} is within 15 days."
                    elif atr_pct > 70:
                        action = "🔴 Avoid (Extended Vol)"
                        reason = "ATR is too extended; IV is likely inflated."
                    else:
                        action = "🟢 Deploy Calendar (Watch Spread)"
                        reason = f"Good setup, but bid/ask spread ({spread_pct:.1f}%) is slightly wide."
                        
                    rows.append({
                        "Ticker": sym,
                        "Health Score": f"{r['Health Score']:.1f}",
                        "Stage": r["Stage"],
                        "FDTS": r["FDTS"],
                        "ATR Percentile": f"{atr_pct:.1f}%",
                        "Option Liquidity": liq,
                        "Spread %": f"{spread_pct:.1f}%",
                        "Earnings Date": earn_date_str,
                        "Action": action,
                        "Notes": reason
                    })
                    
                df_candidates = pd.DataFrame(rows)
                if selected_actions:
                    df_candidates = df_candidates[df_candidates["Action"].isin(selected_actions)]
                
                if df_candidates.empty:
                    st.info("No candidates matching the selected actions filters.")
                else:
                    try:
                        styled_candidates = df_candidates.style.map(color_stage_cell, subset=["Stage"]).map(color_fdts_cell, subset=["FDTS"])
                    except AttributeError:
                        styled_candidates = df_candidates.style.applymap(color_stage_cell, subset=["Stage"]).applymap(color_fdts_cell, subset=["FDTS"])
                    
                    st.dataframe(styled_candidates, use_container_width=True, hide_index=True)

        # =================================================================
        # TAB 5: Ticker Story Detail
        # =================================================================
        with tab5:
            st.markdown("### Ticker Deep-Dive & Price Action Lifecycle Chart")
            
            # Format dropdown choices with FDTS indicator
            ticker_fdts = dict(zip(scanned_df["Ticker"], scanned_df["FDTS"]))
            ticker_names = get_ticker_names(universe_name)
            
            def format_ticker_with_fdts(t):
                base_display = format_ticker_display(t, ticker_names)
                fdts_sig = ticker_fdts.get(t, "⚪ No Trade")
                return f"{base_display} ({fdts_sig})"
                
            selected_ticker = st.selectbox(
                "Select Ticker for Detailed Narrative:", 
                options=tickers, 
                index=0,
                format_func=format_ticker_with_fdts
            )
            
            # Reset visual canvas on ticker transition to clear stale visualizations immediately
            if "pa_last_selected_ticker" not in st.session_state:
                st.session_state["pa_last_selected_ticker"] = selected_ticker

            if selected_ticker != st.session_state["pa_last_selected_ticker"]:
                st.session_state["pa_last_selected_ticker"] = selected_ticker
                st.info(f"🔄 Loading detailed narrative and price action charts for {selected_ticker}...")
                st.rerun()
            
            ticker_df = ticker_cache.get(selected_ticker, pd.DataFrame())
            
            if ticker_df.empty:
                st.error("Historical data for selected ticker is missing.")
            else:
                # 1. Narrative Output
                t_score = scanned_df.loc[scanned_df["Ticker"] == selected_ticker, "Health Score"].values[0]
                t_stage = scanned_df.loc[scanned_df["Ticker"] == selected_ticker, "Stage"].values[0]
                t_vpr = scanned_df.loc[scanned_df["Ticker"] == selected_ticker, "VPR"].values[0]
                t_diverg = scanned_df.loc[scanned_df["Ticker"] == selected_ticker, "Divergence"].values[0]
                t_close = scanned_df.loc[scanned_df["Ticker"] == selected_ticker, "Close"].values[0]
                
                # Fetch Name details
                t_name = selected_ticker
                
                # Generate dynamic narrative text
                stage_color = STAGE_DETAILS[t_stage]["color"]
                stage_desc = STAGE_DETAILS[t_stage]["desc"]
                
                # Divergence warning text
                div_text = "Divergence warning detected! Price is rising on fading momentum; trend fragility is high." if t_diverg else "Momentum is aligning positively with the price trend."
                vpr_text = f"Volume participation is healthy (VPR: {t_vpr:.2f}), confirming institutional buying." if t_vpr >= 0.75 else f"Volume participation is fading (VPR: {t_vpr:.2f}), suggesting fewer buyers are chasing."
                
                if t_stage in ["Early Bull / Expansion", "Strong Bull"]:
                    act_rec = "DEPLOY candidate. Perfect for option calendar setups because realized volatility is compressed while price acceleration begins."
                elif t_stage == "Mature Bull":
                    act_rec = "HOLD existing. Volume and momentum are beginning to plateau; do not initiate new bullish calendar positions."
                elif t_stage == "Fading Bull":
                    act_rec = "MONITOR for exit. Trend is showing exhaustion. Risk of rollover is increasing."
                else:
                    act_rec = "AVOID/CUT. The stock is distributing or breaking down. Look for short candidates or sit in cash."
                    
                st.markdown(
                    f"""
                    <div style="background:rgba(30,41,59,0.5); border:1px solid #334155; border-left:6px solid {stage_color}; border-radius:10px; padding:20px; margin-bottom:20px;">
                        <h3 style="color:#f8fafc; margin-top:0; margin-bottom:8px;">{t_name} ({selected_ticker}) — {t_stage}</h3>
                        <div style="font-size:24px; font-weight:700; color:{stage_color};">{t_score:.1f} <span style="font-size:12px; color:#94a3b8;">Health Score</span></div>
                        <p style="color:#cbd5e1; font-size:14.5px; margin: 12px 0; line-height:1.5;">
                            <b>State Details:</b> {stage_desc}<br>
                            <b>Volume Health:</b> {vpr_text}<br>
                            <b>Momentum Alert:</b> {div_text}
                        </p>
                        <hr style="border-color:#334155; margin:12px 0;">
                        <div style="color:#94a3b8; font-size:13px; font-weight:600;">RECOMMENDED ACTION:</div>
                        <div style="color:#f8fafc; font-size:14px; font-weight:700; margin-top:4px;">{act_rec}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # 2. Price Action Lifecycle Chart (Candlestick + Shaded Background Zones)
                # Compute historical stage for each date to draw background shading!
                chart_df = ticker_df.tail(120).copy() # last 120 sessions for visual quality
                
                # Fetch database records if available, otherwise compute historical stages on-the-fly
                db_hist = get_historical_stages(selected_ticker)
                
                stages_by_date = {}
                if not db_hist.empty:
                    db_hist["trade_date"] = pd.to_datetime(db_hist["trade_date"])
                    for _, row in db_hist.iterrows():
                        stages_by_date[row["trade_date"].date()] = row["stage"]
                
                # Fill missing dates using on-the-fly computation
                for d in chart_df.index:
                    dt = d.date()
                    if dt not in stages_by_date:
                        hist_slice = ticker_df.loc[:d].tail(100)
                        if len(hist_slice) >= 30:
                            slice_bench = bench_df.loc[:d].tail(100)
                            calc = evaluate_ticker_price_action(hist_slice, slice_bench)
                            if calc:
                                stages_by_date[dt] = calc["Stage"]
                            else:
                                stages_by_date[dt] = "Breakdown"
                        else:
                            stages_by_date[dt] = "Breakdown"

                # Standard Candlestick chart
                fig_cand = go.Figure()
                fig_cand.add_trace(go.Candlestick(
                    x=chart_df.index,
                    open=chart_df["Open"],
                    high=chart_df["High"],
                    low=chart_df["Low"],
                    close=chart_df["Close"],
                    name="Candlestick",
                    increasing_line_color="#22c55e",
                    decreasing_line_color="#ef4444"
                ))
                
                # Moving Averages
                ma20 = ticker_df["Close"].rolling(20).mean().reindex(chart_df.index)
                ma50 = ticker_df["Close"].rolling(50).mean().reindex(chart_df.index)
                ma200 = ticker_df["Close"].rolling(200).mean().reindex(chart_df.index)
                
                fig_cand.add_trace(go.Scatter(x=chart_df.index, y=ma20, mode="lines", name="20 SMA", line=dict(color="#3b82f6", width=1.5)))
                fig_cand.add_trace(go.Scatter(x=chart_df.index, y=ma50, mode="lines", name="50 SMA", line=dict(color="#f59e0b", width=1.5)))
                fig_cand.add_trace(go.Scatter(x=chart_df.index, y=ma200, mode="lines", name="200 SMA", line=dict(color="#6366f1", width=1.5)))
                
                # Add background shading for each date segment
                dates = chart_df.index
                i = 0
                while i < len(dates):
                    current_stage = stages_by_date.get(dates[i].date(), "Breakdown")
                    bg_color = STAGE_DETAILS.get(current_stage, {"bg": "rgba(0,0,0,0)"})["bg"]
                    
                    # Find how far this block goes
                    start_idx = i
                    while i < len(dates) and stages_by_date.get(dates[i].date(), "Breakdown") == current_stage:
                        i += 1
                    end_idx = min(i, len(dates) - 1)
                    
                    fig_cand.add_vrect(
                        x0=dates[start_idx].strftime("%Y-%m-%d"),
                        x1=dates[end_idx].strftime("%Y-%m-%d"),
                        fillcolor=bg_color,
                        opacity=1.0,
                        layer="below",
                        line_width=0
                    )
                
                fig_cand.update_layout(
                    title=f"Price Action Lifecycle — Shaded Stage Zones",
                    paper_bgcolor="#0f172a",
                    plot_bgcolor="#0f172a",
                    font=dict(color="#f8fafc", family="Inter"),
                    xaxis=dict(gridcolor="#1e293b", rangeslider=dict(visible=False)),
                    yaxis=dict(gridcolor="#1e293b"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=500
                )
                st.plotly_chart(fig_cand, use_container_width=True, key="detail_candlestick")
                
                # Indicator Subplots: Volume/VPR & RSI/MACD & RS & CVD & ADX
                sub1, sub2 = st.columns(2)
                
                with sub1:
                    # 1. Volume Participation Chart
                    fig_vpr = go.Figure()
                    vpr_series = ticker_df["Volume"].rolling(20).mean() / ticker_df["Volume"].rolling(20).mean().rolling(60).max()
                    fig_vpr.add_trace(go.Scatter(x=chart_df.index, y=vpr_series.reindex(chart_df.index), mode="lines", name="VPR", line=dict(color="#10b981", width=2)))
                    fig_vpr.add_hline(y=0.75, line_dash="dash", line_color="#ef4444", annotation_text="Threshold (0.75)")
                    fig_vpr.update_layout(
                        title="Volume Participation Ratio (VPR)",
                        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                        font=dict(color="#f8fafc"), xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b", range=[0.3, 1.1]),
                        height=250, margin=dict(l=10, r=10, t=35, b=10)
                    )
                    st.plotly_chart(fig_vpr, use_container_width=True, key="detail_vpr")
                    
                    # 2. Relative Strength Chart
                    fig_rs = go.Figure()
                    ticker_rs_ratio, _ = calculate_rrg_values(ticker_df["Close"], bench_df["Close"].reindex(ticker_df.index).ffill())
                    fig_rs.add_trace(go.Scatter(x=chart_df.index, y=ticker_rs_ratio.reindex(chart_df.index), mode="lines", name="RS Ratio", line=dict(color="#6366f1", width=2)))
                    fig_rs.add_hline(y=100, line_color="#94a3b8")
                    fig_rs.update_layout(
                        title="Relative Strength (RS Ratio) vs Benchmark",
                        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                        font=dict(color="#f8fafc"), xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                        height=250, margin=dict(l=10, r=10, t=35, b=10)
                    )
                    st.plotly_chart(fig_rs, use_container_width=True, key="detail_rs")
                    
                with sub2:
                    # 3. Momentum Chart
                    fig_rsi = go.Figure()
                    hist_rsi = calculate_rsi(ticker_df["Close"]).reindex(chart_df.index)
                    fig_rsi.add_trace(go.Scatter(x=chart_df.index, y=hist_rsi, mode="lines", name="RSI (14)", line=dict(color="#f59e0b", width=2)))
                    fig_rsi.add_hline(y=70, line_color="#ef4444", line_dash="dot")
                    fig_rsi.add_hline(y=30, line_color="#22c55e", line_dash="dot")
                    fig_rsi.update_layout(
                        title="RSI Momentum",
                        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                        font=dict(color="#f8fafc"), xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b", range=[10, 90]),
                        height=250, margin=dict(l=10, r=10, t=35, b=10)
                    )
                    st.plotly_chart(fig_rsi, use_container_width=True, key="detail_rsi_plot")
                    
                    # 4. CVD Divergence Chart
                    fig_cvd = go.Figure()
                    diff_close = ticker_df["Close"].diff()
                    cvd_full = (np.where(diff_close >= 0, 1, -1) * ticker_df["Volume"]).cumsum()
                    fig_cvd.add_trace(go.Scatter(x=chart_df.index, y=cvd_full.reindex(chart_df.index), mode="lines", name="CVD Proxy", line=dict(color="#06b6d4", width=2)))
                    fig_cvd.update_layout(
                        title="Cumulative Volume Delta (CVD) Proxy",
                        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                        font=dict(color="#f8fafc"), xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                        height=250, margin=dict(l=10, r=10, t=35, b=10)
                    )
                    st.plotly_chart(fig_cvd, use_container_width=True, key="detail_cvd")

        # =================================================================
        # TAB 6: Backtest Lab
        # =================================================================
        with tab6:
            st.markdown("### Price Action Stage Performance Backtest")
            st.caption("Review historical forward returns for each stage over the last 60 sessions to confirm options/trade entry edge.")
            
            # Run simulation on scanned tickers
            with st.spinner("Simulating historical stage performance..."):
                backtest_results = []
                
                # Check cache length
                for ticker, t_df in ticker_cache.items():
                    if len(t_df) < 100: continue
                    t_bench = bench_df.reindex(t_df.index).ffill()
                    
                    # Loop over past 40 trading days (dates where we have at least 20 days of future returns)
                    lookback_backtest = min(40, len(t_df) - 60)
                    for offset in range(1, lookback_backtest + 1):
                        idx = -20 - offset
                        date = t_df.index[idx]
                        
                        # Evaluate stage at that historical date
                        hist_slice = t_df.iloc[:idx + 1]
                        hist_bench = t_bench.iloc[:idx + 1]
                        eval_res = evaluate_ticker_price_action(hist_slice, hist_bench)
                        
                        if eval_res:
                            # Forward returns
                            close_then = t_df["Close"].iloc[idx]
                            
                            ret20 = (t_df["Close"].iloc[idx + 20] / close_then - 1) * 100
                            # Map stage
                            backtest_results.append({
                                "Stage": eval_res["Stage"],
                                "Ret20": ret20
                            })
                            
                if backtest_results:
                    bt_df = pd.DataFrame(backtest_results)
                    summary_rows = []
                    
                    for stage in STAGE_DETAILS.keys():
                        stage_slice = bt_df[bt_df["Stage"] == stage]
                        if stage_slice.empty:
                            summary_rows.append({
                                "Stage": stage,
                                "Sample Count": 0,
                                "Average 20D Return": "0.00%",
                                "Win Rate (>0)": "0.0%"
                            })
                        else:
                            avg_ret = stage_slice["Ret20"].mean()
                            win_rate = (stage_slice["Ret20"] > 0).mean() * 100
                            summary_rows.append({
                                "Stage": stage,
                                "Sample Count": len(stage_slice),
                                "Average 20D Return": f"{avg_ret:+.2f}%",
                                "Win Rate (>0)": f"{win_rate:.1f}%"
                            })
                            
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
                else:
                    st.warning("Not enough historical data points to perform the backtest simulation.")
