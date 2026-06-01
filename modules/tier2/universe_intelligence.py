"""
FazDane Analytics — Tier 2
Universe Intelligence System & Executive Dashboard
"""

import logging
import os
import json
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.cluster.vq import kmeans, vq

from modules.base_module import FazDaneModule
from utils.universe_manager import render_universe_manager, get_universe_names, get_ticker_names, format_ticker_display
from utils.portfolio_performance_store import get_latest_portfolio_positions, get_latest_portfolio_details

logger = logging.getLogger("UniverseIntelligence")

# Ideal Color System
STAGE_COLORS = {
    "Early Accumulation": "#38bdf8",     # Blue
    "Expansion / Leadership": "#22c55e", # Green
    "Late Stage / Exhaustion": "#f97316", # Orange
    "Deterioration / Distribution": "#ef4444", # Red
    "Cash / Other": "#64748b"            # Slate Grey
}

REGIME_COLORS = {
    "Risk-On": "#22c55e",
    "Neutral": "#facc15",
    "Defensive": "#f97316",
    "High Volatility": "#ef4444"
}

# Sector lists
SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Cons Discretionary",
    "XLP": "Cons Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication",
}

# --- Utility to resolve Option symbols to their Underlying ---
def get_underlying_ticker(ticker: str) -> str:
    ticker = str(ticker).strip().upper()
    if not ticker or ticker in ["CASH", "USD", "MMDA12", "MMDA"] or "CASH" in ticker:
        return "CASH"
    # Match standard option prefixes (e.g. AAPL  240621C00180000) or letters
    match = re.match(r"^([A-Z]+)", ticker)
    if match:
        val = match.group(1)
        if val in ["CASH", "USD", "MMDA"]:
            return "CASH"
        return val
    return "CASH"

# --- Caching Data Fetches ---

from modules.calendar_scoring.technical_indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_fdts_ha_signal,
    compute_rrg_ratio_sma as compute_rrg_metrics
)

def calculate_fdts_signal(symbol: str, ticker_df: pd.DataFrame, period: int = 20) -> str:
    """Calculate the FDTS + MACD Trade Signal (Buy/No Trade/Sell)."""
    return calculate_fdts_ha_signal(ticker_df, period)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_prices(tickers: tuple[str, ...], lookback_days: int = 365) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Fetch price, volume and info for tickers in a single batch to reduce yfinance hits."""
    if not tickers:
        return pd.DataFrame(), pd.DataFrame(), {}, {}
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days + 150) # fetch extra for indicators buffer
    
    ticker_list = sorted(list(set(tickers)))
    try:
        data = yf.download(
            ticker_list,
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column"
        )
    except Exception as exc:
        logger.error(f"yfinance download failed for universe: {exc}")
        return pd.DataFrame(), pd.DataFrame(), {}, {}

    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, {}

    # Extract Close and Volume
    if isinstance(data.columns, pd.MultiIndex):
        close_df = data["Close"].copy() if "Close" in data else data["Adj Close"].copy()
        volume_df = data["Volume"].copy() if "Volume" in data else pd.DataFrame()
    else:
        close_df = data[["Close"]].copy() if "Close" in data else data[["Adj Close"]].copy()
        close_df.columns = [ticker_list[0]]
        volume_df = data[["Volume"]].copy() if "Volume" in data else pd.DataFrame()
        volume_df.columns = [ticker_list[0]]

    close_df.index = pd.to_datetime(close_df.index).tz_localize(None)
    close_df = close_df.ffill().bfill()
    
    if not volume_df.empty:
        volume_df.index = pd.to_datetime(volume_df.index).tz_localize(None)
        volume_df = volume_df.ffill().fillna(0)

    # Compute FDTS + MACD Signals
    fdts_signals_map = {}
    for ticker in ticker_list:
        try:
            ticker_df = pd.DataFrame(index=data.index)
            if isinstance(data.columns, pd.MultiIndex):
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col in data and ticker in data[col].columns:
                        ticker_df[col] = data[col][ticker]
            else:
                ticker_df = data.copy()
                if "Adj Close" in ticker_df.columns and "Close" not in ticker_df.columns:
                    ticker_df["Close"] = ticker_df["Adj Close"]
            
            fdts_signals_map[ticker] = calculate_fdts_signal(ticker, ticker_df)
        except Exception as e:
            logger.warning(f"FDTS calculation failed for {ticker}: {e}")
            fdts_signals_map[ticker] = "No Trade"

    # Fetch basic company info (sector, market cap)
    info_map = {}
    for ticker in ticker_list:
        info_map[ticker] = {
            "sector": "Other/ETF" if ticker.startswith("^") or ticker in ["SPY", "QQQ", "IWM", "GLD", "TLT"] else None,
            "market_cap": 1e9 # default
        }
        
    return close_df, volume_df, info_map, fdts_signals_map

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_info_details(ticker: str) -> dict:
    """Fetch additional details for a specific ticker to populate Drilldown Level 1."""
    try:
        tick = yf.Ticker(ticker)
        info = tick.get_info()
        return {
            "name": info.get("shortName") or info.get("longName") or ticker,
            "sector": info.get("sector") or "Other/ETF",
            "industry": info.get("industry") or "N/A",
            "market_cap": info.get("marketCap") or 1e9,
            "beta": info.get("beta") or 1.0,
            "description": info.get("longBusinessSummary") or "No business summary available."
        }
    except Exception:
        return {
            "name": ticker,
            "sector": "Other/ETF",
            "industry": "N/A",
            "market_cap": 1e9,
            "beta": 1.0,
            "description": "Information unavailable."
        }

def calculate_historical_beta(asset_series: pd.Series, index_series: pd.Series) -> float:
    """Calculate historical beta of an asset relative to an index using daily returns."""
    aligned = pd.concat([asset_series, index_series], axis=1).dropna()
    if len(aligned) < 30:
        return 1.0
    returns = aligned.pct_change().dropna()
    if returns.empty:
        return 1.0
    asset_ret = returns.iloc[:, 0]
    index_ret = returns.iloc[:, 1]
    cov = asset_ret.cov(index_ret)
    var = index_ret.var()
    if var == 0 or pd.isna(var) or pd.isna(cov):
        return 1.0
    return float(cov / var)

# --- Core Calculations ---

def classify_stage(rs: float, mom: float) -> str:
    """Classify based on RS-Ratio and RS-Momentum relative to 100."""
    if rs >= 100 and mom >= 100:
        return "Expansion / Leadership"
    elif rs >= 100 and mom < 100:
        return "Late Stage / Exhaustion"
    elif rs < 100 and mom < 100:
        return "Deterioration / Distribution"
    else:
        return "Early Accumulation"

def compute_technical_indicators(prices: pd.Series, volumes: pd.Series) -> dict:
    """Calculate standard metrics: RSI, MACD, ADX, CVD, ATR, expected move."""
    if len(prices) < 30:
        return {"rsi": 50, "macd_line": 0, "macd_signal": 0, "adx": 20, "cvd": 0, "atr": 1.0, "bollinger_sqz": False}
        
    # RSI (14) using core
    rsi_s = calculate_rsi(prices)
    rsi = rsi_s.iloc[-1]
    
    # MACD using core
    macd_line, signal_line, _ = calculate_macd(prices)
    
    # CVD (Cumulative Volume Delta) proxy
    delta = prices.diff()
    cvd_series = (np.where(delta >= 0, 1, -1) * volumes).cumsum()
    cvd_val = cvd_series.iloc[-1] if len(cvd_series) else 0

    # ATR (14) proxy
    high_low = prices.rolling(2).max() - prices.rolling(2).min()
    atr = high_low.rolling(14).mean().iloc[-1]

    # Bollinger Bands & Squeeze Indicator
    sma20 = prices.rolling(20).mean()
    std20 = prices.rolling(20).std()
    bandwidth = (4 * std20) / sma20
    is_sqz = bandwidth.iloc[-1] < bandwidth.rolling(120).mean().iloc[-1] * 0.8 if len(bandwidth) > 120 else False

    return {
        "rsi": rsi,
        "macd_line": macd_line.iloc[-1],
        "macd_signal": signal_line.iloc[-1],
        "cvd": cvd_val,
        "atr": atr,
        "bollinger_sqz": is_sqz
    }

class UniverseIntelligenceModule(FazDaneModule):
    MODULE_NAME = "Universe Intelligence System"
    MODULE_ICON = "🪐"
    MODULE_DESCRIPTION = "Ticker Universe Executive Dashboard with Relative Rotation Graphs, Multi-Timeframe Alignment, Capital Allocation Models, and Portfolio Risk Drilldowns."
    TIER = 2
    SOURCE_NOTEBOOK = "FazDane Universe Intelligence Dashboard"
    CACHE_TTL = 3600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance", "schwab_portfolio_sqlite"]

    def render_sidebar(self):
        st.markdown("**Intelligence Scope**")
        self.universe_name, self.tickers, self.benchmark = render_universe_manager(
            key_prefix="ui",
            show_benchmark=True,
            label="Target Universe:"
        )
        
        # Ensure benchmark is in the ticker download list
        self.tickers = list(self.tickers)
        if self.benchmark not in self.tickers:
            self.tickers.append(self.benchmark)
            
        st.caption(f"Loaded {len(self.tickers)} tickers (including benchmark {self.benchmark}).")

        st.markdown("**Chart Customisation**")
        self.lookback_days = st.slider("Historical Lookback (Days):", 120, 500, 252, step=20, key="ui_lookback")
        self.trail_periods = st.slider("Rotation Trail Steps (Periods):", 2, 20, 5, key="ui_trail_periods")
        self.trail_interval = st.selectbox("Rotation Step Scale:", ["Daily", "Weekly"], index=1, key="ui_trail_scale")

        st.markdown("**Option Calculations**")
        self.greek_pref = st.selectbox("Greeks Normalisation Type:", ["Both (Split View)", "Raw Dollar-Equivalent", "Normalized % of Portfolio"], index=0, key="ui_greek_pref")
        self.bubble_size_pref = st.selectbox("Default Bubble Sizing:", ["Both (Split / Toggle)", "Portfolio Weight", "Market Cap"], index=0, key="ui_bubble_size_pref")

        st.markdown("**Beta Calculations**")
        self.beta_index = st.selectbox("Beta Reference Index:", ["SPY", "QQQ"], index=0, key="ui_beta_index")

        if st.button("🔄 Refresh Market Data", use_container_width=True, type="primary"):
            fetch_historical_prices.clear()
            st.rerun()

    def render_main(self):
        # 0. Load Data
        if not self.tickers or len(self.tickers) <= 1:
            st.warning("⚠️ Please configure a ticker universe in the sidebar with at least one symbol.")
            return

        with st.spinner("Analyzing ticker universe and portfolio allocations..."):
            # Load Portfolio positions from Portfolio Performance Store
            portfolio_df, portfolio_meta = get_latest_portfolio_positions()
            portfolio_details, detail_meta = get_latest_portfolio_details()

        # Prepare portfolio mapping and clean tickers
        portfolio_weights = {}
        portfolio_greeks = {}
        has_portfolio = False
        net_liq = 0.0
        portfolio_tickers = []
        clean_portfolio_rows = []

        if not portfolio_df.empty:
            has_portfolio = True
            net_liq = float(portfolio_df["market_value"].sum())
            if net_liq == 0:
                net_liq = 1.0
            
            for _, row in portfolio_df.iterrows():
                raw_ticker = str(row["ticker"]).upper()
                underlying = get_underlying_ticker(raw_ticker)
                market_val = float(row["market_value"])
                
                clean_portfolio_rows.append({
                    "raw_ticker": raw_ticker,
                    "underlying": underlying,
                    "market_value": market_val,
                    "delta": float(row.get("delta", 0.0)),
                    "gamma": float(row.get("gamma", 0.0)),
                    "theta": float(row.get("theta", 0.0)),
                    "vega": float(row.get("vega", 0.0))
                })
                
                if underlying != "CASH" and underlying not in portfolio_tickers:
                    portfolio_tickers.append(underlying)

        # Merge target universe and portfolio underlying stocks for batch yfinance downloading
        all_download_tickers = list(self.tickers)
        # Ensure SPY and QQQ are in all_download_tickers for beta weighting calculations
        for index_ticker in ["SPY", "QQQ"]:
            if index_ticker not in all_download_tickers:
                all_download_tickers.append(index_ticker)
        for pt in portfolio_tickers:
            if pt not in all_download_tickers:
                all_download_tickers.append(pt)

        with st.spinner("Fetching yfinance price history for universe and portfolio positions..."):
            close_df, volume_df, info_map, fdts_signals = fetch_historical_prices(tuple(all_download_tickers), self.lookback_days)

        if close_df.empty:
            st.error("❌ Failed to retrieve price history for the selected universe.")
            return

        # Prepare weights mapping for the selected universe only
        for tick in self.tickers:
            # Map from portfolio
            tick_up = tick.upper()
            matching_vals = sum(r["market_value"] for r in clean_portfolio_rows if r["underlying"] == tick_up)
            portfolio_weights[tick_up] = matching_vals / net_liq if net_liq > 0 else 0.0

        # Calculate RRG Rotational coordinates
        rs_df, mom_df = compute_rrg_metrics(close_df, self.benchmark)
        if rs_df.empty or mom_df.empty:
            st.error("❌ Relative Rotation analysis failed. Ensure the benchmark ticker has active history.")
            return

        latest_date = rs_df.index[-1]
        
        # Build master dataframe of metrics (for universe display)
        ticker_list = [t for t in self.tickers if t in rs_df.columns and t != self.benchmark]
        if not ticker_list:
            st.warning("⚠️ Selected universe contains only the benchmark index. Please select a broader list.")
            return

        rows = []
        for ticker in ticker_list:
            curr_rs = float(rs_df[ticker].iloc[-1])
            curr_mom = float(mom_df[ticker].iloc[-1])
            stage = classify_stage(curr_rs, curr_mom)
            weight = portfolio_weights.get(ticker.upper(), 0.0)

            # Get sector and market cap from cached info or default
            meta = fetch_info_details(ticker)
            
            # Simple technical calculations
            tech = compute_technical_indicators(close_df[ticker], volume_df[ticker] if ticker in volume_df.columns else pd.Series(0, index=close_df.index))

            fdts_sig = fdts_signals.get(ticker, "No Trade")

            # Calculate historical beta relative to selected SPY or QQQ index
            ref_index = self.beta_index
            if ref_index in close_df.columns and ticker in close_df.columns:
                calculated_beta = calculate_historical_beta(close_df[ticker], close_df[ref_index])
            else:
                calculated_beta = meta.get("beta") or 1.0

            rows.append({
                "Ticker": ticker,
                "Name": meta["name"],
                "Sector": meta["sector"],
                "RS-Ratio": curr_rs,
                "RS-Momentum": curr_mom,
                "Stage": stage,
                "Weight": weight,
                "Market Cap (B)": meta["market_cap"] / 1e9,
                "RSI": tech["rsi"],
                "Bollinger Squeeze": "Squeeze" if tech["bollinger_sqz"] else "Normal",
                "ATR": tech["atr"],
                "CVD": tech["cvd"],
                "Beta": calculated_beta,
                "FDTS Signal": fdts_sig
            })
            
        data_df = pd.DataFrame(rows)

        # 1. TOP HEADER (Macro Regime & Universe Health) - Kept globally visible
        st.markdown(
            f"""
            <div style="background:linear-gradient(90deg, #152847 0%, #0d1b2e 100%); padding: 12px 20px; border-radius:10px; border-left: 5px solid #3ab54a; margin-bottom: 20px;">
                <div style="font-size: 20px; font-weight:700; color: #3ab54a;">FazDane Executive Universe Intelligence System</div>
                <div style="font-size: 13px; color: #94a3b8; margin-top:2px;">Analytic State Snapshot for: <b>{self.universe_name}</b> | Data Date: {latest_date.strftime('%d %b %Y')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Evaluate Regime & Breadth
        pct_above_50 = float((close_df[ticker_list].iloc[-1] > close_df[ticker_list].rolling(50).mean().iloc[-1]).mean() * 100)
        pct_above_200 = float((close_df[ticker_list].iloc[-1] > close_df[ticker_list].rolling(200).mean().iloc[-1]).mean() * 100)
        pct_above_20 = float((close_df[ticker_list].iloc[-1] > close_df[ticker_list].rolling(20).mean().iloc[-1]).mean() * 100)

        # Bench trend
        bench_close = close_df[self.benchmark]
        bench_ma50 = bench_close.rolling(50).mean().iloc[-1]
        bench_ma200 = bench_close.rolling(200).mean().iloc[-1]
        bench_val = bench_close.iloc[-1]
        bench_trend = "Strong Bull" if (bench_val > bench_ma50 and bench_ma50 > bench_ma200) else "Bullish" if bench_val > bench_ma200 else "Bearish"

        # Mock/Retrieve Volatility Proxy
        vix_val = float(close_df["^VIX"].iloc[-1]) if "^VIX" in close_df.columns else 16.5
        
        # Decide overall status
        if vix_val > 24.0:
            regime = "High Volatility"
        elif bench_trend == "Bearish":
            regime = "Defensive"
        elif pct_above_50 > 60:
            regime = "Risk-On"
        else:
            regime = "Neutral"

        # Universe Health Score
        health_score = int(0.3 * pct_above_50 + 0.3 * pct_above_20 + 0.2 * (80 if bench_trend == "Strong Bull" else 60 if bench_trend == "Bullish" else 30) + 0.2 * max(0, 100 - (vix_val - 12) * 4))
        health_score = max(0, min(100, health_score))

        # Render Macro metrics row
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Macro Regime", regime, f"VIX: {vix_val:.2f}", delta_color="inverse")
        with c2:
            st.metric("Universe Health", f"{health_score}/100", f"Regime: {regime}")
        with c3:
            st.metric(f"{self.benchmark} Trend", bench_trend, f"Price: {bench_val:.2f}")
        with c4:
            st.metric("Universe Breadth (>50MA)", f"{pct_above_50:.1f}%", f"{pct_above_200:.1f}% above 200MA")
        with c5:
            # Stage Candidates Count
            early_cnt = len(data_df[data_df["Stage"] == "Early Accumulation"])
            exp_cnt = len(data_df[data_df["Stage"] == "Expansion / Leadership"])
            st.metric("Expansion / Early Candidates", f"{exp_cnt} / {early_cnt}", f"Total Universe: {len(data_df)}")

        # AI Generated Insights
        narrative = self.generate_ai_insight(regime, health_score, data_df, pct_above_50)
        st.markdown(
            f"""
            <div style="background:rgba(21, 40, 71, 0.4); border: 1px solid #1e3a5f; border-radius:8px; padding:12px 18px; margin-bottom:24px;">
                <span style="color:#38bdf8; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:1px;">Advanced AI Insight Engine</span>
                <p style="color:#e2e8f0; font-size:13.5px; margin: 6px 0 0 0; line-height:1.5;">{narrative}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------- TABS SETUP -----------------
        tab_list = [
            "🎯 Executive battlefield Map",
            "📈 Performance Sorter",
            "📊 Capital Allocation Engine",
            "⚡ Internals & Leadership",
            "🛡️ Risk & Volatility Structure",
            "🧬 Cluster Analysis",
            "🔍 Interactive Drill-Downs"
        ]
        
        tab_battlefield, tab_performance, tab_allocation, tab_internals, tab_risk, tab_cluster, tab_drilldowns = st.tabs(tab_list)

        # --- TAB 1: EXECUTIVE BATTLEFIELD MAP ---
        with tab_battlefield:
            self.render_section_header("1. Master Quadrant Model (Centerpiece)", "Rotational battlefield map of relative strength and momentum metrics.")

            # Interactive controls for Bubble chart
            sc1, sc2 = st.columns(2)
            with sc1:
                bubble_size_mode = st.radio(
                    "Bubble Sizing Basis:",
                    ["Portfolio Weight", "Market Cap"],
                    index=0 if self.bubble_size_pref in ["Both (Split / Toggle)", "Portfolio Weight"] else 1,
                    horizontal=True,
                    key="bubble_size_select"
                )
            with sc2:
                show_trails = st.checkbox("Show Rotation Trails (Past path history)", value=True, key="show_rot_trails")

            # Scatter plot construction
            fig_quad = go.Figure()

            # Calculate limits dynamically
            min_x = max(70, min(data_df["RS-Ratio"].min() - 2, 95))
            max_x = min(130, max(data_df["RS-Ratio"].max() + 2, 105))
            min_y = max(70, min(data_df["RS-Momentum"].min() - 2, 95))
            max_y = min(130, max(data_df["RS-Momentum"].max() + 2, 105))

            # Shading regions
            fig_quad.add_shape(type="rect", x0=min_x, x1=100, y0=100, y1=max_y, fillcolor="rgba(56,189,248,0.06)", line_width=0) # Blue
            fig_quad.add_shape(type="rect", x0=100, x1=max_x, y0=100, y1=max_y, fillcolor="rgba(34,197,94,0.06)", line_width=0)  # Green
            fig_quad.add_shape(type="rect", x0=100, x1=max_x, y0=min_y, y1=100, fillcolor="rgba(249,115,22,0.06)", line_width=0) # Orange
            fig_quad.add_shape(type="rect", x0=min_x, x1=100, y0=min_y, y1=100, fillcolor="rgba(239,68,68,0.06)", line_width=0)   # Red

            # Axis crosshairs
            fig_quad.add_hline(y=100, line_dash="dash", line_color="#475569", line_width=1.5)
            fig_quad.add_vline(x=100, line_dash="dash", line_color="#475569", line_width=1.5)

            # Plot current bubbles
            for stage, color in STAGE_COLORS.items():
                stage_df = data_df[data_df["Stage"] == stage]
                if stage_df.empty:
                    continue

                sizes = []
                if bubble_size_mode == "Portfolio Weight":
                    sizes = [max(12, int(w * 100) + 12) if w > 0 else 10 for w in stage_df["Weight"]]
                    hover_text = stage_df.apply(lambda r: f"<b>{r['Ticker']}</b> ({r['Name']})<br>Weight: {r['Weight']*100:.2f}%<br>RS: {r['RS-Ratio']:.2f}<br>Mom: {r['RS-Momentum']:.2f}<br>FDTS Signal: {'🟢 Buy' if r['FDTS Signal'] == 'Buy' else '🔴 Sell' if r['FDTS Signal'] == 'Sell' else '⚪ No Trade'}", axis=1)
                else:
                    sizes = [max(10, int(np.sqrt(mc)) * 3) for mc in stage_df["Market Cap (B)"]]
                    hover_text = stage_df.apply(lambda r: f"<b>{r['Ticker']}</b> ({r['Name']})<br>Mkt Cap: ${r['Market Cap (B)']:.1f}B<br>RS: {r['RS-Ratio']:.2f}<br>Mom: {r['RS-Momentum']:.2f}<br>FDTS Signal: {'🟢 Buy' if r['FDTS Signal'] == 'Buy' else '🔴 Sell' if r['FDTS Signal'] == 'Sell' else '⚪ No Trade'}", axis=1)

                fig_quad.add_trace(go.Scatter(
                    x=stage_df["RS-Ratio"],
                    y=stage_df["RS-Momentum"],
                    mode="markers+text",
                    name=stage,
                    text=stage_df["Ticker"],
                    textposition="top center",
                    hoverinfo="text",
                    hovertext=hover_text,
                    marker=dict(
                        size=sizes,
                        color=color,
                        line=dict(width=1.5, color="#0d1b2e"),
                        opacity=0.85
                    )
                ))

            # Add Trails if requested
            if show_trails:
                trail_step = 5 if self.trail_interval == "Weekly" else 1
                periods = self.trail_periods
                for ticker in ticker_list:
                    if len(rs_df) < (periods * trail_step):
                        continue
                    xs = []
                    ys = []
                    for p in range(periods, -1, -1):
                        idx = -(p * trail_step) - 1
                        if idx >= -len(rs_df):
                            xs.append(rs_df[ticker].iloc[idx])
                            ys.append(mom_df[ticker].iloc[idx])
                    
                    fig_quad.add_trace(go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color="#475569", width=1, dash="solid"),
                        hoverinfo="none",
                        showlegend=False,
                        opacity=0.35
                    ))
                    fig_quad.add_trace(go.Scatter(
                        x=[xs[0]],
                        y=[ys[0]],
                        mode="markers",
                        marker=dict(size=4, color="#475569"),
                        hoverinfo="none",
                        showlegend=False,
                        opacity=0.4
                    ))

            # Quadrant Titles in corners
            fig_quad.add_annotation(x=min_x + 2, y=max_y - 2, text="<b>EARLY ACCUMULATION (Q1)</b><br>Improving Momentum", showarrow=False, font=dict(color="#38bdf8", size=10), align="left")
            fig_quad.add_annotation(x=max_x - 2, y=max_y - 2, text="<b>EXPANSION LEADERS (Q2)</b><br>Alpha Leadership", showarrow=False, font=dict(color="#22c55e", size=10), align="right")
            fig_quad.add_annotation(x=max_x - 2, y=min_y + 2, text="<b>LATE STAGE / EXHAUSTION (Q3)</b><br>Trim & Profit Take", showarrow=False, font=dict(color="#f97316", size=10), align="right")
            fig_quad.add_annotation(x=min_x + 2, y=min_y + 2, text="<b>DETERIORATION (Q4)</b><br>Capital Preservation", showarrow=False, font=dict(color="#ef4444", size=10), align="left")

            fig_quad.update_layout(
                xaxis=dict(title="Relative Strength vs Benchmark ->", range=[min_x, max_x], gridcolor="rgba(148,163,184,0.08)"),
                yaxis=dict(title="Momentum Acceleration ->", range=[min_y, max_y], gridcolor="rgba(148,163,184,0.08)"),
                paper_bgcolor="#0d1b2e",
                plot_bgcolor="rgba(21, 40, 71, 0.2)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(13,27,46,0.6)"),
                margin=dict(l=20, r=20, t=40, b=20),
                height=580,
                font=dict(color="#e2e8f0", family="Inter")
            )
            st.plotly_chart(fig_quad, use_container_width=True, key="universe_rot_map")

            st.divider()
            
            # Stage transitions (Rotation Engine)
            st.markdown("### Stage Transition Log")
            trans_rows = []
            past_step = 5 if self.trail_interval == "Weekly" else 1
            if len(rs_df) > past_step:
                for ticker in ticker_list:
                    p_rs = float(rs_df[ticker].iloc[-past_step - 1])
                    p_mom = float(mom_df[ticker].iloc[-past_step - 1])
                    past_stage = classify_stage(p_rs, p_mom)
                    
                    curr_stage = data_df.loc[data_df["Ticker"] == ticker, "Stage"].values[0]
                    if past_stage != curr_stage:
                        curr_fdts = data_df.loc[data_df["Ticker"] == ticker, "FDTS Signal"].values[0]
                        fd_emoji = {"Buy": "🟢 Buy", "Sell": "🔴 Sell", "No Trade": "⚪ No Trade"}.get(curr_fdts, "⚪ No Trade")
                        trans_rows.append({
                            "Ticker": ticker,
                            "FDTS Signal": fd_emoji,
                            "Prior State": past_stage,
                            "Current State": curr_stage,
                            "RS Change": f"{data_df.loc[data_df['Ticker'] == ticker, 'RS-Ratio'].values[0] - p_rs:+.2f}",
                            "Mom Change": f"{data_df.loc[data_df['Ticker'] == ticker, 'RS-Momentum'].values[0] - p_mom:+.2f}"
                        })
            
            if trans_rows:
                st.dataframe(
                    pd.DataFrame(trans_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "FDTS Signal": st.column_config.TextColumn("FDTS Signal", width="small")
                    }
                )
            else:
                st.info("No tickers transitioned stages over the selected trail step period.")

        # --- TAB: UNIVERSE PERFORMANCE SORTER ---
        with tab_performance:
            self.render_section_header("2. Universe Performance Sorter", "Track relative performance in % across Daily, Weekly, Monthly, Yearly, YTD, or Custom intervals.")
            
            # Timeframe selector
            tf_choice = st.radio(
                "Select Performance Timeframe to Rank:",
                ["Daily (1D)", "Weekly (1W)", "Monthly (1M)", "Yearly (1Y)", "YTD", "Custom Period 🎯"],
                index=1,
                horizontal=True,
                key="ui_perf_timeframe"
            )
            
            # Show custom inputs if Custom Period is chosen
            custom_days = 15
            custom_label = "Custom"
            if tf_choice == "Custom Period 🎯":
                cc1, cc2 = st.columns(2)
                with cc1:
                    custom_unit = st.selectbox("Custom Unit:", ["Trading Days", "Weeks", "Months", "Years"], index=0, key="ui_perf_custom_unit")
                with cc2:
                    if custom_unit == "Trading Days":
                        custom_days = st.slider("Lookback Days:", min_value=1, max_value=252, value=15, step=1, key="ui_perf_custom_days_val")
                        custom_label = f"Custom ({custom_days}D) %"
                    elif custom_unit == "Weeks":
                        custom_weeks = st.slider("Lookback Weeks:", min_value=1, max_value=52, value=3, step=1, key="ui_perf_custom_weeks_val")
                        custom_days = custom_weeks * 5
                        custom_label = f"Custom ({custom_weeks}W) %"
                    elif custom_unit == "Months":
                        custom_months = st.slider("Lookback Months:", min_value=1, max_value=24, value=1, step=1, key="ui_perf_custom_months_val")
                        custom_days = custom_months * 21
                        custom_label = f"Custom ({custom_months}M) %"
                    else:
                        custom_years = st.slider("Lookback Years:", min_value=1, max_value=5, value=1, step=1, key="ui_perf_custom_years_val")
                        custom_days = custom_years * 252
                        custom_label = f"Custom ({custom_years}Y) %"
                
                if custom_days > self.lookback_days:
                    st.info(f"💡 Note: Your selected lookback of {custom_days} trading days exceeds the historical lookback configuration in the sidebar ({self.lookback_days} days). Calculated returns will fallback to inception/maximum available history.")
            else:
                custom_label = "Custom (15D) %"

            # Compute performance metrics for all tickers
            perf_rows = []
            for ticker in ticker_list:
                price_series = close_df[ticker].dropna()
                n = len(price_series)
                
                ret_1d = (price_series.iloc[-1] / price_series.iloc[-2] - 1) * 100 if n >= 2 else 0.0
                ret_1w = (price_series.iloc[-1] / price_series.iloc[-6] - 1) * 100 if n >= 6 else 0.0
                ret_1m = (price_series.iloc[-1] / price_series.iloc[-22] - 1) * 100 if n >= 22 else 0.0
                
                if n >= 253:
                    ret_1y = (price_series.iloc[-1] / price_series.iloc[-253] - 1) * 100
                elif n >= 2:
                    ret_1y = (price_series.iloc[-1] / price_series.iloc[0] - 1) * 100
                else:
                    ret_1y = 0.0
                
                # YTD calculation
                curr_year = price_series.index[-1].year
                prev_year_prices = price_series[price_series.index.year < curr_year]
                if not prev_year_prices.empty:
                    ret_ytd = (price_series.iloc[-1] / prev_year_prices.iloc[-1] - 1) * 100
                elif len(price_series) >= 2:
                    ret_ytd = (price_series.iloc[-1] / price_series.iloc[0] - 1) * 100
                else:
                    ret_ytd = 0.0

                # Custom period calculation
                idx_back = -1 - custom_days
                if n >= abs(idx_back):
                    ret_custom = (price_series.iloc[-1] / price_series.iloc[idx_back] - 1) * 100
                elif n >= 2:
                    ret_custom = (price_series.iloc[-1] / price_series.iloc[0] - 1) * 100
                else:
                    ret_custom = 0.0
                    
                # Get FDTS signal and map to colored emoji
                fd_val = data_df.loc[data_df["Ticker"] == ticker, "FDTS Signal"].values[0]
                fd_emoji = {"Buy": "🟢 Buy", "Sell": "🔴 Sell", "No Trade": "⚪ No Trade"}.get(fd_val, "⚪ No Trade")
                
                perf_rows.append({
                    "Ticker": ticker,
                    "Name": fetch_info_details(ticker)["name"],
                    "FDTS Signal": fd_emoji,
                    "Daily (1D) %": round(ret_1d, 2),
                    "Weekly (1W) %": round(ret_1w, 2),
                    "Monthly (1M) %": round(ret_1m, 2),
                    "Yearly (1Y) %": round(ret_1y, 2),
                    "YTD %": round(ret_ytd, 2),
                    "Custom %": round(ret_custom, 2)
                })
            
            perf_df = pd.DataFrame(perf_rows)
            
            tf_col = "Custom %" if tf_choice == "Custom Period 🎯" else {
                "Daily (1D)": "Daily (1D) %",
                "Weekly (1W)": "Weekly (1W) %",
                "Monthly (1M)": "Monthly (1M) %",
                "Yearly (1Y)": "Yearly (1Y) %",
                "YTD": "YTD %"
            }[tf_choice]
            
            # Sort the data frame
            sorted_df = perf_df.sort_values(by=tf_col, ascending=False)
            
            # Key statistics for this timeframe
            if not sorted_df.empty:
                stat_c1, stat_c2, stat_c3 = st.columns(3)
                best_tick = sorted_df.iloc[0]["Ticker"]
                best_val = sorted_df.iloc[0][tf_col]
                worst_tick = sorted_df.iloc[-1]["Ticker"]
                worst_val = sorted_df.iloc[-1][tf_col]
                
                pos_count = len(sorted_df[sorted_df[tf_col] > 0])
                total_count = len(sorted_df)
                
                with stat_c1:
                    st.metric("🏆 Best Performer", f"{best_tick}", f"{best_val:+.2f}%")
                with stat_c2:
                    st.metric("⚠️ Underperformer", f"{worst_tick}", f"{worst_val:+.2f}%")
                with stat_c3:
                    st.metric("📈 Advancing Breadth", f"{pos_count} / {total_count}", f"{(pos_count/total_count)*100:.1f}% Positive")
            
            # Chart
            colors = ["#22c55e" if val >= 0 else "#ef4444" for val in sorted_df[tf_col]]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=sorted_df["Ticker"],
                y=sorted_df[tf_col],
                marker_color=colors,
                text=[f"{val:+.2f}%" for val in sorted_df[tf_col]],
                textposition="outside",
                cliponaxis=False
            ))
            
            chart_title = f"Relative Performance Rankings ({custom_label if tf_choice == 'Custom Period 🎯' else tf_choice})"
            fig_bar.update_layout(
                title=chart_title,
                xaxis_title="Ticker",
                yaxis_title="Return %",
                paper_bgcolor="#0d1b2e",
                plot_bgcolor="rgba(21, 40, 71, 0.2)",
                font=dict(color="#e2e8f0"),
                margin=dict(l=40, r=40, t=65, b=45),
                height=450
            )
            fig_bar.add_hline(y=0.0, line_width=1.5, line_color="#475569")
            
            st.plotly_chart(fig_bar, use_container_width=True, key=f"perf_plotly_{tf_col.lower().replace(' ', '_').replace('%', '').strip()}")
            
            # Full table
            st.markdown("### 📋 Universe Performance Leaderboard")
            st.dataframe(
                perf_df.sort_values(by=tf_col, ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Name": st.column_config.TextColumn("Name", width="medium"),
                    "FDTS Signal": st.column_config.TextColumn("FDTS Signal", width="small"),
                    "Daily (1D) %": st.column_config.NumberColumn("Daily (1D) %", format="%+.2f%%"),
                    "Weekly (1W) %": st.column_config.NumberColumn("Weekly (1W) %", format="%+.2f%%"),
                    "Monthly (1M) %": st.column_config.NumberColumn("Monthly (1M) %", format="%+.2f%%"),
                    "Yearly (1Y) %": st.column_config.NumberColumn("Yearly (1Y) %", format="%+.2f%%"),
                    "YTD %": st.column_config.NumberColumn("YTD %", format="%+.2f%%"),
                    "Custom %": st.column_config.NumberColumn(custom_label, format="%+.2f%%")
                }
            )

        # --- TAB 3: CAPITAL ALLOCATION ENGINE ---
        with tab_allocation:
            st.markdown("### Recommended vs Actual Capital Allocations")
            
            # Determine recommended weights based on Regime
            if regime == "Risk-On":
                rec_alloc = {"Early Accumulation": 40.0, "Expansion / Leadership": 50.0, "Late Stage / Exhaustion": 10.0, "Deterioration / Distribution": 0.0, "Cash / Other": 0.0}
            elif regime == "Defensive":
                rec_alloc = {"Early Accumulation": 20.0, "Expansion / Leadership": 30.0, "Late Stage / Exhaustion": 20.0, "Deterioration / Distribution": 10.0, "Cash / Other": 20.0}
            elif regime == "High Volatility":
                rec_alloc = {"Early Accumulation": 20.0, "Expansion / Leadership": 30.0, "Late Stage / Exhaustion": 20.0, "Deterioration / Distribution": 10.0, "Cash / Other": 20.0}
            else: # Neutral
                rec_alloc = {"Early Accumulation": 30.0, "Expansion / Leadership": 50.0, "Late Stage / Exhaustion": 15.0, "Deterioration / Distribution": 5.0, "Cash / Other": 0.0}

            # Calculate actual weights across ALL positions from Portfolio Performance Store
            actual_alloc = {
                "Early Accumulation": 0.0,
                "Expansion / Leadership": 0.0,
                "Late Stage / Exhaustion": 0.0,
                "Deterioration / Distribution": 0.0,
                "Cash / Other": 0.0
            }
            
            if has_portfolio:
                for crow in clean_portfolio_rows:
                    under = crow["underlying"]
                    val_pct = (crow["market_value"] / net_liq) * 100.0
                    
                    if under == "CASH":
                        actual_alloc["Cash / Other"] += val_pct
                    elif under in rs_df.columns:
                        # Dynamic classification based on its RRG coordinates
                        curr_rs = float(rs_df[under].iloc[-1])
                        curr_mom = float(mom_df[under].iloc[-1])
                        stage = classify_stage(curr_rs, curr_mom)
                        actual_alloc[stage] += val_pct
                    else:
                        actual_alloc["Cash / Other"] += val_pct
            else:
                # Mock default if no portfolio data exists
                actual_alloc = {"Early Accumulation": 0.0, "Expansion / Leadership": 0.0, "Late Stage / Exhaustion": 0.0, "Deterioration / Distribution": 0.0, "Cash / Other": 100.0}

            ac1, ac2 = st.columns([1, 1.2])
            with ac1:
                st.markdown(f"**Model Breakdown ({regime} Regime)**")
                alloc_rows = []
                for stage in rec_alloc.keys():
                    alloc_rows.append({
                        "Stage": stage,
                        "Model %": f"{rec_alloc[stage]:.1f}%",
                        "Actual Portfolio %": f"{actual_alloc[stage]:.1f}%",
                        "Variance %": f"{actual_alloc[stage] - rec_alloc[stage]:+.1f}%"
                    })
                st.dataframe(pd.DataFrame(alloc_rows), use_container_width=True, hide_index=True)
                st.caption("💡 Recommended weights are adjusted dynamically by the system according to macro regime shifts.")
                
            with ac2:
                fig_alloc = go.Figure()
                stages = list(rec_alloc.keys())
                colors = [STAGE_COLORS[s] for s in stages]
                
                fig_alloc.add_trace(go.Pie(
                    labels=stages,
                    values=[rec_alloc[s] for s in stages],
                    domain=dict(x=[0, 0.45]),
                    name="Recommended",
                    hole=0.4,
                    marker=dict(colors=colors)
                ))
                fig_alloc.add_trace(go.Pie(
                    labels=stages,
                    values=[actual_alloc[s] for s in stages],
                    domain=dict(x=[0.55, 1]),
                    name="Actual Portfolio",
                    hole=0.4,
                    marker=dict(colors=colors)
                ))
                fig_alloc.update_layout(
                    annotations=[
                        dict(text="Model", x=0.20, y=0.5, font_size=12, showarrow=False, font_color="#cbd5e1"),
                        dict(text="Actual", x=0.80, y=0.5, font_size=12, showarrow=False, font_color="#cbd5e1")
                    ],
                    paper_bgcolor="#0d1b2e",
                    plot_bgcolor="#0d1b2e",
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
                    margin=dict(l=10, r=10, t=20, b=20),
                    height=280,
                    font=dict(color="#e2e8f0")
                )
                st.plotly_chart(fig_alloc, use_container_width=True, key="alloc_comparison_pie")

        # --- TAB 3: INTERNALS & LEADERSHIP ---
        with tab_internals:
            ic1, ic2 = st.columns([1.5, 1])
            with ic1:
                st.markdown("### Actionable Leadership & Opportunity Table")
                
                lead_df = data_df[["Ticker", "Stage", "FDTS Signal", "RS-Ratio", "RS-Momentum", "RSI", "Bollinger Squeeze", "Beta"]].copy()
                
                # Format signal with color emojis
                lead_df["FDTS Signal"] = lead_df["FDTS Signal"].map({
                    "Buy": "🟢 Buy",
                    "Sell": "🔴 Sell",
                    "No Trade": "⚪ No Trade"
                }).fillna("⚪ No Trade")
                
                def get_opp(row):
                    if row["Stage"] == "Expansion / Leadership":
                        return "Hold winner / Add on pullbacks"
                    elif row["Stage"] == "Early Accumulation":
                        return "Buy / Position calendars & LEAPS"
                    elif row["Stage"] == "Late Stage / Exhaustion":
                        return "Trim / Write covered calls"
                    else:
                        return "Avoid / Exit / Short candidates"
                        
                lead_df["Opportunity Action"] = lead_df.apply(get_opp, axis=1)
                lead_df["RS Rank"] = lead_df["RS-Ratio"].rank(pct=True).map(lambda r: int(r * 100))
                
                st.dataframe(
                    lead_df.sort_values("RS-Ratio", ascending=False).round(2),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "RS Rank": st.column_config.ProgressColumn("RS Rank", format="%d", min_value=0, max_value=100),
                        "FDTS Signal": st.column_config.TextColumn("FDTS Signal", width="small"),
                        "Beta": st.column_config.NumberColumn(f"Beta vs {self.beta_index}", format="%.2f")
                    }
                )
                
            with ic2:
                st.markdown("### Market Internals & Breadth Profiles")
                st.markdown(
                    f"""
                    <div style="background:rgba(21,40,71,0.25); border:1px solid #1e3a5f; border-radius:8px; padding:15px;">
                        <div style="font-size:12px; text-transform:uppercase; color:#94a3b8; letter-spacing:1px; margin-bottom:8px;">Universe Breadth Profile</div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                            <span>Stocks above 20-day SMA</span>
                            <b style="color:#22c55e;">{pct_above_20:.1f}%</b>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                            <span>Stocks above 50-day SMA</span>
                            <b style="color:#facc15;">{pct_above_50:.1f}%</b>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                            <span>Stocks above 200-day SMA</span>
                            <b style="color:#38bdf8;">{pct_above_200:.1f}%</b>
                        </div>
                        <hr style="border-top:1px solid #1e3a5f; margin:10px 0;">
                        <div style="font-size:12px; text-transform:uppercase; color:#94a3b8; letter-spacing:1px; margin-bottom:8px;">Universe Velocity & Alignment</div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                            <span>Rotation Velocity (Weekly)</span>
                            <b style="color:#e2e8f0;">{(data_df['RS-Momentum'].abs().mean() / 100):.2f} rad/wk</b>
                        </div>
                        <div style="display:flex; justify-content:space-between;">
                            <span>Benchmark Trend Alignment</span>
                            <b style="color:#3ab54a;">{bench_trend}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            st.divider()
            st.markdown("### Multi-Timeframe Alignment Conviction Matrix")
            align_rows = []
            for ticker in ticker_list:
                prices = close_df[ticker]
                d_trend = "Bull" if prices.iloc[-1] > prices.rolling(20).mean().iloc[-1] else "Weak"
                w_trend = "Bull" if prices.iloc[-1] > prices.rolling(50).mean().iloc[-1] else "Neutral" if prices.iloc[-1] > prices.rolling(100).mean().iloc[-1] else "Weak"
                m_trend = "Bull" if prices.iloc[-1] > prices.rolling(200).mean().iloc[-1] else "Weak"
                q_trend = "Bull" if prices.rolling(50).mean().iloc[-1] > prices.rolling(200).mean().iloc[-1] else "Weak"
                
                conv_score = sum([2 if d=="Bull" else 0 for d in [d_trend, w_trend, m_trend, q_trend]])
                conv_label = "Highest" if conv_score >= 7 else "High" if conv_score >= 5 else "Neutral" if conv_score >= 3 else "Low"

                # Get FDTS signal and map to colored emoji
                fd_val = data_df.loc[data_df["Ticker"] == ticker, "FDTS Signal"].values[0]
                fd_emoji = {"Buy": "🟢 Buy", "Sell": "🔴 Sell", "No Trade": "⚪ No Trade"}.get(fd_val, "⚪ No Trade")

                align_rows.append({
                    "Ticker": ticker,
                    "FDTS Signal": fd_emoji,
                    "Daily (20MA)": d_trend,
                    "Weekly (50MA)": w_trend,
                    "Monthly (200MA)": m_trend,
                    "Quarterly (SMA Cross)": q_trend,
                    "Conviction Status": conv_label
                })
            st.dataframe(pd.DataFrame(align_rows), use_container_width=True, hide_index=True)

        # --- TAB 4: RISK & VOLATILITY STRUCTURE ---
        with tab_risk:
            rc1, rc2 = st.columns([1, 1.2])
            with rc1:
                st.markdown("**Sector & Theme Exposure Concentration**")
                sector_counts = data_df.groupby("Sector")["Weight"].sum() * 100.0
                if sector_counts.sum() == 0:
                    sector_counts = data_df["Sector"].value_counts(normalize=True) * 100.0
                    
                fig_sec = px.bar(
                    x=sector_counts.values,
                    y=sector_counts.index,
                    orientation="h",
                    labels={"x": "Portfolio Value / Capital %", "y": "Sector"},
                    color_discrete_sequence=["#1a3a8f"]
                )
                fig_sec.update_layout(
                    paper_bgcolor="#0d1b2e",
                    plot_bgcolor="rgba(21, 40, 71, 0.2)",
                    font=dict(color="#e2e8f0"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=240
                )
                st.plotly_chart(fig_sec, use_container_width=True, key="sector_exposure_bar")

            with rc2:
                st.markdown("**Option Portfolio Risk (Greeks)**")
                if not has_portfolio or not clean_portfolio_rows:
                    st.info("No active options portfolio uploaded. Register a Schwab Position statement to load options Greeks.")
                else:
                    total_delta = sum(g["delta"] for g in clean_portfolio_rows)
                    total_gamma = sum(g["gamma"] for g in clean_portfolio_rows)
                    total_theta = sum(g["theta"] for g in clean_portfolio_rows)
                    total_vega = sum(g["vega"] for g in clean_portfolio_rows)
                    
                    gc1, gc2 = st.columns(2)
                    with gc1:
                        st.markdown("**Raw Dollar-Equivalent Greeks**")
                        st.write(f"• **Delta Exposure:** `${total_delta * 100:,.2f}` equivalent")
                        st.write(f"• **Gamma Concentration:** `{total_gamma:,.2f}` delta/pt")
                        st.write(f"• **Theta Decay:** `${total_theta:,.2f}` / day")
                        st.write(f"• **Vega Volatility Risk:** `${total_vega:,.2f}` / 1% vol")
                    with gc2:
                        st.markdown("**Normalized Portfolio % Greeks**")
                        st.write(f"• **Delta % of Capital:** `{((total_delta * 100) / net_liq) * 100:.2f}%`")
                        st.write(f"• **Gamma Impact (1% move):** `{((total_gamma * 0.01 * 100) / net_liq) * 100:.3f}%` cap shift")
                        st.write(f"• **Theta Yield (Daily):** `{(total_theta / net_liq) * 100:.3f}%` yield")
                        st.write(f"• **Vega Impact (1% volatility):** `{(total_vega / net_liq) * 100:.3f}%` drop")

            st.divider()
            
            # Correlation clustering
            st.markdown("**Hierarchical Correlation Clustering**")
            returns_df = close_df[ticker_list].pct_change().dropna(how="all")
            if len(returns_df) > 10:
                corr = returns_df.corr().fillna(0.0)
                if len(corr) >= 3:
                    try:
                        dists = pdist(corr.values)
                        link = linkage(dists, method="complete")
                        ordered_idx = leaves_list(link)
                        ordered_tickers = [corr.columns[i] for i in ordered_idx]
                        corr = corr.loc[ordered_tickers, ordered_tickers]
                    except Exception:
                        pass
                
                # Visual Adjustment Panel
                with st.expander("🛠️ Heatmap Sizing & Layout Controls", expanded=False):
                    hc1, hc2, hc3 = st.columns(3)
                    with hc1:
                        # Auto-calculate default height based on ticker count to ensure nice spacing
                        calc_default = max(800, len(corr.columns) * 35)
                        heat_height = st.slider("Heatmap Height (px)", min_value=400, max_value=2500, value=int(calc_default), step=50, key="corr_heat_height")
                    with hc2:
                        # Auto-calculate label size
                        calc_font = max(6, min(14, int(400 / max(1, len(corr.columns)))))
                        label_size = st.slider("Tick/Label Font Size (px)", min_value=5, max_value=24, value=int(calc_font), step=1, key="corr_label_size")
                    with hc3:
                        show_vals = st.checkbox("Show Correlation Values inside cells", value=(len(corr.columns) <= 25), key="corr_show_vals")
                        # Add a slider for cell text font size if values shown
                        if show_vals:
                            val_size = st.slider("Cell Text Font Size (px)", min_value=5, max_value=20, value=max(5, label_size - 2), step=1, key="corr_val_size")
                        else:
                            val_size = 8

                fig_heat = px.imshow(
                    corr,
                    x=corr.columns,
                    y=corr.index,
                    color_continuous_scale="RdBu",
                    zmin=-1,
                    zmax=1,
                    text_auto=".2f" if show_vals else False,
                    title="Hierarchical Correlation Sorter (Avoid Overcrowding Risk)"
                )
                fig_heat.update_layout(
                    paper_bgcolor="#0d1b2e",
                    plot_bgcolor="#0d1b2e",
                    font=dict(color="#e2e8f0"),
                    margin=dict(l=60, r=40, t=60, b=60),
                    height=heat_height
                )
                fig_heat.update_xaxes(tickfont=dict(size=label_size))
                fig_heat.update_yaxes(tickfont=dict(size=label_size))
                if show_vals:
                    fig_heat.update_annotations(font=dict(size=val_size))

                st.plotly_chart(fig_heat, use_container_width=True, key="hier_corr_heat")
            else:
                st.caption("Not enough return history to calculate correlation clustering.")

            st.divider()
            st.markdown("### Implied Volatility & Options Squeezes")
            vol_df = data_df[["Ticker", "FDTS Signal", "RSI", "Bollinger Squeeze", "ATR", "Beta"]].copy()
            vol_df["FDTS Signal"] = vol_df["FDTS Signal"].map({
                "Buy": "🟢 Buy",
                "Sell": "🔴 Sell",
                "No Trade": "⚪ No Trade"
            }).fillna("⚪ No Trade")
            vol_df["IV Rank (Sim)"] = vol_df["RSI"].map(lambda r: int(abs(r - 50) * 2))
            vol_df["Expected 5D Move %"] = (vol_df["Beta"] * 2.5).round(2)
            st.dataframe(
                vol_df.sort_values("IV Rank (Sim)", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "FDTS Signal": st.column_config.TextColumn("FDTS Signal", width="small"),
                    "Beta": st.column_config.NumberColumn(f"Beta vs {self.beta_index}", format="%.2f")
                }
            )

        # --- TAB 5: CLUSTER ANALYSIS ---
        with tab_cluster:
            self.render_section_header("5. Dimensionality Reduction & Theme Clustering", "Dynamically group universe assets using Principal Component Analysis and K-Means clustering to uncover hidden thematic drivers.")
            
            if len(ticker_list) < 3:
                st.warning("⚠️ Cluster Analysis requires at least 3 tickers in the active universe. Please select a larger universe in the sidebar.")
            else:
                # 0. Show Universe-wide KPIs
                univ_count = len(ticker_list)
                univ_beta = data_df["Beta"].mean() if "Beta" in data_df.columns else 1.0
                univ_rsi = data_df["RSI"].mean() if "RSI" in data_df.columns else 50.0
                
                buys_univ = len(data_df[data_df["FDTS Signal"] == "Buy"]) if "FDTS Signal" in data_df.columns else 0
                sells_univ = len(data_df[data_df["FDTS Signal"] == "Sell"]) if "FDTS Signal" in data_df.columns else 0
                no_trades_univ = len(data_df[data_df["FDTS Signal"] == "No Trade"]) if "FDTS Signal" in data_df.columns else 0
                
                kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                with kpi_col1:
                    st.metric("Total Universe Assets", f"{univ_count}")
                with kpi_col2:
                    st.metric("Universe Average Beta", f"{univ_beta:.2f}")
                with kpi_col3:
                    st.metric("Universe Average RSI", f"{univ_rsi:.1f}")
                with kpi_col4:
                    st.metric("Universe FDTS Signals", f"🟢 {buys_univ} | ⚪ {no_trades_univ} | 🔴 {sells_univ}")
                
                st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

                # 1. Math PCA engine
                # Gather daily returns for tickers in the universe
                returns_df = close_df[ticker_list].pct_change().fillna(0.0)
                
                # Standardize returns (z-score)
                std = returns_df.std()
                std = std.replace(0.0, 1.0).fillna(1.0)
                standardized_returns = (returns_df - returns_df.mean()) / std
                
                # Calculate correlation matrix
                corr_matrix = standardized_returns.corr().fillna(0.0)
                
                # Eigenvalue decomposition
                pca_success = False
                try:
                    eigenvals, eigenvectors = np.linalg.eigh(corr_matrix.values)
                    
                    # eigenvalues and eigenvectors are sorted in ascending order.
                    # The two largest eigenvalues are at the end.
                    pc1_val = max(0.0, eigenvals[-1])
                    pc2_val = max(0.0, eigenvals[-2]) if len(eigenvals) >= 2 else 0.0
                    
                    x_coords = eigenvectors[:, -1] * np.sqrt(pc1_val)
                    y_coords = eigenvectors[:, -2] * np.sqrt(pc2_val) if len(eigenvals) >= 2 else np.zeros_like(x_coords)
                    pca_success = True
                except Exception as ex:
                    logger.error(f"Failed to calculate PCA: {ex}")
                    st.error(f"Failed to calculate PCA components: {ex}")
                    x_coords = np.zeros(len(ticker_list))
                    y_coords = np.zeros(len(ticker_list))
                    corr_matrix = pd.DataFrame(np.eye(len(ticker_list)), index=ticker_list, columns=ticker_list)

                if pca_success:
                    # 2. UI Control Sliders
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        k_val = st.slider("Number of Clusters (K):", min_value=2, max_value=6, value=4, step=1, key="ui_cluster_k_val")
                    with cc2:
                        threshold_val = st.slider("Similarity Edge Threshold (Correlation):", min_value=0.3, max_value=0.9, value=0.5, step=0.05, key="ui_cluster_threshold_val")
                        
                    k_adjusted = min(k_val, len(ticker_list))
                    
                    # 3. K-Means clustering
                    features = np.column_stack((x_coords, y_coords)).astype(np.float64)
                    cluster_ids = np.zeros(len(ticker_list), dtype=int)
                    try:
                        centroids, _ = kmeans(features, k_adjusted)
                        cluster_ids, _ = vq(features, centroids)
                    except Exception as e:
                        logger.warning(f"K-Means failed, defaulting all to cluster 0: {e}")

                    # 4. Define theme helper and labels
                    def get_ticker_theme_scores(ticker: str, meta: dict, beta: float) -> dict:
                        scores = {
                            "AI & High-Beta Tech": 0.0,
                            "Defensive (Value/Utilities/Staples)": 0.0,
                            "Commodities & Inflation Sensitive": 0.0,
                            "Rate-Sensitive (Financials/Real Estate)": 0.0
                        }
                        t_up = ticker.upper()
                        sector = (meta.get("sector") or "").lower()
                        industry = (meta.get("industry") or "").lower()
                        
                        # AI & High-Beta Tech
                        tech_tickers = {"NVDA", "AMD", "AVGO", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "NFLX", "SMH", "XLK", "QQQ"}
                        if t_up in tech_tickers:
                            scores["AI & High-Beta Tech"] += 3.0
                        if any(s in sector for s in ["technology", "communication"]):
                            scores["AI & High-Beta Tech"] += 2.0
                        if any(ind in industry for ind in ["semiconductor", "software", "hardware", "consumer electronics", "internet"]):
                            scores["AI & High-Beta Tech"] += 2.0
                        if beta > 1.25:
                            scores["AI & High-Beta Tech"] += 1.0
                            
                        # Defensive
                        defensive_tickers = {"XLV", "XLP", "XLU", "LLY", "UNH", "JNJ", "PG", "KO", "PEP", "WMT", "COST", "DHR", "NEE", "SO", "SPY", "DIA"}
                        if t_up in defensive_tickers:
                            scores["Defensive (Value/Utilities/Staples)"] += 3.0
                        if any(s in sector for s in ["healthcare", "consumer defensive", "utilities"]):
                            scores["Defensive (Value/Utilities/Staples)"] += 2.0
                        if any(ind in industry for ind in ["utilities", "drug", "pharmaceutical", "medical", "beverage", "packaged foods", "discount stores"]):
                            scores["Defensive (Value/Utilities/Staples)"] += 2.0
                        if beta < 0.85:
                            scores["Defensive (Value/Utilities/Staples)"] += 1.0
                            
                        # Commodities
                        commodity_tickers = {"XLE", "XLB", "XOP", "GDX", "GLD", "USO", "SLV", "FCX", "COP", "XOM", "CVX", "NEM"}
                        if t_up in commodity_tickers:
                            scores["Commodities & Inflation Sensitive"] += 3.0
                        if any(s in sector for s in ["energy", "basic materials"]):
                            scores["Commodities & Inflation Sensitive"] += 2.0
                        if any(ind in industry for ind in ["oil", "gas", "gold", "silver", "copper", "metal", "mining", "steel", "chemical"]):
                            scores["Commodities & Inflation Sensitive"] += 2.0
                            
                        # Rate-Sensitive
                        rate_tickers = {"XLF", "XLRE", "KRE", "JPM", "BAC", "MS", "GS", "WFC", "BLK", "SCHW", "AMT", "PLD", "CCI"}
                        if t_up in rate_tickers:
                            scores["Rate-Sensitive (Financials/Real Estate)"] += 3.0
                        if any(s in sector for s in ["financial services", "financials", "real estate"]):
                            scores["Rate-Sensitive (Financials/Real Estate)"] += 2.0
                        if any(ind in industry for ind in ["bank", "insurance", "credit", "capital markets", "savings", "reit", "real estate investment"]):
                            scores["Rate-Sensitive (Financials/Real Estate)"] += 2.0
                            
                        return scores

                    cluster_themes = {}
                    used_themes = {}
                    for c_id in range(k_adjusted):
                        cluster_indices = np.where(cluster_ids == c_id)[0]
                        c_tickers = [ticker_list[idx] for idx in cluster_indices]
                        
                        totals = {
                            "AI & High-Beta Tech": 0.0,
                            "Defensive (Value/Utilities/Staples)": 0.0,
                            "Commodities & Inflation Sensitive": 0.0,
                            "Rate-Sensitive (Financials/Real Estate)": 0.0
                        }
                        for t in c_tickers:
                            meta = fetch_info_details(t)
                            beta = meta.get("beta", 1.0)
                            t_scores = get_ticker_theme_scores(t, meta, beta)
                            for theme, val in t_scores.items():
                                totals[theme] += val
                                
                        best_theme = max(totals, key=totals.get)
                        if totals[best_theme] == 0.0:
                            fallbacks = [
                                "AI & High-Beta Tech",
                                "Defensive (Value/Utilities/Staples)",
                                "Commodities & Inflation Sensitive",
                                "Rate-Sensitive (Financials/Real Estate)"
                            ]
                            best_theme = fallbacks[c_id % len(fallbacks)]
                            
                        if best_theme not in used_themes:
                            used_themes[best_theme] = 1
                            final_label = best_theme
                        else:
                            used_themes[best_theme] += 1
                            num = used_themes[best_theme]
                            roman = {2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}.get(num, str(num))
                            final_label = f"{best_theme} {roman}"
                            
                        cluster_themes[c_id] = final_label

                    def get_theme_color(theme_label: str) -> str:
                        if "AI & High-Beta Tech" in theme_label:
                            return "#a855f7" # Purple
                        elif "Defensive" in theme_label:
                            return "#10b981" # Emerald Green
                        elif "Commodities" in theme_label:
                            return "#f59e0b" # Amber/Gold
                        elif "Rate-Sensitive" in theme_label:
                            return "#3b82f6" # Blue
                        return "#64748b" # Slate/Grey

                    # 5. Build hover texts
                    hover_texts = []
                    for idx, t in enumerate(ticker_list):
                        row = data_df[data_df["Ticker"] == t]
                        if not row.empty:
                            name = row["Name"].values[0]
                            beta = row["Beta"].values[0]
                            rsi = row["RSI"].values[0]
                            fdts = row["FDTS Signal"].values[0]
                        else:
                            meta = fetch_info_details(t)
                            name = meta["name"]
                            beta = meta["beta"]
                            rsi = 50.0
                            fdts = "No Trade"
                            
                        c_id = cluster_ids[idx]
                        theme = cluster_themes[c_id]
                        fd_emoji = {"Buy": "🟢 Buy", "Sell": "🔴 Sell", "No Trade": "⚪ No Trade"}.get(fdts, fdts)
                        
                        hover_text = f"<b>{t}</b> ({name})<br>"
                        hover_text += f"Theme: {theme}<br>"
                        hover_text += f"Cluster ID: {c_id}<br>"
                        hover_text += f"Beta vs {self.beta_index}: {beta:.2f}<br>"
                        hover_text += f"RSI (14): {rsi:.1f}<br>"
                        hover_text += f"FDTS Signal: {fd_emoji}"
                        hover_texts.append(hover_text)

                    # 6. Render Plots (Side-by-Side)
                    fig_pca = go.Figure()
                    fig_pca.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)
                    fig_pca.add_vline(x=0, line_dash="dash", line_color="#475569", line_width=1)
                    
                    for c_id in range(k_adjusted):
                        c_indices = np.where(cluster_ids == c_id)[0]
                        if len(c_indices) == 0:
                            continue
                        theme_name = cluster_themes[c_id]
                        theme_color = get_theme_color(theme_name)
                        
                        fig_pca.add_trace(go.Scatter(
                            x=x_coords[c_indices],
                            y=y_coords[c_indices],
                            mode="markers+text",
                            name=theme_name,
                            text=[ticker_list[i] for i in c_indices],
                            textposition="top center",
                            hoverinfo="text",
                            hovertext=[hover_texts[i] for i in c_indices],
                            marker=dict(
                                size=14,
                                color=theme_color,
                                line=dict(width=1.5, color="#0d1b2e")
                            )
                        ))
                        
                    fig_pca.update_layout(
                         xaxis=dict(title="Principal Component 1 (PC1) ->", gridcolor="rgba(148,163,184,0.08)"),
                         yaxis=dict(title="Principal Component 2 (PC2) ->", gridcolor="rgba(148,163,184,0.08)"),
                         paper_bgcolor="#0d1b2e",
                         plot_bgcolor="rgba(21, 40, 71, 0.2)",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor="rgba(13,27,46,0.6)"),
                         margin=dict(l=20, r=20, t=30, b=20),
                         height=500,
                         font=dict(color="#e2e8f0", family="Inter")
                    )

                    fig_net = go.Figure()
                    edge_x = []
                    edge_y = []
                    for i in range(len(ticker_list)):
                        for j in range(i + 1, len(ticker_list)):
                            corr_coef = corr_matrix.iloc[i, j]
                            if corr_coef > threshold_val:
                                edge_x.extend([x_coords[i], x_coords[j], None])
                                edge_y.extend([y_coords[i], y_coords[j], None])
                                
                    fig_net.add_trace(go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        mode="lines",
                        line=dict(width=1.5, color="rgba(148, 163, 184, 0.3)"),
                        hoverinfo="none",
                        showlegend=False
                    ))
                    
                    for c_id in range(k_adjusted):
                        c_indices = np.where(cluster_ids == c_id)[0]
                        if len(c_indices) == 0:
                            continue
                        theme_name = cluster_themes[c_id]
                        theme_color = get_theme_color(theme_name)
                        
                        fig_net.add_trace(go.Scatter(
                            x=x_coords[c_indices],
                            y=y_coords[c_indices],
                            mode="markers+text",
                            name=theme_name,
                            text=[ticker_list[i] for i in c_indices],
                            textposition="top center",
                            hoverinfo="text",
                            hovertext=[hover_texts[i] for i in c_indices],
                            marker=dict(
                                size=14,
                                color=theme_color,
                                line=dict(width=1.5, color="#0d1b2e")
                            )
                        ))
                        
                    fig_net.update_layout(
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        paper_bgcolor="#0d1b2e",
                        plot_bgcolor="rgba(21, 40, 71, 0.2)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor="rgba(13,27,46,0.6)"),
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=500,
                        font=dict(color="#e2e8f0", family="Inter")
                    )

                    plot_col1, plot_col2 = st.columns(2)
                    with plot_col1:
                        st.markdown("<h5 style='text-align: left; margin-bottom: 0px;'>📍 PCA Asset Quadrant Projection</h5>", unsafe_allow_html=True)
                        st.plotly_chart(fig_pca, use_container_width=True, key="cluster_pca_chart")
                    with plot_col2:
                        st.markdown(f"<h5 style='text-align: left; margin-bottom: 0px;'>🕸️ Similarity Network Graph (Correlation > {threshold_val:.2f})</h5>", unsafe_allow_html=True)
                        st.plotly_chart(fig_net, use_container_width=True, key="cluster_network_chart")

                    # 7. Thematic Breakdown Grid
                    st.markdown("### 📊 Thematic Breakdown & Dynamic Insights")
                    for i in range(0, k_adjusted, 2):
                        col1, col2 = st.columns(2)
                        
                        # Cluster 1
                        with col1:
                            c_id = i
                            c_indices = np.where(cluster_ids == c_id)[0]
                            c_tickers = [ticker_list[idx] for idx in c_indices]
                            theme_name = cluster_themes[c_id]
                            theme_color = get_theme_color(theme_name)
                            
                            # Metrics
                            c_betas = []
                            for t in c_tickers:
                                row = data_df[data_df["Ticker"] == t]
                                c_betas.append(row["Beta"].values[0] if not row.empty else fetch_info_details(t).get("beta", 1.0))
                            avg_beta = np.mean(c_betas) if c_betas else 1.0
                            
                            c_rsis = []
                            for t in c_tickers:
                                row = data_df[data_df["Ticker"] == t]
                                c_rsis.append(row["RSI"].values[0] if not row.empty else 50.0)
                            avg_rsi = np.mean(c_rsis) if c_rsis else 50.0
                            
                            signals = []
                            for t in c_tickers:
                                row = data_df[data_df["Ticker"] == t]
                                signals.append(row["FDTS Signal"].values[0] if not row.empty else "No Trade")
                                
                            buys = signals.count("Buy")
                            sells = signals.count("Sell")
                            no_trades = signals.count("No Trade")
                            signal_str = f"🟢 {buys} Buy | ⚪ {no_trades} No Trade | 🔴 {sells} Sell"
                            
                            base_theme = theme_name
                            for r in [" VI", " V", " IV", " III", " II"]:
                                if base_theme.endswith(r):
                                    base_theme = base_theme[:-len(r)]
                                    break
                                    
                            fdts_sentiment = f"{buys} Buy, {sells} Sell, {no_trades} No Trade"
                            if buys > sells and buys > no_trades:
                                fdts_sentiment += " (Bullish Connotation)"
                            elif sells > buys and sells > no_trades:
                                fdts_sentiment += " (Bearish Connotation)"
                            else:
                                fdts_sentiment += " (Neutral/Accumulation)"
                                
                            if "AI & High-Beta Tech" in base_theme:
                                narrative = f"This cluster represents high-beta growth equities, showing an average beta of **{avg_beta:.2f}** and current RSI of **{avg_rsi:.1f}**. The cluster is heavily driven by secular technology expansion, semiconductor demand, and AI infrastructure capital expenditures. Current FDTS sentiment is **{fdts_sentiment}**. Recommended actions involve utilizing option structures like bull call spreads or calendar spreads to capture high implied volatility while mitigating tail risk."
                            elif "Defensive" in base_theme:
                                narrative = f"Composed of low-beta, stable cash flow entities with an average beta of **{avg_beta:.2f}** and RSI of **{avg_rsi:.1f}**. These companies act as market shock-absorbers, demonstrating resilience during liquidity contractions or rate hikes. With FDTS signaling **{fdts_sentiment}**, this cluster serves as a capital preservation vehicle. Writing covered calls or cash-secured puts is favored here for yield enhancement."
                            elif "Commodities" in base_theme:
                                narrative = f"This group represents raw materials, energy producers, and miners, presenting an average beta of **{avg_beta:.2f}** and RSI of **{avg_rsi:.1f}**. These assets are highly sensitive to supply chain bottlenecks, geopolitical risks, and global inflation. Current FDTS signal profile is **{fdts_sentiment}**. Option play: long calls or ratio spreads to capitalize on supply-driven price spikes."
                            elif "Rate-Sensitive" in base_theme:
                                narrative = f"Comprising banking institutions, asset managers, and yield-sensitive real estate assets, with an average beta of **{avg_beta:.2f}** and RSI of **{avg_rsi:.1f}**. Highly influenced by yield curve dynamics, Federal Reserve policy, and credit spreads. FDTS signals are **{fdts_sentiment}**. Position strategy: interest-rate sensitive options or credit vertical spreads."
                            else:
                                narrative = f"This cluster is categorized by general market themes with an average beta of **{avg_beta:.2f}** and RSI of **{avg_rsi:.1f}**. The constituent elements present mixed profiles with FDTS signaling **{fdts_sentiment}**. Standard defensive positioning or index replication is suggested."

                            st.markdown(
                                f"""
                                <div style="border: 1px solid {theme_color}60; border-radius: 8px; padding: 15px; margin-bottom: 12px; background-color: rgba(21, 40, 71, 0.25);">
                                    <h4 style="color: {theme_color}; margin-top: 0; margin-bottom: 8px;">🧬 Theme: {theme_name}</h4>
                                    <p style="font-size: 13px; color: #cbd5e1; margin-bottom: 12px; line-height: 1.4;">{narrative}</p>
                                    <div style="display: flex; gap: 15px; font-size: 12px; color: #94a3b8; margin-bottom: 5px;">
                                        <span><b>Avg Beta:</b> {avg_beta:.2f}</span>
                                        <span><b>Avg RSI:</b> {avg_rsi:.1f}</span>
                                        <span><b>Signals:</b> {signal_str}</span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            c_data = []
                            for t in c_tickers:
                                row = data_df[data_df["Ticker"] == t]
                                if not row.empty:
                                    name = row["Name"].values[0]
                                    sector = row["Sector"].values[0]
                                    rsi_val = row["RSI"].values[0]
                                    beta_val = row["Beta"].values[0]
                                    fdts_val = row["FDTS Signal"].values[0]
                                else:
                                    meta = fetch_info_details(t)
                                    name = meta["name"]
                                    sector = meta["sector"]
                                    rsi_val = 50.0
                                    beta_val = meta["beta"]
                                    fdts_val = "No Trade"
                                    
                                fd_emoji = {"Buy": "🟢 Buy", "Sell": "🔴 Sell", "No Trade": "⚪ No Trade"}.get(fdts_val, fdts_val)
                                c_data.append({
                                    "Ticker": t,
                                    "Company Name": name,
                                    "Sector": sector,
                                    "Beta": round(beta_val, 2),
                                    "RSI": round(rsi_val, 1),
                                    "FDTS Signal": fd_emoji
                                })
                            st.dataframe(
                                pd.DataFrame(c_data),
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Beta": st.column_config.NumberColumn(f"Beta vs {self.beta_index}", format="%.2f")
                                }
                            )

                        # Cluster 2
                        if i + 1 < k_adjusted:
                            with col2:
                                c_id = i + 1
                                c_indices = np.where(cluster_ids == c_id)[0]
                                c_tickers = [ticker_list[idx] for idx in c_indices]
                                theme_name = cluster_themes[c_id]
                                theme_color = get_theme_color(theme_name)
                                
                                # Metrics
                                c_betas = []
                                for t in c_tickers:
                                    row = data_df[data_df["Ticker"] == t]
                                    c_betas.append(row["Beta"].values[0] if not row.empty else fetch_info_details(t).get("beta", 1.0))
                                avg_beta = np.mean(c_betas) if c_betas else 1.0
                                
                                c_rsis = []
                                for t in c_tickers:
                                    row = data_df[data_df["Ticker"] == t]
                                    c_rsis.append(row["RSI"].values[0] if not row.empty else 50.0)
                                avg_rsi = np.mean(c_rsis) if c_rsis else 50.0
                                
                                signals = []
                                for t in c_tickers:
                                    row = data_df[data_df["Ticker"] == t]
                                    signals.append(row["FDTS Signal"].values[0] if not row.empty else "No Trade")
                                    
                                buys = signals.count("Buy")
                                sells = signals.count("Sell")
                                no_trades = signals.count("No Trade")
                                signal_str = f"🟢 {buys} Buy | ⚪ {no_trades} No Trade | 🔴 {sells} Sell"
                                
                                base_theme = theme_name
                                for r in [" VI", " V", " IV", " III", " II"]:
                                    if base_theme.endswith(r):
                                        base_theme = base_theme[:-len(r)]
                                        break
                                        
                                fdts_sentiment = f"{buys} Buy, {sells} Sell, {no_trades} No Trade"
                                if buys > sells and buys > no_trades:
                                    fdts_sentiment += " (Bullish Connotation)"
                                elif sells > buys and sells > no_trades:
                                    fdts_sentiment += " (Bearish Connotation)"
                                else:
                                    fdts_sentiment += " (Neutral/Accumulation)"
                                    
                                if "AI & High-Beta Tech" in base_theme:
                                    narrative = f"This cluster represents high-beta growth equities, showing an average beta of **{avg_beta:.2f}** and current RSI of **{avg_rsi:.1f}**. The cluster is heavily driven by secular technology expansion, semiconductor demand, and AI infrastructure capital expenditures. Current FDTS sentiment is **{fdts_sentiment}**. Recommended actions involve utilizing option structures like bull call spreads or calendar spreads to capture high implied volatility while mitigating tail risk."
                                elif "Defensive" in base_theme:
                                    narrative = f"Composed of low-beta, stable cash flow entities with an average beta of **{avg_beta:.2f}** and RSI of **{avg_rsi:.1f}**. These companies act as market shock-absorbers, demonstrating resilience during liquidity contractions or rate hikes. With FDTS signaling **{fdts_sentiment}**, this cluster serves as a capital preservation vehicle. Writing covered calls or cash-secured puts is favored here for yield enhancement."
                                elif "Commodities" in base_theme:
                                    narrative = f"This group represents raw materials, energy producers, and miners, presenting an average beta of **{avg_beta:.2f}** and RSI of **{avg_rsi:.1f}**. These assets are highly sensitive to supply chain bottlenecks, geopolitical risks, and global inflation. Current FDTS signal profile is **{fdts_sentiment}**. Option play: long calls or ratio spreads to capitalize on supply-driven price spikes."
                                elif "Rate-Sensitive" in base_theme:
                                    narrative = f"Comprising banking institutions, asset managers, and yield-sensitive real estate assets, with an average beta of **{avg_beta:.2f}** and RSI of **{avg_rsi:.1f}**. Highly influenced by yield curve dynamics, Federal Reserve policy, and credit spreads. FDTS signals are **{fdts_sentiment}**. Position strategy: interest-rate sensitive options or credit vertical spreads."
                                else:
                                    narrative = f"This cluster is categorized by general market themes with an average beta of **{avg_beta:.2f}** and RSI of **{avg_rsi:.1f}**. The constituent elements present mixed profiles with FDTS signaling **{fdts_sentiment}**. Standard defensive positioning or index replication is suggested."

                                st.markdown(
                                    f"""
                                    <div style="border: 1px solid {theme_color}60; border-radius: 8px; padding: 15px; margin-bottom: 12px; background-color: rgba(21, 40, 71, 0.25);">
                                        <h4 style="color: {theme_color}; margin-top: 0; margin-bottom: 8px;">🧬 Theme: {theme_name}</h4>
                                        <p style="font-size: 13px; color: #cbd5e1; margin-bottom: 12px; line-height: 1.4;">{narrative}</p>
                                        <div style="display: flex; gap: 15px; font-size: 12px; color: #94a3b8; margin-bottom: 5px;">
                                            <span><b>Avg Beta:</b> {avg_beta:.2f}</span>
                                            <span><b>Avg RSI:</b> {avg_rsi:.1f}</span>
                                            <span><b>Signals:</b> {signal_str}</span>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                c_data = []
                                for t in c_tickers:
                                    row = data_df[data_df["Ticker"] == t]
                                    if not row.empty:
                                        name = row["Name"].values[0]
                                        sector = row["Sector"].values[0]
                                        rsi_val = row["RSI"].values[0]
                                        beta_val = row["Beta"].values[0]
                                        fdts_val = row["FDTS Signal"].values[0]
                                    else:
                                        meta = fetch_info_details(t)
                                        name = meta["name"]
                                        sector = meta["sector"]
                                        rsi_val = 50.0
                                        beta_val = meta["beta"]
                                        fdts_val = "No Trade"
                                        
                                    fd_emoji = {"Buy": "🟢 Buy", "Sell": "🔴 Sell", "No Trade": "⚪ No Trade"}.get(fdts_val, fdts_val)
                                    c_data.append({
                                        "Ticker": t,
                                        "Company Name": name,
                                        "Sector": sector,
                                        "Beta": round(beta_val, 2),
                                        "RSI": round(rsi_val, 1),
                                        "FDTS Signal": fd_emoji
                                    })
                                st.dataframe(
                                    pd.DataFrame(c_data),
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Beta": st.column_config.NumberColumn(f"Beta vs {self.beta_index}", format="%.2f")
                                    }
                                )

        # --- TAB 7: INTERACTIVE DRILL-DOWNS ---
        with tab_drilldowns:
            st.markdown("### Interactive Drill-Down Explorer")
            drill_tabs = st.tabs(["Level 1: Ticker Detail", "Level 2: Sector/Theme Analysis", "Level 3: Portfolio Impact Model"])
            
            with drill_tabs[0]:
                st.markdown("#### Drilldown Level 1 — Ticker Detail Profile")
                ticker_names = get_ticker_names(getattr(self, "universe_name", "Options Default Watchlist"))
                sel_ticker = st.selectbox(
                    "Select Ticker for deep-dive analysis:",
                    options=ticker_list,
                    key="drill_ticker",
                    format_func=lambda ticker: format_ticker_display(ticker, ticker_names)
                )
                
                # Reset visual canvas on ticker transition to clear stale visualizations immediately
                if "ui_last_drill_ticker" not in st.session_state:
                    st.session_state["ui_last_drill_ticker"] = sel_ticker

                if sel_ticker != st.session_state["ui_last_drill_ticker"]:
                    st.session_state["ui_last_drill_ticker"] = sel_ticker
                    st.info(f"🔄 Fetching profile details and price action chart for {sel_ticker}...")
                    st.rerun()
                
                if sel_ticker:
                    tick_info = fetch_info_details(sel_ticker)
                    
                    dc1, dc2 = st.columns([1, 1.5])
                    with dc1:
                        st.markdown(f"##### {tick_info['name']} ({sel_ticker})")
                        st.caption(f"**Sector:** {tick_info['sector']} | **Industry:** {tick_info['industry']}")
                        if "Beta" in data_df.columns:
                            calc_beta_val = data_df.loc[data_df["Ticker"] == sel_ticker, "Beta"].values[0]
                            st.write(f"• **Beta vs {self.beta_index}:** `{calc_beta_val:.2f}`")
                        else:
                            st.write(f"• **Beta vs SPY:** `{tick_info['beta']}`")
                        st.write(f"• **Market Capitalisation:** `${tick_info['market_cap'] / 1e9:.2f}B`")
                        
                        # Display FDTS Signal in Detail Profile
                        if "FDTS Signal" in data_df.columns:
                            fd_val = data_df.loc[data_df["Ticker"] == sel_ticker, "FDTS Signal"].values[0]
                            fd_emoji = {"Buy": "🟢 Buy", "Sell": "🔴 Sell", "No Trade": "⚪ No Trade"}.get(fd_val, "⚪ No Trade")
                            st.write(f"• **FDTS + MACD Signal:** **{fd_emoji}**")
                            
                        st.markdown(f"**Description Summary:**\n*{tick_info['description'][:400]}...*")
                    with dc2:
                        fig_price = go.Figure()
                        fig_price.add_trace(go.Scatter(x=close_df.index, y=close_df[sel_ticker], mode="lines", name="Price", line=dict(color="#3ab54a")))
                        fig_price.add_trace(go.Scatter(x=close_df.index, y=close_df[sel_ticker].rolling(50).mean(), mode="lines", name="50 SMA", line=dict(color="#93c5fd", dash="dash")))
                        fig_price.add_trace(go.Scatter(x=close_df.index, y=close_df[sel_ticker].rolling(200).mean(), mode="lines", name="200 SMA", line=dict(color="#ef4444", dash="dot")))
                        
                        fig_price.update_layout(
                            paper_bgcolor="#0d1b2e",
                            plot_bgcolor="rgba(21, 40, 71, 0.2)",
                            font=dict(color="#e2e8f0"),
                            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                            margin=dict(l=20, r=20, t=10, b=20),
                            height=250
                        )
                        st.plotly_chart(fig_price, use_container_width=True, key="drill_price_chart")

                    st.markdown("###### Technical Momentum & Risk Metrics")
                    tc1, tc2, tc3 = st.columns(3)
                    with tc1:
                        rsi_val = float(data_df.loc[data_df["Ticker"] == sel_ticker, "RSI"].values[0])
                        st.metric("RSI (14)", f"{rsi_val:.1f}", "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral")
                    with tc2:
                        max_val = close_df[sel_ticker].max()
                        curr_val = close_df[sel_ticker].iloc[-1]
                        dd = ((curr_val / max_val) - 1) * 100
                        st.metric("Drawdown from High", f"{dd:.2f}%")
                    with tc3:
                        sqz_status = data_df.loc[data_df["Ticker"] == sel_ticker, "Bollinger Squeeze"].values[0]
                        st.metric("Bollinger Band state", sqz_status)

            with drill_tabs[1]:
                st.markdown("#### Drilldown Level 2 — Sector & Theme Analysis")
                sec_rot = data_df.groupby("Sector")[["RS-Ratio", "RS-Momentum"]].mean().reset_index()
                
                fig_sec_rot = go.Figure()
                fig_sec_rot.add_hline(y=100, line_dash="dash", line_color="#475569")
                fig_sec_rot.add_vline(x=100, line_dash="dash", line_color="#475569")
                
                fig_sec_rot.add_trace(go.Scatter(
                    x=sec_rot["RS-Ratio"],
                    y=sec_rot["RS-Momentum"],
                    mode="markers+text",
                    text=sec_rot["Sector"],
                    textposition="top center",
                    marker=dict(size=14, color="#3ab54a", line=dict(width=1.5, color="#0d1b2e"))
                ))
                fig_sec_rot.update_layout(
                    title="Sector Rotation Graph (RRG Model)",
                    xaxis=dict(title="Sector Relative Strength"),
                    yaxis=dict(title="Sector Momentum"),
                    paper_bgcolor="#0d1b2e",
                    plot_bgcolor="rgba(21, 40, 71, 0.2)",
                    font=dict(color="#e2e8f0"),
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=300
                )
                st.plotly_chart(fig_sec_rot, use_container_width=True, key="sector_rot_chart")

            with drill_tabs[2]:
                st.markdown("#### Drilldown Level 3 — Portfolio Impact Simulator")
                ic1, ic2 = st.columns([1, 1.2])
                with ic1:
                    st.markdown("**Simulated Capital Deployment**")
                    ticker_names = get_ticker_names(getattr(self, "universe_name", "Options Default Watchlist"))
                    sim_ticker = st.selectbox(
                        "Ticker to add/deploy:",
                        options=ticker_list,
                        key="sim_ticker",
                        format_func=lambda ticker: format_ticker_display(ticker, ticker_names)
                    )
                    sim_qty = st.number_input("Shares to trade:", min_value=1, value=100, step=10, key="sim_qty")
                    sim_price = close_df[sim_ticker].iloc[-1]
                    sim_cost = sim_qty * sim_price
                    st.write(f"Estimated Capital Outlay: **${sim_cost:,.2f}** at price `${sim_price:.2f}`")
                with ic2:
                    st.markdown("**Simulated Risk Shifts**")
                    if has_portfolio:
                        cur_port_beta = float((data_df["Beta"] * data_df["Weight"]).sum())
                        new_net_liq = net_liq + sim_cost
                        
                        sim_w = sim_cost / new_net_liq
                        new_port_beta = cur_port_beta * (net_liq / new_net_liq) + (data_df.loc[data_df["Ticker"] == sim_ticker, "Beta"].values[0] * sim_w)
                        
                        st.metric(f"Portfolio Beta Shift (vs {self.beta_index})", f"{new_port_beta:.3f}", f"{new_port_beta - cur_port_beta:+.3f} change")
                        
                        sim_sec = data_df.loc[data_df["Ticker"] == sim_ticker, "Sector"].values[0]
                        curr_sec_w = float(data_df[data_df["Sector"] == sim_sec]["Weight"].sum() * 100.0)
                        new_sec_w = (curr_sec_w * net_liq + sim_cost * 100.0) / new_net_liq
                        st.metric(f"Concentration in {sim_sec}", f"{new_sec_w:.2f}%", f"{new_sec_w - curr_sec_w:+.2f}% change")
                    else:
                        st.info("Please load an options portfolio to calculate simulated risk shifts.")

    # --- Insight Narrative Builder ---
    def generate_ai_insight(self, regime: str, score: int, data_df: pd.DataFrame, breadth: float) -> str:
        """Rule-based natural language generator to produce executive comments."""
        exp_df = data_df[data_df["Stage"] == "Expansion / Leadership"]
        leaders_str = ", ".join(exp_df["Ticker"].head(3).tolist()) if not exp_df.empty else "none"
        
        early_df = data_df[data_df["Stage"] == "Early Accumulation"]
        accum_str = ", ".join(early_df["Ticker"].head(3).tolist()) if not early_df.empty else "none"

        comment = f"Market conditions are currently classified as **{regime}** with a Universe Health score of **{score}/100**. "
        
        if regime == "Risk-On":
            comment += f"Breadth is highly supportive at **{breadth:.1f}%** above the 50-day moving average. "
            comment += f"Leadership is highly concentrated in expansion candidates ({leaders_str}). "
            comment += f"Conditions favor aggressive swing positioning and accumulation of early stage breakouts ({accum_str})."
        elif regime == "Defensive":
            comment += f"Strong headwinds persist with breadth declining to **{breadth:.1f}%**. "
            comment += "Risk management dictates trimming late stage exhaustion names and focusing on capital preservation. "
            comment += "Tactical hedges or short positions should be maintained."
        elif regime == "High Volatility":
            comment += f"Volatility indices are elevated. Breadth stands at **{breadth:.1f}%**. "
            comment += "Standard long exposures should be reduced in favor of calendar spreads, iron condors, or premium-selling strategies."
        else:
            comment += f"Neutral conditions prevail. Participation is mixed at **{breadth:.1f}%**. "
            comment += f"Focus on sector rotation and selective entry into early accumulation setups ({accum_str}) while avoiding deteriorating symbols."
            
        return comment
