"""
FazDane Analytics — Tier 1
Sector Rotation Monitor (Notebook Migration)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import splprep, splev
import logging
from modules.base_module import FazDaneModule
from utils.universe_manager import render_universe_manager

logger = logging.getLogger("SectorRotation")


# Fixed distinct colors for Plotly
COLORS = ["#06b6d4", "#3b82f6", "#10b981", "#f59e0b", "#6366f1", "#ec4899", "#8b5cf6", "#14b8a6", "#f97316", "#84cc16", "#ef4444", "#a855f7", "#fbbf24", "#34d399", "#f87171"]

def get_color(idx):
    return COLORS[idx % len(COLORS)]

# ============================================================
# Logic functions
# ============================================================

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _tema(series: pd.Series, period: int) -> pd.Series:
    ema1 = _ema(series, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3

def calculate_fdts_signal(symbol: str, ticker_df: pd.DataFrame, period: int = 20) -> str:
    """Calculate the FDTS + MACD Trade Signal (Buy/No Trade/Sell)."""
    required = {"Open", "High", "Low", "Close"}
    if ticker_df.empty or not required.issubset(ticker_df.columns):
        return "No Trade"

    data = ticker_df[["Open", "High", "Low", "Close"]].dropna().copy()
    if len(data) < 60:
        return "No Trade"

    price = (data["High"] + data["Low"] + data["Close"]) / 3
    tma1 = _tema(price, period)
    tma2 = _tema(tma1, period)
    typical_tema = tma1 + (tma1 - tma2)

    raw_ha_close = (data["Open"] + data["High"] + data["Low"] + data["Close"]) / 4
    ha_open = pd.Series(index=data.index, dtype="float64")
    ha_open.iloc[0] = (data["High"].iloc[0] + data["Low"].iloc[0]) / 2
    for i in range(1, len(data)):
        ha_open.iloc[i] = (raw_ha_close.iloc[i - 1] + ha_open.iloc[i - 1]) / 2

    ha_close = (
        raw_ha_close
        + ha_open
        + pd.concat([data["High"], ha_open], axis=1).max(axis=1)
        + pd.concat([data["Low"], ha_open], axis=1).min(axis=1)
    ) / 4

    ha_tma1 = _tema(ha_close, period)
    ha_tma2 = _tema(ha_tma1, period)
    ha_tema = ha_tma1 + (ha_tma1 - ha_tma2)
    fdts_dev = typical_tema - ha_tema

    macd_long = _ema(data["Close"], 3) - _ema(data["Close"], 10)
    macd_long_dev = macd_long - _ema(macd_long, 16)
    macd_short = _ema(data["Close"], 12) - _ema(data["Close"], 26)
    macd_short_dev = macd_short - _ema(macd_short, 9)

    state = pd.Series(0, index=data.index, dtype="int64")
    state[(fdts_dev > 0) & (macd_long_dev > 0)] = 1
    state[(fdts_dev < 0) & (macd_short_dev < 0)] = -1

    current_state = int(state.iloc[-1])
    return "Buy" if current_state == 1 else "Sell" if current_state == -1 else "No Trade"

def extract_ticker_df(raw, symbol):
    if raw.empty:
        return pd.DataFrame()
    ticker_df = pd.DataFrame(index=raw.index)
    if isinstance(raw.columns, pd.MultiIndex):
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in raw and symbol in raw[col].columns:
                ticker_df[col] = raw[col][symbol]
    else:
        # If single symbol was downloaded
        ticker_df = raw.copy()
        if "Adj Close" in ticker_df.columns and "Close" not in ticker_df.columns:
            ticker_df["Close"] = ticker_df["Adj Close"]
    return ticker_df.dropna(how="all")

@st.cache_data(ttl=3600, show_spinner=False)
def download_rotation_data(tickers, benchmark, period_type="weeks", period_number=12):
    if benchmark not in tickers:
        tickers = tickers + [benchmark]

    if period_type == "days":
        download_period = f"{max(period_number * 4, 90)}d"
        interval = "1d"
    elif period_type == "weeks":
        download_period = f"{max(period_number * 20, 250)}d"
        interval = "1d"
    else:
        download_period = f"{max(period_number * 55, 700)}d"
        interval = "1d"

    raw = yf.download(tickers, period=download_period, interval=interval, progress=False)

    if raw.empty:
        return pd.DataFrame(), {}

    # Calculate FDTS signals for all tickers (except benchmark) using raw daily data
    fdts_signals = {}
    for ticker in tickers:
        if ticker == benchmark:
            continue
        ticker_df = extract_ticker_df(raw, ticker)
        sig = calculate_fdts_signal(ticker, ticker_df)
        fdts_signals[ticker] = {"Buy": "🟢 Buy", "Sell": "🔴 Sell", "No Trade": "⚪ No Trade"}.get(sig, "⚪ No Trade")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    prices = prices.dropna(how="all")
    valid_tickers = [t for t in tickers if t in prices.columns]
    prices = prices[valid_tickers]

    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark {benchmark} is missing from data.")

    if period_type == "weeks":
        prices = prices.resample("W-FRI").last()
    elif period_type == "months":
        prices = prices.resample("ME").last() # 'ME' is newer pandas equivalent to 'M'

    prices = prices.dropna(axis=1, how="all").dropna()
    return prices, fdts_signals

def calculate_rotation(prices, ticker_dict, benchmark, ema_span=4):
    benchmark_price = prices[benchmark]
    rs_ratio_data = {}
    rs_momentum_data = {}

    for ticker in ticker_dict.keys():
        if ticker == benchmark or ticker not in prices.columns:
            continue

        asset_price = prices[ticker]
        relative_strength = asset_price / benchmark_price

        relative_strength_smooth = relative_strength.ewm(span=ema_span, adjust=False).mean()

        rs_ratio_raw = 100 * (relative_strength_smooth / relative_strength_smooth.rolling(10).mean())
        rs_ratio = rs_ratio_raw.ewm(span=ema_span, adjust=False).mean()

        rs_momentum_raw = 100 * (rs_ratio / rs_ratio.rolling(5).mean())
        rs_momentum = rs_momentum_raw.ewm(span=ema_span, adjust=False).mean()

        rs_ratio_data[ticker] = rs_ratio
        rs_momentum_data[ticker] = rs_momentum

    return pd.DataFrame(rs_ratio_data).dropna(), pd.DataFrame(rs_momentum_data).dropna()

def smooth_tail_path(x, y, points=120):
    x, y = np.array(x), np.array(y)
    if len(x) < 3:
        return x, y
    try:
        tck, u = splprep([x, y], s=0.4)
        u_new = np.linspace(0, 1, points)
        return splev(u_new, tck)
    except Exception:
        return x, y

def get_quadrant(x, y):
    if x >= 100 and y >= 100: return "Leading"
    elif x >= 100 and y < 100: return "Weakening"
    elif x < 100 and y < 100: return "Lagging"
    else: return "Improving"

# ============================================================
# Main Module Class
# ============================================================

class SectorRotationModule(FazDaneModule):
    MODULE_NAME = "Sector Rotation Monitor"
    MODULE_ICON = "🔄"
    MODULE_DESCRIPTION = "RRG-style matrix featuring custom universes and smoothing"
    TIER = 1
    SOURCE_NOTEBOOK = "05-SPX Sector Rotation / RRG-Style Visualization.ipynb"
    CACHE_TTL = 3600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def render_sidebar(self):
        st.markdown("**Matrix Configuration**")

        universe_name, tickers_list, benchmark = render_universe_manager(
            key_prefix="sr",
            show_benchmark=True,
            label="Ticker Universe:"
        )
        # Store selections for render_main
        st.session_state["sr_universe_name"] = universe_name
        st.session_state["sr_tickers_list"] = tickers_list
        st.session_state["sr_benchmark_live"] = benchmark

        st.markdown("**Chart Parameters**")
        period_type = st.selectbox("Period Type:", ["days", "weeks", "months"], index=1, key="sr_ptype")
        period_number = st.slider("Periods:", 3, 52, 12, 1, key="sr_pnum")
        tail_len = st.slider("Tail Length:", 3, 15, 6, 1, key="sr_tail")
        ema_span = st.slider("Smooth (EMA Span):", 2, 10, 4, 1, key="sr_ema")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        scan_clicked = st.button("🔄 Generate Matrix", use_container_width=True, type="primary")

        if scan_clicked:
            st.session_state["sr_state"] = {
                "universe_name": universe_name,
                "ticker_dict": {t: t for t in tickers_list},
                "benchmark": benchmark,
                "period_type": period_type,
                "period_number": period_number,
                "tail_length": tail_len,
                "ema_span": ema_span
            }

    def render_main(self):
        state = st.session_state.get("sr_state", {
            "universe_name": "SPX Sectors",
            "ticker_dict": {t: t for t in ["XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLRE","XLK","XLU"]},
            "benchmark": "SPY",
            "period_type": "weeks",
            "period_number": 12,
            "tail_length": 6,
            "ema_span": 4
        })

        ticker_dict = state["ticker_dict"]
        caption = f"{state['universe_name']} Rotation Matrix vs {state['benchmark']}"

        self.render_section_header(
            "🔄 " + caption,
            f"Tail Length: {state['tail_length']} {state['period_type']} | Smoothing: {state['ema_span']} EMA"
        )

        with st.spinner("Fetching data and computing matrix..."):
            try:
                prices, fdts_signals = download_rotation_data(
                    list(ticker_dict.keys()), 
                    state["benchmark"], 
                    state["period_type"], 
                    state["period_number"]
                )
            except Exception as e:
                st.error(f"Data Fetch Error: {e}")
                return
                
        if prices.empty:
            st.error("No data returned. Please check tickers and benchmark.")
            return

        rs_ratio_df, rs_momentum_df = calculate_rotation(
            prices, ticker_dict, state["benchmark"], state["ema_span"]
        )

        tail_length = state["tail_length"]
        period_number = state["period_number"]
        
        rs_ratio_df = rs_ratio_df.tail(max(period_number, tail_length))
        rs_momentum_df = rs_momentum_df.tail(max(period_number, tail_length))
        
        self._render_plot(rs_ratio_df, rs_momentum_df, ticker_dict, tail_length, fdts_signals)

    def _render_plot(self, ratio_df, mom_df, ticker_dict, tail_length, fdts_signals):
        fig = go.Figure()
        
        all_x_vals, all_y_vals = [], []
        latest_rows = []
        
        color_idx = 0
        for ticker, asset_name in ticker_dict.items():
            if ticker not in ratio_df.columns:
                continue

            x = ratio_df[ticker].tail(tail_length).dropna()
            y = mom_df[ticker].tail(tail_length).dropna()

            common_idx = x.index.intersection(y.index)
            x, y = x.loc[common_idx], y.loc[common_idx]

            if len(x) < 2: continue

            latest_x, latest_y = x.iloc[-1], y.iloc[-1]
            all_x_vals.extend(x.values)
            all_y_vals.extend(y.values)
            
            color = get_color(color_idx)
            color_idx += 1
            
            x_smooth, y_smooth = smooth_tail_path(x.values, y.values, points=120)
            
            # --- The Dotted Trace exactly as requested ---
            fig.add_trace(go.Scatter(
                x=x_smooth, y=y_smooth,
                mode="lines",
                name=f"{asset_name} ({ticker})",
                line=dict(color=color, width=1.5, dash="dot"),
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Trail points
            fig.add_trace(go.Scatter(
                x=x.values, y=y.values,
                mode="markers",
                marker=dict(size=4, color=color, opacity=0.75),
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Head point
            fig.add_trace(go.Scatter(
                x=[latest_x], y=[latest_y],
                mode="markers+text",
                name=f"{asset_name} ({ticker})",
                text=[f"<b>{ticker}</b>"],
                textposition="top right",
                textfont=dict(color=color, size=13, family="Inter"),
                marker=dict(size=14, color=color, line=dict(color="black", width=1)),
                hovertemplate=f"<b>{asset_name} ({ticker})</b><br>RS-Ratio: %{{x:.2f}}<br>RS-Mom: %{{y:.2f}}<br>FDTS Signal: {fdts_signals.get(ticker, '⚪ No Trade')}<extra></extra>"
            ))
            
            quad = get_quadrant(latest_x, latest_y)
            latest_rows.append({
                "Ticker": ticker,
                "Name": asset_name,
                "RS Ratio": latest_x,
                "RS Momentum": latest_y,
                "Quadrant": quad,
                "Color": color,
                "FDTS": fdts_signals.get(ticker, "⚪ No Trade")
            })

        if not all_x_vals:
            st.warning("Not enough data to plot. Try increasing periods.")
            return

        x_min, x_max = min(all_x_vals) - 1.0, max(all_x_vals) + 1.0
        y_min, y_max = min(all_y_vals) - 1.0, max(all_y_vals) + 1.0
        x_min, x_max = min(x_min, 99), max(x_max, 101)
        y_min, y_max = min(y_min, 99), max(y_max, 101)
        
        # Quadrant backgrounds
        fig.add_shape(type="rect", x0=100, y0=100, x1=x_max, y1=y_max, fillcolor="rgba(58,181,74,0.08)", line_width=0)
        fig.add_shape(type="rect", x0=100, y0=y_min, x1=x_max, y1=100, fillcolor="rgba(245,158,11,0.08)", line_width=0)
        fig.add_shape(type="rect", x0=x_min, y0=y_min, x1=100, y1=100, fillcolor="rgba(239,68,68,0.08)", line_width=0)
        fig.add_shape(type="rect", x0=x_min, y0=100, x1=100, y1=y_max, fillcolor="rgba(59,130,246,0.08)", line_width=0)
        
        fig.add_hline(y=100, line_color="#1e3a5f", line_width=1.5)
        fig.add_vline(x=100, line_color="#1e3a5f", line_width=1.5)
        
        diff_x = (x_max - x_min) * 0.05
        diff_y = (y_max - y_min) * 0.05
        
        fig.add_annotation(x=x_max - diff_x, y=y_max - diff_y, text="LEADING", showarrow=False, font=dict(color="rgba(58,181,74,0.3)", size=26, family="Inter", weight="bold"))
        fig.add_annotation(x=x_max - diff_x, y=y_min + diff_y, text="WEAKENING", showarrow=False, font=dict(color="rgba(245,158,11,0.3)", size=26, family="Inter", weight="bold"))
        fig.add_annotation(x=x_min + diff_x, y=y_min + diff_y, text="LAGGING", showarrow=False, font=dict(color="rgba(239,68,68,0.3)", size=26, family="Inter", weight="bold"))
        fig.add_annotation(x=x_min + diff_x, y=y_max - diff_y, text="IMPROVING", showarrow=False, font=dict(color="rgba(59,130,246,0.3)", size=26, family="Inter", weight="bold"))

        fig.update_layout(
            paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(title="Relative Strength Ratio", range=[x_min, x_max], gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
            yaxis=dict(title="Relative Strength Momentum", range=[y_min, y_max], gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0"), scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, t=20, b=0),
            height=700,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Matrix Status Summary")
        
        leading, weakening, lagging, improving = [], [], [], []
        
        for row in latest_rows:
            # Create a styled item
            item = f"<span style='color:{row['Color']};font-weight:bold;'>■</span> <b>{row['Ticker']}</b> ({row['FDTS']}) <span style='color:#94a3b8;font-size:12px;'>({row['Name']})</span>"
            quad = row['Quadrant']
            if quad == "Leading": leading.append(item)
            elif quad == "Weakening": weakening.append(item)
            elif quad == "Lagging": lagging.append(item)
            else: improving.append(item)
            
        c1, c2, c3, c4 = st.columns(4)
        
        def render_col(col, title, color, items):
            with col:
                st.markdown(
                    f"""
                    <div style="background:rgba(21,40,71,0.6); border-top:3px solid {color}; padding:15px; border-radius:8px; min-height:220px;">
                        <h4 style="color:{color}; margin-top:0; margin-bottom:12px; font-family:'Inter',sans-serif;">{title}</h4>
                        <div style="color:#e2e8f0; font-size:14px; line-height:2.0;">
                            {'<br>'.join(items) if items else '<i style="color:#64748b;">None</i>'}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
        render_col(c1, "LEADING", "#3ab54a", leading)
        render_col(c2, "WEAKENING", "#f59e0b", weakening)
        render_col(c3, "LAGGING", "#ef4444", lagging)
        render_col(c4, "IMPROVING", "#3b82f6", improving)
