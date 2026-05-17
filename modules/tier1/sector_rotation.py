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

logger = logging.getLogger("SectorRotation")

# ============================================================
# Universe Definitions
# ============================================================

UNIVERSES = {
    "SPX Sectors": {
        "tickers": {
            "XLC": "Communication Services", "XLY": "Consumer Discretionary", 
            "XLP": "Consumer Staples", "XLE": "Energy", "XLF": "Financials", 
            "XLV": "Health Care", "XLI": "Industrials", "XLB": "Materials", 
            "XLRE": "Real Estate", "XLK": "Technology", "XLU": "Utilities"
        },
        "benchmark": "SPY",
        "caption": "SPX Sector Rotation Matrix vs SPY"
    },
    "MAG 7": {
        "tickers": {
            "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", 
            "AMZN": "Amazon", "META": "Meta Platforms", "GOOGL": "Alphabet", 
            "TSLA": "Tesla"
        },
        "benchmark": "QQQ",
        "caption": "MAG 7 Rotation Matrix vs QQQ"
    },
    "Leading ETFs": {
        "tickers": {
            "QQQ": "Nasdaq 100 ETF", "SPY": "S&P 500 ETF", "IWM": "Russell 2000 ETF", 
            "DIA": "Dow Jones ETF", "SMH": "Semiconductor ETF", "XLK": "Technology ETF", 
            "XLF": "Financial ETF", "XLE": "Energy ETF", "GLD": "Gold ETF", 
            "SLV": "Silver ETF", "TLT": "Long Bond ETF", "HYG": "High Yield Bond ETF"
        },
        "benchmark": "SPY",
        "caption": "Leading ETF Rotation Matrix vs SPY"
    },
    "Major Indexes": {
        "tickers": {
            "^GSPC": "S&P 500", "^IXIC": "Nasdaq Composite", "^DJI": "Dow Jones", 
            "^RUT": "Russell 2000", "^NYA": "NYSE Composite", "^VIX": "Volatility Index"
        },
        "benchmark": "^GSPC",
        "caption": "Major Index Rotation Matrix vs S&P 500"
    }
}

# Fixed distinct colors for Plotly
COLORS = ["#06b6d4", "#3b82f6", "#10b981", "#f59e0b", "#6366f1", "#ec4899", "#8b5cf6", "#14b8a6", "#f97316", "#84cc16", "#ef4444", "#a855f7", "#fbbf24", "#34d399", "#f87171"]

def get_color(idx):
    return COLORS[idx % len(COLORS)]

# ============================================================
# Logic functions
# ============================================================

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
        return pd.DataFrame()

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
    return prices

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
        
        # State management to update benchmark when universe changes
        def on_universe_change():
            univ = st.session_state.sr_univ_sel
            if univ != "Custom Tickers":
                st.session_state.sr_bench_input = UNIVERSES[univ]["benchmark"]
            else:
                st.session_state.sr_bench_input = "SPY"
        
        universe = st.selectbox(
            "Universe:", 
            options=list(UNIVERSES.keys()) + ["Custom Tickers"], 
            index=0, 
            key="sr_univ_sel",
            on_change=on_universe_change
        )
        
        custom_tickers = ""
        if universe == "Custom Tickers":
            custom_tickers = st.text_area("Custom Tickers:", "AAPL, MSFT, NVDA, AMD, META, TSLA, GOOGL", key="sr_custom")
            
        benchmark = st.text_input(
            "Benchmark:", 
            value=UNIVERSES.get(universe, {}).get("benchmark", "SPY"), 
            key="sr_bench_input"
        )
        
        st.markdown("**Chart Parameters**")
        period_type = st.selectbox("Period Type:", ["days", "weeks", "months"], index=1, key="sr_ptype")
        period_number = st.slider("Periods:", 3, 52, 12, 1, key="sr_pnum")
        tail_len = st.slider("Tail Length:", 3, 15, 6, 1, key="sr_tail")
        ema_span = st.slider("Smooth (EMA Span):", 2, 10, 4, 1, key="sr_ema")
        
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        scan_clicked = st.button("🔄 Generate Matrix", use_container_width=True, type="primary")
        
        if scan_clicked:
            st.session_state["sr_state"] = {
                "universe": universe,
                "custom_tickers": custom_tickers,
                "benchmark": benchmark.strip().upper(),
                "period_type": period_type,
                "period_number": period_number,
                "tail_length": tail_len,
                "ema_span": ema_span
            }

    def render_main(self):
        state = st.session_state.get("sr_state", {
            "universe": "SPX Sectors",
            "custom_tickers": "",
            "benchmark": "SPY",
            "period_type": "weeks",
            "period_number": 12,
            "tail_length": 6,
            "ema_span": 4
        })
        
        # Resolve Universe
        if state["universe"] == "Custom Tickers":
            t_list = [t.strip().upper() for t in state["custom_tickers"].replace("\n", ",").split(",") if t.strip()]
            ticker_dict = {t: t for t in t_list}
            caption = f"Custom Rotation Matrix vs {state['benchmark']}"
        else:
            ticker_dict = UNIVERSES[state["universe"]]["tickers"]
            caption = UNIVERSES[state["universe"]]["caption"]

        self.render_section_header(
            "🔄 " + caption,
            f"Tail Length: {state['tail_length']} {state['period_type']} | Smoothing: {state['ema_span']} EMA"
        )

        with st.spinner("Fetching data and computing matrix..."):
            try:
                prices = download_rotation_data(
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
        
        self._render_plot(rs_ratio_df, rs_momentum_df, ticker_dict, tail_length)

    def _render_plot(self, ratio_df, mom_df, ticker_dict, tail_length):
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
                hovertemplate=f"<b>{asset_name} ({ticker})</b><br>RS-Ratio: %{{x:.2f}}<br>RS-Mom: %{{y:.2f}}<extra></extra>"
            ))
            
            quad = get_quadrant(latest_x, latest_y)
            latest_rows.append({
                "Ticker": ticker,
                "Name": asset_name,
                "RS Ratio": latest_x,
                "RS Momentum": latest_y,
                "Quadrant": quad,
                "Color": color
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
            item = f"<span style='color:{row['Color']};font-weight:bold;'>■</span> <b>{row['Ticker']}</b> <span style='color:#94a3b8;font-size:12px;'>({row['Name']})</span>"
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
