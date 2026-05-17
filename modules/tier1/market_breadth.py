"""
FazDane Analytics — Tier 1
Market Breadth Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import logging
from modules.base_module import FazDaneModule

logger = logging.getLogger("MarketBreadth")

# Using S&P 100 proxies for breadth calculations
SP100_SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "XOM", "HD", "MA", "CVX", "ABBV", "MRK", "PEP",
    "KO", "AVGO", "COST", "MCD", "TMO", "CSCO", "PFE", "CRM", "BAC", "WMT",
    "ABT", "LIN", "DIS", "ACN", "TXN", "DHR", "NFLX", "AMD", "CMCSA", "ADBE",
    "WFC", "PM", "NKE", "RTX", "HON", "UNP", "INTC", "QCOM", "BMY", "SPGI",
    "INTU", "COP", "CAT", "BA", "IBM", "AMAT", "GE", "NOW", "ISRG", "GS",
    "T", "PLD", "BLK", "SYK", "MDT", "TJX", "AXP", "C", "LMT", "ZTS",
    "EL", "MDLZ", "CB", "GILD", "ADI", "MMC", "CVS", "CI", "BDX", "SLB",
    "EOG", "PGR", "MO", "VRTX", "SO", "REGN", "TGT", "BSX", "CME", "PYPL",
    "DUK", "NOC", "AON", "KLAC", "ITW", "WM", "CSX", "EW", "HUM", "FCX"
]

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
    "XLC": "Communication"
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(lookback_days: int):
    # Need 1 year (252 trading days) of data PRIOR to the lookback window for 52-week highs/lows
    end_date = datetime.today()
    # 365 calendar days = ~252 trading days. Add buffer.
    start_date = end_date - timedelta(days=lookback_days + 380)
    
    tickers = SP100_SYMBOLS + list(SECTORS.keys()) + ["SPY"]
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        close_df = data["Close"].ffill()
        high_df = data["High"].ffill()
        low_df = data["Low"].ffill()
        return close_df, high_df, low_df
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def calculate_breadth(close_df, high_df, low_df, lookback_days):
    # Filter to SP100
    sp100_close = close_df[SP100_SYMBOLS].dropna(axis=1, how='all')
    sp100_high = high_df[SP100_SYMBOLS].dropna(axis=1, how='all')
    sp100_low = low_df[SP100_SYMBOLS].dropna(axis=1, how='all')
    
    # Daily returns
    returns = sp100_close.pct_change()
    
    advances = (returns > 0).sum(axis=1)
    declines = (returns < 0).sum(axis=1)
    
    ad_diff = advances - declines
    ad_line = ad_diff.cumsum()
    
    # McClellan Oscillator
    ema19 = ad_diff.ewm(span=19, adjust=False).mean()
    ema39 = ad_diff.ewm(span=39, adjust=False).mean()
    mcclellan = ema19 - ema39
    
    # New Highs / New Lows (52-week = 252 trading days)
    rolling_high = sp100_high.rolling(window=252).max()
    rolling_low = sp100_low.rolling(window=252).min()
    
    new_highs = (sp100_high >= rolling_high).sum(axis=1)
    new_lows = (sp100_low <= rolling_low).sum(axis=1)
    
    df = pd.DataFrame({
        "Advances": advances,
        "Declines": declines,
        "AD_Line": ad_line,
        "McClellan": mcclellan,
        "New_Highs": new_highs,
        "New_Lows": new_lows,
        "SPY_Close": close_df["SPY"]
    }).iloc[-lookback_days:]
    
    return df

def calculate_sector_breadth(close_df, lookback_days):
    sector_tickers = list(SECTORS.keys())
    sector_close = close_df[sector_tickers].iloc[-lookback_days-250:] # Need 200 days for SMA200
    
    results = []
    
    for ticker in sector_tickers:
        prices = sector_close[ticker]
        sma20 = prices.rolling(20).mean()
        sma50 = prices.rolling(50).mean()
        sma200 = prices.rolling(200).mean()
        
        current_price = prices.iloc[-1]
        
        pct_from_20 = ((current_price - sma20.iloc[-1]) / sma20.iloc[-1]) * 100
        pct_from_50 = ((current_price - sma50.iloc[-1]) / sma50.iloc[-1]) * 100
        pct_from_200 = ((current_price - sma200.iloc[-1]) / sma200.iloc[-1]) * 100
        
        results.append({
            "Sector": SECTORS[ticker],
            "Ticker": ticker,
            "Price": round(current_price, 2),
            "SMA20_Dist": round(pct_from_20, 2),
            "SMA50_Dist": round(pct_from_50, 2),
            "SMA200_Dist": round(pct_from_200, 2)
        })
        
    return pd.DataFrame(results).sort_values("SMA20_Dist", ascending=False)

class MarketBreadthModule(FazDaneModule):
    MODULE_NAME = "Market Breadth Dashboard"
    MODULE_ICON = "📊"
    MODULE_DESCRIPTION = "A/D line, McClellan Oscillator, and Sector Breadth"
    TIER = 1
    SOURCE_NOTEBOOK = "Market_Breadth.ipynb"
    CACHE_TTL = 3600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def render_sidebar(self):
        st.markdown("**Parameters**")
        lookback = st.slider("Lookback Window (Days)", 30, 252, 90, 30, key="mb_lookback_widget")
        
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        scan_clicked = st.button("📊 Calculate Breadth", use_container_width=True, type="primary", key="mb_calc")
        
        if scan_clicked:
            st.session_state["mb_lookback_val"] = lookback
            st.session_state.pop("mb_results", None)

    def render_main(self):
        self.render_section_header(
            "📊 Market Breadth Dashboard",
            "Monitor market internals, advance/decline metrics, and sector momentum"
        )
        
        lookback = st.session_state.get("mb_lookback_val", 90)
        
        with st.spinner("Fetching market data and calculating breadth metrics..."):
            close_df, high_df, low_df = fetch_market_data(lookback)
            
        if close_df.empty:
            st.error("Failed to fetch market data.")
            return
            
        breadth_df = calculate_breadth(close_df, high_df, low_df, lookback)
        sector_df = calculate_sector_breadth(close_df, lookback)
        
        # Latest metrics
        latest = breadth_df.iloc[-1]
        prev = breadth_df.iloc[-2]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("S&P 100 Advances", f"{int(latest['Advances'])}", f"{int(latest['Advances'] - prev['Advances'])}")
        m2.metric("S&P 100 Declines", f"{int(latest['Declines'])}", f"{int(latest['Declines'] - prev['Declines'])}")
        m3.metric("McClellan Oscillator", f"{latest['McClellan']:.2f}", f"{latest['McClellan'] - prev['McClellan']:.2f}")
        m4.metric("New 52-Wk Highs", f"{int(latest['New_Highs'])}", f"{int(latest['New_Highs'] - prev['New_Highs'])}")
        
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["📈 A/D & McClellan", "🔥 Sector Heatmap", "📊 New Highs/Lows"])
        
        with tab1:
            self._render_ad_mcclellan(breadth_df)
            
        with tab2:
            self._render_sector_heatmap(sector_df)
            
        with tab3:
            self._render_highs_lows(breadth_df)
            
    def _render_ad_mcclellan(self, df):
        # A/D Line Chart overlaid with SPY
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=df.index, y=df['SPY_Close'], name='SPY',
            line=dict(color='#e2e8f0', width=2), yaxis='y1'
        ))
        
        fig1.add_trace(go.Scatter(
            x=df.index, y=df['AD_Line'], name='A/D Line',
            line=dict(color='#3ab54a', width=2), yaxis='y2'
        ))
        
        fig1.update_layout(
            title=dict(text="Advance/Decline Line vs SPY", font=dict(color="#e2e8f0")),
            paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
            yaxis=dict(title="SPY Price", gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
            yaxis2=dict(title="A/D Line", overlaying='y', side='right', tickfont=dict(color="#3ab54a")),
            legend=dict(bgcolor="rgba(21,40,71,0.85)", bordercolor="#1e3a5f", borderwidth=1, font=dict(color="#e2e8f0", size=12)),
            margin=dict(l=0, r=0, t=50, b=0), height=380
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # McClellan Oscillator Chart
        fig2 = go.Figure()
        colors = ['#3ab54a' if val >= 0 else '#ef4444' for val in df['McClellan']]
        
        fig2.add_trace(go.Bar(
            x=df.index, y=df['McClellan'], name='McClellan',
            marker_color=colors
        ))
        
        fig2.update_layout(
            title=dict(text="McClellan Oscillator", font=dict(color="#e2e8f0")),
            paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
            yaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
            legend=dict(bgcolor="rgba(21,40,71,0.85)", bordercolor="#1e3a5f", borderwidth=1, font=dict(color="#e2e8f0", size=12)),
            margin=dict(l=0, r=0, t=50, b=0), height=300
        )
        st.plotly_chart(fig2, use_container_width=True)

    def _render_sector_heatmap(self, df):
        st.markdown("### Sector Breadth & Momentum")
        st.markdown("Distance from moving averages indicating sector strength.")
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Sector", "Ticker", "Price", "Dist from 20-SMA", "Dist from 50-SMA", "Dist from 200-SMA"],
                fill_color='#152847',
                align='left',
                font=dict(color='#94a3b8', size=12)
            ),
            cells=dict(
                values=[
                    df['Sector'], 
                    df['Ticker'], 
                    df['Price'].apply(lambda x: f"${x:.2f}"),
                    df['SMA20_Dist'].apply(lambda x: f"{x:.2f}%"),
                    df['SMA50_Dist'].apply(lambda x: f"{x:.2f}%"),
                    df['SMA200_Dist'].apply(lambda x: f"{x:.2f}%")
                ],
                fill_color='#0d1b2e',
                align='left',
                font=dict(
                    color=[
                        '#e2e8f0', '#e2e8f0', '#e2e8f0',
                        ['#3ab54a' if val >= 0 else '#ef4444' for val in df['SMA20_Dist']],
                        ['#3ab54a' if val >= 0 else '#ef4444' for val in df['SMA50_Dist']],
                        ['#3ab54a' if val >= 0 else '#ef4444' for val in df['SMA200_Dist']]
                    ], 
                    size=12
                ),
                height=30
            )
        )])
        
        fig.update_layout(
            paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
            margin=dict(l=0, r=0, t=0, b=0), height=450
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_highs_lows(self, df):
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df.index, y=df['New_Highs'], name='New 52w Highs',
            marker_color='#3ab54a', opacity=0.8
        ))
        
        fig.add_trace(go.Bar(
            x=df.index, y=-df['New_Lows'], name='New 52w Lows',
            marker_color='#ef4444', opacity=0.8
        ))
        
        fig.update_layout(
            title=dict(text="New 52-Week Highs vs Lows (S&P 100)", font=dict(color="#e2e8f0")),
            barmode='relative',
            paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
            yaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
            legend=dict(bgcolor="rgba(21,40,71,0.85)", bordercolor="#1e3a5f", borderwidth=1, font=dict(color="#e2e8f0", size=12)),
            margin=dict(l=0, r=0, t=50, b=0), height=380
        )
        st.plotly_chart(fig, use_container_width=True)
