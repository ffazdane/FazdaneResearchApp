"""
FazDane Analytics — Tier 1
Iron Condor Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import yfinance as yf
from datetime import datetime, timedelta
import logging
from modules.base_module import FazDaneModule
from utils.universe_manager import render_universe_manager

logger = logging.getLogger("IronCondor")

# ================================================================
#  CONFIGURATION
# ================================================================

INDICES = {"SPX": "^GSPC", "NDX": "^NDX", "RUT": "^RUT", "VIX": "^VIX", "DJIA": "^DJI"}
MAG7 = {"NVDA": "NVDA", "AAPL": "AAPL", "MSFT": "MSFT", "AMZN": "AMZN", "META": "META", "GOOGL": "GOOGL", "TSLA": "TSLA"}
CUSTOM_TICKERS = {"AVGO": "AVGO", "NFLX": "NFLX", "AMD": "AMD", "SPY": "SPY", "QQQ": "QQQ", "IWM": "IWM"}
ALL_MAP = {**INDICES, **MAG7, **CUSTOM_TICKERS}

def _label_for_ticker(ticker):
    for label, mapped in ALL_MAP.items():
        if mapped == ticker:
            return label
    return ticker

def _vix_info(v):
    if v < 15: return "Low", "#10b981"
    if v < 20: return "Normal", "#10b981"
    if v < 25: return "Elevated", "#f59e0b"
    if v < 30: return "High", "#ef4444"
    return "Extreme Fear", "#ef4444"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_data(days, ticker_items):
    ticker_map = dict(ticker_items)
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days + 15)
    tickers = sorted(set(list(ticker_map.values()) + ["^VIX"]))

    raw = yf.download(tickers, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), auto_adjust=True, progress=False, threads=True)
    if raw.empty: return {}

    single = len(tickers) == 1
    result = {}
    for label, ticker in ticker_map.items():
        try:
            close = (raw["Close"].dropna() if single else raw["Close"][ticker].dropna()).tail(days)
            if close.empty: continue
            p = float(close.iloc[-1])
            p0 = float(close.iloc[0])
            pprev = float(close.iloc[-2]) if len(close) > 1 else p
            result[ticker] = {
                "label": label, "close": close, "price": p, "p0": p0, "pprev": pprev,
                "pct": (p - p0) / p0 * 100, "dpct": (p - pprev) / pprev * 100,
                "high": float(close.max()), "low": float(close.min()),
            }
        except Exception:
            pass
    return result

def _rs(v, price):
    if price > 5000: return round(v / 5) * 5
    if price > 1000: return round(v / 2) * 2
    if price > 200:  return round(v)
    return round(v, 2)

class IronCondorModule(FazDaneModule):
    MODULE_NAME = "Iron Condor Analyzer"
    MODULE_ICON = "🦅"
    MODULE_DESCRIPTION = "Advanced Iron Condor strategy builder and payoff visualization"
    TIER = 1
    SOURCE_NOTEBOOK = "IRON CONDOR DASHBOARD v11"
    CACHE_TTL = 3600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def render_sidebar(self):
        st.markdown("""
        <style>
        /* Match Iron Condor number inputs to the app sidebar theme */
        [data-testid="stSidebar"] .stNumberInput,
        [data-testid="stSidebar"] .stNumberInput * {
            color: #e2e8f0 !important;
            -webkit-text-fill-color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] .stNumberInput label,
        [data-testid="stSidebar"] .stNumberInput label *,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #cbd5e1 !important;
            -webkit-text-fill-color: #cbd5e1 !important;
        }
        [data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"],
        [data-testid="stSidebar"] .stNumberInput div[data-baseweb="base-input"] {
            background: rgba(21,40,71,0.9) !important;
            border-color: #1e3a5f !important;
        }
        input[type="number"] {
            color: #e2e8f0 !important;
            -webkit-text-fill-color: #e2e8f0 !important;
            font-weight: 600 !important;
            background: rgba(21,40,71,0.9) !important;
        }
        input[type="number"]::placeholder {
            color: #94a3b8 !important;
            -webkit-text-fill-color: #94a3b8 !important;
        }
        div[data-baseweb="base-input"] svg {
            fill: #e2e8f0 !important;
            color: #e2e8f0 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("**Data Window**")
        days = st.slider("Lookback Days:", 7, 365, 30, 1, key="ic_days")
        
        st.markdown("**Strategy Builder**")
        universe_name, tickers_list, _ = render_universe_manager(
            key_prefix="ic",
            show_benchmark=False,
            label="Ticker Universe:",
        )
        ticker_map = {_label_for_ticker(ticker): ticker for ticker in tickers_list if ticker != "^VIX"}
        if not ticker_map:
            ticker_map = {label: ticker for label, ticker in ALL_MAP.items() if ticker != "^VIX"}
        st.session_state["ic_ticker_map"] = ticker_map
        st.caption(f"{len(ticker_map)} instruments selected from {universe_name}.")

        tradeable = sorted([(f"{label} ({t})", t) for label, t in ticker_map.items()])
        ticker = st.selectbox("Instrument:", options=[t for _, t in tradeable], format_func=lambda x: [l for l, t in tradeable if t == x][0], index=0, key="ic_ticker")
        
        bias = st.selectbox("Market Bias:", ["Neutral", "Bullish", "Bearish"], index=0, key="ic_bias").lower()
        dte = st.number_input("Target DTE:", min_value=7, max_value=90, value=30, key="ic_dte")
        
        st.markdown("**Leg Parameters (%)**")
        put_otm = st.number_input("Short Put OTM %:", min_value=1.0, max_value=20.0, value=5.0, step=0.5, key="ic_put")
        call_otm = st.number_input("Short Call OTM %:", min_value=1.0, max_value=20.0, value=4.0, step=0.5, key="ic_call")
        wing_pct = st.number_input("Wing Width %:", min_value=0.25, max_value=10.0, value=1.0, step=0.25, key="ic_wing")
        stop_pct = st.number_input("Stop Loss %:", min_value=50.0, max_value=400.0, value=200.0, step=25.0, key="ic_stop")

    def render_main(self):
        state = st.session_state
        days = state.get("ic_days", 30)
        ticker = state.get("ic_ticker", "^GSPC")
        ticker_map = state.get("ic_ticker_map", {label: ticker for label, ticker in ALL_MAP.items() if ticker != "^VIX"})
        
        self.render_section_header("🦅 Iron Condor Dashboard", f"{days}-Day Window & Options Strategy Monitor")
        
        with st.spinner(f"Fetching {days}-day data..."):
            DATA = fetch_all_data(days, tuple(sorted(ticker_map.items())))
            
        if not DATA:
            st.error("Failed to load data. Please try again.")
            return

        self._render_performance_cards(DATA, days)
        st.divider()
        self._render_calculator(DATA, ticker, state)

    def _render_performance_cards(self, DATA, days):
        st.markdown("### Market Overview")
        
        idx_tks = [t for t in INDICES.values() if t in DATA]
        if not idx_tks:
            idx_tks = list(DATA.keys())[:5]
            st.caption("No index symbols found in the selected universe. Showing selected instruments instead.")
        if not idx_tks:
            st.warning("No instruments are available for the market overview.")
            return
        cols = st.columns(len(idx_tks))
        
        for i, t in enumerate(idx_tks):
            d = DATA[t]
            pc, dc = d["pct"], d["dpct"]
            col_color = "normal" if pc >= 0 else "inverse"
            sub = f"1d: {dc:+.2f}%"
            if t == "^VIX":
                lbl, _ = _vix_info(d["price"])
                sub += f" | {lbl}"
            
            with cols[i]:
                st.metric(f"{d['label']} ({t})", f"{d['price']:,.2f}", f"{pc:+.2f}% ({days}d)", delta_color=col_color)
                st.caption(sub)
                
        # Line Chart
        fig = go.Figure()
        colors = {"^GSPC": "#3b82f6", "^NDX": "#10b981", "^RUT": "#ef4444", "^DJI": "#8b5cf6", "^VIX": "#f59e0b"}
        for t in idx_tks:
            if t == "^VIX": continue
            d = DATA[t]
            norm = (d["close"] / d["close"].iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(x=norm.index, y=norm.values, mode='lines', name=d['label'], line=dict(color=colors.get(t, "#94a3b8"), width=2)))
        if not fig.data:
            return
            
        fig.update_layout(
            title=dict(text="Index Relative Performance", font=dict(color="#e2e8f0")),
            paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
            font=dict(color="#e2e8f0"), margin=dict(l=0, r=0, t=40, b=0),
            height=300, 
            yaxis=dict(title=dict(text="% Change", font=dict(color="#e2e8f0")), gridcolor="#1e3a5f", tickfont=dict(color="#94a3b8")), 
            xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#94a3b8")),
            legend=dict(font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_calculator(self, DATA, ticker, state):
        d = DATA.get(ticker)
        if not d:
            st.warning(f"No data for {ticker}")
            return
            
        price = d["price"]
        put_otm = state.get("ic_put", 5.0)
        call_otm = state.get("ic_call", 4.0)
        wing_pct = state.get("ic_wing", 1.0)
        bias = state.get("ic_bias", "neutral")
        stop_pct = state.get("ic_stop", 200.0)
        dte = state.get("ic_dte", 30)
        
        pa, ca = ((put_otm * 0.85, call_otm * 1.20) if bias == "bearish" else (put_otm * 1.20, call_otm * 0.85) if bias == "bullish" else (put_otm, call_otm))

        sp_raw = price * (1 - pa/100);       lp_raw = price * (1 - pa/100 - wing_pct/100)
        sc_raw = price * (1 + ca/100);       lc_raw = price * (1 + ca/100 + wing_pct/100)
        sp = _rs(sp_raw, price);  lp = _rs(lp_raw, price)
        sc = _rs(sc_raw, price);  lc = _rs(lc_raw, price)

        put_w = sp - lp;    call_w = lc - sc
        max_r = max(put_w, call_w)
        vix_v = float(DATA.get("^VIX", {}).get("price", 20.0))
        cred = max_r * 0.28 * (vix_v / 20.0)
        net_r = max_r - cred
        stop = cred * (stop_pct / 100.0)
        tgt = cred * 0.50
        be_lo = sp - cred;  be_hi = sc + cred
        prob = max(5, min(85, 70 - abs(pa-5)*1.5 - abs(ca-4)*1.2))

        st.markdown(f"### Strategy Builder: {ticker} @ ${price:,.2f}")
        st.caption(f"DTE {dte} | {bias.title()} | VIX {vix_v:.1f}")

        # Legs Table
        legs_df = pd.DataFrame([
            {"Leg": "Long Call (buy)", "Strike": lc, "OTM": f"{(lc_raw/price-1)*100:+.1f}%"},
            {"Leg": "Short Call (SELL)", "Strike": sc, "OTM": f"{(sc_raw/price-1)*100:+.1f}%"},
            {"Leg": "— Current Price —", "Strike": price, "OTM": ""},
            {"Leg": "Short Put (SELL)", "Strike": sp, "OTM": f"{(sp_raw/price-1)*100:+.1f}%"},
            {"Leg": "Long Put (buy)", "Strike": lp, "OTM": f"{(lp_raw/price-1)*100:+.1f}%"},
        ])
        
        st.dataframe(legs_df, use_container_width=True, hide_index=True)

        st.info(f"⚠️ Credit estimate scaled to VIX {vix_v:.1f}. Verify bid/ask in your broker before entry. Not financial advice.")

        # Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Est. Credit", f"{cred:,.1f} pts")
        c2.metric("Max Risk", f"{net_r:,.1f} pts")
        c3.metric("50% Profit Target", f"{tgt:,.1f} pts")
        c4.metric("Stop-Loss", f"{stop:,.1f} pts")
        
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Breakeven Low", f"{be_lo:,.1f}")
        c6.metric("Breakeven High", f"{be_hi:,.1f}")
        c7.metric("Prob. Profit (est)", f"~{prob:.0f}%")
        c8.metric("Wing Width (P/C)", f"{put_w:,.1f} / {call_w:,.1f}")

        # Payoff Diagram
        st.markdown("### Payoff at Expiry")
        xr = np.linspace(lp * 0.94, lc * 1.06, 600)
        pnl = cred - (np.maximum(sp-xr,0) - np.maximum(lp-xr,0)) - (np.maximum(xr-sc,0) - np.maximum(xr-lc,0))
        
        fig2 = go.Figure()
        
        # Fill positive
        fig2.add_trace(go.Scatter(x=xr[pnl>=0], y=pnl[pnl>=0], fill='tozeroy', mode='lines', line=dict(color="#10b981", width=0), fillcolor="rgba(16,185,129,0.3)", showlegend=False))
        # Fill negative
        fig2.add_trace(go.Scatter(x=xr[pnl<0], y=pnl[pnl<0], fill='tozeroy', mode='lines', line=dict(color="#ef4444", width=0), fillcolor="rgba(239,68,68,0.3)", showlegend=False))
        # Pnl Line
        fig2.add_trace(go.Scatter(x=xr, y=pnl, mode='lines', name='P&L', line=dict(color="#3b82f6", width=3)))
        
        # Markers
        for val, label, col in [(price, "Now", "#f59e0b"), (sp, "SP", "#ef4444"), (sc, "SC", "#10b981"), (be_lo, "BE", "#94a3b8"), (be_hi, "BE", "#94a3b8")]:
            fig2.add_vline(x=val, line_width=1, line_dash="dash", line_color=col)
            fig2.add_annotation(x=val, y=max(pnl)*0.9, text=f"{label}<br>{val:,.0f}", showarrow=False, font=dict(color=col, size=11), yshift=10)

        fig2.add_hline(y=0, line_width=1, line_color="#94a3b8")
        fig2.add_hline(y=tgt, line_width=1, line_dash="dot", line_color="#10b981", annotation_text=f"50% Target ({tgt:.1f})")

        fig2.update_layout(
            paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", font=dict(color="#e2e8f0"),
            margin=dict(l=0, r=0, t=20, b=0), height=400,
            yaxis=dict(title=dict(text="P&L (points)", font=dict(color="#e2e8f0")), gridcolor="#1e3a5f", tickfont=dict(color="#94a3b8")), 
            xaxis=dict(title=dict(text="Underlying Price", font=dict(color="#e2e8f0")), gridcolor="#1e3a5f", tickfont=dict(color="#94a3b8")),
            legend=dict(font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Volatility
        closes = d["close"]
        if len(closes) >= 10:
            st.markdown("### Realized Vol vs VIX")
            rets = closes.pct_change().dropna()
            rv10 = rets.rolling(10).std() * np.sqrt(252) * 100
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=rv10.index, y=rv10.values, fill='tozeroy', mode='lines', name="10d Realized Vol (ann.)", line=dict(color="#8b5cf6", width=2), fillcolor="rgba(139,92,246,0.15)"))
            
            if "^VIX" in DATA:
                vix_s = DATA["^VIX"]["close"].reindex(rv10.index, method="nearest")
                fig3.add_trace(go.Scatter(x=vix_s.index, y=vix_s.values, mode='lines', name="VIX", line=dict(color="#f59e0b", width=2, dash="dash")))

            fig3.update_layout(
                paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", font=dict(color="#e2e8f0"),
                margin=dict(l=0, r=0, t=20, b=0), height=300,
                yaxis=dict(title=dict(text="Ann. Vol %", font=dict(color="#e2e8f0")), gridcolor="#1e3a5f", tickfont=dict(color="#94a3b8")), 
                xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#94a3b8")),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#e2e8f0"))
            )
            st.plotly_chart(fig3, use_container_width=True)
