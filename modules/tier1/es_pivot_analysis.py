"""
FazDane Analytics — Tier 1
ES Pivot Analysis & Volume Profile
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import logging
from io import BytesIO
from modules.base_module import FazDaneModule

logger = logging.getLogger("ESPivotAnalysis")
EASTERN_TZ = pytz.timezone("America/New_York")

@st.cache_data(ttl=900, show_spinner=False)
def fetch_es_data(days=12, interval="1h"):
    symbol = "ES=F"
    end = datetime.now(EASTERN_TZ)
    start = end - timedelta(days=days)
    
    raw = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if raw.empty:
        return pd.DataFrame()
        
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(-1)
        
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC").tz_convert(EASTERN_TZ)
    else:
        raw.index = raw.index.tz_convert(EASTERN_TZ)
        
    # De-duplicate
    raw = raw[~raw.index.duplicated(keep="last")]
    return raw

def compute_pivots(es_hourly):
    es_rth = es_hourly.between_time("09:30", "16:00")
    pivot_rows = []
    rth_dates = sorted(es_rth.index.normalize().unique())

    today_inputs = None
    today_pivot_levels = None

    for date in rth_dates:
        day_data = es_rth[es_rth.index.normalize() == date]
        if day_data.empty:
            continue

        high = day_data["High"].max().item()
        low = day_data["Low"].min().item()
        close = day_data["Close"].iloc[-1].item()

        P  = round((high + low + close) / 3, 2)
        R1 = round((2 * P) - low, 2)
        S1 = round((2 * P) - high, 2)
        R2 = round(P + (high - low), 2)
        S2 = round(P - (high - low), 2)
        R3 = round(high + 2 * (P - low), 2)
        S3 = round(low - 2 * (high - P), 2)

        pivot_date = (pd.Timestamp(date) + pd.tseries.offsets.BusinessDay(1)).strftime("%Y-%m-%d")

        row = {
            "Date": pivot_date,
            "R3": R3, "R2": R2, "R1": R1,
            "Pivot": P,
            "S1": S1, "S2": S2, "S3": S3
        }
        pivot_rows.append(row)

        today_inputs = {
            "Previous RTH Date": date.strftime("%Y-%m-%d"),
            "High": round(high, 2),
            "Low": round(low, 2),
            "Close": round(close, 2)
        }
        today_pivot_levels = row

    pivot_df = pd.DataFrame(pivot_rows).sort_values("Date").tail(5).reset_index(drop=True)
    pivot_df["Pivot_Deviation"] = pivot_df["Pivot"].diff().round(2)
    
    def arrow(val):
        if pd.isna(val): return ""
        if val > 0: return f"🟩 ↑ ({val:.2f})"
        if val < 0: return f"🟥 ↓ ({val:.2f})"
        return f"➡️ ({val:.2f})"
        
    pivot_df["Trend"] = pivot_df["Pivot_Deviation"].apply(arrow)
    return pivot_df, today_inputs, today_pivot_levels

def compute_volume_profile(es_5m, vp_days=5):
    end = datetime.now(EASTERN_TZ)
    cutoff = end - timedelta(days=vp_days)
    
    vp_rth = es_5m.between_time("09:30","16:00").dropna(subset=["Close","Volume"])
    vp_rth = vp_rth[vp_rth.index >= cutoff]
    
    if vp_rth.empty: return None
    
    prices = vp_rth["Close"].astype(float)
    volumes = vp_rth["Volume"].astype(float)

    bin_size = 2.0
    price_min, price_max = prices.min(), prices.max()
    bins = np.arange(price_min - bin_size, price_max + bin_size, bin_size)

    price_bins = pd.cut(prices, bins=bins, labels=bins[:-1], include_lowest=True)
    vp_series = volumes.groupby(price_bins).sum().dropna()
    vp_series.index = vp_series.index.astype(float)
    return vp_series.sort_index()

def compute_deviation_engine(es_5m):
    rth_dev = es_5m.between_time("09:30","16:00").dropna(subset=["Close", "Volume"])
    if rth_dev.empty: return None
    
    last_rth_day = rth_dev.index.normalize().max()
    day = rth_dev[rth_dev.index.normalize() == last_rth_day].copy()

    day["LastLow15"]  = day["Low"].rolling(3).min()
    day["LastHigh15"] = day["High"].rolling(3).max()
    day["Dev_from_LastLow"]  = day["Close"] - day["LastLow15"]
    day["Dev_from_LastHigh"] = day["Close"] - day["LastHigh15"]

    day_export = day.tz_convert("UTC")
    day_export.index = day_export.index.tz_localize(None)
    day_export = day_export.reset_index().rename(columns={"index":"Datetime"})
    return day_export, last_rth_day

class ESPivotAnalysisModule(FazDaneModule):
    MODULE_NAME = "ES Pivot Analysis"
    MODULE_ICON = "⚙️"
    MODULE_DESCRIPTION = "ES Futures Volume Profile, Pivot Confluence, and 5-Min Deviation Engine"
    TIER = 1
    SOURCE_NOTEBOOK = "ES Volume Profile and Pivot Confluence.ipynb"
    CACHE_TTL = 900
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def render_sidebar(self):
        st.markdown("**Module Controls**")
        if st.button("🔄 Refresh Data", width="stretch"):
            fetch_es_data.clear()
            st.rerun()

    def render_main(self):
        self.render_section_header("⚙️ ES Pivot Analysis & Volume Profile", "ES Futures (ES=F) rolling pivots, volume confluence, and 5-min RTH deviation engine")
        
        with st.spinner("Fetching ES=F 1h and 5m data..."):
            es_hourly = fetch_es_data(days=12, interval="1h")
            es_5m = fetch_es_data(days=7, interval="5m")
            
        if es_hourly.empty or es_5m.empty:
            st.error("Could not fetch ES=F data. Please try again later.")
            return

        # 1. Pivot Engine
        pivot_df, today_inputs, today_pivot_levels = compute_pivots(es_hourly)
        
        st.markdown("### 📌 Rolling 5-Day Pivot Table")
        if today_inputs:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Previous RTH Date", today_inputs["Previous RTH Date"])
            c2.metric("Previous High", f"{today_inputs['High']:,.2f}")
            c3.metric("Previous Low", f"{today_inputs['Low']:,.2f}")
            c4.metric("Previous Close", f"{today_inputs['Close']:,.2f}")
            
        # Style dataframe for streamlit
        styled_df = pivot_df.style.apply(
            lambda row: ["background-color: rgba(59, 130, 246, 0.2); font-weight: bold;" if row.name == len(pivot_df)-1 else "" for _ in row], 
            axis=1
        ).format({c: "{:.2f}" for c in ["R3","R2","R1","Pivot","S1","S2","S3","Pivot_Deviation"]})
        
        st.dataframe(styled_df, width="stretch", hide_index=True)

        st.divider()

        # 2. Volume Profile & Confluence
        vp_series = compute_volume_profile(es_5m)
        if vp_series is not None and today_pivot_levels:
            st.markdown("### 📊 ES Volume Profile + Pivot Confluence")
            
            fig_vp = go.Figure()
            fig_vp.add_trace(go.Bar(
                x=vp_series.values, y=vp_series.index, orientation='h',
                marker=dict(color="rgba(148, 163, 184, 0.5)"), name="Volume"
            ))
            
            pivot_colors = {"Pivot": "#3b82f6", "R": "#ef4444", "S": "#10b981"}
            for name, level in today_pivot_levels.items():
                if name in ["Date"]: continue
                if vp_series.index.min() <= level <= vp_series.index.max():
                    color = pivot_colors["Pivot"] if name == "Pivot" else pivot_colors["R"] if "R" in name else pivot_colors["S"]
                    fig_vp.add_hline(y=level, line_color=color, line_width=2, opacity=0.8, annotation_text=f"{name} ({level:.2f})", annotation_position="right", annotation_font_color=color)

            fig_vp.update_layout(
                paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
                font=dict(color="#e2e8f0"), height=600,
                xaxis=dict(title="Volume", gridcolor="#1e3a5f", tickfont=dict(color="#94a3b8"), title_font=dict(color="#e2e8f0")),
                yaxis=dict(title="Price", gridcolor="#1e3a5f", tickfont=dict(color="#94a3b8"), title_font=dict(color="#e2e8f0")),
                showlegend=False, margin=dict(l=0, r=40, t=20, b=0)
            )
            st.plotly_chart(fig_vp, width="stretch")
            
        st.divider()

        # 3. 5-Min Deviation Engine
        dev_res = compute_deviation_engine(es_5m)
        if dev_res:
            dev_df, last_rth_day = dev_res
            st.markdown(f"### ⏱️ ES 5-Min RTH Deviation Engine")
            st.caption(f"Last Full RTH Session: {last_rth_day.strftime('%Y-%m-%d')}")
            
            fig_dev = go.Figure()
            fig_dev.add_trace(go.Scatter(
                x=dev_df["Datetime"], y=dev_df["Close"], mode="lines+markers", name="Close",
                line=dict(color="#f59e0b", width=2), marker=dict(size=4),
                customdata=dev_df[["Volume", "Dev_from_LastLow", "Dev_from_LastHigh"]],
                hovertemplate="<b>Time:</b> %{x}<br><b>Price:</b> %{y:.2f}<br><b>Vol:</b> %{customdata[0]:,.0f}<br><b>Dev Low:</b> %{customdata[1]:.2f}<br><b>Dev High:</b> %{customdata[2]:.2f}<extra></extra>"
            ))
            
            fig_dev.update_layout(
                paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
                font=dict(color="#e2e8f0"), height=400,
                xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#94a3b8")),
                yaxis=dict(title="Price", gridcolor="#1e3a5f", tickfont=dict(color="#94a3b8"), title_font=dict(color="#e2e8f0")),
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig_dev, width="stretch")

            # Excel Download
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                dev_df.to_excel(writer, index=False, sheet_name="ES_Deviations")
            
            st.download_button(
                label="📥 Download Deviation Data (Excel)",
                data=excel_buffer.getvalue(),
                file_name=f"ES_5min_deviation_{last_rth_day.strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
