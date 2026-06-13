"""
FazDane Analytics - Macro Intelligence Dashboard

Streamlit-native version of the Colab macro dashboard. Uses Yahoo Finance for
market proxies and FRED when FRED_API_KEY is available in the environment.
"""

import os
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


MARKET_PROXIES = {
    "oil": ("CL=F", "WTI Crude Oil"),
    "vix": ("^VIX", "VIX"),
    "spx": ("^GSPC", "S&P 500"),
    "iwm": ("IWM", "Russell 2000 ETF"),
    "tlt": ("TLT", "20+ Yr Treasury ETF"),
    "hyg": ("HYG", "High Yield ETF"),
    "lqd": ("LQD", "Investment Grade Credit ETF"),
    "dxy": ("DX-Y.NYB", "Dollar Index"),
    "gold": ("GC=F", "Gold"),
    "btc": ("BTC-USD", "Bitcoin"),
    "dbc": ("DBC", "Commodity Index"),
}

FRED_SERIES = {
    "dgs10": ("DGS10", "US 10Y Treasury"),
    "dgs2": ("DGS2", "US 2Y Treasury"),
    "t5yie": ("T5YIE", "5Y Breakeven Inflation"),
    "walcl": ("WALCL", "Fed Balance Sheet"),
    "m2sl": ("M2SL", "M2 Money Supply"),
    "totalsl": ("TOTALSL", "Consumer Credit Outstanding"),
    "unrate": ("UNRATE", "Unemployment Rate"),
    "hyspread": ("BAMLH0A0HYM2", "High Yield OAS"),
    "claims": ("ICSA", "Initial Jobless Claims"),
    "corepce": ("DPCCRV1Q225SBEA", "Core PCE YoY"),
}

GREEN = "#22c55e"
YELLOW = "#facc15"
ORANGE = "#f97316"
RED = "#ef4444"
BLUE = "#38bdf8"
BG = "#0d1b2e"


def safe_last(series):
    s = pd.Series(series).dropna()
    return np.nan if s.empty else float(s.iloc[-1])


def safe_prev(series, n=2):
    s = pd.Series(series).dropna()
    return np.nan if len(s) < n else float(s.iloc[-n])


def pct_change(value, prev):
    if pd.isna(value) or pd.isna(prev) or prev == 0:
        return np.nan
    return (value / prev - 1) * 100


def clamp(value, low=0, high=100):
    if pd.isna(value):
        return np.nan
    return max(low, min(high, float(value)))


@st.cache_data(ttl=3600, show_spinner=False)
def download_close(ticker: str, period: str = "3mo") -> pd.Series:
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        close_cols = [col for col in data.columns if col[0] in ("Adj Close", "Close")]
        return data[close_cols[0]].dropna() if close_cols else pd.Series(dtype=float)
    for col in ["Adj Close", "Close"]:
        if col in data.columns:
            return data[col].dropna()
    return pd.Series(dtype=float)


def yf_snapshot(ticker: str, label: str) -> dict:
    series = download_close(ticker)
    last = safe_last(series)
    prev = safe_prev(series)
    month_prev = safe_prev(series, 22) if len(series.dropna()) >= 22 else np.nan
    return {
        "label": label,
        "ticker": ticker,
        "value": last,
        "perf_1d": pct_change(last, prev),
        "perf_1m": pct_change(last, month_prev),
        "series": series,
        "source": "Yahoo Finance",
    }


@st.cache_data(ttl=21600, show_spinner=False)
def fred_snapshot(api_key: str | None, months: int = 36) -> dict:
    output = {}
    if not api_key:
        for key, (series_id, label) in FRED_SERIES.items():
            output[key] = {"label": label, "ticker": series_id, "value": np.nan, "change": np.nan, "series": pd.Series(dtype=float)}
        return output

    try:
        from fredapi import Fred

        fred = Fred(api_key=api_key)
    except Exception:
        return fred_snapshot(None, months)

    for key, (series_id, label) in FRED_SERIES.items():
        try:
            series = pd.Series(fred.get_series(series_id)).dropna()
            series.index = pd.to_datetime(series.index)
            cutoff = series.index.max() - pd.DateOffset(months=months)
            series = series[series.index >= cutoff]
            value = safe_last(series)
            output[key] = {
                "label": label,
                "ticker": series_id,
                "value": value,
                "change": value - safe_prev(series) if not pd.isna(value) else np.nan,
                "series": series,
                "source": "FRED",
            }
        except Exception:
            output[key] = {"label": label, "ticker": series_id, "value": np.nan, "change": np.nan, "series": pd.Series(dtype=float)}
    return output


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_data(fred_api_key: str | None) -> tuple[dict, dict]:
    market = {key: yf_snapshot(ticker, label) for key, (ticker, label) in MARKET_PROXIES.items()}
    fred = fred_snapshot(fred_api_key)
    return market, fred


def latest_monthly_change(series: pd.Series, months_back: int = 1):
    s = pd.Series(series).dropna()
    if s.empty or not isinstance(s.index, pd.DatetimeIndex):
        return np.nan
    anchor = s.index.max() - pd.DateOffset(months=months_back)
    past = s[s.index <= anchor]
    return np.nan if past.empty else float(s.iloc[-1] - past.iloc[-1])


def compute_sahm_proxy(unrate_series: pd.Series):
    s = pd.Series(unrate_series).dropna()
    if s.empty or len(s) < 12:
        return np.nan
    if isinstance(s.index, pd.DatetimeIndex):
        s = s.resample("ME").last().dropna()
    ma3 = s.rolling(3).mean().dropna()
    if len(ma3) < 12:
        return np.nan
    return float(ma3.iloc[-1] - ma3.iloc[-12:].min())


def score_growth(iwm_spx_ratio_1m, sahm_proxy, claims_change):
    score = 50
    if not pd.isna(iwm_spx_ratio_1m):
        score += -8 if iwm_spx_ratio_1m > 0 else 8
    if not pd.isna(sahm_proxy):
        score += min(20, sahm_proxy * 40)
    if not pd.isna(claims_change):
        score += 8 if claims_change > 15000 else -3
    return clamp(score)


def score_inflation(oil_1m, dbc_1m, t5yie, core_pce):
    score = 50
    if not pd.isna(oil_1m):
        score += 12 if oil_1m > 5 else (-8 if oil_1m < -5 else 0)
    if not pd.isna(dbc_1m):
        score += 8 if dbc_1m > 2 else (-5 if dbc_1m < -2 else 0)
    if not pd.isna(t5yie):
        score += 10 if t5yie >= 2.4 else (-5 if t5yie < 2.0 else 0)
    if not pd.isna(core_pce):
        score += 10 if core_pce >= 2.8 else (-5 if core_pce < 2.3 else 0)
    return clamp(score)


def score_liquidity(walcl_change, m2_change, totalsl_change):
    score = 50
    if not pd.isna(walcl_change):
        score += -10 if walcl_change > 0 else 8
    if not pd.isna(m2_change):
        score += -8 if m2_change > 0 else 6
    if not pd.isna(totalsl_change):
        score += -4 if totalsl_change > 0 else 2
    return clamp(score)


def score_credit(hyg_1m, lqd_1m, hy_oas):
    score = 50
    if not pd.isna(hyg_1m):
        score += -8 if hyg_1m > 0 else 8
    if not pd.isna(lqd_1m):
        score += -5 if lqd_1m > 0 else 5
    if not pd.isna(hy_oas):
        score += 14 if hy_oas > 4.5 else (-6 if hy_oas < 3.5 else 4)
    return clamp(score)


def score_vol(vix):
    score = 40
    if not pd.isna(vix):
        if vix < 15:
            score -= 15
        elif vix < 20:
            score -= 5
        elif vix < 25:
            score += 8
        elif vix < 30:
            score += 18
        else:
            score += 28
    return clamp(score)


def score_dollar(dxy_1m):
    score = 50
    if not pd.isna(dxy_1m):
        score += 12 if dxy_1m > 1 else (-8 if dxy_1m < -1 else 0)
    return clamp(score)


def classify_regime(growth_score, inflation_score):
    if pd.isna(growth_score) or pd.isna(inflation_score):
        return "Mixed / incomplete data"
    if growth_score >= 55 and inflation_score >= 55:
        return "Stagflation / slowdown risk"
    if growth_score < 55 and inflation_score >= 55:
        return "Reflation / inflationary expansion"
    if growth_score >= 55 and inflation_score < 55:
        return "Disinflationary slowdown"
    return "Goldilocks / supportive"


def risk_label(score):
    if pd.isna(score):
        return "N/A"
    if score < 25:
        return "Calm"
    if score < 45:
        return "Normal"
    if score < 60:
        return "Caution"
    if score < 75:
        return "Elevated"
    return "Stress"


def build_macro_snapshot(market: dict, fred: dict) -> dict:
    spx_1m = market["spx"]["perf_1m"]
    iwm_1m = market["iwm"]["perf_1m"]
    iwm_spx_ratio = iwm_1m - spx_1m if not pd.isna(spx_1m) and not pd.isna(iwm_1m) else np.nan
    sahm = compute_sahm_proxy(fred["unrate"]["series"])
    claims_change = latest_monthly_change(fred["claims"]["series"])
    walcl_change = latest_monthly_change(fred["walcl"]["series"])
    m2_change = latest_monthly_change(fred["m2sl"]["series"])
    totalsl_change = latest_monthly_change(fred["totalsl"]["series"])

    growth = score_growth(iwm_spx_ratio, sahm, claims_change)
    inflation = score_inflation(market["oil"]["perf_1m"], market["dbc"]["perf_1m"], fred["t5yie"]["value"], fred["corepce"]["value"])
    liquidity = score_liquidity(walcl_change, m2_change, totalsl_change)
    credit = score_credit(market["hyg"]["perf_1m"], market["lqd"]["perf_1m"], fred["hyspread"]["value"])
    vol = score_vol(market["vix"]["value"])
    dollar = score_dollar(market["dxy"]["perf_1m"])
    macro_risk = clamp(np.nanmean([growth, inflation, 100 - liquidity, credit, vol, dollar]))

    return {
        "date": datetime.now().strftime("%d %b %Y"),
        "growth_score": growth,
        "inflation_score": inflation,
        "liquidity_score": liquidity,
        "credit_score": credit,
        "vol_score": vol,
        "dollar_score": dollar,
        "macro_risk": macro_risk,
        "regime": classify_regime(growth, inflation),
        "sahm_proxy": sahm,
        "iwm_spx_ratio_1m": iwm_spx_ratio,
        "oil": market["oil"]["value"],
        "oil_1m": market["oil"]["perf_1m"],
        "vix": market["vix"]["value"],
        "dxy": market["dxy"]["value"],
        "dxy_1m": market["dxy"]["perf_1m"],
        "spx": market["spx"]["value"],
        "iwm": market["iwm"]["value"],
        "gold": market["gold"]["value"],
        "btc": market["btc"]["value"],
        "tlt": market["tlt"]["value"],
        "us10y": fred["dgs10"]["value"],
        "us2y": fred["dgs2"]["value"],
        "curve_10_2": fred["dgs10"]["value"] - fred["dgs2"]["value"] if not pd.isna(fred["dgs10"]["value"]) and not pd.isna(fred["dgs2"]["value"]) else np.nan,
        "hy_oas": fred["hyspread"]["value"],
        "t5yie": fred["t5yie"]["value"],
        "core_pce": fred["corepce"]["value"],
        "walcl_change": walcl_change,
        "m2_change": m2_change,
        "totalsl_change": totalsl_change,
    }


def asset_outlook(snapshot: dict) -> pd.DataFrame:
    rows = []
    rows.append(("Large-Cap Equities", "Constructive" if snapshot["macro_risk"] < 60 else "Neutral", "Risk score and yields drive broad beta tolerance."))
    rows.append(("Small Caps", "Cautious" if snapshot["growth_score"] >= 55 or snapshot["dollar_score"] >= 55 else "Neutral", "Small caps are sensitive to growth and dollar/liquidity pressure."))
    rows.append(("Long-Duration Bonds", "Pressured" if pd.notna(snapshot["us10y"]) and snapshot["us10y"] > 4.3 else "Tactical", "Duration needs either growth weakness or softer inflation."))
    rows.append(("Credit", "Cautious" if pd.notna(snapshot["hy_oas"]) and snapshot["hy_oas"] > 4.5 else "Neutral", "Spread widening is the stress confirmation."))
    rows.append(("Gold", "Bullish" if snapshot["inflation_score"] >= 55 or snapshot["macro_risk"] >= 55 else "Neutral", "Macro stress and inflation hedging support demand."))
    rows.append(("U.S. Dollar", "Bullish" if snapshot["dollar_score"] >= 55 else "Neutral", "Dollar strength tightens global liquidity."))
    rows.append(("Bitcoin", "Two-way" if snapshot["vol_score"] >= 55 else "Constructive", "Liquidity helps, but volatility and dollar strength matter."))
    return pd.DataFrame(rows, columns=["Asset", "Outlook", "Why"])


def key_indicators(snapshot: dict) -> pd.DataFrame:
    rows = [
        ("Oil", snapshot["oil"], "90-95 = stagflation pressure"),
        ("VIX", snapshot["vix"], "30+ = stress escalation"),
        ("US 10Y", snapshot["us10y"], "4.30+ = risk headwind"),
        ("Credit OAS", snapshot["hy_oas"], "Widening = stress confirmation"),
        ("Sahm Proxy", snapshot["sahm_proxy"], "0.50+ = recession risk"),
        ("DXY", snapshot["dxy"], "Dollar strength = tighter liquidity"),
    ]
    table = pd.DataFrame(rows, columns=["Indicator", "Current", "Threshold / Message"])
    table["Current"] = table["Current"].map(lambda value: "N/A" if pd.isna(value) else f"{value:,.2f}")
    return table


def gauge_figure(score: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=0 if pd.isna(score) else score,
            number={"font": {"color": "#e2e8f0", "size": 42}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": "#d4af37"},
                "bgcolor": BG,
                "borderwidth": 1,
                "bordercolor": "#1e3a5f",
                "steps": [
                    {"range": [0, 25], "color": GREEN},
                    {"range": [25, 45], "color": "#84cc16"},
                    {"range": [45, 60], "color": YELLOW},
                    {"range": [60, 75], "color": ORANGE},
                    {"range": [75, 100], "color": RED},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor=BG, font=dict(color="#e2e8f0"))
    return fig


def factor_bar_figure(snapshot: dict):
    names = ["Growth", "Inflation", "Liquidity Stress", "Credit", "Volatility", "Dollar"]
    scores = [
        snapshot["growth_score"],
        snapshot["inflation_score"],
        100 - snapshot["liquidity_score"] if not pd.isna(snapshot["liquidity_score"]) else np.nan,
        snapshot["credit_score"],
        snapshot["vol_score"],
        snapshot["dollar_score"],
    ]
    colors = [GREEN if s < 40 else YELLOW if s < 60 else ORANGE if s < 75 else RED for s in scores]
    fig = go.Figure(go.Bar(x=scores, y=names, orientation="h", marker_color=colors, text=[f"{s:.0f}" if pd.notna(s) else "N/A" for s in scores], textposition="outside"))
    fig.update_layout(height=330, xaxis=dict(range=[0, 105], title="Risk / Pressure Score"), template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, margin=dict(l=10, r=20, t=20, b=20))
    return fig


def regime_map_figure(snapshot: dict):
    x = 100 - snapshot["growth_score"] if pd.notna(snapshot["growth_score"]) else 50
    y = snapshot["inflation_score"] if pd.notna(snapshot["inflation_score"]) else 50
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=50, y0=0, y1=50, fillcolor="#214e34", opacity=0.65, line_width=0)
    fig.add_shape(type="rect", x0=50, x1=100, y0=0, y1=50, fillcolor="#6a4c1b", opacity=0.65, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=50, y0=50, y1=100, fillcolor="#4d3040", opacity=0.65, line_width=0)
    fig.add_shape(type="rect", x0=50, x1=100, y0=50, y1=100, fillcolor="#3f4c6b", opacity=0.65, line_width=0)
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=["Current"], textposition="top right", marker=dict(size=16, color="#d4af37"), name="Current"))
    fig.add_annotation(x=25, y=25, text="Improving Growth<br>Cooling Inflation", showarrow=False, font=dict(color="#e2e8f0"))
    fig.add_annotation(x=75, y=25, text="Improving Growth<br>High Inflation", showarrow=False, font=dict(color="#e2e8f0"))
    fig.add_annotation(x=25, y=75, text="Weak Growth<br>Cooling Inflation", showarrow=False, font=dict(color="#e2e8f0"))
    fig.add_annotation(x=75, y=75, text="Weak Growth<br>High Inflation", showarrow=False, font=dict(color="#e2e8f0"))
    fig.update_layout(height=330, xaxis=dict(title="Growth Improving ->", range=[0, 100]), yaxis=dict(title="Inflation Pressure", range=[0, 100]), template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False, margin=dict(l=10, r=10, t=20, b=20))
    return fig


def market_snapshot_table(market: dict) -> pd.DataFrame:
    rows = []
    for key in ["spx", "iwm", "tlt", "hyg", "dxy", "gold", "oil", "vix", "btc"]:
        item = market[key]
        rows.append(
            {
                "Market": item["label"],
                "Ticker": item["ticker"],
                "Last": item["value"],
                "1D %": item["perf_1d"],
                "1M %": item["perf_1m"],
            }
        )
    df = pd.DataFrame(rows)
    return df


def get_fred_api_key() -> str | None:
    key = os.getenv("FRED_API_KEY")
    if key:
        return key
    try:
        key = st.secrets.get("FRED_API_KEY", None)
        if key:
            return key
    except Exception:
        pass
    secrets_path = Path(".streamlit") / "secrets.toml"
    if secrets_path.exists():
        try:
            with secrets_path.open("rb") as handle:
                return tomllib.load(handle).get("FRED_API_KEY")
        except Exception:
            return None
    return None


def render_macro_dashboard(show_download: bool = True, module_tabs: list[dict] | None = None, launch_callback=None):
    fred_key = get_fred_api_key()
    with st.spinner("Loading macro intelligence dashboard..."):
        market, fred = fetch_macro_data(fred_key)
        snapshot = build_macro_snapshot(market, fred)
        asset_df = asset_outlook(snapshot)
        indicators_df = key_indicators(snapshot)

    st.markdown("## Macro Intelligence Dashboard")
    st.caption(f"Updated {snapshot['date']} | Regime: {snapshot['regime']}")
    if not fred_key:
        st.info("FRED_API_KEY is not configured. Yahoo market proxies are live; FRED macro series display as N/A until a key is added.")

    from modules.tier2.market_regime_ui import render_market_regime_center
    
    tab_configs = module_tabs or []
    tab_labels = ["Market Regime Center", "Dashboard", "Asset Outlook", "Key Indicators", "Market Snapshot", "Raw Snapshot"]
    tab_labels.extend(tab_config["label"] for tab_config in tab_configs)
    tabs = st.tabs(tab_labels)
    tab_regime, tab_dashboard, tab_assets, tab_indicators, tab_market, tab_raw = tabs[:6]
    
    with tab_regime:
        render_market_regime_center()
        
    with tab_dashboard:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Macro Risk", f"{snapshot['macro_risk']:.0f}", risk_label(snapshot["macro_risk"]))
        c2.metric("VIX", "N/A" if pd.isna(snapshot["vix"]) else f"{snapshot['vix']:.2f}")
        c3.metric("US 10Y", "N/A" if pd.isna(snapshot["us10y"]) else f"{snapshot['us10y']:.2f}%")
        c4.metric("DXY", "N/A" if pd.isna(snapshot["dxy"]) else f"{snapshot['dxy']:.2f}", "N/A" if pd.isna(snapshot["dxy_1m"]) else f"{snapshot['dxy_1m']:+.2f}% 1M")

        left, right = st.columns([1, 1.35])
        with left:
            st.plotly_chart(gauge_figure(snapshot["macro_risk"]), width="stretch", key="macro_home_gauge")
        with right:
            st.markdown("### Macro Story")
            st.write(
                f"Current regime is **{snapshot['regime']}**. "
                f"Growth risk is {snapshot['growth_score']:.0f}, inflation pressure is {snapshot['inflation_score']:.0f}, "
                f"and volatility stress is {snapshot['vol_score']:.0f}. "
                "Use the dashboard as a regime map, not a trading signal by itself."
            )

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(factor_bar_figure(snapshot), width="stretch", key="macro_home_factors")
        with col_b:
            st.plotly_chart(regime_map_figure(snapshot), width="stretch", key="macro_home_regime")

    with tab_assets:
        st.dataframe(asset_df, width="stretch", hide_index=True)
    with tab_indicators:
        st.dataframe(indicators_df, width="stretch", hide_index=True)
    with tab_market:
        st.dataframe(market_snapshot_table(market).round(2), width="stretch", hide_index=True)
    with tab_raw:
        raw = pd.DataFrame([snapshot]).T.reset_index()
        raw.columns = ["Metric", "Value"]
        raw["Value"] = raw["Value"].astype(str)
        st.dataframe(raw, width="stretch", hide_index=True)
        if show_download:
            st.download_button(
                "Download Macro Snapshot CSV",
                data=raw.to_csv(index=False),
                file_name="fazdane_macro_snapshot.csv",
                mime="text/csv",
                width="stretch",
            )

    for tab, tab_config in zip(tabs[5:], tab_configs):
        with tab:
            items = tab_config.get("items", [])
            if not items:
                st.info("No modules configured.")
                continue
            columns = st.columns(2)
            for index, item in enumerate(items):
                with columns[index % 2]:
                    if st.button(item["label"], key=item["key"], width="stretch"):
                        if launch_callback:
                            launch_callback(item["module"], item["tier"])
                        else:
                            st.session_state["pending_nav"] = {"module": item["module"], "tier": item["tier"]}
                            st.rerun()

    return snapshot
