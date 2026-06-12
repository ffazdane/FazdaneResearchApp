"""
FazDane Analytics - Tier 1
Enhanced Market Breadth Dashboard
"""

from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from modules.base_module import FazDaneModule
from utils.universe_manager import get_universe_names, render_universe_manager


logger = logging.getLogger("MarketBreadth")

SPX_TICKER = "^GSPC"
VIX_TICKER = "^VIX"
STOCK_PROXY = "SPY"
TREASURY_PROXY = "TLT"
JUNK_BOND_PROXY = "HYG"
INVESTMENT_GRADE_PROXY = "LQD"

MARKET_BREADTH_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BRK-B", "C",
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY", "VLO", "HAL",
    "CAT", "DE", "GE", "HON", "UPS", "RTX", "BA", "LMT", "NOC", "ETN",
    "WMT", "COST", "HD", "LOW", "MCD", "SBUX", "NKE", "TGT", "TJX", "DIS",
    "XLE", "XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
    "SPY", "QQQ", "IWM", "DIA", "RSP", "MDY", "IJR", "VTI",
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
    "XLC": "Communication",
}


def clean_tickers(values) -> list[str]:
    cleaned = []
    for value in values:
        symbol = str(value).strip().upper()
        if symbol and symbol not in cleaned:
            cleaned.append(symbol)
    return cleaned


def clean_last(series: pd.Series) -> float:
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.iloc[-1]) if len(clean) else np.nan


def percentile_score(series: pd.Series, value=None, bullish_when_high: bool = True) -> float:
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 20:
        return np.nan
    if value is None:
        value = clean.iloc[-1]
    score = (clean <= value).mean() * 100
    if not bullish_when_high:
        score = 100 - score
    return float(np.clip(score, 0, 100))


def score_label(score: float) -> str:
    if pd.isna(score):
        return "Unavailable"
    if score <= 20:
        return "Extreme Weakness"
    if score <= 40:
        return "Weak"
    if score <= 60:
        return "Neutral"
    if score <= 80:
        return "Strong"
    return "Extreme Strength"


def score_color(score: float) -> str:
    if pd.isna(score):
        return "#7a8799"
    if score <= 40:
        return "#ef4444"
    if score <= 60:
        return "#facc15"
    return "#22c55e"


def rolling_percent_above_ma(close_df: pd.DataFrame, window: int) -> pd.Series:
    ma = close_df.rolling(window).mean()
    counts = close_df.notna().sum(axis=1).replace(0, np.nan)
    result = (close_df > ma).sum(axis=1) / counts * 100
    return result.replace([np.inf, -np.inf], np.nan)


def flatten_yfinance_columns(data: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if field not in data.columns.get_level_values(0):
            return pd.DataFrame()
        frame = data[field].copy()
    else:
        if field not in data.columns:
            return pd.DataFrame()
        frame = data[[field]].copy()
        frame.columns = [tickers[0]] if tickers else [field]

    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame.dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)
def download_market_data(tickers: tuple[str, ...], lookback_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * lookback_years + 120)
    ticker_list = clean_tickers(tickers)
    try:
        data = yf.download(
            ticker_list,
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )
    except Exception as exc:
        logger.error("Market breadth yfinance download failed: %s", exc)
        return pd.DataFrame(), pd.DataFrame()

    close = flatten_yfinance_columns(data, "Close", ticker_list).ffill()
    volume = flatten_yfinance_columns(data, "Volume", ticker_list).ffill()
    return close, volume


def endpoint_urls_last_n_days(days_back=15) -> list[str]:
    urls = ["https://production.dataviz.cnn.io/index/fearandgreed/graphdata"]
    for i in range(days_back):
        day = datetime.today() - timedelta(days=i)
        urls.append(f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{day.strftime('%Y-%m-%d')}")
    return urls


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_component_json(days_back=15) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.cnn.com/markets/fear-and-greed",
    }
    for url in endpoint_urls_last_n_days(days_back):
        try:
            response = requests.get(url, headers=headers, timeout=12)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and data:
                    return data
        except Exception:
            continue
    return {}


def parse_component_dataframe(component) -> pd.DataFrame:
    if isinstance(component, dict):
        containers = [component[key] for key in ["data", "chart", "history", "historical", "values", "series"] if key in component]
        if not containers:
            containers = [component]
    elif isinstance(component, list):
        containers = [component]
    else:
        return pd.DataFrame(columns=["Value"])

    for rows in containers:
        try:
            df = pd.DataFrame(rows)
            if df.empty:
                continue
            date_col = next((col for col in ["x", "date", "timestamp", "time"] if col in df.columns), None)
            value_col = next((col for col in ["y", "value", "score", "close"] if col in df.columns), None)
            if date_col is None or value_col is None:
                continue

            if date_col == "x":
                raw_date = pd.to_numeric(df[date_col], errors="coerce")
                unit = "ms" if raw_date.dropna().median() > 10**11 else "s"
                df["Date"] = pd.to_datetime(raw_date, unit=unit, errors="coerce")
            else:
                df["Date"] = pd.to_datetime(df[date_col], errors="coerce")

            df["Value"] = pd.to_numeric(df[value_col], errors="coerce")
            df = df[["Date", "Value"]].dropna().sort_values("Date").drop_duplicates("Date").set_index("Date")
            if len(df) > 5:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df
        except Exception:
            continue
    return pd.DataFrame(columns=["Value"])


def recursive_component_search(obj, include_words=None, exclude_words=None) -> tuple[pd.DataFrame, str | None]:
    include_words = [word.lower() for word in include_words] if include_words else []
    exclude_words = [word.lower() for word in exclude_words] if exclude_words else []
    matches = []

    def path_ok(path: str) -> bool:
        path_lower = path.lower()
        return all(word in path_lower for word in include_words) and not any(word in path_lower for word in exclude_words)

    def meta_ok(node) -> bool:
        if not isinstance(node, dict):
            return False
        meta = " ".join(str(node.get(key, "")) for key in ["name", "title", "label", "description", "slug", "key"]).lower()
        return all(word in meta for word in include_words) and not any(word in meta for word in exclude_words)

    def walk(node, path=""):
        if isinstance(node, dict):
            if path_ok(path) or meta_ok(node):
                df = parse_component_dataframe(node)
                if not df.empty:
                    matches.append((path, df))
            for key, value in node.items():
                next_path = f"{path}.{key}" if path else str(key)
                if path_ok(next_path):
                    df = parse_component_dataframe(value)
                    if not df.empty:
                        matches.append((next_path, df))
                walk(value, next_path)
        elif isinstance(node, list):
            if path_ok(path):
                df = parse_component_dataframe(node)
                if not df.empty:
                    matches.append((path, df))
            for idx, item in enumerate(node[:10]):
                walk(item, f"{path}[{idx}]")

    walk(obj)
    if not matches:
        return pd.DataFrame(columns=["Value"]), None
    matches = sorted(matches, key=lambda item: len(item[1]), reverse=True)
    return matches[0][1], matches[0][0]


def fetch_market_component_series(component_key_candidates=None, include_words=None, exclude_words=None):
    data = fetch_component_json(days_back=15)
    if not data:
        return pd.DataFrame(columns=["Value"]), None
    for key in component_key_candidates or []:
        if key in data:
            df = parse_component_dataframe(data[key])
            if not df.empty:
                return df, key
    return recursive_component_search(data, include_words=include_words, exclude_words=exclude_words)


def fetch_options_ratio_fallback(vix: pd.Series) -> tuple[pd.Series, str]:
    if vix.empty:
        return pd.Series(dtype=float), "Neutral Engine"
    simulated = 0.8 + (vix - vix.rolling(20).mean()) / vix.rolling(20).std() * 0.15
    return simulated.dropna(), "VIX-Derived Options Proxy"


def compute_breadth_dashboard(close: pd.DataFrame, volume: pd.DataFrame, breadth_symbols: list[str]) -> dict:
    breadth_cols = [ticker for ticker in breadth_symbols if ticker in close.columns]
    breadth_prices = close[breadth_cols].dropna(how="all") if breadth_cols else pd.DataFrame()
    breadth_volume_cols = [ticker for ticker in breadth_symbols if ticker in volume.columns]
    breadth_volumes = volume[breadth_volume_cols].dropna(how="all") if breadth_volume_cols else pd.DataFrame()

    spx = close[SPX_TICKER].dropna() if SPX_TICKER in close.columns else pd.Series(dtype=float)
    vix = close[VIX_TICKER].dropna() if VIX_TICKER in close.columns else pd.Series(dtype=float)
    spy = close[STOCK_PROXY].dropna() if STOCK_PROXY in close.columns else pd.Series(dtype=float)
    tlt = close[TREASURY_PROXY].dropna() if TREASURY_PROXY in close.columns else pd.Series(dtype=float)
    hyg = close[JUNK_BOND_PROXY].dropna() if JUNK_BOND_PROXY in close.columns else pd.Series(dtype=float)
    lqd = close[INVESTMENT_GRADE_PROXY].dropna() if INVESTMENT_GRADE_PROXY in close.columns else pd.Series(dtype=float)

    spx_ma125 = spx.rolling(125).mean()
    spx_distance_125 = ((spx / spx_ma125) - 1) * 100
    market_momentum_score = percentile_score(spx_distance_125, bullish_when_high=True)

    stock_price_strength_df, strength_source = fetch_market_component_series(
        component_key_candidates=["stock_price_strength", "stockPriceStrength", "price_strength"],
        include_words=["stock", "strength"],
    )
    if not stock_price_strength_df.empty:
        stock_price_strength_component = stock_price_strength_df["Value"].dropna()
        net_new_highs_lows = stock_price_strength_component.copy()
        net_new_highs_lows_pct = stock_price_strength_component.copy()
        stock_price_strength_status = f"Stock Price Strength loaded from {strength_source}"
    elif not breadth_prices.empty:
        rolling_252_high = breadth_prices.rolling(252).max()
        rolling_252_low = breadth_prices.rolling(252).min()
        new_highs = (breadth_prices >= rolling_252_high).sum(axis=1)
        new_lows = (breadth_prices <= rolling_252_low).sum(axis=1)
        net_new_highs_lows = new_highs - new_lows
        net_new_highs_lows_pct = net_new_highs_lows / breadth_prices.notna().sum(axis=1).replace(0, np.nan) * 100
        stock_price_strength_component = net_new_highs_lows_pct.copy()
        stock_price_strength_status = "Stock Price Strength using selected universe sample"
    else:
        net_new_highs_lows = pd.Series(dtype=float)
        net_new_highs_lows_pct = pd.Series(dtype=float)
        stock_price_strength_component = pd.Series(dtype=float)
        stock_price_strength_status = "Stock Price Strength unavailable"
    stock_price_strength_score = percentile_score(stock_price_strength_component, bullish_when_high=True)

    returns = breadth_prices.pct_change() if not breadth_prices.empty else pd.DataFrame()
    advances = (returns > 0).sum(axis=1) if not returns.empty else pd.Series(dtype=float)
    declines = (returns < 0).sum(axis=1) if not returns.empty else pd.Series(dtype=float)
    ad_line = (advances - declines).cumsum() if not returns.empty else pd.Series(dtype=float)
    mcclellan_osc = (advances - declines).ewm(span=19, adjust=False).mean() - (advances - declines).ewm(span=39, adjust=False).mean() if not returns.empty else pd.Series(dtype=float)

    if not breadth_volumes.empty and not returns.empty:
        common_cols = [col for col in breadth_prices.columns if col in breadth_volumes.columns]
        returns_common = returns[common_cols]
        volumes_common = breadth_volumes[common_cols]
        advancing_volume = volumes_common.where(returns_common > 0, 0).sum(axis=1)
        declining_volume = volumes_common.where(returns_common < 0, 0).sum(axis=1)
        total_volume = advancing_volume + declining_volume
        volume_net_pct = ((advancing_volume - declining_volume) / total_volume.replace(0, np.nan)) * 100
        mcclellan_volume_osc = volume_net_pct.ewm(span=19, adjust=False).mean() - volume_net_pct.ewm(span=39, adjust=False).mean()
        mcclellan_volume_summation = mcclellan_volume_osc.cumsum()
    else:
        mcclellan_volume_summation = pd.Series(index=breadth_prices.index, dtype=float) if not breadth_prices.empty else pd.Series(dtype=float)
    mcclellan_volume_score = percentile_score(mcclellan_volume_summation, bullish_when_high=True)

    above_50 = rolling_percent_above_ma(breadth_prices, 50) if not breadth_prices.empty else pd.Series(dtype=float)
    above_200 = rolling_percent_above_ma(breadth_prices, 200) if not breadth_prices.empty else pd.Series(dtype=float)
    breadth_composite = (above_50 * 0.6) + (above_200 * 0.4)
    stock_price_breadth_score = percentile_score(breadth_composite, bullish_when_high=True)

    options_df, options_source = fetch_market_component_series(
        component_key_candidates=["put_call_options", "putCall", "options_index"],
        include_words=["put", "call"],
    )
    if not options_df.empty:
        put_call = options_df["Value"].dropna()
        put_call_source = f"Options Ratio Engine: {options_source}"
    else:
        put_call, options_source = fetch_options_ratio_fallback(vix)
        put_call_source = options_source
    put_call_ma = put_call.rolling(5).mean().dropna()
    put_call_score = percentile_score(put_call_ma, bullish_when_high=False)
    if pd.isna(put_call_score):
        put_call_score = 50.0

    vix_ma50 = vix.rolling(50).mean()
    vix_distance = ((vix / vix_ma50) - 1) * 100
    volatility_score = percentile_score(vix_distance, bullish_when_high=False)

    safe_haven_spread = (spy.pct_change(20) * 100) - (tlt.pct_change(20) * 100)
    safe_haven_score = percentile_score(safe_haven_spread, bullish_when_high=True)

    junk_spread = (hyg.pct_change(20) * 100) - (lqd.pct_change(20) * 100)
    junk_bond_score = percentile_score(junk_spread, bullish_when_high=True)

    component_scores = pd.Series(
        {
            "S&P 500 vs 125DMA": market_momentum_score,
            "52W Highs vs Lows": stock_price_strength_score,
            "McClellan Volume Summation": mcclellan_volume_score,
            "Stock Price Breadth": stock_price_breadth_score,
            "Put / Call Options": put_call_score,
            "Market Volatility": volatility_score,
            "Safe Haven Demand": safe_haven_score,
            "Junk Bond Demand": junk_bond_score,
        },
        dtype=float,
    )
    composite_score = component_scores.dropna().mean()

    return {
        "breadth_prices": breadth_prices,
        "spx": spx,
        "spx_ma125": spx_ma125,
        "spx_distance_125": spx_distance_125,
        "vix": vix,
        "net_new_highs_lows": net_new_highs_lows,
        "net_new_highs_lows_pct": net_new_highs_lows_pct,
        "stock_price_strength_component": stock_price_strength_component,
        "mcclellan_volume_summation": mcclellan_volume_summation,
        "mcclellan_osc": mcclellan_osc,
        "ad_line": ad_line,
        "advances": advances,
        "declines": declines,
        "above_50": above_50,
        "above_200": above_200,
        "breadth_composite": breadth_composite,
        "put_call_ma": put_call_ma,
        "safe_haven_spread": safe_haven_spread,
        "junk_spread": junk_spread,
        "component_scores": component_scores,
        "composite_score": composite_score,
        "composite_label": score_label(composite_score),
        "stock_price_strength_status": stock_price_strength_status,
        "put_call_status": put_call_source,
    }


def base_layout(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e2e8f0")),
        paper_bgcolor="#0d1b2e",
        plot_bgcolor="#111622",
        font=dict(color="#e2e8f0", family="Inter"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.16)", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="rgba(148,163,184,0.16)", tickfont=dict(color="#94a3b8")),
        legend=dict(bgcolor="rgba(21,40,71,0.85)", bordercolor="#1e3a5f", borderwidth=1),
        margin=dict(l=20, r=20, t=54, b=20),
        height=height,
    )
    return fig


def line_chart(series_map: dict[str, pd.Series], title: str, height: int = 360, hline: float | None = None) -> go.Figure:
    fig = go.Figure()
    colors = ["#3ab54a", "#93c5fd", "#facc15", "#ef4444", "#a78bfa"]
    for idx, (name, series) in enumerate(series_map.items()):
        clean = pd.Series(series).dropna()
        fig.add_trace(go.Scatter(x=clean.index, y=clean.values, mode="lines", name=name, line=dict(color=colors[idx % len(colors)], width=1.8)))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="#facc15", line_width=1)
    return base_layout(fig, title, height)


def bar_chart(series: pd.Series, title: str, positive_color="#22c55e", negative_color="#ef4444", height: int = 360) -> go.Figure:
    clean = pd.Series(series).dropna()
    colors = np.where(clean >= 0, positive_color, negative_color)
    fig = go.Figure(go.Bar(x=clean.index, y=clean.values, marker_color=colors))
    fig.add_hline(y=0, line_dash="dash", line_color="#facc15", line_width=1)
    return base_layout(fig, title, height)


def component_bar(scores: pd.Series) -> go.Figure:
    clean = scores.sort_values()
    fig = go.Figure(
        go.Bar(
            x=clean.values,
            y=clean.index,
            orientation="h",
            marker_color=[score_color(value) for value in clean.values],
            text=[f"{value:.1f}" if not pd.isna(value) else "N/A" for value in clean.values],
            textposition="outside",
        )
    )
    fig.add_vline(x=50, line_dash="dash", line_color="#facc15", line_width=1)
    fig.update_xaxes(range=[0, 105], title="0 = Bearish Distribution | 100 = Bullish Concentration")
    return base_layout(fig, "Component Scores Matrix", 430)


def gauge(score: float, label: str) -> go.Figure:
    display_score = 50 if pd.isna(score) else score
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=display_score,
            domain={"x": [0.08, 0.92], "y": [0.02, 0.82]},
            number={"font": {"color": "#e2e8f0", "size": 38}, "suffix": ""},
            title={"text": f"<b>{label}</b><br><span style='font-size:0.68em;color:#94a3b8'>Composite Market Breadth Index</span>", "font": {"color": score_color(score), "size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": score_color(score), "thickness": 0.18},
                "bgcolor": "#111622",
                "borderwidth": 1,
                "bordercolor": "#1e3a5f",
                "steps": [
                    {"range": [0, 20], "color": "#7f1d1d"},
                    {"range": [20, 40], "color": "#b91c1c"},
                    {"range": [40, 60], "color": "#854d0e"},
                    {"range": [60, 80], "color": "#166534"},
                    {"range": [80, 100], "color": "#047857"},
                ],
                "threshold": {"line": {"color": "#ffffff", "width": 4}, "thickness": 0.75, "value": display_score},
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#0d1b2e",
        plot_bgcolor="#111622",
        height=380,
        margin=dict(l=16, r=16, t=12, b=12),
    )
    return fig


def sector_breadth_table(close: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, sector in SECTORS.items():
        if ticker not in close.columns:
            continue
        prices = close[ticker].dropna()
        if len(prices) < 200:
            continue
        current = prices.iloc[-1]
        sma20 = prices.rolling(20).mean().iloc[-1]
        sma50 = prices.rolling(50).mean().iloc[-1]
        sma200 = prices.rolling(200).mean().iloc[-1]
        rows.append(
            {
                "Sector": sector,
                "Ticker": ticker,
                "Price": current,
                "Dist from 20-SMA": ((current / sma20) - 1) * 100,
                "Dist from 50-SMA": ((current / sma50) - 1) * 100,
                "Dist from 200-SMA": ((current / sma200) - 1) * 100,
            }
        )
    return pd.DataFrame(rows).sort_values("Dist from 20-SMA", ascending=False)


class MarketBreadthModule(FazDaneModule):
    MODULE_NAME = "Market Breadth Dashboard"
    MODULE_ICON = "Chart"
    MODULE_DESCRIPTION = "Composite breadth, momentum, volatility, options, credit, and sector internals"
    TIER = 1
    SOURCE_NOTEBOOK = "FazDane Market Breadth Analysis Dashboard"
    CACHE_TTL = 3600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance", "CNN Fear & Greed endpoint fallback"]

    def render_sidebar(self):
        self._default_universe()
        st.markdown("**Breadth Universe**")
        universe_name, breadth_symbols, _ = render_universe_manager(
            key_prefix="mb",
            show_benchmark=False,
            label="Breadth Universe:",
        )
        if not breadth_symbols:
            breadth_symbols = MARKET_BREADTH_SAMPLE

        self.universe_name = universe_name
        self.breadth_symbols = clean_tickers(breadth_symbols)
        st.caption(f"{len(self.breadth_symbols)} breadth symbols selected from {universe_name}.")

        st.markdown("**Dashboard Settings**")
        self.lookback_years = int(st.slider("Context Window (Years):", 1, 5, 3, key="mb_lookback_years"))
        self.chart_months = int(st.slider("Chart History (Months):", 3, 36, 18, step=3, key="mb_chart_months"))
        self.show_raw = st.checkbox("Show data tables", value=True, key="mb_show_tables")

        if st.button("Calculate Breadth", width="stretch", type="primary", key="mb_calc"):
            download_market_data.clear()
            fetch_component_json.clear()
            st.rerun()

    def render_main(self):
        self.render_section_header(
            "Market Breadth Analysis",
            "Composite market internals across momentum, high/low strength, options, volatility, safe-haven demand, and credit risk",
        )

        required = [SPX_TICKER, VIX_TICKER, STOCK_PROXY, TREASURY_PROXY, JUNK_BOND_PROXY, INVESTMENT_GRADE_PROXY]
        all_tickers = tuple(sorted(set(required + list(SECTORS.keys()) + self.breadth_symbols)))

        with st.spinner(f"Fetching market breadth data for {len(all_tickers)} symbols..."):
            close, volume = download_market_data(all_tickers, self.lookback_years)

        if close.empty:
            st.error("Failed to fetch market data from yfinance.")
            return

        dashboard = compute_breadth_dashboard(close, volume, self.breadth_symbols)
        if dashboard["breadth_prices"].empty:
            st.warning("No breadth data found for the selected universe.")
            return

        cutoff = pd.Timestamp(datetime.today() - timedelta(days=int(self.chart_months * 30.5)))
        component_scores = dashboard["component_scores"]
        composite_score = dashboard["composite_score"]
        composite_label = dashboard["composite_label"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Market Breadth Index", f"{composite_score:.1f}", composite_label)
        m2.metric("Universe Symbols", f"{len(dashboard['breadth_prices'].columns):,}")
        m3.metric("Above 50DMA", f"{clean_last(dashboard['above_50']):.1f}%")
        m4.metric("Above 200DMA", f"{clean_last(dashboard['above_200']):.1f}%")

        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(gauge(composite_score, composite_label), width="stretch")
        with right:
            st.plotly_chart(component_bar(component_scores), width="stretch")

        tabs = st.tabs(["Breadth Dashboard", "Risk Profiles", "Sectors", "Data"])

        with tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    line_chart(
                        {
                            "S&P 500": dashboard["spx"].loc[dashboard["spx"].index >= cutoff],
                            "125DMA": dashboard["spx_ma125"].loc[dashboard["spx_ma125"].index >= cutoff],
                        },
                        "1. S&P 500 vs 125-Day Moving Average",
                    ),
                    width="stretch",
                )
            with c2:
                st.plotly_chart(
                    bar_chart(
                        dashboard["net_new_highs_lows"].loc[dashboard["net_new_highs_lows"].index >= cutoff],
                        "2. Net New 52-Week Highs vs Lows",
                    ),
                    width="stretch",
                )

            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(
                    line_chart(
                        {"Volume Summation": dashboard["mcclellan_volume_summation"].loc[dashboard["mcclellan_volume_summation"].index >= cutoff]},
                        "3. McClellan Volume Summation Index",
                        hline=0,
                    ),
                    width="stretch",
                )
            with c4:
                st.plotly_chart(
                    line_chart(
                        {"Price Strength": dashboard["stock_price_strength_component"].loc[dashboard["stock_price_strength_component"].index >= cutoff]},
                        "4. Stock Price Strength Oscillator",
                    ),
                    width="stretch",
                )

            c5, c6 = st.columns(2)
            with c5:
                st.plotly_chart(
                    line_chart(
                        {
                            "% Above 50DMA": dashboard["above_50"].loc[dashboard["above_50"].index >= cutoff],
                            "% Above 200DMA": dashboard["above_200"].loc[dashboard["above_200"].index >= cutoff],
                        },
                        "5. Stock Price Breadth Profiles",
                        hline=50,
                    ),
                    width="stretch",
                )
            with c6:
                st.plotly_chart(
                    bar_chart(
                        dashboard["mcclellan_osc"].loc[dashboard["mcclellan_osc"].index >= cutoff],
                        "Advance / Decline McClellan Oscillator",
                    ),
                    width="stretch",
                )

        with tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    line_chart(
                        {"Put/Call Momentum": dashboard["put_call_ma"].loc[dashboard["put_call_ma"].index >= cutoff]},
                        "6. Options Allocation Momentum",
                    ),
                    width="stretch",
                )
            with c2:
                st.plotly_chart(
                    line_chart(
                        {"VIX": dashboard["vix"].loc[dashboard["vix"].index >= cutoff]},
                        "7. Market Volatility Profiles",
                    ),
                    width="stretch",
                )

            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(
                    line_chart(
                        {"SPY minus TLT 20D Return": dashboard["safe_haven_spread"].loc[dashboard["safe_haven_spread"].index >= cutoff]},
                        "8. Safe Haven Multi-Asset Spreads",
                        hline=0,
                    ),
                    width="stretch",
                )
            with c4:
                st.plotly_chart(
                    line_chart(
                        {"HYG minus LQD 20D Return": dashboard["junk_spread"].loc[dashboard["junk_spread"].index >= cutoff]},
                        "9. High Yield Risk Appetite Spreads",
                        hline=0,
                    ),
                    width="stretch",
                )

            st.info(
                f"Status Framework: {dashboard['stock_price_strength_status']} | {dashboard['put_call_status']}"
            )

        with tabs[2]:
            sector_df = sector_breadth_table(close)
            if sector_df.empty:
                st.info("Sector ETF data was not available.")
            else:
                st.dataframe(sector_df.round(2), width="stretch", hide_index=True)

        with tabs[3]:
            component_table = pd.DataFrame(
                {
                    "Component Segment": component_scores.index,
                    "Calculated Score": component_scores.values,
                    "Assigned Condition": [score_label(value) for value in component_scores.values],
                }
            ).sort_values("Calculated Score", ascending=False)
            st.dataframe(component_table.round(2), width="stretch", hide_index=True)

            if self.show_raw:
                daily_frame = pd.DataFrame(
                    {
                        "SPX": dashboard["spx"],
                        "SPX_125DMA": dashboard["spx_ma125"],
                        "SPX_Dist_125DMA": dashboard["spx_distance_125"],
                        "Net_New_Highs_Lows": dashboard["net_new_highs_lows"],
                        "Above_50DMA": dashboard["above_50"],
                        "Above_200DMA": dashboard["above_200"],
                        "McClellan_Volume_Summation": dashboard["mcclellan_volume_summation"],
                        "Put_Call_MA": dashboard["put_call_ma"],
                        "VIX": dashboard["vix"],
                        "Safe_Haven_Spread": dashboard["safe_haven_spread"],
                        "Junk_Bond_Spread": dashboard["junk_spread"],
                    }
                ).dropna(how="all")
                st.dataframe(daily_frame.tail(500).round(3), width="stretch")
                st.download_button(
                    "Download Market Breadth Data CSV",
                    data=daily_frame.to_csv(index=True),
                    file_name="fazdane_market_breadth_data.csv",
                    mime="text/csv",
                    width="stretch",
                )

    def _default_universe(self):
        key = "mb_sel"
        target = "FazDane Portfolio"
        if target in get_universe_names() and key not in st.session_state:
            st.session_state[key] = target
