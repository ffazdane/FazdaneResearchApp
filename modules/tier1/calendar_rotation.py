"""
FazDane Analytics — Tier 1
Calendar Option Strategy Rotation Matrix
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
from utils.universe_manager import render_universe_multiselect

logger = logging.getLogger("CalendarRotation")

# Global Parameters
LOOKBACK_DAYS = 90
TRAIL_DAYS = 20
PLOT_TOP_N = 18
TRAIL_SMOOTH_WINDOW = 4

UNIVERSES = {
    "Calendar Candidates": {
        "tickers": {
            "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF", "IWM": "Russell 2000 ETF",
            "DIA": "Dow Jones ETF", "GLD": "Gold ETF", "NVDA": "Nvidia",
            "TSLA": "Tesla", "AAPL": "Apple", "MSFT": "Microsoft",
            "AMZN": "Amazon", "META": "Meta Platforms", "GOOGL": "Alphabet",
            "AVGO": "Broadcom", "AMD": "Advanced Micro Devices", "NFLX": "Netflix",
            "INTC": "Intel", "QCOM": "Qualcomm", "CSCO": "Cisco",
            "AMAT": "Applied Materials", "COIN": "Coinbase", "HOOD": "Robinhood",
            "PLTR": "Palantir", "IBM": "IBM", "CRM": "Salesforce",
            "ADBE": "Adobe", "ORCL": "Oracle", "CRWD": "CrowdStrike",
            "JPM": "JPMorgan Chase", "GS": "Goldman Sachs", "UNH": "UnitedHealth",
            "LLY": "Eli Lilly", "COST": "Costco", "HD": "Home Depot",
            "BA": "Boeing", "CAT": "Caterpillar"
        },
        "benchmark": "SPY",
    },
    "SPX Sectors": {
        "tickers": {
            "XLC": "Communication Services", "XLY": "Consumer Discretionary",
            "XLP": "Consumer Staples", "XLE": "Energy", "XLF": "Financials",
            "XLV": "Health Care", "XLI": "Industrials", "XLB": "Materials",
            "XLRE": "Real Estate", "XLK": "Technology", "XLU": "Utilities"
        },
        "benchmark": "SPY",
    },
    "MAG 7": {
        "tickers": {
            "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia",
            "AMZN": "Amazon", "META": "Meta Platforms", "GOOGL": "Alphabet",
            "TSLA": "Tesla"
        },
        "benchmark": "QQQ",
    },
    "Leading ETFs": {
        "tickers": {
            "QQQ": "Nasdaq 100 ETF", "SPY": "S&P 500 ETF", "IWM": "Russell 2000 ETF",
            "DIA": "Dow Jones ETF", "SMH": "Semiconductor ETF", "XLK": "Technology ETF",
            "XLF": "Financial ETF", "XLE": "Energy ETF", "GLD": "Gold ETF",
            "SLV": "Silver ETF", "TLT": "Long Bond ETF", "HYG": "High Yield Bond ETF"
        },
        "benchmark": "SPY",
    },
    "Custom Tickers": {
        "tickers": {},
        "benchmark": "SPY",
    }
}

def configure_universe(selected_universe):
    uni = UNIVERSES[selected_universe]
    bench = uni["benchmark"]
    candidates = sorted(set([t for t in uni["tickers"].keys() if t != bench]))
    return {
        "selected_universe": selected_universe, "universe": uni, "benchmark": bench,
        "ticker_names": uni["tickers"],
        "tickers": list(uni["tickers"].keys()), "candidates": candidates,
    }

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

def download_price_data(tickers, benchmark, period="6mo"):
    symbols = sorted(set(list(tickers) + [benchmark]))
    raw = yf.download(symbols, period=period, auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": symbols[0]})
    return close.dropna(how="all"), volume.reindex(close.index).fillna(0), raw

def compute_rotation(close, benchmark, trail_days=TRAIL_DAYS):
    if benchmark not in close: return pd.DataFrame()
    bench = close[benchmark]
    rows = []
    for ticker in close.columns:
        if ticker == benchmark:
            continue
        px = close[ticker].dropna()
        aligned = pd.concat([px, bench], axis=1, join="inner").dropna()
        if len(aligned) < 70:
            continue
        rel_log = np.log(aligned.iloc[:, 0] / aligned.iloc[:, 1])
        rs_mean = rel_log.rolling(50, min_periods=30).mean()
        rs_std = rel_log.rolling(50, min_periods=30).std().replace(0, np.nan)
        rs_ratio = (100 + 2.0 * ((rel_log - rs_mean) / rs_std).clip(-3, 3)).ewm(span=5, adjust=False).mean()
        mom_raw = rs_ratio.diff(5)
        mom_std = mom_raw.rolling(30, min_periods=15).std().replace(0, np.nan)
        rs_momentum = (100 + 1.4 * (mom_raw / mom_std).clip(-3, 3)).ewm(span=5, adjust=False).mean()
        out = pd.DataFrame({
            "date": aligned.index, "ticker": ticker, "close": aligned.iloc[:, 0].values,
            "rs_ratio": rs_ratio.values, "rs_momentum": rs_momentum.values,
        }).dropna()
        rows.append(out.tail(trail_days))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def compute_price_features(close, volume, benchmark):
    features = []
    bench_ret_20 = close[benchmark].pct_change(20).iloc[-1] if benchmark in close else np.nan
    for ticker in close.columns:
        if ticker == benchmark:
            continue
        s = close[ticker].dropna()
        if len(s) < 50:
            continue
        v = volume[ticker].reindex(s.index).fillna(0)
        vol_ratio = v.iloc[-1] / max(v.rolling(20).mean().iloc[-1], 1)
        true_range = pd.concat([s.diff().abs(), (s - s.shift()).abs()], axis=1).max(axis=1)
        atr20 = true_range.rolling(20).mean().iloc[-1]
        trend_score = 0
        trend_score += 20 if s.iloc[-1] > s.ewm(span=8).mean().iloc[-1] else 0
        trend_score += 20 if s.ewm(span=8).mean().iloc[-1] > s.ewm(span=21).mean().iloc[-1] else 0
        trend_score += 15 if s.ewm(span=21).mean().iloc[-1] > s.ewm(span=21).mean().iloc[-6] else 0
        trend_score += 15 if s.iloc[-1] > s.rolling(50).mean().iloc[-1] else 0
        trend_score += 15 if s.iloc[-1] >= 0.97 * s.rolling(20).max().iloc[-1] else 0
        v_last = float(v.iloc[-1]) if not v.empty else 0
        trend_score += 15 if vol_ratio >= 1.2 else 0
        features.append({
            "ticker": ticker, "spot": float(s.iloc[-1]), "atr20": float(atr20),
            "option_oi": int(max(v_last * 0.005, 500)),
            "option_volume": int(max(v_last * 0.001, 100)),
            "trend_score": min(float(trend_score), 100.0),
            "rel_strength_20": float(s.pct_change(20).iloc[-1] - bench_ret_20)
        })
    return pd.DataFrame(features)

def add_scores(df, rotation_latest):
    if df.empty or rotation_latest.empty: return pd.DataFrame()
    df = df.merge(rotation_latest[["ticker", "rs_ratio", "rs_momentum"]], on="ticker", how="left")
    df["option_liquidity_score"] = np.clip(60 + (df["option_volume"] / df["option_volume"].max()) * 40, 60, 100).fillna(60)
    df["calendar_score"] = (0.4 * df["trend_score"] + 0.3 * df["rs_ratio"] + 0.3 * df["rs_momentum"]).clip(0, 100)
    df["target_strike"] = (df["spot"] * 1.03).round(1)
    df["distance_atr"] = 1.1
    df["stage"] = "Stage 3 Active"
    df["quality"] = np.where(df["calendar_score"] >= 75, "Best", "Watch")
    return df.sort_values("calendar_score", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_universe(univ_name, tickers, benchmark):
    ticker_list = list(tickers)
    candidates = sorted(set([t for t in ticker_list if t != benchmark]))
    ctx = {
        "selected_universe": univ_name,
        "benchmark": benchmark,
        "ticker_names": {t: t for t in ticker_list},
        "tickers": ticker_list,
        "candidates": candidates,
    }
    close, volume, raw = download_price_data(ctx["candidates"], ctx["benchmark"])
    if close.empty: return None
    rotation = compute_rotation(close, ctx["benchmark"])
    price_feats = compute_price_features(close, volume, ctx["benchmark"])
    if rotation.empty or price_feats.empty: return None
    latest_rot = rotation.sort_values("date").groupby("ticker").tail(1)
    
    # Calculate FDTS signals for all candidates
    fdts_signals = {}
    for ticker in ctx["candidates"]:
        ticker_df = extract_ticker_df(raw, ticker)
        sig = calculate_fdts_signal(ticker, ticker_df)
        fdts_signals[ticker] = {"Buy": "🟢 Buy", "Sell": "🔴 Sell", "No Trade": "⚪ No Trade"}.get(sig, "⚪ No Trade")

    final_scores = add_scores(price_feats, latest_rot)
    final_scores["fdts_signal"] = final_scores["ticker"].map(fdts_signals).fillna("⚪ No Trade")
    final_scores["universe"] = univ_name
    return {
        "context": ctx, "rotation": rotation, "scores": final_scores,
        "close": close, "volume": volume
    }

class CalendarRotationModule(FazDaneModule):
    MODULE_NAME = "Calendar Strategy Matrix"
    MODULE_ICON = "📅"
    MODULE_DESCRIPTION = "Multi-Universe Rotation Dashboard for Calendar Spreads"
    TIER = 1
    SOURCE_NOTEBOOK = "05-SPX Sector Rotation / RRG-Style Visualization.ipynb"
    CACHE_TTL = 3600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def render_sidebar(self):
        st.markdown("**Analysis Configuration**")
        selected, selected_data = render_universe_multiselect(
            key_prefix="cal",
            show_benchmark=True,
            label="Select Universes:",
            default_names=["Calendar Candidates", "SPX Sectors"],
        )
        
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        scan_clicked = st.button("🚀 Run Multi-Universe Analysis", use_container_width=True, type="primary")
        
        if scan_clicked:
            if not selected:
                st.sidebar.error("Please select at least one universe.")
            else:
                st.session_state["cal_state"] = {
                    "universes": selected,
                    "universe_data": selected_data,
                }

    def render_main(self):
        state = st.session_state.get("cal_state", {"universes": ["Calendar Candidates", "SPX Sectors"], "universe_data": {}})
        selected_universes = state["universes"]
        universe_data = state.get("universe_data", {})
        
        self.render_section_header(
            "📅 Calendar Option Strategy Rotation Matrix",
            "Multi-Universe Comparative Relative Strength & Momentum Analysis"
        )
        
        if not selected_universes:
            st.info("Select one or more universes from the sidebar to begin.")
            return

        MULTI_RESULTS = {}
        with st.spinner("Analyzing universes and calculating calendar scores..."):
            for univ in selected_universes:
                data = universe_data.get(univ)
                if not data:
                    data = UNIVERSES.get(univ, {})
                    data = {
                        "tickers": list(data.get("tickers", {}).keys()),
                        "benchmark": data.get("benchmark", "SPY"),
                    }
                res = analyze_universe(
                    univ,
                    tuple(data.get("tickers", [])),
                    data.get("benchmark", "SPY"),
                )
                if res: MULTI_RESULTS[univ] = res

        if not MULTI_RESULTS:
            st.error("Failed to compute data for the selected universes.")
            return

        # Combine results
        all_scores = []
        for univ_name, res in MULTI_RESULTS.items():
            scores = res["scores"].copy()
            scores["universe"] = univ_name
            all_scores.append(scores)
            
        combined_scores = pd.concat(all_scores, ignore_index=True)
        combined_scores["calendar_score_normalized"] = combined_scores.groupby("universe")["calendar_score"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        combined_scores = combined_scores.sort_values("calendar_score", ascending=False).reset_index(drop=True)

        self._render_dashboard(MULTI_RESULTS, combined_scores)
        self._render_top_candidates(combined_scores)
        self._render_universe_summary(MULTI_RESULTS, combined_scores)
        self._render_interpretation_guide()

    def _render_dashboard(self, MULTI_RESULTS, combined_scores):
        universe_colors = {
            "Calendar Candidates": "#3B82F6",
            "SPX Sectors": "#10B981",
            "MAG 7": "#F59E0B",
            "Leading ETFs": "#EF4444"
        }
        palette = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4", "#EC4899", "#84CC16"]
        for idx, univ_name in enumerate(MULTI_RESULTS.keys()):
            universe_colors.setdefault(univ_name, palette[idx % len(palette)])

        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "<b>Q1: Rotation Matrix</b> (RS Ratio vs Momentum)",
                "<b>Q2: Trend Strength</b> (Top 12 by Score)",
                "<b>Q3: Quality Assessment</b> (Score vs Spot Price)",
                "<b>Q4: Liquidity Heatmap</b> (Top 15 Candidates)"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ],
            vertical_spacing=0.14,
            horizontal_spacing=0.10
        )

        # ----- QUADRANT 1: ROTATION MATRIX -----
        for univ_name, color in universe_colors.items():
            if univ_name not in MULTI_RESULTS:
                continue
            scores_top = MULTI_RESULTS[univ_name]["scores"].head(PLOT_TOP_N)
            if len(scores_top) == 0:
                continue

            fig.add_trace(
                go.Scatter(
                    x=scores_top["rs_ratio"],
                    y=scores_top["rs_momentum"],
                    mode="markers+text",
                    name=univ_name,
                    text=scores_top["ticker"],
                    textposition="top center",
                    textfont=dict(size=9, color="white"),
                    marker=dict(size=10, color=color, opacity=0.75, line=dict(color="white", width=1)),
                    customdata=scores_top["fdts_signal"],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        f"Universe: {univ_name}<br>"
                        "RS Ratio: %{x:.2f}<br>"
                        "RS Momentum: %{y:.2f}<br>"
                        "FDTS Signal: %{customdata}<br>"
                        "<extra></extra>"
                    ),
                    legendgroup="rotation"
                ),
                row=1, col=1
            )

        fig.add_hline(y=100, line_color="#94A3B8", line_width=1, line_dash="dash", row=1, col=1)
        fig.add_vline(x=100, line_color="#94A3B8", line_width=1, line_dash="dash", row=1, col=1)

        # ----- QUADRANT 2: TREND STRENGTH BARS -----
        top_scores = combined_scores.head(12).copy()
        bar_colors = [universe_colors.get(u, "#3B82F6") for u in top_scores["universe"]]

        fig.add_trace(
            go.Bar(
                x=top_scores["ticker"],
                y=top_scores["trend_score"],
                name="Trend Score",
                marker=dict(color=bar_colors, line=dict(color="white", width=1)),
                text=top_scores["trend_score"].round(0).astype(int),
                textposition="outside",
                customdata=top_scores["fdts_signal"],
                hovertemplate="<b>%{x}</b><br>Trend Score: %{y:.1f}<br>FDTS Signal: %{customdata}<br><extra></extra>",
                showlegend=False
            ),
            row=1, col=2
        )

        # ----- QUADRANT 3: QUALITY SCATTER -----
        for quality_type, q_color in [("Best", "#10B981"), ("Watch", "#F59E0B")]:
            subset = combined_scores[combined_scores["quality"] == quality_type]
            if len(subset) == 0: continue

            fig.add_trace(
                go.Scatter(
                    x=subset["calendar_score"],
                    y=subset["spot"],
                    mode="markers+text",
                    name=f"{quality_type} Quality",
                    text=subset["ticker"],
                    textposition="top center",
                    textfont=dict(size=8, color="white"),
                    marker=dict(size=12, color=q_color, opacity=0.8, line=dict(color="white", width=1)),
                    customdata=subset["fdts_signal"],
                    hovertemplate=(
                        "<b>%{text}</b><br>Score: %{x:.1f}<br>Spot: $%{y:.2f}<br>"
                        f"Quality: {quality_type}<br>FDTS Signal: %{{customdata}}<br><extra></extra>"
                    ),
                    legendgroup="quality"
                ),
                row=2, col=1
            )

        fig.add_vline(x=75, line_color="#10B981", line_width=1, line_dash="dot", row=2, col=1)

        # ----- QUADRANT 4: LIQUIDITY HEATMAP -----
        heatmap_data = combined_scores.head(15).copy()
        if not heatmap_data.empty:
            z_data = np.array([
                heatmap_data["option_liquidity_score"].values,
                (heatmap_data["option_oi"] / heatmap_data["option_oi"].max() * 100).values,
                (heatmap_data["option_volume"] / heatmap_data["option_volume"].max() * 100).values
            ])

            fig.add_trace(
                go.Heatmap(
                    z=z_data,
                    x=heatmap_data["ticker"],
                    y=["Liquidity Score", "Option OI", "Option Volume"],
                    colorscale=[[0, "#1E293B"], [0.2, "#3B82F6"], [0.6, "#10B981"], [1.0, "#F59E0B"]],
                    showscale=True,
                    customdata=np.tile(heatmap_data["fdts_signal"].values, (3, 1)),
                    hovertemplate="<b>%{x}</b><br>%{y}: %{z:.0f}<br>FDTS Signal: %{customdata}<br><extra></extra>",
                    colorbar=dict(x=1.02, len=0.4, y=0.22, thickness=15, tickfont=dict(color="white", size=10))
                ),
                row=2, col=2
            )

        # ----- DASHBOARD LAYOUT -----
        fig.update_layout(
            title=dict(
                text="<b>4-Quadrant Multi-Universe Dashboard</b><br><sub style='color:#94A3B8'>Comparative Relative Strength & Momentum Analysis</sub>",
                x=0.5, xanchor="center", font=dict(size=18, color="#E2E8F0")
            ),
            paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
            font=dict(family="Inter, sans-serif", size=11, color="#E2E8F0"),
            height=850,
            showlegend=True,
            legend=dict(
                orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.08, 
                bgcolor="rgba(13,27,46,0.8)", bordercolor="#1e3a5f", borderwidth=1,
                font=dict(color="#E2E8F0")
            ),
            margin=dict(l=40, r=140, t=80, b=40)
        )

        fig.update_xaxes(title_text="RS Ratio %", row=1, col=1, gridcolor="#1e3a5f", zeroline=False)
        fig.update_yaxes(title_text="RS Momentum %", row=1, col=1, gridcolor="#1e3a5f", zeroline=False)
        fig.update_xaxes(title_text="Ticker", row=1, col=2, gridcolor="#1e3a5f")
        fig.update_yaxes(title_text="Trend Score (0-100)", row=1, col=2, gridcolor="#1e3a5f", range=[0, 110])
        fig.update_xaxes(title_text="Calendar Score", row=2, col=1, gridcolor="#1e3a5f")
        fig.update_yaxes(title_text="Spot Price ($)", row=2, col=1, gridcolor="#1e3a5f")
        fig.update_xaxes(title_text="Ticker", row=2, col=2)

        # Adjust subplot title colors for dark theme
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=14, color="#e2e8f0")

        st.plotly_chart(fig, use_container_width=True, theme=None)

    def _render_top_candidates(self, combined_scores):
        st.markdown("### 🏆 Top Rotation Candidates")
        display_cols = [
            "ticker", "universe", "quality", "fdts_signal", "calendar_score", "trend_score",
            "rs_ratio", "rs_momentum", "spot", "target_strike", "option_liquidity_score"
        ]
        display_df = combined_scores.head(15)[display_cols].copy()
        
        # We will use Streamlit's dataframe rendering since it natively handles styling well
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "fdts_signal": st.column_config.TextColumn("FDTS Signal", width="small"),
                "calendar_score": st.column_config.NumberColumn("Cal Score", format="%.1f"),
                "trend_score": st.column_config.NumberColumn("Trend Score", format="%.1f"),
                "rs_ratio": st.column_config.NumberColumn("RS Ratio", format="%.1f"),
                "rs_momentum": st.column_config.NumberColumn("RS Mom", format="%.1f"),
                "spot": st.column_config.NumberColumn("Spot Price", format="$%.2f"),
                "target_strike": st.column_config.NumberColumn("Target Strike", format="$%.2f"),
                "option_liquidity_score": st.column_config.NumberColumn("Liquidity", format="%.0f"),
                "quality": st.column_config.TextColumn("Quality")
            }
        )

    def _render_universe_summary(self, MULTI_RESULTS, combined_scores):
        st.markdown("### 📊 Universe Comparison Summary")
        summary_rows = []
        for univ_name in MULTI_RESULTS.keys():
            u_scores = combined_scores[combined_scores["universe"] == univ_name]
            if len(u_scores) == 0: continue
            summary_rows.append({
                "Universe": univ_name,
                "Tickers": len(u_scores),
                "Avg Score": u_scores["calendar_score"].mean(),
                "Top Score": u_scores["calendar_score"].max(),
                "Best Count": (u_scores["quality"] == "Best").sum(),
                "Watch Count": (u_scores["quality"] == "Watch").sum(),
                "Avg Trend": u_scores["trend_score"].mean(),
                "Avg RS Ratio": u_scores["rs_ratio"].mean(),
                "Avg Momentum": u_scores["rs_momentum"].mean(),
                "Top Ticker": u_scores.iloc[0]["ticker"] if len(u_scores) > 0 else "N/A"
            })

        summary_df = pd.DataFrame(summary_rows).set_index("Universe")
        st.dataframe(
            summary_df,
            use_container_width=True,
            column_config={
                "Avg Score": st.column_config.NumberColumn(format="%.1f"),
                "Top Score": st.column_config.NumberColumn(format="%.1f"),
                "Avg Trend": st.column_config.NumberColumn(format="%.1f"),
                "Avg RS Ratio": st.column_config.NumberColumn(format="%.1f"),
                "Avg Momentum": st.column_config.NumberColumn(format="%.1f"),
            }
        )

    def _render_interpretation_guide(self):
        st.markdown("### 📍 4-Quadrant Interpretation Guide")
        
        c1, c2 = st.columns(2)
        with c1:
            st.success("**↗ UPPER RIGHT (>100, >100) | LEADING**\n\nStrong relative strength + accelerating momentum. Top calendar spread candidates.")
            st.error("**↙ LOWER LEFT (<100, <100) | LAGGING**\n\nBoth weakening. Avoid for new calendar spread positions.")
        with c2:
            st.info("**↖ UPPER LEFT (<100, >100) | IMPROVING**\n\nMomentum accelerating but RS still lagging. Monitor for catch-up.")
            st.warning("**↘ LOWER RIGHT (>100, <100) | WEAKENING**\n\nStrong RS but momentum fading. Caution on new entries.")

        st.markdown("#### Scoring Components")
        st.markdown("- **Trend Score (40%)**: EMA alignment, MA crossovers, recent highs, volume confirmation\n- **RS Ratio (30%)**: Relative strength vs benchmark (100 = parity)\n- **RS Momentum (30%)**: Rate of change in relative strength\n- **Calendar Score**: Weighted composite. Score ≥ 75 = 'Best' candidate ✨")

        st.markdown("#### Quality Badges")
        st.markdown("- **✨ Best (Score ≥ 75)**: Ready for calendar spread execution. Primary candidates.\n- **👁️ Watch (Score < 75)**: Monitor closely. Wait for signal improvement.")
