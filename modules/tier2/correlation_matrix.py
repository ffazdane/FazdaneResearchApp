"""
FazDane Analytics - Tier 2
Correlation Matrix
"""

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from pandas.io.formats.style import Styler

from modules.base_module import FazDaneModule
from utils.universe_manager import get_universe_names, render_universe_manager


DEFAULT_RENAME_MAP = {
    "SPY": "S&P 500",
    "QQQ": "NASDAQ",
    "IWM": "RUSSELL",
    "^VIX": "VOLATILITY",
    "TLT": "BONDS",
    "GLD": "GOLD",
    "CL=F": "OIL",
    "BTC=F": "BITCOIN",
    "DX-Y.NYB": "USD",
    "HG=F": "COPPER",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_close_prices(tickers: tuple[str, ...], start_date, end_date) -> pd.DataFrame:
    prices = pd.DataFrame()
    for symbol in tickers:
        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )
            if data.empty:
                continue

            close = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            prices[symbol] = close
        except Exception:
            continue

    return prices.dropna(how="all")


def compute_correlation(prices: pd.DataFrame, method: str) -> pd.DataFrame:
    returns = prices.ffill().pct_change().dropna(how="all")
    return returns.corr(method=method)


def format_correlation_table(corr: pd.DataFrame) -> Styler:
    table = corr.copy()
    table.index.name = "Symbol"
    return (
        table.style.format("{:.0%}")
        .set_properties(
            **{
                "color": "#ffffff",
                "font-size": "15px",
                "font-weight": "700",
                "text-align": "center",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#1f6f8b"),
                        ("color", "#ffffff"),
                        ("font-size", "15px"),
                        ("font-weight", "700"),
                        ("text-align", "center"),
                    ],
                }
            ]
        )
    )


class CorrelationMatrixModule(FazDaneModule):
    MODULE_NAME = "Correlation Matrix"
    MODULE_ICON = "🧮"
    MODULE_DESCRIPTION = "Cross-asset return correlation heatmap"
    TIER = 2
    SOURCE_NOTEBOOK = "Colab Correlation Matrix"
    CACHE_TTL = 3600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def render_sidebar(self):
        self._default_correlation_universe()
        st.markdown("**Correlation Universe**")
        self.universe_name, tickers, _ = render_universe_manager(
            key_prefix="corr",
            show_benchmark=False,
            label="Ticker Universe:",
        )
        self.tickers = tickers
        st.caption(f"{len(self.tickers)} assets selected from {self.universe_name}.")

        st.markdown("**Date Range**")
        today = datetime.today().date()
        default_start = today - pd.DateOffset(months=3)
        self.start_date = st.date_input("Start Date:", value=default_start.date(), key="corr_start")
        self.end_date = st.date_input("End Date:", value=today, key="corr_end")

        st.markdown("**Calculation**")
        self.method = st.selectbox("Correlation Method:", ["pearson", "spearman", "kendall"], index=0, key="corr_method")
        self.use_friendly_names = st.checkbox("Use friendly asset names", value=True, key="corr_friendly")

        if st.button("Refresh Correlations", use_container_width=True, type="primary", key="corr_refresh"):
            fetch_close_prices.clear()
            st.rerun()

    def render_main(self):
        self.render_section_header(
            "🧮 Correlation Matrix",
            "Cross-asset return correlations from the selected ticker universe",
        )

        if len(self.tickers) < 2:
            st.warning("Select at least two tickers for correlation analysis.")
            return

        if self.start_date >= self.end_date:
            st.warning("Start date must be before end date.")
            return

        with st.spinner(f"Fetching price history for {len(self.tickers)} assets..."):
            prices = fetch_close_prices(tuple(self.tickers), self.start_date, self.end_date)

        if prices.empty or prices.shape[1] < 2:
            st.warning("Not enough data was returned to compute a correlation matrix.")
            return

        corr = compute_correlation(prices, self.method)
        if self.use_friendly_names:
            corr = corr.rename(columns=DEFAULT_RENAME_MAP, index=DEFAULT_RENAME_MAP)
        asset_labels = corr.columns.tolist()
        corr = corr.reindex(index=asset_labels, columns=asset_labels)

        m1, m2, m3 = st.columns(3)
        m1.metric("Universe", self.universe_name)
        m2.metric("Assets Used", str(corr.shape[0]))
        m3.metric("Observations", str(max(len(prices.dropna(how="all")) - 1, 0)))

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=asset_labels,
                y=asset_labels,
                zmin=-1,
                zmax=1,
                zmid=0,
                colorscale=[
                    [0.0, "#ef4444"],
                    [0.5, "#facc15"],
                    [1.0, "#22c55e"],
                ],
                text=(corr * 100).round(0).astype(int).astype(str).add("%").values,
                texttemplate="%{text}",
                hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{customdata:.1f}%<extra></extra>",
                customdata=(corr * 100).values,
                colorbar=dict(title="Corr"),
                xgap=1,
                ygap=1,
            )
        )
        fig.update_traces(textfont=dict(size=16, color="#111827", family="Arial Black"))
        fig.update_layout(
            height=max(520, 42 * len(corr.index)),
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#0d1b2e",
            font=dict(color="#e2e8f0", size=16, family="Arial Black"),
            margin=dict(l=140, r=20, t=80, b=20),
            xaxis=dict(
                side="top",
                tickmode="array",
                tickvals=asset_labels,
                ticktext=asset_labels,
                tickangle=-35,
                tickfont=dict(size=15, family="Arial Black"),
                automargin=True,
            ),
            yaxis=dict(
                autorange="reversed",
                tickmode="array",
                tickvals=asset_labels,
                ticktext=asset_labels,
                tickfont=dict(size=15, family="Arial Black"),
                automargin=True,
            ),
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

        st.markdown("### Correlation Table")
        st.dataframe(
            format_correlation_table(corr),
            use_container_width=True,
            height=min(760, 38 * (len(corr.index) + 1)),
        )

        st.download_button(
            "Download Correlations CSV",
            data=corr.to_csv(index=True),
            file_name=f"correlations_{self.universe_name.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    def _default_correlation_universe(self):
        key = "corr_sel"
        target = "Correlation Matrix Assets"
        names = get_universe_names()
        if target in names and key not in st.session_state:
            st.session_state[key] = target
