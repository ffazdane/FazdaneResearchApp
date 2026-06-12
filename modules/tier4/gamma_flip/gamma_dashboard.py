"""Streamlit dashboard for Gamma Flip Line / GEX Engine."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from modules.base_module import FazDaneModule
from utils.universe_manager import format_ticker_display, get_ticker_names, render_universe_manager

from .data_loader import get_available_expirations, load_option_chain
from .export import analysis_to_excel
from .gex_engine import build_gex_analysis
from .visualization import expiration_heatmap, net_gex_by_strike_chart, simulated_gex_chart


class GammaFlipLineModule(FazDaneModule):
    MODULE_NAME = "Gamma Flip Line Module"
    MODULE_ICON = "GEX"
    MODULE_DESCRIPTION = "Dealer Gamma Exposure & Volatility Regime Dashboard"
    TIER = 4
    SOURCE_NOTEBOOK = "Gamma Flip Line / GEX Engine"
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def render_sidebar(self):
        st.markdown("**Dealer Gamma Exposure**")
        universe_name, tickers_list, _ = render_universe_manager(
            key_prefix="gex_universe",
            show_benchmark=False,
            label="Ticker Universe:",
        )
        ticker_names = get_ticker_names(universe_name)
        tradeable = tickers_list or ["SPY"]
        previous = st.session_state.get("gex_selected_ticker", "SPY")
        index = tradeable.index(previous) if previous in tradeable else 0
        selected_ticker = st.selectbox(
            "Instrument:",
            options=tradeable,
            index=index,
            key="gex_selected_ticker",
            format_func=lambda ticker: format_ticker_display(ticker, ticker_names),
        )
        custom_ticker = st.text_input(
            "Or Enter Custom Ticker:",
            value=st.session_state.get("gex_custom_ticker", ""),
            placeholder="e.g. SPY, QQQ, AAPL...",
            key="gex_custom_ticker",
        ).strip().upper()
        ticker = custom_ticker or selected_ticker or "SPY"
        st.session_state["gex_ticker"] = ticker.strip().upper()

        if st.button("Refresh Gamma Data", width="stretch", key="gex_refresh"):
            st.cache_data.clear()
            st.rerun()

        expirations = get_available_expirations(ticker)
        st.session_state["gex_available_expirations"] = expirations
        mode = st.selectbox(
            "Expiration Selector",
            ["All expirations", "0DTE only", "Weekly", "Monthly", "Custom selected expirations"],
            key="gex_expiration_mode",
        )

        custom = []
        if mode == "Custom selected expirations":
            custom = st.multiselect("Expirations", options=expirations, default=expirations[:4], key="gex_custom_expirations")

        st.slider("Simulation Range (+/- %)", 2.0, 30.0, 10.0, 0.5, key="gex_range_pct")
        st.slider("Simulation Step (%)", 0.1, 2.0, 0.5, 0.1, key="gex_step_pct")
        st.session_state["gex_custom_selected"] = tuple(custom)

    def render_main(self):
        st.markdown(
            """
            <style>
            .gex-callout { background: linear-gradient(135deg, rgba(26,58,143,0.26), rgba(58,181,74,0.10)); border: 1px solid #1e3a5f; border-left: 4px solid #3ab54a; border-radius: 10px; padding: 16px 18px; margin-bottom: 18px; }
            .gex-title { color: #3ab54a; font-size: 22px; font-weight: 800; margin-bottom: 4px; }
            .gex-subtitle { color: #94a3b8; font-size: 13px; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="gex-callout">
                <div class="gex-title">Gamma Flip Line Module</div>
                <div class="gex-subtitle">Dealer Gamma Exposure & Volatility Regime Dashboard</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ticker = st.session_state.get("gex_ticker", "SPY").strip().upper() or "SPY"
        expirations = st.session_state.get("gex_available_expirations") or get_available_expirations(ticker)
        selected = self._select_expirations(expirations)
        if not expirations:
            st.warning(f"No yfinance option expirations found for {ticker}. Try another ticker or refresh later.")
            return
        if not selected:
            st.warning("No expirations match the selected filter.")
            return

        with st.spinner(f"Loading {ticker} option chains..."):
            result = load_option_chain(ticker, tuple(selected))

        for warning in result.warnings:
            st.warning(warning)
        if result.chain.empty:
            st.info("No usable option chain rows were available after filtering.")
            return

        analysis = build_gex_analysis(
            result.chain,
            result.ticker,
            result.spot_price,
            st.session_state.get("gex_range_pct", 10.0),
            st.session_state.get("gex_step_pct", 0.5),
        )
        summary = analysis["summary"].iloc[0].to_dict()
        self._render_summary(summary, analysis["message"], len(selected))

        tab1, tab2, tab3 = st.tabs(["GEX Charts", "Tables", "Export"])
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    net_gex_by_strike_chart(
                        analysis["by_strike"],
                        summary["Spot Price"],
                        summary["Gamma Flip Line"],
                        summary["Call Wall"],
                        summary["Put Wall"],
                    ),
                    width="stretch",
                )
            with col2:
                st.plotly_chart(
                    simulated_gex_chart(analysis["simulation"], summary["Spot Price"], summary["Gamma Flip Line"]),
                    width="stretch",
                )
            with st.expander("Strike x Expiration GEX Heatmap", expanded=False):
                st.plotly_chart(expiration_heatmap(analysis["gex_rows"], summary["Spot Price"]), width="stretch")

        with tab2:
            st.markdown("### Summary Table")
            st.dataframe(self._format_numeric_table(analysis["summary"]), width="stretch", hide_index=True)
            st.markdown("### GEX by Strike")
            st.dataframe(self._format_numeric_table(analysis["by_strike"]), width="stretch", hide_index=True)
            st.markdown("### GEX by Expiration")
            st.dataframe(self._format_numeric_table(analysis["by_expiration"]), width="stretch", hide_index=True)

        with tab3:
            excel_bytes = analysis_to_excel(analysis)
            st.download_button(
                "Export Gamma Flip Analysis (.xlsx)",
                data=excel_bytes,
                file_name=f"FazDane_GEX_{ticker}_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

    def _select_expirations(self, expirations: list[str]) -> list[str]:
        mode = st.session_state.get("gex_expiration_mode", "All expirations")
        exp_dates = pd.to_datetime(pd.Series(expirations), errors="coerce")
        today = pd.Timestamp.today().normalize()
        if mode == "All expirations":
            return expirations
        if mode == "Custom selected expirations":
            return list(st.session_state.get("gex_custom_selected", ()))
        dte = (exp_dates.dt.normalize() - today).dt.days
        if mode == "0DTE only":
            return [exp for exp, days in zip(expirations, dte) if days == 0]
        if mode == "Weekly":
            return [exp for exp, days in zip(expirations, dte) if 0 <= days <= 8]
        if mode == "Monthly":
            monthly = []
            for exp, exp_date in zip(expirations, exp_dates):
                if pd.isna(exp_date):
                    continue
                if exp_date.weekday() == 4 and 15 <= exp_date.day <= 21:
                    monthly.append(exp)
            return monthly
        return expirations

    def _render_summary(self, summary: dict, message: str, expiration_count: int):
        metric_cols = st.columns(4)
        metric_cols[0].metric("Spot Price", f"${summary['Spot Price']:,.2f}")
        metric_cols[1].metric("Net GEX", f"{summary['Net GEX']:,.0f}")
        flip = summary.get("Gamma Flip Line")
        metric_cols[2].metric("Gamma Flip", "N/A" if pd.isna(flip) else f"${flip:,.2f}")
        metric_cols[3].metric("Regime", summary["Gamma Regime"])

        metric_cols = st.columns(4)
        dist = summary.get("Distance to Flip %")
        metric_cols[0].metric("Distance to Flip", "N/A" if pd.isna(dist) else f"{dist:+.2f}%")
        metric_cols[1].metric("Call Wall", "N/A" if pd.isna(summary.get("Call Wall")) else f"${summary['Call Wall']:,.2f}")
        metric_cols[2].metric("Put Wall", "N/A" if pd.isna(summary.get("Put Wall")) else f"${summary['Put Wall']:,.2f}")
        metric_cols[3].metric("Expirations", expiration_count)

        if summary["Gamma Regime"] == "Positive Gamma":
            st.success(message)
        elif summary["Gamma Regime"] == "Negative Gamma":
            st.error(message)
        elif summary["Gamma Regime"] == "Transition Zone":
            st.warning(message)
        else:
            st.info(message)

    def _format_numeric_table(self, table: pd.DataFrame) -> pd.DataFrame:
        formatted = table.copy()
        for col in formatted.columns:
            if pd.api.types.is_numeric_dtype(formatted[col]):
                formatted[col] = formatted[col].map(lambda value: None if pd.isna(value) else round(float(value), 2))
        return formatted
