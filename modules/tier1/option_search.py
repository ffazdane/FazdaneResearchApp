"""Focused option search for high-volume, tight-spread contracts."""

from __future__ import annotations

from datetime import datetime
import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.base_module import FazDaneModule
from modules.tier1.options_liquidity import fetch_options_data
from utils.tastytrade_provider import load_config
from utils.universe_manager import load_universes, render_universe_manager


logger = logging.getLogger("OptionSearch")


def _filter_option_search(
    df: pd.DataFrame,
    min_underlying_price: float,
    max_spread_pct: float,
    max_spread: float,
    prefer_tastytrade: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df.copy()
    for col in ["spot", "volume", "open_interest", "spread", "spread_pct", "bid", "ask"]:
        if col in filtered.columns:
            filtered[col] = pd.to_numeric(filtered[col], errors="coerce")

    if prefer_tastytrade and "data_source" in filtered.columns:
        tasty_rows = filtered["data_source"].astype(str).str.contains("Tastytrade", case=False, na=False)
        if tasty_rows.any():
            filtered = filtered[tasty_rows].copy()

    if "spot" in filtered.columns:
        filtered = filtered[filtered["spot"] >= float(min_underlying_price)]

    if "spread_pct" in filtered.columns:
        filtered = filtered[filtered["spread_pct"].notna() & (filtered["spread_pct"] <= float(max_spread_pct))]

    if max_spread > 0 and "spread" in filtered.columns:
        filtered = filtered[filtered["spread"].notna() & (filtered["spread"] <= float(max_spread))]

    if filtered.empty:
        return filtered

    sort_cols = [col for col in ["volume", "spread_pct", "open_interest"] if col in filtered.columns]
    ascending = [False if col != "spread_pct" else True for col in sort_cols]
    return filtered.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _ticker_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame()

    rows = []
    for symbol, group in df.groupby("symbol", dropna=True):
        top = group.sort_values("volume", ascending=False).iloc[0] if "volume" in group.columns else group.iloc[0]
        calls = group.loc[group.get("option_type") == "Call", "volume"].sum() if "option_type" in group.columns else 0
        puts = group.loc[group.get("option_type") == "Put", "volume"].sum() if "option_type" in group.columns else 0
        rows.append(
            {
                "symbol": symbol,
                "spot": group["spot"].median() if "spot" in group.columns else None,
                "contracts": len(group),
                "total_volume": group["volume"].sum() if "volume" in group.columns else None,
                "max_contract_volume": group["volume"].max() if "volume" in group.columns else None,
                "total_open_interest": group["open_interest"].sum() if "open_interest" in group.columns else None,
                "median_spread_pct": group["spread_pct"].median() if "spread_pct" in group.columns else None,
                "median_spread": group["spread"].median() if "spread" in group.columns else None,
                "call_volume": calls,
                "put_volume": puts,
                "top_contract": top.get("contract"),
                "top_expiration": top.get("expiration"),
                "top_strike": top.get("strike"),
                "top_type": top.get("option_type"),
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["total_volume", "median_spread_pct"],
        ascending=[False, True],
    ).reset_index(drop=True)


class OptionSearchModule(FazDaneModule):
    MODULE_NAME = "Option Search"
    MODULE_ICON = "OS"
    MODULE_DESCRIPTION = "Find tickers above $100 with high option volume and low spreads"
    TIER = 1
    CACHE_TTL = 300
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["Tastytrade API", "yfinance fallback"]

    def render_sidebar(self):
        st.markdown("**Search Universe**")
        if "os_sel" not in st.session_state and "Best Option Spread Tickers" in load_universes():
            st.session_state["os_sel"] = "Best Option Spread Tickers"
        universe_name, symbols, _ = render_universe_manager(
            key_prefix="os",
            show_benchmark=False,
            label="Ticker Universe:",
        )
        st.caption(f"{len(symbols)} symbols selected from {universe_name}.")

        st.markdown("**Contract Filters**")
        min_underlying_price = st.number_input(
            "Min Underlying Price",
            min_value=0.0,
            value=100.0,
            step=5.0,
            key="os_min_price",
        )
        min_volume = st.slider("Min Contract Volume", 0, 20000, 1000, 250, key="os_min_volume")
        min_oi = st.slider("Min Open Interest", 0, 50000, 1000, 500, key="os_min_oi")
        max_spread_pct = st.slider("Max Spread %", 1.0, 50.0, 10.0, 0.5, key="os_max_spread_pct")
        max_spread = st.number_input(
            "Max Spread $",
            min_value=0.0,
            value=0.50,
            step=0.05,
            help="Set to 0 to ignore the absolute spread filter.",
            key="os_max_spread",
        )
        option_types = st.multiselect(
            "Option Type",
            ["Call", "Put"],
            default=["Call", "Put"],
            key="os_types",
        )
        exp_pref = st.selectbox(
            "Expiration Window",
            ["Weekly (<=8 days)", "Monthly (9-45 days)", "Any"],
            index=1,
            key="os_exp",
        )

        st.markdown("**Data Source**")
        config = load_config()
        st.caption(f"Tastytrade: {'configured' if config.is_configured else 'not configured'}")
        prefer_tastytrade = st.checkbox(
            "Prefer Tastytrade rows",
            value=True,
            help="When tastytrade returns matches, hide fallback rows from the final search.",
            key="os_prefer_tt",
        )
        st.info(f"Active Source: {st.session_state.get('os_active_data_source', 'Not scanned yet')}")

        scan_clicked = st.button("Search Options", use_container_width=True, type="primary", key="os_scan")
        export_clicked = st.button("Export CSV", use_container_width=True, key="os_export")

        if scan_clicked:
            if not symbols:
                st.error("Select at least one ticker.")
            elif not option_types:
                st.error("Select at least one option type.")
            else:
                st.session_state["os_last_params"] = {
                    "symbols": tuple(symbols),
                    "min_volume": min_volume,
                    "min_oi": min_oi,
                    "option_types": tuple(option_types),
                    "exp_pref": exp_pref,
                    "min_underlying_price": min_underlying_price,
                    "max_spread_pct": max_spread_pct,
                    "max_spread": max_spread,
                    "prefer_tastytrade": prefer_tastytrade,
                }
                st.session_state.pop("os_results", None)
                st.session_state.pop("os_summary", None)

        if export_clicked and "os_results" in st.session_state:
            df = st.session_state["os_results"]
            if not df.empty:
                st.download_button(
                    "Download Now",
                    data=df.to_csv(index=False),
                    file_name=f"option_search_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    key="os_dl",
                )

    def render_main(self):
        self.render_section_header(
            "Option Search",
            "High-volume, low-spread option contracts on underlyings above $100",
        )

        if "os_last_params" not in st.session_state:
            self._render_welcome()
            return

        params = st.session_state["os_last_params"]
        scan_params = {
            key: params[key]
            for key in ["symbols", "min_volume", "min_oi", "option_types", "exp_pref"]
        }

        if "os_results" not in st.session_state:
            with st.spinner(f"Searching {len(params['symbols'])} tickers..."):
                raw = fetch_options_data(**scan_params)
                source = raw.attrs.get("active_data_source", "No provider status")
                if "data_source" in raw.columns and not raw.empty:
                    source = ", ".join(sorted(raw["data_source"].dropna().unique()))

                results = _filter_option_search(
                    raw,
                    params["min_underlying_price"],
                    params["max_spread_pct"],
                    params["max_spread"],
                    params["prefer_tastytrade"],
                )
                st.session_state["os_results"] = results
                st.session_state["os_summary"] = _ticker_summary(results)
                st.session_state["os_active_data_source"] = source
                st.rerun()

        df = st.session_state["os_results"]
        summary = st.session_state.get("os_summary", pd.DataFrame())

        if df.empty:
            st.warning("No contracts matched the price, volume, and spread filters.")
            st.info(f"Provider status: {st.session_state.get('os_active_data_source', 'No provider status')}")
            return

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tickers Found", f"{df['symbol'].nunique():,}" if "symbol" in df.columns else "0")
        m2.metric("Contracts", f"{len(df):,}")
        m3.metric("Avg Volume", f"{int(df['volume'].mean()):,}" if "volume" in df.columns else "-")
        m4.metric("Median Spread", f"{df['spread_pct'].median():.1f}%" if "spread_pct" in df.columns else "-")
        st.caption(f"Data source: {st.session_state.get('os_active_data_source', 'Unknown')}")

        tab1, tab2, tab3 = st.tabs(["Ticker Ranking", "Contracts", "Volume Map"])
        with tab1:
            self._render_summary(summary)
        with tab2:
            self._render_contracts(df)
        with tab3:
            self._render_volume_map(df)

    def _render_summary(self, summary: pd.DataFrame):
        st.markdown("### Tickers Above $100 With Tight Option Markets")
        if summary.empty:
            st.info("No ticker summary is available.")
            return
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "spot": st.column_config.NumberColumn("Spot", format="$%.2f"),
                "total_volume": st.column_config.NumberColumn("Total Volume", format="%d"),
                "max_contract_volume": st.column_config.NumberColumn("Max Contract Vol.", format="%d"),
                "total_open_interest": st.column_config.NumberColumn("Open Interest", format="%d"),
                "median_spread_pct": st.column_config.NumberColumn("Median Spread %", format="%.1f%%"),
                "median_spread": st.column_config.NumberColumn("Median Spread $", format="$%.2f"),
            },
        )

    def _render_contracts(self, df: pd.DataFrame):
        st.markdown("### Matching Contracts")
        symbols = sorted(df["symbol"].dropna().unique()) if "symbol" in df.columns else []
        selected = st.multiselect("Filter Symbol", symbols, key="os_contract_symbol_filter")
        display = df[df["symbol"].isin(selected)].copy() if selected else df.copy()
        display_cols = [
            col for col in [
                "symbol", "spot", "option_type", "expiration", "dte", "strike",
                "volume", "open_interest", "bid", "ask", "spread", "spread_pct",
                "last_price", "contract", "data_source",
            ]
            if col in display.columns
        ]
        st.dataframe(display[display_cols], use_container_width=True, hide_index=True, height=520)

    def _render_volume_map(self, df: pd.DataFrame):
        if not {"symbol", "volume", "spread_pct", "option_type"}.issubset(df.columns):
            st.info("Volume map needs symbol, volume, spread, and option type fields.")
            return

        plot_df = df.sort_values("volume", ascending=False).head(250)
        fig = px.scatter(
            plot_df,
            x="symbol",
            y="volume",
            size="volume",
            color="spread_pct",
            symbol="option_type",
            color_continuous_scale=["#3ab54a", "#f59e0b", "#ef4444"],
            hover_data=["option_type", "expiration", "strike", "bid", "ask", "spread_pct"],
            labels={"spread_pct": "Spread %", "volume": "Contract Volume"},
        )
        fig.update_layout(
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#152847",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#1e3a5f"),
            yaxis=dict(gridcolor="#1e3a5f"),
            margin=dict(l=0, r=0, t=20, b=0),
            height=460,
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_welcome(self):
        st.markdown(
            """
            <div style="
                background:rgba(21,40,71,0.58);
                border:1px solid #1e3a5f;
                border-left:4px solid #3ab54a;
                border-radius:8px;
                padding:24px 28px;
                margin-top:16px;
            ">
                <div style="color:#3ab54a;font-size:20px;font-weight:700;margin-bottom:8px;">
                    Option Search
                </div>
                <div style="color:#94a3b8;font-size:14px;line-height:1.7;">
                    Search a liquid options universe for contracts with strong volume, tight spreads,
                    and underlying prices above $100. Configure the filters in the sidebar and run the search.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
