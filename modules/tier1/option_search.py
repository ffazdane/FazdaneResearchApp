"""Focused option search module with SQLite database persistence, scoring, and multi-screen analysis."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import os
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.base_module import FazDaneModule
from modules.tier1.options_liquidity import fetch_options_data
from utils.tastytrade_provider import load_config
from utils.universe_manager import format_ticker_display

from modules.tier1.option_search_db import (
    init_option_search_db,
    create_search_run,
    save_ticker_summary,
    save_contract_history,
    get_active_ticker_list,
    get_comma_delimited_tickers,
    deactivate_stale_tickers,
    get_universe_summary,
    get_new_tickers_for_run,
    get_requalified_tickers_for_run,
    get_distinct_contract_tickers,
    get_contracts_from_db,
)

logger = logging.getLogger("OptionSearch")


def _filter_option_search(
    df: pd.DataFrame,
    min_underlying_price: float,
    max_spread_pct: float,
    max_spread: float,
    min_volume: int,
    min_oi: int,
    min_dte: int,
    max_dte: int,
    prefer_tastytrade: bool = True,
) -> pd.DataFrame:
    """Filter raw contract rows to match the contract-level thresholds."""
    if df.empty:
        return df

    filtered = df.copy()

    # Normalize columns
    for col in ["spot", "volume", "open_interest", "spread", "spread_pct", "bid", "ask", "dte", "strike"]:
        if col in filtered.columns:
            filtered[col] = pd.to_numeric(filtered[col], errors="coerce")

    # Prefer Tastytrade filter
    if prefer_tastytrade and "data_source" in filtered.columns:
        tasty_rows = filtered["data_source"].astype(str).str.contains("Tastytrade", case=False, na=False)
        if tasty_rows.any():
            filtered = filtered[tasty_rows].copy()

    # Underlying Price filter
    if "spot" in filtered.columns:
        filtered = filtered[filtered["spot"] >= float(min_underlying_price)]

    # DTE filter
    if "dte" in filtered.columns:
        filtered = filtered[(filtered["dte"] >= min_dte) & (filtered["dte"] <= max_dte)]

    # Contract Volume and Open Interest filters
    if "volume" in filtered.columns:
        filtered = filtered[filtered["volume"] >= min_volume]
    if "open_interest" in filtered.columns:
        filtered = filtered[filtered["open_interest"] >= min_oi]

    # Bid & Ask & Mid constraints
    if "bid" in filtered.columns:
        filtered = filtered[filtered["bid"] > 0]
    if "ask" in filtered.columns:
        filtered = filtered[filtered["ask"] > 0]

    if "bid" in filtered.columns and "ask" in filtered.columns:
        filtered["mid_price"] = (filtered["bid"] + filtered["ask"]) / 2
        filtered = filtered[filtered["mid_price"] > 0]
        filtered["spread"] = filtered["ask"] - filtered["bid"]
        filtered["spread_pct"] = filtered["spread"] / filtered["mid_price"]

    # Filter spread thresholds
    if "spread_pct" in filtered.columns:
        filtered = filtered[filtered["spread_pct"].notna() & (filtered["spread_pct"] <= (float(max_spread_pct) / 100.0))]
    if max_spread > 0 and "spread" in filtered.columns:
        filtered = filtered[filtered["spread"].notna() & (filtered["spread"] <= float(max_spread))]

    if filtered.empty:
        return filtered

    # Add categorization columns
    # 1. Expiration Bucket
    def get_exp_bucket(d):
        if d <= 7:
            return "0-7D"
        elif d <= 14:
            return "8-14D"
        elif d <= 30:
            return "15-30D"
        return "30D+"

    filtered["expiration_bucket"] = filtered["dte"].map(get_exp_bucket)

    # 2. Spread Quality Label
    def get_spread_quality(sp_pct):
        if sp_pct <= 0.03:
            return "Excellent"
        elif sp_pct <= 0.05:
            return "Good"
        elif sp_pct <= 0.10:
            return "Acceptable"
        return "Avoid"

    filtered["spread_quality_label"] = filtered["spread_pct"].map(get_spread_quality)

    # 3. Weekly Candidate Flag
    filtered["weekly_candidate"] = filtered["dte"].map(lambda d: 1 if d <= 14 else 0)

    # Ensure all required columns exist
    filtered["ticker"] = filtered["symbol"]
    filtered["underlying_price"] = filtered["spot"]
    filtered["DTE"] = filtered["dte"]
    filtered["source"] = filtered.get("data_source", "Unknown")

    for col in ["implied_volatility", "delta", "gamma", "theta", "vega"]:
        if col not in filtered.columns:
            # check if they are in yfinance / tastytrade aliases
            alias = "impliedVolatility" if col == "implied_volatility" else col
            if alias in filtered.columns:
                filtered[col] = filtered[alias]
            elif col == "implied_volatility" and "iv_%" in filtered.columns:
                filtered[col] = filtered["iv_%"] / 100.0
            else:
                filtered[col] = None

    # Map to final output format columns
    final_cols = [
        "symbol", "ticker", "underlying_price", "expiration", "DTE", "expiration_bucket",
        "option_type", "strike", "bid", "ask", "mid_price", "spread", "spread_pct",
        "spread_quality_label", "volume", "open_interest", "implied_volatility",
        "delta", "gamma", "theta", "vega", "source", "weekly_candidate"
    ]
    available_cols = [c for c in final_cols if c in filtered.columns]
    filtered = filtered[available_cols].copy()

    # Sort
    sort_cols = [col for col in ["volume", "spread_pct", "open_interest"] if col in filtered.columns]
    ascending = [False if col != "spread_pct" else True for col in sort_cols]
    return filtered.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _ticker_summary(df: pd.DataFrame, min_oi: int) -> pd.DataFrame:
    """Aggregate qualified contracts by ticker and compute trade scoring."""
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()

    rows = []
    for ticker, group in df.groupby("ticker", dropna=True):
        spot = group["underlying_price"].iloc[0]
        contracts = len(group)
        tot_vol = group["volume"].sum()
        max_vol = group["volume"].max()
        tot_oi = group["open_interest"].sum()
        med_spread_dlr = group["spread"].median()
        med_spread_pct = group["spread_pct"].median()
        best_spread_pct = group["spread_pct"].min()
        worst_spread_pct = group["spread_pct"].max()

        calls = group.loc[group["option_type"] == "Call", "volume"].sum()
        puts = group.loc[group["option_type"] == "Put", "volume"].sum()
        
        call_pct = calls / tot_vol if tot_vol > 0 else 0.5
        put_pct = puts / tot_vol if tot_vol > 0 else 0.5

        # Bias logic
        if call_pct >= 0.65:
            bias = "Call Heavy"
        elif put_pct >= 0.65:
            bias = "Put Heavy"
        elif call_pct >= 0.55:
            bias = "Moderately Bullish"
        elif put_pct >= 0.55:
            bias = "Moderately Bearish"
        else:
            bias = "Neutral"

        # Top Contract
        top_contract_row = group.sort_values("volume", ascending=False).iloc[0]
        top_contract = f"{ticker} {top_contract_row['expiration']} {top_contract_row['strike']:.1f} {top_contract_row['option_type']}"
        top_strike = top_contract_row["strike"]
        top_type = top_contract_row["option_type"]
        top_exp = top_contract_row["expiration"]
        top_dte = top_contract_row["DTE"]

        # Weekly listed flag
        has_weekly = "Yes" if (group["weekly_candidate"] == 1).any() else "No"

        # ATM spread percentage (contract closest to spot)
        atm_contract = group.iloc[(group["strike"] - spot).abs().argsort()[:1]]
        atm_spread_pct = atm_contract["spread_pct"].values[0] if not atm_contract.empty else med_spread_pct

        # Spread quality label
        if med_spread_pct <= 0.03:
            spread_quality = "Excellent"
        elif med_spread_pct <= 0.05:
            spread_quality = "Good"
        elif med_spread_pct <= 0.10:
            spread_quality = "Acceptable"
        else:
            spread_quality = "Avoid"

        # Weekly listed flag DTE check
        weekly_listed_flag = "Yes" if (group["DTE"] <= 14).any() else "No"

        # Warning Flags
        flags = []
        if med_spread_pct > 0.10:
            flags.append("Wide Spread")
        if tot_oi < min_oi:
            flags.append("Low OI")
        if put_pct > 0.65:
            flags.append("Put Heavy")
        if call_pct > 0.65:
            flags.append("Call Heavy")
        if weekly_listed_flag == "No":
            flags.append("No Weekly")
        if contracts < 3:
            flags.append("Low Contract Count")
        
        warning_flags = " | ".join(flags) if flags else "Normal"

        rows.append({
            "Ticker": ticker,
            "Underlying Price": spot,
            "Qualified Contracts": contracts,
            "Total Option Volume": tot_vol,
            "Max Contract Volume": max_vol,
            "Total Open Interest": tot_oi,
            "Median Spread $": med_spread_dlr,
            "Median Spread %": med_spread_pct,
            "Best Spread %": best_spread_pct,
            "Worst Spread %": worst_spread_pct,
            "Call Volume": calls,
            "Put Volume": puts,
            "Call %": call_pct,
            "Put %": put_pct,
            "Call/Put Bias": bias,
            "Top Contract": top_contract,
            "Top Strike": top_strike,
            "Top Option Type": top_type,
            "Top Expiration": top_exp,
            "Top DTE": top_dte,
            "Weekly Listed": weekly_listed_flag,
            "ATM Spread %": atm_spread_pct,
            "Spread Quality": spread_quality,
            "Warning Flags": warning_flags
        })

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    # Scoring Framework
    # Percentile ranks (from 0 to 100)
    if len(summary) <= 1:
        summary["Liquidity Score"] = 100.0
        summary["Spread Score"] = 100.0
        summary["Open Interest Score"] = 100.0
    else:
        summary["Liquidity Score"] = summary["Total Option Volume"].rank(pct=True) * 100.0
        summary["Spread Score"] = summary["Median Spread %"].rank(pct=True, ascending=False) * 100.0
        summary["Open Interest Score"] = summary["Total Open Interest"].rank(pct=True) * 100.0

    summary["Weekly Score"] = summary["Weekly Listed"].map(lambda x: 100.0 if x == "Yes" else 0.0)

    # Call/Put Signal Score
    def get_signal_score(cp_bias):
        if cp_bias == "Call Heavy":
            return 100.0
        elif cp_bias == "Moderately Bullish":
            return 75.0
        elif cp_bias == "Neutral":
            return 50.0
        elif cp_bias == "Moderately Bearish":
            return 25.0
        else: # Put Heavy
            return 0.0

    summary["Call/Put Signal Score"] = summary["Call/Put Bias"].map(get_signal_score)

    # Final Composite Score
    summary["Option Trade Score"] = (
        summary["Liquidity Score"] * 0.30
        + summary["Spread Score"] * 0.25
        + summary["Open Interest Score"] * 0.20
        + summary["Weekly Score"] * 0.15
        + summary["Call/Put Signal Score"] * 0.10
    )

    # Sort
    summary = summary.sort_values(
        by=["Option Trade Score", "Median Spread %", "Total Option Volume", "Ticker"],
        ascending=[False, True, False, True]
    ).reset_index(drop=True)

    summary["Rank"] = summary.index + 1

    # Shift Rank to be the first column
    cols = ["Rank"] + [c for c in summary.columns if c != "Rank"]
    return summary[cols]


class OptionSearchModule(FazDaneModule):
    MODULE_NAME = "Option Search Universe Engine"
    MODULE_ICON = "OS"
    MODULE_DESCRIPTION = "Persistent options trading universe engine featuring SQLite history scans, multi-factor liquidity ranking, and flow bias analysis."
    TIER = 1
    CACHE_TTL = 300
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["Tastytrade API", "yfinance fallback", "SQLite DB"]

    def render_sidebar(self):
        st.markdown("**Search Universe**")
        st.caption("Scanning the highly liquid CBOE Weekly-Listed Options Universe (120+ assets).")
        
        # Hardcoded liquid weekly option listed tickers
        symbols = [
            # ETFs
            "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "SMH", "EEM", "FXI", "GDX", "GDXJ",
            "XLE", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XOP", "KRE",
            "ARKK", "USO", "UNG",
            # Megacap / Growth / Liquid Stocks
            "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "GOOG", "META", "NFLX", "AMD", "AVGO",
            "COIN", "MSTR", "HOOD", "PLTR", "CRM", "ADBE", "ORCL", "CRWD", "PANW", "JPM", "GS",
            "UNH", "LLY", "COST", "HD", "BA", "CAT", "DIS", "V", "MA", "BAC", "WFC", "C", "MS",
            "SCHW", "XOM", "CVX", "COP", "FCX", "NEM", "ABBV", "MRK", "PFE", "JNJ", "GILD", "AMGN",
            "NKE", "SBUX", "MCD", "WMT", "TGT", "PG", "KO", "PEP", "GE", "HON", "LMT", "RTX",
            "FDX", "UPS", "CSX", "UNP", "IBKR", "AAL", "DAL", "UAL", "LUV", "MAR", "F", "GM",
            "TM", "TSM", "ASML", "INTC", "QCOM", "TXN", "MU", "LRCX", "KLAC", "NXPI", "ADI",
            "ON", "MRVL", "ANET", "ROKU", "DKNG", "PINS", "SNAP", "NET", "DDOG", "ZS",
            "WDAY", "NOW", "SNOW", "TEAM", "MDB", "SOFI", "PYPL", "SQ", "BABA", "JD", "PDD",
            "BIDU", "NIO", "LI", "XPEV", "FUTU", "RIVN"
        ]

        custom_input = st.text_area("Custom Comma Tickers:", placeholder="NVDA,PLTR,TSLA", key="os_custom_input")
        if custom_input:
            custom_symbols = [s.strip().upper() for s in custom_input.split(",") if s.strip()]
            symbols = list(set(symbols) | set(custom_symbols))

        st.caption(f"Loaded {len(symbols)} unique tickers.")

        st.markdown("**Contract Filters**")
        min_price = st.number_input("Min Underlying Price", min_value=0.0, value=100.0, step=5.0, key="os_min_price")
        min_volume = st.slider("Min Contract Volume", 0, 10000, 1000, 250, key="os_min_vol")
        min_oi = st.slider("Min Open Interest", 0, 25000, 1000, 500, key="os_min_oi")
        max_spread = st.number_input("Max Spread $ (0 to disable)", min_value=0.0, value=0.50, step=0.05, key="os_max_spread")
        max_spread_pct = st.slider("Max Spread %", 1.0, 50.0, 10.0, 0.5, key="os_max_spread_pct")

        st.markdown("**Expiration Filters**")
        min_dte = st.slider("Min DTE", 0, 180, 0, 1, key="os_min_dte")
        max_dte = st.slider("Max DTE", 0, 180, 45, 5, key="os_max_dte")

        option_types = st.multiselect("Option Type", ["Call", "Put"], default=["Call", "Put"], key="os_types")

        st.markdown("**Universe Configuration**")
        stale_days = st.slider("Stale Universe Window (Days)", 5, 90, 30, 5, key="os_stale_days")
        save_db = st.checkbox("Save Scan Runs to Database", value=True, key="os_save_db")
        show_weekly_only = st.checkbox("Show Only Weekly Candidates", value=False, key="os_weekly_only")
        show_active_only = st.checkbox("Show Only Active DB Universe", value=False, key="os_active_only")

        prefer_tastytrade = st.checkbox("Prefer Tastytrade rows", value=True, key="os_prefer_tt")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        scan_clicked = st.button("Search & Update Universe", use_container_width=True, type="primary", key="os_scan")

        if scan_clicked:
            if not symbols:
                st.error("Select at least one ticker.")
            elif not option_types:
                st.error("Select at least one option type.")
            else:
                st.session_state["os_last_params"] = {
                    "symbols": tuple(symbols),
                    "min_underlying_price": min_price,
                    "min_volume": min_volume,
                    "min_oi": min_oi,
                    "max_spread": max_spread,
                    "max_spread_pct": max_spread_pct,
                    "min_dte": min_dte,
                    "max_dte": max_dte,
                    "option_types": tuple(option_types),
                    "prefer_tastytrade": prefer_tastytrade,
                    "stale_days": stale_days,
                    "save_db": save_db,
                    "show_weekly_only": show_weekly_only,
                    "show_active_only": show_active_only,
                }
                # Clear cached results to force rerun
                st.session_state.pop("os_results", None)
                st.session_state.pop("os_summary", None)
                st.session_state.pop("os_run_id", None)

        # Database Tickers copy section
        init_option_search_db()
        db_tickers = get_active_ticker_list()
        if db_tickers:
            st.divider()
            with st.expander("📋 Copy Active DB Tickers", expanded=False):
                st.caption("Copy the comma-delimited active tickers list currently stored in your database:")
                st.text_area("Active DB Tickers:", value=",".join(db_tickers), height=120, key="os_sidebar_db_tickers_copy")

    def render_main(self):
        self.render_section_header("Option Search Universe", "Maintain and search a persistent option trading universe of liquid contracts")

        # Initialize Database
        init_option_search_db()

        if "os_last_params" not in st.session_state:
            # Show welcome screen and history summary
            self._render_welcome_screen()
            return

        params = st.session_state["os_last_params"]
        
        # Load Scan results
        if "os_results" not in st.session_state:
            # Deactivate stale tickers in DB
            deactivate_stale_tickers(days=params["stale_days"])

            with st.spinner("Scanning option chains and resolving Greek variables..."):
                # Call batch options fetch with "Any" window so we can filter locally
                raw = fetch_options_data(
                    symbols=params["symbols"],
                    min_volume=params["min_volume"],
                    min_oi=params["min_oi"],
                    option_types=params["option_types"],
                    exp_pref="Any"
                )

                # Filter contracts
                filtered_df = _filter_option_search(
                    raw,
                    min_underlying_price=params["min_underlying_price"],
                    max_spread_pct=params["max_spread_pct"],
                    max_spread=params["max_spread"],
                    min_volume=params["min_volume"],
                    min_oi=params["min_oi"],
                    min_dte=params["min_dte"],
                    max_dte=params["max_dte"],
                    prefer_tastytrade=params["prefer_tastytrade"]
                )

                # Aggregate ticker summary
                summary_df = _ticker_summary(filtered_df, params["min_oi"])

                # Post-filters (weekly, active flags)
                if params["show_weekly_only"] and not summary_df.empty:
                    summary_df = summary_df[summary_df["Weekly Listed"] == "Yes"]
                    filtered_df = filtered_df[filtered_df["weekly_candidate"] == 1]

                if params["show_active_only"] and not summary_df.empty:
                    db_actives = get_active_ticker_list()
                    if db_actives:
                        summary_df = summary_df[summary_df["Ticker"].isin(db_actives)]
                        filtered_df = filtered_df[filtered_df["ticker"].isin(db_actives)]

                # Save search run to database
                source = raw.attrs.get("active_data_source", "Unknown Source")
                run_id = None
                if params["save_db"]:
                    try:
                        filters_dict = {
                            "min_underlying_price": params["min_underlying_price"],
                            "min_volume": params["min_volume"],
                            "min_oi": params["min_oi"],
                            "max_spread": params["max_spread"],
                            "max_spread_pct": params["max_spread_pct"],
                            "min_dte": params["min_dte"],
                            "max_dte": params["max_dte"],
                            "prefer_tastytrade": params["prefer_tastytrade"]
                        }
                        run_id = create_search_run(
                            filters_dict=filters_dict,
                            scanned_count=len(params["symbols"]),
                            qualified_ticker_count=len(summary_df),
                            qualified_contract_count=len(filtered_df),
                            notes=source
                        )
                        save_ticker_summary(run_id, summary_df)
                        save_contract_history(run_id, filtered_df)
                        try:
                            from utils.persistence import backup_database
                            backup_database("option_search", reason=f"Scan Run {run_id}")
                        except Exception as e:
                            logger.warning(f"Cloud backup failed for option_search: {e}")
                        st.session_state["os_save_status"] = f"✅ Database save successful! Run ID: {run_id}"
                    except Exception as e:
                        st.session_state["os_save_status"] = f"❌ Database save failed: {e}"
                        logger.error(f"Failed to save search run: {e}")
                else:
                    st.session_state["os_save_status"] = "ℹ️ Scan complete. Database save bypassed."

                st.session_state["os_results"] = filtered_df
                st.session_state["os_summary"] = summary_df
                st.session_state["os_active_data_source"] = source
                st.session_state["os_run_id"] = run_id
                st.rerun()

        # Retrieve session values
        filtered_df = st.session_state["os_results"]
        summary_df = st.session_state["os_summary"]
        source_note = st.session_state.get("os_active_data_source", "Unknown")
        save_status = st.session_state.get("os_save_status", "")
        run_id = st.session_state.get("os_run_id")

        if summary_df.empty:
            st.warning("⚠️ No tickers or contracts qualified under your current thresholds.")
            st.info("""
            **How to resolve this:**
            * Try relaxing DTE parameters or absolute dollar spread limits in the sidebar.
            * Decrease the contract volume or open interest floors.
            * Verify that tastytrade API is configured or fallbacks to yfinance are active.
            """)
            st.caption(f"Data status: {source_note}")
            return

        # Render Save message
        if save_status:
            st.success(save_status)

        # ----------------- TABS SETUP -----------------
        tabs = st.tabs([
            "📊 Executive Dashboard",
            "🏆 Ticker Ranking",
            "📋 Best Contracts",
            "📅 Weekly Candidates",
            "⚖️ Call/Put Flow",
            "🗺️ Volume Map",
            "🪐 Persistent Option Universe",
            "📥 Comma Export"
        ])

        with tabs[0]:
            self._render_dashboard_tab(summary_df, filtered_df, run_id)

        with tabs[1]:
            self._render_ticker_ranking_tab(summary_df)

        with tabs[2]:
            self._render_contracts_tab(filtered_df)

        with tabs[3]:
            self._render_weekly_candidates_tab(summary_df, filtered_df)

        with tabs[4]:
            self._render_flow_tab(summary_df)

        with tabs[5]:
            self._render_volume_map_tab(filtered_df)

        with tabs[6]:
            self._render_persistent_universe_tab()

        with tabs[7]:
            self._render_comma_export_tab(summary_df)

    # ---------------- TAB RENDERERS ----------------

    def _render_welcome_screen(self):
        st.markdown(
            """
            <div style="background:rgba(21,40,71,0.58); border-left:5px solid #3ab54a; padding:18px 25px; border-radius:8px; margin-bottom:25px;">
                <h4 style="color:#3ab54a; margin-top:0;">Option Search Universe Engine</h4>
                <p style="color:#cbd5e1; font-size:13.5px; margin:0; line-height:1.6;">
                    Analyze options chains to discover liquid tickers, score call/put flow biases, and track how the universe changes over time.
                    Configure the contract parameters in the sidebar and run the scan.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 🪐 Database Universe Status")
        db_stats = get_universe_summary()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Universe Tickers", f"{db_stats['total_active']}")
        c2.metric("Historical Scans Tickers", f"{db_stats['total_historical']}")
        c3.metric("New Today", f"{db_stats['new_today']}")
        c4.metric("Stale (Deactivated) Tickers", f"{db_stats['stale_count']}")

        st.divider()

        # Database Tickers copy section (Welcome Screen)
        active_tickers = get_active_ticker_list()
        if active_tickers:
            with st.expander("📋 Copy Active Database Ticker List (Comma Delimited)", expanded=True):
                st.caption("Copy this list to customize or merge with your search list:")
                st.text_area("Database Tickers:", value=",".join(active_tickers), height=100, key="os_welcome_db_tickers_copy")
            st.divider()

        cl1, cl2 = st.columns(2)
        with cl1:
            st.markdown("#### 🏆 Top Scoring Tickers (Last Scan)")
            if db_stats["highest_scoring"]:
                high_df = pd.DataFrame(db_stats["highest_scoring"])
                st.dataframe(
                    high_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "ticker": st.column_config.TextColumn("Ticker"),
                        "score": st.column_config.NumberColumn("Option Score", format="%.1f"),
                        "volume": st.column_config.NumberColumn("Volume", format="%d"),
                        "spread": st.column_config.NumberColumn("Spread %", format="%.2f%%")
                    }
                )
            else:
                st.info("No tickers scored yet. Please run a scan.")

        with cl2:
            st.markdown("#### 🔄 Most Frequently Qualified Tickers")
            if db_stats["most_frequent"]:
                freq_df = pd.DataFrame(db_stats["most_frequent"])
                st.dataframe(
                    freq_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "ticker": st.column_config.TextColumn("Ticker"),
                        "count": st.column_config.NumberColumn("Scan Matches Count"),
                        "score": st.column_config.NumberColumn("Last Option Score", format="%.1f")
                    }
                )
            else:
                st.info("No search records found yet.")

    def _render_dashboard_tab(self, summary_df: pd.DataFrame, filtered_df: pd.DataFrame, run_id: int | None):
        st.markdown("### Executive Options Dashboard")
        
        # Load DB statistics
        db_stats = get_universe_summary()
        
        # Compute Discoveries / Re-qualifications for run
        new_count = 0
        req_count = 0
        if run_id:
            new_count = len(get_new_tickers_for_run(run_id))
            req_count = len(get_requalified_tickers_for_run(run_id))

        best_ticker = summary_df.iloc[0]["Ticker"] if not summary_df.empty else "N/A"
        best_score = summary_df.iloc[0]["Option Trade Score"] if not summary_df.empty else 0.0

        # Best weekly candidate
        weekly_df = summary_df[summary_df["Weekly Listed"] == "Yes"]
        best_weekly = weekly_df.iloc[0]["Ticker"] if not weekly_df.empty else "None"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Qualified Tickers", f"{len(summary_df)}", f"New: {new_count} | Requalified: {req_count}")
        c2.metric("Qualified Contracts", f"{len(filtered_df)}")
        c3.metric("Leader Ticker (Score)", f"{best_ticker} ({best_score:.1f})")
        c4.metric("Best Weekly Ticker", f"{best_weekly}")

        st.divider()

        # Graphs
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🏆 Top 20 Option Trade Scores")
            top_scores = summary_df.head(20)
            fig_score = px.bar(
                top_scores,
                x="Ticker",
                y="Option Trade Score",
                color="Option Trade Score",
                color_continuous_scale="Viridis",
                labels={"Option Trade Score": "Score"},
                height=350
            )
            fig_score.update_layout(paper_bgcolor="#0d1b2e", plot_bgcolor="rgba(21,40,71,0.15)", font=dict(color="#e2e8f0"))
            st.plotly_chart(fig_score, use_container_width=True)

        with col2:
            st.markdown("##### 📈 Top 20 Options Volume")
            top_vol = summary_df.sort_values("Total Option Volume", ascending=False).head(20)
            fig_vol = px.bar(
                top_vol,
                x="Ticker",
                y="Total Option Volume",
                color="Total Option Volume",
                color_continuous_scale="Cividis",
                labels={"Total Option Volume": "Volume"},
                height=350
            )
            fig_vol.update_layout(paper_bgcolor="#0d1b2e", plot_bgcolor="rgba(21,40,71,0.15)", font=dict(color="#e2e8f0"))
            st.plotly_chart(fig_vol, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### 🛡️ Bid-Ask Spread Quality Distribution")
            quality_counts = summary_df["Spread Quality"].value_counts().reset_index()
            quality_counts.columns = ["Quality", "Count"]
            color_map = {"Excellent": "#22c55e", "Good": "#3ab54a", "Acceptable": "#f59e0b", "Avoid": "#ef4444"}
            fig_pie = px.pie(
                quality_counts,
                names="Quality",
                values="Count",
                color="Quality",
                color_discrete_map=color_map,
                height=320
            )
            fig_pie.update_layout(paper_bgcolor="#0d1b2e", font=dict(color="#e2e8f0"))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col4:
            st.markdown("##### 📅 Weekly Option Availability")
            weekly_counts = summary_df["Weekly Listed"].value_counts().reset_index()
            weekly_counts.columns = ["Weekly Listed", "Count"]
            fig_week = px.pie(
                weekly_counts,
                names="Weekly Listed",
                values="Count",
                hole=0.4,
                color="Weekly Listed",
                color_discrete_map={"Yes": "#38bdf8", "No": "#64748b"},
                height=320
            )
            fig_week.update_layout(paper_bgcolor="#0d1b2e", font=dict(color="#e2e8f0"))
            st.plotly_chart(fig_week, use_container_width=True)

    def _render_ticker_ranking_tab(self, summary_df: pd.DataFrame):
        st.markdown("### Options Trade Scoring Rankings")
        
        # Scatter Plot
        st.markdown("##### Spread vs Volume Trade Matrix")
        fig_scatter = px.scatter(
            summary_df,
            x="Median Spread %",
            y="Total Option Volume",
            size="Total Open Interest",
            color="Option Trade Score",
            hover_data=["Ticker", "Underlying Price", "Call/Put Bias", "Qualified Contracts"],
            color_continuous_scale="Plotly3",
            labels={"Median Spread %": "Median Spread %", "Total Option Volume": "Volume"},
            height=400
        )
        fig_scatter.update_layout(paper_bgcolor="#0d1b2e", plot_bgcolor="rgba(21,40,71,0.15)", font=dict(color="#e2e8f0"))
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Rankings Table
        st.markdown("##### Ranked Option Trading Tickers")
        
        # Display with conditional formatting or neat mapping
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Underlying Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "Option Trade Score": st.column_config.ProgressColumn("Composite Score", format="%.1f", min_value=0, max_value=100),
                "Liquidity Score": st.column_config.NumberColumn("Liq Score", format="%.0f"),
                "Spread Score": st.column_config.NumberColumn("Spread Score", format="%.0f"),
                "Open Interest Score": st.column_config.NumberColumn("OI Score", format="%.0f"),
                "Weekly Score": st.column_config.NumberColumn("Weekly Score", format="%.0f"),
                "Call/Put Signal Score": st.column_config.NumberColumn("Flow Score", format="%.0f"),
                "Total Option Volume": st.column_config.NumberColumn("Total Volume", format="%d"),
                "Total Open Interest": st.column_config.NumberColumn("Total OI", format="%d"),
                "Median Spread %": st.column_config.NumberColumn("Median Spread %", format="%.2f%%"),
                "Call %": st.column_config.NumberColumn("Call %", format="%.1%"),
                "Put %": st.column_config.NumberColumn("Put %", format="%.1%"),
                "Call/Put Bias": st.column_config.TextColumn("Bias"),
                "Weekly Listed": st.column_config.TextColumn("Weekly Listed"),
                "Top Contract": st.column_config.TextColumn("Top Contract"),
                "Warning Flags": st.column_config.TextColumn("Warning Flags")
            }
        )

    def _render_contracts_tab(self, filtered_df: pd.DataFrame):
        st.markdown("### Qualified Option Contracts Database")

        # Toggle between Current Scan and SQLite Database
        db_path = "data/option_search.db"
        db_exists = os.path.exists(db_path)
        
        source_mode = "Current Scan Results"
        if db_exists:
            source_mode = st.radio(
                "Data Source:",
                ["Current Scan Results", "Local SQLite Database (All Scans)"],
                horizontal=True,
                key="contracts_source_mode"
            )

        # Get list of tickers depending on mode
        if source_mode == "Local SQLite Database (All Scans)":
            db_tickers = get_distinct_contract_tickers()
            tickers_list = ["All"] + db_tickers
        else:
            tickers_list = ["All"] + sorted(filtered_df["ticker"].dropna().unique().tolist())

        # Tab-level filter columns
        c1, c2, c3 = st.columns(3)
        with c1:
            sel_ticker = st.selectbox(
                "Filter Ticker:",
                tickers_list,
                key="contract_sel_ticker",
                format_func=lambda t: "All" if t == "All" else format_ticker_display(t)
            )
        with c2:
            sel_type = st.selectbox("Option Type:", ["All", "Call", "Put"], key="contract_sel_type")
        with c3:
            # Determine base dataframe to compute quality labels and other fields
            if source_mode == "Local SQLite Database (All Scans)":
                base_df = get_contracts_from_db(ticker=sel_ticker)
            else:
                base_df = filtered_df.copy()
                if sel_ticker != "All":
                    base_df = base_df[base_df["ticker"] == sel_ticker]
            
            qualities = ["All"] + sorted(base_df["spread_quality_label"].dropna().unique().tolist()) if not base_df.empty else ["All"]
            sel_quality = st.selectbox("Spread Quality:", qualities, key="contract_sel_quality")

        # Determine stable slider bounds based on overall data source to avoid reset state bugs
        if source_mode == "Local SQLite Database (All Scans)":
            all_db_df = get_contracts_from_db(ticker="All")
            if not all_db_df.empty:
                min_dte_val = int(all_db_df["DTE"].min())
                max_dte_val = int(all_db_df["DTE"].max())
                min_vol_val = int(all_db_df["volume"].min())
                max_vol_val = int(all_db_df["volume"].max())
            else:
                min_dte_val, max_dte_val = 0, 180
                min_vol_val, max_vol_val = 0, 10000
        else:
            if not filtered_df.empty:
                min_dte_val = int(filtered_df["DTE"].min())
                max_dte_val = int(filtered_df["DTE"].max())
                min_vol_val = int(filtered_df["volume"].min())
                max_vol_val = int(filtered_df["volume"].max())
            else:
                min_dte_val, max_dte_val = 0, 180
                min_vol_val, max_vol_val = 0, 10000

        # Safety check for equal boundaries
        if min_dte_val == max_dte_val:
            max_dte_val = min_dte_val + 1
        if min_vol_val == max_vol_val:
            max_vol_val = min_vol_val + 1

        c4, c5 = st.columns(2)
        with c4:
            sel_dte_range = st.slider("DTE Range:", min_dte_val, max_dte_val, (min_dte_val, max_dte_val), key="contract_sel_dte")
        with c5:
            sel_vol = st.slider("Contract Volume Floor:", min_vol_val, max_vol_val, min_vol_val, key="contract_sel_vol")

        # Apply remaining filters on base_df
        display_df = base_df.copy()
        if sel_type != "All":
            display_df = display_df[display_df["option_type"] == sel_type]
        if sel_quality != "All":
            display_df = display_df[display_df["spread_quality_label"] == sel_quality]
        
        if not display_df.empty:
            display_df = display_df[
                (display_df["DTE"] >= sel_dte_range[0]) & 
                (display_df["DTE"] <= sel_dte_range[1]) & 
                (display_df["volume"] >= sel_vol)
            ]

        total_base_count = len(base_df)
        st.markdown(f"Showing **{len(display_df)}** of **{total_base_count}** contracts.")

        if display_df.empty:
            st.info("No contracts matched the selected filters.")
            return

        # Prepare column configs
        col_config = {
            "symbol": st.column_config.TextColumn("Symbol"),
            "ticker": st.column_config.TextColumn("Ticker"),
            "underlying_price": st.column_config.NumberColumn("Underlying", format="$%.2f"),
            "expiration": st.column_config.TextColumn("Expiration"),
            "DTE": st.column_config.NumberColumn("DTE"),
            "expiration_bucket": st.column_config.TextColumn("Bucket"),
            "option_type": st.column_config.TextColumn("Type"),
            "strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
            "bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
            "ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
            "mid_price": st.column_config.NumberColumn("Mid Price", format="$%.2f"),
            "spread": st.column_config.NumberColumn("Spread $", format="$%.2f"),
            "spread_pct": st.column_config.NumberColumn("Spread %", format="%.2f%%"),
            "spread_quality_label": st.column_config.TextColumn("Quality"),
            "volume": st.column_config.NumberColumn("Volume", format="%d"),
            "open_interest": st.column_config.NumberColumn("Open Interest", format="%d"),
            "implied_volatility": st.column_config.NumberColumn("IV %", format="%.2f%%"),
            "delta": st.column_config.NumberColumn("Delta", format="%.3f"),
            "gamma": st.column_config.NumberColumn("Gamma", format="%.4f"),
            "theta": st.column_config.NumberColumn("Theta", format="%.3f"),
            "vega": st.column_config.NumberColumn("Vega", format="%.3f"),
            "source": st.column_config.TextColumn("Source")
        }
        if "run_timestamp" in display_df.columns:
            col_config["run_timestamp"] = st.column_config.TextColumn("Recorded At")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config=col_config
        )

    def _render_weekly_candidates_tab(self, summary_df: pd.DataFrame, filtered_df: pd.DataFrame):
        st.markdown("### Weekly Option Trading Candidates (DTE <= 14)")
        
        # Filter Weekly candidates
        weeklies_tickers = summary_df[summary_df["Weekly Listed"] == "Yes"].copy()
        
        if weeklies_tickers.empty:
            st.info("No tickers qualified as weekly option candidates under current volume and spread limits.")
            return

        st.markdown("##### 📅 Weekly Leaderboard by Score")
        
        weekly_summary_rows = []
        for _, row in weeklies_tickers.iterrows():
            t = row["Ticker"]
            t_contracts = filtered_df[(filtered_df["ticker"] == t) & (filtered_df["weekly_candidate"] == 1)]
            if t_contracts.empty:
                continue
            
            top_w = t_contracts.sort_values("volume", ascending=False).iloc[0]
            w_vol = t_contracts["volume"].sum()
            w_oi = t_contracts["open_interest"].sum()
            w_spread_pct = t_contracts["spread_pct"].median()
            
            w_calls = t_contracts.loc[t_contracts["option_type"] == "Call", "volume"].sum()
            w_puts = t_contracts.loc[t_contracts["option_type"] == "Put", "volume"].sum()
            w_call_pct = w_calls / w_vol if w_vol > 0 else 0.5
            
            if w_call_pct >= 0.65:
                w_bias = "Call Heavy"
            elif w_call_pct <= 0.35:
                w_bias = "Put Heavy"
            elif w_call_pct >= 0.55:
                w_bias = "Bullish"
            elif w_call_pct <= 0.45:
                w_bias = "Bearish"
            else:
                w_bias = "Neutral"

            weekly_summary_rows.append({
                "Ticker": t,
                "Option Trade Score": row["Option Trade Score"],
                "Top Weekly Contract": f"{t} {top_w['expiration']} {top_w['strike']:.1f} {top_w['option_type']}",
                "Top Weekly Expiration": top_w["expiration"],
                "Top Weekly DTE": top_w["DTE"],
                "Weekly Call Volume": w_calls,
                "Weekly Put Volume": w_puts,
                "Weekly Bias": w_bias,
                "Weekly Median Spread %": w_spread_pct,
                "Weekly Total Open Interest": w_oi
            })
            
        weekly_df = pd.DataFrame(weekly_summary_rows)
        if weekly_df.empty:
            st.info("No contracts qualified within 14 DTE.")
            return

        weekly_df = weekly_df.sort_values("Option Trade Score", ascending=False).reset_index(drop=True)
        st.dataframe(
            weekly_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Option Trade Score": st.column_config.ProgressColumn("Composite Score", format="%.1f", min_value=0, max_value=100),
                "Weekly Median Spread %": st.column_config.NumberColumn("Weekly Median Spread %", format="%.2f%%"),
                "Weekly Call Volume": st.column_config.NumberColumn("Call Vol.", format="%d"),
                "Weekly Put Volume": st.column_config.NumberColumn("Put Vol.", format="%d"),
                "Weekly Total Open Interest": st.column_config.NumberColumn("Weekly OI", format="%d")
            }
        )

        st.divider()

        # Visuals
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🏆 Top Weekly Candidates by Score")
            fig_w_score = px.bar(
                weekly_df.head(15),
                x="Ticker",
                y="Option Trade Score",
                color="Option Trade Score",
                color_continuous_scale="Teal",
                height=350
            )
            fig_w_score.update_layout(paper_bgcolor="#0d1b2e", plot_bgcolor="rgba(21,40,71,0.15)", font=dict(color="#e2e8f0"))
            st.plotly_chart(fig_w_score, use_container_width=True)

        with c2:
            st.markdown("##### 📅 DTE Bucket Expirations Comparison")
            w_contracts = filtered_df[filtered_df["weekly_candidate"] == 1]
            bucket_counts = w_contracts["expiration_bucket"].value_counts().reset_index()
            bucket_counts.columns = ["Bucket", "Contracts Count"]
            fig_bucket = px.pie(
                bucket_counts,
                names="Bucket",
                values="Contracts Count",
                color="Bucket",
                color_discrete_map={"0-7D": "#ef4444", "8-14D": "#f59e0b"},
                height=350
            )
            fig_bucket.update_layout(paper_bgcolor="#0d1b2e", font=dict(color="#e2e8f0"))
            st.plotly_chart(fig_bucket, use_container_width=True)

    def _render_flow_tab(self, summary_df: pd.DataFrame):
        st.markdown("### Call/Put Volume Flow & Sentiment Bias")

        # Stacked Volume Comparison
        st.markdown("##### Call vs Put Option Volume Flow")
        balance = summary_df.copy().head(25)
        balance = balance[["Ticker", "Call Volume", "Put Volume"]].melt(
            id_vars="Ticker",
            value_vars=["Call Volume", "Put Volume"],
            var_name="Option Side",
            value_name="Contracts Volume"
        )
        balance["Option Side"] = balance["Option Side"].map({"Call Volume": "Call", "Put Volume": "Put"})
        fig_flow = px.bar(
            balance,
            x="Contracts Volume",
            y="Ticker",
            color="Option Side",
            orientation="h",
            barmode="stack",
            color_discrete_map={"Call": "#22c55e", "Put": "#ef4444"},
            height=500
        )
        fig_flow.update_layout(paper_bgcolor="#0d1b2e", plot_bgcolor="rgba(21,40,71,0.15)", font=dict(color="#e2e8f0"))
        st.plotly_chart(fig_flow, use_container_width=True)

        st.divider()

        # Sentiment Distribution
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("##### ⚖️ Sentiment Bias Distribution")
            bias_counts = summary_df["Call/Put Bias"].value_counts().reset_index()
            bias_counts.columns = ["Bias", "Count"]
            fig_bias = px.pie(
                bias_counts,
                names="Bias",
                values="Count",
                color="Bias",
                color_discrete_map={
                    "Call Heavy": "#22c55e",
                    "Moderately Bullish": "#86efac",
                    "Neutral": "#94a3b8",
                    "Moderately Bearish": "#fca5a5",
                    "Put Heavy": "#ef4444"
                },
                height=300
            )
            fig_bias.update_layout(paper_bgcolor="#0d1b2e", font=dict(color="#e2e8f0"))
            st.plotly_chart(fig_bias, use_container_width=True)

        with c2:
            st.markdown("##### 📋 Ticker Lists by Sentiment Heavyweights")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown("**🟢 Call Heavy (>=65% Call)**")
                ch_df = summary_df[summary_df["Call/Put Bias"] == "Call Heavy"][["Ticker", "Call %"]].head(10)
                st.dataframe(ch_df, use_container_width=True, hide_index=True, column_config={"Call %": st.column_config.NumberColumn(format="%.0%")})
            with sc2:
                st.markdown("**🔴 Put Heavy (>=65% Put)**")
                ph_df = summary_df[summary_df["Call/Put Bias"] == "Put Heavy"][["Ticker", "Put %"]].head(10)
                st.dataframe(ph_df, use_container_width=True, hide_index=True, column_config={"Put %": st.column_config.NumberColumn(format="%.0%")})
            with sc3:
                st.markdown("**⚪ Neutral (45-55% split)**")
                nt_df = summary_df[summary_df["Call/Put Bias"] == "Neutral"][["Ticker", "Call %"]].head(10)
                st.dataframe(nt_df, use_container_width=True, hide_index=True, column_config={"Call %": st.column_config.NumberColumn(format="%.0%")})

    def _render_volume_map_tab(self, filtered_df: pd.DataFrame):
        st.markdown("### Options Chain Strike-Volume Heatmap")
        
        # Filters for volume map
        c1, c2 = st.columns(2)
        with c1:
            y_axis_choice = st.radio("Y-Axis metric:", ["Strike Price", "Days to Expiration (DTE)"], horizontal=True, key="vol_map_y_choice")
        with c2:
            color_choice = st.radio("Color metric:", ["Spread Percentage (%)", "Volume"], horizontal=True, key="vol_map_color_choice")

        y_col = "strike" if y_axis_choice == "Strike Price" else "DTE"
        color_col = "spread_pct" if color_choice == "Spread Percentage (%)" else "volume"
        color_scale = ["#22c55e", "#f59e0b", "#ef4444"] if color_col == "spread_pct" else "Viridis"

        plot_df = filtered_df.sort_values("volume", ascending=False).head(300)

        fig_vol_map = px.scatter(
            plot_df,
            x="ticker",
            y=y_col,
            size="volume",
            color=color_col,
            symbol="option_type",
            color_continuous_scale=color_scale,
            hover_data=["expiration", "strike", "option_type", "bid", "ask", "open_interest", "source"],
            labels={"ticker": "Ticker", y_col: y_axis_choice, color_col: color_choice},
            height=500
        )
        fig_vol_map.update_traces(marker=dict(line=dict(width=1, color="#0d1b2e"), opacity=0.85))
        fig_vol_map.update_layout(paper_bgcolor="#0d1b2e", plot_bgcolor="rgba(21,40,71,0.15)", font=dict(color="#e2e8f0"))
        st.plotly_chart(fig_vol_map, use_container_width=True)

    def _render_persistent_universe_tab(self):
        st.markdown("### Database Persistent Ticker Universe")

        db_stats = get_universe_summary()

        # Metrics cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Universe Active", f"{db_stats['total_active']}")
        c2.metric("Universe Inactive (Stale)", f"{db_stats['stale_count']}")
        c3.metric("Discovery Count", f"{db_stats['total_historical']}")
        c4.metric("Scored Above 80", len([x for x in db_stats["highest_scoring"] if x["score"] >= 80]))

        st.divider()

        # Persistent table view
        st.markdown("##### 📋 SQLite option_ticker_universe Table")
        db_path = "data/option_search.db"
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                universe_df = pd.read_sql_query("SELECT * FROM option_ticker_universe ORDER BY ticker ASC;", conn)
                conn.close()
                
                st.dataframe(
                    universe_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "ticker": st.column_config.TextColumn("Ticker"),
                        "first_seen_date": st.column_config.TextColumn("First Seen"),
                        "last_seen_date": st.column_config.TextColumn("Last Seen"),
                        "last_price": st.column_config.NumberColumn("Last Price", format="$%.2f"),
                        "last_option_score": st.column_config.NumberColumn("Last Score", format="%.1f"),
                        "last_liquidity_score": st.column_config.NumberColumn("Last Liq", format="%.0f"),
                        "last_spread_score": st.column_config.NumberColumn("Last Spread Score", format="%.0f"),
                        "last_open_interest_score": st.column_config.NumberColumn("Last OI Score", format="%.0f"),
                        "last_spread_pct": st.column_config.NumberColumn("Last Spread %", format="%.2f%%"),
                        "last_total_volume": st.column_config.NumberColumn("Last Volume", format="%d"),
                        "last_open_interest": st.column_config.NumberColumn("Last OI", format="%d"),
                        "last_call_pct": st.column_config.NumberColumn("Last Call %", format="%.1%"),
                        "last_put_pct": st.column_config.NumberColumn("Last Put %", format="%.1%"),
                        "last_bias": st.column_config.TextColumn("Last Bias"),
                        "weekly_listed_flag": st.column_config.TextColumn("Weekly Flag"),
                        "qualified_count": st.column_config.NumberColumn("Times Qualified"),
                        "active_flag": st.column_config.NumberColumn("Active Flag")
                    }
                )
            except Exception as e:
                st.error(f"Error loading universe database: {e}")
        else:
            st.info("Database file not discovered. Run a search to create it.")

    def _render_comma_export_tab(self, summary_df: pd.DataFrame):
        st.markdown("### 📥 Option Trading Universe Tickers Export")

        # 1. Current Scan qualified tickers
        current_scan_tickers = summary_df["Ticker"].dropna().astype(str).str.upper().unique().tolist()
        current_scan_tickers.sort()
        current_comma = ",".join(current_scan_tickers)

        # 2. Active Database Universe
        database_tickers = get_active_ticker_list()
        database_comma = ",".join(database_tickers)

        # 3. Final Combined Deduped Sorted Options Trading Universe
        final_universe = sorted(list(set(current_scan_tickers) | set(database_tickers)))
        final_comma = ",".join(final_universe)

        st.markdown("##### 📍 Current Scan Qualified Tickers")
        st.code(current_comma, language="text")

        st.markdown("##### 🪐 Active Database Universe Tickers")
        st.code(database_comma, language="text")

        st.markdown("##### 🚀 Final Combined Deduped Option Trading Universe")
        st.code(final_comma, language="text")

        st.divider()

        # Download Buttons
        st.markdown("##### Download Datasets")
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        with dl_col1:
            st.download_button(
                "Download Current Scan TXT",
                data=current_comma,
                file_name="current_scan_tickers.txt",
                mime="text/plain",
                use_container_width=True
            )
        with dl_col2:
            st.download_button(
                "Download Active Universe TXT",
                data=database_comma,
                file_name="active_database_tickers.txt",
                mime="text/plain",
                use_container_width=True
            )
        with dl_col3:
            st.download_button(
                "Download Final Deduped Universe TXT",
                data=final_comma,
                file_name="final_deduped_universe.txt",
                mime="text/plain",
                use_container_width=True
            )

        csv_col1, csv_col2 = st.columns(2)
        with csv_col1:
            st.download_button(
                "Download Ticker Summary CSV",
                data=summary_df.to_csv(index=False),
                file_name="options_ticker_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
        with csv_col2:
            # Get filtered_df from session
            filtered_df = st.session_state.get("os_results", pd.DataFrame())
            st.download_button(
                "Download Contracts Details CSV",
                data=filtered_df.to_csv(index=False),
                file_name="options_contracts_results.csv",
                mime="text/csv",
                use_container_width=True
            )
