"""Ticker Pullback Strategy search module.

Scans momentum stock universes for pullback setups near key support zones (EMA9, EMA21, AVWAP),
calculates risk efficiency and setup triggers, and persists the results in the option search database.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from modules.base_module import FazDaneModule
from modules.tier1.option_search_db import (
    DB_PATH,
    init_option_search_db,
    upsert_pullback_ticker,
    get_pullback_candidates,
    refresh_final_universe_scores,
    get_active_ticker_list,
    get_comma_delimited_tickers,
)

logger = logging.getLogger("TickerPullbackStrategy")


def fetch_historical_ohlcv(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Fetch history using yfinance download and return a dict of DataFrame per ticker."""
    if not tickers:
        return {}
    try:
        df = yf.download(tickers, period=period, group_by="ticker", progress=False)
        result = {}
        if len(tickers) == 1:
            ticker = tickers[0]
            if not df.empty:
                result[ticker] = df
        else:
            for ticker in tickers:
                if ticker in df.columns.levels[0]:
                    sub_df = df[ticker].dropna(how="all")
                    if not sub_df.empty:
                        result[ticker] = sub_df
        return result
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        # fallback to individual downloads
        result = {}
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                h = t.history(period=period)
                if not h.empty:
                    result[ticker] = h
            except Exception as ex:
                logger.error(f"Fallback fetch failed for {ticker}: {ex}")
        return result


def analyze_pullback_setup(ticker: str, df_history: pd.DataFrame, min_momentum: float) -> dict | None:
    """Analyze if a ticker meets momentum and pullback criteria."""
    if len(df_history) < 50:
        return None

    df = df_history.copy()
    # Normalize columns to lowercase
    df.columns = [c.lower() for c in df.columns]

    close = float(df["close"].iloc[-1])
    high = float(df["high"].iloc[-1])
    low = float(df["low"].iloc[-1])
    volume = float(df["volume"].iloc[-1])

    if close < 10.0:  # Min price filter
        return None

    # 20d Average Volume
    avg_vol_20 = df["volume"].rolling(20).mean().iloc[-1]
    if pd.isna(avg_vol_20) or avg_vol_20 < 500000:
        return None

    # 1. Momentum Screen
    # Returns over 1M (21 days), 3M (63 days), 6M (126 days)
    ret_1m = (df["close"].iloc[-1] / df["close"].iloc[-21]) - 1.0 if len(df) >= 21 else 0.0
    ret_3m = (df["close"].iloc[-1] / df["close"].iloc[-63]) - 1.0 if len(df) >= 63 else 0.0
    ret_6m = (df["close"].iloc[-1] / df["close"].iloc[-126]) - 1.0 if len(df) >= 126 else 0.0

    # Composite Momentum: 30% 1M, 40% 3M, 30% 6M
    composite_mom = (0.30 * ret_1m + 0.40 * ret_3m + 0.30 * ret_6m) * 100.0

    if composite_mom < min_momentum:
        return None

    # Calculate EMAs
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    ema9_val = float(df["ema9"].iloc[-1])
    ema21_val = float(df["ema21"].iloc[-1])
    ema50_val = float(df["ema50"].iloc[-1])
    ema200_val = float(df["ema200"].iloc[-1])

    # Slope of EMA 21 (over 5 days)
    ema21_slope = (df["ema21"].iloc[-1] - df["ema21"].iloc[-5]) / 5.0 if len(df) >= 5 else 0.0

    # Trend filter: close > EMA50, EMA21 > EMA50, EMA50 > EMA200
    is_trending = (close > ema50_val) and (ema21_val > ema50_val) and (ema50_val > ema200_val) and (ema21_slope > 0)
    if not is_trending:
        return None

    # Calculate AVWAP anchored to the 20-day swing low
    swing_low_window = 20
    recent_lows = df["low"].iloc[-swing_low_window:]
    swing_low_idx = recent_lows.idxmin()

    df_since_low = df.loc[swing_low_idx:]
    if not df_since_low.empty:
        pv_sum = (df_since_low["close"] * df_since_low["volume"]).sum()
        vol_sum = df_since_low["volume"].sum()
        avwap_val = pv_sum / vol_sum if vol_sum > 0 else close
    else:
        avwap_val = close

    # 2. Pullback setup detection
    dist_ema9 = (close - ema9_val) / ema9_val
    dist_ema21 = (close - ema21_val) / ema21_val
    dist_avwap = (close - avwap_val) / avwap_val

    # Pullback setup: close within 3% of EMA9, EMA21, or AVWAP from above
    near_ema9 = 0.0 <= dist_ema9 <= 0.03
    near_ema21 = 0.0 <= dist_ema21 <= 0.03
    near_avwap = 0.0 <= dist_avwap <= 0.03

    if not (near_ema9 or near_ema21 or near_avwap):
        return None

    # ATR compression
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(np.abs(df["high"] - df["prev_close"]), np.abs(df["low"] - df["prev_close"])),
    )
    df["atr10"] = df["tr"].rolling(10).mean()
    df["atr50"] = df["tr"].rolling(50).mean()

    atr10_val = df["atr10"].iloc[-1]
    atr50_val = df["atr50"].iloc[-1]
    atr_ratio = atr10_val / atr50_val if (atr50_val and not pd.isna(atr50_val)) else 1.0
    compression_score = max(0.0, min(100.0, (1.0 - atr_ratio) * 200.0 + 50.0))

    # Support confluence calculation
    support_levels = [("EMA9", ema9_val), ("EMA21", ema21_val), ("EMA50", ema50_val), ("AVWAP", avwap_val)]
    confluent_supports = []
    for s_name, s_val in support_levels:
        if 0.0 <= (close - s_val) / s_val <= 0.03:
            confluent_supports.append((s_name, s_val))

    support_confluence_count = len(confluent_supports)
    support_confluence_score = min(100.0, support_confluence_count * 25.0)

    # Nearest support
    if confluent_supports:
        nearest_support = min(confluent_supports, key=lambda x: close - x[1])
        nearest_support_type = nearest_support[0]
        support_zone_low = nearest_support[1] * 0.99
        support_zone_high = nearest_support[1] * 1.01
    else:
        nearest_support_type = "EMA21"
        support_zone_low = ema21_val * 0.99
        support_zone_high = ema21_val * 1.01

    # Trigger filter: intrabar reclaim
    support_val = ema21_val
    if nearest_support_type == "EMA9":
        support_val = ema9_val
    elif nearest_support_type == "AVWAP":
        support_val = avwap_val

    is_reclaim = (low < support_val) and (close > support_val)
    setup_status = "TRIGGERED" if is_reclaim else "WATCH"
    ema_reclaim_score = 100.0 if is_reclaim else 50.0

    # Risk pct
    stop_loss = min(low, support_zone_low)
    risk_pct = (close - stop_loss) / close if close > 0 else 0.0
    risk_efficiency_score = max(40.0, min(100.0, 100.0 - (risk_pct * 100.0 - 1.0) * 15.0))

    # Base score
    base_score = 70.0 + (5.0 if close > ema9_val else 0.0) + (5.0 if ema9_val > ema21_val else 0.0)

    # Final composite pullback_score
    pullback_score = (
        0.30 * min(100.0, composite_mom)
        + 0.20 * support_confluence_score
        + 0.20 * ema_reclaim_score
        + 0.15 * compression_score
        + 0.15 * risk_efficiency_score
    )

    return {
        "last_price": close,
        "pullback_score": pullback_score,
        "momentum_score": min(100.0, composite_mom),
        "base_score": base_score,
        "support_confluence_score": support_confluence_score,
        "ema_reclaim_score": ema_reclaim_score,
        "compression_score": compression_score,
        "risk_efficiency_score": risk_efficiency_score,
        "risk_pct": risk_pct,
        "support_zone_low": support_zone_low,
        "support_zone_high": support_zone_high,
        "nearest_support_type": nearest_support_type,
        "avwap_distance_pct": dist_avwap * 100.0,
        "ema_9_distance_pct": dist_ema9 * 100.0,
        "ema_21_distance_pct": dist_ema21 * 100.0,
        "setup_status": setup_status,
    }


class TickerPullbackStrategyModule(FazDaneModule):
    MODULE_NAME = "Ticker Pullback Strategy"
    MODULE_ICON = "📈"
    MODULE_DESCRIPTION = "Scans liquid assets for high-quality momentum stock pullback opportunities at key moving average and AVWAP supports."
    TIER = 1
    CACHE_TTL = 300
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance API", "SQLite DB"]

    def render_sidebar(self):
        st.markdown("**Starting Universe**")

        universe_type = st.radio(
            "Select Base Tickers",
            options=["Liquid Option Listed (120+)", "SP500 Preset (Top 100)", "Custom Ticker List"],
            key="tps_universe_type",
        )

        # Standard liquid weekly lists
        liquid_weekly_tickers = [
            "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "SMH", "EEM", "FXI", "GDX", "GDXJ",
            "XLE", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XOP", "KRE",
            "ARKK", "USO", "UNG",
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

        sp500_preset = [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK.B", "LLY", "AVGO",
            "JPM", "TSLA", "UNH", "XOM", "V", "MA", "PG", "HD", "COST", "JNJ",
            "MRK", "ABBV", "BAC", "CRM", "WMT", "KO", "AMD", "PEP", "ADBE", "CVX",
            "ORCL", "QCOM", "TMO", "WFC", "ACN", "COST", "MCD", "DIS", "GE", "INTU",
            "PM", "CAT", "IBM", "TXN", "AMGN", "MS", "AXP", "HON", "SPGI", "UNP",
            "COP", "NEE", "GS", "PGR", "PLTR", "RTX", "ISRG", "LOW", "BKNG", "ETN",
            "TJX", "REGN", "LMT", "C", "MDLZ", "VRTX", "BLK", "CI", "ADI", "ANET",
            "SYK", "TJX", "BSX", "DE", "ELV", "ADP", "MDT", "CVS", "PANW", "MMC",
            "LRCX", "WM", "HCA", "MU", "AMT", "FI", "CB", "GILD", "SBUX", "GEV"
        ]

        if universe_type == "Liquid Option Listed (120+)":
            tickers = liquid_weekly_tickers
        elif universe_type == "SP500 Preset (Top 100)":
            tickers = sp500_preset
        else:
            custom_input = st.text_area("Custom Comma Tickers:", placeholder="NVDA,PLTR,TSLA", key="tps_custom_input")
            if custom_input:
                tickers = [s.strip().upper() for s in custom_input.split(",") if s.strip()]
            else:
                tickers = []

        # Add active tickers from DB
        include_db_tickers = st.checkbox("Include Existing Database Active Tickers", value=True, key="tps_include_db")
        if include_db_tickers:
            db_actives = get_active_ticker_list()
            tickers = list(set(tickers) | set(db_actives))

        st.caption(f"Loaded {len(tickers)} unique tickers for analysis.")

        st.markdown("**Pullback Strategy Parameters**")
        min_momentum = st.slider("Min Momentum Return (Composite %)", min_value=0, max_value=100, value=30, step=5, key="tps_min_mom")
        stale_days = st.slider("Deactivate Stale Window (Days)", min_value=5, max_value=90, value=30, step=5, key="tps_stale_days")
        save_db = st.checkbox("Save Scan to Master Universe", value=True, key="tps_save_db")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        scan_clicked = st.button("Run Pullback Scan", use_container_width=True, type="primary", key="tps_scan")

        if scan_clicked:
            if not tickers:
                st.error("Please load or type at least one ticker.")
            else:
                st.session_state["tps_last_params"] = {
                    "tickers": tuple(tickers),
                    "min_momentum": min_momentum,
                    "stale_days": stale_days,
                    "save_db": save_db,
                }
                # Clear cached results to force rerun
                st.session_state.pop("tps_results", None)

    def render_main(self):
        self.render_section_header("Ticker Pullback Strategy", "Identify high-momentum stocks executing pullback setups at key supports")

        init_option_search_db()

        if "tps_last_params" not in st.session_state:
            self._render_welcome_screen()
            return

        params = st.session_state["tps_last_params"]
        tickers = list(params["tickers"])
        min_mom = float(params["min_momentum"])

        # Fetch and analyze
        if "tps_results" not in st.session_state:
            with st.spinner("Downloading historical data & executing technical screening..."):
                ohlcv_data = fetch_historical_ohlcv(tickers, period="1y")
                
                results_list = []
                for t in tickers:
                    if t in ohlcv_data:
                        result = analyze_pullback_setup(t, ohlcv_data[t], min_mom)
                        if result:
                            result["ticker"] = t
                            results_list.append(result)

                results_df = pd.DataFrame(results_list)
                
                # Write to DB
                if params["save_db"] and not results_df.empty:
                    try:
                        trade_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for _, row in results_df.iterrows():
                            upsert_pullback_ticker(row["ticker"], trade_date, row)
                        refresh_final_universe_scores()
                        st.session_state["tps_save_status"] = f"✅ Saved {len(results_df)} opportunities to SQLite option_ticker_universe."
                    except Exception as e:
                        logger.error(f"Error saving pullback results: {e}")
                        st.session_state["tps_save_status"] = f"❌ Database save failed: {e}"
                else:
                    st.session_state["tps_save_status"] = "ℹ️ Scan complete. Database save bypassed."

                st.session_state["tps_results"] = results_df
                st.rerun()

        results_df = st.session_state["tps_results"]
        save_status = st.session_state.get("tps_save_status", "")

        if save_status:
            st.success(save_status)

        if results_df.empty:
            st.warning("⚠️ No tickers met the momentum and pullback confluence criteria in the current scan.")
            st.info("""
            **Adjust parameters to find opportunities:**
            - Lower the **Min Momentum Return** threshold in the sidebar.
            - Add more tickers or check if yfinance download is blocked.
            - Ensure base stocks are trending upwards (above EMA50 and EMA200).
            """)
            return

        # ----------------- TABS SETUP -----------------
        tabs = st.tabs([
            "📊 Scan Results",
            "🎯 Pullback Opportunities",
            "🏆 Triggered Setups",
            "👁️ Watch List",
            "🪐 Combined Universe",
            "📥 Ticker Export"
        ])

        with tabs[0]:
            st.markdown("### Qualified Pullback Candidates")
            st.caption(f"Found {len(results_df)} setups in momentum trend.")
            
            # Simple results table
            display_cols = [
                "ticker", "last_price", "pullback_score", "momentum_score",
                "setup_status", "nearest_support_type", "risk_pct"
            ]
            st.dataframe(
                results_df[display_cols].sort_values("pullback_score", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ticker": "Ticker",
                    "last_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "pullback_score": st.column_config.NumberColumn("Pullback Score", format="%.1f"),
                    "momentum_score": st.column_config.NumberColumn("Momentum %", format="%.1f%%"),
                    "setup_status": "Status",
                    "nearest_support_type": "Nearest Support",
                    "risk_pct": st.column_config.NumberColumn("Risk to Stop", format="%.2%"),
                }
            )

        with tabs[1]:
            st.markdown("### Ranked Pullback Opportunities")
            
            # Advanced interactive matrix
            fig_matrix = px.scatter(
                results_df,
                x="risk_pct",
                y="pullback_score",
                color="setup_status",
                size="momentum_score",
                hover_data=["ticker", "last_price", "nearest_support_type"],
                labels={"risk_pct": "Risk % to Stop", "pullback_score": "Pullback Setup Score"},
                title="Risk vs Setup Score Matrix",
                height=400
            )
            st.plotly_chart(fig_matrix, use_container_width=True)

            # Details
            st.dataframe(
                results_df.sort_values("pullback_score", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ticker": "Ticker",
                    "last_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "pullback_score": st.column_config.ProgressColumn("Composite Score", format="%.1f", min_value=0, max_value=100),
                    "momentum_score": st.column_config.NumberColumn("Momentum %", format="%.1f%%"),
                    "support_confluence_score": st.column_config.NumberColumn("Confluence", format="%.0f"),
                    "ema_reclaim_score": st.column_config.NumberColumn("Reclaim", format="%.0f"),
                    "compression_score": st.column_config.NumberColumn("Compression", format="%.0f"),
                    "risk_efficiency_score": st.column_config.NumberColumn("Risk Efficiency", format="%.0f"),
                    "risk_pct": st.column_config.NumberColumn("Risk %", format="%.2%"),
                    "nearest_support_type": "Support Type",
                    "ema_9_distance_pct": st.column_config.NumberColumn("Dist EMA9 %", format="%.2f%%"),
                    "ema_21_distance_pct": st.column_config.NumberColumn("Dist EMA21 %", format="%.2f%%"),
                    "avwap_distance_pct": st.column_config.NumberColumn("Dist AVWAP %", format="%.2f%%"),
                    "setup_status": "Status"
                }
            )

        with tabs[2]:
            st.markdown("### Triggered Reclaim Setups")
            st.caption("These setups broke below key support during the session but closed back above, signaling an active buyer reclaim trigger.")
            triggered_df = results_df[results_df["setup_status"] == "TRIGGERED"]
            if triggered_df.empty:
                st.info("No active reclaim triggers detected in this scan. Watch list candidates are still digesting support.")
            else:
                st.dataframe(
                    triggered_df.sort_values("pullback_score", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

        with tabs[3]:
            st.markdown("### Watch List Pullbacks")
            st.caption("Price is resting tightly near key support zones. Ready to alert on next reclaim or bounce trigger.")
            watch_df = results_df[results_df["setup_status"] == "WATCH"]
            if watch_df.empty:
                st.info("No watch list pullbacks found.")
            else:
                st.dataframe(
                    watch_df.sort_values("pullback_score", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

        with tabs[4]:
            st.markdown("### Combined Persistent Universe")
            st.caption("This is the shared, deduped Master Universe table showing both options liquidity scores and pullback scores.")
            
            # Query direct from SQLite
            conn = sqlite3.connect(DB_PATH)
            db_df = pd.read_sql_query("SELECT * FROM option_ticker_universe WHERE active_flag = 1 ORDER BY final_universe_score DESC;", conn)
            conn.close()
            
            st.dataframe(
                db_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ticker": "Ticker",
                    "last_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "final_universe_score": st.column_config.ProgressColumn("Final Score", format="%.1f", min_value=0, max_value=100),
                    "last_option_score": st.column_config.NumberColumn("Option Score", format="%.1f"),
                    "pullback_score": st.column_config.NumberColumn("Pullback Score", format="%.1f"),
                    "last_liquidity_score": st.column_config.NumberColumn("Liq Score", format="%.0f"),
                    "risk_pct": st.column_config.NumberColumn("Risk %", format="%.2%"),
                    "setup_status": "Setup Status",
                    "source_strategy": "Last Engine Source",
                    "last_seen_date": "Last Updated"
                }
            )

        with tabs[5]:
            st.markdown("### Master Universe Comma Export")
            st.caption("Use this output to quickly copy-paste list of all active tickers into other trading software.")
            
            conn = sqlite3.connect(DB_PATH)
            active_tickers = [row[0] for row in conn.execute("SELECT ticker FROM option_ticker_universe WHERE active_flag = 1 ORDER BY ticker ASC;").fetchall()]
            conn.close()
            
            st.text_area("Copy List:", value=",".join(active_tickers), height=150, key="tps_export_copy")

    def _render_welcome_screen(self):
        st.markdown(
            """
            <div style="background:rgba(21,40,71,0.58); border-left:5px solid #3ab54a; padding:18px 25px; border-radius:8px; margin-bottom:25px;">
                <h4 style="color:#3ab54a; margin-top:0;">Ticker Pullback Strategy Engine</h4>
                <p style="color:#cbd5e1; font-size:13.5px; margin:0; line-height:1.6;">
                    Select or customize your scan universe, configure filters and minimum momentum, then execute the pullback scanner.
                    Results will enrich the master database and rank assets on a risk/reward scale.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Load stats
        conn = sqlite3.connect(DB_PATH)
        c_cursor = conn.cursor()
        c_cursor.execute("SELECT COUNT(*) FROM option_ticker_universe WHERE active_flag = 1;")
        active_count = c_cursor.fetchone()[0]
        c_cursor.execute("SELECT COUNT(*) FROM option_ticker_universe WHERE pullback_score IS NOT NULL;")
        pullback_count = c_cursor.fetchone()[0]
        conn.close()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Active Master Universe Tickers", active_count)
        with col2:
            st.metric("Tickers with Pullback Scoring Data", pullback_count)
