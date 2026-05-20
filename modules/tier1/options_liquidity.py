"""
FazDane Analytics — Tier 1
Options Liquidity Discovery Engine
Source: 05-FazDane Options Liquidity Discovery Engine.ipynb
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import logging
from modules.base_module import FazDaneModule
from utils.tastytrade_provider import TastytradeProviderError, fetch_nested_option_chain, load_config
from utils.universe_manager import render_universe_manager

logger = logging.getLogger("OptionsLiquidity")

# ══════════════════════════════════════════════════════════════════════
# DEFAULT WATCHLIST
# ══════════════════════════════════════════════════════════════════════

DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "IWM", "GLD", "TLT",
    "AAPL", "MSFT", "NVDA", "AMZN", "TSLA",
    "GOOGL", "META", "JPM", "XLK", "XLF",
]

# ══════════════════════════════════════════════════════════════════════
# DATA ENGINE
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_options_data(symbols: tuple, min_volume: int, min_oi: int,
                       option_types: tuple, exp_pref: str) -> pd.DataFrame:
    """
    Scan options chains for each symbol and return filtered results.
    Tastytrade is preferred for option-chain metadata; yfinance remains the
    fallback and quote/liquidity enrichment source when needed.
    Cached for 5 minutes.
    """
    tasty_df = _fetch_tastytrade_options_data(symbols, min_volume, min_oi, option_types, exp_pref)
    if not tasty_df.empty:
        return tasty_df

    logger.info("Tastytrade option scan unavailable or empty; falling back to yfinance.")
    fallback_df = _fetch_yfinance_options_data(symbols, min_volume, min_oi, option_types, exp_pref)
    if fallback_df.empty:
        fallback_df.attrs["active_data_source"] = "No matching contracts; tried Tastytrade API, then yfinance fallback"
    return fallback_df


def _expiration_bounds(exp_pref: str) -> tuple[int, int]:
    exp_label = str(exp_pref).lower()
    if "weekly" in exp_label:
        return 0, 8
    if "monthly" in exp_label:
        return 9, 45
    return 0, 365


def _fetch_tastytrade_options_data(symbols: tuple, min_volume: int, min_oi: int,
                                   option_types: tuple, exp_pref: str) -> pd.DataFrame:
    config = load_config()
    if not config.is_configured:
        return pd.DataFrame()

    min_dte, max_dte = _expiration_bounds(exp_pref)
    results = []

    for symbol in symbols:
        try:
            tasty_chain = fetch_nested_option_chain(symbol, config=config)
            if tasty_chain.empty:
                continue

            tasty_chain = tasty_chain[
                (tasty_chain["dte"] >= min_dte) &
                (tasty_chain["dte"] <= max_dte) &
                (tasty_chain["option_type"].isin(option_types))
            ].copy()
            if tasty_chain.empty:
                continue

            enriched = _enrich_tasty_chain_with_yfinance_quotes(symbol, tasty_chain)
            if enriched.empty:
                continue

            enriched["volume"] = pd.to_numeric(enriched["volume"], errors="coerce").fillna(0)
            oi_source = "open_interest" if "open_interest" in enriched.columns else "openInterest"
            enriched["open_interest"] = pd.to_numeric(enriched[oi_source], errors="coerce").fillna(0)
            enriched = enriched[
                (enriched["volume"] >= min_volume) &
                (enriched["open_interest"] >= min_oi)
            ]
            if not enriched.empty:
                results.append(enriched)
        except TastytradeProviderError as e:
            logger.info(f"Tastytrade not available for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Tastytrade option fetch failed for {symbol}: {e}")
            continue

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return _finalize_options_frame(combined)


def _enrich_tasty_chain_with_yfinance_quotes(symbol: str, tasty_chain: pd.DataFrame) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    spot = _fetch_yfinance_spot(ticker)
    if not spot or spot == 0:
        return pd.DataFrame()

    quote_frames = []
    matched_expirations = 0
    for exp_str in sorted(tasty_chain["expiration"].dropna().unique()):
        try:
            chain = ticker.option_chain(exp_str)
        except Exception as e:
            logger.warning(f"yfinance quote enrichment failed for {symbol} {exp_str}: {e}")
            continue

        exp_has_rows = False
        for opt_type, df_opts in [("Call", chain.calls), ("Put", chain.puts)]:
            if df_opts is None or df_opts.empty:
                continue
            df_opts = df_opts.copy()
            df_opts["symbol"] = symbol
            df_opts["option_type"] = opt_type
            df_opts["expiration"] = exp_str
            quote_frames.append(df_opts)
            exp_has_rows = True

        if exp_has_rows:
            matched_expirations += 1
        if matched_expirations >= 8:
            break

    if not quote_frames:
        return pd.DataFrame()

    quotes = pd.concat(quote_frames, ignore_index=True)
    quotes["strike"] = pd.to_numeric(quotes["strike"], errors="coerce")
    tasty_chain = tasty_chain.copy()
    tasty_chain["strike"] = pd.to_numeric(tasty_chain["strike"], errors="coerce")

    merged = tasty_chain.merge(
        quotes,
        on=["symbol", "option_type", "expiration", "strike"],
        how="inner",
        suffixes=("", "_yf"),
    )
    if merged.empty:
        return pd.DataFrame()

    merged["spot"] = round(spot, 2)
    merged["moneyness"] = merged["strike"] / spot
    if "impliedVolatility" in merged.columns:
        merged["iv_pct"] = (pd.to_numeric(merged["impliedVolatility"], errors="coerce") * 100).round(1)
    if {"ask", "bid"}.issubset(merged.columns):
        merged["spread"] = (merged["ask"] - merged["bid"]).round(3)
        merged["spread_pct"] = (
            merged["spread"] / merged["ask"].replace(0, np.nan) * 100
        ).round(1)
    merged["data_source"] = "Tastytrade API chain + yfinance quotes"
    return merged


def _fetch_yfinance_spot(ticker) -> float | None:
    spot = None
    try:
        info = ticker.fast_info
        spot = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
    except Exception:
        spot = None
    if not spot or spot == 0:
        try:
            hist = ticker.history(period="5d", interval="1d")
            if not hist.empty and "Close" in hist.columns:
                spot = float(hist["Close"].dropna().iloc[-1])
        except Exception:
            spot = None
    return spot


def _fetch_yfinance_options_data(symbols: tuple, min_volume: int, min_oi: int,
                                 option_types: tuple, exp_pref: str) -> pd.DataFrame:
    results = []
    min_dte, max_dte = _expiration_bounds(exp_pref)

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            spot = _fetch_yfinance_spot(ticker)
            if not spot or spot == 0:
                continue

            expirations = ticker.options
            if not expirations:
                continue

            today = datetime.today().date()
            matched_expirations = 0

            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if dte < min_dte or dte > max_dte:
                    continue

                chain = ticker.option_chain(exp_str)
                exp_has_results = False

                for opt_type, df_opts in [("Call", chain.calls), ("Put", chain.puts)]:
                    if opt_type not in option_types:
                        continue
                    if df_opts is None or df_opts.empty:
                        continue

                    df_opts = df_opts.copy()
                    df_opts["symbol"] = symbol
                    df_opts["option_type"] = opt_type
                    df_opts["expiration"] = exp_str
                    df_opts["dte"] = dte
                    df_opts["spot"] = round(spot, 2)

                    # Moneyness
                    df_opts["moneyness"] = df_opts["strike"] / spot

                    # IV as percentage
                    df_opts["iv_pct"] = (df_opts["impliedVolatility"] * 100).round(1)

                    # Bid-ask spread
                    df_opts["spread"] = (df_opts["ask"] - df_opts["bid"]).round(3)
                    df_opts["spread_pct"] = (
                        df_opts["spread"] / df_opts["ask"].replace(0, np.nan) * 100
                    ).round(1)

                    # Filter
                    vol_col = "volume"
                    oi_col = "openInterest"
                    df_opts[vol_col] = pd.to_numeric(df_opts[vol_col], errors="coerce").fillna(0)
                    df_opts[oi_col] = pd.to_numeric(df_opts[oi_col], errors="coerce").fillna(0)

                    df_filtered = df_opts[
                        (df_opts[vol_col] >= min_volume) &
                        (df_opts[oi_col] >= min_oi)
                    ]

                    if not df_filtered.empty:
                        df_filtered = df_filtered.copy()
                        df_filtered["data_source"] = "yfinance"
                        results.append(df_filtered)
                        exp_has_results = True

                if exp_has_results:
                    matched_expirations += 1
                if matched_expirations >= 8:
                    break

        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
            continue

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return _finalize_options_frame(combined)


def _finalize_options_frame(combined: pd.DataFrame) -> pd.DataFrame:
    combined = combined.copy()
    if "open_interest" in combined.columns and "openInterest" in combined.columns:
        combined.drop(columns=["openInterest"], inplace=True)
    if "last_price" in combined.columns and "lastPrice" in combined.columns:
        combined.drop(columns=["lastPrice"], inplace=True)

    # Select & rename columns
    keep_cols = [
        "symbol", "option_type", "expiration", "dte", "spot",
        "strike", "moneyness", "iv_pct", "volume", "openInterest",
        "open_interest", "bid", "ask", "spread", "spread_pct", "lastPrice",
        "last_price", "contract", "streamer_symbol", "data_source",
    ]
    available = [c for c in keep_cols if c in combined.columns]
    combined = combined[available].copy()

    combined.rename(columns={
        "openInterest": "open_interest",
        "lastPrice": "last_price",
        "iv_pct": "iv_%",
    }, inplace=True)

    combined = combined.sort_values("volume", ascending=False).reset_index(drop=True)
    if "data_source" in combined.columns:
        sources = ", ".join(sorted(combined["data_source"].dropna().unique()))
        combined.attrs["active_data_source"] = sources
    return combined


@st.cache_data(ttl=300, show_spinner=False)
def fetch_iv_rank(symbols: tuple) -> dict:
    """
    Estimate IV Rank for each symbol using 1-year historical close prices.
    IV Rank = (current HV30 - min HV252) / (max HV252 - min HV252) * 100
    """
    ranks = {}
    for sym in symbols:
        try:
            hist = yf.download(sym, period="1y", interval="1d",
                               progress=False, auto_adjust=True)
            if hist.empty or len(hist) < 30:
                ranks[sym] = None
                continue

            # 30-day rolling historical volatility (annualised)
            log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
            hv = log_ret.rolling(30).std() * np.sqrt(252) * 100  # in %

            current_hv = hv.iloc[-1]
            if hasattr(current_hv, '__len__'):
                current_hv = float(current_hv.iloc[0])
            else:
                current_hv = float(current_hv)

            hv_min = float(hv.min())
            hv_max = float(hv.max())
            rng = hv_max - hv_min

            iv_rank = round((current_hv - hv_min) / rng * 100, 1) if rng > 0 else 50.0
            ranks[sym] = iv_rank
        except Exception as e:
            logger.warning(f"IV Rank failed for {sym}: {e}")
            ranks[sym] = None
    return ranks


# ══════════════════════════════════════════════════════════════════════
# MODULE CLASS
# ══════════════════════════════════════════════════════════════════════

class OptionsLiquidityModule(FazDaneModule):
    MODULE_NAME = "Options Liquidity Discovery"
    MODULE_ICON = "💧"
    MODULE_DESCRIPTION = "Scan for high-liquidity options with elevated IV"
    TIER = 1
    SOURCE_NOTEBOOK = "05-FazDane Options Liquidity Discovery Engine.ipynb"
    CACHE_TTL = 300
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["Tastytrade API", "yfinance fallback"]

    # ── Sidebar ────────────────────────────────────────────────────────

    def render_sidebar(self):
        st.markdown("**Watchlist**")
        universe_name, symbols, _ = render_universe_manager(
            key_prefix="ol",
            show_benchmark=False,
            label="Ticker Universe:",
        )
        st.caption(f"{len(symbols)} symbols selected from {universe_name}.")

        st.markdown("**Filters**")
        min_volume = st.slider("Min Volume", 0, 5000, 500, 100, key="ol_min_vol")
        min_oi = st.slider("Min Open Interest", 0, 10000, 1000, 500, key="ol_min_oi")

        option_types = st.multiselect(
            "Option Type",
            ["Call", "Put"],
            default=["Call", "Put"],
            key="ol_types",
        )

        exp_pref = st.selectbox(
            "Expiration Window",
            ["Weekly (≤8 days)", "Monthly (9–45 days)", "Any"],
            index=1,
            key="ol_exp",
        )

        st.markdown("**Data Source**")
        config = load_config()
        st.caption(f"Tastytrade: {'configured' if config.is_configured else 'not configured'}")
        st.caption("Priority: Tastytrade API -> yfinance fallback")
        active_source = st.session_state.get("ol_active_data_source", "Not scanned yet")
        st.info(f"Active Source: {active_source}")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        scan_clicked = st.button("🔍 Scan Options", use_container_width=True,
                                 type="primary", key="ol_scan")
        export_clicked = st.button("📥 Export CSV", use_container_width=True,
                                   key="ol_export")

        if scan_clicked:
            if not symbols:
                st.error("Enter at least one symbol.")
            elif not option_types:
                st.error("Select at least one option type.")
            else:
                st.session_state["ol_last_symbols"] = symbols
                st.session_state["ol_last_params"] = {
                    "symbols": tuple(symbols),
                    "min_volume": min_volume,
                    "min_oi": min_oi,
                    "option_types": tuple(option_types),
                    "exp_pref": exp_pref,
                }
                st.session_state.pop("ol_results", None)  # force refresh

        if export_clicked and "ol_results" in st.session_state:
            df = st.session_state["ol_results"]
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Now",
                    data=csv,
                    file_name=f"options_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    key="ol_dl",
                )

        # Summary stats in sidebar when results exist
        if "ol_results" in st.session_state:
            df = st.session_state["ol_results"]
            if not df.empty:
                st.divider()
                st.markdown("**Scan Summary**")
                st.metric("Results", f"{len(df):,}")
                if "volume" in df.columns:
                    st.metric("Avg Volume", f"{int(df['volume'].mean()):,}")
                if "iv_%" in df.columns:
                    st.metric("Avg IV", f"{df['iv_%'].mean():.1f}%")

    # ── Main ───────────────────────────────────────────────────────────

    def render_main(self):
        self.render_section_header(
            "💧 Options Liquidity Discovery",
            "Real-time scan for high-liquidity options opportunities"
        )

        # No scan yet
        if "ol_last_params" not in st.session_state:
            self._render_welcome()
            return

        params = st.session_state["ol_last_params"]
        symbols = params["symbols"]

        # Run scan (cached)
        if "ol_results" not in st.session_state:
            with st.spinner(f"Scanning {len(symbols)} symbols…"):
                df = fetch_options_data(**params)
                st.session_state["ol_results"] = df
                if "data_source" in df.columns and not df.empty:
                    sources = ", ".join(sorted(df["data_source"].dropna().unique()))
                else:
                    sources = df.attrs.get(
                        "active_data_source",
                        "No matching contracts; provider returned no displayable rows",
                    )
                st.session_state["ol_active_data_source"] = sources

                # Fetch IV ranks separately
                with st.spinner("Calculating IV Ranks…"):
                    iv_ranks = fetch_iv_rank(symbols)
                    st.session_state["ol_iv_ranks"] = iv_ranks
                st.rerun()

        df = st.session_state["ol_results"]
        iv_ranks = st.session_state.get("ol_iv_ranks", {})

        if df.empty:
            st.warning(
                "⚠️ No options found matching your criteria. "
                "Try lowering Min Volume / Min Open Interest or widening the expiration window."
            )
            return

        # ── Top Metrics Row ──────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Results", f"{len(df):,}")
        m2.metric("Symbols Hit", df["symbol"].nunique())
        m3.metric("Avg Volume", f"{int(df['volume'].mean()):,}" if "volume" in df.columns else "—")
        m4.metric("Avg IV", f"{df['iv_%'].mean():.1f}%" if "iv_%" in df.columns else "—")
        m5.metric("Avg Spread", f"${df['spread'].mean():.2f}" if "spread" in df.columns else "—")

        if "data_source" in df.columns:
            sources = ", ".join(sorted(df["data_source"].dropna().unique()))
            st.caption(f"Data source: {sources}")

        st.divider()

        # ── IV Rank Banner (if available) ────────────────────────────
        valid_ranks = {k: v for k, v in iv_ranks.items() if v is not None}
        if valid_ranks:
            self._render_iv_rank_bar(valid_ranks)
            st.divider()

        # ── Tabs ─────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Volume Heatmap", "Ticker Drilldown", "Options Chain", "Analytics", "IV Landscape"]
        )

        with tab1:
            self._tab_heatmap(df)

        with tab2:
            self._tab_ticker_drilldown(df)

        with tab3:
            self._tab_chain(df)

        with tab4:
            self._tab_analytics(df)

        with tab5:
            self._tab_iv_landscape(df, valid_ranks)

    # ── IV Rank Banner ─────────────────────────────────────────────────

    def _render_iv_rank_bar(self, iv_ranks: dict):
        st.markdown(
            "<div style='color:#64748b;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;'>📡 IV Rank by Symbol (Historical Volatility Percentile)</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(min(len(iv_ranks), 8))
        for i, (sym, rank) in enumerate(sorted(iv_ranks.items(), key=lambda x: x[1], reverse=True)[:8]):
            color = "#ef4444" if rank >= 75 else "#f59e0b" if rank >= 50 else "#3ab54a"
            with cols[i]:
                st.markdown(
                    f"""
                    <div style="
                        background:rgba(21,40,71,0.8);
                        border:1px solid #1e3a5f;
                        border-top:3px solid {color};
                        border-radius:8px;
                        padding:10px;
                        text-align:center;
                    ">
                        <div style="color:#e2e8f0;font-weight:700;font-size:14px;">{sym}</div>
                        <div style="color:{color};font-size:20px;font-weight:700;">{rank:.0f}</div>
                        <div style="color:#475569;font-size:10px;">IV Rank</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Tab 1: Heatmap ─────────────────────────────────────────────────

    def _tab_heatmap(self, df: pd.DataFrame):
        st.markdown("### Volume Heatmap by Symbol & Option Type")
        if "volume" not in df.columns or "symbol" not in df.columns:
            st.info("Insufficient data for heatmap.")
            return

        pivot = (
            df.groupby(["symbol", "option_type"])["volume"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        pivot = pivot.sort_values(
            by=[c for c in ["Call", "Put"] if c in pivot.columns],
            ascending=False
        ).head(20)

        call_vals = pivot.get("Call", pd.Series([0] * len(pivot))).tolist()
        put_vals  = pivot.get("Put",  pd.Series([0] * len(pivot))).tolist()
        syms      = pivot["symbol"].tolist()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Calls",
            x=syms,
            y=call_vals,
            marker_color="#3ab54a",
            opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            name="Puts",
            x=syms,
            y=put_vals,
            marker_color="#ef4444",
            opacity=0.85,
        ))

        fig.update_layout(
            barmode="group",
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#0d1b2e",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0", size=11)),
            yaxis=dict(gridcolor="#1e3a5f", title="Total Volume", tickfont=dict(color="#e2e8f0")),
            legend=dict(
                bgcolor="rgba(21,40,71,0.85)",
                bordercolor="#1e3a5f",
                borderwidth=1,
                font=dict(color="#e2e8f0", size=13),
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Scatter: Volume vs IV
        if "iv_%" in df.columns:
            st.markdown("### Volume vs IV — Opportunity Scatter")
            fig2 = px.scatter(
                df.head(200),
                x="iv_%",
                y="volume",
                color="symbol",
                size="volume",
                size_max=28,
                hover_data=["symbol", "strike", "option_type", "dte", "bid", "ask"],
                labels={"iv_%": "Implied Volatility (%)", "volume": "Volume"},
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig2.update_layout(
                paper_bgcolor="#0d1b2e",
                plot_bgcolor="#152847",
                font=dict(color="#e2e8f0", family="Inter"),
                xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
                yaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
                legend=dict(
                    bgcolor="rgba(21,40,71,0.85)",
                    bordercolor="#1e3a5f",
                    borderwidth=1,
                    font=dict(color="#e2e8f0", size=12),
                    title=dict(font=dict(color="#94a3b8", size=11)),
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                height=360,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Chain Table ─────────────────────────────────────────────

    def _tab_ticker_drilldown(self, df: pd.DataFrame):
        st.markdown("### Individual Ticker Activity")
        required = {"symbol", "expiration", "strike", "volume", "option_type"}
        if not required.issubset(df.columns):
            st.info("Insufficient option-chain fields for ticker drilldown.")
            return

        symbol_totals = df.groupby("symbol")["volume"].sum().sort_values(ascending=False)
        selected_symbol = st.selectbox(
            "Select Ticker",
            symbol_totals.index.tolist(),
            index=0,
            key="ol_drill_symbol",
        )

        drill = df[df["symbol"] == selected_symbol].copy()
        if drill.empty:
            st.info("No contracts available for the selected ticker.")
            return

        drill["expiration"] = pd.to_datetime(drill["expiration"], errors="coerce").dt.strftime("%Y-%m-%d")
        drill["contract"] = (
            drill["expiration"].astype(str)
            + " "
            + drill["option_type"].astype(str).str[0]
            + " "
            + drill["strike"].map(lambda value: f"{value:g}")
        )

        total_volume = int(drill["volume"].sum())
        call_volume = int(drill.loc[drill["option_type"] == "Call", "volume"].sum())
        put_volume = int(drill.loc[drill["option_type"] == "Put", "volume"].sum())
        active_expirations = drill["expiration"].nunique()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Volume", f"{total_volume:,}")
        m2.metric("Call Volume", f"{call_volume:,}")
        m3.metric("Put Volume", f"{put_volume:,}")
        m4.metric("Expirations", f"{active_expirations:,}")

        st.markdown("#### Volume by Expiration and Strike")
        agg_map = {"volume": ("volume", "sum")}
        if "open_interest" in drill.columns:
            agg_map["open_interest"] = ("open_interest", "sum")
        activity = (
            drill.groupby(["expiration", "strike", "option_type"], as_index=False)
            .agg(**agg_map)
            .sort_values("volume", ascending=False)
        )
        hover_data = {
            "expiration": True,
            "strike": ":.2f",
            "volume": ":,",
            "option_type": True,
        }
        if "open_interest" in activity.columns:
            hover_data["open_interest"] = ":,"
        fig = px.scatter(
            activity,
            x="expiration",
            y="strike",
            size="volume",
            color="option_type",
            size_max=42,
            color_discrete_map={"Call": "#3ab54a", "Put": "#ef4444"},
            hover_data=hover_data,
            labels={"expiration": "Expiration Date", "strike": "Strike", "volume": "Volume"},
        )
        fig.update_layout(
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#152847",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
            yaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0"), title="Strike"),
            legend=dict(
                bgcolor="rgba(21,40,71,0.85)",
                bordercolor="#1e3a5f",
                borderwidth=1,
                font=dict(color="#e2e8f0", size=12),
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=440,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"ol_drill_scatter_{selected_symbol}")

        st.markdown("#### Expiration Summary")
        expiry_summary = (
            drill.groupby(["expiration", "option_type"], as_index=False)["volume"]
            .sum()
            .sort_values(["expiration", "option_type"])
        )
        fig2 = px.bar(
            expiry_summary,
            x="expiration",
            y="volume",
            color="option_type",
            barmode="stack",
            color_discrete_map={"Call": "#3ab54a", "Put": "#ef4444"},
            labels={"expiration": "Expiration Date", "volume": "Volume"},
        )
        fig2.update_layout(
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#152847",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
            yaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
            margin=dict(l=0, r=0, t=30, b=0),
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True, key=f"ol_drill_expiry_{selected_symbol}")

        st.markdown("#### Top Contracts Feeding Volume")
        top_contracts = drill.nlargest(30, "volume")
        display_cols = [
            c for c in [
                "contract", "option_type", "expiration", "dte", "strike", "volume",
                "open_interest", "iv_%", "bid", "ask", "spread", "last_price",
            ]
            if c in top_contracts.columns
        ]
        st.dataframe(top_contracts[display_cols], use_container_width=True, hide_index=True)

    def _tab_chain(self, df: pd.DataFrame):
        st.markdown("### Options Chain Results")

        fc1, fc2, fc3 = st.columns(3)
        sym_filter = fc1.multiselect(
            "Filter Symbol", sorted(df["symbol"].unique()),
            key="chain_sym"
        )
        type_filter = fc2.multiselect(
            "Option Type", ["Call", "Put"],
            default=["Call", "Put"],
            key="chain_type"
        )
        sort_col = fc3.selectbox(
            "Sort By",
            [c for c in ["volume", "open_interest", "iv_%", "spread"] if c in df.columns],
            key="chain_sort"
        )

        display = df.copy()
        if sym_filter:
            display = display[display["symbol"].isin(sym_filter)]
        if type_filter:
            display = display[display["option_type"].isin(type_filter)]
        display = display.sort_values(sort_col, ascending=False)

        st.markdown(f"*Showing {len(display):,} contracts*")

        # Column config
        col_cfg = {}
        if "volume" in display.columns:
            col_cfg["volume"] = st.column_config.NumberColumn("Volume", format="%d")
        if "open_interest" in display.columns:
            col_cfg["open_interest"] = st.column_config.NumberColumn("Open Int.", format="%d")
        if "iv_%" in display.columns:
            col_cfg["iv_%"] = st.column_config.NumberColumn("IV %", format="%.1f%%")
        if "bid" in display.columns:
            col_cfg["bid"] = st.column_config.NumberColumn("Bid", format="$%.2f")
        if "ask" in display.columns:
            col_cfg["ask"] = st.column_config.NumberColumn("Ask", format="$%.2f")
        if "spread" in display.columns:
            col_cfg["spread"] = st.column_config.NumberColumn("Spread", format="$%.3f")
        if "moneyness" in display.columns:
            col_cfg["moneyness"] = st.column_config.NumberColumn("Moneyness", format="%.3f")

        st.dataframe(
            display,
            use_container_width=True,
            height=500,
            column_config=col_cfg,
        )

    # ── Tab 3: Analytics ───────────────────────────────────────────────

    def _tab_analytics(self, df: pd.DataFrame):
        st.markdown("### Distribution Analytics")
        col1, col2 = st.columns(2)

        with col1:
            if "volume" in df.columns:
                fig = px.histogram(
                    df[df["volume"] < df["volume"].quantile(0.95)],
                    x="volume", nbins=40,
                    title="Volume Distribution",
                    color_discrete_sequence=["#3ab54a"],
                )
                fig.update_layout(
                    paper_bgcolor="#0d1b2e", plot_bgcolor="#152847",
                    font=dict(color="#e2e8f0"),
                    xaxis=dict(gridcolor="#1e3a5f"),
                    yaxis=dict(gridcolor="#1e3a5f"),
                    margin=dict(l=0, r=0, t=40, b=0), height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "iv_%" in df.columns:
                fig2 = px.histogram(
                    df, x="iv_%", nbins=40,
                    title="IV % Distribution",
                    color_discrete_sequence=["#1a3a8f"],
                )
                fig2.update_layout(
                    paper_bgcolor="#0d1b2e", plot_bgcolor="#152847",
                    font=dict(color="#e2e8f0"),
                    xaxis=dict(gridcolor="#1e3a5f"),
                    yaxis=dict(gridcolor="#1e3a5f"),
                    margin=dict(l=0, r=0, t=40, b=0), height=300,
                )
                st.plotly_chart(fig2, use_container_width=True)

        if {"symbol", "volume", "iv_%", "option_type"}.issubset(df.columns):
            st.markdown("### Volume Source by Ticker")
            source = df.copy()
            source = source[source["volume"] > 0].sort_values("volume", ascending=False).head(350)
            hover_cols = [
                c for c in [
                    "symbol", "option_type", "expiration", "dte", "strike",
                    "volume", "open_interest", "iv_%", "bid", "ask",
                ]
                if c in source.columns
            ]
            fig3 = px.scatter(
                source,
                x="symbol",
                y="volume",
                color="option_type",
                size="volume",
                size_max=34,
                hover_data=hover_cols,
                labels={"symbol": "Ticker", "volume": "Contract Volume", "iv_%": "IV %"},
                color_discrete_map={"Call": "#3ab54a", "Put": "#ef4444"},
            )
            fig3.update_traces(
                marker=dict(opacity=0.72, line=dict(width=1, color="rgba(226,232,240,0.35)"))
            )
            fig3.update_layout(
                paper_bgcolor="#0d1b2e",
                plot_bgcolor="#152847",
                font=dict(color="#e2e8f0", family="Inter"),
                xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
                yaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
                legend=dict(
                    bgcolor="rgba(21,40,71,0.85)",
                    bordercolor="#1e3a5f",
                    borderwidth=1,
                    font=dict(color="#e2e8f0", size=12),
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                height=420,
            )
            st.plotly_chart(fig3, use_container_width=True, key="ol_analytics_ticker_source")

        # Top opportunities table
        st.markdown("### 🏆 Top Opportunities by Volume")
        top = df.nlargest(15, "volume") if "volume" in df.columns else df.head(15)
        display_cols = [c for c in ["symbol", "option_type", "strike", "expiration",
                                     "dte", "volume", "open_interest", "iv_%", "bid", "ask"] if c in top.columns]
        st.dataframe(top[display_cols], use_container_width=True, hide_index=True)

    # ── Tab 4: IV Landscape ────────────────────────────────────────────

    def _tab_iv_landscape(self, df: pd.DataFrame, iv_ranks: dict):
        st.markdown("### IV Landscape")

        if iv_ranks:
            rank_df = pd.DataFrame([
                {"Symbol": sym, "IV Rank": rank}
                for sym, rank in sorted(iv_ranks.items(), key=lambda x: x[1], reverse=True)
                if rank is not None
            ])
            colors = ["#ef4444" if r >= 75 else "#f59e0b" if r >= 50 else "#3ab54a"
                      for r in rank_df["IV Rank"]]
            fig = go.Figure(go.Bar(
                x=rank_df["Symbol"],
                y=rank_df["IV Rank"],
                marker_color=colors,
                text=rank_df["IV Rank"].round(0).astype(int),
                textposition="outside",
            ))
            fig.add_hline(y=75, line_dash="dash", line_color="#ef4444",
                          annotation_text="High IV (75)", annotation_position="top right")
            fig.add_hline(y=50, line_dash="dash", line_color="#f59e0b",
                          annotation_text="Mid IV (50)", annotation_position="top right")
            fig.update_layout(
                title=dict(text="IV Rank by Symbol (🟢 Low | 🟡 Mid | 🔴 High)", font=dict(color="#e2e8f0")),
                paper_bgcolor="#0d1b2e", plot_bgcolor="#152847",
                font=dict(color="#e2e8f0", family="Inter"),
                xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
                yaxis=dict(gridcolor="#1e3a5f", range=[0, 105], title="IV Rank", tickfont=dict(color="#e2e8f0")),
                margin=dict(l=0, r=0, t=50, b=0), height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Call/Put volume by symbol
        if "symbol" in df.columns and "volume" in df.columns and "option_type" in df.columns:
            cp = df.groupby(["symbol", "option_type"])["volume"].sum().reset_index()
            cp["IV Rank"] = cp["symbol"].map(iv_ranks)
            fig2 = px.bar(
                cp, x="symbol", y="volume", color="option_type",
                title="Call vs Put Volume by Symbol",
                barmode="stack",
                hover_data={"symbol": True, "option_type": True, "volume": ":,", "IV Rank": ":.1f"},
                color_discrete_map={"Call": "#3ab54a", "Put": "#ef4444"},
            )
            fig2.update_layout(
                paper_bgcolor="#0d1b2e", plot_bgcolor="#152847",
                font=dict(color="#e2e8f0"),
                xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
                yaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
                legend=dict(
                    bgcolor="rgba(21,40,71,0.85)",
                    bordercolor="#1e3a5f",
                    borderwidth=1,
                    font=dict(color="#e2e8f0", size=13),
                    title=dict(text="Type", font=dict(color="#94a3b8", size=11)),
                ),
                margin=dict(l=0, r=0, t=50, b=0), height=320,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Welcome state ──────────────────────────────────────────────────

    def _render_welcome(self):
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, rgba(26,58,143,0.2) 0%, rgba(58,181,74,0.08) 100%);
                border: 1px solid #1e3a5f;
                border-left: 4px solid #3ab54a;
                border-radius: 12px;
                padding: 28px 32px;
                margin: 20px 0;
            ">
                <div style="font-size:32px;margin-bottom:12px;">💧</div>
                <div style="color:#3ab54a;font-size:20px;font-weight:700;margin-bottom:8px;">
                    Options Liquidity Discovery Engine
                </div>
                <div style="color:#94a3b8;font-size:14px;line-height:1.8;">
                    Scan your options watchlist for the highest-liquidity contracts with elevated Implied Volatility.<br><br>
                    <strong style="color:#e2e8f0;">What you get:</strong><br>
                    📊 Volume & Open Interest heatmaps<br>
                    📈 IV Rank percentile for each symbol<br>
                    🔥 Top contracts ranked by liquidity<br>
                    📥 Export results to CSV
                </div>
                <div style="margin-top:20px;color:#64748b;font-size:13px;">
                    ← Configure your watchlist and filters in the sidebar, then click <strong>🔍 Scan Options</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick-start default symbols preview
        st.markdown("### Default Watchlist Preview")
        preview = pd.DataFrame({"Symbol": DEFAULT_SYMBOLS, "Status": ["Ready to scan"] * len(DEFAULT_SYMBOLS)})
        cols = st.columns(5)
        for i, sym in enumerate(DEFAULT_SYMBOLS):
            with cols[i % 5]:
                st.markdown(
                    f"<div style='background:rgba(21,40,71,0.6);border:1px solid #1e3a5f;border-radius:8px;padding:10px;text-align:center;margin-bottom:8px;'><span style='color:#e2e8f0;font-weight:600;'>{sym}</span></div>",
                    unsafe_allow_html=True,
                )
