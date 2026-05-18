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
    Cached for 5 minutes.
    """
    results = []

    exp_map = {"Weekly (≤8 days)": 8, "Monthly (9–45 days)": 45, "Any": 365}
    max_dte = exp_map.get(exp_pref, 365)

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            spot = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
            if not spot or spot == 0:
                continue

            expirations = ticker.options
            if not expirations:
                continue

            today = datetime.today().date()

            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if dte < 0 or dte > max_dte:
                    continue

                chain = ticker.option_chain(exp_str)

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
                        results.append(df_filtered)

                # Only first matching expiration per symbol for speed
                break

        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
            continue

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    # Select & rename columns
    keep_cols = [
        "symbol", "option_type", "expiration", "dte", "spot",
        "strike", "moneyness", "iv_pct", "volume", "openInterest",
        "bid", "ask", "spread", "spread_pct", "lastPrice",
    ]
    available = [c for c in keep_cols if c in combined.columns]
    combined = combined[available].copy()

    combined.rename(columns={
        "openInterest": "open_interest",
        "lastPrice": "last_price",
        "iv_pct": "iv_%",
    }, inplace=True)

    return combined.sort_values("volume", ascending=False).reset_index(drop=True)


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
    DATA_SOURCES = ["yfinance"]

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

                # Fetch IV ranks separately
                with st.spinner("Calculating IV Ranks…"):
                    iv_ranks = fetch_iv_rank(symbols)
                    st.session_state["ol_iv_ranks"] = iv_ranks

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

        st.divider()

        # ── IV Rank Banner (if available) ────────────────────────────
        valid_ranks = {k: v for k, v in iv_ranks.items() if v is not None}
        if valid_ranks:
            self._render_iv_rank_bar(valid_ranks)
            st.divider()

        # ── Tabs ─────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(
            ["🔥 Volume Heatmap", "📋 Options Chain", "📊 Analytics", "📈 IV Landscape"]
        )

        with tab1:
            self._tab_heatmap(df)

        with tab2:
            self._tab_chain(df)

        with tab3:
            self._tab_analytics(df)

        with tab4:
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
            fig2 = px.bar(
                cp, x="symbol", y="volume", color="option_type",
                title="Call vs Put Volume by Symbol",
                barmode="stack",
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
