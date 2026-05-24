"""
FazDane Analytics - Tier 2
Portfolio Performance
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from modules.base_module import FazDaneModule
from utils.tastytrade_provider import (
    TastytradeProviderError,
    fetch_market_data_by_type,
    load_config,
)
from utils.portfolio_performance_store import (
    get_database_status,
    get_latest_portfolio_details,
    get_latest_portfolio_positions,
    get_portfolio_history,
    get_recent_portfolio_snapshots,
    parse_schwab_position_details_csv,
    parse_schwab_positions_csv,
    save_portfolio_snapshot,
    summarize_positions,
)


BRAND = {
    "bg": "#0d1b2e",
    "panel": "#152847",
    "grid": "#1e3a5f",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "green": "#3ab54a",
    "red": "#ef4444",
    "blue": "#93c5fd",
    "yellow": "#facc15",
    "purple": "#a78bfa",
}


def fmt_money(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def fmt_num(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):,.2f}"


def style_figure(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=BRAND["bg"],
        plot_bgcolor=BRAND["panel"],
        font=dict(color=BRAND["text"], family="Inter"),
        margin=dict(l=24, r=24, t=56, b=64),
        legend=dict(
            bgcolor="rgba(21,40,71,0.82)",
            bordercolor=BRAND["grid"],
            borderwidth=1,
            font=dict(color=BRAND["text"]),
        ),
        xaxis=dict(
            gridcolor=BRAND["grid"],
            zerolinecolor=BRAND["muted"],
            tickfont=dict(color=BRAND["text"]),
            automargin=True,
        ),
        yaxis=dict(
            gridcolor=BRAND["grid"],
            zerolinecolor=BRAND["muted"],
            tickfont=dict(color=BRAND["text"]),
            automargin=True,
        ),
    )
    return fig


def metric_delta(value: float, suffix: str = "") -> str:
    return f"{value:+,.2f}{suffix}"


@st.cache_data(ttl=180, show_spinner=False)
def fetch_daily_net_change(symbols: tuple[str, ...], source_preference: str) -> pd.DataFrame:
    """Fetch current daily price change for portfolio tickers."""
    clean_symbols = tuple(dict.fromkeys(str(symbol).upper() for symbol in symbols if str(symbol).strip()))
    if not clean_symbols:
        return pd.DataFrame()

    frames = []
    source_label = source_preference.lower()
    use_tasty = source_label in {"tastytrade first", "tastytrade only"}
    use_yahoo = source_label in {"tastytrade first", "yfinance only"}

    if use_tasty:
        tasty_df = _fetch_tastytrade_daily_change(clean_symbols)
        if not tasty_df.empty:
            frames.append(tasty_df)
        if source_label == "tastytrade only":
            return _dedupe_daily_change(frames)

    existing = set(pd.concat(frames)["ticker"]) if frames else set()
    missing_symbols = tuple(symbol for symbol in clean_symbols if symbol not in existing)
    if use_yahoo and missing_symbols:
        yahoo_df = _fetch_yfinance_daily_change(missing_symbols)
        if not yahoo_df.empty:
            frames.append(yahoo_df)

    return _dedupe_daily_change(frames)


def _fetch_tastytrade_daily_change(symbols: tuple[str, ...]) -> pd.DataFrame:
    try:
        config = load_config()
        if not config.is_configured:
            return pd.DataFrame()
        quotes = fetch_market_data_by_type(equities=symbols, config=config)
    except (TastytradeProviderError, Exception):
        return pd.DataFrame()

    if quotes.empty:
        return pd.DataFrame()

    rows = []
    for _, quote in quotes.iterrows():
        ticker = str(quote.get("market_symbol") or "").upper()
        last = _first_numeric(quote, ["last_price", "mark", "ask", "bid"])
        previous = _first_numeric(quote, ["prev_close", "close"])
        net_change = _first_numeric(quote, ["net_change"])
        pct_change = _first_numeric(quote, ["percent_change"])
        if previous is None and last is not None and net_change is not None:
            previous = float(last) - float(net_change)
        if pct_change is not None and abs(pct_change) <= 1:
            pct_change = pct_change * 100
        if not ticker or last is None or previous is None or previous == 0:
            continue
        if pct_change is None:
            pct_change = (last / previous - 1) * 100
        price_change = net_change if net_change is not None else last - previous
        rows.append(
            {
                "ticker": ticker,
                "last_price": float(last),
                "previous_close": float(previous),
                "price_change": float(price_change),
                "pct_change": float(pct_change),
                "market_source": "Tastytrade",
            }
        )
    return pd.DataFrame(rows)


def _fetch_yfinance_daily_change(symbols: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            last = _first_attr_or_key(info, ["last_price", "lastPrice", "regular_market_price", "regularMarketPrice"])
            previous = _first_attr_or_key(
                info,
                ["previous_close", "previousClose", "regular_market_previous_close", "regularMarketPreviousClose"],
            )

            if not last or not previous:
                hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
                closes = hist["Close"].dropna() if not hist.empty and "Close" in hist.columns else pd.Series(dtype=float)
                if len(closes) >= 2:
                    last = float(closes.iloc[-1])
                    previous = float(closes.iloc[-2])

            if last is None or previous is None or float(previous) == 0:
                continue

            last_float = float(last)
            previous_float = float(previous)
            rows.append(
                {
                    "ticker": symbol,
                    "last_price": last_float,
                    "previous_close": previous_float,
                    "price_change": last_float - previous_float,
                    "pct_change": (last_float / previous_float - 1) * 100,
                    "market_source": "yfinance",
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def _dedupe_daily_change(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return combined
    return combined.drop_duplicates("ticker", keep="first")


def _first_numeric(row: pd.Series, columns: list[str]) -> float | None:
    for column in columns:
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            return float(value)
    return None


def _first_attr_or_key(value: object, names: list[str]) -> float | None:
    for name in names:
        try:
            found = getattr(value, name)
        except Exception:
            found = None
        if found is None:
            try:
                found = value.get(name)  # type: ignore[attr-defined]
            except Exception:
                found = None
        try:
            if found is not None:
                return float(found)
        except (TypeError, ValueError):
            continue
    return None


class PortfolioPerformanceModule(FazDaneModule):
    MODULE_NAME = "Portfolio Performance"
    MODULE_ICON = "PF"
    MODULE_DESCRIPTION = "Schwab position snapshots, risk exposure, and saved daily portfolio progress"
    TIER = 2
    SOURCE_NOTEBOOK = "FazDane Analytics Portfolio Dashboard Generator"
    CACHE_TTL = 300
    REQUIRES_LIVE_DATA = False
    DATA_SOURCES = ["Schwab Position Statement CSV", "Local SQLite"]

    def render_sidebar(self):
        st.markdown("**Snapshot Source**")
        self.uploaded_file = st.file_uploader(
            "Schwab Position Statement CSV",
            type=["csv"],
            key="pp_csv_upload",
        )
        self.load_latest_saved = st.checkbox(
            "Use latest saved snapshot when no file is uploaded",
            value=True,
            key="pp_use_latest",
        )
        self.auto_save = st.checkbox(
            "Save parsed upload to daily database",
            value=True,
            key="pp_auto_save",
        )

        st.markdown("**View Settings**")
        self.top_n = st.slider("Rows per leaderboard", min_value=3, max_value=15, value=7, key="pp_top_n")
        self.history_days = st.selectbox("History Window", [30, 60, 90, 180, 365], index=2, key="pp_hist_days")
        self.delta_caution = st.number_input(
            "High Delta Caution Level",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
            key="pp_delta_caution",
        )
        self.market_data_source = st.selectbox(
            "Daily Net Change Source",
            ["yfinance Only", "Tastytrade First", "Tastytrade Only"],
            index=0,
            key="pp_daily_change_source",
        )

        if st.button("Refresh Current Pricing", use_container_width=True, type="primary", key="pp_refresh_prices"):
            fetch_daily_net_change.clear()
            st.rerun()

        if st.button("Refresh Portfolio View", use_container_width=True, key="pp_refresh"):
            st.session_state.pop("pp_last_saved_hash", None)
            fetch_daily_net_change.clear()
            st.rerun()

        self._render_database_status()

    def render_main(self):
        self.render_section_header(
            "Portfolio Performance",
            "Upload Schwab positions, save daily snapshots, and monitor P/L, theta, and delta concentration.",
        )

        positions, details, metadata, source_label = self._load_active_snapshot()
        if positions.empty:
            self._render_welcome()
            self._render_history()
            return

        saved_info = None
        if self.uploaded_file is not None and self.auto_save:
            saved_info = self._save_uploaded_snapshot_once(positions, details, metadata)

        totals = summarize_positions(positions)
        self._render_source_banner(metadata, source_label, saved_info)
        self._render_metrics(totals, positions)

        tab_overview, tab_risk, tab_history, tab_data, tab_details = st.tabs(
            ["Overview", "Risk Map", "Daily History", "Position Data", "Raw Details"]
        )
        with tab_overview:
            self._render_overview(positions, totals)
        with tab_risk:
            self._render_risk_map(positions)
        with tab_history:
            self._render_history()
        with tab_data:
            self._render_data_tab(positions, metadata)
        with tab_details:
            self._render_details_tab(details, metadata)

    def _load_active_snapshot(self) -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
        if self.uploaded_file is not None:
            content = self.uploaded_file.getvalue()
            positions, metadata = parse_schwab_positions_csv(content, self.uploaded_file.name)
            details, detail_metadata = parse_schwab_position_details_csv(content, self.uploaded_file.name)
            metadata.update({key: value for key, value in detail_metadata.items() if key.endswith("_count")})
            return positions, details, metadata, "Uploaded Schwab CSV"

        if self.load_latest_saved:
            latest_positions, latest_metadata = get_latest_portfolio_positions()
            if latest_metadata:
                latest_details, _ = get_latest_portfolio_details()
                return latest_positions, latest_details, latest_metadata, "Latest saved snapshot"

        return pd.DataFrame(), pd.DataFrame(), {}, "No snapshot"

    def _save_uploaded_snapshot_once(self, positions: pd.DataFrame, details: pd.DataFrame, metadata: dict) -> dict | None:
        file_hash = metadata.get("file_sha256")
        last_hash = st.session_state.get("pp_last_saved_hash")
        if file_hash and last_hash == file_hash:
            return st.session_state.get("pp_last_save_info")

        try:
            saved = save_portfolio_snapshot(positions, metadata, details=details)
            st.session_state["pp_last_saved_hash"] = file_hash
            st.session_state["pp_last_save_info"] = saved
            return saved
        except Exception as exc:
            st.warning(f"Snapshot parsed, but database save failed: {exc}")
            return None

    def _render_source_banner(self, metadata: dict, source_label: str, saved_info: dict | None):
        source_file = metadata.get("source_file", "Unknown file")
        snapshot_ts = metadata.get("snapshot_ts", "")
        saved_text = ""
        if saved_info:
            saved_text = f"<div style='color:{BRAND['green']};font-size:12px;margin-top:6px;'>Saved run {saved_info['run_id']}</div>"

        st.markdown(
            f"""
            <div style="
                background:linear-gradient(135deg, rgba(26,58,143,0.28), rgba(58,181,74,0.08));
                border:1px solid {BRAND['grid']};
                border-left:4px solid {BRAND['green']};
                border-radius:10px;
                padding:14px 18px;
                margin:8px 0 16px 0;
            ">
                <div style="color:{BRAND['green']};font-size:15px;font-weight:700;">{source_label}</div>
                <div style="color:{BRAND['text']};font-size:13px;margin-top:4px;">{source_file}</div>
                <div style="color:{BRAND['muted']};font-size:12px;margin-top:3px;">Snapshot time: {snapshot_ts}</div>
                {saved_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_metrics(self, totals: dict[str, float], positions: pd.DataFrame):
        winners = int((positions["pl_day"] > 0).sum())
        losers = int((positions["pl_day"] < 0).sum())
        win_rate = winners / max(winners + losers, 1) * 100

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("P/L Open", fmt_money(totals["total_pl_open"]), fmt_money(totals["positive_open"]))
        m2.metric("P/L Day", fmt_money(totals["total_pl_day"]), f"{win_rate:.0f}% day win rate")
        m3.metric("Theta", fmt_num(totals["total_theta"]), metric_delta(totals["total_theta"]))
        m4.metric("Delta", fmt_num(totals["total_delta"]), "Net exposure")
        m5.metric("Tickers", f"{len(positions):,}", f"{winners} up / {losers} down")

    def _render_overview(self, positions: pd.DataFrame, totals: dict[str, float]):
        left, right = st.columns([1.35, 1.0])
        with left:
            st.markdown("### P/L by Ticker")
            value_col = st.radio(
                "Performance Field",
                ["pl_open", "pl_day"],
                horizontal=True,
                format_func=lambda value: "P/L Open" if value == "pl_open" else "P/L Day",
                key="pp_perf_field",
                label_visibility="collapsed",
            )
            chart_df = positions.sort_values(value_col, ascending=True).copy()
            chart_df["color"] = chart_df[value_col].map(lambda value: "Gain" if value >= 0 else "Loss")
            fig = px.bar(
                chart_df,
                x=value_col,
                y="ticker",
                orientation="h",
                color="color",
                color_discrete_map={"Gain": BRAND["green"], "Loss": BRAND["red"]},
                hover_data=["theta", "delta", "market_value", "account_group"],
                labels={value_col: "Dollars", "ticker": "Ticker"},
            )
            fig.add_vline(x=0, line_width=1, line_color=BRAND["muted"])
            fig.update_traces(texttemplate="%{x:,.0f}", textposition="outside", cliponaxis=False)
            style_figure(fig, height=max(380, 28 * len(chart_df) + 80))
            fig.update_layout(
                yaxis=dict(
                    anchor="free",
                    position=0.0,
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    title=None,
                ),
                margin=dict(l=24, r=24, t=56, b=85),
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)
            self._render_daily_net_change_chart(positions)

        with right:
            self._render_insights(positions, totals)

        st.markdown("### Exposure Leaders")
        c1, c2 = st.columns(2)
        with c1:
            theta_df = positions.sort_values("theta", ascending=False).head(self.top_n)
            fig = px.bar(
                theta_df,
                x="ticker",
                y="theta",
                color="theta",
                color_continuous_scale=[[0, BRAND["red"]], [0.5, BRAND["yellow"]], [1, BRAND["green"]]],
                labels={"theta": "Theta", "ticker": "Ticker"},
                title="Theta Generators",
            )
            style_figure(fig, height=330)
            st.plotly_chart(fig, use_container_width=True, theme=None)

        with c2:
            delta_df = positions.reindex(positions["delta"].abs().sort_values(ascending=False).index).head(self.top_n)
            fig = px.bar(
                delta_df,
                x="ticker",
                y="delta",
                color="delta",
                color_continuous_scale=[[0, BRAND["red"]], [0.5, BRAND["yellow"]], [1, BRAND["blue"]]],
                labels={"delta": "Delta", "ticker": "Ticker"},
                title="Largest Delta Exposures",
            )
            fig.add_hline(y=self.delta_caution, line_dash="dash", line_color=BRAND["yellow"])
            style_figure(fig, height=330)
            st.plotly_chart(fig, use_container_width=True, theme=None)

    def _render_daily_net_change_chart(self, positions: pd.DataFrame):
        st.markdown("### Daily Net Change by Portfolio Ticker")
        symbols = tuple(positions["ticker"].dropna().astype(str).str.upper().unique())
        with st.spinner("Fetching daily market change for portfolio tickers..."):
            market = fetch_daily_net_change(symbols, self.market_data_source)

        if market.empty:
            st.info("Daily market change is unavailable from the selected source. Try Tastytrade First or yfinance Only.")
            return

        quantity = positions[["ticker", "quantity", "market_value"]].copy()
        quantity["ticker"] = quantity["ticker"].astype(str).str.upper()
        daily = market.merge(quantity, on="ticker", how="left")
        daily["quantity"] = pd.to_numeric(daily["quantity"], errors="coerce").fillna(0)
        daily["market_value"] = pd.to_numeric(daily["market_value"], errors="coerce").fillna(0)
        daily["estimated_dollar_change"] = daily["quantity"] * daily["price_change"]
        fallback = daily["market_value"] * daily["pct_change"] / 100
        daily["estimated_dollar_change"] = daily["estimated_dollar_change"].where(
            daily["estimated_dollar_change"].abs() > 0,
            fallback,
        )
        daily = daily.sort_values("pct_change", ascending=True)
        daily["direction"] = daily["pct_change"].map(lambda value: "Up" if value >= 0 else "Down")
        daily["pct_label"] = daily["pct_change"].map(lambda value: f"{value:+.2f}%")

        fig = px.bar(
            daily,
            x="pct_change",
            y="ticker",
            text="pct_label",
            orientation="h",
            color="direction",
            color_discrete_map={"Up": BRAND["green"], "Down": BRAND["red"]},
            hover_data={
                "last_price": ":$.2f",
                "previous_close": ":$.2f",
                "price_change": ":$.2f",
                "pct_change": ":.2f",
                "pct_label": False,
                "estimated_dollar_change": ":$.2f",
                "market_source": True,
                "direction": False,
            },
            labels={"pct_change": "Daily Change (%)", "ticker": "Ticker"},
            title="Market-Sourced Daily Price Performance",
        )
        fig.add_vline(x=0, line_width=1, line_color=BRAND["muted"])
        style_figure(fig, height=max(360, 26 * len(daily) + 96))
        fig.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False)
        fig.update_xaxes(ticksuffix="%", tickformat=".2f", separatethousands=False)
        fig.update_layout(
            yaxis=dict(
                anchor="free",
                position=0.0,
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                title=None,
            ),
            margin=dict(l=24, r=24, t=56, b=85),
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

        total_estimate = daily["estimated_dollar_change"].sum()
        source_mix = ", ".join(sorted(daily["market_source"].dropna().unique()))
        st.caption(
            f"Estimated net dollar move from displayed quantities: {fmt_money(total_estimate)}. "
            f"Source: {source_mix}."
        )
        if st.button("Refresh Current Pricing", key="pp_inline_refresh_prices", use_container_width=True):
            fetch_daily_net_change.clear()
            st.rerun()

    def _render_insights(self, positions: pd.DataFrame, totals: dict[str, float]):
        top_open = positions.sort_values("pl_open", ascending=False).head(self.top_n)
        open_drags = positions.sort_values("pl_open").head(self.top_n)
        day_leaders = positions.sort_values("pl_day", ascending=False).head(self.top_n)
        watch = self._build_watchlist(positions)

        st.markdown("### Insights")
        self._leaderboard("Top Open Winners", top_open, "pl_open", BRAND["green"], money=True)
        self._leaderboard("Largest Open Drags", open_drags, "pl_open", BRAND["red"], money=True)
        self._leaderboard("Day Leaders", day_leaders, "pl_day", BRAND["green"], money=True)

        watch_text = ", ".join(watch) if watch else "No concentrated caution names in this snapshot."
        st.markdown(
            f"""
            <div style="
                background:rgba(21,40,71,0.70);
                border:1px solid {BRAND['grid']};
                border-radius:8px;
                padding:14px 16px;
                margin-top:10px;
            ">
                <div style="color:{BRAND['yellow']};font-weight:700;font-size:13px;">Watchlist / Caution</div>
                <div style="color:{BRAND['muted']};font-size:13px;line-height:1.55;margin-top:6px;">
                    {watch_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _leaderboard(self, title: str, df: pd.DataFrame, column: str, color: str, money: bool = False):
        rows = []
        for _, row in df.iterrows():
            value = fmt_money(row[column]) if money else f"{row[column]:,.2f}"
            rows.append(
                f"<div style='display:flex;justify-content:space-between;gap:16px;margin:4px 0;'>"
                f"<span style='color:{BRAND['text']};font-weight:600;'>{row['ticker']}</span>"
                f"<span style='color:{color};font-weight:700;'>{value}</span>"
                f"</div>"
            )
        st.markdown(
            f"""
            <div style="
                background:rgba(21,40,71,0.55);
                border:1px solid {BRAND['grid']};
                border-radius:8px;
                padding:12px 14px;
                margin-bottom:10px;
            ">
                <div style="color:{color};font-weight:700;font-size:13px;margin-bottom:6px;">{title}</div>
                {''.join(rows)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_risk_map(self, positions: pd.DataFrame):
        st.markdown("### Delta vs Theta Risk Map")
        risk = positions.copy()
        risk["bubble_size"] = risk["pl_open"].abs().clip(lower=25)
        risk["quadrant"] = risk.apply(self._risk_quadrant, axis=1)

        fig = px.scatter(
            risk,
            x="delta",
            y="theta",
            size="bubble_size",
            color="quadrant",
            text="ticker",
            hover_data=["pl_open", "pl_day", "market_value", "account_group"],
            color_discrete_map={
                "Constructive": BRAND["green"],
                "Directional": BRAND["blue"],
                "Decay Drag": BRAND["yellow"],
                "Pressure": BRAND["red"],
            },
            labels={"delta": "Delta", "theta": "Theta"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color=BRAND["muted"])
        fig.add_vline(x=self.delta_caution, line_dash="dash", line_color=BRAND["yellow"])
        fig.update_traces(textposition="top center", marker=dict(opacity=0.78, line=dict(width=1, color="#e2e8f0")))
        style_figure(fig, height=520)
        st.plotly_chart(fig, use_container_width=True, theme=None)

        st.markdown("### Caution Candidates")
        caution = positions.loc[positions["ticker"].isin(self._build_watchlist(positions))].copy()
        if caution.empty:
            st.info("No names matched the current caution rules.")
        else:
            self._render_position_table(caution.sort_values(["delta", "theta"], ascending=[False, True]))

    def _render_history(self):
        st.markdown("### Saved Daily Progress")
        history = get_portfolio_history(days=self.history_days)
        recent = get_recent_portfolio_snapshots(limit=15)
        if history.empty:
            st.info("No saved portfolio snapshots yet. Upload a Schwab CSV and keep database save enabled to start the daily progress record.")
            return

        h1, h2, h3 = st.columns(3)
        latest = history.iloc[-1]
        previous = history.iloc[-2] if len(history) > 1 else None
        h1.metric(
            "Latest Open P/L",
            fmt_money(latest["total_pl_open"]),
            fmt_money(latest["total_pl_open"] - previous["total_pl_open"]) if previous is not None else None,
        )
        h2.metric(
            "Latest Day P/L",
            fmt_money(latest["total_pl_day"]),
            fmt_money(latest["total_pl_day"] - previous["total_pl_day"]) if previous is not None else None,
        )
        h3.metric(
            "Net Theta / Delta",
            f"{latest['total_theta']:,.1f} / {latest['total_delta']:,.1f}",
            "Saved snapshot totals",
        )

        chart = history.copy()
        chart["snapshot_ts"] = pd.to_datetime(chart["snapshot_ts"], errors="coerce")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart["snapshot_ts"],
            y=chart["total_pl_open"],
            mode="lines+markers",
            name="P/L Open",
            line=dict(color=BRAND["green"], width=3),
        ))
        fig.add_trace(go.Bar(
            x=chart["snapshot_ts"],
            y=chart["total_pl_day"],
            name="P/L Day",
            marker_color=chart["total_pl_day"].map(lambda value: BRAND["green"] if value >= 0 else BRAND["red"]),
            opacity=0.58,
            yaxis="y2",
        ))
        fig.update_layout(
            title="Portfolio Progress by Saved Snapshot",
            yaxis=dict(title="P/L Open", gridcolor=BRAND["grid"]),
            yaxis2=dict(title="P/L Day", overlaying="y", side="right", showgrid=False),
            barmode="overlay",
        )
        style_figure(fig, height=420)
        st.plotly_chart(fig, use_container_width=True, theme=None)

        if not recent.empty:
            st.markdown("### Recent Saved Snapshots")
            st.dataframe(
                recent,
                use_container_width=True,
                hide_index=True,
                height=460,
                column_config={
                    "snapshot_ts": st.column_config.TextColumn("Snapshot Time"),
                    "snapshot_date": st.column_config.TextColumn("Date"),
                    "source_file": st.column_config.TextColumn("Source"),
                    "row_count": st.column_config.NumberColumn("Tickers", format="%d"),
                    "total_pl_open": st.column_config.NumberColumn("P/L Open", format="$%.2f"),
                    "total_pl_day": st.column_config.NumberColumn("P/L Day", format="$%.2f"),
                    "total_theta": st.column_config.NumberColumn("Theta", format="%.2f"),
                    "total_delta": st.column_config.NumberColumn("Delta", format="%.2f"),
                },
            )

    def _render_database_status(self):
        status = get_database_status()
        with st.expander("Database Storage", expanded=False):
            st.caption(status["db_path"])
            if status.get("warning"):
                st.warning(status["warning"])
            elif status["configured_env_path"]:
                st.success("Using PORTFOLIO_PERFORMANCE_DB_PATH.")
            else:
                st.info("Using the default local development database path.")

            c1, c2 = st.columns(2)
            c1.metric("Saved Runs", f"{status['run_count']:,}")
            c2.metric("Saved Positions", f"{status['position_count']:,}")
            if status.get("latest_snapshot_ts"):
                st.caption(
                    f"Latest snapshot: {status['latest_snapshot_ts']} "
                    f"from {status.get('latest_source_file') or 'unknown source'}"
                )

    def _render_data_tab(self, positions: pd.DataFrame, metadata: dict):
        st.markdown("### Parsed Position Snapshot")
        self._render_position_table(positions)
        csv_bytes = positions.to_csv(index=False).encode("utf-8")
        date_part = str(metadata.get("snapshot_date") or datetime.now().date())
        st.download_button(
            "Download Parsed Snapshot CSV",
            data=csv_bytes,
            file_name=f"portfolio_performance_{date_part}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    def _render_details_tab(self, details: pd.DataFrame, metadata: dict):
        st.markdown("### Raw Broker Position Details")
        if details.empty:
            st.info("No raw detail rows were saved for this snapshot. Re-upload the Schwab CSV to persist option legs.")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detail Rows", f"{len(details):,}")
        c2.metric("Option Legs", f"{int((details['row_type'] == 'option_leg').sum()):,}")
        c3.metric("Strategies", f"{int((details['row_type'] == 'strategy').sum()):,}")
        c4.metric("Tickers", f"{details['underlying'].dropna().nunique():,}")

        row_types = sorted(details["row_type"].dropna().unique().tolist())
        selected_types = st.multiselect(
            "Detail Row Types",
            row_types,
            default=[row_type for row_type in ["ticker_summary", "strategy", "option_leg"] if row_type in row_types],
            key="pp_detail_row_types",
        )
        underlyings = sorted(details["underlying"].dropna().unique().tolist())
        selected_underlying = st.selectbox("Underlying", ["All"] + underlyings, key="pp_detail_underlying")

        view = details.copy()
        if selected_types:
            view = view[view["row_type"].isin(selected_types)]
        if selected_underlying != "All":
            view = view[view["underlying"] == selected_underlying]

        display_cols = [
            "account_group",
            "underlying",
            "row_type",
            "strategy",
            "instrument",
            "side",
            "quantity",
            "days",
            "expiration",
            "strike",
            "call_put",
            "trade_price",
            "mark_price",
            "delta",
            "theta",
            "gamma",
            "vega",
            "pl_open",
            "pl_day",
        ]
        available = [column for column in display_cols if column in view.columns]
        st.dataframe(
            view[available],
            use_container_width=True,
            hide_index=True,
            height=min(820, max(460, 34 * (len(view) + 1))),
            column_config={
                "quantity": st.column_config.NumberColumn("Contracts", format="%.0f"),
                "trade_price": st.column_config.NumberColumn("Trade Price", format="$%.3f"),
                "mark_price": st.column_config.NumberColumn("Mark", format="$%.3f"),
                "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                "delta": st.column_config.NumberColumn("Delta", format="%.2f"),
                "theta": st.column_config.NumberColumn("Theta", format="%.2f"),
                "gamma": st.column_config.NumberColumn("Gamma", format="%.2f"),
                "vega": st.column_config.NumberColumn("Vega", format="%.2f"),
                "pl_open": st.column_config.NumberColumn("P/L Open", format="$%.2f"),
                "pl_day": st.column_config.NumberColumn("P/L Day", format="$%.2f"),
            },
        )

        csv_bytes = view.to_csv(index=False).encode("utf-8")
        date_part = str(metadata.get("snapshot_date") or datetime.now().date())
        st.download_button(
            "Download Raw Detail Rows CSV",
            data=csv_bytes,
            file_name=f"portfolio_details_{date_part}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    def _render_position_table(self, positions: pd.DataFrame):
        display_cols = [
            "ticker",
            "account_group",
            "quantity",
            "mark_price",
            "market_value",
            "pl_open",
            "pl_day",
            "theta",
            "delta",
            "bp_effect",
        ]
        available = [column for column in display_cols if column in positions.columns]
        st.dataframe(
            positions[available],
            use_container_width=True,
            hide_index=True,
            height=min(820, max(460, 38 * (len(positions) + 1))),
            column_config={
                "ticker": st.column_config.TextColumn("Ticker"),
                "account_group": st.column_config.TextColumn("Group"),
                "quantity": st.column_config.NumberColumn("Qty", format="%.2f"),
                "mark_price": st.column_config.NumberColumn("Mark", format="$%.2f"),
                "market_value": st.column_config.NumberColumn("Market Value", format="$%.2f"),
                "pl_open": st.column_config.NumberColumn("P/L Open", format="$%.2f"),
                "pl_day": st.column_config.NumberColumn("P/L Day", format="$%.2f"),
                "theta": st.column_config.NumberColumn("Theta", format="%.2f"),
                "delta": st.column_config.NumberColumn("Delta", format="%.2f"),
                "bp_effect": st.column_config.NumberColumn("BP Effect", format="$%.2f"),
            },
        )

    def _build_watchlist(self, positions: pd.DataFrame) -> list[str]:
        if positions.empty:
            return []
        delta_threshold = max(self.delta_caution, float(positions["delta"].quantile(0.75)))
        risk = positions.copy()
        risk["risk_score"] = (
            risk["pl_open"].lt(0).astype(int)
            + risk["pl_day"].lt(0).astype(int)
            + risk["theta"].lt(0).astype(int)
            + risk["delta"].gt(delta_threshold).astype(int)
        )
        return (
            risk[risk["risk_score"] > 0]
            .sort_values(["risk_score", "delta"], ascending=False)
            .head(self.top_n)["ticker"]
            .tolist()
        )

    @staticmethod
    def _risk_quadrant(row: pd.Series) -> str:
        if row["pl_open"] < 0 and row["theta"] < 0:
            return "Pressure"
        if row["theta"] < 0:
            return "Decay Drag"
        if row["delta"] > 0 and row["pl_open"] >= 0:
            return "Constructive"
        return "Directional"

    def _render_welcome(self):
        st.markdown(
            f"""
            <div style="
                background:linear-gradient(135deg, rgba(26,58,143,0.24), rgba(58,181,74,0.08));
                border:1px solid {BRAND['grid']};
                border-left:4px solid {BRAND['green']};
                border-radius:10px;
                padding:22px 24px;
                margin:12px 0 22px 0;
            ">
                <div style="color:{BRAND['green']};font-size:18px;font-weight:700;margin-bottom:8px;">Portfolio Performance</div>
                <div style="color:{BRAND['muted']};font-size:14px;line-height:1.7;">
                    Upload a Schwab Position Statement CSV from the sidebar. The module parses ticker-level
                    positions, builds performance and risk views, and can save each upload as a local daily snapshot.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
