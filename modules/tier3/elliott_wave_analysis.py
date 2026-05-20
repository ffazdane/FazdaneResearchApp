"""
FazDane Analytics - Tier 3
Elliott Wave analysis using ZigZag pivots and rules-based impulse scoring.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from modules.base_module import FazDaneModule
from utils.universe_manager import format_ticker_display, get_ticker_names, get_universe_names, render_universe_manager


TICKER_ALIASES = {
    "SPX": "^GSPC",
    "NDX": "^NDX",
    "RUT": "^RUT",
    "NASDAQ": "^IXIC",
}


@dataclass
class Pivot:
    idx: int
    price: float
    kind: str


@dataclass
class ImpulseCandidate:
    pivot_idxs: tuple[int, ...]
    score: float
    direction: str


def normalize_symbol(symbol: str) -> str:
    clean = symbol.strip().upper()
    return TICKER_ALIASES.get(clean, clean)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_elliott_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    symbol = normalize_symbol(ticker)
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df[~df.index.duplicated(keep="last")].sort_index().dropna(how="any")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(part) for part in col if part).strip() for col in df.columns]

    close_col = pick_close_column(df)
    df["Close_PLOT"] = df[close_col].astype("float64")
    if all(col in df.columns for col in ["High", "Low"]):
        df["Mid"] = (df["High"] + df["Low"]) / 2.0
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def pick_close_column(df: pd.DataFrame) -> str:
    candidates = []
    if "Close" in df.columns:
        candidates.append("Close")
    if "Adj Close" in df.columns:
        candidates.append("Adj Close")
    candidates.extend([col for col in df.columns if "close" in str(col).lower() and col not in candidates])
    if not candidates:
        return df.columns[0]
    return candidates[0]


def zigzag_percent(prices: pd.Series, pct: float) -> list[Pivot]:
    arr = np.asarray(prices, dtype="float64").reshape(-1)
    n = int(arr.shape[0])
    if n < 3:
        return []

    threshold = float(pct) / 100.0
    pivots = []
    last_pivot_idx = 0
    last_pivot_price = float(arr[0])
    trend = None

    for i in range(1, n):
        price = float(arr[i])
        move = (price - last_pivot_price) / max(1e-12, last_pivot_price)

        if trend is None:
            if abs(move) >= threshold:
                trend = "up" if move > 0.0 else "down"
        elif trend == "up":
            if price > last_pivot_price:
                last_pivot_idx = i
                last_pivot_price = price
            else:
                drop = (last_pivot_price - price) / max(1e-12, last_pivot_price)
                if drop >= threshold:
                    pivots.append(Pivot(last_pivot_idx, last_pivot_price, "H"))
                    trend = "down"
                    last_pivot_idx = i
                    last_pivot_price = price
        else:
            if price < last_pivot_price:
                last_pivot_idx = i
                last_pivot_price = price
            else:
                rise = (price - last_pivot_price) / max(1e-12, last_pivot_price)
                if rise >= threshold:
                    pivots.append(Pivot(last_pivot_idx, last_pivot_price, "L"))
                    trend = "up"
                    last_pivot_idx = i
                    last_pivot_price = price

    if pivots:
        trailing_kind = "H" if pivots[-1].kind == "L" else "L"
        pivots.append(Pivot(last_pivot_idx, last_pivot_price, trailing_kind))
    else:
        trailing_kind = "H" if last_pivot_price >= float(arr[0]) else "L"
        pivots.append(Pivot(last_pivot_idx, last_pivot_price, trailing_kind))

    cleaned = []
    for pivot in pivots:
        if not cleaned or cleaned[-1].kind != pivot.kind:
            cleaned.append(pivot)
        elif pivot.kind == "H" and pivot.price >= cleaned[-1].price:
            cleaned[-1] = pivot
        elif pivot.kind == "L" and pivot.price <= cleaned[-1].price:
            cleaned[-1] = pivot
    return cleaned


def wave_length(a: Pivot, b: Pivot) -> float:
    return float(b.price - a.price)


def valid_impulse_sequence(seq: tuple[int, ...], pivots: list[Pivot]) -> tuple[str | None, bool]:
    points = [pivots[i] for i in seq]
    for i in range(1, len(points)):
        if points[i].kind == points[i - 1].kind:
            return None, False

    w1 = wave_length(points[0], points[1])
    w2 = wave_length(points[1], points[2])
    w3 = wave_length(points[2], points[3])
    w4 = wave_length(points[3], points[4])
    w5 = wave_length(points[4], points[5])
    direction = "bull" if w1 > 0.0 else "bear"

    if direction == "bull":
        if not (w1 > 0 and w2 < 0 and w3 > 0 and w4 < 0 and w5 > 0):
            return None, False
        if points[2].price < points[0].price:
            return None, False
    else:
        if not (w1 < 0 and w2 > 0 and w3 < 0 and w4 > 0 and w5 < 0):
            return None, False
        if points[2].price > points[0].price:
            return None, False

    if abs(w3) < min(abs(w1), abs(w3), abs(w5)):
        return None, False
    return direction, True


def impulse_score(seq: tuple[int, ...], pivots: list[Pivot]) -> float:
    points = [pivots[i] for i in seq]
    waves = [wave_length(points[i], points[i + 1]) for i in range(5)]
    score = abs(waves[2]) * 2.0

    retr2 = abs(waves[1]) / max(abs(waves[0]), 1e-9)
    if retr2 > 0.786:
        score -= 2.0
    elif retr2 > 0.618:
        score -= 1.0
    else:
        score += 0.5

    w1_high = max(points[0].price, points[1].price)
    w1_low = min(points[0].price, points[1].price)
    w4_high = max(points[3].price, points[4].price)
    w4_low = min(points[3].price, points[4].price)
    overlap = max(0.0, min(w1_high, w4_high) - max(w1_low, w4_low))
    overlap_ratio = overlap / max(1e-9, w1_high - w1_low)
    score -= overlap_ratio * 3.0

    ideal_3 = 1.618 * abs(waves[0])
    score -= abs(abs(waves[2]) - ideal_3) / (abs(ideal_3) + 1e-9)
    return float(score)


def find_best_impulse(pivots: list[Pivot], max_windows: int) -> ImpulseCandidate | None:
    best = None
    tested = 0
    for start in range(0, max(0, len(pivots) - 5)):
        if tested >= int(max_windows):
            break
        seq = tuple(range(start, start + 6))
        direction, ok = valid_impulse_sequence(seq, pivots)
        if not ok:
            continue
        score = impulse_score(seq, pivots)
        tested += 1
        if best is None or score > best.score:
            best = ImpulseCandidate(seq, score, direction or "unknown")
    return best


def fib_retracements(start: float, end: float, levels=(0.382, 0.5, 0.618, 0.786)) -> dict[str, float]:
    length = float(end) - float(start)
    return {f"{int(level * 100)}%": float(end) - level * length for level in levels}


def fib_extensions(start: float, end: float, levels=(1.0, 1.272, 1.618, 2.0, 2.618)) -> dict[str, float]:
    length = float(end) - float(start)
    return {f"{level}x": float(end) + level * length for level in levels}


def build_levels(pivots: list[Pivot], best: ImpulseCandidate | None, show_abc: bool) -> tuple[pd.DataFrame, dict | None]:
    rows = []
    abc = None
    if best is None:
        return pd.DataFrame(columns=["Group", "Level", "Price"]), abc

    seq = [pivots[i] for i in best.pivot_idxs]
    w1 = (seq[0], seq[1])
    w3 = (seq[2], seq[3])
    for key, val in fib_retracements(w1[0].price, w1[1].price).items():
        rows.append(["Wave2 retr", key, float(val)])
    for key, val in fib_retracements(w3[0].price, w3[1].price).items():
        rows.append(["Wave4 retr", key, float(val)])
    for key, val in fib_extensions(w1[0].price, w1[1].price).items():
        rows.append(["Wave3 ext", key, float(val)])
    for key, val in fib_extensions(w3[0].price, w3[1].price).items():
        rows.append(["Wave5 ext", key, float(val)])

    last_idx = best.pivot_idxs[-1]
    if show_abc and last_idx + 2 < len(pivots):
        a = (pivots[last_idx], pivots[last_idx + 1])
        b = (pivots[last_idx + 1], pivots[last_idx + 2])
        c_target = b[1].price + (a[1].price - a[0].price)
        abc = {"A": a, "B": b, "C_target": float(c_target)}
        rows.append(["ABC", "C_target", float(c_target)])

    return pd.DataFrame(rows, columns=["Group", "Level", "Price"]), abc


def pivots_to_frame(df: pd.DataFrame, pivots: list[Pivot]) -> pd.DataFrame:
    rows = []
    for pivot in pivots:
        dt = df.index[pivot.idx] if 0 <= pivot.idx < len(df.index) else None
        rows.append(
            {
                "df_index": int(pivot.idx),
                "date": dt.strftime("%Y-%m-%d %H:%M") if hasattr(dt, "strftime") else str(dt),
                "price": float(pivot.price),
                "kind": pivot.kind,
            }
        )
    return pd.DataFrame(rows)


def build_elliott_chart(
    df: pd.DataFrame,
    pivots: list[Pivot],
    best: ImpulseCandidate | None,
    levels: pd.DataFrame,
    abc: dict | None,
    symbol: str,
    zigzag_pct: float,
    plot_last_n: int,
    show_levels: bool,
) -> go.Figure:
    tail = df if plot_last_n <= 0 else df.iloc[-plot_last_n:]
    start_pos = len(df) - len(tail)
    end_pos = len(df) - 1

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tail.index,
            y=tail["Close_PLOT"],
            mode="lines",
            name="Close",
            line=dict(color="#93c5fd", width=1.6),
        )
    )

    visible_pivots = [pivot for pivot in pivots if start_pos <= pivot.idx <= end_pos]
    if visible_pivots:
        fig.add_trace(
            go.Scatter(
                x=[df.index[pivot.idx] for pivot in visible_pivots],
                y=[pivot.price for pivot in visible_pivots],
                mode="markers+text",
                name="ZigZag Pivots",
                text=[pivot.kind for pivot in visible_pivots],
                textposition="top center",
                marker=dict(size=8, color="#facc15", line=dict(color="#0d1b2e", width=1)),
            )
        )

    if best is not None:
        seq = [pivots[i] for i in best.pivot_idxs]
        labels = ["0", "1", "2", "3", "4", "5"]
        for i in range(5):
            a, b = seq[i], seq[i + 1]
            fig.add_trace(
                go.Scatter(
                    x=[df.index[a.idx], df.index[b.idx]],
                    y=[a.price, b.price],
                    mode="lines",
                    name=f"Wave {i + 1}",
                    line=dict(color="#22c55e" if best.direction == "bull" else "#ef4444", width=2.4),
                    showlegend=i == 0,
                    legendgroup="impulse",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[df.index[pivot.idx] for pivot in seq],
                y=[pivot.price for pivot in seq],
                mode="text",
                name="Wave Labels",
                text=labels,
                textposition="bottom center",
                textfont=dict(size=15, color="#ffffff"),
                showlegend=False,
            )
        )

    if show_levels and not levels.empty:
        for _, row in levels.iterrows():
            color = "#64748b" if row["Group"] != "ABC" else "#f97316"
            dash = "dash" if row["Group"] != "ABC" else "dot"
            fig.add_hline(
                y=row["Price"],
                line_width=1,
                line_dash=dash,
                line_color=color,
                annotation_text=f"{row['Group']} {row['Level']}",
                annotation_position="right",
                annotation_font_size=10,
                annotation_font_color="#cbd5e1",
            )

    if abc is not None:
        for label in ("A", "B"):
            a, b = abc[label]
            fig.add_trace(
                go.Scatter(
                    x=[df.index[a.idx], df.index[b.idx]],
                    y=[a.price, b.price],
                    mode="lines",
                    name=f"ABC {label}",
                    line=dict(color="#f97316", width=1.8, dash="dot"),
                )
            )

    title = f"{symbol} | ZigZag {zigzag_pct:.1f}%"
    if best is not None:
        title += f" | Impulse {best.direction.upper()} | Score {best.score:.2f}"
    else:
        title += " | No impulse detected"

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#0d1b2e",
        plot_bgcolor="#0d1b2e",
        height=660,
        margin=dict(l=24, r=24, t=84, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)")
    return fig


class ElliottWaveAnalysisModule(FazDaneModule):
    MODULE_NAME = "Elliott Wave Analysis"
    MODULE_ICON = "Wave"
    MODULE_DESCRIPTION = "ZigZag pivot detection, 5-wave impulse scoring, Fibonacci levels, and ABC projection"
    TIER = 3
    SOURCE_NOTEBOOK = "Forecasting/Cycle Analysis/Elliott Wave Analysis"
    CACHE_TTL = 3600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def render_sidebar(self):
        self._default_universe()
        st.markdown("**Elliott Universe**")
        self.universe_name, tickers, _ = render_universe_manager(
            key_prefix="elliott",
            show_benchmark=False,
            label="Ticker Universe:",
        )
        ticker_names = get_ticker_names(self.universe_name)
        if "^GSPC" not in tickers:
            tickers = ["^GSPC"] + tickers
            ticker_names.setdefault("^GSPC", "S&P 500 Index")
        self.tickers = tickers
        default_idx = self.tickers.index("^GSPC") if "^GSPC" in self.tickers else 0
        if st.session_state.get("elliott_ticker") not in self.tickers:
            st.session_state["elliott_ticker"] = self.tickers[default_idx]
        self.ticker = st.selectbox(
            "Ticker / Index:",
            self.tickers,
            index=default_idx,
            key="elliott_ticker",
            format_func=lambda ticker: format_ticker_display(ticker, ticker_names),
        )

        st.markdown("**Data Window**")
        self.period = st.selectbox("Period:", ["6mo", "1y", "2y", "5y", "10y", "max"], index=2, key="elliott_period")
        self.interval = st.selectbox("Interval:", ["1d", "1h", "15m"], index=0, key="elliott_interval")

        st.markdown("**Wave Detection**")
        self.zigzag_pct = float(st.slider("ZigZag Reversal %:", 1.0, 12.0, 5.0, step=0.5, key="elliott_zigzag"))
        self.max_windows = int(st.slider("Max Pivot Windows:", 50, 1000, 500, step=50, key="elliott_windows"))
        self.plot_last_n = int(st.slider("Plot Last N Bars:", 100, 1500, 600, step=50, key="elliott_plot_n"))
        self.show_levels = st.checkbox("Show Fibonacci levels", value=True, key="elliott_levels")
        self.show_abc = st.checkbox("Show ABC after impulse", value=True, key="elliott_abc")

        if st.button("Refresh Elliott Wave", use_container_width=True, type="primary", key="elliott_refresh"):
            fetch_elliott_data.clear()
            st.rerun()

    def render_main(self):
        self.render_section_header(
            "Elliott Wave Analysis",
            "Rules-based 5-wave impulse scan from ZigZag pivots with Fibonacci retracements and extensions",
        )

        symbol = normalize_symbol(self.ticker)
        with st.spinner(f"Fetching {symbol} and scanning Elliott Wave pivots..."):
            df = fetch_elliott_data(symbol, self.period, self.interval)

        if df.empty:
            st.warning(f"No data returned for {symbol}. Try another ticker, period, or interval.")
            return
        if len(df) < 30:
            st.warning("Not enough bars returned for Elliott Wave analysis.")
            return

        pivots = zigzag_percent(df["Close_PLOT"], self.zigzag_pct)
        best = find_best_impulse(pivots, self.max_windows)
        levels, abc = build_levels(pivots, best, self.show_abc)
        pivots_df = pivots_to_frame(df, pivots)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", symbol)
        c2.metric("Bars", f"{len(df):,}")
        c3.metric("ZigZag Pivots", f"{len(pivots):,}")
        c4.metric("Impulse", best.direction.upper() if best else "None", f"{best.score:.2f}" if best else None)

        st.plotly_chart(
            build_elliott_chart(
                df=df,
                pivots=pivots,
                best=best,
                levels=levels,
                abc=abc,
                symbol=symbol,
                zigzag_pct=self.zigzag_pct,
                plot_last_n=self.plot_last_n,
                show_levels=self.show_levels,
            ),
            use_container_width=True,
        )

        tab_summary, tab_pivots, tab_levels, tab_exports = st.tabs(["Summary", "Pivots", "Levels", "Exports"])

        with tab_summary:
            if best is None:
                st.info("No clean 5-wave impulse was detected with the current ZigZag and timeframe settings.")
            else:
                seq = [pivots[i] for i in best.pivot_idxs]
                waves = [abs(seq[i + 1].price - seq[i].price) for i in range(5)]
                summary = pd.DataFrame(
                    {
                        "Wave": ["1", "2", "3", "4", "5"],
                        "Start": [df.index[seq[i].idx].strftime("%Y-%m-%d") for i in range(5)],
                        "End": [df.index[seq[i + 1].idx].strftime("%Y-%m-%d") for i in range(5)],
                        "Length": waves,
                    }
                )
                st.dataframe(summary.round(2), use_container_width=True, hide_index=True)
                col_a, col_b = st.columns(2)
                col_a.metric("Wave 3 / Wave 1", f"{waves[2] / max(waves[0], 1e-9):.3f}")
                col_b.metric("Wave 5 / Wave 3", f"{waves[4] / max(waves[2], 1e-9):.3f}")

        with tab_pivots:
            st.dataframe(pivots_df.round(2), use_container_width=True, hide_index=True)

        with tab_levels:
            if levels.empty:
                st.info("No Fibonacci levels are available until an impulse is detected.")
            else:
                st.dataframe(levels.round(2), use_container_width=True, hide_index=True)

        with tab_exports:
            run_log = self._run_log(symbol, df, pivots, best)
            st.download_button(
                "Download Pivots CSV",
                data=pivots_df.to_csv(index=False),
                file_name=f"elliott_pivots_{symbol.replace('^', '').lower()}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download Wave Levels CSV",
                data=levels.to_csv(index=False),
                file_name=f"elliott_levels_{symbol.replace('^', '').lower()}.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=levels.empty,
            )
            st.download_button(
                "Download Process Log",
                data=run_log,
                file_name=f"elliott_process_log_{symbol.replace('^', '').lower()}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    def _default_universe(self):
        key = "elliott_sel"
        target = "Index Universe"
        if target in get_universe_names() and key not in st.session_state:
            st.session_state[key] = target

    def _run_log(self, symbol: str, df: pd.DataFrame, pivots: list[Pivot], best: ImpulseCandidate | None) -> str:
        start = df.index.min().strftime("%Y-%m-%d")
        end = df.index.max().strftime("%Y-%m-%d")
        lines = [
            "Elliott Wave run log",
            "--------------------",
            f"Ticker  : {symbol}",
            f"Period  : {self.period}",
            f"Interval: {self.interval}",
            f"ZigZag% : {self.zigzag_pct}",
            f"Range   : {start} to {end}",
            f"Bars    : {len(df)}",
            f"Pivots  : {len(pivots)}",
        ]
        if best is None:
            lines.append("Impulse : None detected")
        else:
            lines.extend(
                [
                    f"Impulse : {best.direction.upper()}",
                    f"Score   : {best.score:.2f}",
                    f"Pivots  : {best.pivot_idxs}",
                ]
            )
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)
