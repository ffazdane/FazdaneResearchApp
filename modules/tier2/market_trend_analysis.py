"""
FazDane Analytics — Tier 2
Market Trend Analysis — MA + FazDane Cloud Chart Module

Displays:
  - Candlestick price chart with SMA / EMA overlays
  - FazDane Cloud (Ichimoku Span A / Span B with shaded fill)
  - VWAP line
  - KPI deck: deviation from every selected MA, cloud levels, VWAP, trend stack
  - Universe scanner table with score + trade interpretation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import sqlite3
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from modules.base_module import FazDaneModule
from utils.universe_manager import (
    render_universe_manager,
    load_universes,
    get_universe_names,
    format_ticker_display,
    get_company_name,
)
from utils.persistence import get_db_path, backup_database

logger = logging.getLogger("MarketTrendAnalysis")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ALL_MA_PERIODS = [9, 14, 20, 50, 100, 150, 200]
DEFAULT_MA_PERIODS = [9, 14, 20, 50, 200]
DEFAULT_MA_TYPE = "EMA"
DEFAULT_TIMEFRAME = "3M"
DEFAULT_AGGREGATION = "Daily"

NEAR_THRESHOLD_PCT = 0.50  # percent band for "Near" status

# Colour for each MA period (consistent across runs)
MA_COLORS = {
    9:   "#06b6d4",   # cyan
    14:  "#3b82f6",   # blue
    20:  "#10b981",   # emerald
    50:  "#f59e0b",   # amber
    100: "#8b5cf6",   # violet
    150: "#ec4899",   # pink
    200: "#ef4444",   # red
}

SPAN_A_COLOR = "rgba(16, 185, 129, 0.80)"    # green-teal line
SPAN_B_COLOR = "rgba(239, 68, 68, 0.80)"     # red line
CLOUD_BULL_FILL = "rgba(16, 185, 129, 0.15)"
CLOUD_BEAR_FILL = "rgba(239, 68, 68, 0.15)"
VWAP_COLOR = "#a855f7"                         # purple
PRICE_LINE_COLOR = "#e2e8f0"

STATUS_COLORS = {
    "Above":        "#22c55e",
    "Near":         "#eab308",
    "Below":        "#ef4444",
    "Above Cloud":  "#22c55e",
    "Inside Cloud": "#eab308",
    "Below Cloud":  "#ef4444",
    "Bullish Cloud":"#22c55e",
    "Bearish Cloud":"#ef4444",
    "Above VWAP":   "#22c55e",
    "Below VWAP":   "#ef4444",
    "🟢 Buy":       "#22c55e",
    "🔴 Sell":      "#ef4444",
    "⚪ No Trade":  "#94a3b8",
}

# Timeframe UI label → (yf period, default yf interval)
TIMEFRAME_MAP = {
    # label: (download_period, default_interval, display_days)
    "1D":  ("5d",   "5m",  1),
    "5D":  ("30d",  "15m", 5),
    "1M":  ("3mo",  "1h",  30),
    "3M":  ("1y",   "1d",  90),
    "6M":  ("2y",   "1d",  180),
    "1Y":  ("3y",   "1d",  365),
    "2Y":  ("5y",   "1wk", 730),
    "5Y":  ("10y",  "1wk", 1825),
}

# Aggregation UI label → yf interval override (where feasible)
AGGREGATION_MAP = {
    "Daily":  "1d",
    "Hourly": "1h",
    "Weekly": "1wk",
}

# ─────────────────────────────────────────────────────────────────────────────
# PURE CALCULATION FUNCTIONS  (no Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_moving_average(df: pd.DataFrame, period: int, ma_type: str) -> pd.Series:
    """Return SMA or EMA series for the given period."""
    if ma_type.upper() == "SMA":
        return df["close"].rolling(window=period).mean()
    elif ma_type.upper() == "EMA":
        return df["close"].ewm(span=period, adjust=False).mean()
    raise ValueError(f"Invalid MA type: {ma_type!r}. Use 'SMA' or 'EMA'.")


def calculate_fazdane_cloud(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
) -> pd.DataFrame:
    """
    Compute FazDane Cloud (Ichimoku-style) columns in-place:
        tenkan, kijun, span_a, span_b, chikou
    """
    df = df.copy()

    df["tenkan"] = (
        df["high"].rolling(window=tenkan_period).max()
        + df["low"].rolling(window=tenkan_period).min()
    ) / 2

    df["kijun"] = (
        df["high"].rolling(window=kijun_period).max()
        + df["low"].rolling(window=kijun_period).min()
    ) / 2

    df["span_a"] = ((df["tenkan"] + df["kijun"]) / 2).shift(kijun_period)

    df["span_b"] = (
        df["high"].rolling(window=2 * kijun_period).max()
        + df["low"].rolling(window=2 * kijun_period).min()
    ) / 2
    df["span_b"] = df["span_b"].shift(kijun_period)

    # Chikou (lagging span) — stored for future confirmation logic
    df["chikou"] = df["close"].shift(-kijun_period)

    return df


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative VWAP approximation (typical-price × volume).
    For daily charts this is a multi-bar running VWAP.
    """
    df = df.copy()
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].cumsum()
    return df


def calculate_deviation(
    current_price: float,
    reference_value: float,
    near_threshold: float = NEAR_THRESHOLD_PCT,
) -> dict:
    """
    Return a deviation dict matching the FazDane ThinkScript formula:
        percent = ((price - reference) / price) * 100
    """
    if reference_value == 0 or np.isnan(reference_value):
        return {
            "value": None,
            "deviation_points": None,
            "deviation_percent": None,
            "status": "N/A",
        }

    points = current_price - reference_value
    percent = ((current_price - reference_value) / current_price) * 100

    if percent > near_threshold:
        status = "Above"
    elif percent < -near_threshold:
        status = "Below"
    else:
        status = "Near"

    return {
        "value": round(reference_value, 2),
        "deviation_points": round(points, 2),
        "deviation_percent": round(percent, 2),
        "status": status,
    }


def calculate_trend_stack(
    latest: pd.Series,
    ma_type: str,
    selected_periods: list[int],
) -> dict:
    """Determine MA stack status and details string."""
    ma_type = ma_type.upper()

    def _get(period: int) -> Optional[float]:
        key = f"{ma_type}_{period}"
        val = latest.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)

    price = float(latest["close"])
    ma9   = _get(9)
    ma14  = _get(14)
    ma20  = _get(20)
    ma50  = _get(50)
    ma200 = _get(200)

    # Strong bullish: price > 9 > 14 > 20 > 50 > 200
    if all(v is not None for v in [ma9, ma14, ma20, ma50, ma200]):
        if price > ma9 > ma14 > ma20 > ma50 > ma200:
            return {
                "status": "Strong Bullish Stack",
                "details": (
                    f"Price > 9 {ma_type} > 14 {ma_type} > "
                    f"20 {ma_type} > 50 {ma_type} > 200 {ma_type}"
                ),
            }
        if price < ma9 < ma14 < ma20 < ma50 < ma200:
            return {
                "status": "Strong Bearish Stack",
                "details": (
                    f"Price < 9 {ma_type} < 14 {ma_type} < "
                    f"20 {ma_type} < 50 {ma_type} < 200 {ma_type}"
                ),
            }

    # Moderate bullish: price > 20 > 50 > 200
    if all(v is not None for v in [ma20, ma50, ma200]):
        if price > ma20 and ma20 > ma50 and ma50 > ma200:
            return {
                "status": "Bullish Trend",
                "details": f"Price > 20 {ma_type} > 50 {ma_type} > 200 {ma_type}",
            }
        if price < ma20 and ma20 < ma50 and ma50 < ma200:
            return {
                "status": "Bearish Trend",
                "details": f"Price < 20 {ma_type} < 50 {ma_type} < 200 {ma_type}",
            }

    return {
        "status": "Mixed / Sideways",
        "details": "Moving averages are not cleanly stacked",
    }


def calculate_scanner_score(kpis: dict) -> int:
    """
    Score 0-100 based on MA/cloud alignment.
    """
    score = 0
    ma_list = kpis.get("moving_averages", [])
    ma_by_period = {m["period"]: m for m in ma_list}

    if ma_by_period.get(20, {}).get("status") == "Above":
        score += 10
    elif ma_by_period.get(20, {}).get("status") == "Near":
        score += 5

    if ma_by_period.get(50, {}).get("status") == "Above":
        score += 15

    if ma_by_period.get(200, {}).get("status") == "Above":
        score += 20

    cloud = kpis.get("cloud", {})
    if cloud.get("cloud_status") == "Above Cloud":
        score += 20
    if cloud.get("cloud_trend") == "Bullish Cloud":
        score += 10

    vwap = kpis.get("vwap", {})
    if vwap.get("status") == "Above VWAP":
        score += 10

    stack = kpis.get("trend_stack", {})
    if stack.get("status") == "Strong Bullish Stack":
        score += 15

    return min(score, 100)


def get_trade_interpretation(kpis: dict) -> str:
    """Return a plain-text trade interpretation."""
    cloud = kpis.get("cloud", {})
    ma_list = kpis.get("moving_averages", [])
    ma_by_period = {m["period"]: m for m in ma_list}
    stack = kpis.get("trend_stack", {}).get("status", "")

    above_20  = ma_by_period.get(20,  {}).get("status") == "Above"
    near_20   = ma_by_period.get(20,  {}).get("status") == "Near"
    above_50  = ma_by_period.get(50,  {}).get("status") == "Above"
    above_200 = ma_by_period.get(200, {}).get("status") == "Above"
    cloud_status = cloud.get("cloud_status", "")
    cloud_trend  = cloud.get("cloud_trend", "")

    # Bullish continuation
    if above_20 and above_50 and above_200 and cloud_status == "Above Cloud" and cloud_trend == "Bullish Cloud":
        return "Bullish continuation setup. Suitable for bullish calendar, call diagonal, or trend-following debit spread."

    # Pullback buy zone
    if above_50 and near_20 and cloud_status == "Above Cloud" and cloud_trend == "Bullish Cloud":
        return "Pullback into short-term support. Watch for bullish reversal candle before entry."

    # Sideways / Neutral
    if cloud_status == "Inside Cloud" or "Mixed" in stack or "Sideways" in stack:
        return "Sideways structure. Better suited for iron condor, short premium, or neutral calendar."

    # Bearish risk
    if not above_50 and cloud_status == "Below Cloud" and cloud_trend == "Bearish Cloud":
        return "Bearish structure. Avoid bullish calendar. Consider bearish put spread, bear call spread, or wait."

    return "Mixed signal. Evaluate on multiple timeframes before deploying."


def calculate_ma_cloud_indicators(
    df: pd.DataFrame,
    ma_type: str = "EMA",
    ma_periods: list[int] = None,
    tenkan_period: int = 9,
    kijun_period: int = 26,
) -> dict:
    """
    Master calculation function.  Returns:
        {
            "dataframe": df_with_all_columns,
            "kpis": { current_price, ma_type, moving_averages, cloud, vwap, trend_stack }
        }
    """
    if ma_periods is None:
        ma_periods = DEFAULT_MA_PERIODS

    df = df.copy()
    ma_type = ma_type.upper()

    for period in ma_periods:
        df[f"{ma_type}_{period}"] = calculate_moving_average(df, period, ma_type)

    df = calculate_fazdane_cloud(df, tenkan_period, kijun_period)
    df = calculate_vwap(df)

    # Use the last row that has at least span_a and span_b non-NaN
    valid_rows = df.dropna(subset=["span_a", "span_b"])
    if valid_rows.empty:
        # Fall back to the last non-NaN close row
        valid_rows = df.dropna(subset=["close"])
    if valid_rows.empty:
        return {"dataframe": df, "kpis": {}}

    latest = valid_rows.iloc[-1]
    current_price = float(latest["close"])

    # MA KPIs
    ma_kpis = []
    for period in ma_periods:
        col = f"{ma_type}_{period}"
        ma_val = latest.get(col, float("nan"))
        if isinstance(ma_val, float) and not np.isnan(ma_val):
            dev = calculate_deviation(current_price, float(ma_val))
        else:
            dev = {"value": None, "deviation_points": None, "deviation_percent": None, "status": "N/A"}
        ma_kpis.append({"period": period, "type": ma_type, **dev})

    # Cloud KPIs
    span_a = float(latest.get("span_a", float("nan")))
    span_b = float(latest.get("span_b", float("nan")))
    cloud_payload = {}
    if not np.isnan(span_a) and not np.isnan(span_b):
        cloud_top    = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)

        if current_price > cloud_top:
            cloud_status = "Above Cloud"
        elif current_price < cloud_bottom:
            cloud_status = "Below Cloud"
        else:
            cloud_status = "Inside Cloud"

        cloud_trend = "Bullish Cloud" if span_a > span_b else "Bearish Cloud"

        span_a_dev    = calculate_deviation(current_price, span_a)
        cloud_top_dev = calculate_deviation(current_price, cloud_top)

        cloud_payload = {
            "span_a": round(span_a, 2),
            "span_b": round(span_b, 2),
            "cloud_top": round(cloud_top, 2),
            "cloud_bottom": round(cloud_bottom, 2),
            "deviation_from_span_a_points":   span_a_dev["deviation_points"],
            "deviation_from_span_a_percent":  span_a_dev["deviation_percent"],
            "deviation_from_cloud_top_points":  cloud_top_dev["deviation_points"],
            "deviation_from_cloud_top_percent": cloud_top_dev["deviation_percent"],
            "cloud_status": cloud_status,
            "cloud_trend":  cloud_trend,
        }

    # VWAP KPIs
    vwap_val = float(latest.get("vwap", float("nan")))
    vwap_payload = {}
    if not np.isnan(vwap_val):
        vwap_dev = calculate_deviation(current_price, vwap_val)
        vwap_payload = {
            "value": round(vwap_val, 2),
            "deviation_points":  vwap_dev["deviation_points"],
            "deviation_percent": vwap_dev["deviation_percent"],
            "status": "Above VWAP" if current_price > vwap_val else "Below VWAP",
        }

    trend_stack = calculate_trend_stack(latest, ma_type, ma_periods)

    kpis = {
        "current_price": round(current_price, 2),
        "ma_type": ma_type,
        "moving_averages": ma_kpis,
        "cloud": cloud_payload,
        "vwap": vwap_payload,
        "trend_stack": trend_stack,
    }

    return {"dataframe": df, "kpis": kpis}


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_interval(timeframe: str, aggregation: str) -> str:
    """Pick the best interval given timeframe + aggregation override."""
    _, default_interval, _ = TIMEFRAME_MAP.get(timeframe, ("3mo", "1d", 90))
    agg_interval = AGGREGATION_MAP.get(aggregation, "1d")

    # Guard: hourly only supported up to 60 days by yfinance
    if agg_interval == "1h" and timeframe in ("2Y", "5Y"):
        return default_interval
    return agg_interval


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(ticker: str, timeframe: str, aggregation: str) -> pd.DataFrame:
    """
    Download OHLCV for a single ticker and normalise column names to lower-case.
    Cached for 15 minutes.

    BUG FIX: reset_index() must run BEFORE the lowercase pass so that the
    'Date' / 'Datetime' index column (added by reset_index) also gets
    lowercased — otherwise the date-column lookup silently fails.
    """
    period, _, _ = TIMEFRAME_MAP.get(timeframe, ("3mo", "1d", 90))
    interval  = _resolve_interval(timeframe, aggregation)

    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            logger.warning(f"yfinance returned empty DataFrame for {ticker} ({period}/{interval})")
            return pd.DataFrame()

        # Flatten multi-index if present (newer yfinance always returns MultiIndex)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # ── IMPORTANT: reset_index FIRST, THEN lowercase ALL columns ──
        # reset_index moves Date/Datetime from index → column with capital D.
        # Lowercasing after ensures it becomes 'date'/'datetime' consistently.
        raw = raw.reset_index()
        raw.columns = [str(c).lower() for c in raw.columns]

        # Normalise the date column name to 'date'
        if "datetime" in raw.columns:
            raw = raw.rename(columns={"datetime": "date"})
        elif "date" not in raw.columns:
            # Fallback: use first column as date
            raw = raw.rename(columns={raw.columns[0]: "date"})

        raw["date"] = pd.to_datetime(raw["date"])
        # Strip timezone so Plotly renders consistently
        if hasattr(raw["date"].dt, "tz") and raw["date"].dt.tz is not None:
            raw["date"] = raw["date"].dt.tz_localize(None)

        raw = raw.dropna(subset=["close"])
        return raw

    except Exception as e:
        logger.error(f"yfinance fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def _normalise_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Shared helper: reset_index first, then lowercase all columns including date."""
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})
    elif "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    if hasattr(df["date"].dt, "tz") and df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    return df.dropna(subset=["close"])


@st.cache_data(ttl=900, show_spinner=False)
def fetch_universe_ohlcv(
    tickers: tuple,      # tuple for hashability
    timeframe: str,
    aggregation: str,
) -> dict:
    """
    Batch-download OHLCV for all universe tickers and return a dict of DataFrames.
    Cached for 15 minutes.
    """
    period, _, _ = TIMEFRAME_MAP.get(timeframe, ("3mo", "1d", 90))
    interval  = _resolve_interval(timeframe, aggregation)

    result: dict[str, pd.DataFrame] = {}

    try:
        raw = yf.download(
            list(tickers),
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
        if raw.empty:
            return result

        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in tickers:
                try:
                    df = raw[ticker].copy()
                    df = _normalise_ohlcv_df(df)
                    if not df.empty:
                        result[ticker] = df
                except Exception as e:
                    logger.warning(f"Could not extract {ticker} from batch download: {e}")
        else:
            # Single-ticker batch — treat the whole frame as one
            df = _normalise_ohlcv_df(raw.copy())
            if not df.empty and tickers:
                result[tickers[0]] = df

    except Exception as e:
        logger.error(f"Batch yfinance fetch failed: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_snapshot_schema(db_path: Path) -> None:
    """Create the ticker_ma_cloud_snapshot table if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ticker_ma_cloud_snapshot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                snapshot_date DATE NOT NULL,
                current_price REAL,
                ma_type TEXT,
                ma_period INTEGER,
                ma_value REAL,
                deviation_points REAL,
                deviation_percent REAL,
                ma_status TEXT,
                span_a REAL,
                span_b REAL,
                cloud_top REAL,
                cloud_bottom REAL,
                cloud_status TEXT,
                cloud_trend TEXT,
                vwap REAL,
                vwap_status TEXT,
                trend_stack_status TEXT,
                scanner_score INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_macs_ticker_date
            ON ticker_ma_cloud_snapshot(ticker, snapshot_date)
        """)


def save_scanner_snapshot(scan_results: list[dict], ma_type: str) -> None:
    """Persist today's scanner results to SQLite for backtesting."""
    try:
        db_path = get_db_path("market_trend_analysis")
        _ensure_snapshot_schema(db_path)
        today = datetime.now().date().isoformat()

        records = []
        for row in scan_results:
            ticker = row.get("ticker", "")
            kpis   = row.get("kpis", {})
            price  = kpis.get("current_price")
            cloud  = kpis.get("cloud", {})
            vwap   = kpis.get("vwap", {})
            stack  = kpis.get("trend_stack", {})
            score  = row.get("scanner_score")

            for ma in kpis.get("moving_averages", []):
                records.append((
                    ticker, today, price, ma_type,
                    ma.get("period"), ma.get("value"),
                    ma.get("deviation_points"), ma.get("deviation_percent"),
                    ma.get("status"),
                    cloud.get("span_a"), cloud.get("span_b"),
                    cloud.get("cloud_top"), cloud.get("cloud_bottom"),
                    cloud.get("cloud_status"), cloud.get("cloud_trend"),
                    vwap.get("value"), vwap.get("status"),
                    stack.get("status"),
                    score,
                ))

        with sqlite3.connect(db_path) as conn:
            conn.executemany("""
                INSERT INTO ticker_ma_cloud_snapshot (
                    ticker, snapshot_date, current_price, ma_type,
                    ma_period, ma_value, deviation_points, deviation_percent, ma_status,
                    span_a, span_b, cloud_top, cloud_bottom, cloud_status, cloud_trend,
                    vwap, vwap_status, trend_stack_status, scanner_score
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, records)

        backup_database("market_trend_analysis", reason="MA Cloud Scanner")
    except Exception as e:
        logger.error(f"Failed to save scanner snapshot: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _status_dot(status: str) -> str:
    color = STATUS_COLORS.get(status, "#94a3b8")
    return f"<span style='color:{color};'>●</span>"


def _render_kpi_card(
    title: str,
    value_line: str,
    sub_line: str,
    status: str,
    col,
) -> None:
    color = STATUS_COLORS.get(status, "#94a3b8")
    with col:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(21,40,71,0.85) 0%, rgba(13,27,46,0.95) 100%);
                border: 1px solid #1e3a5f;
                border-top: 3px solid {color};
                border-radius: 10px;
                padding: 14px 16px;
                margin-bottom: 8px;
                min-height: 110px;
            ">
                <div style="color:#94a3b8;font-size:11px;font-weight:700;
                            text-transform:uppercase;letter-spacing:1px;
                            margin-bottom:6px;">{title}</div>
                <div style="color:#f8fafc;font-size:17px;font-weight:700;
                            margin-bottom:4px;">{value_line}</div>
                <div style="color:#94a3b8;font-size:12px;margin-bottom:6px;">{sub_line}</div>
                <div style="font-size:12px;font-weight:600;color:{color};">
                    {_status_dot(status)} {status}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_wide_card(title: str, value: str, sub: str, color: str, col) -> None:
    with col:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(21,40,71,0.85) 0%, rgba(13,27,46,0.95) 100%);
                border: 1px solid #1e3a5f;
                border-left: 4px solid {color};
                border-radius: 10px;
                padding: 14px 18px;
                margin-bottom: 8px;
            ">
                <div style="color:#94a3b8;font-size:11px;font-weight:700;
                            text-transform:uppercase;letter-spacing:1px;
                            margin-bottom:5px;">{title}</div>
                <div style="color:{color};font-size:18px;font-weight:700;
                            margin-bottom:3px;">{value}</div>
                <div style="color:#64748b;font-size:12px;">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _fmt_dev(points, pct) -> str:
    if points is None:
        return "N/A"
    sign = "+" if points >= 0 else ""
    return f"{sign}{points:.2f} pts  ({sign}{pct:.2f}%)"


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_candlestick_chart(
    df: pd.DataFrame,
    kpis: dict,
    ma_type: str,
    selected_periods: list[int],
    show_vwap: bool,
    show_cloud: bool,
    show_current_price: bool,
    plotly_template: str = "plotly_dark",
    bg_color: str = "#0d1b2e",
    ticker: str = "",
    company_name: str = "",
) -> go.Figure:
    """Build and return the full Plotly candlestick figure."""

    fig = go.Figure()

    # ── Candlestick ───────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price",
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e",
        decreasing_fillcolor="#ef4444",
        line_width=1,
    ))

    # ── FazDane Cloud ─────────────────────────────────────────────────
    if show_cloud and "span_a" in df.columns and "span_b" in df.columns:
        cloud = kpis.get("cloud", {})
        cloud_trend = cloud.get("cloud_trend", "Bullish Cloud")
        fill_color = CLOUD_BULL_FILL if cloud_trend == "Bullish Cloud" else CLOUD_BEAR_FILL

        fig.add_trace(go.Scatter(
            x=df["date"], y=df["span_a"],
            mode="lines",
            name="FazDane Cloud Span A",
            line=dict(color=SPAN_A_COLOR, width=1.5),
            hovertemplate="Span A: %{y:.2f}<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=df["date"], y=df["span_b"],
            mode="lines",
            name="FazDane Cloud Span B",
            line=dict(color=SPAN_B_COLOR, width=1.5),
            fill="tonexty",
            fillcolor=fill_color,
            hovertemplate="Span B: %{y:.2f}<extra></extra>",
        ))

    # ── Moving Averages ───────────────────────────────────────────────
    for period in selected_periods:
        col = f"{ma_type}_{period}"
        if col in df.columns:
            color = MA_COLORS.get(period, "#94a3b8")
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                mode="lines",
                name=f"{period} {ma_type}",
                line=dict(color=color, width=1.8),
                hovertemplate=f"{period} {ma_type}: %{{y:.2f}}<extra></extra>",
            ))

    # ── VWAP ──────────────────────────────────────────────────────────
    if show_vwap and "vwap" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["vwap"],
            mode="lines",
            name="VWAP",
            line=dict(color=VWAP_COLOR, width=1.5, dash="dot"),
            hovertemplate="VWAP: %{y:.2f}<extra></extra>",
        ))

    # ── Current Price Line ────────────────────────────────────────────
    if show_current_price and kpis:
        price = kpis.get("current_price")
        if price:
            fig.add_hline(
                y=price,
                line_dash="dash",
                line_color=PRICE_LINE_COLOR,
                line_width=1,
                annotation_text=f"  ${price:.2f}",
                annotation_font_color=PRICE_LINE_COLOR,
                annotation_font_size=11,
            )

    # ── Layout ────────────────────────────────────────────────────────
    title_text = ""
    if ticker:
        title_text = f"📊 {ticker}"
        if company_name:
            title_text += f" : {company_name}"

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(family="Inter, sans-serif", size=16, color="#f8fafc"),
            x=0,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ) if title_text else None,
        template=plotly_template,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        xaxis=dict(
            gridcolor="#1e3a5f",
            showgrid=True,
            zeroline=False,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            gridcolor="#1e3a5f",
            showgrid=True,
            zeroline=False,
            side="right",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        height=560,
        hovermode="x unified",
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MODULE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class MarketTrendAnalysisModule(FazDaneModule):
    MODULE_NAME        = "Market Trend Analysis"
    MODULE_ICON        = "📊"
    MODULE_DESCRIPTION = (
        "Candlestick chart with SMA/EMA overlays, FazDane Cloud, VWAP, "
        "KPI deviation deck, and universe-level MA/Cloud scanner."
    )
    TIER               = 2
    SOURCE_NOTEBOOK    = "FazDane MA + Cloud Command Center"
    CACHE_TTL          = 900
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES       = ["yfinance", "market_trend_analysis_sqlite"]

    # ── Sidebar ───────────────────────────────────────────────────────

    def render_sidebar(self) -> None:
        st.markdown("**Universe**")
        universe_name, tickers_list, _ = render_universe_manager(
            key_prefix="mta",
            label="Ticker Universe:",
        )
        st.session_state["mta_universe_name"] = universe_name
        st.session_state["mta_tickers"] = list(tickers_list)

        st.markdown("**Ticker**")
        tickers = st.session_state.get("mta_tickers", ["SPY"])
        if not tickers:
            tickers = ["SPY"]
        # Resolve index so a scanner click-to-load is reflected in the dropdown
        _ticker_default = st.session_state.get("mta_selected_ticker", tickers[0])
        _ticker_idx = tickers.index(_ticker_default) if _ticker_default in tickers else 0
        selected_ticker = st.selectbox(
            "Ticker:",
            options=tickers,
            index=_ticker_idx,
            key="mta_ticker",
        )
        # mta_selected_ticker is NOT a widget key — safe to write
        st.session_state["mta_selected_ticker"] = selected_ticker

        st.markdown("**Chart Controls**")
        # Note: do NOT write back to session_state keys that match widget key=
        # Streamlit owns those automatically after the widget is rendered.
        st.selectbox(
            "Moving Average Type:",
            options=["EMA", "SMA"],
            index=0,
            key="mta_ma_type",
        )

        selected_periods = st.multiselect(
            "MA Periods:",
            options=ALL_MA_PERIODS,
            default=DEFAULT_MA_PERIODS,
            key="mta_periods",
        )
        if not selected_periods:
            # If user deselects all, restore default into session state
            st.session_state["mta_periods"] = DEFAULT_MA_PERIODS

        st.selectbox(
            "Timeframe:",
            options=list(TIMEFRAME_MAP.keys()),
            index=list(TIMEFRAME_MAP.keys()).index(DEFAULT_TIMEFRAME),
            key="mta_timeframe",
        )

        st.selectbox(
            "Aggregation:",
            options=list(AGGREGATION_MAP.keys()),
            index=0,
            key="mta_aggregation",
        )

        st.markdown("**Display**")
        st.checkbox("FazDane Cloud", value=True, key="mta_show_cloud")
        st.checkbox("VWAP", value=True, key="mta_show_vwap")
        st.checkbox("Current Price Line", value=True, key="mta_show_price_line")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("📊 Run Universe Scanner", use_container_width=True, type="primary", key="mta_scan_btn"):
            st.session_state["mta_trigger_scan"] = True

    # ── Main ──────────────────────────────────────────────────────────

    def render_main(self) -> None:
        ticker     = st.session_state.get("mta_selected_ticker", "SPY")
        ma_type    = st.session_state.get("mta_ma_type", "EMA")
        periods    = st.session_state.get("mta_periods", DEFAULT_MA_PERIODS)
        timeframe  = st.session_state.get("mta_timeframe", "3M")
        aggregation= st.session_state.get("mta_aggregation", "Daily")
        show_cloud = st.session_state.get("mta_show_cloud", True)
        show_vwap  = st.session_state.get("mta_show_vwap", True)
        show_price = st.session_state.get("mta_show_price_line", True)
        universe_name = st.session_state.get("mta_universe_name", "")
        all_tickers   = st.session_state.get("mta_tickers", [ticker])
        theme  = st.session_state.get("theme_colors", {})
        bg_col = theme.get("bg_color", "#0d1b2e")
        pl_tmpl= theme.get("plotly_template", "plotly_dark")

        # ── Module header ─────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
                padding: 16px 22px;
                border-radius: 12px;
                border-left: 6px solid #3ab54a;
                margin-bottom: 22px;
            ">
                <div style="font-size:22px;font-weight:700;color:#f8fafc;">
                    📊 Market Trend Analysis
                </div>
                <div style="font-size:13px;color:#94a3b8;margin-top:3px;">
                    MA + FazDane Cloud Command Center &nbsp;|&nbsp;
                    Universe: <b>{universe_name}</b> &nbsp;|&nbsp;
                    Ticker: <b>{ticker}</b> &nbsp;|&nbsp;
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Global Scanner Execution (Displays progress at top of screen) ──
        trigger = st.session_state.pop("mta_trigger_scan", False)
        if trigger or "mta_scan_results" not in st.session_state:
            progress_container = st.container()
            with progress_container:
                scan_banner = st.empty()
                scan_banner.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(90deg, rgba(26,58,143,0.4) 0%, rgba(58,181,74,0.15) 100%);
                        border: 1px solid #1e3a5f;
                        border-left: 5px solid #3ab54a;
                        border-radius: 12px;
                        padding: 18px 22px;
                        margin-bottom: 16px;
                    ">
                        <div style="color:#3ab54a;font-size:16px;font-weight:700;margin-bottom:4px;">
                            🔄 Running Universe Scanner…
                        </div>
                        <div style="color:#94a3b8;font-size:13px;">
                            Scanning <b>{len(all_tickers)}</b> tickers · {ma_type} · {timeframe} · {aggregation}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                self._run_universe_scan(
                    all_tickers, ma_type, periods, timeframe, aggregation,
                    universe_name, scan_banner, progress_container
                )


        tab1, tab2, tab3 = st.tabs([
            "📈 Chart & KPI Deck",
            "🔭 Universe Scanner",
            "📥 Data Export",
        ])

        # ═══════════════════════════════════════════════════════════════
        # TAB 1 — CHART + KPI DECK
        # ═══════════════════════════════════════════════════════════════
        with tab1:
            self._render_chart_tab(
                ticker, ma_type, periods, timeframe, aggregation,
                show_cloud, show_vwap, show_price, bg_col, pl_tmpl,
            )

        # ═══════════════════════════════════════════════════════════════
        # TAB 2 — UNIVERSE SCANNER
        # ═══════════════════════════════════════════════════════════════
        with tab2:
            self._render_scanner_tab(
                all_tickers, ma_type, periods, timeframe, aggregation,
                universe_name,
            )

        # ═══════════════════════════════════════════════════════════════
        # TAB 3 — DATA EXPORT
        # ═══════════════════════════════════════════════════════════════
        with tab3:
            self._render_export_tab(ticker)

    # ── Tab 1: Chart & KPI Deck ────────────────────────────────────────

    def _render_chart_tab(
        self,
        ticker, ma_type, periods, timeframe, aggregation,
        show_cloud, show_vwap, show_price, bg_col, pl_tmpl,
    ) -> None:

        with st.spinner(f"Loading {ticker} data…"):
            raw_df = fetch_ohlcv(ticker, timeframe, aggregation)

        if raw_df is None or raw_df.empty:
            st.error(f"Could not fetch data for **{ticker}**. Check the ticker or try a different timeframe.")
            return

        result = calculate_ma_cloud_indicators(
            raw_df,
            ma_type=ma_type,
            ma_periods=periods,
        )
        df   = result["dataframe"]
        kpis = result["kpis"]

        if not kpis:
            st.warning("Insufficient data to compute indicators. Try a longer timeframe or fewer MA periods.")
            return

        # Cache for Export tab
        st.session_state["mta_last_df"]   = df
        st.session_state["mta_last_kpis"] = kpis
        st.session_state["mta_last_ticker"] = ticker

        # Slice for chart display based on the display_days mapping
        display_days = TIMEFRAME_MAP.get(timeframe, ("3mo", "1d", 90))[2]
        last_date = df["date"].max()
        cutoff_date = last_date - pd.Timedelta(days=display_days)
        chart_df = df[df["date"] >= cutoff_date]

        self._render_kpi_deck(kpis, ma_type, periods)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        company_name = get_company_name(ticker)
        fig = build_candlestick_chart(
            df=chart_df,
            kpis=kpis,
            ma_type=ma_type,
            selected_periods=periods,
            show_vwap=show_vwap,
            show_cloud=show_cloud,
            show_current_price=show_price,
            plotly_template=pl_tmpl,
            bg_color=bg_col,
            ticker=ticker,
            company_name=company_name,
        )
        st.plotly_chart(fig, use_container_width=True, key="mta_main_chart")

        # Trend stack + interpretation banner
        stack = kpis.get("trend_stack", {})
        interp = get_trade_interpretation(kpis)
        stack_status = stack.get("status", "")
        stack_color  = STATUS_COLORS.get(
            "Above" if "Bullish" in stack_status else
            ("Below" if "Bearish" in stack_status else "Near"),
            "#94a3b8",
        )
        st.markdown(
            f"""
            <div style="
                background: rgba(21,40,71,0.5);
                border: 1px solid #1e3a5f;
                border-left: 5px solid {stack_color};
                border-radius: 10px;
                padding: 16px 20px;
                margin-top: 12px;
            ">
                <span style="color:{stack_color};font-weight:700;
                             text-transform:uppercase;font-size:11px;
                             letter-spacing:1px;">MA Stack</span>
                <div style="color:#f8fafc;font-size:16px;font-weight:700;
                            margin: 4px 0 6px 0;">{stack_status}</div>
                <div style="color:#94a3b8;font-size:12px;margin-bottom:8px;">
                    {stack.get("details", "")}
                </div>
                <div style="color:#cbd5e1;font-size:13px;font-style:italic;">
                    💡 {interp}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── KPI Deck ──────────────────────────────────────────────────────

    def _render_kpi_deck(self, kpis: dict, ma_type: str, periods: list[int]) -> None:
        price = kpis.get("current_price", 0)
        cloud = kpis.get("cloud", {})
        vwap  = kpis.get("vwap", {})
        cloud_status = cloud.get("cloud_status", "N/A")
        cloud_trend  = cloud.get("cloud_trend", "N/A")

        # ── Row 1: Current Price + Cloud Summary + VWAP ───────────────
        r1 = st.columns(4)
        _render_wide_card(
            "Current Price",
            f"${price:,.2f}",
            "Last Close",
            "#f8fafc",
            r1[0],
        )
        _render_wide_card(
            "Cloud Status",
            cloud_status,
            f"Span A: ${cloud.get('span_a', 0):,.2f}  |  Span B: ${cloud.get('span_b', 0):,.2f}",
            STATUS_COLORS.get(cloud_status, "#94a3b8"),
            r1[1],
        )
        _render_wide_card(
            "Cloud Trend",
            cloud_trend,
            f"Top: ${cloud.get('cloud_top', 0):,.2f}  |  Bottom: ${cloud.get('cloud_bottom', 0):,.2f}",
            STATUS_COLORS.get(cloud_trend, "#94a3b8"),
            r1[2],
        )
        vwap_status = vwap.get("status", "N/A")
        _render_wide_card(
            "VWAP",
            f"${vwap.get('value', 0):,.2f}" if vwap.get("value") else "N/A",
            f"Dev: {_fmt_dev(vwap.get('deviation_points'), vwap.get('deviation_percent'))}",
            STATUS_COLORS.get(vwap_status, "#94a3b8"),
            r1[3],
        )

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ── Row 2: MA deviation cards ─────────────────────────────────
        ma_list = kpis.get("moving_averages", [])
        ma_by_period = {m["period"]: m for m in ma_list}
        display_periods = [p for p in ALL_MA_PERIODS if p in [m["period"] for m in ma_list]]

        # Chunk into rows of 4
        chunks = [display_periods[i:i+4] for i in range(0, len(display_periods), 4)]
        for chunk in chunks:
            cols = st.columns(len(chunk))
            for i, period in enumerate(chunk):
                m = ma_by_period.get(period, {})
                status = m.get("status", "N/A")
                val    = m.get("value")
                _render_kpi_card(
                    title=f"{period} {ma_type}",
                    value_line=f"${val:,.2f}" if val else "N/A",
                    sub_line=_fmt_dev(m.get("deviation_points"), m.get("deviation_percent")),
                    status=status,
                    col=cols[i],
                )

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ── Row 3: Cloud deviation cards ──────────────────────────────
        cr = st.columns(3)
        span_a_dev_pts = cloud.get("deviation_from_span_a_points")
        span_a_dev_pct = cloud.get("deviation_from_span_a_percent")
        cloud_top_dev_pts = cloud.get("deviation_from_cloud_top_points")
        cloud_top_dev_pct = cloud.get("deviation_from_cloud_top_percent")

        _render_kpi_card(
            title="Span A Deviation",
            value_line=f"${cloud.get('span_a', 0):,.2f}" if cloud.get("span_a") else "N/A",
            sub_line=_fmt_dev(span_a_dev_pts, span_a_dev_pct),
            status="Above" if (span_a_dev_pts or 0) > 0 else "Below",
            col=cr[0],
        )
        _render_kpi_card(
            title="Cloud Top Deviation",
            value_line=f"${cloud.get('cloud_top', 0):,.2f}" if cloud.get("cloud_top") else "N/A",
            sub_line=_fmt_dev(cloud_top_dev_pts, cloud_top_dev_pct),
            status="Above" if (cloud_top_dev_pts or 0) > 0 else ("Near" if (cloud_top_dev_pts or 0) == 0 else "Below"),
            col=cr[1],
        )
        _render_kpi_card(
            title="Cloud Bottom",
            value_line=f"${cloud.get('cloud_bottom', 0):,.2f}" if cloud.get("cloud_bottom") else "N/A",
            sub_line=f"Span B: ${cloud.get('span_b', 0):,.2f}",
            status=cloud_status,
            col=cr[2],
        )

    # ── Tab 2: Universe Scanner ────────────────────────────────────────

    def _render_scanner_tab(
        self,
        all_tickers: list[str],
        ma_type: str,
        periods: list[int],
        timeframe: str,
        aggregation: str,
        universe_name: str,
    ) -> None:

        st.markdown("### Universe MA + Cloud Scanner")
        st.caption(
            f"Runs MA + FazDane Cloud analysis across all {len(all_tickers)} tickers "
            f"in **{universe_name}**. Uses {ma_type} with {timeframe} / {aggregation} data."
        )

        scan_results = st.session_state.get("mta_scan_results", [])

        if not scan_results:
            st.info("Click **Run Universe Scanner** in the sidebar to populate this table.")
            return

        # ── Scanner sort + filter controls ────────────────────────────
        fc1, fc2, fc3 = st.columns([2, 2, 1])
        with fc1:
            search = st.text_input("Search Ticker:", key="mta_scanner_search").upper()
        with fc2:
            cloud_filter = st.multiselect(
                "Filter Cloud Status:",
                options=["Above Cloud", "Inside Cloud", "Below Cloud"],
                key="mta_scanner_cloud_filter",
            )
        with fc3:
            sort_col = st.selectbox(
                "Sort By:",
                options=["Score", "Price", "20 MA Dev%", "Cloud Top Dev%"],
                key="mta_scanner_sort",
            )

        # Build display dataframe
        rows = []
        for r in scan_results:
            ticker = r.get("ticker", "")
            kpis   = r.get("kpis", {})
            if not kpis:
                continue
            cloud  = kpis.get("cloud", {})
            vwap   = kpis.get("vwap", {})
            stack  = kpis.get("trend_stack", {})
            score  = r.get("scanner_score", 0)
            interp = r.get("interpretation", "")
            ma_list = {m["period"]: m for m in kpis.get("moving_averages", [])}

            def _dev_pct(period):
                m = ma_list.get(period, {})
                v = m.get("deviation_percent")
                return round(v, 2) if v is not None else None

            # Calculate FDTS Signal for display in dataframe and boxes
            interp_lower = str(interp).lower()
            if "bullish continuation" in interp_lower or "pullback" in interp_lower:
                fdts_val = "🟢 Buy"
            elif "bearish" in interp_lower:
                fdts_val = "🔴 Sell"
            else:
                fdts_val = "⚪ No Trade"

            rows.append({
                "Ticker": ticker,
                "Price":  kpis.get("current_price"),
                f"20 {ma_type} Dev%": _dev_pct(20),
                f"50 {ma_type} Dev%": _dev_pct(50),
                f"200 {ma_type} Dev%": _dev_pct(200),
                "Cloud Status": cloud.get("cloud_status", "N/A"),
                "Cloud Trend":  cloud.get("cloud_trend", "N/A"),
                "VWAP Status":  vwap.get("status", "N/A"),
                "MA Stack":     stack.get("status", "N/A"),
                "Score":        score,
                "FDTS Signal":  fdts_val,
                "Signal":       interp[:60] + "…" if len(interp) > 60 else interp,
            })

        scanner_df = pd.DataFrame(rows)
        if scanner_df.empty:
            st.warning("No results. Try a different universe or timeframe.")
            return

        # Apply filters
        if search:
            scanner_df = scanner_df[scanner_df["Ticker"].str.contains(search, na=False)]
        if cloud_filter:
            scanner_df = scanner_df[scanner_df["Cloud Status"].isin(cloud_filter)]

        # Sort
        sort_map = {
            "Score":          ("Score", False),
            "Price":          ("Price", False),
            "20 MA Dev%":     (f"20 {ma_type} Dev%", False),
            "Cloud Top Dev%": (f"50 {ma_type} Dev%", False),
        }
        sort_key, asc = sort_map.get(sort_col, ("Score", False))
        if sort_key in scanner_df.columns:
            scanner_df = scanner_df.sort_values(sort_key, ascending=asc)

        # Colour-code score column
        def _score_color(val):
            if val is None:
                return ""
            if val >= 70:
                return "background-color: rgba(34,197,94,0.15); color: #22c55e; font-weight: bold;"
            if val >= 45:
                return "background-color: rgba(234,179,8,0.12); color: #eab308; font-weight: bold;"
            return "background-color: rgba(239,68,68,0.12); color: #ef4444; font-weight: bold;"

        def _cloud_color(val):
            c = STATUS_COLORS.get(str(val), "")
            return f"color: {c}; font-weight: bold;" if c else ""

        try:
            styled = (
                scanner_df.style
                .applymap(_score_color, subset=["Score"])
                .applymap(_cloud_color, subset=["Cloud Status", "Cloud Trend", "VWAP Status", "FDTS Signal"])
                .format({
                    "Price": "${:,.2f}",
                    f"20 {ma_type} Dev%": "{:+.2f}%",
                    f"50 {ma_type} Dev%": "{:+.2f}%",
                    f"200 {ma_type} Dev%": "{:+.2f}%",
                }, na_rep="N/A")
            )
        except Exception:
            styled = scanner_df

        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── Summary metrics row ───────────────────────────────────────
        n_above   = len(scanner_df[scanner_df["Cloud Status"] == "Above Cloud"])
        n_inside  = len(scanner_df[scanner_df["Cloud Status"] == "Inside Cloud"])
        n_below   = len(scanner_df[scanner_df["Cloud Status"] == "Below Cloud"])
        n_bull_cl = len(scanner_df[scanner_df["Cloud Trend"] == "Bullish Cloud"])
        avg_score = scanner_df["Score"].mean() if not scanner_df["Score"].isna().all() else 0

        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        sm1.metric("Above Cloud",   n_above)
        sm2.metric("Inside Cloud",  n_inside)
        sm3.metric("Below Cloud",   n_below)
        sm4.metric("Bullish Cloud", n_bull_cl)
        sm5.metric("Avg Score",     f"{avg_score:.0f}")

        # ── MA Stack & FDTS Segmentation ─────────────────────────────
        def get_stack_group(status: str) -> str:
            status_lower = str(status).lower()
            if "bullish" in status_lower:
                return "Bullish"
            elif "bearish" in status_lower:
                return "Bearish"
            else:
                return "Sideways"

        def get_fdts_signal(interpretation: str) -> str:
            interp_lower = str(interpretation).lower()
            if "bullish continuation" in interp_lower or "pullback" in interp_lower:
                return "Buy"
            elif "bearish" in interp_lower:
                return "Sell"
            else:
                return "No Trade"

        # Group scanner results
        bullish_tickers = []
        sideways_tickers = []
        bearish_tickers = []

        for r in scan_results:
            ticker = r.get("ticker", "")
            kpis   = r.get("kpis", {})
            if not kpis:
                continue
            stack_status = kpis.get("trend_stack", {}).get("status", "Mixed / Sideways")
            group = get_stack_group(stack_status)
            
            interp = r.get("interpretation", "")
            fdts_sig = get_fdts_signal(interp)
            
            ticker_info = {"ticker": ticker, "fdts": fdts_sig, "status": stack_status}
            if group == "Bullish":
                bullish_tickers.append(ticker_info)
            elif group == "Bearish":
                bearish_tickers.append(ticker_info)
            else:
                sideways_tickers.append(ticker_info)

        n_bull_stack = len(bullish_tickers)
        n_side_stack = len(sideways_tickers)
        n_bear_stack = len(bearish_tickers)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown("#### MA Stack Analysis Summary")
        kcol1, kcol2, kcol3 = st.columns(3)
        _render_wide_card("Bullish Stack (MA)", f"{n_bull_stack}", "Strong Bullish / Bullish Trend", "#22c55e", kcol1)
        _render_wide_card("Sideways Stack (MA)", f"{n_side_stack}", "Mixed / Sideways", "#eab308", kcol2)
        _render_wide_card("Bearish Stack (MA)", f"{n_bear_stack}", "Strong Bearish / Bearish Trend", "#ef4444", kcol3)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        
        def _render_segmented_box(title: str, icon: str, tickers: list[dict], border_color: str, bg_opacity: str) -> str:
            sorted_tickers = sorted(tickers, key=lambda x: x["ticker"])
            html = f"""
            <div style="background:rgba({border_color},{bg_opacity}); border:1px solid rgba({border_color},0.25); 
                        border-radius:12px; padding:14px; min-height:260px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="color:rgb({border_color}); font-size:14px; font-weight:700; margin-bottom:12px; 
                            display:flex; align-items:center; gap:6px; border-bottom:1px solid rgba({border_color},0.15); padding-bottom:6px;">
                    <span>{icon}</span> {title} ({len(sorted_tickers)})
                </div>
                <div style="max-height: 280px; overflow-y: auto; padding-right: 2px;">
            """
            if not sorted_tickers:
                html += f"""
                <div style="color:#64748b; font-size:12px; text-align:center; padding:30px 0; font-style:italic;">
                    No tickers in this group
                </div>
                """
            else:
                for t in sorted_tickers:
                    if t["fdts"] == "Buy":
                        dot_bg = "radial-gradient(circle at 3px 3px, #ffffff 0%, #22c55e 50%, #15803d 100%)"
                        text_color = "#22c55e"
                        label = "Buy"
                    elif t["fdts"] == "Sell":
                        dot_bg = "radial-gradient(circle at 3px 3px, #ffffff 0%, #ef4444 50%, #b91c1c 100%)"
                        text_color = "#ef4444"
                        label = "Sell"
                    else:
                        dot_bg = "radial-gradient(circle at 3px 3px, #ffffff 0%, #d8b4fe 40%, #8b5cf6 100%)"
                        text_color = "#94a3b8"
                        label = "No Trade"

                    html += f"""
                    <div style="background:rgba(15,23,42,0.85); border:1px solid #1e3a5f; 
                                border-radius:8px; padding:8px 12px; margin-bottom:8px; 
                                display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; color:#f8fafc; font-size:13px;">{t['ticker']}</span>
                        <div style="display:flex; align-items:center; gap:6px;">
                            <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:{dot_bg}; box-shadow:inset -1px -1px 2px rgba(0,0,0,0.5), 1px 1px 1px rgba(0,0,0,0.3); vertical-align:middle;"></span>
                            <span style="font-size:12px; font-weight:700; color:{text_color}; vertical-align:middle;">{label}</span>
                        </div>
                    </div>
                    """
            html += """
                </div>
            </div>
            """
            # Collapse whitespace and newlines so the Markdown parser treats it as raw HTML, not a code block
            return re.sub(r'\s+', ' ', html).strip()

        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            st.markdown(_render_segmented_box("Bullish Stack Tickers", "🐂", bullish_tickers, "34,197,94", "0.04"), unsafe_allow_html=True)
        with scol2:
            st.markdown(_render_segmented_box("Sideways Stack Tickers", "⚖️", sideways_tickers, "234,179,8", "0.04"), unsafe_allow_html=True)
        with scol3:
            st.markdown(_render_segmented_box("Bearish Stack Tickers", "🐻", bearish_tickers, "239,68,68", "0.04"), unsafe_allow_html=True)

        # ── Copy tickers in comma-delimited format ─────────────────────
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown("#### 📋 Copy Tickers by Category (Comma-Delimited)")
        
        c_cols = st.columns(3)
        with c_cols[0]:
            st.markdown("<span style='color:#34c759;font-weight:700;'>🐂 Bullish Stack Tickers</span>", unsafe_allow_html=True)
            bullish_list = [t["ticker"] for t in sorted(bullish_tickers, key=lambda x: x["ticker"])]
            if bullish_list:
                st.code(", ".join(bullish_list), language=None)
            else:
                st.caption("*No bullish tickers*")
                
        with c_cols[1]:
            st.markdown("<span style='color:#eab308;font-weight:700;'>⚖️ Sideways Stack Tickers</span>", unsafe_allow_html=True)
            sideways_list = [t["ticker"] for t in sorted(sideways_tickers, key=lambda x: x["ticker"])]
            if sideways_list:
                st.code(", ".join(sideways_list), language=None)
            else:
                st.caption("*No sideways tickers*")
                
        with c_cols[2]:
            st.markdown("<span style='color:#ef4444;font-weight:700;'>🐻 Bearish Stack Tickers</span>", unsafe_allow_html=True)
            bearish_list = [t["ticker"] for t in sorted(bearish_tickers, key=lambda x: x["ticker"])]
            if bearish_list:
                st.code(", ".join(bearish_list), language=None)
            else:
                st.caption("*No bearish tickers*")

    def _run_universe_scan(
        self,
        tickers: list[str],
        ma_type: str,
        periods: list[int],
        timeframe: str,
        aggregation: str,
        universe_name: str,
        scan_banner=None,
        progress_container=None,
    ) -> None:
        """Batch-download and analyse all universe tickers with rich live progress UI."""
        if not tickers:
            return

        total = len(tickers)

        # ── Progress elements ──────────────────────────────────────────
        if progress_container is not None:
            with progress_container:
                progress_bar  = st.progress(0.0)
                status_box    = st.empty()   # current ticker / phase
                counter_box   = st.empty()   # X / N counter row
                live_log      = st.empty()   # scrolling ticker log
        else:
            progress_bar  = st.progress(0.0)
            status_box    = st.empty()   # current ticker / phase
            counter_box   = st.empty()   # X / N counter row
            live_log      = st.empty()   # scrolling ticker log

        def _counter_html(done: int, ok: int, skipped: int) -> str:
            return f"""
            <div style="display:flex;gap:20px;margin:6px 0 10px 0;">
                <div style="background:rgba(21,40,71,0.8);border:1px solid #1e3a5f;
                            border-radius:8px;padding:8px 16px;text-align:center;">
                    <div style="color:#94a3b8;font-size:10px;text-transform:uppercase;
                                letter-spacing:1px;">Analysed</div>
                    <div style="color:#f8fafc;font-size:20px;font-weight:700;">{done}/{total}</div>
                </div>
                <div style="background:rgba(21,40,71,0.8);border:1px solid #1e3a5f;
                            border-radius:8px;padding:8px 16px;text-align:center;">
                    <div style="color:#94a3b8;font-size:10px;text-transform:uppercase;
                                letter-spacing:1px;">Scored</div>
                    <div style="color:#22c55e;font-size:20px;font-weight:700;">{ok}</div>
                </div>
                <div style="background:rgba(21,40,71,0.8);border:1px solid #1e3a5f;
                            border-radius:8px;padding:8px 16px;text-align:center;">
                    <div style="color:#94a3b8;font-size:10px;text-transform:uppercase;
                                letter-spacing:1px;">Skipped</div>
                    <div style="color:#eab308;font-size:20px;font-weight:700;">{skipped}</div>
                </div>
            </div>
            """

        # ── Phase 1: Download ──────────────────────────────────────────
        status_box.markdown(
            """<div style='color:#3ab54a;font-weight:600;font-size:14px;
               margin:4px 0;'>⬇ Downloading market data for all tickers…</div>""",
            unsafe_allow_html=True,
        )
        counter_box.markdown(_counter_html(0, 0, 0), unsafe_allow_html=True)
        progress_bar.progress(0.02)  # show a sliver while waiting

        all_data = fetch_universe_ohlcv(tuple(tickers), timeframe, aggregation)
        n_downloaded = len(all_data)

        status_box.markdown(
            f"""<div style='color:#3ab54a;font-weight:600;font-size:14px;margin:4px 0;'>
               ✅ Downloaded data for {n_downloaded}/{total} tickers. Analysing…</div>""",
            unsafe_allow_html=True,
        )

        # ── Phase 2: Analyse each ticker ───────────────────────────────
        results      = []
        skipped      = []
        log_lines    = []
        min_bars     = max(periods + [26 * 2]) if periods else 52

        for idx, ticker in enumerate(tickers):
            pct = (idx + 1) / total
            progress_bar.progress(pct)

            # Live status line
            status_box.markdown(
                f"""<div style='color:#94a3b8;font-size:13px;margin:4px 0;'>
                   🔍 Analysing <b style='color:#f8fafc;'>{ticker}</b>
                   &nbsp;—&nbsp; {idx + 1} / {total}
                </div>""",
                unsafe_allow_html=True,
            )
            counter_box.markdown(
                _counter_html(idx + 1, len(results), len(skipped)),
                unsafe_allow_html=True,
            )

            df = all_data.get(ticker)
            if df is None or df.empty or len(df) < min_bars:
                reason = "no data" if (df is None or df.empty) else f"only {len(df)} bars"
                skipped.append(ticker)
                log_lines.append(
                    f"<span style='color:#64748b;'>⚠ {ticker} — skipped ({reason})</span>"
                )
                live_log.markdown(
                    "<div style='font-size:11px;font-family:monospace;line-height:1.8;'>"
                    + "<br>".join(log_lines[-12:]) + "</div>",
                    unsafe_allow_html=True,
                )
                continue

            try:
                result = calculate_ma_cloud_indicators(df, ma_type=ma_type, ma_periods=periods)
                kpis   = result.get("kpis", {})
                if not kpis:
                    skipped.append(ticker)
                    log_lines.append(
                        f"<span style='color:#64748b;'>⚠ {ticker} — no KPIs computed</span>"
                    )
                else:
                    score  = calculate_scanner_score(kpis)
                    interp = get_trade_interpretation(kpis)
                    results.append({
                        "ticker":          ticker,
                        "kpis":            kpis,
                        "scanner_score":   score,
                        "interpretation":  interp,
                    })
                    cloud_s = kpis.get("cloud", {}).get("cloud_status", "")
                    score_color = (
                        "#22c55e" if score >= 70 else
                        ("#eab308" if score >= 45 else "#ef4444")
                    )
                    log_lines.append(
                        f"<span style='color:{score_color};font-weight:600;'>"
                        f"✓ {ticker}</span>"
                        f"<span style='color:#64748b;'> — Score: "
                        f"<b style='color:{score_color};'>{score}</b>"
                        f" | {cloud_s}</span>"
                    )
            except Exception as e:
                skipped.append(ticker)
                log_lines.append(
                    f"<span style='color:#ef4444;'>✗ {ticker} — error: {str(e)[:40]}</span>"
                )
                logger.warning(f"Scan failed for {ticker}: {e}")

            live_log.markdown(
                "<div style='font-size:11px;font-family:monospace;line-height:1.8;'>"
                + "<br>".join(log_lines[-12:]) + "</div>",
                unsafe_allow_html=True,
            )

        # ── Done: clear live elements, show final summary ───────────────
        progress_bar.progress(1.0)
        status_box.empty()
        counter_box.empty()
        live_log.empty()
        progress_bar.empty()
        if scan_banner:
            scan_banner.empty()

        st.session_state["mta_scan_results"] = results

        # Final summary card
        n_ok  = len(results)
        n_skip = len(skipped)
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(34,197,94,0.10) 0%,
                            rgba(21,40,71,0.80) 100%);
                border: 1px solid #1e3a5f;
                border-left: 5px solid #22c55e;
                border-radius: 12px;
                padding: 16px 20px;
                margin-bottom: 16px;
            ">
                <div style="color:#22c55e;font-size:15px;font-weight:700;
                            margin-bottom:6px;">✅ Scanner Complete</div>
                <div style="display:flex;gap:24px;">
                    <div><span style="color:#94a3b8;font-size:12px;">Scored</span><br>
                         <span style="color:#22c55e;font-size:22px;font-weight:700;">{n_ok}</span></div>
                    <div><span style="color:#94a3b8;font-size:12px;">Skipped</span><br>
                         <span style="color:#eab308;font-size:22px;font-weight:700;">{n_skip}</span></div>
                    <div><span style="color:#94a3b8;font-size:12px;">Universe</span><br>
                         <span style="color:#f8fafc;font-size:22px;font-weight:700;">{universe_name}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Persist to SQLite
        if results:
            save_scanner_snapshot(results, ma_type)

    # ── Tab 3: Data Export ────────────────────────────────────────────

    def _render_export_tab(self, ticker: str) -> None:
        st.markdown("### Data Export")

        df   = st.session_state.get("mta_last_df")
        kpis = st.session_state.get("mta_last_kpis")
        last_ticker = st.session_state.get("mta_last_ticker", ticker)

        if df is None or df.empty:
            st.info("Run the chart for a ticker first (Tab 1), then return here to export the data.")
            return

        st.caption(
            f"Showing indicator columns for **{last_ticker}**. "
            f"{len(df)} rows."
        )

        # Select columns to display
        indicator_cols = [
            c for c in df.columns
            if c not in ("chikou",)  # exclude raw chikou from default view
        ]
        st.dataframe(df[indicator_cols].tail(50), use_container_width=True)

        # Download buttons
        d1, d2 = st.columns(2)
        with d1:
            csv = df[indicator_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download CSV",
                data=csv,
                file_name=f"{last_ticker}_ma_cloud_indicators.csv",
                mime="text/csv",
                key="mta_dl_csv",
            )
        with d2:
            if kpis:
                import json
                kpi_json = json.dumps(kpis, indent=2, default=str).encode("utf-8")
                st.download_button(
                    label="⬇ Download KPI JSON",
                    data=kpi_json,
                    file_name=f"{last_ticker}_kpis.json",
                    mime="application/json",
                    key="mta_dl_json",
                )

        # KPI summary display
        if kpis:
            st.markdown("#### KPI Summary")
            st.json(kpis)
