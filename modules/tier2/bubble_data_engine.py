"""
Bubble Indicator Data Engine
============================
Grantham-inspired bubble-risk framework.

Design principles (v2):
- Scores are TRUE percentile ranks vs. available history (not clipped z-scores).
- Secular-trend series (QQQ/SPY, SMH/SPY, ...) are de-trended vs. a rolling
  3-year mean before ranking, so "always rising" ratios don't pin at 100.
- The Bubble Score History is genuinely computed from daily component series
  (weight-renormalised when a component's history hasn't started yet).
- Live data: Yahoo Finance (prices), FRED (rates / credit / money / margins),
  multpl.com (S&P 500 valuation metrics with full monthly history).
- Every external call degrades gracefully; `data_quality` reports what is
  live vs. proxy vs. unavailable.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import tomllib
except ImportError:  # pragma: no cover
    tomllib = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CORE_TICKERS = [
    "SPY", "RSP", "QQQ", "SMH", "HYG", "LQD",
    "^GSPC", "^VIX", "^VIX3M", "^TNX",
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
    "EFA", "VWO", "GLD", "TLT", "BIL", "BTC-USD",
]

MAG7 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]

# Static S&P 100 snapshot used for real breadth computation.
SP100_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "AVGO",
    "JPM", "LLY", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ", "NFLX",
    "WMT", "ABBV", "CRM", "BAC", "ORCL", "CVX", "WFC", "KO", "CSCO", "ACN",
    "AMD", "PEP", "LIN", "MCD", "ADBE", "PM", "DIS", "TMO", "ABT", "IBM",
    "GE", "INTU", "CAT", "QCOM", "VZ", "TXN", "AXP", "BKNG", "MS", "SPGI",
    "ISRG", "CMCSA", "RTX", "AMGN", "NEE", "PFE", "UNP", "T", "LOW", "GS",
    "HON", "ETN", "BLK", "SYK", "NKE", "TJX", "BSX", "PLTR", "SCHW", "UBER",
    "CI", "DE", "UPS", "PGR", "BMY", "ADP", "MMC", "MDT", "AMAT", "COP",
    "MO", "SO", "CB", "LMT", "GILD", "ANET", "MU", "ICE", "DUK", "PLD",
    "ELV", "SBUX", "EMR", "CL", "GD", "PYPL", "TGT", "FDX", "USB", "MET",
]

FRED_SERIES = {
    "fed_funds": "DFF",              # Effective Fed Funds (daily)
    "us10y": "DGS10",                # 10Y Treasury yield
    "us2y": "DGS2",                  # 2Y Treasury yield
    "hy_oas": "BAMLH0A0HYM2",        # High Yield OAS
    "ig_oas": "BAMLC0A0CM",          # Investment Grade OAS
    "m2": "M2SL",                    # M2 money stock (monthly)
    "fed_bs": "WALCL",               # Fed balance sheet (weekly)
    "cpi": "CPIAUCSL",               # CPI (monthly)
    "corp_profits": "CPATAX",        # Corporate profits after tax (quarterly)
    "gdp": "GDP",                    # Nominal GDP (quarterly)
    "corp_equities": "NCBEILQ027S",  # Corporate equities market value (Z.1, quarterly)
}

MULTPL_PAGES = {
    "CAPE (Shiller P/E)": "shiller-pe",
    "Trailing P/E": "s-p-500-pe-ratio",
    "Price / Sales": "s-p-500-price-to-sales",
    "Price / Book": "s-p-500-price-to-book",
    "Dividend Yield": "s-p-500-dividend-yield",
}

# Spec weights (sum = 1.0). Profit Margins is displayed but carries no master
# weight per the specification.
MASTER_WEIGHTS = {
    "Valuation": 0.20,
    "Momentum": 0.15,
    "Sentiment": 0.15,
    "Liquidity": 0.10,
    "Credit": 0.10,
    "Breadth": 0.10,
    "Concentration": 0.10,
    "AI Bubble": 0.10,
}

DETREND_WINDOW = 756  # ~3 trading years


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_rank(series: pd.Series) -> pd.Series:
    """Full-sample percentile rank (0-100) of every observation."""
    s = series.dropna()
    if s.empty:
        return series * np.nan
    ranked = s.rank(pct=True) * 100.0
    return ranked.reindex(series.index)


def _detrended(series: pd.Series, window: int = DETREND_WINDOW) -> pd.Series:
    """Series divided by its rolling mean - removes secular drift."""
    roll = series.rolling(window, min_periods=window // 3).mean()
    return series / roll


def get_fred_api_key() -> str | None:
    key = os.getenv("FRED_API_KEY")
    if key:
        return key
    try:
        key = st.secrets.get("FRED_API_KEY", None)
        if key:
            return key
    except Exception:
        pass
    secrets_path = Path(".streamlit") / "secrets.toml"
    if secrets_path.exists() and tomllib is not None:
        try:
            with secrets_path.open("rb") as handle:
                return tomllib.load(handle).get("FRED_API_KEY")
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Fetch layers (each cached + fault-tolerant)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history() -> pd.DataFrame:
    """Daily closes for core tickers, maximum available history."""
    import yfinance as yf

    try:
        data = yf.download(
            CORE_TICKERS, period="max", interval="1d",
            progress=False, group_by="column", auto_adjust=True,
        )
        if isinstance(data.columns, pd.MultiIndex):
            df = data["Close"]
        else:
            df = data[["Close"]]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        # Drop crypto weekends so the index matches equity trading days.
        df = df[df.index.dayofweek < 5]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series() -> dict:
    """Dict of FRED series (pd.Series). Empty dict if key/network missing."""
    key = get_fred_api_key()
    if not key:
        return {}
    try:
        from fredapi import Fred
        fred = Fred(api_key=key)
    except Exception:
        return {}
    out = {}
    for name, sid in FRED_SERIES.items():
        try:
            s = fred.get_series(sid)
            s.index = pd.to_datetime(s.index)
            out[name] = s.dropna()
        except Exception:
            continue
    return out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_multpl_history() -> dict:
    """
    Full monthly history for S&P 500 valuation metrics from multpl.com.
    Returns {metric: pd.Series}. Empty dict on failure.
    """
    import io
    import requests

    out = {}
    headers = {"User-Agent": "Mozilla/5.0 (research dashboard)"}
    for metric, slug in MULTPL_PAGES.items():
        try:
            url = f"https://www.multpl.com/{slug}/table/by-month"
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
            if not tables:
                continue
            tbl = tables[0]
            tbl.columns = ["Date", "Value"]
            tbl["Date"] = pd.to_datetime(tbl["Date"], errors="coerce")
            tbl["Value"] = (
                tbl["Value"].astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.extract(r"([\d.]+)")[0]
                .astype(float)
            )
            s = tbl.dropna().set_index("Date")["Value"].sort_index()
            if len(s) > 24:
                out[metric] = s
        except Exception:
            continue
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_breadth_data() -> pd.DataFrame:
    """2 years of daily closes for the S&P 100 basket (real breadth)."""
    import yfinance as yf

    try:
        data = yf.download(
            SP100_TICKERS, period="2y", interval="1d",
            progress=False, group_by="column", auto_adjust=True,
        )
        if isinstance(data.columns, pd.MultiIndex):
            df = data["Close"]
        else:
            df = data
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.dropna(axis=1, how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_us_vs_world() -> dict:
    """Fundamental ratios for SPY vs. developed/EM world via yfinance .info."""
    import yfinance as yf

    out = {}
    for tk in ["SPY", "EFA", "VWO"]:
        try:
            info = yf.Ticker(tk).info or {}
            out[tk] = {
                "pe": info.get("trailingPE"),
                "pb": info.get("priceToBook"),
                "div_yield": info.get("dividendYield") or info.get("yield"),
                "fwd_pe": info.get("forwardPE"),
            }
        except Exception:
            out[tk] = {}
    return out


# ---------------------------------------------------------------------------
# Component score computation (all daily series, 0-100 percentile ranks)
# ---------------------------------------------------------------------------

def _compute_component_series(px: pd.DataFrame, fred: dict,
                              multpl: dict) -> tuple[pd.DataFrame, dict]:
    """
    Returns (component_scores_df, quality_flags).
    Each column of the df is a 0-100 daily percentile score.
    """
    idx = px.index
    comps = pd.DataFrame(index=idx)
    quality = {}

    def to_daily(s: pd.Series) -> pd.Series:
        return s.reindex(idx.union(s.index)).ffill().reindex(idx)

    spx = px["^GSPC"].dropna() if "^GSPC" in px else pd.Series(dtype=float)

    # -- 1. Valuation: CAPE percentile (true, monthly) blended with SPX
    #       deviation from its rolling 5-year mean.
    parts = []
    if "CAPE (Shiller P/E)" in multpl:
        cape_pct = _pct_rank(multpl["CAPE (Shiller P/E)"])
        parts.append(to_daily(cape_pct))
        quality["valuation"] = "live (Shiller CAPE + trend)"
    else:
        quality["valuation"] = "proxy (price trend only)"
    if not spx.empty:
        dev = spx / spx.rolling(1260, min_periods=252).mean()
        parts.append(to_daily(_pct_rank(dev)))
    if parts:
        comps["Valuation"] = pd.concat(parts, axis=1).mean(axis=1)

    # -- 2. Momentum: 12M + 6M SPX return percentiles.
    if not spx.empty:
        r12 = _pct_rank(spx.pct_change(252))
        r6 = _pct_rank(spx.pct_change(126))
        comps["Momentum"] = to_daily(pd.concat([r12, r6], axis=1).mean(axis=1))
        quality["momentum"] = "live (SPX 6M/12M returns)"

    # -- 3. Sentiment: complacency = low VIX percentile (inverted) +
    #       VIX term-structure contango (VIX3M/VIX high -> complacent).
    parts = []
    if "^VIX" in px:
        vix = px["^VIX"].dropna()
        parts.append(to_daily(100 - _pct_rank(vix)))
        if "^VIX3M" in px:
            ts = (px["^VIX3M"] / px["^VIX"]).dropna()
            parts.append(to_daily(_pct_rank(ts)))
        quality["sentiment"] = "live (VIX level + term structure)"
    if parts:
        comps["Sentiment"] = pd.concat(parts, axis=1).mean(axis=1)

    # -- 4. Liquidity: looseness fuels bubbles. Low real 10Y rate, M2
    #       acceleration, Fed balance-sheet growth -> higher score.
    parts = []
    if "us10y" in fred and "cpi" in fred:
        cpi_yoy = fred["cpi"].pct_change(12) * 100
        real_rate = fred["us10y"].resample("MS").mean() - cpi_yoy
        parts.append(to_daily(100 - _pct_rank(real_rate.dropna())))
    if "m2" in fred:
        m2_yoy = fred["m2"].pct_change(12).dropna()
        parts.append(to_daily(_pct_rank(m2_yoy)))
    if "fed_bs" in fred:
        bs_yoy = fred["fed_bs"].pct_change(52).dropna()
        parts.append(to_daily(_pct_rank(bs_yoy)))
    if parts:
        comps["Liquidity"] = pd.concat(parts, axis=1).mean(axis=1)
        quality["liquidity"] = "live (FRED real rates / M2 / Fed B/S)"
    elif "^TNX" in px:
        tnx = px["^TNX"].dropna()
        comps["Liquidity"] = to_daily(100 - _pct_rank(tnx))
        quality["liquidity"] = "proxy (10Y yield only - add FRED key)"

    # -- 5. Credit: tight spreads = complacent, easy credit -> higher score.
    parts = []
    if "hy_oas" in fred:
        parts.append(to_daily(100 - _pct_rank(fred["hy_oas"])))
        quality["credit"] = "live (FRED HY/IG OAS)"
    if "HYG" in px and "LQD" in px:
        ratio = _detrended((px["HYG"] / px["LQD"]).dropna())
        parts.append(to_daily(_pct_rank(ratio)))
        quality.setdefault("credit", "proxy (HYG/LQD ratio)")
    if parts:
        comps["Credit"] = pd.concat(parts, axis=1).mean(axis=1)

    # -- 6. Breadth: bubble risk = strong index with weak participation.
    #       Rising RSP/SPY (broad rally) LOWERS the score.
    if "RSP" in px and "SPY" in px:
        ratio = _detrended((px["RSP"] / px["SPY"]).dropna())
        narrow = 100 - _pct_rank(ratio)
        # Divergence kicker: index momentum strong while breadth weak.
        if "Momentum" in comps:
            div = (comps["Momentum"] + to_daily(narrow)) / 2
            comps["Breadth"] = div
        else:
            comps["Breadth"] = to_daily(narrow)
        quality["breadth_score"] = "live (RSP/SPY participation)"

    # -- 7. Concentration: QQQ/SPY + MAG7 basket vs SPY (both de-trended).
    parts = []
    if "QQQ" in px and "SPY" in px:
        parts.append(to_daily(_pct_rank(_detrended((px["QQQ"] / px["SPY"]).dropna()))))
    mag7_cols = [t for t in MAG7 if t in px]
    if len(mag7_cols) >= 5 and "SPY" in px:
        mag7_idx = px[mag7_cols].apply(lambda c: c / c.dropna().iloc[0] if c.notna().any() else c).mean(axis=1)
        ratio = _detrended((mag7_idx / px["SPY"]).dropna())
        parts.append(to_daily(_pct_rank(ratio)))
    if parts:
        comps["Concentration"] = pd.concat(parts, axis=1).mean(axis=1)
        quality["concentration"] = "live (QQQ/SPY + MAG7 relative)"

    # -- 8. AI Bubble: SMH/SPY + NVDA/SPY relative strength (de-trended).
    parts = []
    if "SMH" in px and "SPY" in px:
        parts.append(to_daily(_pct_rank(_detrended((px["SMH"] / px["SPY"]).dropna()))))
    if "NVDA" in px and "SPY" in px:
        parts.append(to_daily(_pct_rank(_detrended((px["NVDA"] / px["SPY"]).dropna()))))
    if parts:
        comps["AI Bubble"] = pd.concat(parts, axis=1).mean(axis=1)
        quality["ai"] = "live (SMH + NVDA vs SPY)"

    # -- 9. Profit Margins (display only): corporate profits / GDP percentile.
    if "corp_profits" in fred and "gdp" in fred:
        margin = (fred["corp_profits"] / fred["gdp"]).dropna()
        comps["Profit Margins"] = to_daily(_pct_rank(margin))
        quality["margins"] = "live (FRED corp profits / GDP)"
    else:
        quality["margins"] = "unavailable (needs FRED key)"

    return comps, quality


def _master_history(comps: pd.DataFrame) -> pd.Series:
    """Weighted master score with weight renormalisation for missing comps."""
    cols = [c for c in MASTER_WEIGHTS if c in comps.columns]
    if not cols:
        return pd.Series(dtype=float)
    w = pd.Series({c: MASTER_WEIGHTS[c] for c in cols})
    vals = comps[cols]
    mask = vals.notna()
    weight_sum = mask.mul(w, axis=1).sum(axis=1)
    weighted = vals.mul(w, axis=1).sum(axis=1, min_count=1)
    master = weighted / weight_sum.replace(0, np.nan)
    master = master.dropna()
    # Weekly smoothing keeps the long chart readable without lag distortion.
    return master.rolling(5, min_periods=1).mean()


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def _valuation_snapshot(multpl: dict, fred: dict | None = None) -> list[dict]:
    rows = []
    fred = fred or {}

    # Buffett Indicator: US corporate equities market value / nominal GDP
    # (Fed Z.1 flow-of-funds, quarterly, history to 1945). NCBEILQ027S is in
    # $millions, GDP in $billions.
    if "corp_equities" in fred and "gdp" in fred:
        try:
            eq = (fred["corp_equities"] / 1000.0)
            gdp = fred["gdp"]
            common = eq.index.intersection(gdp.index)
            buffett = (eq.reindex(common) / gdp.reindex(common)).dropna()
            if len(buffett) > 20:
                rows.append({
                    "Metric": "Buffett Indicator (Mkt Cap / GDP)",
                    "Current": round(float(buffett.iloc[-1]), 2),
                    "Historical Avg": round(float(buffett.mean()), 2),
                    "Percentile": int(round(float(_pct_rank(buffett).iloc[-1]))),
                })
        except Exception:
            pass

    for metric, series in multpl.items():
        cur = float(series.iloc[-1])
        avg = float(series.mean())
        pct = float(_pct_rank(series).iloc[-1])
        if metric == "Dividend Yield":
            pct = 100 - pct  # low yield = expensive
            metric = "Dividend Yield (low = rich)"
        rows.append({
            "Metric": metric,
            "Current": round(cur, 2),
            "Historical Avg": round(avg, 2),
            "Percentile": int(round(pct)),
        })
    rows.sort(key=lambda r: -r["Percentile"])
    return rows


def _liquidity_snapshot(fred: dict) -> list[dict]:
    rows = []

    def add(name, series, fmt, invert=False, spark_months=6):
        if series is None or series.empty:
            return
        cur = float(series.iloc[-1])
        pct = float(_pct_rank(series).iloc[-1])
        eff = 100 - pct if invert else pct
        status = ("TIGHT" if eff < 33 else "NEUTRAL" if eff < 66 else "LOOSE")
        cutoff = series.index[-1] - pd.DateOffset(months=spark_months)
        spark = series[series.index >= cutoff]
        rows.append({
            "Indicator": name,
            "Level": fmt.format(cur),
            "Status": status,
            "Spark": [float(v) for v in spark.tail(60).values],
        })

    if "fed_funds" in fred:
        add("Fed Funds Rate", fred["fed_funds"], "{:.2f}%", invert=True)
    if "us10y" in fred:
        add("10Y Treasury Yield", fred["us10y"], "{:.2f}%", invert=True)
    if "us10y" in fred and "us2y" in fred:
        curve = (fred["us10y"] - fred["us2y"]).dropna()
        add("Yield Curve (10Y-2Y)", curve, "{:+.2f}%")
    if "ig_oas" in fred:
        add("IG Credit Spread", fred["ig_oas"] * 100, "{:.0f} bps", invert=True)
    if "hy_oas" in fred:
        add("HY Credit Spread", fred["hy_oas"] * 100, "{:.0f} bps", invert=True)
    if "m2" in fred:
        m2_yoy = (fred["m2"].pct_change(12) * 100).dropna()
        add("M2 Money Growth (YoY)", m2_yoy, "{:.1f}%", spark_months=24)
    return rows


def _breadth_stats(breadth_px: pd.DataFrame, px: pd.DataFrame) -> dict:
    out = {"available": False}
    if breadth_px.empty or breadth_px.shape[1] < 30:
        return out
    closes = breadth_px
    sma200 = closes.rolling(200, min_periods=150).mean()
    above = (closes > sma200).sum(axis=1)
    valid = sma200.notna().sum(axis=1).replace(0, np.nan)
    pct_above = (above / valid * 100).dropna()

    hi52 = closes.rolling(252, min_periods=200).max()
    lo52 = closes.rolling(252, min_periods=200).min()
    at_high = (closes >= hi52 * 0.999).sum(axis=1)
    at_low = (closes <= lo52 * 1.001).sum(axis=1)

    adv = (closes.diff() > 0).sum(axis=1)
    dec = (closes.diff() < 0).sum(axis=1).replace(0, np.nan)

    nh = int(at_high.iloc[-1])
    nl = int(at_low.iloc[-1])
    out.update({
        "available": True,
        "universe": int(closes.shape[1]),
        "pct_above_200": float(pct_above.iloc[-1]),
        "pct_above_series": pct_above.tail(378),
        "new_highs": nh,
        "new_lows": nl,
        "high_low_ratio": round(nh / max(nl, 1), 1),
        "adv_dec": round(float(adv.iloc[-1] / dec.iloc[-1]) if not pd.isna(dec.iloc[-1]) else float("nan"), 2),
    })
    if "^GSPC" in px:
        spx = px["^GSPC"].dropna()
        out["momentum_12m"] = float(spx.pct_change(252).iloc[-1] * 100)
    return out


def _asset_class_ytd(px: pd.DataFrame) -> list[dict]:
    year_start = pd.Timestamp(datetime.now().year, 1, 1)
    assets = [
        ("US Equities", "SPY", "📈"),
        ("Intl Equities", "EFA", "🌍"),
        ("Emerging Mkts", "VWO", "🌏"),
        ("Treasuries (20Y)", "TLT", "🧾"),
        ("Gold", "GLD", "🥇"),
        ("Crypto (BTC)", "BTC-USD", "₿"),
        ("Cash (T-Bills)", "BIL", "💵"),
    ]
    rows = []
    for label, tk, icon in assets:
        if tk not in px:
            continue
        s = px[tk].dropna()
        base = s[s.index >= year_start]
        if len(base) < 2:
            continue
        ytd = (base.iloc[-1] / base.iloc[0] - 1) * 100
        rows.append({"Asset": label, "Icon": icon, "YTD": round(float(ytd), 1)})
    return rows


def _us_vs_world_panel(usvw: dict) -> list[dict]:
    spy = usvw.get("SPY", {})
    rows = []
    for metric, key, better_low in [
        ("Trailing P/E", "pe", True),
        ("Forward P/E", "fwd_pe", True),
        ("Price / Book", "pb", True),
        ("Dividend Yield %", "div_yield", False),
    ]:
        us = spy.get(key)
        efa = usvw.get("EFA", {}).get(key)
        vwo = usvw.get("VWO", {}).get(key)
        world_vals = [v for v in (efa, vwo) if v]
        if not us or not world_vals:
            continue
        world = float(np.mean(world_vals))
        if key == "div_yield" and us < 1:  # normalise fraction -> %
            us, world = us * 100, world * 100
        rows.append({
            "Metric": metric,
            "US": round(float(us), 2),
            "World": round(world, 2),
            "Ratio": round(float(us) / world, 2) if world else None,
        })
    return rows


def _regime_and_context(master: float, comps_now: dict, px: pd.DataFrame,
                        fred: dict) -> dict:
    vix = float(px["^VIX"].dropna().iloc[-1]) if "^VIX" in px else float("nan")
    mom = comps_now.get("Momentum", 50)
    if master >= 80:
        regime = "BUBBLE"
    elif master >= 60:
        regime = "RISK-ON / LATE CYCLE"
    elif mom < 30 and vix > 30:
        regime = "RISK-OFF"
    elif mom < 40:
        regime = "NEUTRAL / RECOVERY"
    else:
        regime = "RISK-ON"

    fed_status = "UNKNOWN"
    if "fed_funds" in fred:
        ff = fred["fed_funds"].dropna()
        now, prev = float(ff.iloc[-1]), float(ff[ff.index <= ff.index[-1] - pd.DateOffset(months=6)].iloc[-1]) if len(ff) > 200 else (float(ff.iloc[-1]), float(ff.iloc[0]))
        chg = now - prev
        fed_status = "CUTTING" if chg < -0.15 else "HIKING" if chg > 0.15 else "ON HOLD"
    return {"regime": regime, "vix": vix, "fed_status": fed_status}


def _allocation(master: float) -> dict:
    """Rule-based allocation guidance by bubble-score band."""
    bands = [
        (20, {"Equities": 75, "International": 15, "Treasuries": 5, "Gold": 0, "Cash": 5}),
        (40, {"Equities": 65, "International": 15, "Treasuries": 10, "Gold": 5, "Cash": 5}),
        (60, {"Equities": 50, "International": 15, "Treasuries": 15, "Gold": 10, "Cash": 10}),
        (80, {"Equities": 35, "International": 15, "Treasuries": 20, "Gold": 10, "Cash": 20}),
        (101, {"Equities": 25, "International": 10, "Treasuries": 25, "Gold": 10, "Cash": 30}),
    ]
    for cap, alloc in bands:
        if master < cap:
            return alloc
    return bands[-1][1]


def _commentary(master: float, master_hist: pd.Series,
                comps_now: dict, trends: dict) -> str:
    prev = float(master_hist.iloc[-22]) if len(master_hist) > 22 else master
    direction = "increased" if master > prev + 1 else "decreased" if master < prev - 1 else "held steady"
    drivers = sorted(
        [(k, v) for k, v in comps_now.items() if k in MASTER_WEIGHTS],
        key=lambda kv: -kv[1],
    )[:3]
    rising = [k for k, d in trends.items() if d > 2]
    txt = (
        f"The Bubble Score {direction} from {prev:.0f} to {master:.0f} over the "
        f"past month. Highest-risk components: "
        + ", ".join(f"{k} ({v:.0f})" for k, v in drivers) + "."
    )
    if rising:
        txt += " Rising pressure in: " + ", ".join(rising) + "."
    if master >= 80:
        txt += " Framework suggests avoiding adds to crowded momentum names and raising defensive allocation."
    elif master >= 60:
        txt += " Framework suggests selectivity and trimming into strength; risk of mean reversion is elevated."
    else:
        txt += " Risk conditions are within normal historical ranges."
    return txt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_bubble_data() -> dict:
    """Assemble everything the Bubble Indicator UI needs."""
    px = fetch_price_history()
    fred = fetch_fred_series()
    multpl = fetch_multpl_history()
    breadth_px = fetch_breadth_data()
    usvw = fetch_us_vs_world()

    quality = {
        "prices": "live (Yahoo Finance)" if not px.empty else "UNAVAILABLE",
        "fred": "live (FRED)" if fred else "unavailable - add FRED_API_KEY",
        "multpl": "live (multpl.com)" if multpl else "unavailable",
        "breadth_universe": "live (S&P 100 constituents)" if not breadth_px.empty else "unavailable",
    }

    if px.empty:
        return {"error": "Market data unavailable (Yahoo Finance unreachable).",
                "data_quality": quality}

    comps, comp_quality = _compute_component_series(px, fred, multpl)
    quality.update(comp_quality)

    master_hist = _master_history(comps)
    master = float(master_hist.iloc[-1]) if not master_hist.empty else 50.0

    # Latest component scores + 1-month trend deltas.
    comps_now, trends = {}, {}
    for col in comps.columns:
        s = comps[col].dropna()
        if s.empty:
            continue
        comps_now[col] = float(s.iloc[-1])
        trends[col] = float(s.iloc[-1] - s.iloc[-22]) if len(s) > 22 else 0.0

    # Sub-scores used by the "Market Excitement" card (spec/mockup naming).
    if "Momentum" in comps_now and "Sentiment" in comps_now:
        comps_now["Market Excitement"] = (comps_now["Momentum"] + comps_now["Sentiment"]) / 2
        trends["Market Excitement"] = (trends.get("Momentum", 0) + trends.get("Sentiment", 0)) / 2

    context = _regime_and_context(master, comps_now, px, fred)
    breadth = _breadth_stats(breadth_px, px)

    # Crash-probability heuristic: logistic mapping of score, labelled as such.
    crash_12m = 100 / (1 + np.exp(-(master - 72) / 9.0))

    return {
        "master_score": master,
        "history": master_hist[master_hist.index >= "1990-01-01"],
        "components": comps_now,
        "trends": trends,
        "val_snapshot": _valuation_snapshot(multpl, fred),
        "liquidity_snapshot": _liquidity_snapshot(fred),
        "breadth": breadth,
        "asset_ytd": _asset_class_ytd(px),
        "us_vs_world": _us_vs_world_panel(usvw),
        "context": context,
        "allocation": _allocation(master),
        "commentary": _commentary(master, master_hist, comps_now, trends),
        "crash_prob_12m": float(crash_12m),
        "data_quality": quality,
        "as_of": px.index[-1].strftime("%b %d, %Y"),
    }
