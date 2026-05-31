"""
FazDane Analytics - Tier 2
Portfolio Performance & Risk Management
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from modules.base_module import FazDaneModule
from modules.tier2.portfolio_performance import BRAND, fmt_money, fmt_num, style_figure
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
    clean_ticker_for_lookup,
    parse_uploaded_files,
    get_broker_dot,
    format_ticker_for_display,
    classify_option_strategy,
    save_portfolio_log,
    delete_portfolio_log,
    get_portfolio_logs,
    get_portfolio_log_images,
)
from utils.universe_manager import update_fazdane_portfolio_universe


ACTION_COLORS = {
    "Add": "#3ab54a",
    "Hold": "#93c5fd",
    "Watch": "#facc15",
    "Trim": "#f97316",
    "Hedge": "#a78bfa",
    "Eliminate": "#ef4444",
    "Redeploy": "#38bdf8",
}

SECTOR_MAP = {
    "AAPL": "Mega Cap Tech",
    "MSFT": "Mega Cap Tech",
    "GOOGL": "Mega Cap Tech",
    "GOOG": "Mega Cap Tech",
    "META": "Mega Cap Tech",
    "AMZN": "Mega Cap Tech",
    "TSLA": "Momentum Growth",
    "NVDA": "Semiconductors",
    "AMD": "Semiconductors",
    "AVGO": "Semiconductors",
    "SMH": "Semiconductors",
    "SOXL": "Levered Semis",
    "SOXS": "Levered Semis",
    "SPY": "Index Beta",
    "QQQ": "Index Beta",
    "IWM": "Index Beta",
    "DIA": "Index Beta",
    "TLT": "Rates",
    "GLD": "Metals",
    "SLV": "Metals",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLP": "Staples",
    "XLU": "Utilities",
}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_regime_snapshot() -> dict[str, float | str]:
    """Fetch a compact market regime snapshot. Falls back cleanly when offline."""
    try:
        end = datetime.today().date() + timedelta(days=1)
        start = end - timedelta(days=90)
        data = yf.download(["SPY", "QQQ", "^VIX"], start=start, end=end, auto_adjust=True, progress=False)
        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
        spy = close["SPY"].dropna()
        qqq = close["QQQ"].dropna()
        vix = close["^VIX"].dropna()
        if spy.empty or vix.empty:
            raise ValueError("No regime data")

        spy_trend = float((spy.iloc[-1] / spy.rolling(20).mean().iloc[-1] - 1) * 100)
        qqq_trend = float((qqq.iloc[-1] / qqq.rolling(20).mean().iloc[-1] - 1) * 100) if not qqq.empty else spy_trend
        vix_last = float(vix.iloc[-1])
        vix_percentile = float((vix.rank(pct=True).iloc[-1]) * 100)
        breadth_score = float(np.clip(50 + spy_trend * 6 + qqq_trend * 4, 0, 100))
        volatility_score = float(np.clip(vix_percentile, 0, 100))
        
        # Skewness of SPY daily returns
        spy_returns = spy.pct_change().dropna()
        spy_skew = float(spy_returns.skew())
        if spy_skew < -0.15:
            skew_label = "Negative Skew (Fat Left Tail / Downside Risk Heavy)"
        elif spy_skew > 0.15:
            skew_label = "Positive Skew (Fat Right Tail / Upside Momentum Heavy)"
        else:
            skew_label = "Symmetrical / Neutral Skew"
    except Exception:
        vix_last = 18.0
        breadth_score = 58.0
        volatility_score = 45.0
        spy_trend = 0.8
        spy_skew = -0.45
        skew_label = "Negative Skew (Fat Left Tail / Downside Risk Heavy)"

    if breadth_score >= 65 and volatility_score < 70:
        regime = "Moderate Bullish Risk-On"
        risk_level = "Moderate"
    elif volatility_score >= 75:
        regime = "Volatility Expansion / Defensive"
        risk_level = "Elevated"
    elif breadth_score < 40:
        regime = "Weak Breadth Risk-Off"
        risk_level = "High"
    else:
        regime = "Neutral Compression"
        risk_level = "Balanced"

    return {
        "regime": regime,
        "risk_level": risk_level,
        "vix": vix_last,
        "breadth_score": breadth_score,
        "volatility_score": volatility_score,
        "spy_trend": spy_trend,
        "gamma_flip": "Above Spot" if breadth_score >= 55 else "Below Spot",
        "spy_skew": spy_skew,
        "skew_label": skew_label,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_position_correlations(tickers: tuple[str, ...], lookback_days: int) -> pd.DataFrame:
    clean = tuple(dict.fromkeys([clean_ticker_for_lookup(ticker) for ticker in tickers if ticker]))[:25]
    if len(clean) < 2:
        return pd.DataFrame()
    try:
        end = datetime.today().date() + timedelta(days=1)
        start = end - timedelta(days=int(lookback_days))
        data = yf.download(list(clean), start=start, end=end, auto_adjust=True, progress=False)
        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
        returns = close.ffill().pct_change().dropna(how="all")
        return returns.corr().dropna(how="all", axis=0).dropna(how="all", axis=1)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_position_betas(
    tickers: tuple[str, ...],
    benchmark: str,
    lookback_days: int,
) -> dict[str, float]:
    """Compute OLS beta for each ticker vs benchmark using daily returns. Falls back to 1.0 on failure."""
    clean = list(dict.fromkeys([clean_ticker_for_lookup(t) for t in tickers if t]))[:25]
    bench_clean = benchmark.upper()
    all_syms = list(dict.fromkeys([bench_clean] + clean))
    if len(all_syms) < 2:
        return {}
    try:
        end = datetime.today().date() + timedelta(days=1)
        start = end - timedelta(days=int(lookback_days))
        raw = yf.download(all_syms, start=start, end=end, auto_adjust=True, progress=False)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        returns = close.ffill().pct_change().dropna(how="all")
        if bench_clean not in returns.columns or len(returns) < 10:
            return {}
        bench_ret = returns[bench_clean].dropna()
        betas: dict[str, float] = {}
        for sym in clean:
            if sym not in returns.columns:
                betas[sym] = 1.0
                continue
            sym_ret = returns[sym].dropna()
            aligned = pd.concat([bench_ret, sym_ret], axis=1).dropna()
            if len(aligned) < 10:
                betas[sym] = 1.0
                continue
            x = aligned.iloc[:, 0].values
            y = aligned.iloc[:, 1].values
            cov = float(np.cov(x, y)[0, 1])
            var = float(np.var(x))
            betas[sym] = round(cov / var, 3) if var > 0 else 1.0
        return betas
    except Exception:
        return {}


def compute_weighted_portfolio_beta(
    df: pd.DataFrame,
    betas: dict[str, float],
) -> float:
    """Compute weight-averaged portfolio beta. Unmapped tickers assume beta=1."""
    if df.empty or not betas:
        return 1.0
    total_weight = max(float(df["weight_pct"].sum()), 1.0)
    weighted_sum = 0.0
    for _, row in df.iterrows():
        clean = clean_ticker_for_lookup(str(row["ticker"]))
        beta = betas.get(clean, 1.0)
        weighted_sum += float(row["weight_pct"]) * beta
    return round(weighted_sum / total_weight, 3)


def enrich_positions(
    positions: pd.DataFrame,
    regime: dict[str, float | str],
    details: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = positions.copy()
    if df.empty:
        return df

    for column in ["quantity", "mark_price", "market_value", "theta", "delta", "pl_open", "pl_day", "bp_effect"]:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["strategy"] = df.get("strategy", pd.Series(index=df.index, dtype=str)).fillna("Position Basket")
    df["expiration"] = df.get("expiration", pd.Series(index=df.index, dtype=str)).fillna("")
    df["dte"] = _numeric_series(df, "dte", 30).clip(lower=0)
    gamma_raw = _numeric_series(df, "gamma", 0)
    vega_raw = _numeric_series(df, "vega", 0)
    df["gamma"] = gamma_raw.where(gamma_raw.abs() > 0, df["delta"].abs() * 0.018)
    df["vega"] = vega_raw.where(vega_raw.abs() > 0, df["market_value"].abs() * 0.006)
    
    # Initialize default stock/equity values before applying option-leg overrides
    df["current_value"] = df["market_value"]
    df["entry_value"] = df["market_value"] - df["pl_open"]
    df["entry_value"] = df["entry_value"].where(df["market_value"].abs() > 0, df["pl_open"].abs())

    df = _apply_leg_detail_metrics(df, details)

    # Extract underlying stock price from details (equity_description row) if available
    df["underlying_price"] = 0.0
    if details is not None and not details.empty:
        eq_rows = details[details["row_type"].astype(str).eq("equity_description")].copy()
        if not eq_rows.empty:
            eq_rows["ticker_clean"] = eq_rows["underlying"].apply(clean_ticker_for_lookup)
            price_map = eq_rows.groupby("ticker_clean")["mark_price"].first().to_dict()
            df["ticker_clean_tmp"] = df["ticker"].apply(clean_ticker_for_lookup)
            df["underlying_price"] = df["ticker_clean_tmp"].map(price_map).fillna(0.0)
            df.drop(columns=["ticker_clean_tmp"], inplace=True)
    for column in [
        "leg_count",
        "gross_contracts",
        "net_contracts",
        "long_contracts",
        "short_contracts",
        "min_dte",
        "max_dte",
        "avg_dte",
        "gross_entry_value",
        "gross_current_value",
        "entry_value",
        "current_value",
        "gross_exposure",
        "net_mark_value",
        "net_trade_value",
    ]:
        if column not in df.columns:
            df[column] = 0.0
    df["delta_exposure"] = df["delta"]
    df["gamma_exposure"] = df["gamma"]
    df["theta_decay_day"] = df["theta"]
    df["vega_exposure"] = df["vega"]
    df["sector"] = df["ticker"].apply(clean_ticker_for_lookup).map(SECTOR_MAP).fillna("Single Name / Other")
    if "entry_value" not in df.columns:
        df["entry_value"] = df["market_value"].where(df["market_value"].abs() > 0, df["pl_open"].abs())
    if "current_value" not in df.columns:
        df["current_value"] = df["market_value"]
    if "gross_exposure" not in df.columns:
        df["gross_exposure"] = _gross_exposure_series(df)
    df["entry_value"] = pd.to_numeric(df["entry_value"], errors="coerce").fillna(0.0)
    df["current_value"] = pd.to_numeric(df["current_value"], errors="coerce").fillna(0.0)
    df["gross_exposure"] = pd.to_numeric(df["gross_exposure"], errors="coerce").fillna(0.0)
    df["notional_abs"] = df["gross_exposure"]
    total_current_value = max(float(df["current_value"].abs().sum()), 1.0)
    total_entry_value = max(float(df["entry_value"].abs().sum()), 1.0)
    df["weight_pct"] = df["current_value"].abs() / total_current_value * 100
    df["current_weight_pct"] = df["weight_pct"]
    df["investment_pct"] = df["entry_value"].abs() / total_entry_value * 100
    calculated_profit = df["current_value"] - df["entry_value"]
    df["profit_value"] = calculated_profit.where(
        df["entry_value"].abs().gt(0) | df["current_value"].abs().gt(0),
        df["pl_open"],
    )
    df["profit_pct"] = df["profit_value"] / df["entry_value"].abs().replace(0, np.nan) * 100
    df["profit_pct"] = df["profit_pct"].replace([np.inf, -np.inf], 0).fillna(0)
    df["profit_status"] = np.where(df["profit_value"] >= 0, "Making Money", "Losing Money")
    df["pl_quality"] = np.where(df["pl_open"] >= 0, 72, 42) + np.clip(df["pl_day"] / total_current_value * 500, -12, 12)
    df["theta_component"] = np.clip(78 + df["theta"] / max(abs(float(df["theta"].sum())), 1) * 18, 10, 95)
    df["gamma_component"] = np.clip(80 - df["gamma_exposure"].abs() / max(df["gamma_exposure"].abs().max(), 1) * 36, 10, 95)
    df["correlation_component"] = np.clip(90 - df["weight_pct"] * 1.8, 15, 95)
    regime_alignment = 72 if str(regime["risk_level"]) in {"Moderate", "Balanced"} else 48
    trend_quality = np.where(df["pl_day"] >= 0, 72, 46)
    iv_environment = 100 - float(regime["volatility_score"]) * 0.45
    liquidity = np.clip(60 + np.log1p(df["current_value"].abs()) * 3, 35, 90)

    df["risk_score"] = (
        trend_quality * 0.15
        + df["pl_quality"] * 0.15
        + df["theta_component"] * 0.10
        + df["gamma_component"] * 0.10
        + iv_environment * 0.15
        + liquidity * 0.10
        + df["correlation_component"] * 0.10
        + regime_alignment * 0.15
    ).round(0).clip(0, 100)
    df["recommendation"] = df.apply(_recommend_position, axis=1)
    df["risk_band"] = df["risk_score"].map(_risk_band)
    df = _add_position_sizing(df)
    return df.sort_values(["risk_score", "weight_pct"], ascending=[True, False])


def _apply_leg_detail_metrics(df: pd.DataFrame, details: pd.DataFrame | None) -> pd.DataFrame:
    if details is None or details.empty or "row_type" not in details.columns:
        return df

    legs = details[details["row_type"].astype(str).eq("option_leg")].copy()
    if legs.empty or "underlying" not in legs.columns:
        return df

    for column in [
        "quantity",
        "days",
        "trade_price",
        "mark_price",
        "mark_change",
        "delta",
        "theta",
        "gamma",
        "vega",
        "pl_open",
        "pl_day",
        "strike",
    ]:
        if column not in legs.columns:
            legs[column] = 0.0
        legs[column] = pd.to_numeric(legs[column], errors="coerce").fillna(0.0)

    legs["underlying"] = legs["underlying"].astype(str).str.upper()
    legs["gross_leg_exposure"] = legs["quantity"].abs() * legs["strike"].abs() * 100
    legs["gross_entry_value"] = legs["quantity"].abs() * legs["trade_price"].abs() * 100
    legs["gross_current_value"] = legs["quantity"].abs() * legs["mark_price"].abs() * 100
    legs["net_mark_value"] = legs["quantity"] * legs["mark_price"] * 100
    legs["net_trade_value"] = legs["quantity"] * legs["trade_price"] * 100
    legs["long_contracts"] = legs["quantity"].clip(lower=0)
    legs["short_contracts"] = legs["quantity"].clip(upper=0).abs()

    grouped = legs.groupby("underlying", as_index=False).agg(
        leg_count=("instrument", "count"),
        gross_contracts=("quantity", lambda values: float(values.abs().sum())),
        net_contracts=("quantity", "sum"),
        long_contracts=("long_contracts", "sum"),
        short_contracts=("short_contracts", "sum"),
        min_dte=("days", "min"),
        max_dte=("days", "max"),
        avg_dte=("days", "mean"),
        min_strike=("strike", "min"),
        max_strike=("strike", "max"),
        leg_gross_exposure=("gross_leg_exposure", "sum"),
        gross_entry_value=("gross_entry_value", "sum"),
        gross_current_value=("gross_current_value", "sum"),
        net_mark_value=("net_mark_value", "sum"),
        net_trade_value=("net_trade_value", "sum"),
        leg_delta=("delta", "sum"),
        leg_theta=("theta", "sum"),
        leg_gamma=("gamma", "sum"),
        leg_vega=("vega", "sum"),
        leg_pl_open=("pl_open", "sum"),
        leg_pl_day=("pl_day", "sum"),
    )

    enriched = df.merge(grouped, left_on="ticker", right_on="underlying", how="left")
    has_legs = enriched["leg_count"].fillna(0).gt(0)
    enriched.loc[has_legs, "delta"] = enriched.loc[has_legs, "leg_delta"]
    enriched.loc[has_legs, "theta"] = enriched.loc[has_legs, "leg_theta"]
    enriched.loc[has_legs, "gamma"] = enriched.loc[has_legs, "leg_gamma"]
    enriched.loc[has_legs, "vega"] = enriched.loc[has_legs, "leg_vega"]
    enriched.loc[has_legs, "pl_open"] = enriched.loc[has_legs, "leg_pl_open"]
    enriched.loc[has_legs, "pl_day"] = enriched.loc[has_legs, "leg_pl_day"]
    enriched.loc[has_legs, "dte"] = enriched.loc[has_legs, "min_dte"]
    enriched.loc[has_legs, "quantity"] = enriched.loc[has_legs, "net_contracts"]
    enriched.loc[has_legs, "mark_price"] = (
        enriched.loc[has_legs, "net_mark_value"] / enriched.loc[has_legs, "net_contracts"].replace(0, np.nan) / 100
    ).fillna(0.0)
    enriched.loc[has_legs, "market_value"] = enriched.loc[has_legs, "net_mark_value"]
    enriched.loc[has_legs, "notional_abs"] = enriched.loc[has_legs, "leg_gross_exposure"]
    enriched.loc[has_legs, "entry_value"] = pd.to_numeric(enriched.loc[has_legs, "net_trade_value"], errors="coerce").fillna(0.0)
    enriched.loc[has_legs, "current_value"] = pd.to_numeric(enriched.loc[has_legs, "net_mark_value"], errors="coerce").fillna(0.0)
    enriched["gross_exposure"] = pd.to_numeric(enriched.get("gross_current_value"), errors="coerce").fillna(0.0)
    enriched.loc[~has_legs, "gross_exposure"] = enriched.loc[~has_legs, "current_value"].abs()
    enriched.loc[has_legs, "gross_exposure"] = enriched.loc[has_legs, "leg_gross_exposure"]
    for column in [
        "leg_count",
        "gross_contracts",
        "net_contracts",
        "long_contracts",
        "short_contracts",
        "min_dte",
        "max_dte",
        "avg_dte",
        "min_strike",
        "max_strike",
        "leg_gross_exposure",
        "gross_entry_value",
        "gross_current_value",
        "net_mark_value",
        "net_trade_value",
    ]:
        enriched[column] = pd.to_numeric(enriched.get(column), errors="coerce").fillna(0.0)

    drop_cols = [column for column in enriched.columns if column.startswith("leg_") and column not in {"leg_count", "leg_gross_exposure"}]
    drop_cols += ["underlying"]
    return enriched.drop(columns=[column for column in drop_cols if column in enriched.columns])


def _numeric_series(df: pd.DataFrame, column: str, default: float | pd.Series) -> pd.Series:
    if column in df.columns:
        value = df[column]
    elif isinstance(default, pd.Series):
        value = default.reindex(df.index)
    else:
        value = pd.Series(default, index=df.index)
    return pd.to_numeric(value, errors="coerce").fillna(0.0)


def _gross_exposure_series(df: pd.DataFrame) -> pd.Series:
    if "leg_gross_exposure" in df.columns:
        leg_exposure = pd.to_numeric(df["leg_gross_exposure"], errors="coerce").abs().fillna(0.0)
    elif "gross_exposure" in df.columns:
        leg_exposure = pd.to_numeric(df["gross_exposure"], errors="coerce").abs().fillna(0.0)
    elif "gross_current_value" in df.columns:
        leg_exposure = pd.to_numeric(df["gross_current_value"], errors="coerce").abs().fillna(0.0)
    elif "net_trade_value" in df.columns:
        leg_exposure = pd.to_numeric(df["net_trade_value"], errors="coerce").abs().fillna(0.0)
    elif "entry_value" in df.columns:
        leg_exposure = pd.to_numeric(df["entry_value"], errors="coerce").abs().fillna(0.0)
    else:
        leg_exposure = pd.Series(0.0, index=df.index)
    market_value = df["market_value"].abs()
    contract_value = (df["quantity"].abs() * df["mark_price"].abs() * 100).fillna(0.0)
    greek_value = (df["delta"].abs() * 100 + df["theta"].abs() * 10).fillna(0.0)
    pnl_value = (df["pl_open"].abs() + df["pl_day"].abs()).fillna(0.0)
    exposure = leg_exposure.where(leg_exposure > 0, market_value)
    exposure = exposure.where(exposure > 0, contract_value)
    exposure = exposure.where(exposure > 0, greek_value)
    exposure = exposure.where(exposure > 0, pnl_value)
    exposure = exposure.where(exposure > 0, 1.0)
    return exposure


def _add_position_sizing(df: pd.DataFrame) -> pd.DataFrame:
    sized = df.copy()
    current_capital = pd.to_numeric(sized.get("current_value"), errors="coerce").abs().fillna(0.0)
    total_capital = max(float(current_capital.sum()), 1.0)
    quality = sized["risk_score"].clip(lower=5) / 100
    concentration_penalty = 1 / (1 + sized["weight_pct"] / 15)
    action_multiplier = sized["recommendation"].map(
        {
            "Add": 1.25,
            "Hold": 1.00,
            "Watch": 0.75,
            "Trim": 0.45,
            "Hedge": 0.55,
            "Redeploy": 0.25,
            "Eliminate": 0.05,
        }
    ).fillna(0.75)
    raw_target = quality * concentration_penalty * action_multiplier
    if float(raw_target.sum()) <= 0:
        raw_target = pd.Series(1.0, index=sized.index)

    target_weight = raw_target / raw_target.sum() * 100
    target_weight = target_weight.clip(lower=1.0, upper=18.0)
    target_weight = target_weight / target_weight.sum() * 100
    target_weight = _apply_action_size_caps(sized, target_weight)

    sized["target_weight_pct"] = target_weight.round(2)
    sized["current_position_value"] = current_capital.round(2)
    sized["target_position_value"] = (total_capital * target_weight / 100).round(2)
    sized["position_gap_value"] = (sized["target_position_value"] - sized["current_position_value"]).round(2)
    sized["position_gap_pct"] = (sized["target_weight_pct"] - sized["weight_pct"]).round(2)
    tolerance = np.maximum(2.0, sized["target_weight_pct"] * 0.15)
    sized["exposure_status"] = np.select(
        [
            sized["position_gap_pct"] < -tolerance,
            sized["position_gap_pct"] > tolerance,
        ],
        ["Overexposed", "Underexposed"],
        default="In Line",
    )
    return sized


def _apply_action_size_caps(df: pd.DataFrame, target_weight: pd.Series) -> pd.Series:
    capped = target_weight.copy()
    current = df["weight_pct"]
    cap_multiplier = df["recommendation"].map(
        {
            "Trim": 0.75,
            "Hedge": 0.70,
            "Redeploy": 0.35,
            "Eliminate": 0.10,
        }
    )
    has_cap = cap_multiplier.notna()
    capped.loc[has_cap] = np.minimum(capped.loc[has_cap], current.loc[has_cap] * cap_multiplier.loc[has_cap])

    deficit = 100 - float(capped.sum())
    eligible = df["recommendation"].isin(["Add", "Hold", "Watch"])
    if deficit > 0.01 and eligible.any():
        receiver = (df.loc[eligible, "risk_score"].clip(lower=10) / 100).astype(float)
        capped.loc[eligible] = capped.loc[eligible] + deficit * receiver / receiver.sum()
    elif float(capped.sum()) > 0:
        capped = capped / capped.sum() * 100

    return capped


def _risk_band(score: float) -> str:
    if score < 20:
        return "Eliminate"
    if score < 40:
        return "High Risk"
    if score < 60:
        return "Watch"
    if score < 80:
        return "Hold"
    return "Add"


def _recommend_position(row: pd.Series) -> str:
    if row["risk_score"] < 25:
        return "Eliminate"
    if row["dte"] < 10 and row["theta"] < 0 and row["pl_day"] < 0:
        return "Eliminate"
    if row["weight_pct"] >= 18 or row["gamma_component"] < 45:
        return "Trim"
    if row["theta"] < 0 and row["risk_score"] < 55:
        return "Redeploy"
    if row["risk_score"] < 45:
        return "Hedge"
    if row["risk_score"] >= 82 and row["pl_day"] >= 0:
        return "Add"
    return "Hold"


def portfolio_heat_score(df: pd.DataFrame, regime: dict[str, float | str]) -> float:
    if df.empty:
        return 0.0
    heat = (
        df["weight_pct"].max() * 1.2
        + df["gamma_exposure"].abs().sum() / max(df["delta_exposure"].abs().sum(), 1) * 10
        + max(-float(df["theta"].sum()), 0) / max(abs(float(df["market_value"].sum())), 1) * 500
        + float(regime["volatility_score"]) * 0.35
    )
    return float(np.clip(heat, 0, 100))


def build_ai_commentary(df: pd.DataFrame, totals: dict[str, float], regime: dict[str, float | str], heat: float) -> str:
    if df.empty:
        return "Upload or load a saved portfolio snapshot to generate AI portfolio commentary."

    top_sector = df.groupby("sector")["weight_pct"].sum().sort_values(ascending=False).head(1)
    sector_text = f"{top_sector.index[0]} concentration is {_fmt_pct(top_sector.iloc[0])}" if not top_sector.empty else "No dominant sector"
    theta_text = "positive theta carry" if totals["total_theta"] >= 0 else "negative theta drag"
    pressure = df[df["recommendation"].isin(["Trim", "Hedge", "Eliminate", "Redeploy"])].head(4)
    actions = "\n".join([f"- {row.ticker}: {row.recommendation} ({row.risk_band}, score {row.risk_score:.0f})" for row in pressure.itertuples()])
    if not actions:
        actions = "- Maintain current exposure; no critical pressure candidates were detected."

    return (
        f"Portfolio is operating in a {regime['regime']} tape with {regime['risk_level']} market risk. "
        f"Portfolio heat is {heat:.0f}/100, net delta is {totals['total_delta']:+.2f}, and the book has {theta_text} "
        f"of {totals['total_theta']:+.2f} per day. {sector_text}, so hidden correlation should be monitored before adding similar beta.\n\n"
        f"Recommended actions:\n{actions}"
    )


def _fmt_pct(value: float, signed: bool = False) -> str:
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:.2f}%"


def _status_from_level(value: float, warning: float, danger: float) -> str:
    if value >= danger:
        return "Danger"
    if value >= warning:
        return "Warning"
    return "Healthy"


def _status_from_abs(value: float, warning: float, danger: float) -> str:
    return _status_from_level(abs(float(value)), warning, danger)


def _near_term_dte_exposure(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    total = max(float(df["notional_abs"].sum()), 1.0)
    near = float(df.loc[df["dte"] <= 14, "notional_abs"].sum())
    return near / total * 100


def _top_contribution_pct(df: pd.DataFrame, column: str, count: int = 3) -> float:
    exposure = df[column].abs()
    total = max(float(exposure.sum()), 1.0)
    return float(exposure.sort_values(ascending=False).head(count).sum() / total * 100)


def _greek_quadrant(row: pd.Series) -> str:
    high_gamma = row["gamma_exposure"] > 0
    high_delta = row["delta_exposure"] > 0
    if high_gamma and high_delta:
        return "Danger Zone"
    if high_gamma and not high_delta:
        return "Convexity Trap"
    if not high_gamma and high_delta:
        return "Directional Exposure"
    return "Low Risk"


def _greek_commentary(
    df: pd.DataFrame,
    totals: dict[str, float],
    regime: dict[str, float | str],
    heat: float,
) -> tuple[str, list[str], list[str]]:
    delta_leader = _leader_name(df, "delta_exposure")
    gamma_leader = _leader_name(df, "gamma_exposure")
    theta_burn = df.sort_values("theta_decay_day").head(1)
    theta_name = theta_burn.iloc[0]["ticker"] if not theta_burn.empty else "No theta burn leader"
    near_term = _near_term_dte_exposure(df)
    top_sector = df.groupby("sector")["weight_pct"].sum().sort_values(ascending=False).head(1)
    sector_name = top_sector.index[0] if not top_sector.empty else "No cluster"
    sector_weight = float(top_sector.iloc[0]) if not top_sector.empty else 0.0

    commentary = (
        f"{gamma_leader} is the primary gamma driver and {delta_leader} is the primary directional driver. "
        f"The book is in a {regime['regime']} regime with portfolio heat at {heat:.0f}/100. "
        f"{sector_name} concentration is {_fmt_pct(sector_weight)}, while near-term expiry exposure is {_fmt_pct(near_term)}. "
        f"Theta profile is {'constructive' if totals['total_theta'] >= 0 else 'a drag'} at {fmt_num(totals['total_theta'])} per day."
    )

    actions = []
    trim = df[df["recommendation"].isin(["Trim", "Eliminate"])].head(2)
    if not trim.empty:
        actions.append(f"Trim or reduce {', '.join(trim['ticker'].tolist())} where convexity and concentration dominate.")
    if near_term >= 25:
        actions.append("Reduce near-term expiries or roll exposure farther out to lower gamma acceleration.")
    if sector_weight >= 35:
        actions.append(f"Cap additional {sector_name} exposure until cluster weight drops below 30.00%.")
    if totals["total_theta"] < 0:
        actions.append("Repair negative theta by closing decay-heavy structures or adding defined-risk premium.")
    if not actions:
        actions.append("Maintain current Greek posture; monitor concentration before adding correlated exposure.")

    stack = [
        f"{gamma_leader} gamma concentration",
        f"{sector_name} cluster risk at {_fmt_pct(sector_weight)}",
        f"Near-term expiry exposure at {_fmt_pct(near_term)}",
        f"{delta_leader} directional bias",
    ]
    return commentary, actions, stack


def _leader_name(df: pd.DataFrame, column: str) -> str:
    if df.empty:
        return "No position"
    ordered = df.reindex(df[column].abs().sort_values(ascending=False).index)
    return str(ordered.iloc[0]["ticker"])


# ── Scenario Shock Engine ─────────────────────────────────────────────────────

_SPY_SCENARIOS: list[float] = [-10.0, -7.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

_VIX_SHOCK_TABLE: list[tuple[float, float]] = [
    (-10.0, 30.0),
    (-7.0, 20.0),
    (-5.0, 12.0),
    (-3.0, 6.0),
    (0.0, 0.0),
    (3.0, -4.0),
    (5.0, -7.0),
    (7.0, -10.0),
    (10.0, -15.0),
]


def _vix_shock_from_spy(spy_pct: float) -> float:
    """Empirical VIX response (% change in IV) for a given SPY % move.
    Uses linear interpolation between calibrated breakpoints."""
    xs = [t[0] for t in _VIX_SHOCK_TABLE]
    ys = [t[1] for t in _VIX_SHOCK_TABLE]
    if spy_pct <= xs[0]:
        return ys[0]
    if spy_pct >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= spy_pct <= xs[i + 1]:
            frac = (spy_pct - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + frac * (ys[i + 1] - ys[i])
    return 0.0


def _reprice_position(
    row: pd.Series,
    spy_pct: float,
    vix_shock_pct: float,
    days_passed: float,
    position_betas: dict[str, float],
) -> dict:
    """Estimate per-position P/L under a given SPY move and IV shock using Greek approximation.

    P/L ≈ Delta × ΔPrice + 0.5 × Gamma × ΔPrice² + Vega × ΔIV + Theta × days
    ΔPrice is beta-adjusted to the underlying price.
    """
    ticker_clean = clean_ticker_for_lookup(str(row.get("ticker", "")))
    beta = float(position_betas.get(ticker_clean, 1.0))

    # Beta-adjusted underlying price change (in dollars)
    underlying_price = float(row.get("underlying_price", 0.0))
    if underlying_price <= 0:
        underlying_price = float(row.get("mark_price", 0.0))
    if underlying_price <= 0:
        # Fallback to strike proxy if options position
        min_s = float(row.get("min_strike", 0.0))
        max_s = float(row.get("max_strike", 0.0))
        if min_s > 0 or max_s > 0:
            underlying_price = (min_s + max_s) / 2.0 if min_s > 0 and max_s > 0 else (min_s if min_s > 0 else max_s)
    if underlying_price <= 0:
        underlying_price = float(row.get("current_value", 0.0))
    ticker_move_pct = spy_pct * beta
    price_change = underlying_price * ticker_move_pct / 100.0

    delta = float(row.get("delta", 0.0))
    gamma = float(row.get("gamma", 0.0))
    vega = float(row.get("vega", 0.0))
    theta = float(row.get("theta", 0.0))

    delta_pnl = delta * price_change
    gamma_pnl = 0.5 * gamma * (price_change ** 2)
    vega_pnl = vega * (vix_shock_pct / 100.0)
    theta_pnl = theta * days_passed
    total_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl

    return {
        "ticker": str(row.get("ticker", ticker_clean)),
        "strategy": str(row.get("strategy", "Position Basket")),
        "beta": round(beta, 3),
        "ticker_move_pct": round(ticker_move_pct, 2),
        "spy_pct": spy_pct,
        "vix_shock_pct": round(vix_shock_pct, 1),
        "delta_pnl": round(delta_pnl, 2),
        "gamma_pnl": round(gamma_pnl, 2),
        "vega_pnl": round(vega_pnl, 2),
        "theta_pnl": round(theta_pnl, 2),
        "total_pnl": round(total_pnl, 2),
    }


def _scenario_action_signal(
    scenario_pnls: list[float],
    base_pnl: float,
    current_pnl: float,
    risk_score: float,
    weight_pct: float,
) -> str:
    """Derive a scenario-aware action signal based on how a position performs across all 9 scenarios."""
    negative_count = sum(1 for v in scenario_pnls if v < 0)
    worst = min(scenario_pnls)
    best = max(scenario_pnls)

    if negative_count >= 7:
        return "Eliminate"
    if negative_count >= 5 and worst < -200:
        return "Trim"
    if risk_score < 35:
        return "Hedge"
    # Take profit: profitable now, upside capped (best scenario not much better than current)
    if current_pnl > 50 and best < current_pnl * 1.15:
        return "Take Profit"
    if weight_pct > 18 and negative_count >= 3:
        return "Trim"
    if base_pnl >= 0 and negative_count <= 2:
        return "Keep"
    if negative_count <= 4 and risk_score >= 60:
        return "Hold"
    return "Redeploy"


def _calendar_risk_checks(
    row: pd.Series,
    spy_pct: float,
    vix_shock_pct: float,
    median_theta_abs: float,
    median_vega: float,
) -> list[tuple[str, str]]:
    """Return list of (check_name, alert_level) for calendar spread positions.
    alert_level: 'Safe', 'Monitor', or 'Danger'
    """
    alerts: list[tuple[str, str]] = []
    strategy = str(row.get("strategy", ""))
    is_calendar = any(kw in strategy for kw in ("Calander", "Calendar"))
    if not is_calendar:
        return alerts

    underlying_price = float(row.get("mark_price", 0.0))
    ticker_clean = clean_ticker_for_lookup(str(row.get("ticker", "")))
    min_strike = float(row.get("min_strike", 0.0))
    max_strike = float(row.get("max_strike", 0.0))
    strike_width = abs(max_strike - min_strike)

    # Strike distance: how far underlying moves under this scenario
    beta = 1.0  # use 1.0 as conservative default here (betas passed separately)
    price_move = abs(underlying_price * spy_pct * beta / 100.0)
    if strike_width > 0:
        breach_ratio = price_move / strike_width
        if breach_ratio >= 0.8:
            alerts.append(("Short Strike Distance", "Danger"))
        elif breach_ratio >= 0.5:
            alerts.append(("Short Strike Distance", "Monitor"))
        else:
            alerts.append(("Short Strike Distance", "Safe"))
    else:
        alerts.append(("Short Strike Distance", "Monitor"))

    # Front expiry DTE decay risk
    min_dte = float(row.get("min_dte", 30.0))
    theta_abs = abs(float(row.get("theta", 0.0)))
    if min_dte < 7:
        alerts.append(("Front-Leg Expiry", "Danger"))
    elif min_dte < 14 and theta_abs > median_theta_abs:
        alerts.append(("Front-Leg Expiry", "Monitor"))
    else:
        alerts.append(("Front-Leg Expiry", "Safe"))

    # IV crush risk (calendars lose if IV drops too much — they are long vega)
    vega = float(row.get("vega", 0.0))
    if vix_shock_pct < -10 and vega > median_vega * 0.5:
        alerts.append(("IV Crush Risk", "Danger"))
    elif vix_shock_pct < -5 and vega > 0:
        alerts.append(("IV Crush Risk", "Monitor"))
    else:
        alerts.append(("IV Crush Risk", "Safe"))

    # Gamma acceleration near short strike
    gamma_component = float(row.get("gamma_component", 80.0))
    if gamma_component < 45:
        alerts.append(("Gamma Near Strike", "Danger"))
    elif gamma_component < 60:
        alerts.append(("Gamma Near Strike", "Monitor"))
    else:
        alerts.append(("Gamma Near Strike", "Safe"))

    return alerts


def _build_scenario_matrix(
    df: pd.DataFrame,
    position_betas: dict[str, float],
    vix_override: float | None,
    days_passed: float,
    details: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run all 13 SPY scenarios × all positions and return a long P/L DataFrame.
    If details (leg-level data) is provided, repricing is done at the leg level with
    directional risk bounds (capping long option losses at premium and short option
    gains at credit received) to avoid quadratic Gamma extrapolation errors.
    """
    if df.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    
    # Extract underlying stock prices from details (equity_description row) if available
    stock_prices = {}
    if details is not None and not details.empty:
        eq_rows = details[details["row_type"].astype(str).eq("equity_description")].copy()
        if not eq_rows.empty:
            eq_rows["ticker_clean"] = eq_rows["underlying"].apply(clean_ticker_for_lookup)
            stock_prices = eq_rows.groupby("ticker_clean")["mark_price"].first().to_dict()

    for spy_pct in _SPY_SCENARIOS:
        vix_shock = vix_override if vix_override is not None else _vix_shock_from_spy(spy_pct)
        
        # Check if we can do leg-level repricing
        if details is not None and not details.empty:
            details_copy = details.copy()
            details_copy["ticker_clean"] = details_copy["underlying"].apply(clean_ticker_for_lookup)
            
            for und, group in details_copy.groupby("ticker_clean"):
                ticker_df = df[df["ticker"].apply(clean_ticker_for_lookup) == und]
                if ticker_df.empty:
                    continue
                
                ticker_name = ticker_df.iloc[0]["ticker"]
                strategy_name = ticker_df.iloc[0]["strategy"]
                beta = float(position_betas.get(und, 1.0))
                ticker_move_pct = spy_pct * beta
                
                # Retrieve stock price
                underlying_price = stock_prices.get(und, 0.0)
                if underlying_price <= 0:
                    eq_in_group = group[group["row_type"] == "equity_description"]
                    if not eq_in_group.empty:
                        underlying_price = float(eq_in_group.iloc[0].get("mark_price", 0.0))
                if underlying_price <= 0:
                    opt_in_group = group[group["row_type"] == "option_leg"]
                    if not opt_in_group.empty:
                        min_s = opt_in_group["strike"].min()
                        max_s = opt_in_group["strike"].max()
                        underlying_price = (min_s + max_s) / 2.0 if min_s > 0 and max_s > 0 else (min_s if min_s > 0 else max_s)
                if underlying_price <= 0:
                    underlying_price = float(group.iloc[0].get("mark_price", 0.0))
                
                price_change = underlying_price * ticker_move_pct / 100.0
                
                tot_delta_pnl = 0.0
                tot_gamma_pnl = 0.0
                tot_vega_pnl = 0.0
                tot_theta_pnl = 0.0
                tot_total_pnl = 0.0
                
                for _, row in group.iterrows():
                    row_type = str(row.get("row_type", ""))
                    qty = float(row.get("quantity", 0.0))
                    mark = float(row.get("mark_price", 0.0))
                    
                    if row_type == "option_leg":
                        delta = float(row.get("delta", 0.0))
                        gamma = float(row.get("gamma", 0.0))
                        vega = float(row.get("vega", 0.0))
                        theta = float(row.get("theta", 0.0))
                        
                        delta_pnl = delta * price_change
                        gamma_pnl = 0.5 * gamma * (price_change ** 2)
                        vega_pnl = vega * (vix_shock / 100.0)
                        theta_pnl = theta * days_passed
                        
                        leg_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
                        
                        # Apply Directional Bounds Capping
                        bounded_pnl = leg_pnl
                        cur_val = qty * mark * 100
                        cp = str(row.get("call_put", "")).upper()
                        
                        if cp == "CALL":
                            if price_change < 0: # Down move
                                if qty > 0: # Long Call
                                    bounded_pnl = np.clip(leg_pnl, -cur_val, 0.0)
                                else: # Short Call
                                    bounded_pnl = np.clip(leg_pnl, 0.0, -cur_val)
                            else: # Up move
                                if qty > 0: # Long Call
                                    bounded_pnl = max(0.0, leg_pnl)
                                else: # Short Call
                                    bounded_pnl = min(0.0, leg_pnl)
                        elif cp == "PUT":
                            if price_change < 0: # Down move
                                if qty > 0: # Long Put
                                    bounded_pnl = max(0.0, leg_pnl)
                                else: # Short Put
                                    bounded_pnl = min(0.0, leg_pnl)
                            else: # Up move
                                if qty > 0: # Long Put
                                    bounded_pnl = np.clip(leg_pnl, -cur_val, 0.0)
                                else: # Short Put
                                    bounded_pnl = np.clip(leg_pnl, 0.0, -cur_val)
                        
                        tot_delta_pnl += delta_pnl
                        tot_gamma_pnl += gamma_pnl
                        tot_vega_pnl += vega_pnl
                        tot_theta_pnl += theta_pnl
                        tot_total_pnl += bounded_pnl
                        
                    elif row_type == "equity_description" and qty != 0:
                        leg_pnl = qty * price_change
                        tot_delta_pnl += leg_pnl
                        tot_total_pnl += leg_pnl
                
                rows.append({
                    "ticker": str(ticker_name),
                    "strategy": str(strategy_name),
                    "beta": round(beta, 3),
                    "ticker_move_pct": round(ticker_move_pct, 2),
                    "spy_pct": spy_pct,
                    "vix_shock_pct": round(vix_shock, 1),
                    "delta_pnl": round(tot_delta_pnl, 2),
                    "gamma_pnl": round(tot_gamma_pnl, 2),
                    "vega_pnl": round(tot_vega_pnl, 2),
                    "theta_pnl": round(tot_theta_pnl, 2),
                    "total_pnl": round(tot_total_pnl, 2),
                })
        else:
            for spy_pct_fallback in _SPY_SCENARIOS:
                vix_shock_fallback = vix_override if vix_override is not None else _vix_shock_from_spy(spy_pct_fallback)
                for _, row in df.iterrows():
                    result = _reprice_position(row, spy_pct_fallback, vix_shock_fallback, days_passed, position_betas)
                    rows.append(result)
                
    return pd.DataFrame(rows)


class PortfolioRiskManagementModule(FazDaneModule):
    MODULE_NAME = "Portfolio Performance & Risk Management"
    MODULE_ICON = "PR"
    MODULE_DESCRIPTION = "Daily portfolio ingestion, Greeks, concentration risk, recommendations, and AI commentary"
    TIER = 2
    SOURCE_NOTEBOOK = "FazDane Portfolio Risk Engine"
    CACHE_TTL = 300
    REQUIRES_LIVE_DATA = False
    DATA_SOURCES = ["Schwab Position Statement CSV", "Local SQLite", "yfinance"]

    def render_sidebar(self):
        st.markdown("**Daily Portfolio Load**")
        self.uploaded_files = st.file_uploader(
            "Upload Broker CSVs (Schwab / Tastytrade)",
            type=["csv"],
            accept_multiple_files=True,
            key="prm_csv_uploads",
        )
        self.load_latest_saved = st.checkbox("Use latest saved snapshot", value=True, key="prm_use_latest")
        self.auto_save = st.checkbox("Save parsed upload", value=True, key="prm_auto_save")

        if st.button("🔄 Sync 'FazDane Portfolio'", key="prm_sync_universe_btn", use_container_width=True):
            pos, det, meta, label = self._load_active_snapshot()
            if not pos.empty:
                raw_tickers = pos["ticker"].apply(clean_ticker_for_lookup).unique().tolist()
                update_fazdane_portfolio_universe(raw_tickers)
                st.success(f"Updated 'FazDane Portfolio' universe with {len(raw_tickers)} tickers!")
            else:
                st.warning("No active portfolio positions loaded to sync.")

        st.markdown("**Risk Controls**")
        self.top_n = st.slider("Focus list size", 3, 20, 8, key="prm_top_n")
        self.lookback_days = st.selectbox("Correlation Lookback", [30, 60, 90, 180, 365], index=2, key="prm_corr_days")
        self.heat_limit = st.slider("Portfolio Heat Alert", 40, 95, 70, key="prm_heat_limit")
        self.include_live_regime = st.checkbox("Fetch market regime", value=True, key="prm_live_regime")
        st.selectbox("Beta Benchmark", ["SPY", "QQQ"], key="prm_beta_benchmark", help="Benchmark index used for portfolio beta regression")

        if st.button("Refresh Risk Engine", use_container_width=True, type="primary", key="prm_refresh"):
            fetch_regime_snapshot.clear()
            fetch_position_correlations.clear()
            st.session_state.pop("prm_last_saved_hash", None)
            st.rerun()

        self._render_database_status()

    def render_main(self):
        self.render_section_header(
            "Portfolio Performance & Risk Management",
            "Institutional options portfolio command center for Greeks, concentration, decay, exits, and AI commentary.",
        )
        st.markdown("<div style='font-size:13px;color:#888;margin-bottom:12px;'>Source: 🔴 Tastytrade | 🔵 Schwab</div>", unsafe_allow_html=True)

        positions, details, metadata, source_label = self._load_active_snapshot()
        
        # Auto-update the "FazDane Portfolio" universe on new statement upload
        if self.uploaded_files and not positions.empty:
            file_hash = metadata.get("file_sha256")
            if file_hash and st.session_state.get("prm_last_synced_universe_hash") != file_hash:
                raw_tickers = positions["ticker"].apply(clean_ticker_for_lookup).unique().tolist()
                update_fazdane_portfolio_universe(raw_tickers)
                st.session_state["prm_last_synced_universe_hash"] = file_hash
        if positions.empty:
            self._render_welcome()
            self._render_history()
            return

        saved_info = None
        if self.uploaded_files and self.auto_save:
            saved_info = self._save_uploaded_snapshot_once(positions, details, metadata)

        regime = fetch_regime_snapshot() if self.include_live_regime else {
            "regime": "Manual / Offline",
            "risk_level": "Balanced",
            "vix": 0.0,
            "breadth_score": 55.0,
            "volatility_score": 45.0,
            "spy_trend": 0.0,
            "gamma_flip": "Unknown",
        }
        enriched = enrich_positions(positions, regime, details=details)
        totals = summarize_positions(enriched)
        heat = portfolio_heat_score(enriched, regime)
        commentary = build_ai_commentary(enriched, totals, regime, heat)

        # Portfolio beta (sidebar control drives benchmark selection)
        beta_benchmark = st.session_state.get("prm_beta_benchmark", "SPY")
        tickers_tuple = tuple(enriched["ticker"].tolist()) if not enriched.empty else ()
        position_betas = fetch_position_betas(tickers_tuple, beta_benchmark, int(self.lookback_days))
        portfolio_beta = compute_weighted_portfolio_beta(enriched, position_betas)

        self._render_source_banner(metadata, source_label, saved_info)
        self._render_command_metrics(enriched, totals, regime, heat, portfolio_beta, beta_benchmark)

        tabs = st.tabs(
            [
                "Executive Command Center",
                "Position Analyzer",
                "Greeks Command Center",
                "What-If Simulator",
                "Correlation Risk",
                "Theta Decay",
                "Profit Taking",
                "Capital Redeployment",
                "AI Portfolio Manager",
                "Portfolio Logs",
            ]
        )
        with tabs[0]:
            self._render_executive(enriched, totals, regime, heat, commentary)
        with tabs[1]:
            self._render_position_analyzer(enriched)
        with tabs[2]:
            self._render_greeks(enriched, totals, regime, heat)
        with tabs[3]:
            self._render_what_if_tab(enriched, totals, position_betas, portfolio_beta, beta_benchmark, details=details, regime=regime)
        with tabs[4]:
            self._render_correlation(enriched)
        with tabs[5]:
            self._render_theta(enriched)
        with tabs[6]:
            self._render_profit_taking(enriched)
        with tabs[7]:
            self._render_redeployment(enriched)
        with tabs[8]:
            self._render_ai_manager(enriched, totals, regime, heat, commentary, metadata)
        with tabs[9]:
            self._render_logs_tab(metadata)

    def _render_logs_tab(self, metadata: dict):
        st.markdown("### Portfolio Daily Logs")
        
        import base64
        from st_img_pastebutton import paste
        
        # Load logs from DB
        logs_df = get_portfolio_logs()
        
        # Handle edit state loading
        edit_id = st.session_state.get("prm_edit_log_id")
        log_to_edit = None
        if edit_id and not logs_df.empty:
            matching = logs_df[logs_df["log_id"] == edit_id]
            if not matching.empty:
                log_to_edit = matching.iloc[0]
                
        # Initialize the current images list in session state
        if "prm_current_images" not in st.session_state or st.session_state.get("prm_loaded_edit_id") != edit_id:
            st.session_state["prm_loaded_edit_id"] = edit_id
            if edit_id and not logs_df.empty and log_to_edit is not None:
                images = []
                if "image_data" in log_to_edit and log_to_edit["image_data"] is not None:
                    if isinstance(log_to_edit["image_data"], bytes) and len(log_to_edit["image_data"]) > 0:
                        images.append(log_to_edit["image_data"])
                try:
                    db_images = get_portfolio_log_images().get(edit_id, [])
                    images.extend(db_images)
                except Exception:
                    pass
                st.session_state["prm_current_images"] = images
            else:
                st.session_state["prm_current_images"] = []
        
        # 1. Input Form
        form_title = "✍️ Edit Portfolio Log Entry" if log_to_edit is not None else "✍️ Add Daily Portfolio Log Entry"
        with st.container(border=True):
            st.markdown(f"**{form_title}**")
            
            # Default values
            if log_to_edit is not None:
                default_date = datetime.strptime(log_to_edit["log_date"], "%Y-%m-%d")
                default_category = log_to_edit["category"]
                default_content = log_to_edit["content"]
                default_snippet = log_to_edit["snippet"] if "snippet" in log_to_edit and log_to_edit["snippet"] is not None else ""
            else:
                active_date_str = metadata.get("snapshot_date")
                if active_date_str:
                    try:
                        default_date = datetime.strptime(active_date_str, "%Y-%m-%d")
                    except Exception:
                        default_date = datetime.today()
                else:
                    default_date = datetime.today()
                default_category = "What Happened"
                default_content = ""
                default_snippet = ""
                
            col_date, col_cat = st.columns(2)
            with col_date:
                log_date = st.date_input("Target Log Date", value=default_date, key="prm_log_date_input")
            with col_cat:
                categories = ["What Happened", "What Can Improve", "Market Context", "Action Items", "General Notes"]
                cat_idx = categories.index(default_category) if default_category in categories else 0
                category = st.selectbox("Log Category", options=categories, index=cat_idx, key="prm_log_category_input")
                
            content = st.text_area("What's on your mind? (Supports Markdown)", value=default_content, height=120, key="prm_log_content_input")
            snippet = st.text_area("Code/Data Snippet (Optional, e.g. terminal output, JSON, trade detail)", value=default_snippet, height=100, key="prm_log_snippet_input")
            
            # Use dynamic keys for uploader/paste to allow resetting
            paste_counter = st.session_state.get("prm_paste_counter", 0)
            uploaded_image = st.file_uploader("Upload Screenshot File (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], key=f"prm_log_image_upload_{paste_counter}")
            
            st.markdown("<p style='font-size:13px;color:#94a3b8;margin-bottom:4px;'>Or paste screenshot directly from clipboard:</p>", unsafe_allow_html=True)
            pasted_image_data = paste(label="📋 Click to Paste Clipboard Image", key=f"prm_log_image_paste_{paste_counter}")
            
            # Check if new image uploaded
            if uploaded_image is not None:
                image_bytes = uploaded_image.read()
                st.session_state["prm_current_images"].append(image_bytes)
                st.session_state["prm_paste_counter"] = paste_counter + 1
                st.rerun()
                
            # Check if new image pasted
            if pasted_image_data is not None:
                try:
                    header, encoded = pasted_image_data.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                    st.session_state["prm_current_images"].append(image_bytes)
                    st.session_state["prm_paste_counter"] = paste_counter + 1
                    st.rerun()
                except Exception:
                    pass
            
            # Display attached screenshots
            current_images = st.session_state.get("prm_current_images", [])
            if current_images:
                st.markdown("<p style='font-size:14px;font-weight:bold;margin-top:10px;margin-bottom:6px;'>Attached Screenshots:</p>", unsafe_allow_html=True)
                for idx, img_bytes in enumerate(current_images):
                    col_img, col_act = st.columns([5, 1])
                    with col_img:
                        st.image(img_bytes, caption=f"Screenshot #{idx + 1}", width=250)
                    with col_act:
                        st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)
                        if st.button("🗑️ Remove", key=f"prm_remove_img_{idx}"):
                            st.session_state["prm_current_images"].pop(idx)
                            st.rerun()
            
            sub_col1, sub_col2 = st.columns([1, 6])
            with sub_col1:
                submit_button = st.button("Save Log Entry", type="primary", key="prm_save_log_btn")
            with sub_col2:
                if log_to_edit is not None:
                    cancel_button = st.button("Cancel Edit", key="prm_cancel_edit_btn")
                    if cancel_button:
                        st.session_state.pop("prm_edit_log_id", None)
                        st.session_state.pop("prm_loaded_edit_id", None)
                        st.session_state.pop("prm_current_images", None)
                        st.session_state["prm_paste_counter"] = paste_counter + 1
                        st.rerun()
                        
            if submit_button:
                if not content.strip():
                    st.error("Log content cannot be empty.")
                else:
                    log_date_str = log_date.strftime("%Y-%m-%d")
                    run_id = metadata.get("run_id") if log_to_edit is None else log_to_edit["run_id"]
                    
                    save_portfolio_log(
                        log_date=log_date_str,
                        category=category,
                        content=content,
                        log_id=edit_id,
                        run_id=run_id,
                        image_data=None,
                        clear_image=False,
                        snippet=snippet if snippet.strip() else None,
                        images_list=current_images,
                    )
                    st.session_state.pop("prm_edit_log_id", None)
                    st.session_state.pop("prm_loaded_edit_id", None)
                    st.session_state.pop("prm_current_images", None)
                    st.session_state["prm_paste_counter"] = paste_counter + 1
                    st.success("Portfolio log saved successfully!")
                    st.rerun()

        # 2. Insights counters
        if not logs_df.empty:
            total_logs = len(logs_df)
            improvements = len(logs_df[logs_df["category"] == "What Can Improve"])
            happened = len(logs_df[logs_df["category"] == "What Happened"])
            
            try:
                current_month_str = datetime.today().strftime("%Y-%m")
                logs_this_month = len(logs_df[logs_df["log_date"].str.startswith(current_month_str)])
            except Exception:
                logs_this_month = 0
                
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Logs", f"{total_logs}")
            c2.metric("What Happened", f"{happened}")
            c3.metric("What Can Improve", f"{improvements}", delta="Opportunities" if improvements > 0 else None, delta_color="normal")
            c4.metric("Logs This Month", f"{logs_this_month}")
            
        st.markdown("---")
        
        # 3. Filtering & View
        st.markdown("### Log Timeline & Reporting")
        
        if logs_df.empty:
            st.info("No logs saved yet. Add a log entry above to populate your timeline!")
            return
        
        # Get all images for logs timeline
        all_images_dict = {}
        try:
            all_images_dict = get_portfolio_log_images()
        except Exception:
            pass
            
        col_search, col_filter = st.columns([2, 1])
        with col_search:
            search_query = st.text_input("🔍 Search logs...", placeholder="Type to search...", key="prm_logs_search")
        with col_filter:
            cat_filter = st.multiselect("Filter by Category", options=["What Happened", "What Can Improve", "Market Context", "Action Items", "General Notes"], key="prm_logs_cat_filter")
            
        view_period = st.radio("Group/Report logs by:", ["Day", "Week", "Month", "Year"], horizontal=True, key="prm_logs_period")
        
        # Apply search and category filters
        filtered_df = logs_df.copy()
        if search_query:
            filtered_df = filtered_df[
                filtered_df["content"].str.contains(search_query, case=False) |
                filtered_df["category"].str.contains(search_query, case=False)
            ]
        if cat_filter:
            filtered_df = filtered_df[filtered_df["category"].isin(cat_filter)]
            
        if filtered_df.empty:
            st.warning("No logs match the current search or filter criteria.")
            return

        filtered_df["log_date_dt"] = pd.to_datetime(filtered_df["log_date"])
        
        cat_styles = {
            "What Happened": ("#3ab54a", "rgba(58, 181, 74, 0.12)"),
            "What Can Improve": ("#ef4444", "rgba(239, 68, 68, 0.12)"),
            "Market Context": ("#facc15", "rgba(250, 204, 21, 0.12)"),
            "Action Items": ("#a78bfa", "rgba(167, 139, 250, 0.12)"),
            "General Notes": ("#93c5fd", "rgba(147, 197, 253, 0.12)"),
        }
        
        if view_period == "Day":
            grouped = filtered_df.groupby("log_date", sort=False)
            for date_str, group in grouped:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                st.markdown(f"#### 📅 {dt.strftime('%A, %B %d, %Y')}")
                for _, row in group.iterrows():
                    self._render_log_item_card(row, cat_styles, all_images_dict=all_images_dict)
                    
        elif view_period == "Week":
            filtered_df["week_start"] = filtered_df["log_date_dt"].dt.to_period("W").dt.start_time
            grouped = filtered_df.groupby("week_start", sort=False)
            for week_start, group in grouped:
                week_end = week_start + pd.Timedelta(days=6)
                st.markdown(f"#### 🗓️ Week of {week_start.strftime('%B %d, %Y')} to {week_end.strftime('%B %d, %Y')}")
                for _, row in group.iterrows():
                    self._render_log_item_card(row, cat_styles, show_date=True, all_images_dict=all_images_dict)
                    
        elif view_period == "Month":
            filtered_df["year_month"] = filtered_df["log_date_dt"].dt.to_period("M")
            grouped = filtered_df.groupby("year_month", sort=False)
            for month_period, group in grouped:
                st.markdown(f"#### 📅 {month_period.start_time.strftime('%B %Y')}")
                for _, row in group.iterrows():
                    self._render_log_item_card(row, cat_styles, show_date=True, all_images_dict=all_images_dict)
                    
        elif view_period == "Year":
            filtered_df["year"] = filtered_df["log_date_dt"].dt.to_period("Y")
            grouped = filtered_df.groupby("year", sort=False)
            for year_period, group in grouped:
                st.markdown(f"#### 🗓️ Year {year_period.start_time.strftime('%Y')}")
                for _, row in group.iterrows():
                    self._render_log_item_card(row, cat_styles, show_date=True, all_images_dict=all_images_dict)

    def _render_log_item_card(self, row: pd.Series, cat_styles: dict, show_date: bool = False, all_images_dict: dict | None = None):
        cat = row["category"]
        color, bg = cat_styles.get(cat, ("#e2e8f0", "rgba(226, 232, 240, 0.15)"))
        
        date_badge = f"<span style='background-color:#1e3a5f;color:#e2e8f0;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;margin-right:8px;'>{row['log_date']}</span>" if show_date else ""
        run_badge = f"<span style='background-color:#1e3a5f;color:#94a3b8;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;margin-right:8px;'>Run: {row['run_id']}</span>" if row["run_id"] else ""
        
        st.markdown(
            f"""<div style="background-color:#152847; border:1px solid #1e3a5f; border-left:4px solid {color}; border-radius:8px; padding:12px 16px; margin-bottom:10px;">
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:6px;">
<div>
{date_badge}
<span style="background-color:{bg}; color:{color}; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; border:1px solid {color};">
{cat.upper()}
</span>
{run_badge}
</div>
<span style="color:#94a3b8; font-size:11px;">Updated: {row['updated_at'][:16].replace('T', ' ')}</span>
</div>
<div style="color:#e2e8f0; font-size:13.5px; line-height:1.6; white-space:pre-wrap; margin-bottom:10px;">{row['content']}</div>
</div>""",
            unsafe_allow_html=True,
        )
        
        # Render legacy + multiple images
        log_images = []
        if "image_data" in row and row["image_data"] is not None:
            if isinstance(row["image_data"], bytes) and len(row["image_data"]) > 0:
                log_images.append(row["image_data"])
                
        if all_images_dict is not None:
            db_images = all_images_dict.get(row["log_id"], [])
        else:
            try:
                db_images = get_portfolio_log_images().get(row["log_id"], [])
            except Exception:
                db_images = []
        log_images.extend(db_images)
        
        for img in log_images:
            st.image(img, use_container_width=True)
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            
        # Check if there is snippet data and render it
        if "snippet" in row and row["snippet"] is not None and str(row["snippet"]).strip() != "":
            st.code(row["snippet"])
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        
        col_btn1, col_btn2, _ = st.columns([1, 1, 15])
        with col_btn1:
            if st.button("Edit", key=f"prm_edit_btn_{row['log_id']}"):
                st.session_state["prm_edit_log_id"] = row["log_id"]
                st.rerun()
        with col_btn2:
            if st.button("Delete", key=f"prm_delete_btn_{row['log_id']}"):
                delete_portfolio_log(row["log_id"])
                st.success("Log entry deleted.")
                st.rerun()


    def _load_active_snapshot(self) -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
        positions, details, metadata, label = pd.DataFrame(), pd.DataFrame(), {}, "No snapshot"
        if self.uploaded_files:
            file_tuples = [(f.getvalue(), f.name) for f in self.uploaded_files]
            positions, details, metadata = parse_uploaded_files(file_tuples)
            label = "Uploaded broker CSV(s)"
        elif self.load_latest_saved:
            latest_positions, latest_metadata = get_latest_portfolio_positions()
            if latest_metadata:
                latest_details, _ = get_latest_portfolio_details()
                positions, details, metadata = latest_positions, latest_details, latest_metadata
                label = "Latest saved snapshot"

        if not positions.empty and "ticker" in positions.columns:
            positions["ticker"] = positions["ticker"].apply(format_ticker_for_display)
        if not details.empty and "underlying" in details.columns:
            details["underlying"] = details["underlying"].apply(format_ticker_for_display)

        # On-the-fly strategy classification for loaded Tastytrade database positions
        if not positions.empty and not details.empty:
            tasty_mask = positions["ticker"].str.contains("🔴") & positions["account_group"].astype(str).str.contains("5WT|TASTY", case=False)
            if tasty_mask.any():
                option_legs = details[details["row_type"] == "option_leg"]
                if not option_legs.empty:
                    strategies = {}
                    for und, group in option_legs.groupby("underlying"):
                        strategies[und] = classify_option_strategy(group)
                    positions.loc[tasty_mask, "account_group"] = positions.loc[tasty_mask, "ticker"].map(strategies).fillna("Position Basket")

        return positions, details, metadata, label

    def _save_uploaded_snapshot_once(self, positions: pd.DataFrame, details: pd.DataFrame, metadata: dict) -> dict | None:
        file_hash = metadata.get("file_sha256")
        if file_hash and st.session_state.get("prm_last_saved_hash") == file_hash:
            return st.session_state.get("prm_last_save_info")
        try:
            saved = save_portfolio_snapshot(positions, metadata, details=details)
            st.session_state["prm_last_saved_hash"] = file_hash
            st.session_state["prm_last_save_info"] = saved
            return saved
        except Exception as exc:
            st.warning(f"Snapshot parsed, but database save failed: {exc}")
            return None

    def _render_source_banner(self, metadata: dict, source_label: str, saved_info: dict | None):
        saved_text = f" | Saved run {saved_info['run_id']}" if saved_info else ""
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg, rgba(21,40,71,0.94), rgba(15,23,42,0.92));
                border:1px solid {BRAND['grid']};border-left:4px solid {BRAND['green']};
                border-radius:8px;padding:14px 18px;margin:8px 0 16px 0;">
                <div style="color:{BRAND['green']};font-size:15px;font-weight:700;">{source_label}</div>
                <div style="color:{BRAND['text']};font-size:13px;margin-top:4px;">{metadata.get('source_file', 'Unknown file')}</div>
                <div style="color:{BRAND['muted']};font-size:12px;margin-top:3px;">Snapshot: {metadata.get('snapshot_ts', '')}{saved_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_command_metrics(
        self,
        df: pd.DataFrame,
        totals: dict[str, float],
        regime: dict[str, float | str],
        heat: float,
        portfolio_beta: float = 1.0,
        beta_benchmark: str = "SPY",
    ):
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Portfolio Value", fmt_money(totals["total_market_value"]))
        c2.metric("Open P/L", fmt_money(totals["total_pl_open"]), fmt_money(totals["total_pl_day"]))
        c3.metric("Net Delta", fmt_num(totals["total_delta"]))
        c4.metric("Net Theta", fmt_num(totals["total_theta"]))
        c5.metric("Portfolio Heat", f"{heat:.0f}/100", "Alert" if heat >= self.heat_limit else "Normal")
        c6.metric("Market Regime", str(regime["risk_level"]), f"VIX {float(regime['vix']):.1f}" if float(regime["vix"]) else None)
        beta_delta = "Defensive" if portfolio_beta < 0.85 else ("Aggressive" if portfolio_beta > 1.25 else "Market-Like")
        c7.metric(f"β vs {beta_benchmark}", f"{portfolio_beta:.2f}", beta_delta)

    def _render_executive(self, df: pd.DataFrame, totals: dict[str, float], regime: dict[str, float | str], heat: float, commentary: str):
        left, right = st.columns([1.35, 1.0])
        with left:
            st.markdown("### Exposure Treemap")
            treemap_df = df.copy()
            treemap_df["weight_pct_label"] = treemap_df["weight_pct"].map(_fmt_pct)
            treemap_df["investment_pct_label"] = treemap_df["investment_pct"].map(_fmt_pct)
            fig = px.treemap(
                treemap_df,
                path=["sector", "ticker"],
                values="notional_abs",
                color="risk_score",
                color_continuous_scale=[[0, BRAND["red"]], [0.55, BRAND["yellow"]], [1, BRAND["green"]]],
                custom_data=[
                    "recommendation",
                    "delta_exposure",
                    "theta_decay_day",
                    "weight_pct_label",
                    "investment_pct_label",
                    "entry_value",
                    "current_value",
                    "gross_exposure",
                ],
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Current Value: $%{customdata[6]:,.2f}<br>"
                    "Current Weight: %{customdata[3]}<br>"
                    "Entry Value: $%{customdata[5]:,.2f}<br>"
                    "Investment Weight: %{customdata[4]}<br>"
                    "Gross Leg Exposure: $%{customdata[7]:,.2f}<br>"
                    "Recommendation: %{customdata[0]}<br>"
                    "Delta Exposure: %{customdata[1]:.2f}<br>"
                    "Theta / Day: %{customdata[2]:.2f}<extra></extra>"
                )
            )
            style_figure(fig, height=470)
            st.plotly_chart(fig, use_container_width=True, theme=None)

        with right:
            st.markdown("### AI Summary")
            st.info(commentary)
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=heat,
                title={"text": "Portfolio Heat"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": BRAND["yellow"] if heat >= self.heat_limit else BRAND["green"]},
                    "steps": [
                        {"range": [0, 45], "color": "rgba(58,181,74,0.25)"},
                        {"range": [45, 70], "color": "rgba(250,204,21,0.25)"},
                        {"range": [70, 100], "color": "rgba(239,68,68,0.30)"},
                    ],
                },
            ))
            style_figure(gauge, height=280)
            st.plotly_chart(gauge, use_container_width=True, theme=None)

        st.markdown("### Recommendation Mix")
        self._render_recommendation_kpis(df)
        self._render_recommendation_ticker_boards(df)
        self._render_position_sizing_leaderboard(df)

    def _render_recommendation_kpis(self, df: pd.DataFrame):
        actions = ["Hold", "Trim", "Add", "Hedge", "Redeploy", "Eliminate"]
        cols = st.columns(6)
        for index, action in enumerate(actions):
            action_df = df[df["recommendation"] == action].sort_values("risk_score", ascending=False)
            tickers = ", ".join(action_df["ticker"].head(4).tolist()) if not action_df.empty else "None"
            avg_score = action_df["risk_score"].mean() if not action_df.empty else 0
            color = ACTION_COLORS.get(action, BRAND["blue"])
            with cols[index]:
                st.markdown(
                    f"""
                    <div style="background:rgba(21,40,71,0.72);border:1px solid {BRAND['grid']};
                        border-top:4px solid {color};border-radius:8px;padding:12px 12px;min-height:128px;">
                        <div style="color:{color};font-size:12px;font-weight:900;text-transform:uppercase;">{action}</div>
                        <div style="color:{BRAND['text']};font-size:30px;font-weight:900;line-height:1.15;margin-top:4px;">
                            {len(action_df)}
                        </div>
                        <div style="color:{BRAND['muted']};font-size:11px;margin-top:2px;">Avg score {avg_score:.0f}</div>
                        <div style="color:{BRAND['text']};font-size:11px;line-height:1.35;margin-top:8px;">{tickers}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    def _render_position_analyzer(self, df: pd.DataFrame):
        st.markdown("### Position Analyzer")
        position_view = df.copy()

        # Render Net Profit by Ticker horizontal bar chart
        chart_data = position_view.sort_values("profit_value").copy()
        chart_data["profit_label"] = chart_data["profit_value"].map(lambda value: f"${value:+,.2f}")
        chart_data["direction"] = np.where(chart_data["profit_value"] >= 0, "Gain", "Loss")
        
        fig = px.bar(
            chart_data,
            x="profit_value",
            y="ticker",
            orientation="h",
            text="profit_label",
            color="direction",
            color_discrete_map={"Gain": BRAND["green"], "Loss": BRAND["red"]},
            labels={"profit_value": "Net Profit ($)", "ticker": "Ticker"},
            title="Net Profit by Ticker Summary",
        )
        fig.add_vline(x=0, line_color=BRAND["muted"])
        fig.update_traces(textposition="outside", cliponaxis=False)
        style_figure(fig, height=max(280, 24 * len(chart_data) + 100))
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

        position_view["Source"] = position_view["ticker"].apply(get_broker_dot)
        position_view["ticker"] = position_view["ticker"].apply(clean_ticker_for_lookup)
        position_view["weight_pct_label"] = position_view["weight_pct"].map(_fmt_pct)
        position_view["target_weight_pct_label"] = position_view["target_weight_pct"].map(_fmt_pct)
        position_view["investment_pct_label"] = position_view["investment_pct"].map(_fmt_pct)
        position_view["profit_pct_label"] = position_view["profit_pct"].map(lambda value: _fmt_pct(value, signed=True))
        display_cols = [
            "Source",
            "ticker",
            "strategy",
            "sector",
            "leg_count",
            "gross_contracts",
            "long_contracts",
            "short_contracts",
            "net_contracts",
            "delta_exposure",
            "gamma_exposure",
            "theta_decay_day",
            "vega_exposure",
            "dte",
            "max_dte",
            "entry_value",
            "current_value",
            "investment_pct_label",
            "profit_value",
            "profit_pct_label",
            "profit_status",
            "pl_open",
            "pl_day",
            "risk_score",
            "risk_band",
            "recommendation",
            "weight_pct_label",
            "target_weight_pct_label",
            "current_position_value",
            "target_position_value",
            "position_gap_value",
            "exposure_status",
        ]
        display = position_view[display_cols].copy()

        def color_profit(value):
            try:
                val = float(value)
                if val > 0:
                    return "color: #bbf7d0; font-weight: 800;"
                elif val < 0:
                    return "color: #fecaca; font-weight: 800;"
                return "color: #fef08a; font-weight: 800;"
            except (ValueError, TypeError):
                return ""

        styled_display = display.style.map(color_profit, subset=["profit_value", "pl_open", "pl_day"])

        st.dataframe(
            styled_display,
            use_container_width=True,
            hide_index=True,
            height=min(820, max(460, 36 * (len(display) + 1))),
            column_config={
                "Source": st.column_config.TextColumn("Source", width="small"),
                "weight_pct_label": st.column_config.TextColumn("Current Wt"),
                "target_weight_pct_label": st.column_config.TextColumn("Target Wt"),
                "leg_count": st.column_config.NumberColumn("Legs", format="%.0f"),
                "gross_contracts": st.column_config.NumberColumn("Gross Contracts", format="%.0f"),
                "long_contracts": st.column_config.NumberColumn("Long", format="%.0f"),
                "short_contracts": st.column_config.NumberColumn("Short", format="%.0f"),
                "net_contracts": st.column_config.NumberColumn("Net Contracts", format="%.0f"),
                "entry_value": st.column_config.NumberColumn("Entry Value", format="$%.2f"),
                "current_value": st.column_config.NumberColumn("Current Value", format="$%.2f"),
                "investment_pct_label": st.column_config.TextColumn("% Total Investment"),
                "profit_value": st.column_config.NumberColumn("Net Profit", format="$%.2f"),
                "profit_pct_label": st.column_config.TextColumn("Profit %"),
                "profit_status": st.column_config.TextColumn("P/L Status"),
                "current_position_value": st.column_config.NumberColumn("Current Size", format="$%.2f"),
                "target_position_value": st.column_config.NumberColumn("Should Be", format="$%.2f"),
                "position_gap_value": st.column_config.NumberColumn("Add / Reduce", format="$%.2f"),
            },
        )

    def _render_recommendation_ticker_boards(self, df: pd.DataFrame):
        st.markdown("### Recommendation Tickers")
        hold = df[df["recommendation"] == "Hold"].sort_values("risk_score", ascending=False)
        trim = df[df["recommendation"] == "Trim"].sort_values(["position_gap_value", "weight_pct"], ascending=[True, False])
        add = df[df["recommendation"] == "Add"].sort_values("risk_score", ascending=False)
        risk_actions = df[df["recommendation"].isin(["Hedge", "Redeploy", "Eliminate"])].sort_values("risk_score")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            self._ticker_board("Hold", hold, BRAND["blue"])
        with c2:
            self._ticker_board("Trim", trim, ACTION_COLORS["Trim"])
        with c3:
            self._ticker_board("Add", add, BRAND["green"])
        with c4:
            self._ticker_board("Risk Actions", risk_actions, BRAND["red"])

    def _ticker_board(self, title: str, data: pd.DataFrame, color: str):
        rows = []
        for row in data.head(self.top_n).itertuples():
            rows.append(
                f"<div style='display:flex;justify-content:space-between;gap:12px;margin:5px 0;'>"
                f"<span style='color:{BRAND['text']};font-weight:700;'>{row.ticker}</span>"
                f"<span style='color:{color};font-weight:700;'>{row.risk_score:.0f}</span>"
                f"</div>"
            )
        body = "".join(rows) if rows else f"<div style='color:{BRAND['muted']};font-size:12px;'>No tickers</div>"
        st.markdown(
            f"""
            <div style="background:rgba(21,40,71,0.58);border:1px solid {BRAND['grid']};
                border-radius:8px;padding:12px 14px;min-height:170px;">
                <div style="color:{color};font-weight:800;font-size:13px;margin-bottom:8px;">{title}</div>
                {body}
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_position_sizing_leaderboard(self, df: pd.DataFrame):
        st.markdown("### Position Sizing Leaderboard")
        st.caption("Target size uses current gross capital, position risk score, recommendation, and concentration penalty.")
        board = df.sort_values("position_gap_value").copy()
        self._render_position_sizing_chart(board)
        board["Source"] = board["ticker"].apply(get_broker_dot)
        board["ticker"] = board["ticker"].apply(clean_ticker_for_lookup)
        board["current_weight_label"] = board["weight_pct"].map(_fmt_pct)
        board["target_weight_label"] = board["target_weight_pct"].map(_fmt_pct)
        board["profit_pct_label"] = board["profit_pct"].map(lambda value: _fmt_pct(value, signed=True))
        display_cols = [
            "Source",
            "ticker",
            "recommendation",
            "exposure_status",
            "current_weight_label",
            "target_weight_label",
            "profit_value",
            "profit_pct_label",
            "current_position_value",
            "target_position_value",
            "position_gap_value",
            "risk_score",
        ]
        st.dataframe(
            self._style_position_sizing_table(board[display_cols]),
            use_container_width=True,
            hide_index=True,
            height=min(620, max(360, 36 * (len(board) + 1))),
            column_config={
                "Source": st.column_config.TextColumn("Source", width="small"),
                "ticker": st.column_config.TextColumn("Ticker"),
                "recommendation": st.column_config.TextColumn("Action"),
                "exposure_status": st.column_config.TextColumn("Exposure"),
                "current_weight_label": st.column_config.TextColumn("Current Wt"),
                "target_weight_label": st.column_config.TextColumn("Target Wt"),
                "profit_value": st.column_config.NumberColumn("Net Profit", format="$%.2f"),
                "profit_pct_label": st.column_config.TextColumn("Profit %"),
                "current_position_value": st.column_config.NumberColumn("Current Size", format="$%.2f"),
                "target_position_value": st.column_config.NumberColumn("Should Be", format="$%.2f"),
                "position_gap_value": st.column_config.NumberColumn("Add / Reduce", format="$%.2f"),
                "risk_score": st.column_config.NumberColumn("Risk Score", format="%.0f"),
            },
        )

    def _style_position_sizing_table(self, data: pd.DataFrame):
        def color_status(value: str) -> str:
            if value == "Overexposed":
                return "background-color: rgba(239,68,68,0.28); color: #fecaca; font-weight: 800;"
            if value == "Underexposed":
                return "background-color: rgba(58,181,74,0.22); color: #bbf7d0; font-weight: 800;"
            return "background-color: rgba(147,197,253,0.18); color: #bfdbfe; font-weight: 800;"

        def color_gap(value: float) -> str:
            if value < 0:
                return "color: #fecaca; font-weight: 800;"
            if value > 0:
                return "color: #bbf7d0; font-weight: 800;"
            return "color: #bfdbfe; font-weight: 800;"

        return (
            data.style
            .map(color_status, subset=["exposure_status"])
            .map(color_gap, subset=["position_gap_value"])
            .set_properties(
                subset=["ticker", "recommendation"],
                **{"font-weight": "800", "color": "#e2e8f0"},
            )
        )

    def _render_position_sizing_chart(self, board: pd.DataFrame):
        visual = board.copy().sort_values("position_gap_pct")
        visual["gap_direction"] = np.where(visual["position_gap_pct"] >= 0, "Add", "Reduce")
        visual["gap_label"] = visual["position_gap_pct"].map(lambda value: _fmt_pct(value, signed=True))
        left, right = st.columns([1.15, 1.0])
        with left:
            fig = px.bar(
                visual,
                x="position_gap_pct",
                y="ticker",
                orientation="h",
                text="gap_label",
                color="gap_direction",
                color_discrete_map={"Add": BRAND["green"], "Reduce": BRAND["red"]},
                custom_data=[
                    "gap_label",
                    "current_position_value",
                    "target_position_value",
                    "position_gap_value",
                    "exposure_status",
                ],
                labels={"position_gap_pct": "Target Gap (%)", "ticker": "Ticker"},
                title="Add / Reduce Gap by Position",
            )
            fig.add_vline(x=0, line_color=BRAND["muted"], line_width=1)
            fig.update_traces(
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Target Gap: %{customdata[0]}<br>"
                    "Current Size: $%{customdata[1]:,.2f}<br>"
                    "Should Be: $%{customdata[2]:,.2f}<br>"
                    "Add / Reduce: $%{customdata[3]:,.2f}<br>"
                    "Exposure: %{customdata[4]}<extra></extra>"
                )
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            style_figure(fig, height=max(360, 30 * len(visual) + 90))
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

        with right:
            compare = visual.melt(
                id_vars=["ticker", "exposure_status"],
                value_vars=["weight_pct", "target_weight_pct"],
                var_name="weight_type",
                value_name="weight",
            )
            compare["weight_type"] = compare["weight_type"].map(
                {"weight_pct": "Current", "target_weight_pct": "Target"}
            )
            fig = px.bar(
                compare,
                x="ticker",
                y="weight",
                color="weight_type",
                barmode="group",
                color_discrete_map={"Current": BRAND["blue"], "Target": BRAND["green"]},
                hover_data=["exposure_status"],
                labels={"weight": "Weight (%)", "ticker": "Ticker"},
                title="Current vs Target Weight",
            )
            fig.update_traces(hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.2f}%<extra></extra>")
            fig.update_yaxes(ticksuffix="%")
            style_figure(fig, height=max(360, 30 * len(visual) + 90))
            st.plotly_chart(fig, use_container_width=True, theme=None)

    def _render_greeks(self, df: pd.DataFrame, totals: dict[str, float], regime: dict[str, float | str], heat: float):
        st.markdown("### Portfolio Risk Intelligence Engine")
        self._render_greek_risk_kpis(df, totals, regime, heat)

        left, right = st.columns([1.25, 1.0])
        with left:
            self._render_greek_leader_charts(df)
        with right:
            self._render_greek_brain(df, totals, regime, heat)

        st.markdown("### Delta / Gamma Risk Map")
        self._render_greek_bubble_map(df)

        c1, c2 = st.columns([1.15, 1.0])
        with c1:
            self._render_greek_concentration(df)
            self._render_expiration_risk_heatmap(df)
        with c2:
            self._render_what_if_panel(df, totals)
            self._render_behavior_metrics(df, totals)

    def _render_greek_risk_kpis(self, df: pd.DataFrame, totals: dict[str, float], regime: dict[str, float | str], heat: float):
        near_term = _near_term_dte_exposure(df)
        top_sector = df.groupby("sector")["weight_pct"].sum().sort_values(ascending=False).head(1)
        sector_name = top_sector.index[0] if not top_sector.empty else "Cluster"
        sector_weight = float(top_sector.iloc[0]) if not top_sector.empty else 0.0
        cash_efficiency = min(100.0, float(df["notional_abs"].sum()) / max(abs(float(df["market_value"].sum())), float(df["notional_abs"].sum()), 1.0) * 100)
        kpis = [
            ("Net Delta", f"{df['delta_exposure'].sum():+,.2f}", "Directional exposure", _status_from_abs(df["delta_exposure"].sum(), 300, 600)),
            ("Portfolio Heat", f"{heat:.0f}/100", "Total stop-loss risk", _status_from_level(heat, 45, 70)),
            ("Theta Burn/Day", fmt_num(totals["total_theta"]), "Daily decay", "Healthy" if totals["total_theta"] >= 0 else "Warning"),
            ("Convexity Risk", f"{df['gamma_exposure'].sum():+,.2f}", "Gamma acceleration", _status_from_abs(df["gamma_exposure"].abs().sum(), 5, 12)),
            ("Correlation Risk", _fmt_pct(sector_weight), sector_name, _status_from_level(sector_weight, 25, 40)),
            ("Near-Term DTE", _fmt_pct(near_term), "Expiration danger", _status_from_level(near_term, 25, 50)),
            ("IV Compression", f"{df['vega_exposure'].sum():+,.2f}", "Vega danger", _status_from_abs(df["vega_exposure"].sum(), 1500, 3500)),
            ("Cash Efficiency", _fmt_pct(cash_efficiency), "Productive capital", "Healthy" if cash_efficiency >= 70 else "Warning"),
        ]
        cols = st.columns(4)
        for index, (label, value, meaning, status) in enumerate(kpis):
            with cols[index % 4]:
                self._risk_kpi_card(label, value, meaning, status)

    def _risk_kpi_card(self, label: str, value: str, meaning: str, status: str):
        color = {"Healthy": BRAND["green"], "Low Risk": BRAND["green"], "Moderate": BRAND["yellow"], "Warning": "#f97316", "Elevated": "#f97316", "Danger": BRAND["red"]}.get(status, BRAND["blue"])
        st.markdown(
            f"""
            <div style="background:rgba(21,40,71,0.70);border:1px solid {BRAND['grid']};border-top:4px solid {color};
                border-radius:8px;padding:12px 12px;margin-bottom:10px;min-height:124px;">
                <div style="color:{BRAND['muted']};font-size:11px;font-weight:800;text-transform:uppercase;">{label}</div>
                <div style="color:{BRAND['text']};font-size:24px;font-weight:900;margin-top:5px;">{value}</div>
                <div style="color:{BRAND['muted']};font-size:11px;margin-top:2px;">{meaning}</div>
                <div style="color:{color};font-size:12px;font-weight:900;margin-top:8px;">{status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_greek_leader_charts(self, df: pd.DataFrame):
        st.markdown("### Top Greek Contributors")
        chart_specs = [
            ("Delta Leaders", "delta_exposure", BRAND["blue"], False),
            ("Gamma Leaders", "gamma_exposure", BRAND["purple"], False),
            ("Theta Income vs Theta Burn", "theta_decay_day", BRAND["yellow"], True),
            ("Vega Exposure", "vega_exposure", BRAND["green"], False),
        ]
        rows = [st.columns(2), st.columns(2)]
        for idx, (title, column, color, ascending) in enumerate(chart_specs):
            with rows[idx // 2][idx % 2]:
                if column == "theta_decay_day":
                    self._render_theta_contribution_chart(df, title=title, show_summary=False, compact=True)
                else:
                    data = df.reindex(df[column].abs().sort_values(ascending=False).index).head(self.top_n)
                    fig = px.bar(
                        data.sort_values(column),
                        x=column,
                        y="ticker",
                        orientation="h",
                        text=data.sort_values(column)[column].map(lambda value: f"{value:+,.2f}"),
                        color_discrete_sequence=[color],
                        hover_data=["recommendation", "risk_score", "exposure_status"],
                        title=title,
                    )
                    fig.add_vline(x=0, line_color=BRAND["muted"], line_width=1)
                    fig.update_traces(textposition="outside", cliponaxis=False)
                    style_figure(fig, height=max(280, 26 * len(data) + 92))
                    fig.update_layout(
                        yaxis=dict(
                            anchor="free",
                            position=0.0,
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None)

    def _render_theta_contribution_chart(
        self,
        df: pd.DataFrame,
        title: str = "Theta Income vs Theta Burn",
        show_summary: bool = True,
        compact: bool = False,
    ):
        theta = df.sort_values("theta_decay_day").copy()
        if show_summary:
            positive_theta = float(theta.loc[theta["theta_decay_day"] > 0, "theta_decay_day"].sum())
            negative_theta = float(theta.loc[theta["theta_decay_day"] < 0, "theta_decay_day"].sum())
            net_theta = float(theta["theta_decay_day"].sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Positive Theta", f"{positive_theta:+,.2f}/day")
            c2.metric("Negative Theta", f"{negative_theta:+,.2f}/day")
            c3.metric("Net Theta", f"{net_theta:+,.2f}/day", "Income positive" if net_theta >= 0 else "Decay drag")

        theta["theta_direction"] = np.where(theta["theta_decay_day"] >= 0, "Theta Income", "Theta Burn")
        theta["theta_label"] = theta["theta_decay_day"].map(lambda value: f"{value:+,.2f}")
        fig = px.bar(
            theta,
            x="theta_decay_day",
            y="ticker",
            orientation="h",
            color="theta_direction",
            text="theta_label",
            color_discrete_map={"Theta Income": BRAND["green"], "Theta Burn": BRAND["red"]},
            hover_data={
                "dte": ":.0f",
                "recommendation": True,
                "risk_score": ":.0f",
                "entry_value": ":$,.2f",
                "profit_value": ":$,.2f",
                "theta_label": False,
                "theta_direction": False,
            },
            labels={"theta_decay_day": "Theta / Day", "ticker": "Ticker"},
            title=title,
        )
        fig.add_vline(x=0, line_color=BRAND["muted"])
        fig.update_traces(textposition="outside", cliponaxis=False)
        style_figure(fig, height=max(360 if compact else 460, 24 * len(theta) + 100))
        fig.update_layout(
            yaxis=dict(
                anchor="free",
                position=0.0,
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22 if compact else -0.15,
                xanchor="center",
                x=0.5,
                title=None,
            ),
            margin=dict(l=24, r=24, t=56, b=85),
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

    def _render_greek_bubble_map(self, df: pd.DataFrame):
        risk = df.copy()
        risk["theta_size"] = risk["theta_decay_day"].abs().clip(lower=1)
        risk["quadrant"] = risk.apply(_greek_quadrant, axis=1)
        fig = px.scatter(
            risk,
            x="delta_exposure",
            y="gamma_exposure",
            size="theta_size",
            color="recommendation",
            text="ticker",
            color_discrete_map=ACTION_COLORS,
            hover_data={
                "quadrant": True,
                "theta_decay_day": ":.2f",
                "vega_exposure": ":.2f",
                "risk_score": ":.0f",
                "weight_pct": False,
            },
        )
        max_x = max(float(risk["delta_exposure"].abs().max()), 1.0) * 1.2
        max_y = max(float(risk["gamma_exposure"].abs().max()), 1.0) * 1.2
        fig.add_hline(y=0, line_color=BRAND["muted"], line_dash="dash")
        fig.add_vline(x=0, line_color=BRAND["muted"], line_dash="dash")
        fig.add_annotation(x=-max_x * 0.55, y=-max_y * 0.78, text="Low Risk", showarrow=False, font=dict(color=BRAND["green"], size=13))
        fig.add_annotation(x=-max_x * 0.55, y=max_y * 0.78, text="Convexity Trap", showarrow=False, font=dict(color=BRAND["yellow"], size=13))
        fig.add_annotation(x=max_x * 0.55, y=-max_y * 0.78, text="Directional Exposure", showarrow=False, font=dict(color=BRAND["blue"], size=13))
        fig.add_annotation(x=max_x * 0.55, y=max_y * 0.78, text="Danger Zone", showarrow=False, font=dict(color=BRAND["red"], size=13))
        fig.update_xaxes(range=[-max_x, max_x])
        fig.update_yaxes(range=[-max_y, max_y])
        fig.update_traces(marker=dict(opacity=0.82, line=dict(width=1, color="#e2e8f0")), textposition="top center")
        style_figure(fig, height=520)
        st.plotly_chart(fig, use_container_width=True, theme=None)

    def _fetch_latest_price_action_info(self, tickers: list[str]) -> dict[str, dict]:
        """Fetch the latest and previous Stage & FDTS signals for given tickers from price_action_story DB."""
        import sqlite3
        from utils.persistence import get_db_path
        
        results = {}
        if not tickers:
            return results
            
        try:
            db_path = get_db_path("price_action_story")
            if not db_path.exists():
                return results
                
            clean_tickers = [clean_ticker_for_lookup(t) for t in tickers]
            clean_tickers = list(set(clean_tickers))
            
            with sqlite3.connect(db_path) as conn:
                placeholders = ",".join(["?"] * len(clean_tickers))
                query = f"""
                    SELECT ticker, stage, fdts, scan_ts
                    FROM ticker_stage_history
                    WHERE ticker IN ({placeholders})
                    ORDER BY ticker ASC, scan_ts DESC
                """
                rows = conn.execute(query, clean_tickers).fetchall()
                
                ticker_history = {}
                for ticker, stage, fdts, scan_ts in rows:
                    t_upper = ticker.upper()
                    if t_upper not in ticker_history:
                        ticker_history[t_upper] = []
                    ticker_history[t_upper].append({
                        "stage": stage,
                        "fdts": fdts,
                        "scan_ts": scan_ts
                    })
                    
                for t_upper, history in ticker_history.items():
                    latest = history[0]
                    prev = history[1] if len(history) > 1 else None
                    
                    results[t_upper] = {
                        "current_stage": latest["stage"],
                        "current_fdts": latest["fdts"],
                        "prev_stage": prev["stage"] if prev else None,
                        "prev_fdts": prev["fdts"] if prev else None,
                        "changed": prev is not None and (latest["stage"] != prev["stage"] or latest["fdts"] != prev["fdts"])
                    }
        except Exception:
            pass
            
        return results

    def _render_greek_concentration(self, df: pd.DataFrame):
        st.markdown("### Greek Concentration Risk")
        metrics = [
            ("Gamma", _top_contribution_pct(df, "gamma_exposure")),
            ("Theta", _top_contribution_pct(df, "theta_decay_day")),
            ("Delta", _top_contribution_pct(df, "delta_exposure")),
        ]
        cols = st.columns(3)
        for idx, (label, value) in enumerate(metrics):
            status = _status_from_level(value, 45, 70)
            with cols[idx]:
                self._risk_kpi_card(f"Top 3 {label}", _fmt_pct(value), f"{label.lower()} from largest contributors", status)

        # Gross Option Exposure Concentration Chart
        st.markdown("<br>", unsafe_allow_html=True)
        df_exp = df.copy()
        total_gross = df_exp["gross_exposure"].sum()
        if total_gross > 0:
            df_exp["gross_exposure_pct"] = (df_exp["gross_exposure"] / total_gross) * 100
            df_exp = df_exp.sort_values("gross_exposure_pct", ascending=True)
            
            # Fetch latest stages and signals from Price Action Story Engine database
            tickers_list = df_exp["ticker"].tolist()
            stages_data = self._fetch_latest_price_action_info(tickers_list)
            
            display_labels = []
            current_stages = []
            current_fdts_list = []
            shift_texts = []
            custom_data_list = []
            
            for idx, row in df_exp.iterrows():
                raw_ticker = row["ticker"]
                t_clean = clean_ticker_for_lookup(raw_ticker)
                pa_info = stages_data.get(t_clean, {})
                
                curr_stage = pa_info.get("current_stage", "N/A")
                curr_fdts = pa_info.get("current_fdts", "⚪ No Trade")
                prev_stage = pa_info.get("prev_stage")
                prev_fdts = pa_info.get("prev_fdts")
                changed = pa_info.get("changed", False)
                
                formatted_ticker = format_ticker_for_display(raw_ticker)
                short_stage = curr_stage.replace(" / Expansion", "")
                
                # Y-axis will show clean ticker (e.g. 🔵 SPY)
                display_labels.append(formatted_ticker)
                
                current_stages.append(curr_stage)
                current_fdts_list.append(curr_fdts)
                
                if changed:
                    parts = []
                    if prev_stage and prev_stage != curr_stage:
                        parts.append(f"{prev_stage.replace(' / Expansion', '')} ➔ {short_stage}")
                    if prev_fdts and prev_fdts != curr_fdts:
                        clean_prev = prev_fdts.replace("🟢 ", "").replace("🔴 ", "").replace("⚪ ", "")
                        clean_curr = curr_fdts.replace("🟢 ", "").replace("🔴 ", "").replace("⚪ ", "")
                        parts.append(f"FDTS: {clean_prev} ➔ {clean_curr}")
                    shift_texts.append(" | ".join(parts))
                else:
                    shift_texts.append("Stable" if curr_stage != "N/A" else "No History")
                
                custom_data_list.append([
                    formatted_ticker,
                    row["gross_exposure_pct"],
                    curr_fdts,
                    prev_fdts if prev_fdts else "N/A",
                    curr_stage,
                    prev_stage if prev_stage else "N/A",
                    "Yes" if changed else "No"
                ])
                
            df_exp["display_label"] = display_labels
            df_exp["stage"] = current_stages
            df_exp["fdts"] = current_fdts_list
            df_exp["shift_text"] = shift_texts
            
            fig_exp = go.Figure(go.Bar(
                x=df_exp["gross_exposure_pct"],
                y=df_exp["display_label"],
                orientation="h",
                marker_color=[BRAND["blue"] if v < 15 else (BRAND["yellow"] if v <= 25 else BRAND["red"]) for v in df_exp["gross_exposure_pct"]],
                text=df_exp["gross_exposure_pct"].map(lambda v: f"{v:.1f}%"),
                textposition="auto",
                customdata=custom_data_list,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Gross Exposure: %{customdata[1]:.2f}%<br>"
                    "FDTS Signal: %{customdata[2]} (Prev: %{customdata[3]})<br>"
                    "PA Lifecycle: %{customdata[4]} (Prev: %{customdata[5]})<br>"
                    "Changed: %{customdata[6]}<extra></extra>"
                )
            ))
            
            # Add aligned annotations on the right side of the chart
            max_x = float(df_exp["gross_exposure_pct"].max()) if not df_exp.empty else 10.0
            for idx, row in df_exp.iterrows():
                raw_ticker = row["ticker"]
                t_clean = clean_ticker_for_lookup(raw_ticker)
                pa_info = stages_data.get(t_clean, {})
                
                curr_stage = pa_info.get("current_stage", "N/A")
                curr_fdts = pa_info.get("current_fdts", "⚪ No Trade")
                changed = pa_info.get("changed", False)
                
                short_stage = curr_stage.replace(" / Expansion", "")
                change_marker = " 🔄" if changed else ""
                
                annotation_text = f"{curr_fdts}  •  {short_stage}{change_marker}"
                
                fig_exp.add_annotation(
                    x=max_x + 0.5,
                    y=format_ticker_for_display(raw_ticker),
                    text=annotation_text,
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(color="#cbd5e1", size=10)
                )
            
            style_figure(fig_exp, height=max(240, 26 * len(df_exp) + 50))
            fig_exp.update_layout(
                title=dict(text="Portfolio Concentration by Gross Option Exposure (%)", font=dict(size=12)),
                margin=dict(l=80, r=180, t=30, b=10),
                xaxis=dict(title="Exposure Percentage (%)", ticksuffix="%", showgrid=True, gridcolor="#1e3a5f", range=[0, max_x + 6.0]),
                yaxis=dict(anchor="free", position=0.0),
            )
            st.plotly_chart(fig_exp, use_container_width=True, theme=None)
            
            # Companion concentration & lifecycle details table
            st.markdown("<br><b>Concentration & Lifecycle Detail Table</b>", unsafe_allow_html=True)
            df_table = df_exp.sort_values("gross_exposure_pct", ascending=False).copy()
            df_table["Ticker"] = df_table["ticker"].apply(format_ticker_for_display)
            
            table_cols = ["Ticker", "gross_exposure_pct", "fdts", "stage", "shift_text"]
            df_display = df_table[table_cols].rename(columns={
                "gross_exposure_pct": "Gross Exposure %",
                "fdts": "FDTS Signal",
                "stage": "Price Action Stage",
                "shift_text": "Stage Shift / Trend"
            })
            
            def style_stage_and_fdts(styler):
                stage_colors = {
                    "Early Bull / Expansion": "background-color: rgba(34,197,94,0.12); color: #22c55e; font-weight: bold;",
                    "Strong Bull": "background-color: rgba(16,185,129,0.12); color: #10b981; font-weight: bold;",
                    "Mature Bull": "background-color: rgba(234,179,8,0.12); color: #eab308; font-weight: bold;",
                    "Fading Bull": "background-color: rgba(249,115,22,0.12); color: #f97316; font-weight: bold;",
                    "Distribution": "background-color: rgba(239,68,68,0.12); color: #ef4444; font-weight: bold;",
                    "Breakdown": "background-color: rgba(153,27,27,0.15); color: #ef4444; font-weight: bold;",
                }
                
                def apply_stage_style(val):
                    return stage_colors.get(val, "color: #94a3b8;")
                    
                def apply_fdts_style(val):
                    if "Buy" in str(val):
                        return "background-color: rgba(34,197,94,0.12); color: #22c55e; font-weight: bold;"
                    elif "Sell" in str(val):
                        return "background-color: rgba(239,68,68,0.12); color: #ef4444; font-weight: bold;"
                    return "color: #94a3b8;"

                def apply_shift_style(val):
                    if "➔" in str(val):
                        return "background-color: rgba(234,179,8,0.08); color: #eab308; font-weight: bold;"
                    elif val == "Stable":
                        return "color: #475569;"
                    return "color: #64748b;"

                return styler.map(apply_stage_style, subset=["Price Action Stage"])\
                             .map(apply_fdts_style, subset=["FDTS Signal"])\
                             .map(apply_shift_style, subset=["Stage Shift / Trend"])\
                             .set_properties(subset=["Ticker"], **{"font-weight": "800", "color": "#e2e8f0"})

            st.dataframe(
                style_stage_and_fdts(df_display.style),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", width="medium"),
                    "Gross Exposure %": st.column_config.ProgressColumn(
                        "Gross Exposure %",
                        help="Percentage of total gross option exposure",
                        format="%.1f%%",
                        min_value=0.0,
                        max_value=100.0,
                        width="medium"
                    ),
                    "FDTS Signal": st.column_config.TextColumn("FDTS Signal", width="small"),
                    "Price Action Stage": st.column_config.TextColumn("Price Action Stage", width="medium"),
                    "Stage Shift / Trend": st.column_config.TextColumn("Stage Shift / Trend", width="medium")
                }
            )

    def _render_expiration_risk_heatmap(self, df: pd.DataFrame):
        st.markdown("### Expiration Risk Heatmap")
        buckets = ["0-7 DTE", "8-14 DTE", "15-30 DTE", "30-60 DTE"]
        rows = []
        top = df.reindex(df["notional_abs"].sort_values(ascending=False).index).head(min(self.top_n, 12))
        for _, row in top.iterrows():
            dte = float(row.get("dte", 30))
            values = []
            for bucket in buckets:
                active = (
                    (bucket == "0-7 DTE" and dte <= 7)
                    or (bucket == "8-14 DTE" and 8 <= dte <= 14)
                    or (bucket == "15-30 DTE" and 15 <= dte <= 30)
                    or (bucket == "30-60 DTE" and 31 <= dte <= 60)
                )
                risk_value = float(row["weight_pct"]) if active else 0.0
                values.append(risk_value)
            rows.append(values)
        heat = pd.DataFrame(rows, index=top["ticker"].tolist(), columns=buckets)
        if heat.empty:
            st.info("No positions available for expiration risk.")
            return
        fig = go.Figure(go.Heatmap(
            z=heat.values,
            x=heat.columns,
            y=heat.index,
            text=heat.map(_fmt_pct),
            texttemplate="%{text}",
            zmin=0,
            zmax=max(float(heat.values.max()), 20),
            colorscale=[[0, BRAND["green"]], [0.55, BRAND["yellow"]], [1, BRAND["red"]]],
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        ))
        style_figure(fig, height=max(320, 30 * len(heat) + 90))
        st.plotly_chart(fig, use_container_width=True, theme=None)

    def _render_greek_brain(self, df: pd.DataFrame, totals: dict[str, float], regime: dict[str, float | str], heat: float):
        st.markdown("### Portfolio Brain")
        commentary, actions, stack = _greek_commentary(df, totals, regime, heat)
        st.info(commentary)
        st.markdown("### Recommended Actions")
        st.markdown("\n".join([f"- {item}" for item in actions]))
        st.markdown("### Top Risk Stack")
        for index, item in enumerate(stack, start=1):
            st.markdown(f"**{index}. {item}**")

    def _render_what_if_panel(self, df: pd.DataFrame, totals: dict[str, float]):
        st.markdown("### Stress Test Scenarios")
        gross = max(float(df["notional_abs"].sum()), 1.0)
        delta = float(df["delta_exposure"].sum())
        gamma = float(df["gamma_exposure"].abs().sum())
        vega = float(df["vega_exposure"].sum())
        scenarios = pd.DataFrame(
            [
                {"Scenario": "SPY -2%", "Portfolio Impact": -(abs(delta) * 2 + gamma * 0.8) / gross * 100},
                {"Scenario": "VIX +15%", "Portfolio Impact": (vega * 0.15) / gross * 100},
                {"Scenario": "Largest Name -5%", "Portfolio Impact": -float(df["weight_pct"].max()) * 0.05},
                {"Scenario": "IV Crush", "Portfolio Impact": -(abs(vega) * 0.12) / gross * 100},
            ]
        )
        scenarios["Impact"] = scenarios["Portfolio Impact"].map(lambda value: _fmt_pct(value, signed=True))
        st.dataframe(scenarios[["Scenario", "Impact"]], use_container_width=True, hide_index=True)
        st.markdown("<p style='font-size:11.5px;color:#94a3b8;margin-top:-4px;'>For dynamic shocks, interactive curve simulations, and proactive hedging playbooks, visit the dedicated <b>What-If Simulator</b> tab.</p>", unsafe_allow_html=True)

    def _render_what_if_tab(
        self,
        df: pd.DataFrame,
        totals: dict[str, float],
        position_betas: dict[str, float] | None = None,
        portfolio_beta: float = 1.0,
        beta_benchmark: str = "SPY",
        details: pd.DataFrame | None = None,
        regime: dict | None = None,
    ):
        """SPY Scenario Portfolio Impact Simulator — full rebuild with 3-layer engine."""
        st.markdown("### SPY Scenario Portfolio Impact Simulator")
        st.markdown("""
        <p style='color:#94a3b8;font-size:13px;margin-bottom:4px;'>
        3-layer engine: <b>SPY Market Shock → Per-Position Repricing (beta-adjusted, VIX-correlated) → Scenario Action Signals</b>
        </p>""", unsafe_allow_html=True)

        position_betas = position_betas or {}
        port_val = float(totals.get("total_market_value", 0.0))
        gross_exp = max(float(df["notional_abs"].sum()), 1.0)
        capital_base = port_val if port_val > 10.0 else gross_exp

        # ── Sub-panel 0: Beta Analytics (preserved) ──────────────────────────
        st.markdown("#### Portfolio Beta Analytics")
        if portfolio_beta < 0.5:
            beta_label, beta_color = "Very Defensive", BRAND["green"]
        elif portfolio_beta < 0.85:
            beta_label, beta_color = "Defensive", "#22d3ee"
        elif portfolio_beta <= 1.15:
            beta_label, beta_color = "Market-Like", BRAND["yellow"]
        elif portfolio_beta <= 1.50:
            beta_label, beta_color = "Aggressive", "#f97316"
        else:
            beta_label, beta_color = "High Beta / Leveraged", BRAND["red"]

        beta_col1, beta_col2, beta_col3 = st.columns([1.0, 1.0, 2.0])
        with beta_col1:
            st.markdown(
                f"""
                <div style="background:#152847;border:1px solid #1e3a5f;border-left:4px solid {beta_color};
                    border-radius:8px;padding:14px;text-align:center;">
                    <div style="color:#94a3b8;font-size:12px;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.5px;">Portfolio Beta vs {beta_benchmark}</div>
                    <div style="color:#e2e8f0;font-size:40px;font-weight:900;margin:6px 0;">{portfolio_beta:.2f}</div>
                    <div style="color:{beta_color};font-size:13px;font-weight:700;">{beta_label}</div>
                    <div style="color:#64748b;font-size:11px;margin-top:4px;">1% {beta_benchmark} move ≈ {portfolio_beta:.2f}% portfolio move</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with beta_col2:
            beta_1pct_dollar = capital_base * portfolio_beta * 0.01
            beta_5pct_dollar = capital_base * portfolio_beta * 0.05
            beta_10pct_dollar = capital_base * portfolio_beta * 0.10
            st.markdown(
                f"""
                <div style="background:#152847;border:1px solid #1e3a5f;border-radius:8px;padding:14px;">
                    <div style="color:#94a3b8;font-size:12px;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.5px;margin-bottom:10px;">Beta-Adjusted Exposure</div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                        <span style="color:#94a3b8;font-size:12px;">{beta_benchmark} -1%</span>
                        <span style="color:#fca5a5;font-weight:700;font-size:13px;">-${beta_1pct_dollar:,.0f}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                        <span style="color:#94a3b8;font-size:12px;">{beta_benchmark} -5%</span>
                        <span style="color:#fca5a5;font-weight:700;font-size:13px;">-${beta_5pct_dollar:,.0f}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#94a3b8;font-size:12px;">{beta_benchmark} -10%</span>
                        <span style="color:#ef4444;font-weight:800;font-size:14px;">-${beta_10pct_dollar:,.0f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with beta_col3:
            if position_betas and not df.empty:
                beta_rows = []
                for _, row in df.iterrows():
                    clean = clean_ticker_for_lookup(str(row["ticker"]))
                    b = position_betas.get(clean, 1.0)
                    beta_rows.append({
                        "ticker": clean_ticker_for_lookup(str(row["ticker"])),
                        "beta": b,
                        "weight_pct": float(row["weight_pct"]),
                        "weighted_beta": round(b * float(row["weight_pct"]) / 100.0, 4),
                    })
                beta_df = pd.DataFrame(beta_rows).sort_values("beta", ascending=True)
                beta_df["color"] = beta_df["beta"].apply(
                    lambda v: BRAND["green"] if v < 0.85 else (BRAND["yellow"] if v <= 1.25 else BRAND["red"])
                )
                fig_beta = go.Figure(go.Bar(
                    x=beta_df["beta"],
                    y=beta_df["ticker"],
                    orientation="h",
                    marker_color=beta_df["color"].tolist(),
                    text=beta_df["beta"].map(lambda v: f"{v:.2f}β"),
                    textposition="outside",
                    cliponaxis=False,
                    customdata=beta_df[["weight_pct", "weighted_beta"]].values,
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        f"Beta vs {beta_benchmark}: %{{x:.2f}}<br>"
                        "Position Weight: %{customdata[0]:.2f}%<br>"
                        "Weighted Contribution: %{customdata[1]:.4f}<extra></extra>"
                    ),
                ))
                fig_beta.add_vline(x=1.0, line_dash="dash", line_color="#475569", line_width=1.5)
                style_figure(fig_beta, height=max(220, 26 * len(beta_df) + 60))
                fig_beta.update_layout(
                    margin=dict(l=10, r=40, t=24, b=10),
                    xaxis=dict(title=f"Beta vs {beta_benchmark}", showgrid=True, gridcolor="#1e3a5f"),
                    yaxis=dict(anchor="free", position=0.0),
                    title=dict(text=f"Per-Position Beta vs {beta_benchmark}", font=dict(size=13)),
                )
                st.plotly_chart(fig_beta, use_container_width=True, theme=None)
            else:
                st.info("Beta data is loading. Ensure \"Fetch market regime\" is enabled and positions are loaded.")

        st.markdown("---")

        # ── Sub-panel 1: Scenario Assumptions (Control Panel) ─────────────────
        st.markdown("#### Scenario Assumptions")
        ctrl1, ctrl2, ctrl3 = st.columns([1.5, 1.5, 1.0])
        with ctrl1:
            vix_auto = st.toggle(
                "Auto-correlate VIX to SPY move",
                value=True,
                key="tab_whatif_vix_auto",
                help="When ON, VIX shock is automatically derived from the SPY move using an empirical table (e.g. SPY -10% → VIX +30%). Turn OFF to set VIX shock manually.",
            )
        with ctrl2:
            days_passed = st.slider(
                "Days of Theta Decay to Include",
                min_value=0,
                max_value=5,
                value=1,
                step=1,
                key="tab_whatif_days_passed",
                help="Number of trading days of theta decay to add to the P/L estimate.",
            )
        with ctrl3:
            if not vix_auto:
                vix_manual = st.slider(
                    "VIX Shock Override (%)",
                    min_value=-50.0,
                    max_value=100.0,
                    value=15.0,
                    step=5.0,
                    key="tab_whatif_vix_manual",
                )
                vix_override: float | None = vix_manual
            else:
                st.markdown(
                    "<div style='color:#94a3b8;font-size:12px;padding-top:28px;'>VIX auto-linked to SPY</div>",
                    unsafe_allow_html=True,
                )
                vix_override = None

        # Display the live VIX shock table for the 9 scenarios
        auto_vix_row = {f"{s:+.2f}%": f"{_vix_shock_from_spy(s):+.2f}%" for s in _SPY_SCENARIOS}
        if vix_auto:
            st.markdown(
                "<p style='font-size:11.5px;color:#64748b;margin-top:-4px;'>Auto VIX shocks applied per scenario ↓</p>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                pd.DataFrame([auto_vix_row], index=["VIX Shock"]),
                use_container_width=True,
                hide_index=False,
            )

        st.markdown("---")

        # ── Build scenario matrix (Layer 1 + 2) ──────────────────────────────
        matrix = _build_scenario_matrix(df, position_betas, vix_override, float(days_passed), details=details)

        # Portfolio-level totals per scenario
        if not matrix.empty:
            port_by_scenario = (
                matrix.groupby("spy_pct")["total_pnl"]
                .sum()
                .reset_index()
                .rename(columns={"spy_pct": "SPY Move", "total_pnl": "Portfolio P/L"})
            )
            port_by_scenario["SPY Label"] = port_by_scenario["SPY Move"].map(lambda v: f"{v:+.2f}%")
            port_by_scenario["color"] = port_by_scenario["Portfolio P/L"].apply(
                lambda v: BRAND["green"] if v >= 0 else BRAND["red"]
            )
            port_by_scenario["pct_of_capital"] = port_by_scenario["Portfolio P/L"] / max(capital_base, 1.0) * 100
        else:
            port_by_scenario = pd.DataFrame()

        st.markdown("#### Portfolio P/L by SPY Scenario")
        if not port_by_scenario.empty:
            # Calculate standard deviation for next 20 calendar days from VIX
            vix_val = 16.0
            if regime is not None and "vix" in regime:
                try:
                    vix_val = float(regime["vix"])
                except Exception:
                    pass
            if vix_val <= 0:
                vix_val = 16.0
                
            import math
            one_std = vix_val * math.sqrt(20.0 / 365.0)

            # Retrieve dynamic skewness from regime
            spy_skew = -0.45
            skew_label = "Negative Skew (Fat Left Tail / Downside Risk Heavy)"
            if regime is not None and "spy_skew" in regime:
                try:
                    spy_skew = float(regime.get("spy_skew", -0.45))
                    skew_label = str(regime.get("skew_label", skew_label))
                except Exception:
                    pass

            # Calculate probability and standard deviation move for each scenario
            def _calc_prob_and_sigma(row):
                move = float(row["SPY Move"])
                if move == 0.0:
                    return 0.0, 100.0, "Base Case"
                
                z = move / one_std
                # Cumulative distribution function of standard normal
                cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
                
                if move < 0.0:
                    prob = cdf * 100.0
                    text = "Prob(SPY &le; " + f"{move:+.1f}%)"
                else:
                    prob = (1.0 - cdf) * 100.0
                    text = "Prob(SPY &ge; " + f"{move:+.1f}%)"
                
                return z, prob, text

            prob_res = port_by_scenario.apply(_calc_prob_and_sigma, axis=1)
            port_by_scenario["sigma_move"] = [r[0] for r in prob_res]
            port_by_scenario["prob_happening"] = [r[1] for r in prob_res]
            port_by_scenario["prob_text"] = [r[2] for r in prob_res]
            port_by_scenario["skew_desc"] = skew_label

            # Format as strings with 2 decimal places in python directly to avoid Plotly parser bugs
            port_by_scenario["pct_of_capital_str"] = port_by_scenario["pct_of_capital"].map(lambda v: f"{v:+.2f}%")
            port_by_scenario["sigma_move_str"] = port_by_scenario["sigma_move"].map(lambda v: f"{v:+.2f}σ")
            port_by_scenario["prob_happening_str"] = port_by_scenario["prob_happening"].map(lambda v: f"{v:.2f}%")

            # Metric cards for key scenarios
            key_scenarios = [-10.0, -5.0, 0.0, 5.0, 10.0]
            metric_cols = st.columns(len(key_scenarios))
            for idx, spy_val in enumerate(key_scenarios):
                row_data = port_by_scenario[port_by_scenario["SPY Move"] == spy_val]
                if not row_data.empty:
                    pnl = float(row_data["Portfolio P/L"].iloc[0])
                    pct = float(row_data["pct_of_capital"].iloc[0])
                    vix_at = _vix_shock_from_spy(spy_val) if vix_override is None else vix_override
                    with metric_cols[idx]:
                        card_color = BRAND["green"] if pnl >= 0 else BRAND["red"]
                        st.markdown(
                            f"""
                            <div style="background:#0f1e36;border:1px solid #1e3a5f;border-top:3px solid {card_color};
                                border-radius:8px;padding:10px;text-align:center;margin-bottom:6px;">
                                <div style="color:#94a3b8;font-size:11px;font-weight:800;text-transform:uppercase;">SPY {spy_val:+.2f}%</div>
                                <div style="color:{card_color};font-size:20px;font-weight:900;margin:4px 0;">${pnl:+,.0f}</div>
                                <div style="color:#64748b;font-size:10px;">{pct:+.2f}% capital</div>
                                <div style="color:#475569;font-size:10px;">VIX {vix_at:+.2f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # 9-scenario bar chart — beta-adjusted via matrix
            fig_scen = go.Figure(go.Bar(
                x=port_by_scenario["SPY Move"],
                y=port_by_scenario["Portfolio P/L"],
                marker_color=port_by_scenario["color"].tolist(),
                text=port_by_scenario["Portfolio P/L"].map(lambda v: f"${v:+,.0f}"),
                textposition="outside",
                cliponaxis=False,
                customdata=port_by_scenario[["pct_of_capital_str", "SPY Label", "sigma_move_str", "prob_happening_str", "prob_text", "skew_desc"]].values,
                hovertemplate=(
                    "<b>SPY %{customdata[1]} Scenario</b><br>"
                    "Portfolio P/L: $%{y:,.2f} (%{customdata[0]})<br>"
                    "Distance from mean: %{customdata[2]}<br>"
                    "%{customdata[4]}: <b>%{customdata[3]}</b><br><br>"
                    "Current Market Skew:<br><i>%{customdata[5]}</i><extra></extra>"
                ),
            ))
            fig_scen.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)

            # Standard Deviation Shaded Regions (Probability Cones)
            # 1st Standard Deviation: [-one_std, one_std] -> Green
            fig_scen.add_vrect(
                x0=-one_std, x1=one_std,
                fillcolor="#10b981", opacity=0.06, line_width=0, layer="below"
            )
            # 2nd Standard Deviation Left: [-2*one_std, -one_std] -> Yellow
            fig_scen.add_vrect(
                x0=-2*one_std, x1=-one_std,
                fillcolor="#facc15", opacity=0.06, line_width=0, layer="below"
            )
            # 2nd Standard Deviation Right: [one_std, 2*one_std] -> Yellow
            fig_scen.add_vrect(
                x0=one_std, x1=2*one_std,
                fillcolor="#facc15", opacity=0.06, line_width=0, layer="below"
            )
            # 3rd Standard Deviation Left: [-3*one_std, -2*one_std] -> Red
            fig_scen.add_vrect(
                x0=-3*one_std, x1=-2*one_std,
                fillcolor="#ef4444", opacity=0.06, line_width=0, layer="below"
            )
            # 3rd Standard Deviation Right: [2*one_std, 3*one_std] -> Red
            fig_scen.add_vrect(
                x0=2*one_std, x1=3*one_std,
                fillcolor="#ef4444", opacity=0.06, line_width=0, layer="below"
            )

            # Boundary vertical bars
            fig_scen.add_vline(
                x=-one_std, line_color="#10b981", line_width=4.0,
                annotation_text="<b>-1σ</b>", annotation_position="top left",
                annotation_font=dict(color="#10b981", size=11)
            )
            fig_scen.add_vline(
                x=one_std, line_color="#10b981", line_width=4.0,
                annotation_text="<b>+1σ</b>", annotation_position="top right",
                annotation_font=dict(color="#10b981", size=11)
            )
            fig_scen.add_vline(
                x=-2*one_std, line_color="#facc15", line_width=4.0,
                annotation_text="<b>-2σ</b>", annotation_position="top left",
                annotation_font=dict(color="#facc15", size=11)
            )
            fig_scen.add_vline(
                x=2*one_std, line_color="#facc15", line_width=4.0,
                annotation_text="<b>+2σ</b>", annotation_position="top right",
                annotation_font=dict(color="#facc15", size=11)
            )
            fig_scen.add_vline(
                x=-3*one_std, line_color="#ef4444", line_width=4.0,
                annotation_text="<b>-3σ</b>", annotation_position="top left",
                annotation_font=dict(color="#ef4444", size=11)
            )
            fig_scen.add_vline(
                x=3*one_std, line_color="#ef4444", line_width=4.0,
                annotation_text="<b>+3σ</b>", annotation_position="top right",
                annotation_font=dict(color="#ef4444", size=11)
            )

            # Breakeven annotation: find first scenario where P/L crosses zero
            pos_vals = port_by_scenario[port_by_scenario["Portfolio P/L"] >= 0]["SPY Move"]
            neg_vals = port_by_scenario[port_by_scenario["Portfolio P/L"] < 0]["SPY Move"]
            if not pos_vals.empty and not neg_vals.empty:
                sorted_df = port_by_scenario.sort_values("SPY Move")
                for i in range(len(sorted_df) - 1):
                    p1 = float(sorted_df.iloc[i]["Portfolio P/L"])
                    p2 = float(sorted_df.iloc[i + 1]["Portfolio P/L"])
                    s1 = float(sorted_df.iloc[i]["SPY Move"])
                    s2 = float(sorted_df.iloc[i + 1]["SPY Move"])
                    if p1 * p2 < 0:
                        breakeven_spy = s1 - p1 * (s2 - s1) / (p2 - p1)
                        fig_scen.add_annotation(
                            x=breakeven_spy,
                            y=0,
                            text=f"Breakeven ≈ SPY {breakeven_spy:+.2f}%",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor=BRAND["yellow"],
                            font=dict(color=BRAND["yellow"], size=11, family="monospace"),
                            bgcolor="#0f1e36",
                            bordercolor=BRAND["yellow"],
                            borderpad=4,
                        )
                        break

            style_figure(fig_scen, height=320)
            fig_scen.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(
                    title="SPY Scenario (%)",
                    showgrid=False,
                    tickmode="array",
                    tickvals=list(port_by_scenario["SPY Move"]),
                    ticktext=list(port_by_scenario["SPY Label"])
                ),
                yaxis=dict(title="Estimated Portfolio P/L ($)", tickprefix="$", showgrid=True, gridcolor="#1e3a5f"),
                showlegend=False,
            )
            st.plotly_chart(fig_scen, use_container_width=True, theme=None)
            st.markdown(
                f"<p style='font-size:11px;color:#94a3b8;margin-top:-8px;'>"  
                f"Delta &amp; Gamma impact scaled by per-ticker beta. VIX shock auto-correlated to SPY move when toggle is ON. "
                f"Theta decay applied for selected days. Shaded regions represent 1-3 Standard Deviations for the next 20 days "
                f"based on current VIX of {vix_val:.2f} (1σ = ±{one_std:.2f}%). P/L is an approximation — not a full revaluation.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No position data to build scenario matrix.")

        st.markdown("---")

        # ── Sub-panel 3: Position-Level Shock Table ───────────────────────────
        st.markdown("#### Position-Level Shock Table")
        if not matrix.empty:
            # Added increments around zero (-2%, -1%, +1%, +2%)
            key_spy_vals = [-10.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
            pivot = matrix[matrix["spy_pct"].isin(key_spy_vals)].pivot_table(
                index=["ticker", "strategy", "beta"],
                columns="spy_pct",
                values="total_pnl",
                aggfunc="sum",
            ).reset_index()
            pivot.columns = (
                ["Ticker", "Strategy", "Beta"]
                + [f"SPY {v:+.2f}%" for v in key_spy_vals]
            )

            # Add scenario-aware action signal using base scenario (0.00%)
            base_col_candidates = [c for c in pivot.columns if "0.00%" in c or "0%" in c]
            base_col = base_col_candidates[0] if base_col_candidates else pivot.columns[3]

            scenario_pnl_cols = [c for c in pivot.columns if c.startswith("SPY ")]
            action_signals = []
            for _, prow in pivot.iterrows():
                pnl_vals = [float(prow.get(c, 0.0)) for c in scenario_pnl_cols]
                base_pnl_val = float(prow.get(base_col, 0.0))
                # Look up risk_score from enriched df
                ticker_clean = clean_ticker_for_lookup(str(prow["Ticker"]))
                risk_row = df[df["ticker"].apply(clean_ticker_for_lookup) == ticker_clean]
                risk_score_val = float(risk_row["risk_score"].iloc[0]) if not risk_row.empty else 50.0
                weight_pct_val = float(risk_row["weight_pct"].iloc[0]) if not risk_row.empty else 0.0
                current_pnl_val = float(risk_row["pl_open"].iloc[0]) if not risk_row.empty else 0.0
                signal = _scenario_action_signal(
                    pnl_vals, base_pnl_val, current_pnl_val, risk_score_val, weight_pct_val
                )
                action_signals.append(signal)
            pivot["Signal"] = action_signals

            # Add Total row to pivot table
            pnl_cols = [c for c in pivot.columns if c.startswith("SPY ")]
            totals_row = {
                "Ticker": "Total",
                "Strategy": "",
                "Beta": portfolio_beta,
                "Signal": "",
            }
            for c in pnl_cols:
                totals_row[c] = pivot[c].sum()
            totals_df = pd.DataFrame([totals_row])
            pivot = pd.concat([pivot, totals_df], ignore_index=True)

            pnl_cols = [c for c in pivot.columns if c.startswith("SPY ")]

            def _color_pnl(val: float) -> str:
                if not isinstance(val, (int, float)):
                    return ""
                if val > 0:
                    return f"color: {BRAND['green']}; font-weight: 700"
                if val < 0:
                    return f"color: {BRAND['red']}; font-weight: 700"
                return "color: #94a3b8"

            def _bold_total_row(row):
                if row["Ticker"] == "Total":
                    return ["font-weight: 900; background-color: #0f274a; border-top: 2px solid #1e3a5f;"] * len(row)
                return [""] * len(row)

            fmt_dict = {c: "${:+,.2f}" for c in pnl_cols}
            fmt_dict["Beta"] = "{:.2f}"
            styled = pivot.style.map(_color_pnl, subset=pnl_cols).apply(_bold_total_row, axis=1).format(fmt_dict)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("Scenario matrix not available.")

        st.markdown("---")

        # ── Sub-panel 4: SPY Stress Heatmap (Ticker × Scenario) ──────────────
        st.markdown("#### SPY Stress Heatmap — Ticker × Scenario")
        if not matrix.empty:
            heat_pivot = matrix.pivot_table(
                index="ticker",
                columns="spy_pct",
                values="total_pnl",
                aggfunc="sum",
            )
            heat_pivot.columns = [f"{v:+.0f}%" for v in heat_pivot.columns]
            heat_pivot = heat_pivot.sort_values(heat_pivot.columns[0], ascending=True)  # worst scenario first

            z_vals = heat_pivot.values
            abs_max = max(float(np.abs(z_vals).max()), 1.0)
            text_vals = [[f"${v:+,.0f}" for v in row_vals] for row_vals in z_vals]

            fig_heat = go.Figure(go.Heatmap(
                z=z_vals,
                x=list(heat_pivot.columns),
                y=list(heat_pivot.index),
                text=text_vals,
                texttemplate="%{text}",
                zmin=-abs_max,
                zmax=abs_max,
                colorscale=[[0, BRAND["red"]], [0.5, "#1e293b"], [1, BRAND["green"]]],
                hovertemplate="<b>%{y}</b><br>SPY %{x}<br>Est. P/L: %{text}<extra></extra>",
            ))
            style_figure(fig_heat, height=max(320, 28 * len(heat_pivot) + 80))
            fig_heat.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="SPY Scenario"),
                yaxis=dict(title="", automargin=True),
            )
            st.plotly_chart(fig_heat, use_container_width=True, theme=None)

        st.markdown("---")

        # ── Sub-panel 5: Contribution Breakdown (Ticker + Strategy) ──────────
        st.markdown("#### Contribution Breakdown")
        if not matrix.empty:
            # Use the SPY -5% scenario for contribution charts
            contrib_scenario = -5.0
            contrib_df = matrix[matrix["spy_pct"] == contrib_scenario].copy()
            if contrib_df.empty and not matrix.empty:
                contrib_scenario = float(matrix["spy_pct"].unique()[0])
                contrib_df = matrix[matrix["spy_pct"] == contrib_scenario].copy()

            cc1, cc2 = st.columns(2)
            with cc1:
                ticker_contrib = (
                    contrib_df.groupby("ticker")["total_pnl"]
                    .sum()
                    .sort_values()
                    .reset_index()
                )
                ticker_contrib["color"] = ticker_contrib["total_pnl"].apply(
                    lambda v: BRAND["green"] if v >= 0 else BRAND["red"]
                )
                fig_tc = go.Figure(go.Bar(
                    x=ticker_contrib["total_pnl"],
                    y=ticker_contrib["ticker"],
                    orientation="h",
                    marker_color=ticker_contrib["color"].tolist(),
                    text=ticker_contrib["total_pnl"].map(lambda v: f"${v:+,.0f}"),
                    textposition="auto",
                    cliponaxis=False,
                    hovertemplate=f"<b>%{{y}}</b><br>P/L at SPY {contrib_scenario:+.0f}%: $%{{x:,.2f}}<extra></extra>",
                ))
                fig_tc.add_vline(x=0, line_color="#475569", line_width=1)
                style_figure(fig_tc, height=max(280, 26 * len(ticker_contrib) + 60))
                fig_tc.update_layout(
                    title=dict(text=f"Ticker P/L at SPY {contrib_scenario:+.0f}%", font=dict(size=12)),
                    margin=dict(l=10, r=40, t=36, b=10),
                    xaxis=dict(tickprefix="$", showgrid=True, gridcolor="#1e3a5f"),
                    yaxis=dict(anchor="free", position=0.0),
                )
                st.plotly_chart(fig_tc, use_container_width=True, theme=None)

            with cc2:
                strat_contrib = (
                    contrib_df.groupby("strategy")["total_pnl"]
                    .sum()
                    .sort_values()
                    .reset_index()
                )
                strat_contrib["color"] = strat_contrib["total_pnl"].apply(
                    lambda v: BRAND["green"] if v >= 0 else BRAND["red"]
                )
                fig_sc = go.Figure(go.Bar(
                    x=strat_contrib["total_pnl"],
                    y=strat_contrib["strategy"],
                    orientation="h",
                    marker_color=strat_contrib["color"].tolist(),
                    text=strat_contrib["total_pnl"].map(lambda v: f"${v:+,.0f}"),
                    textposition="auto",
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>Strategy P/L: $%{x:,.2f}<extra></extra>",
                ))
                fig_sc.add_vline(x=0, line_color="#475569", line_width=1)
                style_figure(fig_sc, height=max(280, 26 * len(strat_contrib) + 60))
                fig_sc.update_layout(
                    title=dict(text=f"Strategy P/L at SPY {contrib_scenario:+.0f}%", font=dict(size=12)),
                    margin=dict(l=10, r=40, t=36, b=10),
                    xaxis=dict(tickprefix="$", showgrid=True, gridcolor="#1e3a5f"),
                    yaxis=dict(anchor="free", position=0.0),
                )
                st.plotly_chart(fig_sc, use_container_width=True, theme=None)

        st.markdown("---")

        # ── Sub-panel 6: Calendar Risk Alerts ────────────────────────────────
        calendar_positions = df[
            df["strategy"].apply(lambda s: any(kw in str(s) for kw in ("Calander", "Calendar")))
        ] if "strategy" in df.columns else pd.DataFrame()

        if not calendar_positions.empty:
            st.markdown("#### 📅 Calendar Spread Risk Monitor")
            # Use SPY -5% as the reference scenario for calendar checks
            cal_spy = -5.0
            cal_vix = vix_override if vix_override is not None else _vix_shock_from_spy(cal_spy)
            median_theta_abs = float(df["theta"].abs().median()) if not df.empty else 0.0
            median_vega = float(df["vega"].median()) if not df.empty else 0.0

            alert_level_map = {"Safe": "🟢", "Monitor": "🟡", "Danger": "🔴"}
            cal_rows = []
            for _, cal_row in calendar_positions.iterrows():
                alerts = _calendar_risk_checks(
                    cal_row, cal_spy, cal_vix, median_theta_abs, median_vega
                )
                row_dict: dict = {
                    "Ticker": clean_ticker_for_lookup(str(cal_row.get("ticker", ""))),
                    "Strategy": str(cal_row.get("strategy", "")),
                    "Front DTE": int(float(cal_row.get("min_dte", 0))),
                    "Back DTE": int(float(cal_row.get("max_dte", 0))),
                }
                for check_name, level in alerts:
                    row_dict[check_name] = f"{alert_level_map.get(level, '⚪')} {level}"
                cal_rows.append(row_dict)

            cal_table = pd.DataFrame(cal_rows)
            st.dataframe(cal_table, use_container_width=True, hide_index=True)

            danger_count = sum(
                1 for r in cal_rows
                for k, v in r.items()
                if k not in {"Ticker", "Strategy", "Front DTE", "Back DTE"} and "Danger" in str(v)
            )
            if danger_count > 0:
                st.error(
                    f"⚠️ **{danger_count} calendar risk alert(s) at Danger level** under SPY {cal_spy:+.0f}% scenario. "
                    "Review short-strike distance and IV crush exposure before earnings or macro events."
                )
            st.markdown("---")

        # ── Sub-panel 7: Proactive Advisory (scenario-driven) ────────────────
        st.markdown("#### Proactive Profit Protection Advisory")
        if not port_by_scenario.empty:
            base_impact = float(
                port_by_scenario.loc[port_by_scenario["SPY Move"] == 0.0, "Portfolio P/L"].sum()
                if 0.0 in port_by_scenario["SPY Move"].values
                else 0.0
            )
            stress_impact = float(
                port_by_scenario.loc[port_by_scenario["SPY Move"] == -5.0, "Portfolio P/L"].sum()
                if -5.0 in port_by_scenario["SPY Move"].values
                else 0.0
            )
            worst_impact = float(port_by_scenario["Portfolio P/L"].min())
            best_impact = float(port_by_scenario["Portfolio P/L"].max())
            negative_scenarios = int((port_by_scenario["Portfolio P/L"] < 0).sum())

            with st.expander("🛡️ Proactive Profit Protection Advisory", expanded=True):
                advisory_color = BRAND["green"] if stress_impact >= 0 else BRAND["red"]
                st.markdown(
                    f"""
                    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;">
                        <div style="background:#0f1e36;border:1px solid #1e3a5f;border-left:4px solid {BRAND['blue']};border-radius:6px;padding:10px 14px;flex:1;min-width:140px;">
                            <div style="color:#94a3b8;font-size:11px;">Best Scenario</div>
                            <div style="color:{BRAND['green']};font-size:18px;font-weight:800;">${best_impact:+,.0f}</div>
                        </div>
                        <div style="background:#0f1e36;border:1px solid #1e3a5f;border-left:4px solid {advisory_color};border-radius:6px;padding:10px 14px;flex:1;min-width:140px;">
                            <div style="color:#94a3b8;font-size:11px;">SPY -5% Impact</div>
                            <div style="color:{advisory_color};font-size:18px;font-weight:800;">${stress_impact:+,.0f}</div>
                        </div>
                        <div style="background:#0f1e36;border:1px solid #1e3a5f;border-left:4px solid {BRAND['red']};border-radius:6px;padding:10px 14px;flex:1;min-width:140px;">
                            <div style="color:#94a3b8;font-size:11px;">Worst Scenario</div>
                            <div style="color:{BRAND['red']};font-size:18px;font-weight:800;">${worst_impact:+,.0f}</div>
                        </div>
                        <div style="background:#0f1e36;border:1px solid #1e3a5f;border-left:4px solid {'#f97316' if negative_scenarios >= 5 else BRAND['green']};border-radius:6px;padding:10px 14px;flex:1;min-width:140px;">
                            <div style="color:#94a3b8;font-size:11px;">Negative Scenarios</div>
                            <div style="color:{'#f97316' if negative_scenarios >= 5 else BRAND['green']};font-size:18px;font-weight:800;">{negative_scenarios}/9</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if negative_scenarios == 0:
                    st.success("✅ **All-Clear**: Portfolio shows positive P/L across all 9 SPY scenarios. Current structure appears resilient to tested market shocks.")
                elif negative_scenarios <= 2:
                    st.info("ℹ️ **Selective Risk**: Portfolio underperforms in extreme tail scenarios only. Monitor positions with high negative scenario contributions in the heatmap above.")
                elif negative_scenarios <= 5:
                    st.warning(
                        f"⚠️ **Scenario Risk Detected**: {negative_scenarios} of 9 SPY scenarios produce negative P/L. "
                        "Review the Ticker Contribution chart to identify which positions drive losses and consider defensive adjustments."
                    )
                else:
                    st.error(
                        f"🚨 **High Scenario Risk**: {negative_scenarios} of 9 scenarios are negative (worst: ${worst_impact:,.0f}). "
                        "Immediate action advised — reduce directional exposure, buy portfolio protection (SPY puts), or trim the highest-beta positions identified in the shock table above."
                    )

                # Scenario-driven action checklist
                net_delta_val = float(totals.get("total_delta", 0.0))
                net_vega_val = float(totals.get("total_vega", 0.0))
                net_theta_val = float(totals.get("total_theta", 0.0))

                actions_text = []
                if stress_impact < -capital_base * 0.03:
                    actions_text.append(f"**Delta Hedge**: SPY -5% costs ~${abs(stress_impact):,.0f}. Buy SPY put spreads or reduce long delta positions.")
                if net_vega_val < -1000 and (vix_override or _vix_shock_from_spy(-5.0)) > 5:
                    actions_text.append("**Vega Hedge**: Short vega book is exposed to IV expansion. Consider VIX calls or calendar back-spreads.")
                if net_theta_val < 0:
                    actions_text.append("**Theta Drain**: Book carries negative theta. Close losing structures with DTE < 14 or add premium-selling trades.")
                if portfolio_beta > 1.3:
                    actions_text.append(f"**Beta Reduction**: Portfolio beta {portfolio_beta:.2f} amplifies market moves. Reduce high-beta names or add inverse ETF hedge.")
                if not actions_text:
                    actions_text.append("Maintain current structure — no critical action signals from scenario analysis.")
                for action in actions_text:
                    st.markdown(f"- {action}")

    def _render_behavior_metrics(self, df: pd.DataFrame, totals: dict[str, float]):
        st.markdown("### Portfolio Behavior")
        avg_dte = float(df["dte"].mean()) if not df.empty else 0.0
        theta_eff = totals["total_theta"] / max(float(df["notional_abs"].sum()), 1.0) * 100
        win_rate = float((df["pl_open"] > 0).mean() * 100) if not df.empty else 0.0
        overexposed = int((df["exposure_status"] == "Overexposed").sum())
        data = pd.DataFrame(
            [
                {"Metric": "Average DTE", "Value": f"{avg_dte:.1f}"},
                {"Metric": "Theta Efficiency", "Value": _fmt_pct(theta_eff, signed=True)},
                {"Metric": "Open P/L Win Rate", "Value": _fmt_pct(win_rate)},
                {"Metric": "Overexposed Positions", "Value": str(overexposed)},
            ]
        )
        st.dataframe(data, use_container_width=True, hide_index=True)

    def _render_correlation(self, df: pd.DataFrame):
        st.markdown("### Correlation Risk")
        corr = fetch_position_correlations(tuple(df["ticker"].tolist()), self.lookback_days)
        if corr.empty:
            st.warning("Correlation data is unavailable. The thematic concentration map below still flags hidden single-theme exposure.")
        else:
            corr_pct = corr * 100
            fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                colorscale=[[0, BRAND["red"]], [0.5, BRAND["yellow"]], [1, BRAND["green"]]],
                text=corr_pct.map(lambda value: f"{value:.0f}%"),
                texttemplate="%{text}",
                customdata=corr_pct.values,
                hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{customdata:.0f}%<extra></extra>",
            ))
            style_figure(fig, height=max(440, 32 * len(corr)))
            st.plotly_chart(fig, use_container_width=True, theme=None)

        sector = df.groupby("sector", as_index=False).agg(weight_pct=("weight_pct", "sum"), net_delta=("delta_exposure", "sum"), tickers=("ticker", "nunique"))
        sector["weight_label"] = sector["weight_pct"].map(_fmt_pct)
        fig = px.bar(
            sector.sort_values("weight_pct"),
            x="weight_pct",
            y="sector",
            orientation="h",
            color="weight_pct",
            text="weight_label",
            color_continuous_scale=[[0, BRAND["green"]], [0.65, BRAND["yellow"]], [1, BRAND["red"]]],
            custom_data=["weight_label", "net_delta", "tickers"],
            labels={"weight_pct": "Weight (%)", "sector": "Sector"},
        )
        fig.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Weight: %{customdata[0]}<br>"
                "Net Delta: %{customdata[1]:.2f}<br>"
                "Tickers: %{customdata[2]}<extra></extra>"
            )
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(ticksuffix="%")
        style_figure(fig, height=360)
        st.plotly_chart(fig, use_container_width=True, theme=None)

    def _render_theta(self, df: pd.DataFrame):
        st.markdown("### Theta Decay Engine")
        self._render_theta_contribution_chart(df, title="Full Portfolio Theta Income vs Theta Burn", show_summary=True)
        traps = df[(df["theta"] < 0) & (df["dte"] < 15)].sort_values("risk_score").copy()
        traps["Source"] = traps["ticker"].apply(get_broker_dot)
        traps["ticker"] = traps["ticker"].apply(clean_ticker_for_lookup)
        st.markdown("### Expiration Danger")
        st.dataframe(
            traps[["Source", "ticker", "dte", "theta", "entry_value", "current_value", "profit_value", "pl_day", "risk_score", "recommendation"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Source": st.column_config.TextColumn("Source", width="small"),
                "entry_value": st.column_config.NumberColumn("Entry Value", format="$%.2f"),
                "current_value": st.column_config.NumberColumn("Current Value", format="$%.2f"),
                "profit_value": st.column_config.NumberColumn("Net Profit", format="$%.2f"),
            },
        )

    def _render_profit_taking(self, df: pd.DataFrame):
        st.markdown("### Profit-Taking Engine")
        winners = df[(df["profit_value"] > 0) | (df["pl_day"] > 0)].copy()
        winners["profit_pressure"] = winners["profit_value"].clip(lower=0) + winners["gamma_exposure"].abs() * 0.25 + winners["weight_pct"] * 10
        winners = winners.sort_values("profit_pressure", ascending=False).head(self.top_n)
        winners["weight_pct_label"] = winners["weight_pct"].map(_fmt_pct)
        winners["profit_pct_label"] = winners["profit_pct"].map(lambda value: _fmt_pct(value, signed=True))
        fig = px.scatter(
            winners,
            x="profit_value",
            y="gamma_exposure",
            size="weight_pct",
            color="recommendation",
            text="ticker",
            color_discrete_map=ACTION_COLORS,
            hover_data={
                "pl_day": ":$,.2f",
                "risk_score": ":.0f",
                "weight_pct": False,
                "weight_pct_label": True,
            },
        )
        style_figure(fig, height=430)
        st.plotly_chart(fig, use_container_width=True, theme=None)
        winners["Source"] = winners["ticker"].apply(get_broker_dot)
        winners["ticker"] = winners["ticker"].apply(clean_ticker_for_lookup)
        st.dataframe(
            winners[["Source", "ticker", "entry_value", "current_value", "gross_exposure", "profit_value", "profit_pct_label", "pl_day", "gamma_exposure", "weight_pct_label", "risk_score", "recommendation"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Source": st.column_config.TextColumn("Source", width="small"),
                "entry_value": st.column_config.NumberColumn("Entry Value", format="$%.2f"),
                "current_value": st.column_config.NumberColumn("Current Value", format="$%.2f"),
                "gross_exposure": st.column_config.NumberColumn("Gross Leg Exposure", format="$%.2f"),
                "profit_value": st.column_config.NumberColumn("Net Profit", format="$%.2f"),
                "profit_pct_label": st.column_config.TextColumn("Profit %"),
                "weight_pct_label": st.column_config.TextColumn("Weight"),
            },
        )

    def _render_redeployment(self, df: pd.DataFrame):
        st.markdown("### Capital Redeployment Engine")
        redeploy = df[df["recommendation"].isin(["Redeploy", "Eliminate", "Trim", "Hedge"])].sort_values(["risk_score", "weight_pct"], ascending=[True, False]).head(self.top_n).copy()
        preserved = df[df["recommendation"].isin(["Hold", "Add"])].sort_values("risk_score", ascending=False).head(self.top_n).copy()
        
        redeploy["Source"] = redeploy["ticker"].apply(get_broker_dot)
        preserved["Source"] = preserved["ticker"].apply(get_broker_dot)
        redeploy["ticker"] = redeploy["ticker"].apply(clean_ticker_for_lookup)
        preserved["ticker"] = preserved["ticker"].apply(clean_ticker_for_lookup)
        
        left, right = st.columns(2)
        with left:
            st.markdown("#### Weak Capital Efficiency")
            st.dataframe(
                redeploy[["Source", "ticker", "sector", "gross_contracts", "entry_value", "current_value", "gross_exposure", "profit_value", "theta", "risk_score", "recommendation"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Source": st.column_config.TextColumn("Source", width="small"),
                },
            )
        with right:
            st.markdown("#### Better Current Setups")
            st.dataframe(
                preserved[["Source", "ticker", "sector", "gross_contracts", "entry_value", "current_value", "gross_exposure", "profit_value", "theta", "risk_score", "recommendation"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Source": st.column_config.TextColumn("Source", width="small"),
                },
            )

    def _render_ai_manager(self, df: pd.DataFrame, totals: dict[str, float], regime: dict[str, float | str], heat: float, commentary: str, metadata: dict):
        st.markdown("### AI Portfolio Manager")
        question = st.selectbox(
            "Ask the portfolio manager",
            [
                "What is my biggest hidden risk?",
                "What should I trim?",
                "What positions are theta traps?",
                "Where can I redeploy capital?",
            ],
            key="prm_ai_question",
        )
        answer = self._answer_question(question, df, totals, regime, heat)
        st.success(answer)
        st.markdown("### Executive Commentary")
        st.text_area("Commentary", value=commentary, height=180, key="prm_commentary")
        report = self._build_pdf_report(df, totals, regime, heat, commentary, metadata)
        st.download_button("Export PDF Executive Report", data=report, file_name=f"portfolio_risk_report_{metadata.get('snapshot_date', datetime.now().date())}.pdf", mime="application/pdf", use_container_width=True)

    def _answer_question(self, question: str, df: pd.DataFrame, totals: dict[str, float], regime: dict[str, float | str], heat: float) -> str:
        if "hidden risk" in question:
            sector = df.groupby("sector")["weight_pct"].sum().sort_values(ascending=False).head(1)
            return f"Biggest hidden risk is {sector.index[0]} clustering at {_fmt_pct(sector.iloc[0])} of gross exposure, with portfolio heat at {heat:.0f}/100."
        if "trim" in question:
            trim = df[df["recommendation"].isin(["Trim", "Eliminate"])].head(5)
            names = ", ".join(trim["ticker"].tolist()) if not trim.empty else "no urgent trim candidates"
            return f"Trim focus: {names}. These names combine lower scores, concentration, gamma, or negative day pressure."
        if "theta traps" in question:
            traps = df[(df["theta"] < 0) & (df["dte"] < 15)].head(5)
            names = ", ".join(traps["ticker"].tolist()) if not traps.empty else "no near-term theta traps detected"
            return f"Theta trap scan: {names}. Net theta is {totals['total_theta']:.1f} per day."
        redeploy = df[df["recommendation"].isin(["Redeploy", "Eliminate"])].head(5)
        names = ", ".join(redeploy["ticker"].tolist()) if not redeploy.empty else "no obvious redeployment source"
        return f"Redeployment source: {names}. Prefer reducing weak score positions before adding exposure in a {regime['risk_level']} regime."

    def _build_pdf_report(self, df: pd.DataFrame, totals: dict[str, float], regime: dict[str, float | str], heat: float, commentary: str, metadata: dict) -> bytes:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Portfolio Performance & Risk Management", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, f"Snapshot: {metadata.get('snapshot_ts', '')}", ln=True)
        pdf.cell(0, 7, f"Regime: {regime['regime']} | Heat: {heat:.0f}/100", ln=True)
        pdf.cell(0, 7, f"Value: {fmt_money(totals['total_market_value'])} | Day P/L: {fmt_money(totals['total_pl_day'])}", ln=True)
        pdf.ln(4)
        
        # Clean commentary text by replacing emoji bullets with text names for Helvetica compatibility
        clean_commentary = commentary.replace("🔵", "[Schwab]").replace("🔴", "[Tasty]").replace("⚪", "[Other]")
        pdf.multi_cell(0, 6, clean_commentary.encode("latin-1", "replace").decode("latin-1"))
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Top Risk Actions", ln=True)
        pdf.set_font("Helvetica", "", 9)
        
        for row in df.head(10).itertuples():
            ticker_pdf = str(row.ticker).replace("🔵", "[Schwab]").replace("🔴", "[Tasty]").replace("⚪", "[Other]")
            pdf.cell(0, 6, f"{ticker_pdf}: {row.recommendation} | Score {row.risk_score:.0f} | {row.risk_band}", ln=True)
            
        raw = pdf.output(dest="S")
        if isinstance(raw, str):
            return raw.encode("latin-1")
        if isinstance(raw, bytearray):
            return bytes(raw)
        return raw

    def _render_history(self):
        history = get_portfolio_history(days=90)
        if history.empty:
            st.info("No saved portfolio snapshots yet. Upload a daily CSV to begin the risk history.")
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["snapshot_ts"], y=history["total_pl_open"], name="Open P/L", mode="lines+markers", line=dict(color=BRAND["green"])))
        fig.add_trace(go.Scatter(x=history["snapshot_ts"], y=history["total_theta"], name="Theta", mode="lines+markers", line=dict(color=BRAND["yellow"]), yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
        style_figure(fig, height=360)
        st.plotly_chart(fig, use_container_width=True, theme=None)

    def _render_database_status(self):
        status = get_database_status()
        with st.expander("Portfolio Storage", expanded=False):
            st.caption(status["db_path"])
            c1, c2 = st.columns(2)
            c1.metric("Saved Runs", f"{status['run_count']:,}")
            c2.metric("Saved Positions", f"{status['position_count']:,}")
            recent = get_recent_portfolio_snapshots(limit=5)
            if not recent.empty:
                st.dataframe(recent, use_container_width=True, hide_index=True)

    def _render_welcome(self):
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg, rgba(26,58,143,0.26), rgba(58,181,74,0.08));
                border:1px solid {BRAND['grid']};border-left:4px solid {BRAND['green']};
                border-radius:8px;padding:22px 24px;margin:12px 0 22px 0;">
                <div style="color:{BRAND['green']};font-size:18px;font-weight:700;margin-bottom:8px;">Daily Options Risk Operating System</div>
                <div style="color:{BRAND['muted']};font-size:14px;line-height:1.7;">
                    Upload the same daily portfolio CSV used by Portfolio Performance, or load the latest saved snapshot.
                    The engine will score positions, flag hidden concentration, map Greeks, and generate executive commentary.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
