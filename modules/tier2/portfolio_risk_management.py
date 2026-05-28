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
)


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
    except Exception:
        vix_last = 18.0
        breadth_score = 58.0
        volatility_score = 45.0
        spy_trend = 0.8

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

        st.markdown("**Risk Controls**")
        self.top_n = st.slider("Focus list size", 3, 20, 8, key="prm_top_n")
        self.lookback_days = st.selectbox("Correlation Lookback", [30, 60, 90, 180, 365], index=2, key="prm_corr_days")
        self.heat_limit = st.slider("Portfolio Heat Alert", 40, 95, 70, key="prm_heat_limit")
        self.include_live_regime = st.checkbox("Fetch market regime", value=True, key="prm_live_regime")

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

        self._render_source_banner(metadata, source_label, saved_info)
        self._render_command_metrics(enriched, totals, regime, heat)

        tabs = st.tabs(
            [
                "Executive Command Center",
                "Position Analyzer",
                "Greeks Command Center",
                "Correlation Risk",
                "Theta Decay",
                "Profit Taking",
                "Capital Redeployment",
                "AI Portfolio Manager",
            ]
        )
        with tabs[0]:
            self._render_executive(enriched, totals, regime, heat, commentary)
        with tabs[1]:
            self._render_position_analyzer(enriched)
        with tabs[2]:
            self._render_greeks(enriched, totals, regime, heat)
        with tabs[3]:
            self._render_correlation(enriched)
        with tabs[4]:
            self._render_theta(enriched)
        with tabs[5]:
            self._render_profit_taking(enriched)
        with tabs[6]:
            self._render_redeployment(enriched)
        with tabs[7]:
            self._render_ai_manager(enriched, totals, regime, heat, commentary, metadata)

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

    def _render_command_metrics(self, df: pd.DataFrame, totals: dict[str, float], regime: dict[str, float | str], heat: float):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Portfolio Value", fmt_money(totals["total_market_value"]))
        c2.metric("Open P/L", fmt_money(totals["total_pl_open"]), fmt_money(totals["total_pl_day"]))
        c3.metric("Net Delta", fmt_num(totals["total_delta"]))
        c4.metric("Net Theta", fmt_num(totals["total_theta"]))
        c5.metric("Portfolio Heat", f"{heat:.0f}/100", "Alert" if heat >= self.heat_limit else "Normal")
        c6.metric("Market Regime", str(regime["risk_level"]), f"VIX {float(regime['vix']):.1f}" if float(regime["vix"]) else None)

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
        st.markdown("### What Happens If")
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
