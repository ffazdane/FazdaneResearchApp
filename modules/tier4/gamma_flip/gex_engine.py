"""Gamma exposure aggregation and gamma flip simulation engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .greeks import black_scholes_gamma, gamma_exposure


def _regime_message(spot: float, gamma_flip: float | None, net_gex: float) -> tuple[str, str]:
    if gamma_flip is None:
        return "No Clear Flip", "No clear gamma flip detected within simulation range."

    distance_pct = abs(spot - gamma_flip) / spot * 100 if spot else 0
    if distance_pct <= 0.5:
        return "Transition Zone", "Price is near the gamma flip line; expect unstable/choppy behavior."
    if spot > gamma_flip and net_gex > 0:
        return "Positive Gamma", "Market structure favors mean reversion and lower volatility."
    if spot < gamma_flip and net_gex < 0:
        return "Negative Gamma", "Market structure favors momentum, wider ranges, and volatility expansion."
    if net_gex > 0:
        return "Positive Gamma", "Dealer gamma is net positive, but spot is not cleanly above the flip line."
    if net_gex < 0:
        return "Negative Gamma", "Dealer gamma is net negative, but spot is not cleanly below the flip line."
    return "Neutral Gamma", "Net gamma exposure is close to balanced."


def _find_zero_crossing(simulation: pd.DataFrame) -> float | None:
    if simulation.empty:
        return None
    ordered = simulation.sort_values("price_level").reset_index(drop=True)
    values = ordered["total_gex"].to_numpy(dtype=float)
    prices = ordered["price_level"].to_numpy(dtype=float)
    for idx in range(1, len(values)):
        prev_val, cur_val = values[idx - 1], values[idx]
        if prev_val == 0:
            return float(prices[idx - 1])
        if np.sign(prev_val) != np.sign(cur_val):
            denom = cur_val - prev_val
            if denom == 0:
                return float(prices[idx])
            weight = -prev_val / denom
            return float(prices[idx - 1] + weight * (prices[idx] - prices[idx - 1]))
    return None


def calculate_row_gex(chain: pd.DataFrame, spot_price: float) -> pd.DataFrame:
    data = chain.copy()
    data["gamma"] = data.apply(
        lambda row: black_scholes_gamma(
            spot=spot_price,
            strike=row["strike"],
            days_to_expiration=row["dte"],
            implied_volatility=row["impliedVolatility"],
        ),
        axis=1,
    )
    data["raw_gex"] = data.apply(
        lambda row: gamma_exposure(row["gamma"], row["openInterest"], spot_price),
        axis=1,
    )
    data["signed_gex"] = np.where(data["option_type"].eq("call"), data["raw_gex"], -data["raw_gex"])
    return data


def aggregate_by_strike(gex_rows: pd.DataFrame) -> pd.DataFrame:
    if gex_rows.empty:
        return pd.DataFrame(columns=["Strike", "Call GEX", "Put GEX", "Net GEX", "Total Open Interest", "Total Volume"])

    grouped = gex_rows.groupby(["strike", "option_type"], as_index=False).agg(
        gex=("signed_gex", "sum"),
        open_interest=("openInterest", "sum"),
        volume=("volume", "sum"),
    )
    pivot = grouped.pivot(index="strike", columns="option_type", values="gex").fillna(0)
    for col in ["call", "put"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    totals = grouped.groupby("strike").agg(open_interest=("open_interest", "sum"), volume=("volume", "sum"))
    result = pivot.join(totals).reset_index()
    result["Net GEX"] = result["call"] + result["put"]
    result = result.rename(
        columns={
            "strike": "Strike",
            "call": "Call GEX",
            "put": "Put GEX",
            "open_interest": "Total Open Interest",
            "volume": "Total Volume",
        }
    )
    return result.sort_values("Strike").reset_index(drop=True)


def aggregate_by_expiration(gex_rows: pd.DataFrame) -> pd.DataFrame:
    if gex_rows.empty:
        return pd.DataFrame(columns=["Expiration", "Call GEX", "Put GEX", "Net GEX"])

    grouped = gex_rows.groupby(["expiration", "option_type"], as_index=False)["signed_gex"].sum()
    pivot = grouped.pivot(index="expiration", columns="option_type", values="signed_gex").fillna(0)
    for col in ["call", "put"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    result = pivot.reset_index()
    result["Net GEX"] = result["call"] + result["put"]
    return result.rename(columns={"expiration": "Expiration", "call": "Call GEX", "put": "Put GEX"}).sort_values("Expiration")


def simulate_total_gex(chain: pd.DataFrame, spot_price: float, range_pct: float, step_pct: float) -> pd.DataFrame:
    if chain.empty or spot_price <= 0:
        return pd.DataFrame(columns=["price_level", "total_gex"])

    low = spot_price * (1 - range_pct / 100.0)
    high = spot_price * (1 + range_pct / 100.0)
    step = max(spot_price * (step_pct / 100.0), 0.01)
    levels = np.arange(low, high + step, step)
    rows = []
    for level in levels:
        total = 0.0
        for row in chain.itertuples(index=False):
            gamma = black_scholes_gamma(level, row.strike, row.dte, row.impliedVolatility)
            gex = gamma_exposure(gamma, row.openInterest, level)
            total += gex if row.option_type == "call" else -gex
        rows.append({"price_level": float(level), "total_gex": float(total)})
    return pd.DataFrame(rows)


def build_gex_analysis(chain: pd.DataFrame, ticker: str, spot_price: float, range_pct: float, step_pct: float) -> dict:
    gex_rows = calculate_row_gex(chain, spot_price)
    by_strike = aggregate_by_strike(gex_rows)
    by_expiration = aggregate_by_expiration(gex_rows)
    simulation = simulate_total_gex(chain, spot_price, range_pct, step_pct)

    net_gex = float(gex_rows["signed_gex"].sum()) if not gex_rows.empty else 0.0
    gamma_flip = _find_zero_crossing(simulation)
    distance_pct = None if gamma_flip is None or spot_price <= 0 else (gamma_flip - spot_price) / spot_price * 100.0
    regime, message = _regime_message(spot_price, gamma_flip, net_gex)

    call_wall = None
    put_wall = None
    peak_gamma = None
    if not by_strike.empty:
        call_wall = float(by_strike.loc[by_strike["Call GEX"].idxmax(), "Strike"])
        put_wall = float(by_strike.loc[by_strike["Put GEX"].idxmin(), "Strike"])
        peak_gamma = float(by_strike.loc[by_strike["Net GEX"].abs().idxmax(), "Strike"])

    summary = pd.DataFrame(
        [
            {
                "Ticker": ticker,
                "Spot Price": spot_price,
                "Net GEX": net_gex,
                "Gamma Flip Line": gamma_flip,
                "Distance to Flip %": distance_pct,
                "Gamma Regime": regime,
                "Call Wall": call_wall,
                "Put Wall": put_wall,
                "Peak Gamma Strike": peak_gamma,
            }
        ]
    )

    return {
        "summary": summary,
        "by_strike": by_strike,
        "by_expiration": by_expiration,
        "simulation": simulation,
        "gex_rows": gex_rows,
        "message": message,
    }

