"""Gamma exposure aggregation and gamma flip simulation engine.

v2 changes:
- Fully vectorized gamma / GEX computation (numpy broadcasting) - the
  simulation previously looped price levels x option rows in Python and
  took minutes for SPY full chains; it now runs in well under a second.
- Gamma flip = zero crossing NEAREST to spot (profiles frequently cross
  zero more than once); all crossings are also reported.
- Call/Put walls are computed within the simulation window around spot,
  so far-dated LEAP strikes with big OI no longer hijack the walls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .greeks import black_scholes_gamma_vec, gamma_exposure_vec


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


def _find_zero_crossings(simulation: pd.DataFrame) -> list[float]:
    """Return every interpolated zero crossing of the simulated GEX curve."""
    if simulation.empty:
        return []
    ordered = simulation.sort_values("price_level").reset_index(drop=True)
    values = ordered["total_gex"].to_numpy(dtype=float)
    prices = ordered["price_level"].to_numpy(dtype=float)
    crossings: list[float] = []
    for idx in range(1, len(values)):
        prev_val, cur_val = values[idx - 1], values[idx]
        if prev_val == 0:
            crossings.append(float(prices[idx - 1]))
            continue
        if np.sign(prev_val) != np.sign(cur_val):
            denom = cur_val - prev_val
            if denom == 0:
                crossings.append(float(prices[idx]))
            else:
                weight = -prev_val / denom
                crossings.append(float(prices[idx - 1] + weight * (prices[idx] - prices[idx - 1])))
    return crossings


def _nearest_crossing(crossings: list[float], spot: float) -> float | None:
    if not crossings:
        return None
    return float(min(crossings, key=lambda level: abs(level - spot)))


def _chain_arrays(chain: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "strike": chain["strike"].to_numpy(dtype=float),
        "dte": chain["dte"].to_numpy(dtype=float),
        "iv": chain["impliedVolatility"].to_numpy(dtype=float),
        "oi": chain["openInterest"].to_numpy(dtype=float),
        "sign": np.where(chain["option_type"].eq("call").to_numpy(), 1.0, -1.0),
    }


def calculate_row_gex(chain: pd.DataFrame, spot_price: float) -> pd.DataFrame:
    """Per-contract gamma and signed GEX (vectorized)."""
    data = chain.copy()
    if data.empty:
        data["gamma"] = []
        data["raw_gex"] = []
        data["signed_gex"] = []
        return data
    arr = _chain_arrays(data)
    gamma = black_scholes_gamma_vec(spot_price, arr["strike"], arr["dte"], arr["iv"])
    raw = gamma_exposure_vec(gamma, arr["oi"], spot_price)
    data["gamma"] = gamma
    data["raw_gex"] = raw
    data["signed_gex"] = raw * arr["sign"]
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
    """
    Total signed GEX across hypothetical price levels (vectorized).

    Broadcasting shape: levels (L, 1) against option rows (1, N).
    """
    if chain.empty or spot_price <= 0:
        return pd.DataFrame(columns=["price_level", "total_gex"])

    low = spot_price * (1 - range_pct / 100.0)
    high = spot_price * (1 + range_pct / 100.0)
    step = max(spot_price * (step_pct / 100.0), 0.01)
    levels = np.arange(low, high + step, step)

    arr = _chain_arrays(chain)
    levels_col = levels[:, None]  # (L, 1)

    # Chunk option rows to bound memory for very large chains.
    n = len(arr["strike"])
    chunk = max(int(4_000_000 / max(len(levels), 1)), 500)
    totals = np.zeros(len(levels))
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        gamma = black_scholes_gamma_vec(
            levels_col,
            arr["strike"][None, start:end],
            arr["dte"][None, start:end],
            arr["iv"][None, start:end],
        )
        gex = gamma_exposure_vec(gamma, arr["oi"][None, start:end], levels_col)
        totals += (gex * arr["sign"][None, start:end]).sum(axis=1)

    return pd.DataFrame({"price_level": levels.astype(float), "total_gex": totals.astype(float)})


def build_gex_analysis(chain: pd.DataFrame, ticker: str, spot_price: float, range_pct: float, step_pct: float) -> dict:
    gex_rows = calculate_row_gex(chain, spot_price)
    by_strike = aggregate_by_strike(gex_rows)
    by_expiration = aggregate_by_expiration(gex_rows)
    simulation = simulate_total_gex(chain, spot_price, range_pct, step_pct)

    net_gex = float(gex_rows["signed_gex"].sum()) if not gex_rows.empty else 0.0
    crossings = _find_zero_crossings(simulation)
    gamma_flip = _nearest_crossing(crossings, spot_price)
    distance_pct = None if gamma_flip is None or spot_price <= 0 else (gamma_flip - spot_price) / spot_price * 100.0
    regime, message = _regime_message(spot_price, gamma_flip, net_gex)

    # Walls restricted to the simulation window so distant LEAP strikes
    # with large OI cannot hijack them.
    call_wall = None
    put_wall = None
    peak_gamma = None
    if not by_strike.empty and spot_price > 0:
        low = spot_price * (1 - range_pct / 100.0)
        high = spot_price * (1 + range_pct / 100.0)
        window = by_strike[(by_strike["Strike"] >= low) & (by_strike["Strike"] <= high)]
        if window.empty:
            window = by_strike
        call_wall = float(window.loc[window["Call GEX"].idxmax(), "Strike"])
        put_wall = float(window.loc[window["Put GEX"].idxmin(), "Strike"])
        peak_gamma = float(window.loc[window["Net GEX"].abs().idxmax(), "Strike"])

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
                "Zero Crossings": len(crossings),
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
        "all_crossings": crossings,
    }
