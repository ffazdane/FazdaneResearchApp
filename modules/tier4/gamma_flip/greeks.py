"""Greek calculations for the Gamma Flip Line / GEX Engine."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


MIN_DAYS_TO_EXPIRATION = 1.0 / 365.0

# Sanity bounds for implied volatility (yfinance in particular can return
# stale zeros or absurd values on illiquid strikes).
IV_MIN = 0.005
IV_MAX = 5.0


def years_to_expiration(days_to_expiration: float) -> float:
    """Convert DTE to years while treating 0DTE as one trading day."""
    if days_to_expiration is None or not np.isfinite(days_to_expiration):
        return MIN_DAYS_TO_EXPIRATION
    return max(float(days_to_expiration) / 365.0, MIN_DAYS_TO_EXPIRATION)


def black_scholes_gamma(
    spot: float,
    strike: float,
    days_to_expiration: float,
    implied_volatility: float,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> float:
    """Return Black-Scholes gamma for calls/puts (scalar version)."""
    if spot <= 0 or strike <= 0 or implied_volatility <= 0:
        return 0.0

    time_years = years_to_expiration(days_to_expiration)
    sigma = max(float(implied_volatility), 1e-6)
    denom = sigma * np.sqrt(time_years)
    if denom <= 0:
        return 0.0

    d1 = (
        np.log(float(spot) / float(strike))
        + (risk_free_rate - dividend_yield + 0.5 * sigma * sigma) * time_years
    ) / denom
    return float(np.exp(-dividend_yield * time_years) * norm.pdf(d1) / (float(spot) * denom))


def black_scholes_gamma_vec(
    spot,
    strikes: np.ndarray,
    days_to_expiration: np.ndarray,
    implied_volatility: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Black-Scholes gamma.

    ``spot`` may be a scalar or an array broadcastable against the option
    arrays (e.g. a column of price levels for simulation). Invalid inputs
    (non-positive strike/IV/spot) produce gamma of 0.
    """
    spot_arr = np.asarray(spot, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    t = np.maximum(np.nan_to_num(days_to_expiration, nan=0.0) / 365.0,
                   MIN_DAYS_TO_EXPIRATION)
    sigma = np.nan_to_num(implied_volatility, nan=0.0)

    valid = (strikes > 0) & (sigma > 0) & (spot_arr > 0)
    sigma_safe = np.clip(sigma, IV_MIN, IV_MAX)
    strikes_safe = np.where(strikes > 0, strikes, 1.0)
    spot_safe = np.where(spot_arr > 0, spot_arr, 1.0)

    denom = sigma_safe * np.sqrt(t)
    d1 = (np.log(spot_safe / strikes_safe) + 0.5 * sigma_safe ** 2 * t) / denom
    pdf = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
    gamma = pdf / (spot_safe * denom)
    return np.where(valid, gamma, 0.0)


def gamma_exposure(gamma: float, open_interest: float, spot_price: float) -> float:
    """Dollar gamma per 1% move: gamma * OI * 100 (contract) * S^2 * 1%."""
    oi = max(float(open_interest or 0), 0.0)
    return float(gamma * oi * 100.0 * float(spot_price) ** 2 * 0.01)


def gamma_exposure_vec(gamma: np.ndarray, open_interest: np.ndarray, spot) -> np.ndarray:
    """Vectorized dollar gamma per 1% move."""
    oi = np.maximum(np.nan_to_num(open_interest, nan=0.0), 0.0)
    spot_arr = np.asarray(spot, dtype=float)
    return gamma * oi * 100.0 * spot_arr ** 2 * 0.01
