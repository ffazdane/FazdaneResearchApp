"""Greek calculations for the Gamma Flip Line / GEX Engine."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


MIN_DAYS_TO_EXPIRATION = 1.0 / 365.0


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
    """Return Black-Scholes gamma for calls/puts."""
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


def gamma_exposure(gamma: float, open_interest: float, spot_price: float) -> float:
    """Simplified GEX formula requested for first-version implementation."""
    oi = max(float(open_interest or 0), 0.0)
    return float(gamma * oi * 100.0 * float(spot_price) ** 2 * 0.01)

