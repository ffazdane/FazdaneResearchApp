"""Data access for the Gamma Flip Line / GEX Engine.

Two sources:
- Tastytrade (preferred when configured): production provider already used by
  the Options Liquidity module. Real-time OI / IV / marks.
- yfinance (fallback): free but OI is previous-day and IV is often stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd
import streamlit as st
import yfinance as yf

from utils.tastytrade_provider import (
    TastytradeProviderError,
    fetch_market_data_by_type,
    fetch_nested_option_chain,
    load_config,
)

IV_SANITY_MIN = 0.005
IV_SANITY_MAX = 5.0


@dataclass(frozen=True)
class OptionChainResult:
    ticker: str
    spot_price: float
    expirations: list[str]
    chain: pd.DataFrame
    warnings: list[str]
    source: str = "yfinance"


def tastytrade_configured() -> bool:
    try:
        return bool(load_config().is_configured)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Expirations
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def get_available_expirations(ticker: str) -> list[str]:
    """Return yfinance expiration strings for a ticker."""
    symbol = ticker.strip().upper()
    if not symbol:
        return []
    try:
        return list(yf.Ticker(symbol).options or [])
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def get_available_expirations_tastytrade(ticker: str) -> list[str]:
    """Return Tastytrade expiration strings for a ticker (sorted)."""
    symbol = ticker.strip().upper()
    if not symbol:
        return []
    try:
        chain = fetch_nested_option_chain(symbol)
        if chain.empty:
            return []
        return sorted(chain["expiration"].dropna().unique().tolist())
    except Exception:
        return []


def get_expirations_for_source(ticker: str, source: str) -> list[str]:
    if source == "tastytrade":
        expirations = get_available_expirations_tastytrade(ticker)
        if expirations:
            return expirations
    return get_available_expirations(ticker)


# ---------------------------------------------------------------------------
# yfinance loader
# ---------------------------------------------------------------------------

def _get_spot_price(stock: yf.Ticker) -> float | None:
    try:
        fast_info = getattr(stock, "fast_info", {})
        fast_price = fast_info.get("lastPrice") if hasattr(fast_info, "get") else fast_info["lastPrice"]
        if fast_price and fast_price > 0:
            return float(fast_price)
    except Exception:
        pass

    try:
        hist = stock.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None


def _days_to_expiration(expiration: str) -> int:
    exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    return max((exp_date - date.today()).days, 0)


def _sanitize_iv(frame: pd.DataFrame) -> pd.DataFrame:
    """Zero out absurd IV values so they contribute no gamma."""
    iv = pd.to_numeric(frame["impliedVolatility"], errors="coerce").fillna(0.0)
    frame["impliedVolatility"] = iv.where((iv >= IV_SANITY_MIN) & (iv <= IV_SANITY_MAX), 0.0)
    return frame


def _apply_strike_window(frame: pd.DataFrame, spot_price: float, strike_window_pct: float) -> pd.DataFrame:
    if frame.empty or not spot_price or strike_window_pct >= 100:
        return frame
    low = spot_price * (1 - strike_window_pct / 100.0)
    high = spot_price * (1 + strike_window_pct / 100.0)
    return frame[(frame["strike"] >= low) & (frame["strike"] <= high)].copy()


def _normalize_side(frame: pd.DataFrame, option_type: str, expiration: str, spot_price: float) -> pd.DataFrame:
    cols = ["strike", "openInterest", "impliedVolatility", "lastPrice", "bid", "ask", "volume"]
    output = frame.reindex(columns=cols).copy()
    output["expiration"] = expiration
    output["option_type"] = option_type
    output["spot_price"] = spot_price
    output["dte"] = _days_to_expiration(expiration)
    for col in cols:
        output[col] = pd.to_numeric(output[col], errors="coerce")
    output["openInterest"] = output["openInterest"].fillna(0)
    output["volume"] = output["volume"].fillna(0)
    output["impliedVolatility"] = output["impliedVolatility"].fillna(0)
    return output


@st.cache_data(ttl=300, show_spinner=False)
def load_option_chain(ticker: str, expirations: tuple[str, ...],
                      strike_window_pct: float = 100.0) -> OptionChainResult:
    """Pull selected options chains from yfinance and return a normalized table."""
    symbol = ticker.strip().upper()
    warnings: list[str] = []
    if not symbol:
        return OptionChainResult(symbol, 0.0, [], pd.DataFrame(), ["Enter a ticker to begin."])

    stock = yf.Ticker(symbol)
    try:
        available = list(stock.options or [])
    except Exception as exc:
        return OptionChainResult(symbol, 0.0, [], pd.DataFrame(), [f"Could not load option expirations for {symbol}: {exc}"])
    if not available:
        return OptionChainResult(symbol, 0.0, [], pd.DataFrame(), [f"No option expirations found for {symbol}."])

    selected = [exp for exp in expirations if exp in available]
    if not selected:
        selected = available

    spot_price = _get_spot_price(stock)
    if not spot_price:
        return OptionChainResult(symbol, 0.0, available, pd.DataFrame(), [f"Could not resolve spot price for {symbol}."])

    frames = []
    for exp in selected:
        try:
            chain = stock.option_chain(exp)
            calls = _normalize_side(chain.calls, "call", exp, spot_price)
            puts = _normalize_side(chain.puts, "put", exp, spot_price)
            frames.extend([calls, puts])
        except Exception as exc:
            warnings.append(f"Skipped {exp}: {exc}")

    if not frames:
        warnings.append(f"No usable option-chain rows found for {symbol}.")
        return OptionChainResult(symbol, spot_price, available, pd.DataFrame(), warnings)

    data = pd.concat(frames, ignore_index=True)
    data = data[data["strike"].notna() & (data["strike"] > 0)].copy()
    data = data[(data["openInterest"] > 0) | (data["volume"] > 0)].copy()
    data = _sanitize_iv(data)
    data = _apply_strike_window(data, spot_price, strike_window_pct)
    if data.empty:
        warnings.append("Option chain loaded, but open interest and volume were missing or zero across selected expirations.")

    return OptionChainResult(symbol, spot_price, available, data, warnings, source="yfinance")


# ---------------------------------------------------------------------------
# Tastytrade loader
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_option_chain_tastytrade(ticker: str, expirations: tuple[str, ...],
                                 strike_window_pct: float = 25.0) -> OptionChainResult:
    """
    Pull option chain + live market data (OI / IV / volume) from Tastytrade.

    The strike window is applied BEFORE quoting so we only request market
    data for contracts near spot (each API call covers 100 contracts).
    """
    symbol = ticker.strip().upper()
    warnings: list[str] = []
    if not symbol:
        return OptionChainResult(symbol, 0.0, [], pd.DataFrame(), ["Enter a ticker to begin."], source="tastytrade")

    config = load_config()
    if not config.is_configured:
        return OptionChainResult(symbol, 0.0, [], pd.DataFrame(),
                                 ["Tastytrade is not configured."], source="tastytrade")

    try:
        meta = fetch_nested_option_chain(symbol, config=config)
    except TastytradeProviderError as exc:
        return OptionChainResult(symbol, 0.0, [], pd.DataFrame(),
                                 [f"Tastytrade chain fetch failed: {exc}"], source="tastytrade")
    if meta.empty:
        return OptionChainResult(symbol, 0.0, [], pd.DataFrame(),
                                 [f"Tastytrade returned no option chain for {symbol}."], source="tastytrade")

    available = sorted(meta["expiration"].dropna().unique().tolist())
    selected = [exp for exp in expirations if exp in available] or available
    meta = meta[meta["expiration"].isin(selected)].copy()

    # Spot from Tastytrade equity market data.
    spot_price = None
    try:
        quote = fetch_market_data_by_type(equities=[symbol], config=config)
        if not quote.empty:
            row = quote.iloc[0]
            for col in ["last_price", "mark", "close"]:
                value = pd.to_numeric(row.get(col), errors="coerce")
                if pd.notna(value) and float(value) > 0:
                    spot_price = float(value)
                    break
    except Exception as exc:
        warnings.append(f"Tastytrade spot quote failed: {exc}")
    if not spot_price:
        return OptionChainResult(symbol, 0.0, available, pd.DataFrame(),
                                 warnings + [f"Could not resolve Tastytrade spot price for {symbol}."],
                                 source="tastytrade")

    meta = _apply_strike_window(meta, spot_price, strike_window_pct)
    if meta.empty:
        return OptionChainResult(symbol, spot_price, available, pd.DataFrame(),
                                 warnings + ["No contracts inside the strike window."], source="tastytrade")

    contracts = meta["contract"].dropna().astype(str).unique().tolist()
    try:
        market = fetch_market_data_by_type(options=contracts, config=config)
    except TastytradeProviderError as exc:
        return OptionChainResult(symbol, spot_price, available, pd.DataFrame(),
                                 warnings + [f"Tastytrade market data failed: {exc}"], source="tastytrade")
    if market.empty:
        return OptionChainResult(symbol, spot_price, available, pd.DataFrame(),
                                 warnings + ["Tastytrade returned no market data for the selected contracts."],
                                 source="tastytrade")

    merged = meta.merge(market, left_on="contract", right_on="market_symbol",
                        how="inner", suffixes=("", "_md"))
    if merged.empty:
        return OptionChainResult(symbol, spot_price, available, pd.DataFrame(),
                                 warnings + ["Could not match Tastytrade contracts to market data."],
                                 source="tastytrade")

    data = pd.DataFrame({
        "strike": pd.to_numeric(merged["strike"], errors="coerce"),
        "openInterest": pd.to_numeric(merged.get("open_interest"), errors="coerce").fillna(0),
        "impliedVolatility": pd.to_numeric(merged.get("implied_volatility"), errors="coerce").fillna(0),
        "lastPrice": pd.to_numeric(merged.get("last_price"), errors="coerce"),
        "bid": pd.to_numeric(merged.get("bid"), errors="coerce"),
        "ask": pd.to_numeric(merged.get("ask"), errors="coerce"),
        "volume": pd.to_numeric(merged.get("volume"), errors="coerce").fillna(0),
        "expiration": merged["expiration"],
        "option_type": merged["option_type"].astype(str).str.lower(),
        "spot_price": spot_price,
        "dte": pd.to_numeric(merged["dte"], errors="coerce").fillna(0).clip(lower=0),
    })
    data = data[data["strike"].notna() & (data["strike"] > 0)].copy()
    data = data[(data["openInterest"] > 0) | (data["volume"] > 0)].copy()
    data = _sanitize_iv(data)
    if data.empty:
        warnings.append("Tastytrade chain loaded, but OI and volume were zero across the window.")

    return OptionChainResult(symbol, spot_price, available, data, warnings, source="tastytrade")


def load_chain_for_source(ticker: str, expirations: tuple[str, ...], source: str,
                          strike_window_pct: float) -> OptionChainResult:
    """Load from the requested source; fall back to yfinance if Tastytrade fails."""
    if source == "tastytrade":
        result = load_option_chain_tastytrade(ticker, expirations, strike_window_pct)
        if not result.chain.empty:
            return result
        fallback = load_option_chain(ticker, expirations, strike_window_pct)
        merged_warnings = result.warnings + ["Fell back to yfinance data."] + fallback.warnings
        return OptionChainResult(fallback.ticker, fallback.spot_price, fallback.expirations,
                                 fallback.chain, merged_warnings, source=fallback.source)
    return load_option_chain(ticker, expirations, strike_window_pct)
