"""yfinance data access for the Gamma Flip Line / GEX Engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import streamlit as st
import yfinance as yf


@dataclass(frozen=True)
class OptionChainResult:
    ticker: str
    spot_price: float
    expirations: list[str]
    chain: pd.DataFrame
    warnings: list[str]


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
def load_option_chain(ticker: str, expirations: tuple[str, ...]) -> OptionChainResult:
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
    if data.empty:
        warnings.append("Option chain loaded, but open interest and volume were missing or zero across selected expirations.")

    return OptionChainResult(symbol, spot_price, available, data, warnings)
