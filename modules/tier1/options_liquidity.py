"""
FazDane Analytics  Tier 1
Options Liquidity Discovery Engine
Source: 05-FazDane Options Liquidity Discovery Engine.ipynb
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import logging
from modules.base_module import FazDaneModule
from utils.tastytrade_provider import (
    TastytradeProviderError,
    fetch_market_data_by_type,
    fetch_nested_option_chain,
    load_config,
)
from utils.universe_manager import render_universe_manager, get_ticker_names, format_ticker_display
from utils.options_liquidity_store import (
    get_latest_contract_snapshot,
    get_recent_snapshots,
    save_options_snapshot,
)

logger = logging.getLogger("OptionsLiquidity")

#
# DEFAULT WATCHLIST
#

DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "IWM", "GLD", "TLT",
    "AAPL", "MSFT", "NVDA", "AMZN", "TSLA",
    "GOOGL", "META", "JPM", "XLK", "XLF",
]

#
# DATA ENGINE
#

@st.cache_data(ttl=300, show_spinner=False)
def fetch_options_data(symbols: tuple, min_volume: int, min_oi: int,
                       option_types: tuple, exp_pref: str) -> pd.DataFrame:
    """
    Scan options chains for each symbol and return filtered results.
    Tastytrade and yfinance are evaluated for data completeness at scan time.
    The source with the better usable field coverage is selected for display.
    Cached for 5 minutes.
    """
    source_notes = []
    try:
        tasty_df = _fetch_tastytrade_options_data(symbols, min_volume, min_oi, option_types, exp_pref)
    except Exception as e:
        tasty_df = pd.DataFrame()
        tasty_df.attrs["active_data_source"] = f"Tastytrade failed: {e}"

    tasty_status = tasty_df.attrs.get("active_data_source", "Tastytrade unavailable")
    source_notes.append(tasty_status)
    tasty_quality = _score_options_source(tasty_df)
    should_check_yahoo = tasty_df.empty or not tasty_quality["has_iv"] or tasty_quality["score"] < 8

    yahoo_df = pd.DataFrame()
    if should_check_yahoo:
        logger.info(f"{tasty_status}; checking yfinance data quality.")
        yahoo_df = _fetch_yfinance_options_data(symbols, min_volume, min_oi, option_types, exp_pref)
        yahoo_status = yahoo_df.attrs.get("active_data_source", "yfinance unavailable")
        source_notes.append(yahoo_status)

    selected_df, selected_source = _select_best_options_source(
        {"Tastytrade": tasty_df, "yfinance": yahoo_df}
    )

    if not selected_df.empty:
        selected_quality = _score_options_source(selected_df)
        selected_df.attrs["active_data_source"] = (
            f"{selected_source} selected "
            f"(score {selected_quality['score']}, IV coverage {selected_quality['iv_coverage']:.0%})"
            + " after checking "
            + " | ".join(source_notes)
        )
        return selected_df

    if selected_df.empty:
        snapshot_df = get_latest_contract_snapshot(symbols, min_volume, min_oi, option_types, exp_pref)
        if not snapshot_df.empty:
            snapshot_status = snapshot_df.attrs.get("active_data_source", "Local snapshot fallback")
            snapshot_df.attrs["active_data_source"] = snapshot_status + " after " + " | ".join(source_notes)
            return snapshot_df

        empty = pd.DataFrame()
        empty.attrs["active_data_source"] = "No matching contracts; " + " | ".join(source_notes)
        return empty


def _expiration_bounds(exp_pref: str) -> tuple[int, int]:
    exp_label = str(exp_pref).lower()
    if "weekly" in exp_label:
        return 0, 8
    if "monthly" in exp_label:
        return 9, 45
    return 0, 365


def _fetch_tastytrade_options_data(symbols: tuple, min_volume: int, min_oi: int,
                                   option_types: tuple, exp_pref: str) -> pd.DataFrame:
    config = load_config()
    if not config.is_configured:
        empty = pd.DataFrame()
        empty.attrs["active_data_source"] = "Tastytrade not configured"
        return empty

    min_dte, max_dte = _expiration_bounds(exp_pref)
    results = []
    errors = []

    for symbol in symbols:
        try:
            tasty_chain = fetch_nested_option_chain(symbol, config=config)
            if tasty_chain.empty:
                continue

            tasty_chain = tasty_chain[
                (tasty_chain["dte"] >= min_dte) &
                (tasty_chain["dte"] <= max_dte) &
                (tasty_chain["option_type"].isin(option_types))
            ].copy()
            if tasty_chain.empty:
                continue

            enriched = _enrich_tasty_chain_with_tastytrade_quotes(symbol, tasty_chain, config)
            if enriched.empty:
                continue

            enriched["volume"] = pd.to_numeric(enriched["volume"], errors="coerce").fillna(0)
            oi_source = "open_interest" if "open_interest" in enriched.columns else "openInterest"
            enriched["open_interest"] = pd.to_numeric(enriched[oi_source], errors="coerce").fillna(0)
            enriched = enriched[
                (enriched["volume"] >= min_volume) &
                (enriched["open_interest"] >= min_oi)
            ]
            if not enriched.empty:
                results.append(enriched)
        except TastytradeProviderError as e:
            message = f"Tastytrade not available for {symbol}: {e}"
            logger.info(message)
            empty = pd.DataFrame()
            empty.attrs["active_data_source"] = message
            return empty
        except Exception as e:
            message = f"Tastytrade option fetch failed for {symbol}: {e}"
            logger.warning(message)
            errors.append(message)
            continue

    if not results:
        empty = pd.DataFrame()
        if errors:
            empty.attrs["active_data_source"] = "; ".join(errors[:3])
        else:
            empty.attrs["active_data_source"] = "Tastytrade returned no matching contracts"
        return empty

    combined = pd.concat(results, ignore_index=True)
    return _finalize_options_frame(combined)


def _enrich_tasty_chain_with_tastytrade_quotes(symbol: str, tasty_chain: pd.DataFrame, config) -> pd.DataFrame:
    spot = _fetch_tastytrade_spot(symbol, config)
    if not spot or spot == 0:
        logger.info(f"Tastytrade market data returned no spot price for {symbol}")
        return pd.DataFrame()

    candidates = tasty_chain.copy()
    candidates["spot"] = round(spot, 2)
    candidates["moneyness"] = candidates["strike"] / spot
    candidates["moneyness_distance"] = (candidates["moneyness"] - 1).abs()
    candidates = (
        candidates.sort_values(["moneyness_distance", "dte", "option_type", "strike"])
        .head(100)
        .copy()
    )

    quote_symbols = candidates["contract"].dropna().astype(str).unique().tolist()
    if not quote_symbols:
        return pd.DataFrame()

    market_data = fetch_market_data_by_type(options=quote_symbols, config=config)
    if market_data.empty:
        return pd.DataFrame()

    merged = candidates.merge(
        market_data,
        left_on="contract",
        right_on="market_symbol",
        how="inner",
        suffixes=("", "_market"),
    )
    if merged.empty:
        return pd.DataFrame()

    for col in ["volume", "open_interest", "bid", "ask", "last_price", "mark"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    if "last_price" in merged.columns:
        merged["last_price"] = merged["last_price"].fillna(merged.get("mark"))
    if "implied_volatility" in merged.columns:
        merged["iv_pct"] = (pd.to_numeric(merged["implied_volatility"], errors="coerce") * 100).round(1)
    if {"ask", "bid"}.issubset(merged.columns):
        merged["spread"] = (merged["ask"] - merged["bid"]).round(3)
        merged["spread_pct"] = (
            merged["spread"] / merged["ask"].replace(0, np.nan) * 100
        ).round(1)

    merged.drop(columns=["moneyness_distance"], inplace=True, errors="ignore")
    merged["data_source"] = "Tastytrade API chain + Tastytrade market data"
    return merged


def _fetch_tastytrade_spot(symbol: str, config) -> float | None:
    try:
        market_data = fetch_market_data_by_type(equities=[symbol], config=config)
    except Exception as e:
        logger.warning(f"Tastytrade spot fetch failed for {symbol}: {e}")
        return None

    if market_data.empty:
        return None

    row = market_data.iloc[0]
    for col in ["last_price", "mark", "close"]:
        value = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(value) and float(value) > 0:
            return float(value)

    bid = pd.to_numeric(row.get("bid"), errors="coerce")
    ask = pd.to_numeric(row.get("ask"), errors="coerce")
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return float((bid + ask) / 2)

    return None


def _enrich_tasty_chain_with_yfinance_quotes(symbol: str, tasty_chain: pd.DataFrame) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    spot = _fetch_yfinance_spot(ticker)
    if not spot or spot == 0:
        return pd.DataFrame()

    quote_frames = []
    matched_expirations = 0
    for exp_str in sorted(tasty_chain["expiration"].dropna().unique()):
        try:
            chain = ticker.option_chain(exp_str)
        except Exception as e:
            logger.warning(f"yfinance quote enrichment failed for {symbol} {exp_str}: {e}")
            continue

        exp_has_rows = False
        for opt_type, df_opts in [("Call", chain.calls), ("Put", chain.puts)]:
            if df_opts is None or df_opts.empty:
                continue
            df_opts = df_opts.copy()
            df_opts["symbol"] = symbol
            df_opts["option_type"] = opt_type
            df_opts["expiration"] = exp_str
            quote_frames.append(df_opts)
            exp_has_rows = True

        if exp_has_rows:
            matched_expirations += 1
        if matched_expirations >= 8:
            break

    if not quote_frames:
        return pd.DataFrame()

    quotes = pd.concat(quote_frames, ignore_index=True)
    quotes["strike"] = pd.to_numeric(quotes["strike"], errors="coerce")
    tasty_chain = tasty_chain.copy()
    tasty_chain["strike"] = pd.to_numeric(tasty_chain["strike"], errors="coerce")

    merged = tasty_chain.merge(
        quotes,
        on=["symbol", "option_type", "expiration", "strike"],
        how="inner",
        suffixes=("", "_yf"),
    )
    if merged.empty:
        return pd.DataFrame()

    merged["spot"] = round(spot, 2)
    merged["moneyness"] = merged["strike"] / spot
    if "impliedVolatility" in merged.columns:
        merged["iv_pct"] = (pd.to_numeric(merged["impliedVolatility"], errors="coerce") * 100).round(1)
    if {"ask", "bid"}.issubset(merged.columns):
        merged["spread"] = (merged["ask"] - merged["bid"]).round(3)
        merged["spread_pct"] = (
            merged["spread"] / merged["ask"].replace(0, np.nan) * 100
        ).round(1)
    merged["data_source"] = "Tastytrade API chain + yfinance quotes"
    return merged


def _fetch_yfinance_spot(ticker) -> float | None:
    spot = None
    try:
        info = ticker.fast_info
        spot = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
    except Exception:
        spot = None
    if not spot or spot == 0:
        try:
            hist = ticker.history(period="5d", interval="1d")
            if not hist.empty and "Close" in hist.columns:
                spot = float(hist["Close"].dropna().iloc[-1])
        except Exception:
            spot = None
    return spot


def _fetch_yfinance_options_data(symbols: tuple, min_volume: int, min_oi: int,
                                 option_types: tuple, exp_pref: str) -> pd.DataFrame:
    results = []
    min_dte, max_dte = _expiration_bounds(exp_pref)
    notes = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            spot = _fetch_yfinance_spot(ticker)
            if not spot or spot == 0:
                notes.append(f"{symbol}: no yfinance spot price")
                continue

            expirations = ticker.options
            if not expirations:
                notes.append(f"{symbol}: no yfinance option expirations")
                continue

            today = datetime.today().date()
            matched_expirations = 0

            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if dte < min_dte or dte > max_dte:
                    continue

                chain = ticker.option_chain(exp_str)
                exp_has_results = False

                for opt_type, df_opts in [("Call", chain.calls), ("Put", chain.puts)]:
                    if opt_type not in option_types:
                        continue
                    if df_opts is None or df_opts.empty:
                        continue

                    df_opts = df_opts.copy()
                    df_opts["symbol"] = symbol
                    df_opts["option_type"] = opt_type
                    df_opts["expiration"] = exp_str
                    df_opts["dte"] = dte
                    df_opts["spot"] = round(spot, 2)

                    # Moneyness
                    df_opts["moneyness"] = df_opts["strike"] / spot

                    # IV as percentage
                    df_opts["iv_pct"] = (df_opts["impliedVolatility"] * 100).round(1)

                    # Bid-ask spread
                    df_opts["spread"] = (df_opts["ask"] - df_opts["bid"]).round(3)
                    df_opts["spread_pct"] = (
                        df_opts["spread"] / df_opts["ask"].replace(0, np.nan) * 100
                    ).round(1)

                    # Filter
                    vol_col = "volume"
                    oi_col = "openInterest"
                    df_opts[vol_col] = pd.to_numeric(df_opts[vol_col], errors="coerce").fillna(0)
                    df_opts[oi_col] = pd.to_numeric(df_opts[oi_col], errors="coerce").fillna(0)

                    df_filtered = df_opts[
                        (df_opts[vol_col] >= min_volume) &
                        (df_opts[oi_col] >= min_oi)
                    ]

                    if not df_filtered.empty:
                        df_filtered = df_filtered.copy()
                        df_filtered["data_source"] = "yfinance"
                        results.append(df_filtered)
                        exp_has_results = True

                if exp_has_results:
                    matched_expirations += 1
                if matched_expirations >= 8:
                    break

        except Exception as e:
            message = f"yfinance failed for {symbol}: {e}"
            logger.warning(message)
            notes.append(message)
            if _is_yfinance_rate_limited(e):
                notes.append("Yahoo/yfinance rate limit detected; stopped fallback scan early")
                break
            continue

    if not results:
        empty = pd.DataFrame()
        if notes:
            empty.attrs["active_data_source"] = "yfinance fallback returned no rows: " + "; ".join(notes[:5])
        else:
            empty.attrs["active_data_source"] = "yfinance fallback returned no rows after filters"
        return empty

    combined = pd.concat(results, ignore_index=True)
    return _finalize_options_frame(combined)


def _is_yfinance_rate_limited(exc: Exception) -> bool:
    text = str(exc).lower()
    return "too many requests" in text or "rate limited" in text or "yfratelimiterror" in text


def _finalize_options_frame(combined: pd.DataFrame) -> pd.DataFrame:
    combined = combined.copy()
    if "open_interest" in combined.columns and "openInterest" in combined.columns:
        combined.drop(columns=["openInterest"], inplace=True)
    if "last_price" in combined.columns and "lastPrice" in combined.columns:
        combined.drop(columns=["lastPrice"], inplace=True)

    # Select & rename columns
    keep_cols = [
        "symbol", "option_type", "expiration", "dte", "spot",
        "strike", "moneyness", "iv_pct", "volume", "openInterest",
        "open_interest", "bid", "ask", "spread", "spread_pct", "lastPrice",
        "last_price", "contract", "streamer_symbol", "data_source",
        "implied_volatility", "impliedVolatility", "delta", "gamma", "theta", "vega"
    ]
    available = [c for c in keep_cols if c in combined.columns]
    combined = combined[available].copy()

    combined.rename(columns={
        "openInterest": "open_interest",
        "lastPrice": "last_price",
        "iv_pct": "iv_%",
    }, inplace=True)

    combined = combined.sort_values("volume", ascending=False).reset_index(drop=True)
    if "data_source" in combined.columns:
        sources = ", ".join(sorted(combined["data_source"].dropna().unique()))
        combined.attrs["active_data_source"] = sources
    return combined


def _has_numeric_values(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any()


def _numeric_mean(df: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def _field_coverage(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    values = pd.to_numeric(df[column], errors="coerce")
    return float(values.notna().mean())


def _score_options_source(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "score": 0,
            "rows": 0,
            "has_iv": False,
            "iv_coverage": 0.0,
        }

    weights = {
        "iv_%": 5,
        "bid": 1,
        "ask": 1,
        "volume": 1,
        "open_interest": 1,
        "spread": 1,
        "spot": 1,
    }
    coverages = {column: _field_coverage(df, column) for column in weights}
    score = sum(weight for column, weight in weights.items() if coverages[column] > 0)
    row_bonus = min(int(len(df) / 25), 3)
    return {
        "score": score + row_bonus,
        "rows": int(len(df)),
        "has_iv": coverages["iv_%"] > 0,
        "iv_coverage": coverages["iv_%"],
    }


def _select_best_options_source(sources: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    candidates = []
    for name, df in sources.items():
        quality = _score_options_source(df)
        candidates.append((quality["score"], quality["iv_coverage"], quality["rows"], name, df))

    _, _, _, selected_name, selected_df = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return selected_df, selected_name


@st.cache_data(ttl=300, show_spinner=False)
def fetch_iv_rank(symbols: tuple) -> dict:
    """
    Estimate IV Rank for each symbol using 1-year historical close prices.
    IV Rank = (current HV30 - min HV252) / (max HV252 - min HV252) * 100
    """
    ranks = {}
    for sym in symbols:
        try:
            hist = yf.download(sym, period="1y", interval="1d",
                               progress=False, auto_adjust=True)
            if hist.empty or len(hist) < 30:
                ranks[sym] = None
                continue

            close_series = hist["Close"]
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]

            # 30-day rolling historical volatility (annualised)
            log_ret = np.log(close_series / close_series.shift(1)).dropna()
            hv = log_ret.rolling(30).std() * np.sqrt(252) * 100  # in %

            current_hv = float(hv.iloc[-1])
            hv_min = float(hv.min())
            hv_max = float(hv.max())
            rng = hv_max - hv_min

            iv_rank = round((current_hv - hv_min) / rng * 100, 1) if rng > 0 else 50.0
            ranks[sym] = iv_rank
        except Exception as e:
            logger.warning(f"IV Rank failed for {sym}: {e}")
            ranks[sym] = None
    return ranks


#
# MODULE CLASS
#

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _tema(series: pd.Series, period: int) -> pd.Series:
    ema1 = _ema(series, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3


def _calculate_trade_signal(symbol: str, hist: pd.DataFrame, period: int = 20) -> dict | None:
    required = {"Open", "High", "Low", "Close"}
    if hist.empty or not required.issubset(hist.columns):
        return None

    data = hist[["Open", "High", "Low", "Close"]].dropna().copy()
    if len(data) < max(60, period * 3):
        return None

    price = (data["High"] + data["Low"] + data["Close"]) / 3
    tma1 = _tema(price, period)
    tma2 = _tema(tma1, period)
    typical_tema = tma1 + (tma1 - tma2)

    raw_ha_close = (data["Open"] + data["High"] + data["Low"] + data["Close"]) / 4
    ha_open = pd.Series(index=data.index, dtype="float64")
    ha_open.iloc[0] = (data["High"].iloc[0] + data["Low"].iloc[0]) / 2
    for i in range(1, len(data)):
        ha_open.iloc[i] = (raw_ha_close.iloc[i - 1] + ha_open.iloc[i - 1]) / 2

    ha_close = (
        raw_ha_close
        + ha_open
        + pd.concat([data["High"], ha_open], axis=1).max(axis=1)
        + pd.concat([data["Low"], ha_open], axis=1).min(axis=1)
    ) / 4

    ha_tma1 = _tema(ha_close, period)
    ha_tma2 = _tema(ha_tma1, period)
    ha_tema = ha_tma1 + (ha_tma1 - ha_tma2)
    fdts_dev = typical_tema - ha_tema

    macd_long = _ema(data["Close"], 3) - _ema(data["Close"], 10)
    macd_long_dev = macd_long - _ema(macd_long, 16)
    macd_short = _ema(data["Close"], 12) - _ema(data["Close"], 26)
    macd_short_dev = macd_short - _ema(macd_short, 9)

    state = pd.Series(0, index=data.index, dtype="int64")
    state[(fdts_dev > 0) & (macd_long_dev > 0)] = 1
    state[(fdts_dev < 0) & (macd_short_dev < 0)] = -1

    valid = pd.DataFrame({"close": data["Close"], "state": state}).dropna()
    if valid.empty:
        return None

    current_state = int(valid["state"].iloc[-1])
    changed = valid["state"].ne(valid["state"].shift(1))
    start_idx = valid.index[changed].tolist()[-1]
    start_loc = valid.index.get_loc(start_idx)
    previous_state = int(valid["state"].iloc[start_loc - 1]) if start_loc > 0 else current_state
    current_close = float(valid["close"].iloc[-1])
    trigger_price = float(valid.loc[start_idx, "close"])
    delta = (
        current_close - trigger_price
        if current_state == 1
        else trigger_price - current_close
        if current_state == -1
        else 0.0
    )

    return {
        "Ticker": symbol,
        "Previous": "Buy" if previous_state == 1 else "Sell" if previous_state == -1 else "No Trade",
        "Signal": "Buy" if current_state == 1 else "Sell" if current_state == -1 else "No Trade",
        "Start": pd.Timestamp(start_idx).strftime("%Y-%m-%d"),
        "Days": int(len(valid) - start_loc),
        "Entry": round(trigger_price, 2),
        "Last": round(current_close, 2),
        "Delta": round(delta, 2),
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_trade_signal_buckets(symbols: tuple, period: int = 20) -> pd.DataFrame:
    symbols = tuple(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))
    if not symbols:
        return pd.DataFrame()

    rows = []
    for symbol in symbols:
        try:
            hist = yf.download(
                symbol,
                period="1y",
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
            if isinstance(hist.columns, pd.MultiIndex):
                ticker_level = hist.columns.get_level_values(-1)
                if symbol in ticker_level:
                    hist = hist.xs(symbol, axis=1, level=-1, drop_level=True)
                elif ticker_level.nunique() == 1:
                    hist = hist.droplevel(-1, axis=1)
            signal = _calculate_trade_signal(symbol, hist, period=period)
            if signal:
                rows.append(signal)
        except Exception as e:
            logger.warning(f"Trade signal failed for {symbol}: {e}")

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    signal_order = pd.CategoricalDtype(["Buy", "No Trade", "Sell"], ordered=True)
    result["Signal"] = result["Signal"].astype(signal_order)
    result["Previous"] = result["Previous"].astype(signal_order)
    return result.sort_values(["Signal", "Days", "Ticker"], ascending=[True, False, True]).reset_index(drop=True)


class OptionsLiquidityModule(FazDaneModule):
    MODULE_NAME = "Options Liquidity Discovery"
    MODULE_ICON = "OL"
    MODULE_DESCRIPTION = "Scan for high-liquidity options with elevated IV"
    TIER = 1
    SOURCE_NOTEBOOK = "05-FazDane Options Liquidity Discovery Engine.ipynb"
    CACHE_TTL = 300
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["Tastytrade API", "yfinance fallback"]

    #  Sidebar

    def render_sidebar(self):
        st.markdown("**Watchlist**")
        self.universe_name, symbols, _ = render_universe_manager(
            key_prefix="ol",
            show_benchmark=False,
            label="Ticker Universe:",
        )
        st.caption(f"{len(symbols)} symbols selected from {self.universe_name}.")

        st.markdown("**Filters**")
        min_volume = st.slider("Min Volume", 0, 5000, 500, 100, key="ol_min_vol")
        min_oi = st.slider("Min Open Interest", 0, 10000, 1000, 500, key="ol_min_oi")

        option_types = st.multiselect(
            "Option Type",
            ["Call", "Put"],
            default=["Call", "Put"],
            key="ol_types",
        )

        exp_pref = st.selectbox(
            "Expiration Window",
            ["Weekly (<=8 days)", "Monthly (9-45 days)", "Any"],
            index=1,
            key="ol_exp",
        )

        st.markdown("**Data Source**")
        config = load_config()
        st.caption(f"Tastytrade: {'configured' if config.is_configured else 'not configured'}")
        st.caption("Priority: Tastytrade API -> yfinance fallback")
        active_source = st.session_state.get("ol_active_data_source", "Not scanned yet")
        st.info(f"Active Source: {active_source}")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        scan_clicked = st.button("Scan Options", use_container_width=True,
                                 type="primary", key="ol_scan")
        export_clicked = st.button("Export CSV", use_container_width=True,
                                   key="ol_export")

        if scan_clicked:
            if not symbols:
                st.error("Enter at least one symbol.")
            elif not option_types:
                st.error("Select at least one option type.")
            else:
                st.session_state["ol_last_symbols"] = symbols
                st.session_state["ol_last_params"] = {
                    "symbols": tuple(symbols),
                    "min_volume": min_volume,
                    "min_oi": min_oi,
                    "option_types": tuple(option_types),
                    "exp_pref": exp_pref,
                }
                st.session_state.pop("ol_results", None)  # force refresh

        if export_clicked and "ol_results" in st.session_state:
            df = st.session_state["ol_results"]
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Now",
                    data=csv,
                    file_name=f"options_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    key="ol_dl",
                )

        # Summary stats in sidebar when results exist
        if "ol_results" in st.session_state:
            df = st.session_state["ol_results"]
            if not df.empty:
                st.divider()
                st.markdown("**Scan Summary**")
                st.metric("Results", f"{len(df):,}")
                if "volume" in df.columns:
                    st.metric("Avg Volume", f"{int(df['volume'].mean()):,}")
                if _has_numeric_values(df, "iv_%"):
                    st.metric("Avg IV", f"{_numeric_mean(df, 'iv_%'):.1f}%")

    #  Main

    def render_main(self):
        self.render_section_header(
            "Options Liquidity Discovery",
            "Real-time scan for high-liquidity options opportunities"
        )

        # No scan yet
        if "ol_last_params" not in st.session_state:
            self._render_welcome()
            return

        params = st.session_state["ol_last_params"]
        symbols = params["symbols"]

        # Run scan (cached)
        if "ol_results" not in st.session_state:
            with st.spinner(f"Scanning {len(symbols)} symbols"):
                df = fetch_options_data(**params)
                st.session_state["ol_results"] = df
                if df.attrs.get("active_data_source"):
                    sources = df.attrs["active_data_source"]
                elif "data_source" in df.columns and not df.empty:
                    sources = ", ".join(sorted(df["data_source"].dropna().unique()))
                else:
                    sources = df.attrs.get(
                        "active_data_source",
                        "No matching contracts; provider returned no displayable rows",
                    )
                st.session_state["ol_active_data_source"] = sources
                try:
                    snapshot = save_options_snapshot(df, params, sources)
                    st.session_state["ol_last_snapshot"] = snapshot
                    st.session_state.pop("ol_snapshot_error", None)
                except Exception as e:
                    logger.warning(f"Failed to save options liquidity snapshot: {e}")
                    st.session_state["ol_snapshot_error"] = str(e)

                # Fetch IV ranks only after a successful scan to avoid adding
                # extra Yahoo requests when providers are already throttled.
                if not df.empty:
                    rank_symbols = tuple(sorted(df["symbol"].dropna().unique())) if "symbol" in df.columns else symbols
                    with st.spinner("Calculating IV Ranks..."):
                        iv_ranks = fetch_iv_rank(rank_symbols)
                        st.session_state["ol_iv_ranks"] = iv_ranks
                else:
                    st.session_state["ol_iv_ranks"] = {}
                st.rerun()

        df = st.session_state["ol_results"]
        iv_ranks = st.session_state.get("ol_iv_ranks", {})

        if df.empty:
            st.warning(
                " No options found matching your criteria. "
                "Try lowering Min Volume / Min Open Interest or widening the expiration window."
            )
            source_status = st.session_state.get("ol_active_data_source")
            if source_status:
                st.info(f"Provider status: {source_status}")
            return

        #  Top Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Results", f"{len(df):,}")
        m2.metric("Symbols Hit", df["symbol"].nunique())
        m3.metric("Avg Volume", f"{int(df['volume'].mean()):,}" if "volume" in df.columns else "")
        m4.metric("Avg IV", f"{_numeric_mean(df, 'iv_%'):.1f}%" if _has_numeric_values(df, "iv_%") else "")
        m5.metric("Avg Spread", f"${df['spread'].mean():.2f}" if "spread" in df.columns else "")

        if "data_source" in df.columns:
            sources = ", ".join(sorted(df["data_source"].dropna().unique()))
            st.caption(f"Data source: {sources}")
        snapshot = st.session_state.get("ol_last_snapshot")
        if snapshot:
            st.caption(
                f"Saved local snapshot {snapshot['run_id']} "
                f"({snapshot['row_count']:,} rows) to {snapshot['db_path']}"
            )
        elif st.session_state.get("ol_snapshot_error"):
            st.warning(f"Snapshot save failed: {st.session_state['ol_snapshot_error']}")

        st.divider()

        #  IV Rank Banner (if available)
        valid_ranks = {k: v for k, v in iv_ranks.items() if v is not None}
        if valid_ranks:
            self._render_iv_rank_bar(valid_ranks)
            st.divider()

        #  Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["Volume Heatmap", "Ticker Drilldown", "Options Chain", "Analytics", "IV Landscape", "Snapshots"]
        )

        with tab1:
            self._tab_heatmap(df)

        with tab2:
            self._tab_ticker_drilldown(df)

        with tab3:
            self._tab_chain(df)

        with tab4:
            self._tab_analytics(df)

        with tab5:
            self._tab_iv_landscape(df, valid_ranks)

        with tab6:
            self._tab_snapshots()

    #  External Engine Interface

    def execute_options_liquidity_scan(
        self,
        tickers: list[str],
        min_volume: int = 100,
        min_oi: int = 500,
        option_types: tuple = ("Call", "Put"),
        exp_pref: str = "Monthly (9-45 days)",
        progress_bar=None,
        status_text=None,
    ) -> dict:
        """
        Callable from Calendar Strategy Matrix Quad-Engine scan.
        Runs Options Liquidity scan across tickers, saves results to SQLite.
        Returns summary dict with run metadata.
        """
        total = len(tickers)
        if status_text:
            status_text.write(f"### 🎬 Engine 4/4: Options Liquidity Discovery — scanning {total} tickers...")
        if progress_bar:
            progress_bar.progress(0.76)

        params = {
            "symbols": tuple(t.strip().upper() for t in tickers if t.strip()),
            "min_volume": min_volume,
            "min_oi": min_oi,
            "option_types": option_types,
            "exp_pref": exp_pref,
        }

        try:
            df = fetch_options_data(**params)
            sources = df.attrs.get("active_data_source", "Unknown")

            if progress_bar:
                progress_bar.progress(0.90)
            if status_text:
                status_text.write(f"💾 Saving Options Liquidity snapshot to database... (source: {sources})")

            result = save_options_snapshot(df, params, sources)

            if progress_bar:
                progress_bar.progress(1.0)
            if status_text:
                status_text.write(
                    f"✅ Engine 4/4 complete — saved {result['row_count']:,} contract rows "
                    f"({result['run_id']}) to `{result['db_path']}`"
                )
            return result

        except Exception as e:
            logger.error(f"execute_options_liquidity_scan failed: {e}", exc_info=True)
            if status_text:
                status_text.write(f"⚠️ Engine 4/4 warning — Options Liquidity scan encountered an error: {e}")
            return {}

    #  IV Rank Banner

    def _render_iv_rank_bar(self, iv_ranks: dict):
        st.markdown(
            "<div style='color:#64748b;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;'> IV Rank by Symbol (Historical Volatility Percentile)</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(min(len(iv_ranks), 8))
        for i, (sym, rank) in enumerate(sorted(iv_ranks.items(), key=lambda x: x[1], reverse=True)[:8]):
            color = "#ef4444" if rank >= 75 else "#f59e0b" if rank >= 50 else "#3ab54a"
            with cols[i]:
                st.markdown(
                    f"""
                    <div style="
                        background:rgba(21,40,71,0.8);
                        border:1px solid #1e3a5f;
                        border-top:3px solid {color};
                        border-radius:8px;
                        padding:10px;
                        text-align:center;
                    ">
                        <div style="color:#e2e8f0;font-weight:700;font-size:14px;">{sym}</div>
                        <div style="color:{color};font-size:20px;font-weight:700;">{rank:.0f}</div>
                        <div style="color:#475569;font-size:10px;">IV Rank</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    #  Tab 1: Heatmap

    def _render_trade_signal_buckets(self, symbols: list[str], focus_symbol: str | None = None, put_hedging_symbols: set[str] = None):
        st.markdown("### FDTS + MACD Trade Signal Buckets")
        if put_hedging_symbols is None:
            put_hedging_symbols = set()
            
        if put_hedging_symbols:
            st.caption("🛡️ indicates symbols with higher Put option volume than Call option volume (hedging bias on Puts).")

        filter_hedging = st.checkbox("Show only Put Hedging symbols (🛡️)", value=False, key="ol_filter_put_hedging")

        with st.spinner("Calculating daily FDTS + MACD regimes..."):
            signals = fetch_trade_signal_buckets(tuple(symbols))

        if signals.empty:
            st.info("No trade signal data available for the current symbols.")
            return

        focus_symbol = str(focus_symbol or "").strip().upper()
        if focus_symbol:
            focus_rows = signals[signals["Ticker"].astype(str).str.upper() == focus_symbol]
            if focus_rows.empty:
                st.info(f"{focus_symbol} is not available in the FDTS + MACD signal table.")
            else:
                focus = focus_rows.iloc[0]
                signal_color = {
                    "Buy": "#3ab54a",
                    "No Trade": "#f59e0b",
                    "Sell": "#ef4444",
                }.get(str(focus["Signal"]), "#94a3b8")
                
                is_focus_hedging = focus_symbol in put_hedging_symbols
                hedging_badge = (
                    f'<span style="background-color:rgba(249, 115, 22, 0.25);color:#fdba74;border:1px solid rgba(249, 115, 22, 0.6);border-radius:4px;padding:2px 6px;font-size:11px;font-weight:700;margin-left:10px;">🛡️ HEDGING PUTS</span>'
                    if is_focus_hedging
                    else ""
                )
                
                st.markdown(
                    f'<div style="border:1px solid {signal_color};border-left:5px solid {signal_color};'
                    f'background:rgba(21,40,71,0.78);border-radius:8px;padding:10px 12px;margin:4px 0 12px;">'
                    f'<span style="color:#e2e8f0;font-weight:800;">{focus_symbol}</span>'
                    f'<span style="color:{signal_color};font-weight:800;margin-left:10px;">{focus["Signal"]}</span>'
                    f'{hedging_badge}'
                    f'<span style="color:#94a3b8;margin-left:10px;">from {focus["Previous"]} on {focus["Start"]} | {focus["Days"]} days | delta {focus["Delta"]:.2f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        bucket_defs = [
            ("Buy", "#3ab54a"),
            ("No Trade", "#f59e0b"),
            ("Sell", "#ef4444"),
        ]
        cols = st.columns(3)
        for col, (bucket, color) in zip(cols, bucket_defs):
            bucket_df = signals[signals["Signal"].astype(str) == bucket].copy()
            if filter_hedging:
                bucket_df = bucket_df[bucket_df["Ticker"].astype(str).str.upper().isin(put_hedging_symbols)]
            with col:
                st.markdown(
                    f'<div style="color:{color};font-size:13px;font-weight:800;text-transform:uppercase;letter-spacing:0.5px;margin:4px 0 8px;">'
                    f'{bucket} ({len(bucket_df)})</div>',
                    unsafe_allow_html=True,
                )
                if bucket_df.empty:
                    st.caption("No symbols in this bucket.")
                    continue

                display_df = bucket_df[["Ticker", "Previous", "Start", "Days", "Entry", "Last", "Delta"]].copy()
                display_df["Ticker"] = display_df["Ticker"].apply(
                    lambda t: f"{t} 🛡️" if str(t).upper() in put_hedging_symbols else t
                )
                if focus_symbol:
                    styled_df = display_df.style.apply(
                        lambda row: [
                            "background-color: rgba(250, 204, 21, 0.22); color: #ffffff; font-weight: 800;"
                            if str(row["Ticker"]).split(" ")[0].upper() == focus_symbol
                            else ""
                            for _ in row
                        ],
                        axis=1,
                    )
                else:
                    styled_df = display_df
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker"),
                        "Previous": st.column_config.TextColumn("From"),
                        "Start": st.column_config.TextColumn("Start"),
                        "Days": st.column_config.NumberColumn("Days", format="%d"),
                        "Entry": st.column_config.NumberColumn("Entry", format="%.2f"),
                        "Last": st.column_config.NumberColumn("Last", format="%.2f"),
                        "Delta": st.column_config.NumberColumn("Delta", format="%.2f"),
                    },
                )

        rotation_signals = signals.copy()
        if filter_hedging:
            rotation_signals = rotation_signals[rotation_signals["Ticker"].astype(str).str.upper().isin(put_hedging_symbols)]
        self._render_trade_signal_rotation(rotation_signals, focus_symbol=focus_symbol, put_hedging_symbols=put_hedging_symbols)

    def _render_trade_signal_rotation(self, signals: pd.DataFrame, focus_symbol: str | None = None, put_hedging_symbols: set[str] = None):
        st.markdown("### Ticker Signal Rotation")
        if put_hedging_symbols is None:
            put_hedging_symbols = set()
            
        if signals.empty or not {"Previous", "Signal", "Ticker"}.issubset(signals.columns):
            st.info("No signal rotation data available.")
            return

        focus_symbol = str(focus_symbol or "").strip().upper()
        bucket_x = {"Buy": -1, "No Trade": 0, "Sell": 1}
        bucket_color = {"Sell": "#ef4444", "No Trade": "#f59e0b", "Buy": "#3ab54a"}
        plot_df = signals.copy()
        plot_df["PreviousText"] = plot_df["Previous"].astype(str)
        plot_df["SignalText"] = plot_df["Signal"].astype(str)
        plot_df["Changed"] = plot_df["PreviousText"] != plot_df["SignalText"]
        plot_df = plot_df.sort_values(["Changed", "SignalText", "Ticker"], ascending=[False, True, True])
        plot_df["Lane"] = list(range(len(plot_df), 0, -1))

        fig = go.Figure()
        for bucket, x in bucket_x.items():
            fig.add_vrect(
                x0=x - 0.33,
                x1=x + 0.33,
                fillcolor=bucket_color[bucket],
                opacity=0.08,
                line_width=0,
            )

        for _, row in plot_df.iterrows():
            previous = row["PreviousText"]
            current = row["SignalText"]
            color = bucket_color.get(current, "#94a3b8")
            is_focus = str(row["Ticker"]).upper() == focus_symbol
            ticker_label = f"{row['Ticker']} 🛡️" if str(row['Ticker']).upper() in put_hedging_symbols else row['Ticker']
            fig.add_trace(go.Scatter(
                x=[bucket_x.get(previous, 0), bucket_x.get(current, 0)],
                y=[row["Lane"], row["Lane"]],
                mode="lines+markers+text",
                line=dict(color="#facc15" if is_focus else color, width=5 if is_focus else 3 if row["Changed"] else 1.5, dash="solid" if is_focus else "dot"),
                marker=dict(
                    size=[12, 16] if is_focus else [8, 11],
                    color=[bucket_color.get(previous, "#94a3b8"), color],
                    line=dict(color="#facc15" if is_focus else color, width=3 if is_focus else 0),
                ),
                text=["", ticker_label],
                textposition="middle right",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "From: %{customdata[1]}<br>"
                    "Now: %{customdata[2]}<br>"
                    "Start: %{customdata[3]}<br>"
                    "Days: %{customdata[4]}<br>"
                    "Delta: %{customdata[5]:.2f}<extra></extra>"
                ),
                customdata=[[row["Ticker"], previous, current, row["Start"], row["Days"], row["Delta"]]] * 2,
                showlegend=False,
            ))

        transition_counts = (
            plot_df.groupby(["PreviousText", "SignalText"], observed=False)
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )
        subtitles = [
            f"{row.PreviousText} -> {row.SignalText}: {row.Count}"
            for row in transition_counts.itertuples(index=False)
            if row.Count > 0
        ]
        summary = " | ".join(subtitles[:6]) if subtitles else "No rotations detected"
        st.markdown(
            f"""
            <div style="
                color:#94a3b8;
                font-size:13px;
                font-weight:600;
                margin:-4px 0 10px;
            ">{summary}</div>
            """,
            unsafe_allow_html=True,
        )

        fig.update_layout(
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#0d1b2e",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(
                tickmode="array",
                tickvals=[-1, 0, 1],
                ticktext=["Buy", "No Trade", "Sell"],
                range=[-1.45, 1.75],
                gridcolor="#1e3a5f",
                zeroline=False,
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[0, max(len(plot_df) + 1, 2)],
            ),
            margin=dict(l=10, r=10, t=10, b=20),
            height=max(320, min(720, 120 + 28 * len(plot_df))),
        )
        st.plotly_chart(fig, use_container_width=True, key="ol_trade_signal_rotation")

    def _tab_heatmap(self, df: pd.DataFrame):
        st.markdown("### Volume Heatmap by Symbol & Option Type")
        if "volume" not in df.columns or "symbol" not in df.columns:
            st.info("Insufficient data for heatmap.")
            return

        full_pivot = (
            df.groupby(["symbol", "option_type"])["volume"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        # Ensure Call and Put columns exist
        for col in ["Call", "Put"]:
            if col not in full_pivot.columns:
                full_pivot[col] = 0

        # Calculate set of symbols with Put hedging bias (Put Volume > Call Volume)
        put_hedging_symbols = set(
            full_pivot[full_pivot["Put"] > full_pivot["Call"]]["symbol"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
        )

        pivot = full_pivot.sort_values(
            by=[c for c in ["Call", "Put"] if c in full_pivot.columns],
            ascending=False
        ).head(20)

        call_vals = pivot.get("Call", pd.Series([0] * len(pivot))).tolist()
        put_vals  = pivot.get("Put",  pd.Series([0] * len(pivot))).tolist()
        syms      = pivot["symbol"].tolist()
        focus_state_key = "ol_heatmap_focus_symbol"
        focus_picker_key = "ol_heatmap_focus_picker"
        selected_symbol = st.session_state.get(focus_state_key)
        if selected_symbol not in syms:
            selected_symbol = syms[0] if syms else None
            st.session_state[focus_state_key] = selected_symbol
        if st.session_state.get(focus_picker_key) not in syms:
            st.session_state[focus_picker_key] = selected_symbol

        ticker_names = get_ticker_names(getattr(self, "universe_name", "Options Default Watchlist"))
        control_cols = st.columns([2, 1])
        with control_cols[0]:
            focus_symbol = st.selectbox(
                "Focus ticker",
                syms,
                index=syms.index(selected_symbol) if selected_symbol in syms else 0,
                key=focus_picker_key,
                format_func=lambda ticker: format_ticker_display(ticker, ticker_names),
                help="Click a heatmap bar when supported, or choose a ticker here to highlight its FDTS + MACD status below.",
            )
            if focus_symbol != st.session_state.get(focus_state_key):
                st.session_state[focus_state_key] = focus_symbol

            # Show orange hedging warning if focus_symbol has higher Put volume
            if focus_symbol and focus_symbol.upper() in put_hedging_symbols:
                fs_data = full_pivot[full_pivot["symbol"] == focus_symbol]
                if not fs_data.empty:
                    c_vol = int(fs_data["Call"].values[0]) if "Call" in fs_data.columns else 0
                    p_vol = int(fs_data["Put"].values[0]) if "Put" in fs_data.columns else 0
                    st.markdown(
                        f'<div style="background:rgba(249,115,22,0.15);border:1px solid rgba(249,115,22,0.4);'
                        f'border-radius:6px;padding:6px 12px;margin-top:8px;color:#fdba74;font-size:12px;font-weight:600;">'
                        f'🛡️ <b>Hedging Bias:</b> Puts Volume ({p_vol:,}) exceeds Call Volume ({c_vol:,}) for {focus_symbol} - Hedging more on Puts.'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        with control_cols[1]:
            volume_cols = [c for c in ["Call", "Put"] if c in pivot.columns]
            focused_volume = 0
            if focus_symbol in syms and volume_cols:
                focused_volume = int(
                    pivot.loc[pivot["symbol"] == focus_symbol, volume_cols]
                    .sum(axis=1)
                    .iloc[0]
                )
            st.metric(
                "Focused Volume",
                f"{focused_volume:,}",
            )

        fig = go.Figure()
        focus_line = ["#facc15" if sym == focus_symbol else "rgba(226,232,240,0.18)" for sym in syms]
        focus_width = [4 if sym == focus_symbol else 1 for sym in syms]
        fig.add_trace(go.Bar(
            name="Calls",
            x=syms,
            y=call_vals,
            marker_color="#3ab54a",
            marker_line_color=focus_line,
            marker_line_width=focus_width,
            opacity=0.85,
            customdata=syms,
            hovertemplate="<b>%{x}</b><br>Calls: %{y:,}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="Puts",
            x=syms,
            y=put_vals,
            marker_color="#ef4444",
            marker_line_color=focus_line,
            marker_line_width=focus_width,
            opacity=0.85,
            customdata=syms,
            hovertemplate="<b>%{x}</b><br>Puts: %{y:,}<extra></extra>",
        ))

        # Check which syms have Put > Call for ticktext
        tick_labels = [f"{sym} 🛡️" if sym.upper() in put_hedging_symbols else sym for sym in syms]

        fig.update_layout(
            barmode="group",
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#0d1b2e",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(
                gridcolor="#1e3a5f",
                tickfont=dict(color="#e2e8f0", size=11),
                tickmode="array",
                tickvals=syms,
                ticktext=tick_labels,
            ),
            yaxis=dict(gridcolor="#1e3a5f", title="Total Volume", tickfont=dict(color="#e2e8f0")),
            legend=dict(
                bgcolor="rgba(21,40,71,0.85)",
                bordercolor="#1e3a5f",
                borderwidth=1,
                font=dict(color="#e2e8f0", size=13),
            ),
            margin=dict(l=0, r=0, t=35, b=0),
            height=380,
            annotations=[
                dict(
                    text="🛡️ = Put Volume > Call Volume (Hedging Bias)",
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=1.05,
                    showarrow=False,
                    font=dict(size=11, color="#f97316", family="Inter"),
                )
            ],
        )
        try:
            heatmap_event = st.plotly_chart(
                fig,
                use_container_width=True,
                key="ol_volume_heatmap",
                on_select="rerun",
                selection_mode="points",
            )
            if isinstance(heatmap_event, dict):
                selection = heatmap_event.get("selection", {})
            else:
                selection = getattr(heatmap_event, "selection", {})
            points = selection.get("points", []) if isinstance(selection, dict) else []
            if points:
                clicked_symbol = str(points[0].get("x") or points[0].get("customdata") or "").upper()
                if clicked_symbol in syms and clicked_symbol != focus_symbol:
                    st.session_state[focus_state_key] = clicked_symbol
                    st.rerun()
        except TypeError:
            st.plotly_chart(fig, use_container_width=True, key="ol_volume_heatmap_static")

        signal_symbols = sorted(df["symbol"].dropna().astype(str).str.upper().unique())
        self._render_trade_signal_buckets(signal_symbols, focus_symbol=focus_symbol, put_hedging_symbols=put_hedging_symbols)


    #  Tab 2: Chain Table

    def _tab_ticker_drilldown(self, df: pd.DataFrame):
        st.markdown("### Individual Ticker Activity")
        required = {"symbol", "expiration", "strike", "volume", "option_type"}
        if not required.issubset(df.columns):
            st.info("Insufficient option-chain fields for ticker drilldown.")
            return

        symbol_totals = df.groupby("symbol")["volume"].sum().sort_values(ascending=False)
        ticker_names = get_ticker_names(getattr(self, "universe_name", "Options Default Watchlist"))
        selected_symbol = st.selectbox(
            "Select Ticker",
            symbol_totals.index.tolist(),
            index=0,
            key="ol_drill_symbol",
            format_func=lambda ticker: format_ticker_display(ticker, ticker_names),
        )

        drill = df[df["symbol"] == selected_symbol].copy()
        if drill.empty:
            st.info("No contracts available for the selected ticker.")
            return

        drill["expiration"] = pd.to_datetime(drill["expiration"], errors="coerce").dt.strftime("%Y-%m-%d")
        drill["contract"] = (
            drill["expiration"].astype(str)
            + " "
            + drill["option_type"].astype(str).str[0]
            + " "
            + drill["strike"].map(lambda value: f"{value:g}")
        )

        total_volume = int(drill["volume"].sum())
        call_volume = int(drill.loc[drill["option_type"] == "Call", "volume"].sum())
        put_volume = int(drill.loc[drill["option_type"] == "Put", "volume"].sum())
        active_expirations = drill["expiration"].nunique()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Volume", f"{total_volume:,}")
        m2.metric("Call Volume", f"{call_volume:,}")
        m3.metric("Put Volume", f"{put_volume:,}")
        m4.metric("Expirations", f"{active_expirations:,}")

        st.markdown("#### Volume by Expiration and Strike")
        agg_map = {"volume": ("volume", "sum")}
        if "open_interest" in drill.columns:
            agg_map["open_interest"] = ("open_interest", "sum")
        activity = (
            drill.groupby(["expiration", "strike", "option_type"], as_index=False)
            .agg(**agg_map)
            .sort_values("volume", ascending=False)
        )
        hover_data = {
            "expiration": True,
            "strike": ":.2f",
            "volume": ":,",
            "option_type": True,
        }
        if "open_interest" in activity.columns:
            hover_data["open_interest"] = ":,"
        fig = px.scatter(
            activity,
            x="expiration",
            y="strike",
            size="volume",
            color="option_type",
            size_max=42,
            color_discrete_map={"Call": "#3ab54a", "Put": "#ef4444"},
            hover_data=hover_data,
            labels={"expiration": "Expiration Date", "strike": "Strike", "volume": "Volume"},
        )
        fig.update_layout(
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#152847",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
            yaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0"), title="Strike"),
            legend=dict(
                bgcolor="rgba(21,40,71,0.85)",
                bordercolor="#1e3a5f",
                borderwidth=1,
                font=dict(color="#e2e8f0", size=12),
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=440,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"ol_drill_scatter_{selected_symbol}")

        st.markdown("#### Expiration Summary")
        expiry_summary = (
            drill.groupby(["expiration", "option_type"], as_index=False)["volume"]
            .sum()
            .sort_values(["expiration", "option_type"])
        )
        fig2 = px.bar(
            expiry_summary,
            x="expiration",
            y="volume",
            color="option_type",
            barmode="stack",
            color_discrete_map={"Call": "#3ab54a", "Put": "#ef4444"},
            labels={"expiration": "Expiration Date", "volume": "Volume"},
        )
        fig2.update_layout(
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#152847",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
            yaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
            margin=dict(l=0, r=0, t=30, b=0),
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True, key=f"ol_drill_expiry_{selected_symbol}")

        st.markdown("#### Top Contracts Feeding Volume")
        top_contracts = drill.nlargest(30, "volume")
        display_cols = [
            c for c in [
                "contract", "option_type", "expiration", "dte", "strike", "volume",
                "open_interest", "iv_%", "bid", "ask", "spread", "last_price",
            ]
            if c in top_contracts.columns
        ]
        st.dataframe(top_contracts[display_cols], use_container_width=True, hide_index=True)

    def _tab_chain(self, df: pd.DataFrame):
        st.markdown("### Options Chain Results")
        if df.empty:
            st.info("No contracts are available for the current scan. Run a scan or lower the sidebar filters.")
            return

        fc1, fc2, fc3 = st.columns(3)
        symbol_options = sorted(df["symbol"].dropna().unique()) if "symbol" in df.columns else []
        sort_options = [c for c in ["volume", "open_interest", "iv_%", "spread"] if c in df.columns]
        sym_filter = fc1.multiselect(
            "Filter Symbol", symbol_options,
            key="chain_sym"
        )
        type_filter = fc2.multiselect(
            "Option Type", ["Call", "Put"],
            default=["Call", "Put"],
            key="chain_type"
        )
        sort_col = fc3.selectbox("Sort By", sort_options, key="chain_sort") if sort_options else None

        display = df.copy()
        if sym_filter and "symbol" in display.columns:
            display = display[display["symbol"].isin(sym_filter)]
        if type_filter and "option_type" in display.columns:
            display = display[display["option_type"].isin(type_filter)]
        if sort_col:
            display = display.sort_values(sort_col, ascending=False)

        st.markdown(f"*Showing {len(display):,} contracts*")
        if display.empty:
            source_status = df.attrs.get("active_data_source") or st.session_state.get("ol_active_data_source")
            st.warning("No contracts match the table filters. Clear the symbol/type filters or widen the scan filters in the sidebar.")
            if source_status:
                st.caption(f"Source status: {source_status}")
            return

        # Column config
        col_cfg = {}
        if "volume" in display.columns:
            col_cfg["volume"] = st.column_config.NumberColumn("Volume", format="%d")
        if "open_interest" in display.columns:
            col_cfg["open_interest"] = st.column_config.NumberColumn("Open Int.", format="%d")
        if "iv_%" in display.columns:
            col_cfg["iv_%"] = st.column_config.NumberColumn("IV %", format="%.1f%%")
        if "bid" in display.columns:
            col_cfg["bid"] = st.column_config.NumberColumn("Bid", format="$%.2f")
        if "ask" in display.columns:
            col_cfg["ask"] = st.column_config.NumberColumn("Ask", format="$%.2f")
        if "spread" in display.columns:
            col_cfg["spread"] = st.column_config.NumberColumn("Spread", format="$%.3f")
        if "moneyness" in display.columns:
            col_cfg["moneyness"] = st.column_config.NumberColumn("Moneyness", format="%.3f")

        st.dataframe(
            display,
            use_container_width=True,
            height=500,
            column_config=col_cfg,
        )

    #  Tab 3: Analytics

    def _tab_analytics(self, df: pd.DataFrame):
        st.markdown("### Distribution Analytics")
        col1, col2 = st.columns(2)

        with col1:
            if "volume" in df.columns:
                fig = px.histogram(
                    df[df["volume"] < df["volume"].quantile(0.95)],
                    x="volume", nbins=40,
                    title="Volume Distribution",
                    color_discrete_sequence=["#3ab54a"],
                )
                fig.update_layout(
                    paper_bgcolor="#0d1b2e", plot_bgcolor="#152847",
                    font=dict(color="#e2e8f0"),
                    xaxis=dict(gridcolor="#1e3a5f"),
                    yaxis=dict(gridcolor="#1e3a5f"),
                    margin=dict(l=0, r=0, t=40, b=0), height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "iv_%" in df.columns:
                fig2 = px.histogram(
                    df, x="iv_%", nbins=40,
                    title="IV % Distribution",
                    color_discrete_sequence=["#1a3a8f"],
                )
                fig2.update_layout(
                    paper_bgcolor="#0d1b2e", plot_bgcolor="#152847",
                    font=dict(color="#e2e8f0"),
                    xaxis=dict(gridcolor="#1e3a5f"),
                    yaxis=dict(gridcolor="#1e3a5f"),
                    margin=dict(l=0, r=0, t=40, b=0), height=300,
                )
                st.plotly_chart(fig2, use_container_width=True)

        if {"symbol", "volume", "iv_%", "option_type"}.issubset(df.columns):
            st.markdown("### Volume Source by Ticker")
            source = df.copy()
            source = source[source["volume"] > 0].sort_values("volume", ascending=False).head(350)
            hover_cols = [
                c for c in [
                    "symbol", "option_type", "expiration", "dte", "strike",
                    "volume", "open_interest", "iv_%", "bid", "ask",
                ]
                if c in source.columns
            ]
            fig3 = px.scatter(
                source,
                x="symbol",
                y="volume",
                color="option_type",
                size="volume",
                size_max=34,
                hover_data=hover_cols,
                labels={"symbol": "Ticker", "volume": "Contract Volume", "iv_%": "IV %"},
                color_discrete_map={"Call": "#3ab54a", "Put": "#ef4444"},
            )
            fig3.update_traces(
                marker=dict(opacity=0.72, line=dict(width=1, color="rgba(226,232,240,0.35)"))
            )
            fig3.update_layout(
                paper_bgcolor="#0d1b2e",
                plot_bgcolor="#152847",
                font=dict(color="#e2e8f0", family="Inter"),
                xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
                yaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
                legend=dict(
                    bgcolor="rgba(21,40,71,0.85)",
                    bordercolor="#1e3a5f",
                    borderwidth=1,
                    font=dict(color="#e2e8f0", size=12),
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                height=420,
            )
            st.plotly_chart(fig3, use_container_width=True, key="ol_analytics_ticker_source")

        # Top opportunities table
        st.markdown("###  Top Opportunities by Volume")
        top = df.nlargest(15, "volume") if "volume" in df.columns else df.head(15)
        display_cols = [c for c in ["symbol", "option_type", "strike", "expiration",
                                     "dte", "volume", "open_interest", "iv_%", "bid", "ask"] if c in top.columns]
        st.dataframe(top[display_cols], use_container_width=True, hide_index=True)

    #  Tab 4: IV Landscape

    def _tab_iv_landscape(self, df: pd.DataFrame, iv_ranks: dict):
        st.markdown("### IV Landscape")

        if iv_ranks:
            rank_df = pd.DataFrame([
                {"Symbol": sym, "IV Rank": rank}
                for sym, rank in sorted(iv_ranks.items(), key=lambda x: x[1], reverse=True)
                if rank is not None
            ])
            colors = ["#ef4444" if r >= 75 else "#f59e0b" if r >= 50 else "#3ab54a"
                      for r in rank_df["IV Rank"]]
            fig = go.Figure(go.Bar(
                x=rank_df["Symbol"],
                y=rank_df["IV Rank"],
                marker_color=colors,
                text=rank_df["IV Rank"].round(0).astype(int),
                textposition="outside",
            ))
            fig.add_hline(y=75, line_dash="dash", line_color="#ef4444",
                          annotation_text="High IV (75)", annotation_position="top right")
            fig.add_hline(y=50, line_dash="dash", line_color="#f59e0b",
                          annotation_text="Mid IV (50)", annotation_position="top right")
            fig.update_layout(
                title=dict(text="IV Rank by Symbol (Low | Mid | High)", font=dict(color="#e2e8f0")),
                paper_bgcolor="#0d1b2e", plot_bgcolor="#152847",
                font=dict(color="#e2e8f0", family="Inter"),
                xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
                yaxis=dict(gridcolor="#1e3a5f", range=[0, 105], title="IV Rank", tickfont=dict(color="#e2e8f0")),
                margin=dict(l=0, r=0, t=50, b=0), height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("IV Rank is unavailable. Production needs Yahoo/yfinance historical price access for this chart.")

        # Scatter: Volume vs IV
        if _has_numeric_values(df, "iv_%"):
            st.markdown("### Volume vs IV - Opportunity Scatter")
            fig3 = px.scatter(
                df.head(200),
                x="iv_%",
                y="volume",
                color="symbol",
                size="volume",
                size_max=28,
                hover_data=["symbol", "strike", "option_type", "dte", "bid", "ask"],
                labels={"iv_%": "Implied Volatility (%)", "volume": "Volume"},
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig3.update_layout(
                paper_bgcolor="#0d1b2e",
                plot_bgcolor="#152847",
                font=dict(color="#e2e8f0", family="Inter"),
                xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
                yaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
                legend=dict(
                    bgcolor="rgba(21,40,71,0.85)",
                    bordercolor="#1e3a5f",
                    borderwidth=1,
                    font=dict(color="#e2e8f0", size=12),
                    title=dict(font=dict(color="#94a3b8", size=11)),
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                height=360,
            )
            st.plotly_chart(fig3, use_container_width=True, key="ol_iv_volume_scatter")
        else:
            st.info(
                "Volume vs IV is unavailable because the active provider did not return implied volatility values."
            )

        # Call/Put volume by symbol
        if "symbol" in df.columns and "volume" in df.columns and "option_type" in df.columns:
            cp = df.groupby(["symbol", "option_type"])["volume"].sum().reset_index()
            cp["IV Rank"] = cp["symbol"].map(iv_ranks)
            fig2 = px.bar(
                cp, x="symbol", y="volume", color="option_type",
                title="Call vs Put Volume by Symbol",
                barmode="stack",
                hover_data={"symbol": True, "option_type": True, "volume": ":,", "IV Rank": ":.1f"},
                color_discrete_map={"Call": "#3ab54a", "Put": "#ef4444"},
            )
            fig2.update_layout(
                paper_bgcolor="#0d1b2e", plot_bgcolor="#152847",
                font=dict(color="#e2e8f0"),
                xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
                yaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#e2e8f0")),
                legend=dict(
                    bgcolor="rgba(21,40,71,0.85)",
                    bordercolor="#1e3a5f",
                    borderwidth=1,
                    font=dict(color="#e2e8f0", size=13),
                    title=dict(text="Type", font=dict(color="#94a3b8", size=11)),
                ),
                margin=dict(l=0, r=0, t=50, b=0), height=320,
            )
            st.plotly_chart(fig2, use_container_width=True)

    #  Welcome state

    def _tab_snapshots(self):
        st.markdown("### Local Snapshot Repository")
        recent = get_recent_snapshots(limit=25)
        if recent.empty:
            st.info("No local snapshots have been saved yet.")
            return

        latest = recent.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Saved Runs", f"{len(recent):,}")
        c2.metric("Latest Rows", f"{int(latest['row_count']):,}")
        c3.metric("Latest Date", str(latest["trade_date"]))

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True,
            column_config={
                "run_id": st.column_config.TextColumn("Run ID"),
                "scan_ts": st.column_config.TextColumn("Scan Time"),
                "trade_date": st.column_config.TextColumn("Trade Date"),
                "row_count": st.column_config.NumberColumn("Rows", format="%d"),
                "data_source": st.column_config.TextColumn("Data Source"),
            },
        )

    def _render_welcome(self):
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, rgba(26,58,143,0.2) 0%, rgba(58,181,74,0.08) 100%);
                border: 1px solid #1e3a5f;
                border-left: 4px solid #3ab54a;
                border-radius: 12px;
                padding: 28px 32px;
                margin: 20px 0;
            ">
                <div style="font-size:20px;margin-bottom:12px;color:#3ab54a;font-weight:700;">Options Liquidity</div>
                <div style="color:#3ab54a;font-size:20px;font-weight:700;margin-bottom:8px;">
                  Options Liquidity Discovery Engine
                </div>
                <div style="color:#94a3b8;font-size:14px;line-height:1.8;">
                    Scan your options watchlist for the highest-liquidity contracts with elevated Implied Volatility.<br><br>
                    <strong style="color:#e2e8f0;">What you get:</strong><br>
                     Volume & Open Interest heatmaps<br>
                     IV Rank percentile for each symbol<br>
                     Top contracts ranked by liquidity<br>
                     Export results to CSV
                </div>
                <div style="margin-top:20px;color:#64748b;font-size:13px;">
                     Configure your watchlist and filters in the sidebar, then click <strong> Scan Options</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick-start default symbols preview
        st.markdown("### Default Watchlist Preview")
        preview = pd.DataFrame({"Symbol": DEFAULT_SYMBOLS, "Status": ["Ready to scan"] * len(DEFAULT_SYMBOLS)})
        cols = st.columns(5)
        for i, sym in enumerate(DEFAULT_SYMBOLS):
            with cols[i % 5]:
                st.markdown(
                    f"<div style='background:rgba(21,40,71,0.6);border:1px solid #1e3a5f;border-radius:8px;padding:10px;text-align:center;margin-bottom:8px;'><span style='color:#e2e8f0;font-weight:600;'>{sym}</span></div>",
                    unsafe_allow_html=True,
                )
