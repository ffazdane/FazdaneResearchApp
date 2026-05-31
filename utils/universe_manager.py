"""
Universal ticker universe manager for FazDane Analytics.

Stores named ticker lists in config/universes.json and exposes reusable
Streamlit widgets for single-universe and multi-universe workflows.
"""

import json
import os
import re

import streamlit as st


UNIVERSE_CONFIG_PATH = os.path.join("config", "universes.json")

KNOWN_TICKER_NAMES = {
    "^GSPC": "S&P 500 Index",
    "^SPX": "S&P 500 Index",
    "SPX": "S&P 500 Index",
    "^NDX": "Nasdaq 100 Index",
    "NDX": "Nasdaq 100 Index",
    "^IXIC": "Nasdaq Composite Index",
    "^DJI": "Dow Jones Industrial Average",
    "DJI": "Dow Jones Industrial Average",
    "^RUT": "Russell 2000 Index",
    "RUT": "Russell 2000 Index",
    "^NYA": "NYSE Composite Index",
    "^VIX": "CBOE Volatility Index",
    "VIX": "CBOE Volatility Index",
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "IWM": "iShares Russell 2000 ETF",
    "DIA": "SPDR Dow Jones Industrial Average ETF Trust",
    "GLD": "SPDR Gold Shares",
    "SLV": "iShares Silver Trust",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "HYG": "iShares iBoxx High Yield Corporate Bond ETF",
    "SMH": "VanEck Semiconductor ETF",
    "XLC": "Communication Services Select Sector SPDR Fund",
    "XLY": "Consumer Discretionary Select Sector SPDR Fund",
    "XLP": "Consumer Staples Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLV": "Health Care Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLB": "Materials Select Sector SPDR Fund",
    "XLRE": "Real Estate Select Sector SPDR Fund",
    "XLK": "Technology Select Sector SPDR Fund",
    "XLU": "Utilities Select Sector SPDR Fund",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "GOOGL": "Alphabet Inc.",
    "GOOG": "Alphabet Inc.",
    "META": "Meta Platforms Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "GS": "Goldman Sachs Group Inc.",
    "AVGO": "Broadcom Inc.",
    "AMD": "Advanced Micro Devices Inc.",
    "NFLX": "Netflix Inc.",
    "INTC": "Intel Corporation",
    "QCOM": "Qualcomm Inc.",
    "CSCO": "Cisco Systems Inc.",
    "AMAT": "Applied Materials Inc.",
    "COIN": "Coinbase Global Inc.",
    "HOOD": "Robinhood Markets Inc.",
    "PLTR": "Palantir Technologies Inc.",
    "IBM": "International Business Machines Corporation",
    "CRM": "Salesforce Inc.",
    "ADBE": "Adobe Inc.",
    "ORCL": "Oracle Corporation",
    "CRWD": "CrowdStrike Holdings Inc.",
    "PANW": "Palo Alto Networks Inc.",
    "UNH": "UnitedHealth Group Inc.",
    "LLY": "Eli Lilly and Company",
    "COST": "Costco Wholesale Corporation",
    "HD": "Home Depot Inc.",
    "BA": "Boeing Company",
    "CAT": "Caterpillar Inc.",
    "DDOG": "Datadog Inc.",
    "MSTR": "MicroStrategy Inc.",
    "ES=F": "E-mini S&P 500 Futures",
    "NQ=F": "E-mini Nasdaq 100 Futures",
    "YM=F": "E-mini Dow Futures",
    "RTY=F": "E-mini Russell 2000 Futures",
    "GC=F": "Gold Futures",
    "CL=F": "Crude Oil Futures",
    "BTC=F": "Bitcoin Futures",
    "DX-Y.NYB": "US Dollar Index",
    "HG=F": "Copper Futures",
}

_DEFAULTS = {
    "Options Default Watchlist": {
        "tickers": ["SPY", "QQQ", "IWM", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "META", "JPM", "XLK", "XLF"],
        "benchmark": "SPY",
        "description": "Default options liquidity scan list",
        "module": "general",
    },
    "Calendar Candidates": {
        "tickers": ["SPY", "QQQ", "IWM", "DIA", "GLD", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO", "AMD", "NFLX", "INTC", "QCOM", "CSCO", "AMAT", "COIN", "HOOD", "PLTR", "IBM", "CRM", "ADBE", "ORCL", "CRWD", "JPM", "GS", "UNH", "LLY", "COST", "HD", "BA", "CAT"],
        "benchmark": "SPY",
        "description": "Calendar spread candidate universe",
        "module": "general",
    },
    "SPX Sectors": {
        "tickers": ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"],
        "benchmark": "SPY",
        "description": "SPX Sector ETFs",
        "module": "general",
    },
    "MAG 7": {
        "tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"],
        "benchmark": "QQQ",
        "description": "Magnificent 7 mega-cap tech stocks",
        "module": "general",
    },
    "Leading ETFs": {
        "tickers": ["QQQ", "SPY", "IWM", "DIA", "SMH", "XLK", "XLF", "XLE", "GLD", "SLV", "TLT", "HYG"],
        "benchmark": "SPY",
        "description": "Leading broad-market ETFs",
        "module": "general",
    },
    "Major Indexes": {
        "tickers": ["^GSPC", "^IXIC", "^DJI", "^RUT", "^NYA", "^VIX"],
        "benchmark": "^GSPC",
        "description": "Major market indices",
        "module": "general",
    },
    "Index Universe": {
        "tickers": ["^GSPC", "^NDX", "^RUT"],
        "benchmark": "^GSPC",
        "description": "Core index universe: SPX, NDX, and RUT",
        "module": "general",
    },
    "Futures and Indexes": {
        "tickers": ["ES=F", "NQ=F", "YM=F", "RTY=F", "GC=F", "CL=F", "^VIX", "SPY", "QQQ", "IWM", "DIA"],
        "benchmark": "SPY",
        "description": "Futures, volatility, and index ETFs",
        "module": "general",
    },
    "Correlation Matrix Assets": {
        "tickers": ["SPY", "QQQ", "IWM", "^VIX", "TLT", "GLD", "CL=F", "BTC=F", "DX-Y.NYB", "HG=F"],
        "benchmark": "SPY",
        "description": "Cross-asset list for correlation matrix analysis",
        "module": "general",
    },
    "FazDane Portfolio": {
        "tickers": ["CSCO", "NVDA", "AMZN", "AAPL", "GOOG", "TSLA", "MSFT", "AMD", "AVGO", "META", "QQQ", "SPY", "GS"],
        "benchmark": "SPY",
        "description": "FazDane personal portfolio",
        "module": "general",
    },
    "Best Option Spread Tickers": {
        "tickers": ["SPY", "QQQ", "IWM", "DIA", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO", "AMD", "NFLX", "COIN", "MSTR", "HOOD", "PLTR", "CRM", "ADBE", "ORCL", "CRWD", "PANW", "JPM", "GS", "UNH", "LLY", "COST", "HD", "BA", "CAT"],
        "benchmark": "SPY",
        "description": "Highly liquid option spread candidates",
        "module": "general",
    },
    "Iron Condor Tradeables": {
        "tickers": ["^GSPC", "^NDX", "^RUT", "^DJI", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "NFLX", "AMD", "SPY", "QQQ", "IWM"],
        "benchmark": "^VIX",
        "description": "Default instruments for iron condor strategy builder",
        "module": "general",
    },
}


def load_universes() -> dict:
    """Load universes from disk and merge in any missing built-in defaults."""
    defaults = dict(_DEFAULTS)
    if os.path.exists(UNIVERSE_CONFIG_PATH):
        try:
            with open(UNIVERSE_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            st.warning("Could not read config/universes.json. Loaded built-in universes without overwriting the file.")
            return _normalize_universes(defaults)

        deleted_defaults = set(data.get("__deleted_defaults__", []))
        defaults = {name: val for name, val in _DEFAULTS.items() if name not in deleted_defaults}
        upgraded = defaults
        for name, val in data.items():
            if name.startswith("__"):
                continue
            if isinstance(val, list):
                tickers = _clean_tickers(val)
                upgraded[name] = {
                    "tickers": tickers,
                    "ticker_names": _build_ticker_names(tickers),
                    "benchmark": "SPY",
                    "description": "",
                    "module": "general",
                }
            else:
                tickers = _clean_tickers(val.get("tickers", []))
                names = _normalize_ticker_names(tickers, val.get("ticker_names", {}))
                upgraded[name] = {
                    "tickers": tickers,
                    "ticker_names": names,
                    "benchmark": str(val.get("benchmark", "SPY")).strip().upper(),
                    "description": val.get("description", ""),
                    "module": val.get("module", "general"),
                }
        return _normalize_universes(upgraded)

    save_universes(defaults)
    return _normalize_universes(defaults)


def save_universes(universes: dict) -> None:
    universes = _normalize_universes(universes)
    os.makedirs(os.path.dirname(UNIVERSE_CONFIG_PATH), exist_ok=True)
    with open(UNIVERSE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(universes, f, indent=4)


def delete_universe(name: str) -> None:
    """Delete a universe and remember deleted built-ins so they do not reappear."""
    deleted_defaults = set()
    if os.path.exists(UNIVERSE_CONFIG_PATH):
        try:
            with open(UNIVERSE_CONFIG_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
            deleted_defaults.update(existing.get("__deleted_defaults__", []))
        except Exception:
            existing = {}
    else:
        existing = {}

    universes = load_universes()
    universes.pop(name, None)
    if name in _DEFAULTS:
        deleted_defaults.add(name)

    if deleted_defaults:
        universes["__deleted_defaults__"] = sorted(deleted_defaults)
    elif "__deleted_defaults__" in existing:
        universes["__deleted_defaults__"] = existing["__deleted_defaults__"]

    save_universes(universes)


def get_universe_names(module_filter: str = None) -> list[str]:
    universes = load_universes()
    if module_filter:
        return sorted(
            name
            for name, data in universes.items()
            if not name.startswith("__") and data.get("module") in (module_filter, "general", None)
        )
    return sorted(name for name in universes.keys() if not name.startswith("__"))


def get_universe(name: str) -> dict:
    return load_universes().get(
        name,
        {"tickers": [], "benchmark": "SPY", "description": "", "module": "general"},
    )


def get_tickers(name: str) -> list[str]:
    return get_universe(name).get("tickers", [])


def get_ticker_names(name: str) -> dict:
    data = get_universe(name)
    return _normalize_ticker_names(data.get("tickers", []), data.get("ticker_names", {}))


def format_ticker_display(ticker: str, ticker_names: dict = None) -> str:
    ticker = str(ticker).strip().upper()
    name = (ticker_names or {}).get(ticker) or get_company_name(ticker)
    if _is_index_symbol(ticker):
        return name if name else ticker
    return f"{name} ({ticker})" if name and name != ticker else ticker


def get_benchmark(name: str) -> str:
    return get_universe(name).get("benchmark", "SPY")


def render_universe_manager(
    key_prefix: str = "um",
    module_filter: str = None,
    show_benchmark: bool = False,
    label: str = "Ticker Universe",
) -> tuple[str, list[str], str]:
    """Render a single-universe selector with create/edit/delete controls."""
    universes = load_universes()
    names = get_universe_names(module_filter)
    if not names:
        st.warning("No ticker universes are available.")
        return "", [], "SPY"

    universe_name = st.selectbox(label, options=names, key=f"{key_prefix}_sel")
    universe_data = universes.get(universe_name, {})
    tickers = universe_data.get("tickers", [])
    ticker_names = _normalize_ticker_names(tickers, universe_data.get("ticker_names", {}))
    benchmark = universe_data.get("benchmark", "SPY")

    if tickers:
        preview = ", ".join(format_ticker_display(ticker, ticker_names) for ticker in tickers[:8])
        suffix = f" +{len(tickers) - 8} more" if len(tickers) > 8 else ""
        st.caption(f"Selected: {preview}{suffix}")

    if show_benchmark:
        benchmark = st.text_input(
            "Benchmark Ticker:",
            value=benchmark,
            key=f"{key_prefix}_bench",
        ).strip().upper()

    _render_editor(
        key_prefix=key_prefix,
        universes=universes,
        names=names,
        selected_name=universe_name,
        module_filter=module_filter,
        show_benchmark=show_benchmark,
    )

    final_data = load_universes().get(universe_name, universe_data)
    return (
        universe_name,
        final_data.get("tickers", tickers),
        benchmark if show_benchmark else final_data.get("benchmark", benchmark),
    )


def render_universe_multiselect(
    key_prefix: str = "um_multi",
    module_filter: str = None,
    show_benchmark: bool = False,
    label: str = "Ticker Universes",
    default_names: list[str] = None,
) -> tuple[list[str], dict]:
    """Render a multi-universe selector with the same shared editor."""
    universes = load_universes()
    names = get_universe_names(module_filter)
    if not names:
        st.warning("No ticker universes are available.")
        return [], {}

    defaults = [name for name in (default_names or names[:1]) if name in names]
    selected_names = st.multiselect(
        label,
        options=names,
        default=defaults,
        key=f"{key_prefix}_multi_sel",
    )

    _render_editor(
        key_prefix=key_prefix,
        universes=universes,
        names=names,
        selected_name=selected_names[0] if selected_names else names[0],
        module_filter=module_filter,
        show_benchmark=show_benchmark,
    )

    latest = load_universes()
    selected_data = {name: latest[name] for name in selected_names if name in latest}
    return selected_names, selected_data


def _render_editor(
    key_prefix: str,
    universes: dict,
    names: list[str],
    selected_name: str,
    module_filter: str,
    show_benchmark: bool,
) -> None:
    with st.expander("Edit / Create Universe", expanded=False):
        tab_edit, tab_new, tab_delete = st.tabs(["Edit Selected", "Create New", "Delete"])

        with tab_edit:
            edit_name = st.selectbox(
                "Universe to edit:",
                options=names,
                index=names.index(selected_name) if selected_name in names else 0,
                key=f"{key_prefix}_edit_name",
            )
            selected_data = universes.get(edit_name, {})
            st.caption(f"Editing: **{edit_name}**")
            edit_key = _safe_key(edit_name)
            edit_tickers = selected_data.get("tickers", [])
            edit_names = _normalize_ticker_names(edit_tickers, selected_data.get("ticker_names", {}))
            edit_tickers_str = st.text_area(
                "Tickers (one per line, optional ': Company / Index Name'):",
                value=_format_ticker_editor_value(edit_tickers, edit_names),
                height=140,
                key=f"{key_prefix}_{edit_key}_edit_tickers",
            )
            if show_benchmark:
                edit_bench = st.text_input(
                    "Benchmark:",
                    value=selected_data.get("benchmark", "SPY"),
                    key=f"{key_prefix}_{edit_key}_edit_bench",
                ).strip().upper()
            else:
                edit_bench = selected_data.get("benchmark", "SPY")
            edit_desc = st.text_input(
                "Description:",
                value=selected_data.get("description", ""),
                key=f"{key_prefix}_{edit_key}_edit_desc",
            )

            if st.button("Save Changes", key=f"{key_prefix}_save_edit", use_container_width=True):
                parsed, parsed_names = _parse_ticker_entries(edit_tickers_str)
                if not parsed:
                    st.error("Please enter at least one valid ticker.")
                else:
                    universes[edit_name] = {
                        "tickers": parsed,
                        "ticker_names": _normalize_ticker_names(parsed, parsed_names),
                        "benchmark": edit_bench,
                        "description": edit_desc,
                        "module": selected_data.get("module", module_filter or "general"),
                    }
                    save_universes(universes)
                    st.success(f"'{edit_name}' updated with {len(parsed)} tickers.")
                    st.rerun()

        with tab_new:
            new_name = st.text_input("Universe Name:", key=f"{key_prefix}_new_name", placeholder="e.g. My Watch List")
            new_tickers_str = st.text_area(
                "Tickers (one per line, optional ': Company / Index Name'):",
                height=130,
                key=f"{key_prefix}_new_tickers",
                placeholder="AAPL\nMSFT: Microsoft Corporation\n^GSPC: S&P 500 Index",
            )
            if show_benchmark:
                new_bench = st.text_input("Benchmark:", value="SPY", key=f"{key_prefix}_new_bench").strip().upper()
            else:
                new_bench = "SPY"
            new_desc = st.text_input("Description:", key=f"{key_prefix}_new_desc", placeholder="Optional description")

            if st.button("Create Universe", key=f"{key_prefix}_create", use_container_width=True, type="primary"):
                clean_name = new_name.strip()
                if not clean_name:
                    st.error("Please enter a universe name.")
                elif clean_name in universes:
                    st.error(f"A universe named '{clean_name}' already exists. Edit it instead.")
                else:
                    parsed, parsed_names = _parse_ticker_entries(new_tickers_str)
                    if not parsed:
                        st.error("Please enter at least one valid ticker.")
                    else:
                        universes[clean_name] = {
                            "tickers": parsed,
                            "ticker_names": _normalize_ticker_names(parsed, parsed_names),
                            "benchmark": new_bench,
                            "description": new_desc,
                            "module": module_filter or "general",
                        }
                        save_universes(universes)
                        st.success(f"Universe '{clean_name}' created with {len(parsed)} tickers.")
                        st.rerun()

        with tab_delete:
            delete_name = st.selectbox(
                "Universe to delete:",
                options=names,
                index=names.index(selected_name) if selected_name in names else 0,
                key=f"{key_prefix}_delete_name",
            )
            st.warning(f"You are about to permanently delete '{delete_name}'.")
            if st.button("Delete This Universe", key=f"{key_prefix}_delete", use_container_width=True):
                if delete_name in universes:
                    delete_universe(delete_name)
                    st.success(f"'{delete_name}' deleted.")
                    st.rerun()


def _parse_tickers(raw: str) -> list[str]:
    tickers, _ = _parse_ticker_entries(raw)
    return tickers


def _parse_ticker_entries(raw: str) -> tuple[list[str], dict]:
    tickers = []
    names = {}
    entries = re.split(r"[\n,]+", raw or "")
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        if ":" in entry:
            ticker_part, name_part = entry.split(":", 1)
        elif "|" in entry:
            ticker_part, name_part = entry.split("|", 1)
        else:
            ticker_part, name_part = entry, ""

        ticker = str(ticker_part).strip().upper()
        name = str(name_part).strip()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
        if ticker and name:
            names[ticker] = name
    return tickers, names


def _clean_tickers(values) -> list[str]:
    cleaned = []
    for value in values:
        ticker = str(value).strip().upper()
        if ticker and ticker not in cleaned:
            cleaned.append(ticker)
    return cleaned


def _normalize_universes(universes: dict) -> dict:
    normalized = {}
    for name, data in universes.items():
        if name.startswith("__"):
            normalized[name] = data
            continue
        if isinstance(data, list):
            tickers = _clean_tickers(data)
            ticker_names = _build_ticker_names(tickers)
            sorted_tickers = sorted(tickers, key=lambda t: ticker_names.get(t, t).strip().lower())
            normalized[name] = {
                "tickers": sorted_tickers,
                "ticker_names": {t: ticker_names[t] for t in sorted_tickers if t in ticker_names},
                "benchmark": "SPY",
                "description": "",
                "module": "general",
            }
            continue
        tickers = _clean_tickers(data.get("tickers", []))
        ticker_names = _normalize_ticker_names(tickers, data.get("ticker_names", {}))
        sorted_tickers = sorted(tickers, key=lambda t: ticker_names.get(t, t).strip().lower())
        normalized[name] = {
            "tickers": sorted_tickers,
            "ticker_names": {t: ticker_names[t] for t in sorted_tickers if t in ticker_names},
            "benchmark": str(data.get("benchmark", "SPY")).strip().upper(),
            "description": data.get("description", ""),
            "module": data.get("module", "general"),
        }
    return normalized


def _normalize_ticker_names(tickers: list[str], existing_names: dict = None) -> dict:
    names = {}
    existing_names = existing_names or {}
    for ticker in _clean_tickers(tickers):
        existing = existing_names.get(ticker) or existing_names.get(ticker.upper())
        names[ticker] = str(existing).strip() if existing else get_company_name(ticker)
    return names


def _build_ticker_names(tickers: list[str]) -> dict:
    return {ticker: get_company_name(ticker) for ticker in _clean_tickers(tickers)}


def _format_ticker_editor_value(tickers: list[str], ticker_names: dict) -> str:
    lines = []
    for ticker in _clean_tickers(tickers):
        name = ticker_names.get(ticker) or get_company_name(ticker)
        lines.append(f"{ticker}: {name}" if name and name != ticker else ticker)
    return "\n".join(lines)


def _is_index_symbol(ticker: str) -> bool:
    ticker = str(ticker).strip().upper()
    return ticker.startswith("^") or ticker in {"SPX", "NDX", "RUT", "VIX", "DJI"}


@st.cache_data(ttl=86400, show_spinner=False)
def get_company_name(ticker: str) -> str:
    ticker = str(ticker).strip().upper()
    if not ticker:
        return ""
    if ticker in KNOWN_TICKER_NAMES:
        return KNOWN_TICKER_NAMES[ticker]
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).get_info()
        name = info.get("shortName") or info.get("longName") or info.get("displayName")
        return str(name).strip() if name else ticker
    except Exception:
        return ticker


def _safe_key(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower() or "universe"


def update_fazdane_portfolio_universe(tickers: list[str]) -> None:
    """Update 'FazDane Portfolio' universe in config/universes.json with clean tickers."""
    universes = load_universes()
    cleaned = []
    for t in tickers:
        clean_t = str(t).strip().upper()
        clean_t = clean_t.replace("🔵", "").replace("🔴", "").replace("⚪", "").strip()
        if "(" in clean_t:
            clean_t = clean_t.split("(")[0].strip()
        if clean_t and clean_t not in cleaned:
            cleaned.append(clean_t)
            
    cleaned = sorted(cleaned)
    ticker_names = {}
    for t in cleaned:
        if t in KNOWN_TICKER_NAMES:
            ticker_names[t] = KNOWN_TICKER_NAMES[t]
        else:
            ticker_names[t] = get_company_name(t)
            
    universes["FazDane Portfolio"] = {
        "tickers": cleaned,
        "ticker_names": ticker_names,
        "benchmark": "SPY",
        "description": "FazDane personal portfolio (auto-updated from portfolio upload)",
        "module": "general"
    }
    save_universes(universes)
