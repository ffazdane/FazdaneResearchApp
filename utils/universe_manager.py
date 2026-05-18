"""
Universal ticker universe manager for FazDane Analytics.

Stores named ticker lists in config/universes.json and exposes reusable
Streamlit widgets for single-universe and multi-universe workflows.
"""

import json
import os

import streamlit as st


UNIVERSE_CONFIG_PATH = os.path.join("config", "universes.json")

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
            return defaults

        deleted_defaults = set(data.get("__deleted_defaults__", []))
        defaults = {name: val for name, val in _DEFAULTS.items() if name not in deleted_defaults}
        upgraded = defaults
        for name, val in data.items():
            if name.startswith("__"):
                continue
            if isinstance(val, list):
                upgraded[name] = {
                    "tickers": _clean_tickers(val),
                    "benchmark": "SPY",
                    "description": "",
                    "module": "general",
                }
            else:
                upgraded[name] = {
                    "tickers": _clean_tickers(val.get("tickers", [])),
                    "benchmark": str(val.get("benchmark", "SPY")).strip().upper(),
                    "description": val.get("description", ""),
                    "module": val.get("module", "general"),
                }
        return upgraded

    save_universes(defaults)
    return defaults


def save_universes(universes: dict) -> None:
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
            if data.get("module") in (module_filter, "general", None)
        )
    return sorted(universes.keys())


def get_universe(name: str) -> dict:
    return load_universes().get(
        name,
        {"tickers": [], "benchmark": "SPY", "description": "", "module": "general"},
    )


def get_tickers(name: str) -> list[str]:
    return get_universe(name).get("tickers", [])


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
    benchmark = universe_data.get("benchmark", "SPY")

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
            edit_tickers_str = st.text_area(
                "Tickers (comma or newline separated):",
                value=", ".join(selected_data.get("tickers", [])),
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
                parsed = _parse_tickers(edit_tickers_str)
                if not parsed:
                    st.error("Please enter at least one valid ticker.")
                else:
                    universes[edit_name] = {
                        "tickers": parsed,
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
                "Tickers (comma or newline separated):",
                height=130,
                key=f"{key_prefix}_new_tickers",
                placeholder="AAPL, MSFT, NVDA, TSLA",
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
                    parsed = _parse_tickers(new_tickers_str)
                    if not parsed:
                        st.error("Please enter at least one valid ticker.")
                    else:
                        universes[clean_name] = {
                            "tickers": parsed,
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
    return _clean_tickers(raw.replace("\n", ",").split(","))


def _clean_tickers(values) -> list[str]:
    cleaned = []
    for value in values:
        ticker = str(value).strip().upper()
        if ticker and ticker not in cleaned:
            cleaned.append(ticker)
    return cleaned


def _safe_key(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower() or "universe"
