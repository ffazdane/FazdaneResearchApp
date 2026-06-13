"""Local storage and parsing for Portfolio Performance snapshots."""

from __future__ import annotations

import csv
import logging
import hashlib
import io
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None


import pandas as pd
from utils.persistence import backup_database

logger = logging.getLogger("PortfolioPerformanceStore")


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "portfolio_performance"
DEFAULT_DB_PATH = DATA_DIR / "portfolio_performance.sqlite"
DB_PATH = Path(os.getenv("PORTFOLIO_PERFORMANCE_DB_PATH", DEFAULT_DB_PATH)).expanduser()
try:
    CENTRAL_TZ = ZoneInfo("America/Chicago") if ZoneInfo is not None else None
except Exception:
    CENTRAL_TZ = None

if CENTRAL_TZ is None:
    from datetime import timezone, timedelta
    CENTRAL_TZ = timezone(timedelta(hours=-5))



POSITION_COLUMNS = [
    "run_id",
    "snapshot_ts",
    "snapshot_date",
    "source_file",
    "account_group",
    "ticker",
    "quantity",
    "mark_price",
    "market_value",
    "theta",
    "delta",
    "gamma",
    "vega",
    "pl_open",
    "pl_day",
    "bp_effect",
]

DETAIL_COLUMNS = [
    "run_id",
    "snapshot_ts",
    "snapshot_date",
    "source_file",
    "account_group",
    "underlying",
    "row_type",
    "strategy",
    "instrument",
    "quantity",
    "days",
    "expiration",
    "strike",
    "call_put",
    "side",
    "trade_price",
    "mark_price",
    "mark_change",
    "delta",
    "theta",
    "gamma",
    "vega",
    "pl_open",
    "pl_day",
    "bp_effect",
    "is_weekly",
]


def clean_ticker_for_lookup(ticker: str) -> str:
    """Extract standard stock/ETF symbol from a potentially account-suffixed ticker."""
    ticker = str(ticker).strip().upper()
    ticker = ticker.replace("🔵", "").replace("🔴", "").replace("⚪", "").strip()
    if "(" in ticker:
        ticker = ticker.split("(")[0].strip()
    return ticker


def get_broker_dot(ticker_or_group: str) -> str:
    """Return visual color dot representation for the broker/source."""
    val = str(ticker_or_group).lower().strip()
    if not val or val in {"none", "unknown", "⚪"}:
        return "⚪"
    if "5wt" in val or "tasty" in val or "🔴" in val:
        return "🔴"
    return "🔵"


def format_ticker_for_display(ticker: str) -> str:
    """Convert any ticker format (suffixed or prefixed) to clean emoji bullet prefix format (e.g. 🔵 NVDA)."""
    raw_ticker = clean_ticker_for_lookup(ticker)
    dot = get_broker_dot(ticker)
    return f"{dot} {raw_ticker}"


def classify_option_strategy(legs: pd.DataFrame) -> str:
    """Classify the options strategy from a group of option legs."""
    if legs.empty:
        return "Position Basket"
    
    # Filter only option legs
    option_legs = legs[legs["row_type"] == "option_leg"]
    if option_legs.empty:
        return "Equity"
        
    num_legs = len(option_legs)
    expirations = option_legs["expiration"].dropna().unique()
    strikes = option_legs["strike"].dropna().unique()
    call_puts = option_legs["call_put"].dropna().unique()
    
    # If 1 leg
    if num_legs == 1:
        cp = call_puts[0] if len(call_puts) > 0 else "Option"
        side_prefix = "Long" if option_legs.iloc[0]["quantity"] > 0 else "Short"
        return f"{side_prefix} {cp}"
        
    # If 2 legs
    if num_legs == 2:
        # Same expiration, different strikes (Vertical)
        if len(expirations) == 1 and len(strikes) == 2:
            cp = call_puts[0] if len(call_puts) == 1 else "Vertical"
            if cp == "CALL":
                return "Call Vertical"
            elif cp == "PUT":
                return "Put Vertical"
            return "Vertical Spread"
            
        # Different expirations
        if len(expirations) == 2:
            cp = call_puts[0] if len(call_puts) == 1 else "Calendar"
            if cp == "CALL":
                return "Call-Calander"  # Match Schwab spelling "Call-Calander"
            elif cp == "PUT":
                return "Put-Calander"
            return "Calendar Spread"
            
    # If 4 legs (Iron Condor)
    if num_legs == 4:
        if len(call_puts) == 2 and len(expirations) == 1:
            return "Iron Condor"
            
    # Fallback/General classification
    if len(expirations) > 1:
        if len(call_puts) == 1:
            return "Call-Calander" if call_puts[0] == "CALL" else "Put-Calander"
        return "Calendar Spread"
        
    if len(strikes) > 1:
        if len(call_puts) == 1:
            return "Call Vertical" if call_puts[0] == "CALL" else "Put Vertical"
        return "Vertical Spread"
        
    return "Options Combo"


def detect_broker_type(text: str) -> str:
    """Inspect the first few lines of file text to identify Schwab vs Tastytrade."""
    first_lines = " ".join(text.splitlines()[:5])
    if "Position Statement for" in first_lines:
        return "schwab"
    if "Account,Symbol,Type,Quantity" in first_lines or "Underlying Last Price" in first_lines:
        return "tastytrade"
    return "unknown"


def extract_schwab_account(text: str) -> str:
    """Extract Schwab account number from the first line of the position statement."""
    first_lines = " ".join(text.splitlines()[:8])
    match = re.search(r"Position\s+Statement\s+for\s+([a-zA-Z0-9]+)", first_lines, flags=re.IGNORECASE)
    if match:
        acct = match.group(1).upper()
        if "SCHWAB" in acct:
            return acct
        return f"SCHWAB_{acct}"
    return "SCHWAB"


def parse_money(value: Any) -> float:
    """Parse Schwab-style money values, including commas and parentheses."""
    if value is None:
        return 0.0

    text = str(value).strip()
    if text == "" or text.upper() in {"N/A", "NA", "--"}:
        return 0.0

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]

    text = text.replace("$", "").replace(",", "").replace("+", "").strip()
    if text in {"", "-"}:
        return 0.0

    try:
        number = float(text)
    except ValueError:
        return 0.0
    return -number if negative else number


def parse_number(value: Any) -> float:
    """Parse numeric fields while tolerating blanks and Schwab placeholders."""
    if value is None:
        return 0.0

    text = str(value).strip()
    if text == "" or text.upper() in {"N/A", "NA", "--"}:
        return 0.0

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = text.replace(",", "").replace("+", "").strip()

    try:
        number = float(text)
    except ValueError:
        return 0.0
    return -number if negative else number


def parse_schwab_positions_csv(
    content: bytes | str,
    source_file: str = "uploaded_positions.csv",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract ticker-level Schwab position rows and report metadata."""
    text = _decode_content(content)
    snapshot_dt = extract_snapshot_datetime(text)
    account = extract_schwab_account(text)
    rows = _extract_positions(text, account)
    if rows.empty:
        raise ValueError("No ticker-level rows found. Check the Schwab Position Statement CSV format.")

    aggregate_map = {
        "account_group": lambda values: ", ".join(sorted({str(v) for v in values if str(v).strip()})),
        "quantity": "sum",
        "mark_price": "mean",
        "market_value": "sum",
        "theta": "sum",
        "delta": "sum",
        "gamma": "sum",
        "vega": "sum",
        "pl_open": "sum",
        "pl_day": "sum",
        "bp_effect": "sum",
    }
    positions = (
        rows.groupby("ticker", as_index=False)
        .agg(aggregate_map)
        .sort_values("pl_open", ascending=False)
        .reset_index(drop=True)
    )

    positions["snapshot_ts"] = snapshot_dt.isoformat(sep=" ")
    positions["snapshot_date"] = snapshot_dt.date().isoformat()
    positions["source_file"] = source_file

    metadata = {
        "source_file": source_file,
        "snapshot_ts": snapshot_dt.isoformat(sep=" "),
        "snapshot_date": snapshot_dt.date().isoformat(),
        "file_sha256": hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest(),
        "row_count": int(len(positions)),
        "raw_row_count": int(len(rows)),
        "detail_row_count": int(len(_extract_position_details(text, account))),
        "totals": summarize_positions(positions),
    }
    return positions, metadata


def parse_schwab_position_details_csv(
    content: bytes | str,
    source_file: str = "uploaded_positions.csv",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract raw Schwab ticker, strategy, equity description, and option-leg rows."""
    text = _decode_content(content)
    snapshot_dt = extract_snapshot_datetime(text)
    account = extract_schwab_account(text)
    details = _extract_position_details(text, account)
    if details.empty:
        raise ValueError("No detailed position rows found. Check the Schwab Position Statement CSV format.")

    details["snapshot_ts"] = snapshot_dt.isoformat(sep=" ")
    details["snapshot_date"] = snapshot_dt.date().isoformat()
    details["source_file"] = source_file

    metadata = {
        "source_file": source_file,
        "snapshot_ts": snapshot_dt.isoformat(sep=" "),
        "snapshot_date": snapshot_dt.date().isoformat(),
        "file_sha256": hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest(),
        "detail_row_count": int(len(details)),
        "option_leg_count": int((details["row_type"] == "option_leg").sum()),
        "strategy_row_count": int((details["row_type"] == "strategy").sum()),
        "summary_row_count": int((details["row_type"] == "ticker_summary").sum()),
    }
    return details, metadata


def format_tasty_exp(exp_str: str) -> str:
    """Format Tastytrade exp date (e.g. Jun 5, 2026) to Schwab format (e.g. 5 JUN 26)."""
    try:
        dt = datetime.strptime(str(exp_str).strip(), "%b %d, %Y")
        return f"{dt.day} {dt.strftime('%b').upper()} {dt.strftime('%y')}"
    except Exception:
        return str(exp_str)


def parse_tastytrade_position_details_csv(
    content: bytes | str,
    source_file: str = "uploaded_positions.csv",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract raw Tastytrade symbol, type, quantity, and option metrics into detail rows."""
    text = _decode_content(content)
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return pd.DataFrame(), {}

    account = "TASTYTRADE"
    if "Account" in df.columns and not df["Account"].empty:
        raw_acct = str(df["Account"].iloc[0]).strip().upper()
        account = raw_acct if "TASTYTRADE" in raw_acct else f"TASTYTRADE_{raw_acct}"

    rows = []
    for _, row in df.iterrows():
        symbol = str(row.get("Symbol", "")).strip().upper()
        type_str = str(row.get("Type", "")).strip().upper()

        underlying = get_underlying_ticker(symbol)
        if not underlying:
            continue

        dot = get_broker_dot(account)
        underlying_suffixed = f"{dot} {underlying}"

        qty = parse_number(row.get("Quantity"))
        dte_str = str(row.get("DTE", "")).strip().lower()
        dte = parse_number(dte_str.replace("d", ""))

        exp_date_str = str(row.get("Exp Date", ""))
        formatted_exp = format_tasty_exp(exp_date_str)
        strike = parse_number(row.get("Strike Price"))
        call_put = str(row.get("Call/Put", "")).upper()
        side = "SELL" if qty < 0 else "BUY"

        trade_price = parse_number(row.get("Trade Price"))
        mark_price = abs(parse_number(row.get("Mark")))

        if underlying == "CASH":
            instrument = "CASH"
            trade_price = 1.0
            mark_price = 1.0
        else:
            instrument = f"100 {formatted_exp} {strike:g} {call_put}" if type_str == "OPTION" else symbol

        pl_open = parse_money(row.get("P/L Open w/ Percent Bar"))
        pl_day = parse_money(row.get("P/L Day w/ Percent Bar"))

        rows.append({
            "account_group": account,
            "underlying": underlying_suffixed,
            "row_type": "option_leg" if type_str == "OPTION" else "equity_description",
            "strategy": "NONE",
            "instrument": instrument,
            "quantity": qty,
            "days": dte,
            "expiration": formatted_exp if type_str == "OPTION" else None,
            "strike": strike if type_str == "OPTION" else None,
            "call_put": call_put if type_str == "OPTION" else None,
            "side": side,
            "trade_price": trade_price,
            "mark_price": mark_price,
            "mark_change": 0.0,
            "delta": parse_number(row.get("Delta")),
            "theta": parse_number(row.get("Theta")),
            "gamma": parse_number(row.get("Gamma")),
            "vega": parse_number(row.get("Vega")),
            "pl_open": pl_open,
            "pl_day": pl_day,
            "bp_effect": 0.0,
            "is_weekly": False,
        })

    details = pd.DataFrame(rows)
    snapshot_dt = datetime.now(CENTRAL_TZ).replace(microsecond=0) if CENTRAL_TZ else datetime.now().replace(microsecond=0)
    details["snapshot_ts"] = snapshot_dt.isoformat(sep=" ")
    details["snapshot_date"] = snapshot_dt.date().isoformat()
    details["source_file"] = source_file

    metadata = {
        "source_file": source_file,
        "snapshot_ts": snapshot_dt.isoformat(sep=" "),
        "snapshot_date": snapshot_dt.date().isoformat(),
        "detail_row_count": len(details),
        "option_leg_count": int((details["row_type"] == "option_leg").sum()),
        "strategy_row_count": 0,
        "summary_row_count": 0,
    }
    return details, metadata


def parse_tastytrade_positions_csv(
    content: bytes | str,
    source_file: str = "uploaded_positions.csv",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract aggregated ticker-level position rows for Tastytrade uploads."""
    text = _decode_content(content)
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return pd.DataFrame(), {}

    account = "TASTYTRADE"
    if "Account" in df.columns and not df["Account"].empty:
        raw_acct = str(df["Account"].iloc[0]).strip().upper()
        account = raw_acct if "TASTYTRADE" in raw_acct else f"TASTYTRADE_{raw_acct}"

    details, detail_meta = parse_tastytrade_position_details_csv(content, source_file)
    if not details.empty:
        for und_suffixed, group in details[details["row_type"] == "option_leg"].groupby("underlying"):
            strategy = classify_option_strategy(group)
            details.loc[details["underlying"] == und_suffixed, "strategy"] = strategy

    if details.empty:
        return pd.DataFrame(), {}

    raw_underlyings = {}
    for _, row in df.iterrows():
        sym = str(row.get("Symbol", "")).strip().upper()
        und = get_underlying_ticker(sym)
        last_px = parse_number(row.get("Underlying Last Price"))
        if und and und != "CASH":
            raw_underlyings[und] = last_px

    rows = []
    grouped = details.groupby("underlying")
    for und_suffixed, group in grouped:
        und = clean_ticker_for_lookup(und_suffixed)
        underlying_px = raw_underlyings.get(und, 0.0)
        
        option_legs = group[group["row_type"] == "option_leg"]
        equity_legs = group[group["row_type"] == "equity_description"]
        
        if not option_legs.empty:
            strategy = option_legs["strategy"].iloc[0] if "strategy" in option_legs.columns else "Position Basket"
        else:
            strategy = "Equity" if und != "CASH" else "Cash"
            
        opt_mv = (option_legs["quantity"] * option_legs["mark_price"] * 100.0).sum()
        eq_mv = (equity_legs["quantity"] * equity_legs["mark_price"]).sum()
        total_mv = opt_mv + eq_mv
        
        if und == "CASH":
            total_mv = eq_mv

        rows.append({
            "account_group": strategy,
            "ticker": und_suffixed,
            "quantity": float(group["quantity"].sum()),
            "mark_price": float(underlying_px if und != "CASH" else 1.0),
            "market_value": float(total_mv),
            "theta": float(group["theta"].sum()),
            "delta": float(group["delta"].sum()),
            "gamma": float(group["gamma"].sum()),
            "vega": float(group["vega"].sum()),
            "pl_open": float(group["pl_open"].sum()),
            "pl_day": float(group["pl_day"].sum()),
            "bp_effect": 0.0,
        })

    positions = pd.DataFrame(rows)
    snapshot_dt = datetime.now(CENTRAL_TZ).replace(microsecond=0) if CENTRAL_TZ else datetime.now().replace(microsecond=0)
    positions["snapshot_ts"] = snapshot_dt.isoformat(sep=" ")
    positions["snapshot_date"] = snapshot_dt.date().isoformat()
    positions["source_file"] = source_file

    metadata = {
        "source_file": source_file,
        "snapshot_ts": snapshot_dt.isoformat(sep=" "),
        "snapshot_date": snapshot_dt.date().isoformat(),
        "file_sha256": hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest(),
        "row_count": len(positions),
        "raw_row_count": len(df),
        "detail_row_count": len(details),
        "totals": summarize_positions(positions),
    }
    return positions, metadata


def parse_uploaded_files(
    uploaded_files: list[tuple[bytes | str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Parse one or more uploaded CSV statements (Schwab and/or Tastytrade) and combine them."""
    if not uploaded_files:
        return pd.DataFrame(), pd.DataFrame(), {}

    positions_list = []
    details_list = []

    source_files = []
    file_hashes = []

    total_raw_rows = 0
    total_detail_rows = 0
    total_option_legs = 0
    total_strategies = 0
    total_summaries = 0

    latest_dt = None

    for content, filename in uploaded_files:
        text = _decode_content(content)
        broker = detect_broker_type(text)

        if broker == "schwab":
            pos, pos_meta = parse_schwab_positions_csv(text, filename)
            det, det_meta = parse_schwab_position_details_csv(text, filename)
        elif broker == "tastytrade":
            pos, pos_meta = parse_tastytrade_positions_csv(text, filename)
            det, det_meta = parse_tastytrade_position_details_csv(text, filename)
        else:
            raise ValueError(
                f"Unknown broker file format for: {filename}. "
                "Please upload a Schwab Position Statement CSV or Tastytrade positions CSV."
            )

        positions_list.append(pos)
        details_list.append(det)

        source_files.append(filename)
        file_hashes.append(hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest())

        total_raw_rows += pos_meta.get("raw_row_count", 0)
        total_detail_rows += det_meta.get("detail_row_count", 0)
        total_option_legs += det_meta.get("option_leg_count", 0)
        total_strategies += det_meta.get("strategy_row_count", 0)
        total_summaries += det_meta.get("summary_row_count", 0)

        # Track the most recent snapshot datetime
        dt_str = pos_meta.get("snapshot_ts")
        if dt_str:
            try:
                dt = datetime.fromisoformat(dt_str)
                if latest_dt is None or dt > latest_dt:
                    latest_dt = dt
            except Exception:
                pass

    if not positions_list:
        return pd.DataFrame(), pd.DataFrame(), {}

    combined_positions = pd.concat(positions_list, ignore_index=True)
    combined_details = pd.concat(details_list, ignore_index=True)

    if latest_dt is None:
        latest_dt = datetime.now(CENTRAL_TZ).replace(microsecond=0) if CENTRAL_TZ else datetime.now().replace(microsecond=0)

    combined_hash = hashlib.sha256(",".join(file_hashes).encode("utf-8")).hexdigest()

    combined_metadata = {
        "source_file": ", ".join(source_files),
        "snapshot_ts": latest_dt.isoformat(sep=" "),
        "snapshot_date": latest_dt.date().isoformat(),
        "file_sha256": combined_hash,
        "row_count": len(combined_positions),
        "raw_row_count": total_raw_rows,
        "detail_row_count": total_detail_rows,
        "option_leg_count": total_option_legs,
        "strategy_row_count": total_strategies,
        "summary_row_count": total_summaries,
        "totals": summarize_positions(combined_positions),
    }

    return combined_positions, combined_details, combined_metadata


def summarize_positions(df: pd.DataFrame) -> dict[str, float]:
    """Return portfolio-level totals used by the dashboard and run table."""
    if df.empty:
        return {
            "total_pl_open": 0.0,
            "total_pl_day": 0.0,
            "total_theta": 0.0,
            "total_delta": 0.0,
            "total_gamma": 0.0,
            "total_vega": 0.0,
            "total_market_value": 0.0,
            "positive_open": 0.0,
            "negative_open": 0.0,
        }

    return {
        "total_pl_open": float(df["pl_open"].sum()),
        "total_pl_day": float(df["pl_day"].sum()),
        "total_theta": float(df["theta"].sum()),
        "total_delta": float(df["delta"].sum()),
        "total_gamma": float(df["gamma"].sum()) if "gamma" in df.columns else 0.0,
        "total_vega": float(df["vega"].sum()) if "vega" in df.columns else 0.0,
        "total_market_value": float(df.get("market_value", pd.Series(dtype=float)).sum()),
        "positive_open": float(df.loc[df["pl_open"] > 0, "pl_open"].sum()),
        "negative_open": float(df.loc[df["pl_open"] < 0, "pl_open"].sum()),
    }


def summarize_position_details(details: pd.DataFrame | None) -> dict[str, float]:
    """Return portfolio totals from option-leg detail rows when available."""
    if details is None or details.empty or "row_type" not in details.columns:
        return {}

    legs = details[details["row_type"].astype(str).eq("option_leg")].copy()
    if legs.empty:
        return {}

    for column in [
        "quantity",
        "trade_price",
        "mark_price",
        "delta",
        "theta",
        "gamma",
        "vega",
        "pl_open",
        "pl_day",
    ]:
        if column not in legs.columns:
            legs[column] = 0.0
        legs[column] = pd.to_numeric(legs[column], errors="coerce").fillna(0.0)

    entry_values = legs["quantity"] * legs["trade_price"] * 100
    current_values = legs["quantity"] * legs["mark_price"] * 100
    gross_entry_values = legs["quantity"].abs() * legs["trade_price"].abs() * 100
    gross_current_values = legs["quantity"].abs() * legs["mark_price"].abs() * 100

    return {
        "total_pl_open": float(legs["pl_open"].sum()),
        "total_pl_day": float(legs["pl_day"].sum()),
        "total_theta": float(legs["theta"].sum()),
        "total_delta": float(legs["delta"].sum()),
        "total_gamma": float(legs["gamma"].sum()),
        "total_vega": float(legs["vega"].sum()),
        "total_entry_value": float(entry_values.sum()),
        "total_market_value": float(current_values.sum()),
        "total_net_profit": float(current_values.sum() - entry_values.sum()),
        "gross_entry_value": float(gross_entry_values.sum()),
        "gross_current_value": float(gross_current_values.sum()),
        "positive_open": float(legs.loc[legs["pl_open"] > 0, "pl_open"].sum()),
        "negative_open": float(legs.loc[legs["pl_open"] < 0, "pl_open"].sum()),
    }


def save_portfolio_snapshot(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    details: pd.DataFrame | None = None,
    db_path: Path = DB_PATH,
) -> dict[str, Any]:
    """Persist one uploaded portfolio snapshot and its ticker-level rows."""
    if df.empty:
        raise ValueError("Cannot save an empty portfolio snapshot.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_ts = str(metadata.get("snapshot_ts") or datetime.now(CENTRAL_TZ).isoformat(sep=" "))
    snapshot_date = str(metadata.get("snapshot_date") or snapshot_ts[:10])
    source_file = str(metadata.get("source_file") or "uploaded_positions.csv")
    totals = summarize_positions(df)
    detail_totals = summarize_position_details(details)
    if detail_totals:
        totals.update(detail_totals)

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        
        # Check if there is already a run for the same snapshot_date
        existing = conn.execute(
            "SELECT run_id FROM pp_runs WHERE snapshot_date = ?",
            (snapshot_date,)
        ).fetchone()
        
        if existing:
            run_id = existing[0]
            # Delete child records so we don't duplicate them
            conn.execute("DELETE FROM pp_position_snapshots WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM pp_position_details WHERE run_id = ?", (run_id,))
            insert_query = """
            INSERT OR REPLACE INTO pp_runs (
                run_id, snapshot_ts, snapshot_date, source_file, file_sha256,
                row_count, raw_row_count, totals_json, total_pl_open, total_pl_day,
                total_theta, total_delta, total_market_value
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        else:
            run_id = f"pp_{datetime.now(CENTRAL_TZ).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            insert_query = """
            INSERT INTO pp_runs (
                run_id, snapshot_ts, snapshot_date, source_file, file_sha256,
                row_count, raw_row_count, totals_json, total_pl_open, total_pl_day,
                total_theta, total_delta, total_market_value
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

        conn.execute(
            insert_query,
            (
                run_id,
                snapshot_ts,
                snapshot_date,
                source_file,
                metadata.get("file_sha256"),
                int(len(df)),
                int(metadata.get("raw_row_count", len(df))),
                json.dumps(totals, sort_keys=True),
                totals["total_pl_open"],
                totals["total_pl_day"],
                totals["total_theta"],
                totals["total_delta"],
                totals["total_market_value"],
            ),
        )

        positions = _prepare_position_rows(df, run_id, snapshot_ts, snapshot_date, source_file)
        positions.to_sql("pp_position_snapshots", conn, if_exists="append", index=False)
        if details is not None and not details.empty:
            detail_rows = _prepare_detail_rows(details, run_id, snapshot_ts, snapshot_date, source_file)
            detail_rows.to_sql("pp_position_details", conn, if_exists="append", index=False)

    # Sync snapshot to cloud storage
    try:
        backup_database("portfolio_performance", reason=f"Upload {source_file}")
    except Exception as e:
        logger.warning(f"Cloud backup failed for portfolio_performance: {e}")

    return {
        "run_id": run_id,
        "snapshot_ts": snapshot_ts,
        "snapshot_date": snapshot_date,
        "row_count": int(len(df)),
        "detail_row_count": int(len(details)) if details is not None else 0,
        "db_path": str(db_path),
    }


def get_recent_portfolio_snapshots(limit: int = 20, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return recent saved portfolio runs."""
    if not db_path.exists():
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        return pd.read_sql_query(
            """
            SELECT
                run_id, snapshot_ts, snapshot_date, source_file, row_count,
                total_pl_open, total_pl_day, total_theta, total_delta, total_market_value
            FROM pp_runs
            ORDER BY snapshot_ts DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )





def save_portfolio_log(
    log_date: str,
    category: str,
    content: str,
    log_id: str | None = None,
    run_id: str | None = None,
    image_data: bytes | None = None,
    clear_image: bool = False,
    snippet: str | None = None,
    images_list: list[bytes] | None = None,
    db_path: Path = DB_PATH,
) -> str:
    """Save a new portfolio log or update an existing one, triggering cloud backup."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not log_id:
        log_id = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        is_new = True
    else:
        is_new = False
        
    now_ts = datetime.now().isoformat()
    
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        if is_new:
            conn.execute(
                """
                INSERT INTO pp_portfolio_logs (
                    log_id, log_date, category, content, created_at, updated_at, run_id, image_data, snippet
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (log_id, log_date, category, content, now_ts, now_ts, run_id, image_data, snippet),
            )
        else:
            if clear_image:
                conn.execute(
                    """
                    UPDATE pp_portfolio_logs
                    SET log_date = ?, category = ?, content = ?, updated_at = ?, run_id = ?, image_data = NULL, snippet = ?
                    WHERE log_id = ?
                    """,
                    (log_date, category, content, now_ts, run_id, snippet, log_id),
                )
            elif image_data is not None:
                conn.execute(
                    """
                    UPDATE pp_portfolio_logs
                    SET log_date = ?, category = ?, content = ?, updated_at = ?, run_id = ?, image_data = ?, snippet = ?
                    WHERE log_id = ?
                    """,
                    (log_date, category, content, now_ts, run_id, image_data, snippet, log_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE pp_portfolio_logs
                    SET log_date = ?, category = ?, content = ?, updated_at = ?, run_id = ?, snippet = ?
                    WHERE log_id = ?
                    """,
                    (log_date, category, content, now_ts, run_id, snippet, log_id),
                )
                
        # Handle multiple images
        if clear_image:
            conn.execute("DELETE FROM pp_log_images WHERE log_id = ?", (log_id,))
        elif images_list is not None:
            # Clear legacy column to keep everything consistent
            conn.execute("UPDATE pp_portfolio_logs SET image_data = NULL WHERE log_id = ?", (log_id,))
            conn.execute("DELETE FROM pp_log_images WHERE log_id = ?", (log_id,))
            for idx, img in enumerate(images_list):
                img_id = f"img_{log_id}_{idx}_{uuid.uuid4().hex[:4]}"
                conn.execute(
                    """
                    INSERT INTO pp_log_images (image_id, log_id, image_data, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (img_id, log_id, sqlite3.Binary(img), now_ts),
                )
            
    # Sync database to cloud storage
    try:
        backup_database("portfolio_performance", reason=f"Log {'Save' if is_new else 'Update'}")
    except Exception as e:
        logger.warning(f"Cloud backup failed for portfolio_performance on log save: {e}")
        
    return log_id


def delete_portfolio_log(log_id: str, db_path: Path = DB_PATH) -> None:
    """Delete a portfolio log by ID, triggering cloud backup."""
    if not db_path.exists():
        return
        
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        conn.execute("DELETE FROM pp_portfolio_logs WHERE log_id = ?", (log_id,))
        conn.execute("DELETE FROM pp_log_images WHERE log_id = ?", (log_id,))
        
    # Sync database to cloud storage
    try:
        backup_database("portfolio_performance", reason="Log Delete")
    except Exception as e:
        logger.warning(f"Cloud backup failed for portfolio_performance on log delete: {e}")


def get_portfolio_log_images(db_path: Path = DB_PATH) -> dict[str, list[bytes]]:
    """Retrieve all log images grouped by log_id."""
    if not db_path.exists():
        return {}
        
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        cursor = conn.cursor()
        cursor.execute("SELECT log_id, image_data FROM pp_log_images ORDER BY created_at ASC")
        rows = cursor.fetchall()
        
    res = {}
    for log_id, image_bytes in rows:
        if log_id not in res:
            res[log_id] = []
        res[log_id].append(image_bytes)
    return res


def get_portfolio_logs(
    start_date: str | None = None,
    end_date: str | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Retrieve portfolio logs within optional date range (YYYY-MM-DD)."""
    if not db_path.exists():
        return pd.DataFrame()
        
    query = "SELECT log_id, log_date, category, content, created_at, updated_at, run_id, image_data, snippet FROM pp_portfolio_logs"
    params = []
    
    conditions = []
    if start_date:
        conditions.append("date(log_date) >= date(?)")
        params.append(start_date)
    if end_date:
        conditions.append("date(log_date) <= date(?)")
        params.append(end_date)
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += " ORDER BY log_date DESC, created_at DESC"
    
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        return pd.read_sql_query(query, conn, params=params)


def get_portfolio_history(days: int = 90, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return one saved total row per snapshot for progress analysis."""

    if not db_path.exists():
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        return pd.read_sql_query(
            """
            SELECT
                snapshot_ts, snapshot_date, row_count, total_pl_open, total_pl_day,
                total_theta, total_delta, total_market_value
            FROM pp_runs
            WHERE date(snapshot_date) >= date('now', ?)
            ORDER BY snapshot_ts ASC
            """,
            conn,
            params=(f"-{int(days)} days",),
        )


def get_latest_portfolio_positions(db_path: Path = DB_PATH) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Load ticker rows from the most recent saved portfolio snapshot."""
    if not db_path.exists():
        return pd.DataFrame(), None

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        run = conn.execute(
            """
            SELECT run_id, snapshot_ts, snapshot_date, source_file, row_count, totals_json
            FROM pp_runs
            ORDER BY snapshot_ts DESC
            LIMIT 1
            """
        ).fetchone()
        if not run:
            return pd.DataFrame(), None

        df = pd.read_sql_query(
            """
            SELECT
                account_group, ticker, quantity, mark_price, market_value,
                theta, delta, gamma, vega, pl_open, pl_day, bp_effect,
                snapshot_ts, snapshot_date, source_file
            FROM pp_position_snapshots
            WHERE run_id = ?
            ORDER BY pl_open DESC
            """,
            conn,
            params=(run[0],),
        )

    metadata = {
        "run_id": run[0],
        "snapshot_ts": run[1],
        "snapshot_date": run[2],
        "source_file": run[3],
        "row_count": run[4],
        "totals": json.loads(run[5]) if run[5] else summarize_positions(df),
    }
    return df, metadata


def get_latest_portfolio_details(db_path: Path = DB_PATH) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Load detailed rows from the most recent saved portfolio snapshot."""
    if not db_path.exists():
        return pd.DataFrame(), None

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        run = conn.execute(
            """
            SELECT run_id, snapshot_ts, snapshot_date, source_file
            FROM pp_runs
            ORDER BY snapshot_ts DESC
            LIMIT 1
            """
        ).fetchone()
        if not run:
            return pd.DataFrame(), None

        df = pd.read_sql_query(
            """
            SELECT
                account_group, underlying, row_type, strategy, instrument,
                quantity, days, expiration, strike, call_put, side,
                trade_price, mark_price, mark_change, delta, theta, gamma, vega,
                pl_open, pl_day, bp_effect, is_weekly,
                snapshot_ts, snapshot_date, source_file
            FROM pp_position_details
            WHERE run_id = ?
            ORDER BY underlying, row_type, instrument
            """,
            conn,
            params=(run[0],),
        )

    metadata = {
        "run_id": run[0],
        "snapshot_ts": run[1],
        "snapshot_date": run[2],
        "source_file": run[3],
        "detail_row_count": int(len(df)),
    }
    return df, metadata


def get_portfolio_details_for_run(run_id: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return detailed rows for a saved run id."""
    if not db_path.exists():
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        return pd.read_sql_query(
            """
            SELECT
                account_group, underlying, row_type, strategy, instrument,
                quantity, days, expiration, strike, call_put, side,
                trade_price, mark_price, mark_change, delta, theta, gamma, vega,
                pl_open, pl_day, bp_effect, is_weekly,
                snapshot_ts, snapshot_date, source_file
            FROM pp_position_details
            WHERE run_id = ?
            ORDER BY underlying, row_type, instrument
            """,
            conn,
            params=(run_id,),
        )


def get_database_status(db_path: Path = DB_PATH) -> dict[str, Any]:
    """Return storage diagnostics so production can detect ephemeral DB paths."""
    resolved = db_path.expanduser().resolve()
    repo_root = REPO_ROOT.resolve()
    env_path = os.getenv("PORTFOLIO_PERFORMANCE_DB_PATH")
    exists = resolved.exists()
    inside_repo = _is_relative_to(resolved, repo_root)

    status = {
        "db_path": str(resolved),
        "configured_env_path": env_path,
        "exists": exists,
        "inside_repo": inside_repo,
        "is_default_path": resolved == DEFAULT_DB_PATH.resolve(),
        "run_count": 0,
        "position_count": 0,
        "latest_snapshot_ts": None,
        "latest_source_file": None,
        "warning": None,
    }

    if inside_repo:
        status["warning"] = (
            "Database is inside the app repository. Git will not overwrite it, "
            "but production hosts with ephemeral app storage can wipe it on reboot or redeploy. "
            "Set PORTFOLIO_PERFORMANCE_DB_PATH to a persistent mounted volume."
        )

    if not exists:
        return status

    with sqlite3.connect(resolved) as conn:
        _ensure_schema(conn)
        status["run_count"] = int(conn.execute("SELECT COUNT(*) FROM pp_runs").fetchone()[0])
        status["position_count"] = int(conn.execute("SELECT COUNT(*) FROM pp_position_snapshots").fetchone()[0])
        latest = conn.execute(
            """
            SELECT snapshot_ts, source_file
            FROM pp_runs
            ORDER BY snapshot_ts DESC
            LIMIT 1
            """
        ).fetchone()
        if latest:
            status["latest_snapshot_ts"] = latest[0]
            status["latest_source_file"] = latest[1]

    return status


def extract_snapshot_datetime(text: str) -> datetime:
    """Extract Schwab's report timestamp when present, otherwise use local time."""
    first_lines = " ".join(text.splitlines()[:8])
    match = re.search(
        r"on\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2}:\d{2})",
        first_lines,
        flags=re.IGNORECASE,
    )
    if match:
        for fmt in ("%m/%d/%y %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
            try:
                return datetime.strptime(f"{match.group(1)} {match.group(2)}", fmt).replace(tzinfo=CENTRAL_TZ)
            except ValueError:
                continue
    return datetime.now(CENTRAL_TZ).replace(microsecond=0)


def get_underlying_ticker(ticker: str) -> str:
    """Extract underlying stock ticker from option leg formats."""
    ticker = str(ticker).strip().upper()
    if not ticker or ticker in ["CASH", "USD", "MMDA12", "MMDA"] or "CASH" in ticker:
        return "CASH"
    
    # Format 1: OCC option symbol format (e.g. AAPL  240621C00180000 or AAPL240621C00180000)
    match_occ = re.match(r"^([A-Z]+)\s*\d{6}[CP]\d{8}$", ticker)
    if match_occ:
        return match_occ.group(1)
        
    # Format 2: Schwab format with date (e.g. AAPL 06/21/2024 180.00 C)
    match_schwab = re.match(r"^([A-Z]+)\s+\d{1,2}/\d{1,2}/\d{2,4}\s+", ticker)
    if match_schwab:
        return match_schwab.group(1)
        
    if ticker.startswith("^"):
        return ticker
        
    return ticker


def _extract_positions(text: str, account: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current_group = "Unknown"
    header: list[str] | None = None
    in_table = False

    reader = csv.reader(io.StringIO(text))
    for raw_row in reader:
        if not raw_row:
            continue

        row = [str(value).strip() for value in raw_row]
        first = row[0] if row else ""

        if first.startswith('Group "'):
            current_group = first.replace('Group "', "").replace('"', "") or "Unknown"
            in_table = False
            header = None
            continue

        if first == "Instrument":
            header = row
            in_table = True
            continue

        if first.startswith("Subtotals") or first.startswith("Overall Totals"):
            in_table = False
            header = None
            continue

        if not in_table or header is None:
            continue

        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))

        record = dict(zip(header, row))
        ticker = str(record.get("Instrument", "")).strip().upper()
        
        # Resolve option leg symbols to their underlying ticker
        underlying = get_underlying_ticker(ticker)
        if underlying == "CASH":
            ticker = "CASH"
        elif underlying:
            ticker = underlying

        is_ticker = bool(re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", ticker))
        is_detail_row = ticker in {
            "CUSTOM", "CALENDAR", "VERTICAL", "DIAGONAL", "SINGLE", "DOUBLE",
            "TRIPLE", "QUAD", "CONDOR", "BUTTERFLY", "STRADDLE", "STRANGLE",
            "COLLAR", "COVERED", "NONE"
        }
        if not is_ticker or is_detail_row:
            continue

        dot = get_broker_dot(account)
        ticker = f"{dot} {ticker}"

        rows.append(
            {
                "account_group": current_group,
                "ticker": ticker,
                "quantity": parse_number(_first_available(record, ["Qty", "Quantity"])),
                "mark_price": parse_money(_first_available(record, ["Mark", "Mark Price", "Price"])),
                "market_value": parse_money(_first_available(record, ["Market Value", "Mkt Val"])),
                "theta": parse_number(record.get("Theta")),
                "delta": parse_number(record.get("Delta")),
                "gamma": parse_number(record.get("Gamma")),
                "vega": parse_number(record.get("Vega")),
                "pl_open": parse_money(record.get("P/L Open")),
                "pl_day": parse_money(record.get("P/L Day")),
                "bp_effect": parse_money(record.get("BP Effect")),
            }
        )

    return pd.DataFrame(rows)


def _extract_position_details(text: str, account: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current_group = "Unknown"
    current_underlying: str | None = None
    current_strategy: str | None = None
    header: list[str] | None = None
    in_table = False
    option_pattern = re.compile(
        r"^100\s+(?:(\(Weeklys\))\s+)?(\d{1,2}\s+[A-Z]{3}\s+\d{2})\s+([0-9.]+)\s+(CALL|PUT)$",
        flags=re.IGNORECASE,
    )

    reader = csv.reader(io.StringIO(text))
    for raw_row in reader:
        if not raw_row:
            continue

        row = [str(value).strip() for value in raw_row]
        first = row[0] if row else ""

        if first.startswith('Group "'):
            current_group = first.replace('Group "', "").replace('"', "") or "Unknown"
            current_underlying = None
            current_strategy = None
            in_table = False
            header = None
            continue

        if first == "Instrument":
            header = row
            in_table = True
            continue

        if first.startswith("Subtotals") or first.startswith("Overall Totals"):
            in_table = False
            header = None
            current_underlying = None
            current_strategy = None
            continue

        if not in_table or header is None:
            continue

        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))

        record = dict(zip(header, row))
        instrument = str(record.get("Instrument", "")).strip()
        if not instrument or instrument == "None":
            continue

        underlying = get_underlying_ticker(instrument)
        if underlying == "CASH":
            instrument = "CASH"

        option_match = option_pattern.match(instrument)
        is_ticker = bool(re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", instrument))
        is_strategy = instrument in {
            "CUSTOM", "CALENDAR", "VERTICAL", "DIAGONAL", "SINGLE", "DOUBLE",
            "TRIPLE", "QUAD", "CONDOR", "BUTTERFLY", "STRADDLE", "STRANGLE",
            "COLLAR", "COVERED", "NONE"
        }

        row_type = "detail"
        strategy = current_strategy
        expiration = None
        strike = None
        call_put = None
        side = None
        is_weekly = False

        if option_match:
            row_type = "option_leg"
            is_weekly = bool(option_match.group(1))
            expiration = option_match.group(2).upper()
            strike = parse_number(option_match.group(3))
            call_put = option_match.group(4).upper()
            quantity = parse_number(_first_available(record, ["Qty", "Quantity"]))
            side = "SELL" if quantity < 0 else "BUY"
        elif is_strategy:
            row_type = "strategy"
            current_strategy = instrument if instrument != "CUSTOM" else current_strategy
            strategy = current_strategy
        elif is_ticker:
            row_type = "ticker_summary"
            dot = get_broker_dot(account)
            current_underlying = f"{dot} {instrument.upper()}"
            current_strategy = None
            strategy = None
        elif parse_number(_first_available(record, ["Qty", "Quantity"])) == 0:
            row_type = "equity_description"

        rows.append(
            {
                "account_group": current_group,
                "underlying": current_underlying,
                "row_type": row_type,
                "strategy": strategy,
                "instrument": instrument,
                "quantity": parse_number(_first_available(record, ["Qty", "Quantity"])),
                "days": parse_number(record.get("Days")),
                "expiration": expiration,
                "strike": strike,
                "call_put": call_put,
                "side": side,
                "trade_price": parse_money(_first_available(record, ["Trade Price"])),
                "mark_price": parse_money(_first_available(record, ["Mark", "Mark Price", "Price"])),
                "mark_change": parse_number(record.get("Mrk Chng")),
                "delta": parse_number(record.get("Delta")),
                "theta": parse_number(record.get("Theta")),
                "gamma": parse_number(record.get("Gamma")),
                "vega": parse_number(record.get("Vega")),
                "pl_open": parse_money(record.get("P/L Open")),
                "pl_day": parse_money(record.get("P/L Day")),
                "bp_effect": parse_money(record.get("BP Effect")),
                "is_weekly": bool(is_weekly),
            }
        )

    details = pd.DataFrame(rows)
    if not details.empty:
        details["underlying"] = details["underlying"].ffill()
    return details


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pp_runs (
            run_id TEXT PRIMARY KEY,
            snapshot_ts TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            source_file TEXT,
            file_sha256 TEXT,
            row_count INTEGER NOT NULL,
            raw_row_count INTEGER NOT NULL,
            totals_json TEXT,
            total_pl_open REAL,
            total_pl_day REAL,
            total_theta REAL,
            total_delta REAL,
            total_market_value REAL
        );

        CREATE TABLE IF NOT EXISTS pp_position_snapshots (
            run_id TEXT NOT NULL,
            snapshot_ts TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            source_file TEXT,
            account_group TEXT,
            ticker TEXT NOT NULL,
            quantity REAL,
            mark_price REAL,
            market_value REAL,
            theta REAL,
            delta REAL,
            gamma REAL,
            vega REAL,
            pl_open REAL,
            pl_day REAL,
            bp_effect REAL,
            FOREIGN KEY (run_id) REFERENCES pp_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_pp_runs_date
            ON pp_runs(snapshot_date, snapshot_ts);
        CREATE INDEX IF NOT EXISTS idx_pp_positions_run
            ON pp_position_snapshots(run_id);
        CREATE INDEX IF NOT EXISTS idx_pp_positions_date_ticker
            ON pp_position_snapshots(snapshot_date, ticker);

        CREATE TABLE IF NOT EXISTS pp_position_details (
            run_id TEXT NOT NULL,
            snapshot_ts TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            source_file TEXT,
            account_group TEXT,
            underlying TEXT,
            row_type TEXT NOT NULL,
            strategy TEXT,
            instrument TEXT NOT NULL,
            quantity REAL,
            days REAL,
            expiration TEXT,
            strike REAL,
            call_put TEXT,
            side TEXT,
            trade_price REAL,
            mark_price REAL,
            mark_change REAL,
            delta REAL,
            theta REAL,
            gamma REAL,
            vega REAL,
            pl_open REAL,
            pl_day REAL,
            bp_effect REAL,
            is_weekly INTEGER,
            FOREIGN KEY (run_id) REFERENCES pp_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_pp_details_run
            ON pp_position_details(run_id);
        CREATE INDEX IF NOT EXISTS idx_pp_details_underlying
            ON pp_position_details(snapshot_date, underlying, row_type);

        CREATE TABLE IF NOT EXISTS pp_portfolio_logs (
            log_id TEXT PRIMARY KEY,
            log_date TEXT NOT NULL,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            run_id TEXT,
            image_data BLOB,
            snippet TEXT,
            FOREIGN KEY (run_id) REFERENCES pp_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_pp_logs_date ON pp_portfolio_logs(log_date);
        CREATE INDEX IF NOT EXISTS idx_pp_logs_run_id ON pp_portfolio_logs(run_id);

        CREATE TABLE IF NOT EXISTS pp_log_images (
            image_id TEXT PRIMARY KEY,
            log_id TEXT NOT NULL,
            image_data BLOB NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (log_id) REFERENCES pp_portfolio_logs(log_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_pp_log_images_log_id ON pp_log_images(log_id);
        """
    )
    _ensure_column(conn, "pp_position_snapshots", "gamma", "REAL")
    _ensure_column(conn, "pp_position_snapshots", "vega", "REAL")
    _ensure_column(conn, "pp_portfolio_logs", "image_data", "BLOB")
    _ensure_column(conn, "pp_portfolio_logs", "snippet", "TEXT")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _prepare_position_rows(
    df: pd.DataFrame,
    run_id: str,
    snapshot_ts: str,
    snapshot_date: str,
    source_file: str,
) -> pd.DataFrame:
    rows = df.copy()
    rows["run_id"] = run_id
    rows["snapshot_ts"] = snapshot_ts
    rows["snapshot_date"] = snapshot_date
    rows["source_file"] = source_file

    for column in POSITION_COLUMNS:
        if column not in rows.columns:
            rows[column] = None

    numeric_cols = [
        "quantity",
        "mark_price",
        "market_value",
        "theta",
        "delta",
        "gamma",
        "vega",
        "pl_open",
        "pl_day",
        "bp_effect",
    ]
    for column in numeric_cols:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")

    text_cols = ["source_file", "account_group", "ticker"]
    for column in text_cols:
        rows[column] = rows[column].astype("string")

    return rows[POSITION_COLUMNS].where(pd.notna(rows[POSITION_COLUMNS]), None)


def _prepare_detail_rows(
    df: pd.DataFrame,
    run_id: str,
    snapshot_ts: str,
    snapshot_date: str,
    source_file: str,
) -> pd.DataFrame:
    rows = df.copy()
    rows["run_id"] = run_id
    rows["snapshot_ts"] = snapshot_ts
    rows["snapshot_date"] = snapshot_date
    rows["source_file"] = source_file

    for column in DETAIL_COLUMNS:
        if column not in rows.columns:
            rows[column] = None

    numeric_cols = [
        "quantity",
        "days",
        "strike",
        "trade_price",
        "mark_price",
        "mark_change",
        "delta",
        "theta",
        "gamma",
        "vega",
        "pl_open",
        "pl_day",
        "bp_effect",
    ]
    for column in numeric_cols:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")

    text_cols = [
        "source_file",
        "account_group",
        "underlying",
        "row_type",
        "strategy",
        "instrument",
        "expiration",
        "call_put",
        "side",
    ]
    for column in text_cols:
        rows[column] = rows[column].astype("string")

    rows["is_weekly"] = rows["is_weekly"].fillna(False).astype(bool).astype(int)
    return rows[DETAIL_COLUMNS].where(pd.notna(rows[DETAIL_COLUMNS]), None)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def _decode_content(content: bytes | str) -> str:
    if isinstance(content, str):
        return content

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _first_available(record: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in record:
            return record.get(name)
    return None
