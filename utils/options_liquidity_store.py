"""Local snapshot storage for Options Liquidity Discovery scans."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "options_liquidity"
DB_PATH = Path(
    os.getenv("OPTIONS_LIQUIDITY_DB_PATH", DATA_DIR / "options_liquidity.sqlite")
).expanduser()


CONTRACT_COLUMNS = [
    "run_id",
    "scan_ts",
    "trade_date",
    "symbol",
    "option_type",
    "expiration",
    "dte",
    "spot",
    "strike",
    "moneyness",
    "iv_pct",
    "volume",
    "open_interest",
    "bid",
    "ask",
    "spread",
    "spread_pct",
    "last_price",
    "contract",
    "streamer_symbol",
    "data_source",
]


def save_options_snapshot(
    df: pd.DataFrame,
    params: dict[str, Any],
    data_source: str,
    db_path: Path = DB_PATH,
) -> dict[str, Any]:
    """Persist one scan run plus granular contract rows and summaries."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    scan_ts = datetime.now().replace(microsecond=0)
    trade_date = scan_ts.date().isoformat()
    run_id = f"ol_{scan_ts.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    symbols = list(params.get("symbols", []))
    filters = {
        key: _json_safe(value)
        for key, value in params.items()
        if key != "symbols"
    }

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        conn.execute(
            """
            INSERT INTO ol_runs (
                run_id, scan_ts, trade_date, symbols_json, filters_json,
                data_source, row_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                scan_ts.isoformat(sep=" "),
                trade_date,
                json.dumps(symbols),
                json.dumps(filters, sort_keys=True),
                data_source,
                int(len(df)),
            ),
        )

        if not df.empty:
            contracts = _prepare_contract_rows(df, run_id, scan_ts, trade_date)
            contracts.to_sql(
                "ol_contract_snapshots",
                conn,
                if_exists="append",
                index=False,
            )

            summaries = _build_symbol_summaries(contracts, run_id, scan_ts, trade_date)
            if not summaries.empty:
                summaries.to_sql(
                    "ol_symbol_snapshot_summary",
                    conn,
                    if_exists="append",
                    index=False,
                )

    return {
        "run_id": run_id,
        "scan_ts": scan_ts.isoformat(sep=" "),
        "trade_date": trade_date,
        "row_count": int(len(df)),
        "db_path": str(db_path),
    }


def get_recent_snapshots(limit: int = 10, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return recent scan runs for display or diagnostics."""
    if not db_path.exists():
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT run_id, scan_ts, trade_date, row_count, data_source
            FROM ol_runs
            ORDER BY scan_ts DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )


def get_latest_contract_snapshot(
    symbols: list[str] | tuple[str, ...],
    min_volume: int = 0,
    min_oi: int = 0,
    option_types: list[str] | tuple[str, ...] | None = None,
    exp_pref: str = "Any",
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Return the newest saved contract rows that match the current scan filters."""
    if not db_path.exists() or not symbols:
        return pd.DataFrame()

    symbol_list = [str(symbol).upper() for symbol in symbols if str(symbol).strip()]
    if not symbol_list:
        return pd.DataFrame()

    placeholders = ",".join("?" for _ in symbol_list)
    min_dte, max_dte = _expiration_bounds(exp_pref)
    type_list = list(option_types or [])
    type_clause = ""
    type_params: list[str] = []
    if type_list:
        type_clause = f" AND option_type IN ({','.join('?' for _ in type_list)})"
        type_params = [str(option_type) for option_type in type_list]

    with sqlite3.connect(db_path) as conn:
        latest_run = conn.execute(
            f"""
            SELECT run_id
            FROM ol_contract_snapshots
            WHERE symbol IN ({placeholders})
            ORDER BY scan_ts DESC
            LIMIT 1
            """,
            symbol_list,
        ).fetchone()

        if not latest_run:
            return pd.DataFrame()

        params = [
            latest_run[0],
            *symbol_list,
            min_dte,
            max_dte,
            int(min_volume),
            int(min_oi),
            *type_params,
        ]
        df = pd.read_sql_query(
            f"""
            SELECT
                symbol, option_type, expiration, dte, spot, strike, moneyness,
                iv_pct, volume, open_interest, bid, ask, spread, spread_pct,
                last_price, contract, streamer_symbol, data_source
            FROM ol_contract_snapshots
            WHERE run_id = ?
              AND symbol IN ({placeholders})
              AND dte BETWEEN ? AND ?
              AND COALESCE(volume, 0) >= ?
              AND COALESCE(open_interest, 0) >= ?
              {type_clause}
            ORDER BY volume DESC
            """,
            conn,
            params=params,
        )

    if df.empty:
        return df

    df.rename(columns={"iv_pct": "iv_%"}, inplace=True)
    df["data_source"] = "Local snapshot fallback"
    df.attrs["active_data_source"] = f"Local snapshot fallback from {latest_run[0]}"
    return df


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS ol_runs (
            run_id TEXT PRIMARY KEY,
            scan_ts TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            symbols_json TEXT NOT NULL,
            filters_json TEXT NOT NULL,
            data_source TEXT,
            row_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ol_contract_snapshots (
            run_id TEXT NOT NULL,
            scan_ts TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            symbol TEXT,
            option_type TEXT,
            expiration TEXT,
            dte REAL,
            spot REAL,
            strike REAL,
            moneyness REAL,
            iv_pct REAL,
            volume REAL,
            open_interest REAL,
            bid REAL,
            ask REAL,
            spread REAL,
            spread_pct REAL,
            last_price REAL,
            contract TEXT,
            streamer_symbol TEXT,
            data_source TEXT,
            FOREIGN KEY (run_id) REFERENCES ol_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS ol_symbol_snapshot_summary (
            run_id TEXT NOT NULL,
            scan_ts TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            contract_count INTEGER NOT NULL,
            call_volume REAL,
            put_volume REAL,
            total_volume REAL,
            total_open_interest REAL,
            avg_iv_pct REAL,
            median_spread_pct REAL,
            top_contract TEXT,
            top_contract_volume REAL,
            FOREIGN KEY (run_id) REFERENCES ol_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_ol_contract_date_symbol
            ON ol_contract_snapshots(trade_date, symbol);
        CREATE INDEX IF NOT EXISTS idx_ol_contract_run
            ON ol_contract_snapshots(run_id);
        CREATE INDEX IF NOT EXISTS idx_ol_summary_date_symbol
            ON ol_symbol_snapshot_summary(trade_date, symbol);
        """
    )


def _expiration_bounds(exp_pref: str) -> tuple[int, int]:
    exp_label = str(exp_pref).lower()
    if "weekly" in exp_label:
        return 0, 8
    if "monthly" in exp_label:
        return 9, 45
    return 0, 365


def _prepare_contract_rows(
    df: pd.DataFrame,
    run_id: str,
    scan_ts: datetime,
    trade_date: str,
) -> pd.DataFrame:
    rows = df.copy()
    rows.rename(columns={"iv_%": "iv_pct"}, inplace=True)

    for col in CONTRACT_COLUMNS:
        if col not in rows.columns:
            rows[col] = None

    rows["run_id"] = run_id
    rows["scan_ts"] = scan_ts.isoformat(sep=" ")
    rows["trade_date"] = trade_date

    numeric_cols = [
        "dte",
        "spot",
        "strike",
        "moneyness",
        "iv_pct",
        "volume",
        "open_interest",
        "bid",
        "ask",
        "spread",
        "spread_pct",
        "last_price",
    ]
    for col in numeric_cols:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")

    text_cols = [
        "symbol",
        "option_type",
        "expiration",
        "contract",
        "streamer_symbol",
        "data_source",
    ]
    for col in text_cols:
        rows[col] = rows[col].astype("string")

    return rows[CONTRACT_COLUMNS].where(pd.notna(rows[CONTRACT_COLUMNS]), None)


def _build_symbol_summaries(
    contracts: pd.DataFrame,
    run_id: str,
    scan_ts: datetime,
    trade_date: str,
) -> pd.DataFrame:
    if contracts.empty or "symbol" not in contracts.columns:
        return pd.DataFrame()

    rows = []
    for symbol, group in contracts.groupby("symbol", dropna=True):
        calls = group.loc[group["option_type"] == "Call", "volume"].sum()
        puts = group.loc[group["option_type"] == "Put", "volume"].sum()
        top_contract = None
        top_contract_volume = None
        if "volume" in group.columns and not group["volume"].dropna().empty:
            top = group.sort_values("volume", ascending=False).iloc[0]
            top_contract = top.get("contract")
            top_contract_volume = top.get("volume")

        rows.append(
            {
                "run_id": run_id,
                "scan_ts": scan_ts.isoformat(sep=" "),
                "trade_date": trade_date,
                "symbol": symbol,
                "contract_count": int(len(group)),
                "call_volume": calls,
                "put_volume": puts,
                "total_volume": group["volume"].sum(),
                "total_open_interest": group["open_interest"].sum(),
                "avg_iv_pct": group["iv_pct"].mean(),
                "median_spread_pct": group["spread_pct"].median(),
                "top_contract": top_contract,
                "top_contract_volume": top_contract_volume,
            }
        )

    summaries = pd.DataFrame(rows)
    return summaries.where(pd.notna(summaries), None)


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    return value
