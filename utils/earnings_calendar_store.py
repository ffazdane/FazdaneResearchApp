"""SQLite storage for Earnings Calendar data and fetch coverage."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from utils.persistence import backup_database


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "earnings_calendar"
DEFAULT_DB_PATH = DATA_DIR / "earnings_calendar.sqlite"
DB_PATH = Path(os.getenv("EARNINGS_CALENDAR_DB_PATH", DEFAULT_DB_PATH)).expanduser()


EVENT_COLUMNS = [
    "scope",
    "date",
    "ticker",
    "name",
    "time",
    "eps_estimate",
    "reported_eps",
    "surprise_pct",
    "price",
    "source",
    "last_fetched_ts",
]


def load_earnings_events(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    scope: str = "universe",
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    symbols = _clean_tickers(tickers)
    if not db_path.exists() or not symbols:
        return pd.DataFrame()

    placeholders = ",".join("?" for _ in symbols)
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        return pd.read_sql_query(
            f"""
            SELECT
                date AS Date,
                ticker AS Ticker,
                name AS Name,
                time AS Time,
                eps_estimate AS "EPS Estimate",
                reported_eps AS "Reported EPS",
                surprise_pct AS "Surprise %",
                price AS Price,
                source AS Source
            FROM ec_earnings_events
            WHERE scope = ?
              AND ticker IN ({placeholders})
              AND date BETWEEN ? AND ?
            ORDER BY date, ticker
            """,
            conn,
            params=[scope, *symbols, start_date, end_date],
        )


def load_market_earnings(
    dates: Iterable[str],
    min_price: float,
    scope: str = "market_next_7",
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    date_list = _clean_dates(dates)
    if not db_path.exists() or not date_list:
        return pd.DataFrame()

    placeholders = ",".join("?" for _ in date_list)
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        return pd.read_sql_query(
            f"""
            SELECT
                date AS Date,
                ticker AS Ticker,
                name AS Name,
                time AS Time,
                price AS Price,
                source AS Source
            FROM ec_earnings_events
            WHERE scope = ?
              AND date IN ({placeholders})
              AND price >= ?
            ORDER BY date, ticker
            """,
            conn,
            params=[scope, *date_list, float(min_price)],
        )


def save_earnings_events(
    rows: list[dict],
    scope: str,
    db_path: Path = DB_PATH,
) -> int:
    if not rows:
        return 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    fetched_ts = datetime.now().replace(microsecond=0).isoformat(sep=" ")
    prepared = [_prepare_event_row(row, scope, fetched_ts) for row in rows]

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        conn.executemany(
            """
            INSERT INTO ec_earnings_events (
                scope, date, ticker, name, time, eps_estimate, reported_eps,
                surprise_pct, price, source, last_fetched_ts
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope, date, ticker) DO UPDATE SET
                name = excluded.name,
                time = excluded.time,
                eps_estimate = excluded.eps_estimate,
                reported_eps = excluded.reported_eps,
                surprise_pct = excluded.surprise_pct,
                price = excluded.price,
                source = excluded.source,
                last_fetched_ts = excluded.last_fetched_ts
            """,
            prepared,
        )

    # Sync to cloud storage
    try:
        backup_database("earnings_calendar", reason=f"Events Save: {scope}")
    except Exception as e:
        logger.warning(f"Cloud backup failed for earnings_calendar: {e}")

    return len(prepared)


def missing_ticker_coverage(
    tickers: Iterable[str],
    dates: Iterable[str],
    scope: str = "universe",
    db_path: Path = DB_PATH,
) -> list[str]:
    symbols = _clean_tickers(tickers)
    date_list = _clean_dates(dates)
    if not symbols or not date_list:
        return []
    if not db_path.exists():
        return symbols

    placeholders_tickers = ",".join("?" for _ in symbols)
    placeholders_dates = ",".join("?" for _ in date_list)
    required_count = len(date_list)

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            f"""
            SELECT ticker, COUNT(DISTINCT coverage_date) AS covered_days
            FROM ec_ticker_coverage
            WHERE scope = ?
              AND ticker IN ({placeholders_tickers})
              AND coverage_date IN ({placeholders_dates})
            GROUP BY ticker
            """,
            [scope, *symbols, *date_list],
        ).fetchall()

    covered = {ticker: count for ticker, count in rows}
    return [ticker for ticker in symbols if covered.get(ticker, 0) < required_count]


def mark_ticker_coverage(
    tickers: Iterable[str],
    dates: Iterable[str],
    scope: str = "universe",
    db_path: Path = DB_PATH,
) -> None:
    symbols = _clean_tickers(tickers)
    date_list = _clean_dates(dates)
    if not symbols or not date_list:
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    fetched_ts = datetime.now().replace(microsecond=0).isoformat(sep=" ")
    rows = [(scope, ticker, date_value, fetched_ts) for ticker in symbols for date_value in date_list]

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        conn.executemany(
            """
            INSERT INTO ec_ticker_coverage (scope, ticker, coverage_date, last_fetched_ts)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(scope, ticker, coverage_date) DO UPDATE SET
                last_fetched_ts = excluded.last_fetched_ts
            """,
            rows,
        )

    # Sync to cloud storage
    try:
        backup_database("earnings_calendar", reason=f"Coverage Mark: {scope}")
    except Exception as e:
        logger.warning(f"Cloud backup failed for earnings_calendar: {e}")


def missing_market_dates(
    dates: Iterable[str],
    scope: str = "market_next_7",
    db_path: Path = DB_PATH,
) -> list[str]:
    date_list = _clean_dates(dates)
    if not date_list:
        return []
    if not db_path.exists():
        return date_list

    placeholders = ",".join("?" for _ in date_list)
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            f"""
            SELECT coverage_date
            FROM ec_market_coverage
            WHERE scope = ?
              AND coverage_date IN ({placeholders})
            """,
            [scope, *date_list],
        ).fetchall()

    covered = {row[0] for row in rows}
    return [date_value for date_value in date_list if date_value not in covered]


def mark_market_dates(
    dates: Iterable[str],
    scope: str = "market_next_7",
    db_path: Path = DB_PATH,
) -> None:
    date_list = _clean_dates(dates)
    if not date_list:
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    fetched_ts = datetime.now().replace(microsecond=0).isoformat(sep=" ")
    rows = [(scope, date_value, fetched_ts) for date_value in date_list]

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        conn.executemany(
            """
            INSERT INTO ec_market_coverage (scope, coverage_date, last_fetched_ts)
            VALUES (?, ?, ?)
            ON CONFLICT(scope, coverage_date) DO UPDATE SET
                last_fetched_ts = excluded.last_fetched_ts
            """,
            rows,
        )

    # Sync to cloud storage
    try:
        backup_database("earnings_calendar", reason=f"Market Coverage Mark: {scope}")
    except Exception as e:
        logger.warning(f"Cloud backup failed for earnings_calendar: {e}")


def get_database_summary(db_path: Path = DB_PATH) -> dict:
    resolved = db_path.expanduser().resolve()
    repo_root = REPO_ROOT.resolve()
    env_path = os.getenv("EARNINGS_CALENDAR_DB_PATH")
    inside_repo = _is_relative_to(resolved, repo_root)
    warning = None
    if inside_repo:
        warning = (
            "Database is inside the app repository. Git will not overwrite it, "
            "but production hosts with ephemeral app storage can wipe it on reboot or redeploy. "
            "Set EARNINGS_CALENDAR_DB_PATH to a persistent mounted volume."
        )

    if not db_path.exists():
        return {
            "db_path": str(resolved),
            "configured_env_path": env_path,
            "exists": False,
            "inside_repo": inside_repo,
            "is_default_path": resolved == DEFAULT_DB_PATH.resolve(),
            "event_count": 0,
            "ticker_count": 0,
            "ticker_coverage_count": 0,
            "market_coverage_count": 0,
            "warning": warning,
        }

    with sqlite3.connect(resolved) as conn:
        _ensure_schema(conn)
        event_count = conn.execute("SELECT COUNT(*) FROM ec_earnings_events").fetchone()[0]
        ticker_count = conn.execute("SELECT COUNT(DISTINCT ticker) FROM ec_earnings_events").fetchone()[0]
        ticker_coverage_count = conn.execute("SELECT COUNT(*) FROM ec_ticker_coverage").fetchone()[0]
        market_coverage_count = conn.execute("SELECT COUNT(*) FROM ec_market_coverage").fetchone()[0]
        latest_fetch = conn.execute(
            """
            SELECT MAX(last_fetched_ts)
            FROM (
                SELECT last_fetched_ts FROM ec_earnings_events
                UNION ALL
                SELECT last_fetched_ts FROM ec_ticker_coverage
                UNION ALL
                SELECT last_fetched_ts FROM ec_market_coverage
            )
            """
        ).fetchone()[0]

    return {
        "db_path": str(resolved),
        "configured_env_path": env_path,
        "exists": True,
        "inside_repo": inside_repo,
        "is_default_path": resolved == DEFAULT_DB_PATH.resolve(),
        "event_count": int(event_count),
        "ticker_count": int(ticker_count),
        "ticker_coverage_count": int(ticker_coverage_count),
        "market_coverage_count": int(market_coverage_count),
        "latest_fetch": latest_fetch,
        "warning": warning,
    }


def get_saved_tickers(scope: str | None = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()

    where = ""
    params: list[str] = []
    if scope:
        where = "WHERE scope = ?"
        params.append(scope)

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        return pd.read_sql_query(
            f"""
            SELECT
                ticker,
                scope,
                COUNT(*) AS event_count,
                MIN(date) AS first_event_date,
                MAX(date) AS last_event_date,
                MAX(last_fetched_ts) AS last_fetched_ts
            FROM ec_earnings_events
            {where}
            GROUP BY ticker, scope
            ORDER BY ticker, scope
            """,
            conn,
            params=params,
        )


def get_recent_events(limit: int = 100, db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        return pd.read_sql_query(
            """
            SELECT
                scope,
                date AS Date,
                ticker AS Ticker,
                name AS Name,
                time AS Time,
                price AS Price,
                source AS Source,
                last_fetched_ts AS "Saved At"
            FROM ec_earnings_events
            ORDER BY last_fetched_ts DESC, date ASC, ticker ASC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )


def get_coverage_sample(limit: int = 100, db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        return pd.read_sql_query(
            """
            SELECT
                scope,
                ticker,
                coverage_date,
                last_fetched_ts
            FROM ec_ticker_coverage
            ORDER BY last_fetched_ts DESC, ticker ASC, coverage_date ASC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS ec_earnings_events (
            scope TEXT NOT NULL,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            name TEXT,
            time TEXT,
            eps_estimate REAL,
            reported_eps REAL,
            surprise_pct REAL,
            price REAL,
            source TEXT,
            last_fetched_ts TEXT NOT NULL,
            PRIMARY KEY (scope, date, ticker)
        );

        CREATE TABLE IF NOT EXISTS ec_ticker_coverage (
            scope TEXT NOT NULL,
            ticker TEXT NOT NULL,
            coverage_date TEXT NOT NULL,
            last_fetched_ts TEXT NOT NULL,
            PRIMARY KEY (scope, ticker, coverage_date)
        );

        CREATE TABLE IF NOT EXISTS ec_market_coverage (
            scope TEXT NOT NULL,
            coverage_date TEXT NOT NULL,
            last_fetched_ts TEXT NOT NULL,
            PRIMARY KEY (scope, coverage_date)
        );

        CREATE INDEX IF NOT EXISTS idx_ec_events_scope_date
            ON ec_earnings_events(scope, date);
        CREATE INDEX IF NOT EXISTS idx_ec_events_scope_ticker_date
            ON ec_earnings_events(scope, ticker, date);
        CREATE INDEX IF NOT EXISTS idx_ec_ticker_coverage_date
            ON ec_ticker_coverage(scope, coverage_date);
        """
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _prepare_event_row(row: dict, scope: str, fetched_ts: str) -> tuple:
    return (
        scope,
        _date_value(_first_present(row, "Date", "date")),
        str(_first_present(row, "Ticker", "ticker") or "").strip().upper(),
        _text_value(_first_present(row, "Name", "name")),
        _text_value(_first_present(row, "Time", "time")),
        _float_value(_first_present(row, "EPS Estimate", "eps_estimate")),
        _float_value(_first_present(row, "Reported EPS", "reported_eps")),
        _float_value(_first_present(row, "Surprise %", "surprise_pct")),
        _float_value(_first_present(row, "Price", "price")),
        _text_value(_first_present(row, "Source", "source")),
        fetched_ts,
    )


def _clean_tickers(values: Iterable[str]) -> list[str]:
    cleaned = []
    for value in values:
        ticker = str(value).strip().upper()
        if ticker and ticker not in cleaned:
            cleaned.append(ticker)
    return cleaned


def _clean_dates(values: Iterable[str]) -> list[str]:
    cleaned = []
    for value in values:
        date_value = _date_value(value)
        if date_value and date_value not in cleaned:
            cleaned.append(date_value)
    return cleaned


def _date_value(value) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _float_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _text_value(value):
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _first_present(row: dict, *keys: str):
    for key in keys:
        if key in row:
            return row.get(key)
    return None
