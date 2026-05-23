"""Local storage and parsing for Portfolio Performance snapshots."""

from __future__ import annotations

import csv
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
from zoneinfo import ZoneInfo

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "portfolio_performance"
DEFAULT_DB_PATH = DATA_DIR / "portfolio_performance.sqlite"
DB_PATH = Path(os.getenv("PORTFOLIO_PERFORMANCE_DB_PATH", DEFAULT_DB_PATH)).expanduser()
CENTRAL_TZ = ZoneInfo("America/Chicago")


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
    "pl_open",
    "pl_day",
    "bp_effect",
]


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
    rows = _extract_positions(text)
    if rows.empty:
        raise ValueError("No ticker-level rows found. Check the Schwab Position Statement CSV format.")

    aggregate_map = {
        "account_group": lambda values: ", ".join(sorted({str(v) for v in values if str(v).strip()})),
        "quantity": "sum",
        "mark_price": "mean",
        "market_value": "sum",
        "theta": "sum",
        "delta": "sum",
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
        "totals": summarize_positions(positions),
    }
    return positions, metadata


def summarize_positions(df: pd.DataFrame) -> dict[str, float]:
    """Return portfolio-level totals used by the dashboard and run table."""
    if df.empty:
        return {
            "total_pl_open": 0.0,
            "total_pl_day": 0.0,
            "total_theta": 0.0,
            "total_delta": 0.0,
            "total_market_value": 0.0,
            "positive_open": 0.0,
            "negative_open": 0.0,
        }

    return {
        "total_pl_open": float(df["pl_open"].sum()),
        "total_pl_day": float(df["pl_day"].sum()),
        "total_theta": float(df["theta"].sum()),
        "total_delta": float(df["delta"].sum()),
        "total_market_value": float(df.get("market_value", pd.Series(dtype=float)).sum()),
        "positive_open": float(df.loc[df["pl_open"] > 0, "pl_open"].sum()),
        "negative_open": float(df.loc[df["pl_open"] < 0, "pl_open"].sum()),
    }


def save_portfolio_snapshot(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    db_path: Path = DB_PATH,
) -> dict[str, Any]:
    """Persist one uploaded portfolio snapshot and its ticker-level rows."""
    if df.empty:
        raise ValueError("Cannot save an empty portfolio snapshot.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = f"pp_{datetime.now(CENTRAL_TZ).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    snapshot_ts = str(metadata.get("snapshot_ts") or datetime.now(CENTRAL_TZ).isoformat(sep=" "))
    snapshot_date = str(metadata.get("snapshot_date") or snapshot_ts[:10])
    source_file = str(metadata.get("source_file") or "uploaded_positions.csv")
    totals = summarize_positions(df)

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        conn.execute(
            """
            INSERT INTO pp_runs (
                run_id, snapshot_ts, snapshot_date, source_file, file_sha256,
                row_count, raw_row_count, totals_json, total_pl_open, total_pl_day,
                total_theta, total_delta, total_market_value
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
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

    return {
        "run_id": run_id,
        "snapshot_ts": snapshot_ts,
        "snapshot_date": snapshot_date,
        "row_count": int(len(df)),
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
                theta, delta, pl_open, pl_day, bp_effect,
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


def _extract_positions(text: str) -> pd.DataFrame:
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
        is_ticker = bool(re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", ticker))
        is_detail_row = ticker in {"CUSTOM", "CALENDAR", "VERTICAL", "NONE"}
        if not is_ticker or is_detail_row:
            continue

        rows.append(
            {
                "account_group": current_group,
                "ticker": ticker,
                "quantity": parse_number(_first_available(record, ["Qty", "Quantity"])),
                "mark_price": parse_money(_first_available(record, ["Mark", "Mark Price", "Price"])),
                "market_value": parse_money(_first_available(record, ["Market Value", "Mkt Val"])),
                "theta": parse_number(record.get("Theta")),
                "delta": parse_number(record.get("Delta")),
                "pl_open": parse_money(record.get("P/L Open")),
                "pl_day": parse_money(record.get("P/L Day")),
                "bp_effect": parse_money(record.get("BP Effect")),
            }
        )

    return pd.DataFrame(rows)


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
        """
    )


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
