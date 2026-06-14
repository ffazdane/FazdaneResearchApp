import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from utils.persistence import get_db_path

logger = logging.getLogger("CycleEngineDB")

def get_connection():
    db_path = get_db_path("cycle_engine")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path, timeout=10)

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    # Table 1: cycle_signal_history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cycle_signal_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        dominant_cycle_days REAL,
        cycle_strength REAL,
        cycle_phase_pct REAL,
        cycle_direction TEXT,
        next_peak_date TEXT,
        next_bottom_date TEXT,
        peak_confidence REAL,
        bottom_confidence REAL,
        alignment_score REAL,
        volatility_cycle_status TEXT,
        liquidity_cycle_status TEXT,
        regime TEXT,
        recommended_strategy TEXT,
        confidence_score REAL,
        reason_code TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Table 2: cycle_backtest_results
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cycle_backtest_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id INTEGER,
        ticker TEXT,
        signal_date TEXT,
        forecast_horizon_days INTEGER,
        expected_direction TEXT,
        actual_return REAL,
        max_favorable_excursion REAL,
        max_adverse_excursion REAL,
        realized_vol_change REAL,
        strategy TEXT,
        strategy_outcome TEXT,
        pnl_estimate REAL,
        win_flag INTEGER,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Table 3: cycle_event_calendar
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cycle_event_calendar (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_date TEXT NOT NULL,
        event_type TEXT NOT NULL,
        event_name TEXT,
        expected_impact TEXT,
        source TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(event_date, event_type, event_name) ON CONFLICT REPLACE
    );
    """)

    # Table 4: cycle_trade_outcomes
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cycle_trade_outcomes (
        outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id INTEGER,
        entry_date TEXT,
        exit_date TEXT,
        entry_price REAL,
        exit_price REAL,
        max_favorable_move REAL,
        max_adverse_move REAL,
        pnl_percent REAL,
        win_loss_flag INTEGER,
        outcome_notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()
    logger.info("Cycle Analysis Engine database tables verified/created.")

def save_signal(data: dict) -> int:
    create_tables()
    conn = get_connection()
    cursor = conn.cursor()
    
    cols = [
        "signal_date", "ticker", "timeframe", "dominant_cycle_days", "cycle_strength",
        "cycle_phase_pct", "cycle_direction", "next_peak_date", "next_bottom_date",
        "peak_confidence", "bottom_confidence", "alignment_score", "volatility_cycle_status",
        "liquidity_cycle_status", "regime", "recommended_strategy", "confidence_score", "reason_code"
    ]
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO cycle_signal_history ({', '.join(cols)}) VALUES ({placeholders})"
    
    vals = [data.get(c) for c in cols]
    cursor.execute(sql, vals)
    signal_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Sync database
    from utils.persistence import backup_database
    try:
        backup_database("cycle_engine", reason=f"Save Cycle Signal: {data.get('ticker')}")
    except Exception as e:
        logger.warning(f"Failed to sync cycle database: {e}")
        
    return signal_id

def load_signal_history(limit: int = 100) -> list:
    create_tables()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cycle_signal_history ORDER BY signal_date DESC, id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def save_backtest_result(data: dict) -> int:
    create_tables()
    conn = get_connection()
    cursor = conn.cursor()
    
    cols = [
        "signal_id", "ticker", "signal_date", "forecast_horizon_days", "expected_direction",
        "actual_return", "max_favorable_excursion", "max_adverse_excursion", "realized_vol_change",
        "strategy", "strategy_outcome", "pnl_estimate", "win_flag", "notes"
    ]
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO cycle_backtest_results ({', '.join(cols)}) VALUES ({placeholders})"
    
    vals = [data.get(c) for c in cols]
    cursor.execute(sql, vals)
    bt_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return bt_id

def load_backtests(ticker: str = None) -> list:
    create_tables()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if ticker:
        cursor.execute("SELECT * FROM cycle_backtest_results WHERE ticker = ? ORDER BY signal_date DESC", (ticker,))
    else:
        cursor.execute("SELECT * FROM cycle_backtest_results ORDER BY signal_date DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def save_events(events: list):
    create_tables()
    conn = get_connection()
    cursor = conn.cursor()
    for ev in events:
        cursor.execute("""
        INSERT INTO cycle_event_calendar (event_date, event_type, event_name, expected_impact, source)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(event_date, event_type, event_name) DO UPDATE SET
            expected_impact = excluded.expected_impact,
            source = excluded.source
        """, (ev["event_date"], ev["event_type"], ev.get("event_name"), ev.get("expected_impact"), ev.get("source")))
    conn.commit()
    conn.close()

def load_events(start_date: str, end_date: str) -> list:
    create_tables()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cycle_event_calendar WHERE event_date BETWEEN ? AND ? ORDER BY event_date ASC", (start_date, end_date))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def save_trade_outcome(data: dict) -> int:
    create_tables()
    conn = get_connection()
    cursor = conn.cursor()
    
    cols = [
        "signal_id", "entry_date", "exit_date", "entry_price", "exit_price",
        "max_favorable_move", "max_adverse_move", "pnl_percent", "win_loss_flag", "outcome_notes"
    ]
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO cycle_trade_outcomes ({', '.join(cols)}) VALUES ({placeholders})"
    
    vals = [data.get(c) for c in cols]
    cursor.execute(sql, vals)
    outcome_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return outcome_id
