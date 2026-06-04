import sqlite3
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from utils.persistence import get_db_path

logger = logging.getLogger("TradeRecommendationDB")

def get_connection():
    db_path = get_db_path("trade_recommendation")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path, timeout=10)

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table 1: ticker_signal_snapshot
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ticker_signal_snapshot (
        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        snapshot_datetime DATETIME NOT NULL,
        price REAL,
        trend_state TEXT,
        market_state TEXT,
        trade_decision TEXT,
        recommended_strategy TEXT,
        trade_score REAL,
        trend_score REAL,
        momentum_score REAL,
        range_score REAL,
        volatility_score REAL,
        liquidity_score REAL,
        event_risk_score REAL,
        expected_40d_path TEXT,
        expected_40d_low REAL,
        expected_40d_high REAL,
        support_level REAL,
        resistance_level REAL,
        trigger_level REAL,
        invalidation_level REAL,
        notes TEXT,
        raw_analysis_json TEXT
    );
    """)
    
    # Check if raw_analysis_json column exists (for backward compatibility)
    cursor.execute("PRAGMA table_info(ticker_signal_snapshot)")
    cols = [row[1] for row in cursor.fetchall()]
    if "raw_analysis_json" not in cols:
        try:
            cursor.execute("ALTER TABLE ticker_signal_snapshot ADD COLUMN raw_analysis_json TEXT")
            conn.commit()
            logger.info("Successfully added raw_analysis_json column to ticker_signal_snapshot.")
        except Exception as e:
            logger.warning(f"Could not add raw_analysis_json column to ticker_signal_snapshot: {e}")

    
    # Table 2: indicator_snapshot
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS indicator_snapshot (
        indicator_id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_id INTEGER,
        ticker TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        price REAL,
        ma20 REAL,
        ma50 REAL,
        ma200 REAL,
        vwap REAL,
        fdts_delta REAL,
        fdts_signal TEXT,
        macd_value REAL,
        macd_avg REAL,
        macd_hist REAL,
        macd_signal TEXT,
        wpr_value REAL,
        wpr_signal TEXT,
        darvas_upper REAL,
        darvas_lower REAL,
        darvas_signal TEXT,
        regression_upper REAL,
        regression_middle REAL,
        regression_lower REAL,
        ichimoku_span_a REAL,
        ichimoku_span_b REAL,
        cloud_signal TEXT,
        atr14 REAL,
        iv_rank REAL,
        FOREIGN KEY(snapshot_id) REFERENCES ticker_signal_snapshot(snapshot_id)
    );
    """)
    
    # Table 3: trade_plan
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trade_plan (
        trade_plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_id INTEGER,
        ticker TEXT NOT NULL,
        decision TEXT,
        strategy TEXT,
        option_structure TEXT,
        entry_trigger TEXT,
        target_zone TEXT,
        invalidation_rule TEXT,
        adjustment_rule TEXT,
        profit_target TEXT,
        max_loss_rule TEXT,
        rationale TEXT,
        created_datetime DATETIME NOT NULL,
        FOREIGN KEY(snapshot_id) REFERENCES ticker_signal_snapshot(snapshot_id)
    );
    """)
    
    # Table 4: trade_outcome_log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trade_outcome_log (
        outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_plan_id INTEGER,
        ticker TEXT NOT NULL,
        entry_date DATE,
        review_date DATE,
        price_at_signal REAL,
        price_after_5d REAL,
        price_after_10d REAL,
        price_after_20d REAL,
        price_after_40d REAL,
        max_favorable_move REAL,
        max_adverse_move REAL,
        strategy_result TEXT,
        estimated_pnl REAL,
        notes TEXT,
        FOREIGN KEY(trade_plan_id) REFERENCES trade_plan(trade_plan_id)
    );
    """)
    
    conn.commit()
    conn.close()
    logger.info("Trade Recommendation Engine tables created/verified.")

def save_signal_snapshot(snapshot: dict, indicators: list, plan: dict = None) -> int:
    """Save ticker signal snapshot, daily/1H indicators, and the trade plan."""
    create_tables()
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # 1. Insert ticker_signal_snapshot
        snap_cols = [
            "ticker", "snapshot_datetime", "price", "trend_state", "market_state",
            "trade_decision", "recommended_strategy", "trade_score", "trend_score",
            "momentum_score", "range_score", "volatility_score", "liquidity_score",
            "event_risk_score", "expected_40d_path", "expected_40d_low", "expected_40d_high",
            "support_level", "resistance_level", "trigger_level", "invalidation_level", "notes",
            "raw_analysis_json"
        ]
        placeholders = ", ".join(["?"] * len(snap_cols))
        snap_sql = f"INSERT INTO ticker_signal_snapshot ({', '.join(snap_cols)}) VALUES ({placeholders})"
        snap_vals = [snapshot.get(c) for c in snap_cols]
        cursor.execute(snap_sql, snap_vals)
        snapshot_id = cursor.lastrowid
        
        # 2. Insert indicator_snapshots
        ind_cols = [
            "snapshot_id", "ticker", "timeframe", "price", "ma20", "ma50", "ma200", "vwap",
            "fdts_delta", "fdts_signal", "macd_value", "macd_avg", "macd_hist", "macd_signal",
            "wpr_value", "wpr_signal", "darvas_upper", "darvas_lower", "darvas_signal",
            "regression_upper", "regression_middle", "regression_lower", "ichimoku_span_a",
            "ichimoku_span_b", "cloud_signal", "atr14", "iv_rank"
        ]
        ind_placeholders = ", ".join(["?"] * len(ind_cols))
        ind_sql = f"INSERT INTO indicator_snapshot ({', '.join(ind_cols)}) VALUES ({ind_placeholders})"
        
        for ind in indicators:
            ind["snapshot_id"] = snapshot_id
            ind_vals = [ind.get(c) for c in ind_cols]
            cursor.execute(ind_sql, ind_vals)
            
        # 3. Insert trade_plan if available
        if plan:
            plan_cols = [
                "snapshot_id", "ticker", "decision", "strategy", "option_structure",
                "entry_trigger", "target_zone", "invalidation_rule", "adjustment_rule",
                "profit_target", "max_loss_rule", "rationale", "created_datetime"
            ]
            plan_placeholders = ", ".join(["?"] * len(plan_cols))
            plan_sql = f"INSERT INTO trade_plan ({', '.join(plan_cols)}) VALUES ({plan_placeholders})"
            plan["snapshot_id"] = snapshot_id
            plan["created_datetime"] = snapshot.get("snapshot_datetime")
            plan_vals = [plan.get(c) for c in plan_cols]
            cursor.execute(plan_sql, plan_vals)
            
        conn.commit()
        return snapshot_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving snapshot: {e}")
        raise e
    finally:
        conn.close()

def log_trade_outcome(outcome: dict) -> int:
    """Insert or update a trade outcome log."""
    create_tables()
    conn = get_connection()
    cursor = conn.cursor()
    
    cols = [
        "trade_plan_id", "ticker", "entry_date", "review_date", "price_at_signal",
        "price_after_5d", "price_after_10d", "price_after_20d", "price_after_40d",
        "max_favorable_move", "max_adverse_move", "strategy_result", "estimated_pnl", "notes"
    ]
    
    # Check if outcome already exists for this trade_plan_id
    cursor.execute("SELECT outcome_id FROM trade_outcome_log WHERE trade_plan_id = ?", (outcome.get("trade_plan_id"),))
    row = cursor.fetchone()
    
    try:
        if row:
            outcome_id = row[0]
            # Update
            set_clause = ", ".join([f"{c} = ?" for c in cols])
            sql = f"UPDATE trade_outcome_log SET {set_clause} WHERE outcome_id = ?"
            vals = [outcome.get(c) for c in cols] + [outcome_id]
            cursor.execute(sql, vals)
        else:
            # Insert
            placeholders = ", ".join(["?"] * len(cols))
            sql = f"INSERT INTO trade_outcome_log ({', '.join(cols)}) VALUES ({placeholders})"
            vals = [outcome.get(c) for c in cols]
            cursor.execute(sql, vals)
            outcome_id = cursor.lastrowid
            
        conn.commit()
        return outcome_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error logging outcome: {e}")
        raise e
    finally:
        conn.close()

def fetch_historical_snapshots(limit: int = 100) -> list:
    """Retrieve history of signal snapshots."""
    create_tables()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, p.trade_plan_id, p.option_structure, p.entry_trigger, p.invalidation_rule, p.adjustment_rule, p.rationale
        FROM ticker_signal_snapshot s
        LEFT JOIN trade_plan p ON s.snapshot_id = p.snapshot_id
        ORDER BY s.snapshot_datetime DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def fetch_indicator_snapshots(snapshot_id: int) -> list:
    create_tables()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM indicator_snapshot WHERE snapshot_id = ?", (snapshot_id,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def fetch_active_trade_plans() -> list:
    """Retrieve trade plans that are deployed and do not have outcomes yet."""
    create_tables()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.*, s.price as current_price_at_signal, s.snapshot_datetime
        FROM trade_plan p
        JOIN ticker_signal_snapshot s ON p.snapshot_id = s.snapshot_id
        LEFT JOIN trade_outcome_log o ON p.trade_plan_id = o.trade_plan_id
        WHERE p.decision = 'Deploy' AND o.outcome_id IS NULL
        ORDER BY p.created_datetime DESC
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def fetch_trade_outcomes(limit: int = 100) -> list:
    create_tables()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT o.*, p.strategy, p.option_structure, p.decision, p.snapshot_id
        FROM trade_outcome_log o
        JOIN trade_plan p ON o.trade_plan_id = p.trade_plan_id
        ORDER BY o.review_date DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# ══════════════════════════════════════════════════════════════════════
# DATA SERIALIZATION & CACHING HELPERS
# ══════════════════════════════════════════════════════════════════════

import json
import numpy as np
import pandas as pd

class CustomAnalysisEncoder(json.JSONEncoder):
    """JSON encoder that handles DataFrames, Series, numpy types, NaN, and Infinity."""
    def default(self, obj):
        # ─ pandas ───────────────────────────────────────────
        if isinstance(obj, pd.DataFrame):
            return {"__type__": "DataFrame", "data": obj.to_json(orient='split', date_format='iso')}
        if isinstance(obj, pd.Series):
            return {"__type__": "Series", "data": obj.to_json(orient='split', date_format='iso')}
        # ─ numpy scalars ───────────────────────────────────
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        # ─ datetime-like ───────────────────────────────────
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        # ─ native float NaN / Inf ────────────────────────────
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        # ─ fallback ───────────────────────────────────────
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def custom_analysis_decoder(pct):
    if isinstance(pct, dict):
        import io
        if pct.get("__type__") == "DataFrame":
            return pd.read_json(io.StringIO(pct["data"]), orient='split')
        elif pct.get("__type__") == "Series":
            return pd.read_json(io.StringIO(pct["data"]), orient='split', typ='series')
        
        new_dict = {}
        for k, v in pct.items():
            new_dict[k] = custom_analysis_decoder(v)
        return new_dict
    elif isinstance(pct, list):
        return [custom_analysis_decoder(item) for item in pct]
    return pct

def serialize_analysis_data(data: dict) -> str:
    """Serialize the analysis results dictionary (including dataframes) to JSON.
    
    NaN and Infinity values are converted to null so the output is valid JSON
    that can be safely parsed by any strict JSON parser.
    """
    return json.dumps(data, cls=CustomAnalysisEncoder)

def deserialize_analysis_data(json_str: str) -> dict:
    """Deserialize the JSON cache back to the full analysis results dictionary.
    
    Handles NaN / Infinity literals that Python's json module may have written
    in previous versions by replacing them with null before parsing.
    """
    if not json_str:
        return {}
    # Replace any bare NaN/Infinity literals (written by older Python json encoder)
    json_str = re.sub(r'\bNaN\b', 'null', json_str)
    json_str = re.sub(r'\bInfinity\b', 'null', json_str)
    json_str = re.sub(r'\b-Infinity\b', 'null', json_str)
    return json.loads(json_str, object_hook=custom_analysis_decoder)

def fetch_latest_ticker_snapshot(ticker: str) -> dict:
    """Fetch the absolute latest snapshot row for a specific ticker."""
    create_tables()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM ticker_signal_snapshot 
        WHERE ticker = ? 
        ORDER BY snapshot_datetime DESC LIMIT 1
    """, (ticker,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

