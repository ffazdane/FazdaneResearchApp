"""
Database Store for Correlation Analysis Runs
============================================
Handles schema definition, custom JSON serialization for pandas DataFrames,
and SQLite persistence for correlation, segmentation, and backtest results.
"""

import io
import json
import math
import logging
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
from utils.persistence import get_db_path

logger = logging.getLogger("CorrelationAnalysisDB")

def get_connection():
    db_path = get_db_path("correlation_analysis")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path, timeout=10)

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS correlation_runs (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        universe_name TEXT NOT NULL,
        tickers TEXT NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        method TEXT NOT NULL,
        run_datetime TEXT NOT NULL,
        price_data_json TEXT,
        results_json TEXT
    );
    """)
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_corr_runs_uni_dt 
    ON correlation_runs(universe_name, run_datetime);
    """)
    conn.commit()
    conn.close()
    logger.info("Correlation analysis SQLite tables verified/created.")

# --------------------------------------------------------------------------- #
# Custom JSON Serialization Helpers for DataFrames
# --------------------------------------------------------------------------- #

class CustomCorrEncoder(json.JSONEncoder):
    """JSON encoder that handles pandas DataFrames, Series, numpy types, and float bounds."""
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return {
                "__type__": "DataFrame",
                "data": obj.to_json(orient='split', date_format='iso')
            }
        if isinstance(obj, pd.Series):
            return {
                "__type__": "Series",
                "data": obj.to_json(orient='split', date_format='iso')
            }
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def custom_corr_decoder(pct):
    if isinstance(pct, dict):
        if pct.get("__type__") == "DataFrame":
            return pd.read_json(io.StringIO(pct["data"]), orient='split')
        elif pct.get("__type__") == "Series":
            return pd.read_json(io.StringIO(pct["data"]), orient='split', typ='series')
        
        new_dict = {}
        for k, v in pct.items():
            new_dict[k] = custom_corr_decoder(v)
        return new_dict
    elif isinstance(pct, list):
        return [custom_corr_decoder(item) for item in pct]
    return pct

def serialize_data(data: dict) -> str:
    return json.dumps(data, cls=CustomCorrEncoder)

def deserialize_data(json_str: str) -> dict:
    if not json_str:
        return {}
    return json.loads(json_str, object_hook=custom_corr_decoder)

# --------------------------------------------------------------------------- #
# DB persistence APIs
# --------------------------------------------------------------------------- #

def save_run(
    universe_name: str,
    tickers: list[str],
    start_date: str,
    end_date: str,
    method: str,
    prices_df: pd.DataFrame,
    results_dict: dict
) -> int:
    """Save correlation run details, closes, and analysis outputs to database."""
    create_tables()
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        run_datetime = datetime.now().isoformat()
        tickers_str = ",".join(tickers)
        
        price_data_json = prices_df.to_json(orient='split', date_format='iso')
        results_json = serialize_data(results_dict)
        
        cursor.execute("""
            INSERT INTO correlation_runs (
                universe_name, tickers, start_date, end_date, method,
                run_datetime, price_data_json, results_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            universe_name, tickers_str, start_date, end_date, method,
            run_datetime, price_data_json, results_json
        ))
        run_id = cursor.lastrowid
        conn.commit()
        logger.info(f"Saved correlation run ID {run_id} for universe '{universe_name}'.")
        return run_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving correlation run: {e}")
        raise e
    finally:
        conn.close()

def fetch_latest_run(universe_name: str) -> dict | None:
    """Retrieve the absolute latest run for a specific universe."""
    create_tables()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT * FROM correlation_runs
            WHERE universe_name = ?
            ORDER BY run_datetime DESC LIMIT 1
        """, (universe_name,))
        row = cursor.fetchone()
        if not row:
            return None
        
        run_dict = dict(row)
        # Reconstruct pandas objects
        run_dict["tickers"] = run_dict["tickers"].split(",")
        run_dict["prices_df"] = pd.read_json(io.StringIO(run_dict["price_data_json"]), orient='split')
        run_dict["results"] = deserialize_data(run_dict["results_json"])
        return run_dict
    except Exception as e:
        logger.error(f"Error fetching latest correlation run for '{universe_name}': {e}")
        return None
    finally:
        conn.close()

def fetch_run_by_id(run_id: int) -> dict | None:
    """Retrieve a specific run by ID."""
    create_tables()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM correlation_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        run_dict = dict(row)
        run_dict["tickers"] = run_dict["tickers"].split(",")
        run_dict["prices_df"] = pd.read_json(io.StringIO(run_dict["price_data_json"]), orient='split')
        run_dict["results"] = deserialize_data(run_dict["results_json"])
        return run_dict
    except Exception as e:
        logger.error(f"Error fetching correlation run by ID {run_id}: {e}")
        return None
    finally:
        conn.close()

def fetch_run_history(limit: int = 50) -> list[dict]:
    """Get metadata summary of recent correlation runs."""
    create_tables()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT run_id, universe_name, tickers, start_date, end_date, method, run_datetime
            FROM correlation_runs
            ORDER BY run_datetime DESC
            LIMIT ?
        """, (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        for r in rows:
            r["tickers"] = r["tickers"].split(",")
        return rows
    except Exception as e:
        logger.error(f"Error fetching correlation run history: {e}")
        return []
    finally:
        conn.close()

# Auto-initialize database schema on import
create_tables()

