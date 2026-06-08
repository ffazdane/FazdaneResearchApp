import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from utils.persistence import get_db_path

logger = logging.getLogger("VolatilityRunDB")

MODEL_VERSION = "v1.0"

def get_connection():
    """Get sqlite3 connection to the volatility engine database, creating parent directories if needed."""
    db_path = get_db_path("volatility_engine")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path, timeout=10)

def create_tables():
    """Create the volatility run history table and necessary indices if they do not exist."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS volatility_run_history (
            run_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_datetime            TEXT NOT NULL,
            run_version             TEXT NOT NULL,
            symbol                  TEXT NOT NULL,
            volatility_risk_score   REAL NOT NULL,
            risk_regime             TEXT NOT NULL,
            delta_action            TEXT NOT NULL,
            vix_regime_score        REAL,
            put_call_score          REAL,
            price_action_score      REAL,
            breadth_score           REAL,
            liquidity_gamma_score   REAL,
            macro_event_score       REAL,
            vix_current             REAL,
            vix_percentile          REAL,
            vvix_current            REAL,
            hvr                     REAL,
            atm_iv                  REAL,
            hv20                    REAL,
            term_shape              TEXT,
            skew_label              TEXT,
            put_call_ratio          REAL,
            spy_vs_20ema_pct        REAL,
            spy_vs_50ema_pct        REAL,
            gamma_regime            TEXT,
            days_to_earnings        INTEGER,
            macro_event_flagged     INTEGER,
            created_at              TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vrh_symbol_datetime ON volatility_run_history(symbol, run_datetime DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vrh_latest ON volatility_run_history(run_datetime DESC);")
        
        conn.commit()
    except Exception as e:
        logger.error(f"Error creating volatility engine tables: {e}", exc_info=True)
        raise e
    finally:
        conn.close()

def save_volatility_run(
    symbol: str,
    volatility_risk_score: float,
    risk_regime: str,
    delta_action: str,
    sub_scores: dict,
    raw_inputs: dict,
    run_datetime: str = None
) -> int:
    """Persist a complete run with all sub-scores and raw inputs to the database."""
    create_tables()
    if run_datetime is None:
        run_datetime = datetime.now().isoformat()
    
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO volatility_run_history (
            run_datetime, run_version, symbol, volatility_risk_score, risk_regime, delta_action,
            vix_regime_score, put_call_score, price_action_score, breadth_score, liquidity_gamma_score, macro_event_score,
            vix_current, vix_percentile, vvix_current, hvr, atm_iv, hv20, term_shape, skew_label,
            put_call_ratio, spy_vs_20ema_pct, spy_vs_50ema_pct, gamma_regime, days_to_earnings, macro_event_flagged
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_datetime,
            MODEL_VERSION,
            symbol.upper(),
            volatility_risk_score,
            risk_regime,
            delta_action,
            sub_scores.get("vix_regime_score"),
            sub_scores.get("put_call_score"),
            sub_scores.get("price_action_score"),
            sub_scores.get("breadth_score"),
            sub_scores.get("liquidity_gamma_score"),
            sub_scores.get("macro_event_score"),
            raw_inputs.get("vix_current"),
            raw_inputs.get("vix_percentile"),
            raw_inputs.get("vvix_current"),
            raw_inputs.get("hvr"),
            raw_inputs.get("atm_iv"),
            raw_inputs.get("hv20"),
            raw_inputs.get("term_shape"),
            raw_inputs.get("skew_label"),
            raw_inputs.get("put_call_ratio"),
            raw_inputs.get("spy_vs_20ema_pct"),
            raw_inputs.get("spy_vs_50ema_pct"),
            raw_inputs.get("gamma_regime"),
            raw_inputs.get("days_to_earnings"),
            1 if raw_inputs.get("macro_event_flagged") else 0
        ))
        conn.commit()
        last_id = cursor.lastrowid
        logger.info(f"Saved volatility run for {symbol} with score {volatility_risk_score} (id: {last_id})")
        return last_id
    except Exception as e:
        logger.error(f"Failed to save volatility run: {e}", exc_info=True)
        return -1
    finally:
        conn.close()

def _row_to_dict(row, description) -> dict:
    if row is None:
        return {}
    cols = [col[0] for col in description]
    res = dict(zip(cols, row))
    # Unpack sub_scores and raw_inputs to conform to expected structure
    res["sub_scores"] = {
        "vix_regime_score": res.get("vix_regime_score"),
        "put_call_score": res.get("put_call_score"),
        "price_action_score": res.get("price_action_score"),
        "breadth_score": res.get("breadth_score"),
        "liquidity_gamma_score": res.get("liquidity_gamma_score"),
        "macro_event_score": res.get("macro_event_score")
    }
    res["raw_inputs"] = {
        "vix_current": res.get("vix_current"),
        "vix_percentile": res.get("vix_percentile"),
        "vvix_current": res.get("vvix_current"),
        "hvr": res.get("hvr"),
        "atm_iv": res.get("atm_iv"),
        "hv20": res.get("hv20"),
        "term_shape": res.get("term_shape"),
        "skew_label": res.get("skew_label"),
        "put_call_ratio": res.get("put_call_ratio"),
        "spy_vs_20ema_pct": res.get("spy_vs_20ema_pct"),
        "spy_vs_50ema_pct": res.get("spy_vs_50ema_pct"),
        "gamma_regime": res.get("gamma_regime"),
        "days_to_earnings": res.get("days_to_earnings"),
        "macro_event_flagged": bool(res.get("macro_event_flagged"))
    }
    return res

def get_latest_run(symbol: str) -> dict | None:
    """Get the most recent run for a specific symbol."""
    create_tables()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT * FROM volatility_run_history 
        WHERE symbol = ? 
        ORDER BY run_datetime DESC 
        LIMIT 1
        """, (symbol.upper(),))
        row = cursor.fetchone()
        if row:
            return _row_to_dict(row, cursor.description)
        return None
    except Exception as e:
        logger.error(f"Error fetching latest volatility run for {symbol}: {e}", exc_info=True)
        return None
    finally:
        conn.close()

def get_latest_run_any() -> dict | None:
    """Get the most recent run across all symbols."""
    create_tables()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT * FROM volatility_run_history 
        ORDER BY run_datetime DESC 
        LIMIT 1
        """)
        row = cursor.fetchone()
        if row:
            return _row_to_dict(row, cursor.description)
        return None
    except Exception as e:
        logger.error(f"Error fetching latest volatility run (any): {e}", exc_info=True)
        return None
    finally:
        conn.close()

def get_run_history(symbol: str, limit: int = 50) -> list:
    """Get historical runs for a symbol, sorted chronologically."""
    create_tables()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT * FROM volatility_run_history 
        WHERE symbol = ? 
        ORDER BY run_datetime DESC 
        LIMIT ?
        """, (symbol.upper(), limit))
        rows = cursor.fetchall()
        description = cursor.description
        runs = [_row_to_dict(row, description) for row in rows]
        # Return chronologically (oldest first)
        runs.reverse()
        return runs
    except Exception as e:
        logger.error(f"Error fetching volatility run history for {symbol}: {e}", exc_info=True)
        return []
    finally:
        conn.close()

def get_active_risk_regime() -> str:
    """Quick accessor for the current risk regime label."""
    run = get_latest_run_any()
    if run:
        return run.get("risk_regime", "LOW")
    return "LOW"

def get_volatility_risk_score(symbol: str) -> float:
    """Quick accessor for the current score of a symbol."""
    run = get_latest_run(symbol)
    if run:
        return run.get("volatility_risk_score", 0.0)
    return 0.0
