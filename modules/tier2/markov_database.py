import sqlite3
import logging
from pathlib import Path
from utils.persistence import get_db_path

logger = logging.getLogger("MarkovDatabase")

def get_connection():
    db_path = get_db_path("calendar_scoring")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path, timeout=15)

def create_markov_tables():
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Daily State Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS markov_daily_state (
        trade_date TEXT,
        ticker TEXT,
        close_price REAL,
        daily_return REAL,
        rolling_20d_return REAL,
        realized_vol_20d REAL,
        price_state TEXT,
        volatility_state TEXT,
        combined_state TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (trade_date, ticker)
    );
    """)
    
    # 2. Transition Matrix
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS markov_transition_matrix (
        as_of_date TEXT,
        ticker TEXT,
        from_state TEXT,
        to_state TEXT,
        transition_count INTEGER,
        transition_probability REAL,
        lookback_days INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (as_of_date, ticker, from_state, to_state)
    );
    """)
    
    # 3. Forecast Output
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS markov_forecast (
        as_of_date TEXT,
        ticker TEXT,
        current_state TEXT,
        bull_prob_1d REAL,
        sideways_prob_1d REAL,
        bear_prob_1d REAL,
        bull_prob_5d REAL,
        sideways_prob_5d REAL,
        bear_prob_5d REAL,
        bull_prob_20d REAL,
        sideways_prob_20d REAL,
        bear_prob_20d REAL,
        markov_signal REAL,
        stickiness_score REAL,
        expected_duration REAL,
        final_regime_label TEXT,
        final_action TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (as_of_date, ticker)
    );
    """)
    
    # 4. Backtest Results
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS markov_backtest_results (
        run_date TEXT,
        ticker TEXT,
        strategy_name TEXT,
        start_date TEXT,
        end_date TEXT,
        total_return REAL,
        max_drawdown REAL,
        sharpe_ratio REAL,
        win_rate REAL,
        prediction_accuracy REAL,
        bull_precision REAL,
        bear_precision REAL,
        sideways_precision REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (run_date, ticker, strategy_name)
    );
    """)
    
    # Migration: add final_action column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE markov_forecast ADD COLUMN final_action TEXT;")
    except sqlite3.OperationalError:
        pass  # Column already exists
        
    conn.commit()
    conn.close()
    logger.info("Markov database tables initialized successfully.")

def save_daily_states(records: list):
    """Save daily states list of dicts to db."""
    if not records:
        return
    conn = get_connection()
    cursor = conn.cursor()
    sql = """
    INSERT OR REPLACE INTO markov_daily_state 
    (trade_date, ticker, close_price, daily_return, rolling_20d_return, realized_vol_20d, price_state, volatility_state, combined_state)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    data = [
        (
            r.get("trade_date"), r.get("ticker"), r.get("close_price"), r.get("daily_return"),
            r.get("rolling_20d_return"), r.get("realized_vol_20d"), r.get("price_state"),
            r.get("volatility_state"), r.get("combined_state")
        )
        for r in records
    ]
    cursor.executemany(sql, data)
    conn.commit()
    conn.close()

def save_transition_matrix(records: list):
    """Save transition matrix counts and probabilities."""
    if not records:
        return
    conn = get_connection()
    cursor = conn.cursor()
    sql = """
    INSERT OR REPLACE INTO markov_transition_matrix
    (as_of_date, ticker, from_state, to_state, transition_count, transition_probability, lookback_days)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    data = [
        (
            r.get("as_of_date"), r.get("ticker"), r.get("from_state"), r.get("to_state"),
            r.get("transition_count"), r.get("transition_probability"), r.get("lookback_days")
        )
        for r in records
    ]
    cursor.executemany(sql, data)
    conn.commit()
    conn.close()

def save_forecast(f: dict):
    """Save forecast output dictionary."""
    conn = get_connection()
    cursor = conn.cursor()
    sql = """
    INSERT OR REPLACE INTO markov_forecast
    (as_of_date, ticker, current_state, bull_prob_1d, sideways_prob_1d, bear_prob_1d,
     bull_prob_5d, sideways_prob_5d, bear_prob_5d, bull_prob_20d, sideways_prob_20d, bear_prob_20d,
     markov_signal, stickiness_score, expected_duration, final_regime_label, final_action)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.execute(sql, (
        f.get("as_of_date"), f.get("ticker"), f.get("current_state"),
        f.get("bull_prob_1d"), f.get("sideways_prob_1d"), f.get("bear_prob_1d"),
        f.get("bull_prob_5d"), f.get("sideways_prob_5d"), f.get("bear_prob_5d"),
        f.get("bull_prob_20d"), f.get("sideways_prob_20d"), f.get("bear_prob_20d"),
        f.get("markov_signal"), f.get("stickiness_score"), f.get("expected_duration"), f.get("final_regime_label"), f.get("final_action")
    ))
    conn.commit()
    conn.close()

def save_backtest_results(b: dict):
    """Save backtest results summary dictionary."""
    conn = get_connection()
    cursor = conn.cursor()
    sql = """
    INSERT OR REPLACE INTO markov_backtest_results
    (run_date, ticker, strategy_name, start_date, end_date, total_return, max_drawdown,
     sharpe_ratio, win_rate, prediction_accuracy, bull_precision, bear_precision, sideways_precision)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.execute(sql, (
        b.get("run_date"), b.get("ticker"), b.get("strategy_name"),
        b.get("start_date"), b.get("end_date"), b.get("total_return"),
        b.get("max_drawdown"), b.get("sharpe_ratio"), b.get("win_rate"),
        b.get("prediction_accuracy"), b.get("bull_precision"), b.get("bear_precision"), b.get("sideways_precision")
    ))
    conn.commit()
    conn.close()

def get_latest_forecast(ticker: str) -> dict:
    """Retrieve the latest forecast for a specific ticker."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT as_of_date, current_state, bull_prob_1d, sideways_prob_1d, bear_prob_1d,
               bull_prob_5d, sideways_prob_5d, bear_prob_5d, bull_prob_20d, sideways_prob_20d, bear_prob_20d,
               markov_signal, stickiness_score, expected_duration, final_regime_label, final_action, created_at
        FROM markov_forecast
        WHERE ticker = ?
        ORDER BY as_of_date DESC LIMIT 1
    """, (ticker,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "as_of_date": row[0],
            "current_state": row[1],
            "bull_prob_1d": row[2],
            "sideways_prob_1d": row[3],
            "bear_prob_1d": row[4],
            "bull_prob_5d": row[5],
            "sideways_prob_5d": row[6],
            "bear_prob_5d": row[7],
            "bull_prob_20d": row[8],
            "sideways_prob_20d": row[9],
            "bear_prob_20d": row[10],
            "markov_signal": row[11],
            "stickiness_score": row[12],
            "expected_duration": row[13],
            "final_regime_label": row[14],
            "final_action": row[15] if row[15] is not None else "Hold",
            "created_at": row[16]
        }
    return {}

def get_latest_transition_matrix(ticker: str) -> tuple:
    """Retrieve the latest transition matrix for a specific ticker."""
    import numpy as np
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(as_of_date) FROM markov_transition_matrix WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        if not row or not row[0]:
            return None, None
        latest_date = row[0]
        
        cursor.execute("""
            SELECT from_state, to_state, transition_probability
            FROM markov_transition_matrix
            WHERE ticker = ? AND as_of_date = ?
        """, (ticker, latest_date))
        rows = cursor.fetchall()
    finally:
        conn.close()
        
    if not rows:
        return None, None
        
    state_list = ["BULL", "SIDEWAYS", "BEAR"]
    state_to_idx = {s: i for i, s in enumerate(state_list)}
    P = np.zeros((3, 3))
    for from_state, to_state, prob in rows:
        if from_state in state_to_idx and to_state in state_to_idx:
            P[state_to_idx[from_state], state_to_idx[to_state]] = prob
    return P, state_list

def get_latest_backtest(ticker: str) -> dict:
    """Retrieve the latest backtest results for a specific ticker."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT run_date, strategy_name, start_date, end_date, total_return, max_drawdown,
               sharpe_ratio, win_rate, prediction_accuracy, bull_precision, bear_precision, sideways_precision
        FROM markov_backtest_results
        WHERE ticker = ?
        ORDER BY run_date DESC LIMIT 1
    """, (ticker,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "run_date": row[0],
            "strategy_name": row[1],
            "start_date": row[2],
            "end_date": row[3],
            "total_return": row[4],
            "max_drawdown": row[5],
            "sharpe_ratio": row[6],
            "win_rate": row[7],
            "prediction_accuracy": row[8],
            "bull_precision": row[9],
            "bear_precision": row[10],
            "sideways_precision": row[11]
        }
    return {}

def get_historical_states(ticker: str) -> list:
    """Retrieve all historical daily state records for a ticker."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT trade_date, close_price, daily_return, rolling_20d_return, realized_vol_20d,
               price_state, volatility_state, combined_state
        FROM markov_daily_state
        WHERE ticker = ?
        ORDER BY trade_date ASC
    """, (ticker,))
    rows = cursor.fetchall()
    conn.close()
    records = []
    for r in rows:
        records.append({
            "trade_date": r[0],
            "close_price": r[1],
            "daily_return": r[2],
            "rolling_20d_return": r[3],
            "realized_vol_20d": r[4],
            "price_state": r[5],
            "volatility_state": r[6],
            "combined_state": r[7]
        })
    return records
