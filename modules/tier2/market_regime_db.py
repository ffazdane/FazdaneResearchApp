import sqlite3
import logging
import json
from datetime import datetime
from pathlib import Path
from utils.persistence import get_db_path

logger = logging.getLogger("MarketRegimeDB")

def get_connection():
    """Retrieve SQLite connection to the calendar_scoring database."""
    db_path = get_db_path("calendar_scoring")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path, timeout=15)

def create_regime_tables():
    """Create the Market Regime Engine tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. market_regime_daily
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS market_regime_daily (
        regime_date TEXT NOT NULL,
        spy_close REAL,
        qqq_close REAL,
        iwm_close REAL,
        dia_close REAL,
        smh_close REAL,
        vix_close REAL,
        trend_score REAL,
        breadth_score REAL,
        volatility_score REAL,
        momentum_score REAL,
        risk_sentiment_score REAL,
        final_regime_score REAL,
        regime_name TEXT,
        confidence_score REAL,
        market_bias TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (regime_date)
    );
    """)
    
    # 2. market_regime_strategy_rules
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS market_regime_strategy_rules (
        rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
        regime_name TEXT,
        strategy_name TEXT,
        strategy_status TEXT, -- 'Preferred' | 'Avoid' | 'Blocked'
        reason TEXT,
        min_score REAL,
        max_score REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 3. market_regime_history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS market_regime_history (
        history_id INTEGER PRIMARY KEY AUTOINCREMENT,
        regime_date TEXT NOT NULL,
        previous_regime TEXT,
        current_regime TEXT,
        regime_change_flag INTEGER, -- 1 if changed, 0 otherwise
        trigger_reason TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    conn.commit()
    conn.close()
    logger.info("Market Regime Engine tables initialized successfully.")
    
    # Pre-populate strategy rules if empty
    prepopulate_strategy_rules()

def prepopulate_strategy_rules():
    """Pre-populate option strategy guidance rules by regime if table is empty."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM market_regime_strategy_rules")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return
        
    rules = [
        # Strong Buy The Dip
        ("Strong Buy The Dip", "Call Diagonal", "Preferred", "Captures bullish trend with controlled risk.", 80.0, 100.0),
        ("Strong Buy The Dip", "Call Calendar", "Preferred", "Benefits from neutral-to-bullish continuation and low front-month IV.", 80.0, 100.0),
        ("Strong Buy The Dip", "Bull Put Spread", "Preferred", "Credits premiums from high-probability support levels.", 80.0, 100.0),
        ("Strong Buy The Dip", "Bear Call Spread", "Avoid", "High risk of losses in strongly trending bull market.", 80.0, 100.0),
        ("Strong Buy The Dip", "Aggressive shorting", "Avoid", "Chasing momentum to the downside is blocked.", 80.0, 100.0),
        ("Strong Buy The Dip", "Heavy negative delta", "Avoid", "Avoid counter-trend exposure in high conviction bull tape.", 80.0, 100.0),
        
        # Buy Dips Selectively
        ("Buy Dips Selectively", "Call Diagonal", "Preferred", "Allows selective bullish trend exposure.", 60.0, 79.0),
        ("Buy Dips Selectively", "Bull Put Spread", "Preferred", "Credits options premiums from pullbacks near support.", 60.0, 79.0),
        ("Buy Dips Selectively", "Selective Call Calendar", "Preferred", "Allowed if term structure is normal and volatility is low.", 60.0, 79.0),
        ("Buy Dips Selectively", "Chasing extended stocks", "Avoid", " Chasing extended high-beta leaders is blocked due to narrow breadth.", 60.0, 79.0),
        ("Buy Dips Selectively", "Large unhedged bullish trades", "Avoid", "Risk of sudden rotation or consolidation remains elevated.", 60.0, 79.0),
        
        # Range Bound
        ("Range Bound", "Iron Condor", "Preferred", "Profitable in high-probability mean-reverting market.", 40.0, 59.0),
        ("Range Bound", "Calendar", "Preferred", "Captures theta decay under compressing volatility.", 40.0, 59.0),
        ("Range Bound", "Double Calendar", "Preferred", "Protects range extremes with long vega and theta.", 40.0, 59.0),
        ("Range Bound", "Butterfly", "Preferred", "High reward-to-risk ratio inside tight consolidation zone.", 40.0, 59.0),
        ("Range Bound", "Large directional debit trades", "Avoid", "Directional trades struggle under range consolidation and decay.", 40.0, 59.0),
        ("Range Bound", "Overpaying for momentum", "Avoid", "Exhaustion risk is high near support/resistance extremes.", 40.0, 59.0),
        
        # Sell The Rip
        ("Sell The Rip", "Put Diagonal", "Preferred", "Captures downside bias while collecting near-term premium.", 20.0, 39.0),
        ("Sell The Rip", "Put Debit Spread", "Preferred", "Capitalizes on rapid directional downward moves.", 20.0, 39.0),
        ("Sell The Rip", "Bear Call Spread", "Preferred", "Credits credit spreads at descending resistance/moving averages.", 20.0, 39.0),
        ("Sell The Rip", "ATM Put Calendar", "Avoid", "Avoid unless volatility term structure is strictly normal.", 20.0, 39.0),
        ("Sell The Rip", "Bullish calendars", "Avoid", "High risk of rapid breakdown below support levels.", 20.0, 39.0),
        ("Sell The Rip", "High positive delta", "Avoid", "Counter-trend longs face heavy overhead resistance.", 20.0, 39.0),
        
        # Risk Off / Volatility Shock
        ("Risk Off / Volatility Shock", "Long Put", "Preferred", "Profitable in sharp risk-off expansions.", 0.0, 19.0),
        ("Risk Off / Volatility Shock", "Debit Spread", "Preferred", "Protects against extreme volatility spikes.", 0.0, 19.0),
        ("Risk Off / Volatility Shock", "Tail Hedge", "Preferred", "Direct protection against tail events.", 0.0, 19.0),
        ("Risk Off / Volatility Shock", "Cash", "Preferred", "Capital preservation is the absolute priority.", 0.0, 19.0),
        ("Risk Off / Volatility Shock", "Calendars", "Avoid", "Highly risky due to term structure inversion and volatility spikes.", 0.0, 19.0),
        ("Risk Off / Volatility Shock", "Short premium without hedge", "Avoid", "Uncapped losses in volatility expansion are prohibited.", 0.0, 19.0),
        ("Risk Off / Volatility Shock", "High positive delta", "Avoid", "Severe losses in index liquidations.", 0.0, 19.0),
        ("Risk Off / Volatility Shock", "Large theta positions with uncontrolled gamma", "Avoid", "Gamma risk is extremely dangerous in volatile regimes.", 0.0, 19.0),
    ]
    
    cursor.executemany("""
    INSERT INTO market_regime_strategy_rules (regime_name, strategy_name, strategy_status, reason, min_score, max_score)
    VALUES (?, ?, ?, ?, ?, ?)
    """, rules)
    
    conn.commit()
    conn.close()
    logger.info("Market strategy rules populated.")

def save_daily_regime(record: dict):
    """Upsert daily regime calculations."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO market_regime_daily (
        regime_date, spy_close, qqq_close, iwm_close, dia_close, smh_close, vix_close,
        trend_score, breadth_score, volatility_score, momentum_score, risk_sentiment_score,
        final_regime_score, regime_name, confidence_score, market_bias
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record.get("regime_date"), record.get("spy_close"), record.get("qqq_close"),
        record.get("iwm_close"), record.get("dia_close"), record.get("smh_close"), record.get("vix_close"),
        record.get("trend_score"), record.get("breadth_score"), record.get("volatility_score"),
        record.get("momentum_score"), record.get("risk_sentiment_score"), record.get("final_regime_score"),
        record.get("regime_name"), record.get("confidence_score"), record.get("market_bias")
    ))
    conn.commit()
    conn.close()

def save_regime_history(regime_date: str, previous_regime: str, current_regime: str, trigger_reason: str):
    """Save history logs. Flags changes."""
    regime_change = 1 if previous_regime != current_regime else 0
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO market_regime_history (
        regime_date, previous_regime, current_regime, regime_change_flag, trigger_reason
    ) VALUES (?, ?, ?, ?, ?)
    """, (regime_date, previous_regime, current_regime, regime_change, trigger_reason))
    conn.commit()
    conn.close()

def get_latest_regime() -> dict:
    """Retrieve the newest daily regime record."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM market_regime_daily 
    ORDER BY regime_date DESC LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return {}

def get_historical_regimes(limit: int = 60) -> list:
    """Retrieve a chronological list of recent daily regime calculations."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM market_regime_daily 
    ORDER BY regime_date ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    # Limit at Python level or query
    return [dict(r) for r in rows][-limit:]

def get_regime_history_logs(limit: int = 30) -> list:
    """Retrieve recent history transition logs."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM market_regime_history 
    ORDER BY regime_date DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def load_strategy_rules(regime_name: str) -> list:
    """Load the strategy guidance and status mapping for a specific regime."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
    SELECT strategy_name, strategy_status, reason 
    FROM market_regime_strategy_rules 
    WHERE regime_name = ?
    """, (regime_name,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
