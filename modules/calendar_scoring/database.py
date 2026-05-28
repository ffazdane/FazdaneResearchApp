import sqlite3
import logging
from pathlib import Path
from utils.persistence import get_db_path
from modules.calendar_scoring.config import DEFAULT_WEIGHTS, MODEL_VERSION

logger = logging.getLogger("CalendarScoringDB")

def get_connection():
    db_path = get_db_path("calendar_scoring")
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path, timeout=10)

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table 1: ticker_decision_log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ticker_decision_log (
        decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_datetime TEXT NOT NULL,
        decision_date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        strategy_type TEXT,
        recommendation TEXT,
        rank_today INTEGER,
        final_score REAL,

        market_regime TEXT,
        fdts_signal TEXT,
        fdts_score REAL,

        trend_score REAL,
        option_structure_score REAL,
        volatility_score REAL,
        fdts_score_val REAL, -- Renamed to avoid syntax issue/alias duplicate if needed, but schema says fdts_score REAL. We will map fdts_score REAL.
        pca_score REAL,
        cluster_score REAL,
        leading_lagging_score REAL,
        liquidity_score REAL,
        event_risk_score REAL,
        institutional_flow_score REAL,

        cluster_label TEXT,
        leading_lagging_state TEXT,

        price_at_decision REAL,
        atr_14 REAL,
        rsi_14 REAL,
        adx_14 REAL,
        ema_20 REAL,
        ema_50 REAL,
        ema_200 REAL,

        iv_rank REAL,
        iv_percentile REAL,
        front_iv REAL,
        back_iv REAL,
        iv_term_structure REAL,

        avg_option_volume REAL,
        avg_open_interest REAL,
        bid_ask_spread_pct REAL,

        earnings_date TEXT,
        event_risk_flag INTEGER,

        reason_summary TEXT,
        model_version TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Table 2: option_trade_setup_log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS option_trade_setup_log (
        setup_id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id INTEGER,
        ticker TEXT NOT NULL,

        strategy_type TEXT,
        short_dte INTEGER,
        long_dte INTEGER,
        target_delta REAL,

        short_expiry TEXT,
        long_expiry TEXT,
        selected_strike REAL,

        short_bid REAL,
        short_ask REAL,
        short_mid REAL,

        long_bid REAL,
        long_ask REAL,
        long_mid REAL,

        net_debit REAL,
        max_risk REAL,

        setup_delta REAL,
        setup_gamma REAL,
        setup_theta REAL,
        setup_vega REAL,

        breakeven_low REAL,
        breakeven_high REAL,

        created_at TEXT DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY(decision_id) REFERENCES ticker_decision_log(decision_id)
    );
    """)

    # Table 3: decision_outcome_log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS decision_outcome_log (
        outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id INTEGER,
        ticker TEXT NOT NULL,

        review_date TEXT,
        review_day INTEGER,

        price_at_review REAL,
        option_value_at_review REAL,

        pnl_amount REAL,
        pnl_pct REAL,

        max_profit_pct REAL,
        max_drawdown_pct REAL,

        result_label TEXT,
        exit_signal TEXT,
        notes TEXT,

        created_at TEXT DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY(decision_id) REFERENCES ticker_decision_log(decision_id)
    );
    """)

    # Table 4: model_weight_config
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_weight_config (
        config_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_version TEXT NOT NULL,

        trend_weight REAL,
        option_structure_weight REAL,
        volatility_weight REAL,
        fdts_weight REAL,
        pca_weight REAL,
        cluster_weight REAL,
        leading_lagging_weight REAL,
        liquidity_weight REAL,
        event_risk_weight REAL,
        institutional_flow_weight REAL,

        is_active INTEGER DEFAULT 1,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Check if we have active weights configured, if not write the default weights
    cursor.execute("SELECT COUNT(*) FROM model_weight_config WHERE is_active = 1")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
        INSERT INTO model_weight_config (
            model_version, trend_weight, option_structure_weight, volatility_weight,
            fdts_weight, pca_weight, cluster_weight, leading_lagging_weight,
            liquidity_weight, event_risk_weight, institutional_flow_weight, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            MODEL_VERSION,
            DEFAULT_WEIGHTS["trend_weight"],
            DEFAULT_WEIGHTS["option_structure_weight"],
            DEFAULT_WEIGHTS["volatility_weight"],
            DEFAULT_WEIGHTS["fdts_weight"],
            DEFAULT_WEIGHTS["pca_weight"],
            DEFAULT_WEIGHTS["cluster_weight"],
            DEFAULT_WEIGHTS["leading_lagging_weight"],
            DEFAULT_WEIGHTS["liquidity_weight"],
            DEFAULT_WEIGHTS["event_risk_weight"],
            DEFAULT_WEIGHTS["institutional_flow_weight"]
        ))
        
    conn.commit()
    conn.close()
    logger.info("Calendar scoring database tables verified/created.")

def get_active_model_weights() -> dict:
    create_tables()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT trend_weight, option_structure_weight, volatility_weight, fdts_weight,
           pca_weight, cluster_weight, leading_lagging_weight, liquidity_weight,
           event_risk_weight, institutional_flow_weight, model_version
    FROM model_weight_config
    WHERE is_active = 1
    ORDER BY config_id DESC LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "trend_weight": row[0],
            "option_structure_weight": row[1],
            "volatility_weight": row[2],
            "fdts_weight": row[3],
            "pca_weight": row[4],
            "cluster_weight": row[5],
            "leading_lagging_weight": row[6],
            "liquidity_weight": row[7],
            "event_risk_weight": row[8],
            "institutional_flow_weight": row[9],
            "model_version": row[10]
        }
    return DEFAULT_WEIGHTS

def save_model_weights(weights: dict, version: str = MODEL_VERSION):
    create_tables()
    conn = get_connection()
    cursor = conn.cursor()
    
    # Deactivate current active weights
    cursor.execute("UPDATE model_weight_config SET is_active = 0 WHERE is_active = 1")
    
    # Insert new active weights
    cursor.execute("""
    INSERT INTO model_weight_config (
        model_version, trend_weight, option_structure_weight, volatility_weight,
        fdts_weight, pca_weight, cluster_weight, leading_lagging_weight,
        liquidity_weight, event_risk_weight, institutional_flow_weight, is_active
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    """, (
        version,
        weights.get("trend_weight", 0.0),
        weights.get("option_structure_weight", 0.0),
        weights.get("volatility_weight", 0.0),
        weights.get("fdts_weight", 0.0),
        weights.get("pca_weight", 0.0),
        weights.get("cluster_weight", 0.0),
        weights.get("leading_lagging_weight", 0.0),
        weights.get("liquidity_weight", 0.0),
        weights.get("event_risk_weight", 0.0),
        weights.get("institutional_flow_weight", 0.0)
    ))
    conn.commit()
    conn.close()
    logger.info(f"Model weights saved for version {version}")

def insert_decision_log(data: dict) -> int:
    create_tables()
    conn = get_connection()
    cursor = conn.cursor()
    
    columns = [
        "decision_datetime", "decision_date", "ticker", "strategy_type", "recommendation",
        "rank_today", "final_score", "market_regime", "fdts_signal", "fdts_score",
        "trend_score", "option_structure_score", "volatility_score", "pca_score",
        "cluster_score", "leading_lagging_score", "liquidity_score", "event_risk_score",
        "institutional_flow_score", "cluster_label", "leading_lagging_state", "price_at_decision",
        "atr_14", "rsi_14", "adx_14", "ema_20", "ema_50", "ema_200", "iv_rank",
        "iv_percentile", "front_iv", "back_iv", "iv_term_structure", "avg_option_volume",
        "avg_open_interest", "bid_ask_spread_pct", "earnings_date", "event_risk_flag",
        "reason_summary", "model_version"
    ]
    
    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT INTO ticker_decision_log ({', '.join(columns)}) VALUES ({placeholders})"
    
    values = [data.get(col) for col in columns]
    cursor.execute(sql, values)
    decision_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return decision_id

def insert_option_setup(data: dict) -> int:
    create_tables()
    conn = get_connection()
    cursor = conn.cursor()
    
    columns = [
        "decision_id", "ticker", "strategy_type", "short_dte", "long_dte", "target_delta",
        "short_expiry", "long_expiry", "selected_strike", "short_bid", "short_ask",
        "short_mid", "long_bid", "long_ask", "long_mid", "net_debit", "max_risk",
        "setup_delta", "setup_gamma", "setup_theta", "setup_vega", "breakeven_low", "breakeven_high"
    ]
    
    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT INTO option_trade_setup_log ({', '.join(columns)}) VALUES ({placeholders})"
    
    values = [data.get(col) for col in columns]
    cursor.execute(sql, values)
    setup_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return setup_id

def insert_outcome_log(data: dict) -> int:
    create_tables()
    conn = get_connection()
    cursor = conn.cursor()
    
    columns = [
        "decision_id", "ticker", "review_date", "review_day", "price_at_review",
        "option_value_at_review", "pnl_amount", "pnl_pct", "max_profit_pct",
        "max_drawdown_pct", "result_label", "exit_signal", "notes"
    ]
    
    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT INTO decision_outcome_log ({', '.join(columns)}) VALUES ({placeholders})"
    
    values = [data.get(col) for col in columns]
    cursor.execute(sql, values)
    outcome_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return outcome_id
