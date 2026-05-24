"""Database engine for Option Search Module."""

from datetime import datetime
import json
import logging
import os
import sqlite3
import pandas as pd

logger = logging.getLogger("OptionSearchDB")
DB_PATH = "data/option_search.db"


def init_option_search_db():
    """Create database folder and all tables if missing."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table 1: option_search_runs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS option_search_runs (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_timestamp TEXT NOT NULL,
        filters_json TEXT,
        scanned_ticker_count INTEGER,
        qualified_ticker_count INTEGER,
        qualified_contract_count INTEGER,
        notes TEXT
    );
    """)

    # Table 2: option_ticker_universe
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS option_ticker_universe (
        ticker TEXT PRIMARY KEY,
        first_seen_date TEXT,
        last_seen_date TEXT,
        last_price REAL,
        last_option_score REAL,
        last_liquidity_score REAL,
        last_spread_score REAL,
        last_open_interest_score REAL,
        last_spread_pct REAL,
        last_total_volume REAL,
        last_open_interest REAL,
        last_call_pct REAL,
        last_put_pct REAL,
        last_bias TEXT,
        weekly_listed_flag TEXT,
        qualified_count INTEGER DEFAULT 1,
        active_flag INTEGER DEFAULT 1,
        notes TEXT
    );
    """)

    # Table 3: option_ticker_score_history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS option_ticker_score_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        run_timestamp TEXT,
        ticker TEXT,
        underlying_price REAL,
        qualified_contracts INTEGER,
        total_option_volume REAL,
        max_contract_volume REAL,
        total_open_interest REAL,
        median_spread REAL,
        median_spread_pct REAL,
        call_volume REAL,
        put_volume REAL,
        call_pct REAL,
        put_pct REAL,
        call_put_bias TEXT,
        top_contract TEXT,
        top_strike REAL,
        top_option_type TEXT,
        top_expiration TEXT,
        top_dte INTEGER,
        weekly_listed_flag TEXT,
        liquidity_score REAL,
        spread_score REAL,
        open_interest_score REAL,
        weekly_score REAL,
        call_put_signal_score REAL,
        option_trade_score REAL,
        warning_flags TEXT,
        FOREIGN KEY(run_id) REFERENCES option_search_runs(run_id)
    );
    """)

    # Table 4: option_contract_history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS option_contract_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        run_timestamp TEXT,
        ticker TEXT,
        underlying_price REAL,
        expiration TEXT,
        dte INTEGER,
        expiration_bucket TEXT,
        option_type TEXT,
        strike REAL,
        bid REAL,
        ask REAL,
        mid_price REAL,
        spread REAL,
        spread_pct REAL,
        spread_quality_label TEXT,
        volume REAL,
        open_interest REAL,
        implied_volatility REAL,
        delta REAL,
        gamma REAL,
        theta REAL,
        vega REAL,
        source TEXT,
        weekly_candidate INTEGER,
        FOREIGN KEY(run_id) REFERENCES option_search_runs(run_id)
    );
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")


def create_search_run(filters_dict, scanned_count, qualified_ticker_count, qualified_contract_count, notes=None):
    """Insert one row into option_search_runs. Return run_id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    filters_json = json.dumps(filters_dict)

    cursor.execute("""
    INSERT INTO option_search_runs (
        run_timestamp, filters_json, scanned_ticker_count,
        qualified_ticker_count, qualified_contract_count, notes
    ) VALUES (?, ?, ?, ?, ?, ?);
    """, (
        run_timestamp, filters_json, scanned_count,
        qualified_ticker_count, qualified_contract_count, notes
    ))

    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id


def save_ticker_summary(run_id, summary_df):
    """Save ticker summary to option_ticker_score_history and upsert to option_ticker_universe."""
    if summary_df.empty:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get run timestamp
    cursor.execute("SELECT run_timestamp FROM option_search_runs WHERE run_id = ?;", (run_id,))
    row = cursor.fetchone()
    run_timestamp = row[0] if row else datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for _, row_data in summary_df.iterrows():
        ticker = str(row_data["Ticker"]).upper()
        underlying_price = float(row_data["Underlying Price"])
        qualified_contracts = int(row_data["Qualified Contracts"])
        total_option_volume = float(row_data["Total Option Volume"]) if pd.notna(row_data["Total Option Volume"]) else 0.0
        max_contract_volume = float(row_data["Max Contract Volume"]) if pd.notna(row_data["Max Contract Volume"]) else 0.0
        total_open_interest = float(row_data["Total Open Interest"]) if pd.notna(row_data["Total Open Interest"]) else 0.0
        median_spread = float(row_data["Median Spread $"]) if pd.notna(row_data["Median Spread $"]) else 0.0
        median_spread_pct = float(row_data["Median Spread %"]) if pd.notna(row_data["Median Spread %"]) else 0.0
        call_volume = float(row_data["Call Volume"]) if pd.notna(row_data["Call Volume"]) else 0.0
        put_volume = float(row_data["Put Volume"]) if pd.notna(row_data["Put Volume"]) else 0.0
        call_pct = float(row_data["Call %"]) if pd.notna(row_data["Call %"]) else 0.0
        put_pct = float(row_data["Put %"]) if pd.notna(row_data["Put %"]) else 0.0
        call_put_bias = str(row_data["Call/Put Bias"])
        top_contract = str(row_data["Top Contract"])
        top_strike = float(row_data["Top Strike"]) if pd.notna(row_data["Top Strike"]) else 0.0
        top_option_type = str(row_data["Top Option Type"])
        top_expiration = str(row_data["Top Expiration"])
        top_dte = int(row_data["Top DTE"]) if pd.notna(row_data["Top DTE"]) else 0
        weekly_listed_flag = str(row_data["Weekly Listed"])
        liquidity_score = float(row_data["Liquidity Score"])
        spread_score = float(row_data["Spread Score"])
        open_interest_score = float(row_data["Open Interest Score"])
        weekly_score = float(row_data["Weekly Score"])
        call_put_signal_score = float(row_data["Call/Put Signal Score"])
        option_trade_score = float(row_data["Option Trade Score"])
        warning_flags = str(row_data["Warning Flags"])

        # Insert Score History
        cursor.execute("""
        INSERT INTO option_ticker_score_history (
            run_id, run_timestamp, ticker, underlying_price, qualified_contracts,
            total_option_volume, max_contract_volume, total_open_interest,
            median_spread, median_spread_pct, call_volume, put_volume,
            call_pct, put_pct, call_put_bias, top_contract, top_strike,
            top_option_type, top_expiration, top_dte, weekly_listed_flag,
            liquidity_score, spread_score, open_interest_score, weekly_score,
            call_put_signal_score, option_trade_score, warning_flags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            run_id, run_timestamp, ticker, underlying_price, qualified_contracts,
            total_option_volume, max_contract_volume, total_open_interest,
            median_spread, median_spread_pct, call_volume, put_volume,
            call_pct, put_pct, call_put_bias, top_contract, top_strike,
            top_option_type, top_expiration, top_dte, weekly_listed_flag,
            liquidity_score, spread_score, open_interest_score, weekly_score,
            call_put_signal_score, option_trade_score, warning_flags
        ))

        # Upsert Ticker Universe
        cursor.execute("""
        INSERT INTO option_ticker_universe (
            ticker, first_seen_date, last_seen_date, last_price, last_option_score,
            last_liquidity_score, last_spread_score, last_open_interest_score,
            last_spread_pct, last_total_volume, last_open_interest,
            last_call_pct, last_put_pct, last_bias, weekly_listed_flag,
            qualified_count, active_flag
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1)
        ON CONFLICT(ticker) DO UPDATE SET
            last_seen_date = excluded.last_seen_date,
            last_price = excluded.last_price,
            last_option_score = excluded.last_option_score,
            last_liquidity_score = excluded.last_liquidity_score,
            last_spread_score = excluded.last_spread_score,
            last_open_interest_score = excluded.last_open_interest_score,
            last_spread_pct = excluded.last_spread_pct,
            last_total_volume = excluded.last_total_volume,
            last_open_interest = excluded.last_open_interest,
            last_call_pct = excluded.last_call_pct,
            last_put_pct = excluded.last_put_pct,
            last_bias = excluded.last_bias,
            weekly_listed_flag = excluded.weekly_listed_flag,
            qualified_count = option_ticker_universe.qualified_count + 1,
            active_flag = 1;
        """, (
            ticker, run_timestamp, run_timestamp, underlying_price, option_trade_score,
            liquidity_score, spread_score, open_interest_score,
            median_spread_pct, total_option_volume, total_open_interest,
            call_pct, put_pct, call_put_bias, weekly_listed_flag
        ))

    conn.commit()
    conn.close()
    logger.info("Saved ticker summary to DB.")


def save_contract_history(run_id, contracts_df):
    """Save every qualified contract row into option_contract_history."""
    if contracts_df.empty:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get run timestamp
    cursor.execute("SELECT run_timestamp FROM option_search_runs WHERE run_id = ?;", (run_id,))
    row = cursor.fetchone()
    run_timestamp = row[0] if row else datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for _, row_data in contracts_df.iterrows():
        ticker = str(row_data["ticker"]).upper()
        underlying_price = float(row_data["underlying_price"])
        expiration = str(row_data["expiration"])
        dte = int(row_data["DTE"])
        expiration_bucket = str(row_data["expiration_bucket"])
        option_type = str(row_data["option_type"])
        strike = float(row_data["strike"])
        bid = float(row_data["bid"])
        ask = float(row_data["ask"])
        mid_price = float(row_data["mid_price"])
        spread = float(row_data["spread"])
        spread_pct = float(row_data["spread_pct"])
        spread_quality_label = str(row_data["spread_quality_label"])
        volume = float(row_data["volume"]) if pd.notna(row_data["volume"]) else 0.0
        open_interest = float(row_data["open_interest"]) if pd.notna(row_data["open_interest"]) else 0.0
        implied_volatility = float(row_data["implied_volatility"]) if pd.notna(row_data["implied_volatility"]) else None
        delta = float(row_data["delta"]) if pd.notna(row_data["delta"]) else None
        gamma = float(row_data["gamma"]) if pd.notna(row_data["gamma"]) else None
        theta = float(row_data["theta"]) if pd.notna(row_data["theta"]) else None
        vega = float(row_data["vega"]) if pd.notna(row_data["vega"]) else None
        source = str(row_data["source"])
        weekly_candidate = int(row_data["weekly_candidate"])

        cursor.execute("""
        INSERT INTO option_contract_history (
            run_id, run_timestamp, ticker, underlying_price, expiration, dte,
            expiration_bucket, option_type, strike, bid, ask, mid_price,
            spread, spread_pct, spread_quality_label, volume, open_interest,
            implied_volatility, delta, gamma, theta, vega, source, weekly_candidate
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            run_id, run_timestamp, ticker, underlying_price, expiration, dte,
            expiration_bucket, option_type, strike, bid, ask, mid_price,
            spread, spread_pct, spread_quality_label, volume, open_interest,
            implied_volatility, delta, gamma, theta, vega, source, weekly_candidate
        ))

    conn.commit()
    conn.close()
    logger.info("Saved contract history to DB.")


def get_active_ticker_list():
    """Return active tickers from option_ticker_universe. Deduped and sorted alphabetically."""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker FROM option_ticker_universe WHERE active_flag = 1 ORDER BY ticker ASC;")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_comma_delimited_tickers():
    """Return comma-delimited ticker list."""
    tickers = get_active_ticker_list()
    return ",".join(tickers)


def deactivate_stale_tickers(days=30):
    """Set active_flag = 0 where last_seen_date is older than selected stale window."""
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Safely update using date comparison
    cursor.execute("""
    UPDATE option_ticker_universe
    SET active_flag = 0
    WHERE date(substr(last_seen_date, 1, 10)) < date('now', '-' || ? || ' days');
    """, (str(days),))
    
    conn.commit()
    conn.close()
    logger.info(f"Stale tickers cleanup completed for stale window: {days} days.")


def get_universe_summary():
    """Return dict of database statistics."""
    summary = {
        "total_active": 0,
        "total_historical": 0,
        "new_today": 0,
        "requalified_today": 0,
        "stale_count": 0,
        "highest_scoring": [],
        "most_frequent": [],
    }

    if not os.path.exists(DB_PATH):
        return summary

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Total active
        cursor.execute("SELECT COUNT(*) FROM option_ticker_universe WHERE active_flag = 1;")
        summary["total_active"] = cursor.fetchone()[0]

        # Total historical
        cursor.execute("SELECT COUNT(*) FROM option_ticker_universe;")
        summary["total_historical"] = cursor.fetchone()[0]

        # New today
        cursor.execute("SELECT COUNT(*) FROM option_ticker_universe WHERE substr(first_seen_date, 1, 10) = date('now');")
        summary["new_today"] = cursor.fetchone()[0]

        # Requalified today (seen today but first seen before today)
        cursor.execute("""
        SELECT COUNT(*) FROM option_ticker_universe 
        WHERE substr(last_seen_date, 1, 10) = date('now') 
          AND substr(first_seen_date, 1, 10) < date('now');
        """)
        summary["requalified_today"] = cursor.fetchone()[0]

        # Stale count
        cursor.execute("SELECT COUNT(*) FROM option_ticker_universe WHERE active_flag = 0;")
        summary["stale_count"] = cursor.fetchone()[0]

        # Highest scoring (active only)
        cursor.execute("""
        SELECT ticker, last_option_score, last_total_volume, last_spread_pct 
        FROM option_ticker_universe 
        WHERE active_flag = 1 
        ORDER BY last_option_score DESC 
        LIMIT 5;
        """)
        summary["highest_scoring"] = [
            {"ticker": r[0], "score": r[1], "volume": r[2], "spread": r[3]} for r in cursor.fetchall()
        ]

        # Most frequent
        cursor.execute("""
        SELECT ticker, qualified_count, last_option_score 
        FROM option_ticker_universe 
        ORDER BY qualified_count DESC 
        LIMIT 5;
        """)
        summary["most_frequent"] = [
            {"ticker": r[0], "count": r[1], "score": r[2]} for r in cursor.fetchall()
        ]
    except Exception as e:
        logger.error(f"Error querying universe summary: {e}")

    conn.close()
    return summary


def get_new_tickers_for_run(run_id):
    """Show tickers discovered for the first time in the latest run."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # We find tickers in this run whose first_seen_date matches the run timestamp date
    cursor.execute("""
    SELECT ticker, underlying_price, option_trade_score, total_option_volume, median_spread_pct
    FROM option_ticker_score_history
    WHERE run_id = ? AND ticker IN (
        SELECT ticker FROM option_ticker_universe 
        WHERE substr(first_seen_date, 1, 10) = (
            SELECT substr(run_timestamp, 1, 10) FROM option_search_runs WHERE run_id = ?
        )
    );
    """, (run_id, run_id))
    
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "Ticker": r[0],
            "Price": r[1],
            "Score": r[2],
            "Volume": r[3],
            "Spread": r[4]
        }
        for r in rows
    ]


def get_requalified_tickers_for_run(run_id):
    """Show tickers that already existed and qualified again in this run."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # We find tickers in this run whose first_seen_date is older than the run timestamp date
    cursor.execute("""
    SELECT ticker, underlying_price, option_trade_score, total_option_volume, median_spread_pct
    FROM option_ticker_score_history
    WHERE run_id = ? AND ticker IN (
        SELECT ticker FROM option_ticker_universe 
        WHERE substr(first_seen_date, 1, 10) < (
            SELECT substr(run_timestamp, 1, 10) FROM option_search_runs WHERE run_id = ?
        )
    );
    """, (run_id, run_id))
    
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "Ticker": r[0],
            "Price": r[1],
            "Score": r[2],
            "Volume": r[3],
            "Spread": r[4]
        }
        for r in rows
    ]
