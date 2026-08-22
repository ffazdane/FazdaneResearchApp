import sqlite3
import pandas as pd
import logging
from datetime import datetime, timedelta
import yfinance as yf
from pathlib import Path
import os
import json

from utils.persistence import get_db_path
from scripts.update_historical_data import YAHOO_MAPPING, fetch_yahoo_prices

logger = logging.getLogger("BacktestStore")

def get_historical_data(symbol: str, months: int = 24) -> pd.DataFrame:
    """
    Get at least `months` of historical data for the given symbol.
    It checks options_liquidity.sqlite first. If there's missing data up to today,
    it fetches it using yfinance, updates options_liquidity.sqlite, and returns the combined DataFrame.
    """
    db_path = get_db_path("options_liquidity")
    if not db_path.exists():
        logger.warning(f"options_liquidity.sqlite not found at {db_path}. Will rely entirely on yf.")
        df_db = pd.DataFrame()
    else:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    date TEXT,
                    symbol TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    open_interest REAL,
                    PRIMARY KEY (date, symbol)
                )
            """)
            try:
                query = "SELECT date, open, high, low, close, volume FROM daily_prices WHERE symbol = ? ORDER BY date ASC"
                df_db = pd.read_sql_query(query, conn, params=(symbol,))
            except Exception as e:
                logger.warning(f"Could not read from daily_prices table: {e}")
                df_db = pd.DataFrame()
            
    today = datetime.now().date()
    target_start = today - timedelta(days=30 * months)
    
    start_fetch_date = target_start.strftime("%Y-%m-%d")
    
    if not df_db.empty:
        df_db['date_obj'] = pd.to_datetime(df_db['date']).dt.date
        last_date = df_db['date_obj'].max()
        first_date = df_db['date_obj'].min()
        
        if last_date < today - timedelta(days=1):
            start_fetch_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            df_filtered = df_db[df_db['date_obj'] >= target_start].copy()
            df_filtered.drop(columns=['date_obj'], inplace=True)
            return df_filtered.reset_index(drop=True)
            
    yahoo_symbol = YAHOO_MAPPING.get(symbol, symbol)
    end_fetch_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    
    logger.info(f"Fetching missing data for {symbol} ({yahoo_symbol}) from {start_fetch_date} to {end_fetch_date}")
    df_new = fetch_yahoo_prices(yahoo_symbol, start_fetch_date, end_fetch_date)
    
    if not df_new.empty and db_path.exists():
        records = []
        for _, row in df_new.iterrows():
            records.append((
                str(row['date'])[:10],
                symbol,
                float(row['open']) if pd.notna(row['open']) else None,
                float(row['high']) if pd.notna(row['high']) else None,
                float(row['low']) if pd.notna(row['low']) else None,
                float(row['close']) if pd.notna(row['close']) else None,
                float(row['volume']) if pd.notna(row['volume']) else 0.0,
                0.0
            ))
            
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO daily_prices (date, symbol, open, high, low, close, volume, open_interest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()
            logger.info(f"Updated {symbol} with {cursor.rowcount} new price records in options_liquidity.sqlite.")
            
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            try:
                query = "SELECT date, open, high, low, close, volume FROM daily_prices WHERE symbol = ? ORDER BY date ASC"
                df_full = pd.read_sql_query(query, conn, params=(symbol,))
                if not df_full.empty:
                    df_full['date_obj'] = pd.to_datetime(df_full['date']).dt.date
                    df_filtered = df_full[df_full['date_obj'] >= target_start].copy()
                    df_filtered.drop(columns=['date_obj'], inplace=True)
                    return df_filtered.reset_index(drop=True)
            except Exception as e:
                logger.warning(f"Could not read from daily_prices table after insert: {e}")
                
    return df_new

def prefetch_historical_data_bulk(tickers: list[str], months: int = 24) -> None:
    """
    Prefetch historical data for a list of tickers in bulk to avoid Yahoo Finance rate limits.
    It checks which tickers are missing data in the local database and uses yfinance's
    multi-threaded downloader to fetch only what's necessary.
    """
    if not tickers:
        return
        
    db_path = get_db_path("options_liquidity")
    today = datetime.now().date()
    target_start = today - timedelta(days=30 * months)
    start_fetch_date = target_start.strftime("%Y-%m-%d")
    end_fetch_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    
    missing_tickers = []
    
    # 1. Identify missing tickers
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                date TEXT,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                open_interest REAL,
                PRIMARY KEY (date, symbol)
            )
        """)
        
        for ticker in tickers:
            try:
                query = "SELECT MAX(date) as last_date FROM daily_prices WHERE symbol = ?"
                df_db = pd.read_sql_query(query, conn, params=(ticker,))
                last_date_str = df_db.iloc[0]['last_date']
                
                if last_date_str:
                    last_date = pd.to_datetime(last_date_str).date()
                    if last_date < today - timedelta(days=1):
                        missing_tickers.append(ticker)
                else:
                    missing_tickers.append(ticker)
            except Exception:
                missing_tickers.append(ticker)
                
    if not missing_tickers:
        logger.info(f"All {len(tickers)} tickers are up-to-date in options_liquidity.sqlite.")
        return
        
    logger.info(f"Prefetching data for {len(missing_tickers)} missing tickers from {start_fetch_date} to {end_fetch_date} via yfinance bulk download...")
    
    # Map symbols to Yahoo symbols
    yahoo_symbols = [YAHOO_MAPPING.get(t, t) for t in missing_tickers]
    
    # 2. Bulk download using yfinance
    try:
        data = yf.download(yahoo_symbols, start=start_fetch_date, end=end_fetch_date, group_by='ticker', threads=True, progress=False)
    except Exception as e:
        logger.error(f"Bulk download failed: {e}")
        return
        
    if data.empty:
        logger.warning("Bulk download returned empty data.")
        return
        
    # 3. Parse and insert data
    records = []
    
    if len(yahoo_symbols) == 1:
        # Single ticker result doesn't have multi-index columns for ticker
        sym = yahoo_symbols[0]
        orig_sym = missing_tickers[0]
        df = data.dropna(subset=['Close'])
        for idx, row in df.iterrows():
            records.append((
                idx.strftime("%Y-%m-%d"),
                orig_sym,
                float(row['Open']) if pd.notna(row['Open']) else None,
                float(row['High']) if pd.notna(row['High']) else None,
                float(row['Low']) if pd.notna(row['Low']) else None,
                float(row['Close']) if pd.notna(row['Close']) else None,
                float(row['Volume']) if pd.notna(row['Volume']) else 0.0,
                0.0
            ))
    else:
        for sym, orig_sym in zip(yahoo_symbols, missing_tickers):
            if sym in data.columns.levels[0]:
                df = data[sym].dropna(subset=['Close'])
                for idx, row in df.iterrows():
                    records.append((
                        idx.strftime("%Y-%m-%d"),
                        orig_sym,
                        float(row['Open']) if pd.notna(row['Open']) else None,
                        float(row['High']) if pd.notna(row['High']) else None,
                        float(row['Low']) if pd.notna(row['Low']) else None,
                        float(row['Close']) if pd.notna(row['Close']) else None,
                        float(row['Volume']) if pd.notna(row['Volume']) else 0.0,
                        0.0
                    ))
                    
    if records:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO daily_prices (date, symbol, open, high, low, close, volume, open_interest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()
            logger.info(f"Inserted {cursor.rowcount} bulk price records into options_liquidity.sqlite.")


def init_backtest_db():
    db_path = get_db_path("backtest_engine")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                kpi_json TEXT,
                run_data_json TEXT
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_symbol ON backtest_runs(symbol)")

def save_backtest_result(symbol: str, kpis: dict, df: pd.DataFrame):
    init_backtest_db()
    db_path = get_db_path("backtest_engine")
    timestamp = datetime.now().isoformat()
    # Serialize df to json string
    import io
    # For large dfs, to_json is better
    run_data_json = df.to_json(orient='records', date_format='iso')
    kpi_json = json.dumps(kpis)
    
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM backtest_runs WHERE symbol = ?", (symbol,))
        conn.execute("""
            INSERT INTO backtest_runs (symbol, timestamp, kpi_json, run_data_json)
            VALUES (?, ?, ?, ?)
        """, (symbol, timestamp, kpi_json, run_data_json))

def get_latest_backtest_result(symbol: str) -> tuple[dict, pd.DataFrame, str]:
    init_backtest_db()
    db_path = get_db_path("backtest_engine")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, kpi_json, run_data_json
            FROM backtest_runs
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (symbol,))
        row = cursor.fetchone()
        if row:
            timestamp, kpi_json, run_data_json = row
            kpis = json.loads(kpi_json)
            import io
            df = pd.read_json(io.StringIO(run_data_json), orient='records')
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")
            return kpis, df, timestamp
    return None, None, None

def get_latest_option_spreads(tickers: list[str]) -> dict:
    """
    Fetch the latest call_spread and put_spread for the given tickers
    from the master_ticker_analysis.sqlite database.
    """
    db_path = get_db_path("master_ticker_analysis")
    results = {}
    if not db_path.exists():
        return results

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in tickers)
        query = f"""
            SELECT ticker, call_spread, put_spread 
            FROM master_analysis 
            WHERE ticker IN ({placeholders})
        """
        try:
            cursor.execute(query, tickers)
            for row in cursor.fetchall():
                ticker, call_spread, put_spread = row
                results[ticker] = {
                    "call_spread": call_spread if call_spread is not None else 0.0,
                    "put_spread": put_spread if put_spread is not None else 0.0
                }
        except Exception as e:
            logger.warning(f"Failed to fetch option spreads from master DB: {e}")
            
    return results
