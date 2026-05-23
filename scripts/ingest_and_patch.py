"""
Ingestion & Online Patching Script for FazDane Research Application
==================================================================
Reads historical index and ETF data from Access DB and VIX CSV,
queries Yahoo Finance API to patch any gaps and update data to 2026,
calculates Option Expiry dates, and saves everything to SQLite.
"""

import os
import sys
import sqlite3
import urllib.request
import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path for relative imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.persistence import get_db_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'ingest_and_patch.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("IngestAndPatch")

# Path definitions
ACCDB_PATH = Path(r"C:\Users\ffazd\OneDrive\Documents\Fazal\Historical Stock Data\Historuical Data.accdb")
SECTORS_ACCDB_PATH = Path(r"C:\Users\ffazd\OneDrive\Documents\Fazal\Historical Stock Data\SPX SECTORS.accdb")
VIX_CSV_PATH = Path(r"C:\Users\ffazd\OneDrive\Documents\Fazal\Historical Stock Data\^VIX.csv")

# Yahoo Finance symbols mapping
YAHOO_MAPPING = {
    "SPX": "^SPX",
    "NDX": "^NDX",
    "RUT": "^RUT",
    "DJI": "^DJI",
    "VIX": "^VIX",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "DIA": "DIA"
}

# ════════════════════════════════════════════════════════════
# Yahoo Finance Fetcher
# ════════════════════════════════════════════════════════════

def fetch_yahoo_prices(yahoo_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily price bars from Yahoo Finance chart API."""
    logger.info(f"Fetching {yahoo_symbol} from Yahoo Finance: {start_date} to {end_date}")
    try:
        dt1 = datetime.strptime(start_date, "%Y-%m-%d")
        dt2 = datetime.strptime(end_date, "%Y-%m-%d")
        # Manual Unix timestamp calculation to support pre-1970 dates on Windows
        epoch = datetime(1970, 1, 1)
        p1 = int((dt1 - epoch).total_seconds())
        p2 = int((dt2 - epoch).total_seconds())
    except Exception as e:
        logger.error(f"Error parsing dates: {e}")
        return pd.DataFrame()

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?period1={p1}&period2={p2}&interval=1d"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        result = data['chart']['result'][0]
        timestamps = result.get('timestamp', [])
        if not timestamps:
            logger.warning(f"No timestamps returned for {yahoo_symbol}")
            return pd.DataFrame()
            
        indicators = result['indicators']['quote'][0]
        
        # Build dataframe
        df = pd.DataFrame({
            'date': [datetime.fromtimestamp(t).strftime('%Y-%m-%d') for t in timestamps],
            'open': indicators.get('open', []),
            'high': indicators.get('high', []),
            'low': indicators.get('low', []),
            'close': indicators.get('close', []),
            'volume': indicators.get('volume', [])
        })
        
        # Drop rows with NaN close prices and reset index
        df = df.dropna(subset=['close']).reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch {yahoo_symbol} from Yahoo Finance: {e}")
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════
# Option Expiry Calculation
# ════════════════════════════════════════════════════════════

def get_third_friday(year: int, month: int) -> date:
    """Calculate the third Friday of a given month."""
    for day in range(15, 22):
        d = date(year, month, day)
        if d.weekday() == 4:  # Friday is 4
            return d
    raise ValueError("Could not find third Friday")


def calculate_option_expiries(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Option Expiry dates for all months in the prices dataset.
    Resolves to the last trading day on or before the third Friday if a holiday.
    """
    # Get set of all active trading dates
    trading_dates = set(df_prices['date'].astype(str).str.slice(0, 10))
    
    # Parse dates to get years and months represented
    df_temp = df_prices.copy()
    df_temp['date_parsed'] = pd.to_datetime(df_temp['date'])
    df_temp['year'] = df_temp['date_parsed'].dt.year
    df_temp['month'] = df_temp['date_parsed'].dt.month
    
    years_months = df_temp[['year', 'month']].drop_duplicates().sort_values(['year', 'month']).values.tolist()
    
    expiries = []
    for year, month in years_months:
        expiry_date = get_third_friday(int(year), int(month))
        
        curr_date = expiry_date
        # Go backwards until we find a valid trading day in our dataset
        while curr_date.strftime("%Y-%m-%d") not in trading_dates:
            curr_date -= timedelta(days=1)
            # Safe check to prevent runaway loop if no data exists in month
            if curr_date.month != month:
                curr_date = expiry_date  # Reset to standard
                break
                
        expiries.append({
            'year': int(year),
            'month': int(month),
            'expiry_date': curr_date.strftime("%Y-%m-%d")
        })
        
    return pd.DataFrame(expiries)


# ════════════════════════════════════════════════════════════
# Database Helpers
# ════════════════════════════════════════════════════════════

def load_accdb_table(db_path: Path, table_name: str) -> pd.DataFrame:
    """Read a table from the MS Access database."""
    logger.info(f"Loading table '{table_name}' from Access DB: {db_path}...")
    try:
        import pyodbc
        conn_str = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_path};"
        conn = pyodbc.connect(conn_str)
        df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
        conn.close()
        logger.info(f"Loaded {len(df)} rows from table '{table_name}'")
        return df
    except Exception as e:
        logger.error(f"Error loading table '{table_name}': {e}")
        return pd.DataFrame()


def init_sqlite_tables(db_path: Path):
    """Ensure SQLite tables assets, daily_prices, and option_expiries are initialized."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        # Create assets
        conn.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                asset_class TEXT,
                sector TEXT,
                industry TEXT
            )
        """)
        # Create daily_prices
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
        # Create option_expiries
        conn.execute("""
            CREATE TABLE IF NOT EXISTS option_expiries (
                symbol TEXT,
                year INTEGER,
                month INTEGER,
                expiry_date TEXT,
                PRIMARY KEY (symbol, year, month)
            )
        """)
        conn.commit()
    logger.info("SQLite database tables initialized successfully.")


def add_asset(db_path: Path, symbol: str, name: str, asset_class: str, sector: str, industry: str):
    """Add or update an asset definition in SQLite."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO assets (symbol, name, asset_class, sector, industry)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, name, asset_class, sector, industry))
        conn.commit()


def ingest_daily_prices(db_path: Path, symbol: str, df: pd.DataFrame) -> int:
    """Bulk insert daily prices into SQLite."""
    records = []
    for _, row in df.iterrows():
        records.append((
            str(row['date'])[:10],
            symbol,
            float(row['open']) if pd.notna(row['open']) else None,
            float(row['high']) if pd.notna(row['high']) else None,
            float(row['low']) if pd.notna(row['low']) else None,
            float(row['close']) if pd.notna(row['close']) else None,
            float(row['volume']) if pd.notna(row['volume']) else 0.0,
            0.0 # Open interest default
        ))
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR REPLACE INTO daily_prices (date, symbol, open, high, low, close, volume, open_interest)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
        conn.commit()
        count = cursor.rowcount
    return count


# ════════════════════════════════════════════════════════════
# Main Ingestion and Update Flow
# ════════════════════════════════════════════════════════════

def main():
    logger.info("Starting FazDane database migration, patching, and 2026 update...")
    
    # 1. Resolve SQLite DB Path
    db_path = get_db_path("options_liquidity")
    init_sqlite_tables(db_path)
    
    # Add active assets to the universe database
    assets_definitions = [
        ("SPX", "S&P 500 Index", "Indices", "Large Cap", ""),
        ("NDX", "NASDAQ 100 Index", "Indices", "Large Cap", ""),
        ("RUT", "Russell 2000 Index", "Indices", "Small Cap", ""),
        ("DJI", "Dow Jones Industrial Average Index", "Indices", "Large Cap", ""),
        ("VIX", "CBOE Volatility Index", "Indices", "Volatility", ""),
        ("SPY", "SPDR S&P 500 ETF Trust", "ETFs", "Large Cap Growth", ""),
        ("QQQ", "Invesco QQQ Trust", "ETFs", "Large Cap Growth", ""),
        ("IWM", "iShares Russell 2000 ETF", "ETFs", "Small Cap Growth", ""),
        ("DIA", "SPDR Dow Jones Industrial Average ETF Trust", "ETFs", "Large Cap Value", "")
    ]
    for symbol, name, asset_class, sector, industry in assets_definitions:
        add_asset(db_path, symbol, name, asset_class, sector, industry)
        
    # 2. Ingest and Patch Each Symbol
    for symbol, yahoo_symbol in YAHOO_MAPPING.items():
        logger.info(f"=== Processing Symbol: {symbol} ===")
        
        # Load local data first
        df_local = pd.DataFrame()
        if symbol == "VIX":
            if os.path.exists(VIX_CSV_PATH):
                logger.info(f"Loading VIX from CSV: {VIX_CSV_PATH}")
                df_local = pd.read_csv(VIX_CSV_PATH)
                df_local = df_local.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
                df_local['symbol'] = "VIX"
            else:
                logger.warning(f"VIX CSV file not found at {VIX_CSV_PATH}")
        elif symbol == "SPX":
            df_local = load_accdb_table(ACCDB_PATH, "SPX")
            if not df_local.empty:
                df_local = df_local.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        elif symbol == "NDX":
            df_local = load_accdb_table(ACCDB_PATH, "Nasdaq")
            if not df_local.empty:
                df_local = df_local.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
                # NDX (Nasdaq 100) was introduced in 1985. The "Nasdaq" table covers from 1971.
                # We filter date >= 1985-10-01 to align with NASDAQ 100.
                df_local['date_temp'] = pd.to_datetime(df_local['date'])
                df_local = df_local[df_local['date_temp'] >= '1985-10-01'].copy()
                df_local = df_local.drop(columns=['date_temp'])
        elif symbol == "RUT":
            df_local = load_accdb_table(ACCDB_PATH, "RUT")
            if not df_local.empty:
                df_local = df_local.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        elif symbol == "DJI":
            df_local = load_accdb_table(ACCDB_PATH, "DOW")
            if not df_local.empty:
                df_local = df_local.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        elif symbol == "SPY":
            # Check SPX SECTORS.accdb SPY table
            if os.path.exists(SECTORS_ACCDB_PATH):
                df_local = load_accdb_table(SECTORS_ACCDB_PATH, "SPY")
                if not df_local.empty:
                    df_local = df_local.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            
        # Clean local dataframe dates to YYYY-MM-DD
        if not df_local.empty:
            df_local['date'] = pd.to_datetime(df_local['date'], format='mixed').dt.strftime('%Y-%m-%d')
            df_local['symbol'] = symbol
            # Ensure sorting and uniqueness
            df_local = df_local.sort_values('date').drop_duplicates('date').reset_index(drop=True)
            logger.info(f"Local {symbol} data range: {df_local['date'].min()} to {df_local['date'].max()} ({len(df_local)} rows)")
        else:
            logger.warning(f"No local data found for {symbol}. Ingesting entirely from online.")
            
        # Determine start date for Yahoo Finance query
        start_query = "1928-01-01"
        if not df_local.empty:
            start_query = df_local['date'].min()
        else:
            # Fallbacks for empty local datasets (ETFs)
            if symbol == "NDX":
                start_query = "1985-10-01"
            elif symbol == "RUT":
                start_query = "1987-09-10"
            elif symbol == "VIX":
                start_query = "1990-01-01"
            elif symbol == "SPY":
                start_query = "1993-01-29"
            elif symbol == "QQQ":
                start_query = "1999-03-10"
            elif symbol == "IWM":
                start_query = "2000-05-26"
            elif symbol == "DIA":
                start_query = "1998-01-20"
            elif symbol == "DJI":
                start_query = "1914-12-12"
            elif symbol == "SPX":
                start_query = "1927-12-30"
            
        end_query = datetime.now().strftime("%Y-%m-%d")
        
        # 3. Query Yahoo Finance for the complete dataset (covers gaps and extends to 2026)
        df_online = fetch_yahoo_prices(yahoo_symbol, start_query, end_query)
        
        if not df_online.empty:
            logger.info(f"Online {symbol} data range: {df_online['date'].min()} to {df_online['date'].max()} ({len(df_online)} rows)")
            
            if not df_local.empty:
                df_local_indexed = df_local.set_index('date')
                df_online_indexed = df_online.set_index('date')
                
                # Identify missing dates in local dataset
                missing_dates = df_online_indexed.index.difference(df_local_indexed.index)
                logger.info(f"Detected {len(missing_dates)} missing dates (gaps or 2024-2026 updates) for {symbol}")
                
                # Combine: Reindex local to union and fill from online
                df_merged = df_local_indexed.combine_first(df_online_indexed).reset_index()
            else:
                df_merged = df_online.copy()
                df_merged['symbol'] = symbol
                
            df_merged = df_merged.sort_values('date').reset_index(drop=True)
            logger.info(f"Cleaned merged dataset for {symbol}: {df_merged['date'].min()} to {df_merged['date'].max()} ({len(df_merged)} rows)")
        else:
            logger.warning(f"Could not retrieve online data for {symbol}. Ingesting local data as-is.")
            df_merged = df_local.copy()
            
        if df_merged.empty:
            logger.error(f"No price data available for {symbol}")
            continue
            
        # Bulk Ingest to SQLite
        inserted = ingest_daily_prices(db_path, symbol, df_merged)
        logger.info(f"Ingested {symbol} into SQLite: {inserted} records added/updated.")
        
        # 4. Calculate and Ingest Option Expiries for this symbol
        logger.info(f"Calculating monthly option expiries for {symbol}...")
        df_expiries = calculate_option_expiries(df_merged)
        
        # Write expiries to SQLite
        logger.info(f"Writing {len(df_expiries)} option expiries to SQLite for {symbol}...")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            expiry_records = [(symbol, int(row['year']), int(row['month']), row['expiry_date']) for _, row in df_expiries.iterrows()]
            cursor.executemany("""
                INSERT OR REPLACE INTO option_expiries (symbol, year, month, expiry_date)
                VALUES (?, ?, ?, ?)
            """, expiry_records)
            conn.commit()
            
    logger.info("Database migration, patching, and 2026 updates successfully completed!")

if __name__ == "__main__":
    main()
