"""
Ongoing Update Script for FazDane Research Application
======================================================
Queries Yahoo Finance for the latest pricing data since the last stored date
in SQLite for all active assets, updates pricing data and monthly option expiries.
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

# Add project root to sys.path for relative imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.persistence import get_db_path, backup_database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'update_historical_data.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("UpdateHistoricalData")

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
        
        df = pd.DataFrame({
            'date': [(datetime(1970, 1, 1) + timedelta(seconds=t)).strftime('%Y-%m-%d') for t in timestamps],
            'open': indicators.get('open', []),
            'high': indicators.get('high', []),
            'low': indicators.get('low', []),
            'close': indicators.get('close', []),
            'volume': indicators.get('volume', [])
        })
        
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
        if d.weekday() == 4:
            return d
    raise ValueError("Could not find third Friday")


def update_option_expiries_for_symbol(db_path: Path, symbol: str):
    """Re-evaluate option expiries for a symbol using all dates in SQLite."""
    with sqlite3.connect(db_path) as conn:
        df_prices = pd.read_sql_query(
            "SELECT date FROM daily_prices WHERE symbol = ? ORDER BY date", 
            conn, 
            params=(symbol,)
        )
    
    if df_prices.empty:
        return
        
    trading_dates = set(df_prices['date'].astype(str))
    
    df_temp = df_prices.copy()
    df_temp['date_parsed'] = pd.to_datetime(df_temp['date'])
    df_temp['year'] = df_temp['date_parsed'].dt.year
    df_temp['month'] = df_temp['date_parsed'].dt.month
    
    years_months = df_temp[['year', 'month']].drop_duplicates().sort_values(['year', 'month']).values.tolist()
    
    expiry_records = []
    for year, month in years_months:
        expiry_date = get_third_friday(int(year), int(month))
        
        curr_date = expiry_date
        while curr_date.strftime("%Y-%m-%d") not in trading_dates:
            curr_date -= timedelta(days=1)
            if curr_date.month != month:
                curr_date = expiry_date
                break
                
        expiry_records.append((symbol, int(year), int(month), curr_date.strftime("%Y-%m-%d")))
        
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR REPLACE INTO option_expiries (symbol, year, month, expiry_date)
            VALUES (?, ?, ?, ?)
        """, expiry_records)
        conn.commit()
    logger.info(f"Updated {len(expiry_records)} option expiries in SQLite for {symbol}.")


# ════════════════════════════════════════════════════════════
# Main Update Flow
# ════════════════════════════════════════════════════════════

def main():
    logger.info("Starting historical data update process...")
    db_path = get_db_path("options_liquidity")
    
    if not db_path.exists():
        logger.error(f"Database does not exist at {db_path}. Run ingest_and_patch.py first.")
        sys.exit(1)
        
    # Get symbols in the database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT symbol FROM assets")
        symbols = [row[0] for row in cursor.fetchall()]
        
    if not symbols:
        logger.warning("No assets defined in the database. Using defaults.")
        symbols = list(YAHOO_MAPPING.keys())
        
    for symbol in symbols:
        if symbol not in YAHOO_MAPPING:
            logger.warning(f"No Yahoo Finance mapping defined for symbol: {symbol}. Skipping.")
            continue
            
        yahoo_symbol = YAHOO_MAPPING[symbol]
        
        # Get latest date in SQLite
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(date) FROM daily_prices WHERE symbol = ?", (symbol,))
            latest_date = cursor.fetchone()[0]
            
        if not latest_date:
            logger.warning(f"No pricing data found for {symbol} in SQLite. Run ingestion first.")
            continue
            
        logger.info(f"Symbol: {symbol} | Latest stored date: {latest_date}")
        
        # Fetch from latest date to today
        start_query = latest_date
        end_query = datetime.now().strftime("%Y-%m-%d")
        
        if start_query == end_query:
            logger.info(f"Symbol {symbol} is already up to date.")
            continue
            
        df_new = fetch_yahoo_prices(yahoo_symbol, start_query, end_query)
        
        if df_new.empty:
            logger.info(f"No new pricing data retrieved for {symbol}.")
            continue
            
        # Ingest new rows
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
            inserted = cursor.rowcount
            
        logger.info(f"Successfully updated {symbol} with {inserted} new price records.")
        
        # Update option expiries for this symbol
        update_option_expiries_for_symbol(db_path, symbol)
        
    # Trigger cloud backup
    try:
        ok, msg = backup_database("options_liquidity", reason="Daily scheduled update")
        if ok:
            logger.info("Database cloud backup triggered successfully.")
        else:
            logger.warning(f"Database cloud backup warning: {msg}")
    except Exception as e:
        logger.warning(f"Cloud backup failed: {e}")
        
    logger.info("Historical data update process successfully completed!")

if __name__ == "__main__":
    main()
