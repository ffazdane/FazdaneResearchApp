import os
import sys
import sqlite3
import logging
import json
from datetime import datetime, date
import pandas as pd
import yfinance as yf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.universe_manager import load_universes
from utils.persistence import get_db_path
from modules.trade_recommendation.indicators import run_indicators_scan
from modules.calendar_scoring.technical_indicators import calculate_rsi, calculate_adx
from modules.tier4.gamma_flip.data_loader import get_available_expirations_tastytrade, get_available_expirations, load_option_chain_tastytrade, load_option_chain, tastytrade_configured

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'master_analysis.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("MasterAnalysis")

def initialize_database():
    db_path = get_db_path("master_ticker_analysis")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS master_analysis (
                ticker TEXT PRIMARY KEY,
                last_processed TEXT,
                price REAL,
                fdts_signal TEXT,
                cloud_signal TEXT,
                macd_signal TEXT,
                wpr_signal TEXT,
                darvas_signal TEXT,
                has_weeklys BOOLEAN,
                above_vwap BOOLEAN,
                call_spread REAL,
                put_spread REAL,
                options_source TEXT,
                next_earnings_date TEXT
            );

            CREATE TABLE IF NOT EXISTS technical_indicators (
                ticker TEXT PRIMARY KEY,
                ma20 REAL,
                ma50 REAL,
                ma200 REAL,
                rsi14 REAL,
                adx14 REAL,
                macd_value REAL,
                macd_hist REAL,
                atr14 REAL,
                bollinger_upper REAL,
                bollinger_lower REAL,
                darvas_upper REAL,
                darvas_lower REAL,
                regression_upper REAL,
                regression_lower REAL,
                vwap REAL,
                cvd REAL,
                ichimoku_span_a REAL,
                ichimoku_span_b REAL,
                FOREIGN KEY(ticker) REFERENCES master_analysis(ticker)
            );
        """)
    return db_path

def get_all_tickers():
    universes = load_universes()
    tickers = set()
    for name, data in universes.items():
        if name.startswith("__"):
            continue
        for t in data.get("tickers", []):
            tickers.add(t)
    return sorted(list(tickers))

def fetch_price_history(ticker, db_path_liq):
    """Fetch from options_liquidity DB, fallback to yfinance."""
    try:
        with sqlite3.connect(db_path_liq) as conn:
            df = pd.read_sql_query("SELECT date, open, high, low, close, volume FROM daily_prices WHERE symbol = ? ORDER BY date", conn, params=(ticker,))
        if len(df) > 50:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            return df
    except Exception as e:
        logger.warning(f"Failed to read from DB for {ticker}: {e}")
        
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if not df.empty:
            return df
    except Exception as e:
        logger.error(f"yfinance failed for {ticker}: {e}")
        
    return pd.DataFrame()

def calculate_additional_indicators(df):
    results = {}
    if len(df) < 20:
        return results
        
    # RSI
    try:
        rsi = calculate_rsi(df['Close'], 14).iloc[-1]
        results['rsi14'] = float(rsi)
    except:
        results['rsi14'] = None
        
    # ADX
    try:
        adx, _, _ = calculate_adx(df['High'], df['Low'], df['Close'], 14)
        results['adx14'] = float(adx.iloc[-1])
    except:
        results['adx14'] = None

    # Bollinger Bands
    try:
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        results['bollinger_upper'] = float((sma20 + 2*std20).iloc[-1])
        results['bollinger_lower'] = float((sma20 - 2*std20).iloc[-1])
    except:
        results['bollinger_upper'] = None
        results['bollinger_lower'] = None

    # CVD (proxy)
    try:
        delta = df['Close'].diff()
        cvd_series = (pd.Series([1 if d >= 0 else -1 for d in delta], index=df.index) * df['Volume']).cumsum()
        results['cvd'] = float(cvd_series.iloc[-1])
    except:
        results['cvd'] = None
        
    return results

def is_third_friday(d: date) -> bool:
    return d.weekday() == 4 and 15 <= d.day <= 21

def fetch_earnings_date(ticker):
    try:
        from utils.persistence import get_db_path
        from utils.earnings_calendar_store import DB_PATH as ec_db_path
        today_str = date.today().strftime("%Y-%m-%d")
        
        cs_db = get_db_path("calendar_scoring")
        if cs_db.exists():
            with sqlite3.connect(cs_db) as conn:
                row = conn.execute("SELECT earnings_date FROM ticker_decision_log WHERE ticker = ? AND earnings_date >= ? ORDER BY decision_id DESC LIMIT 1", (ticker, today_str)).fetchone()
                if row and row[0]: return row[0]
                
        if ec_db_path.exists():
            with sqlite3.connect(ec_db_path) as conn:
                row = conn.execute("SELECT MIN(date) FROM ec_earnings_events WHERE ticker = ? AND date >= ?", (ticker, today_str)).fetchone()
                if row and row[0]: return row[0]
    except:
        pass
        
    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        if calendar is not None and not calendar.empty:
            return calendar.iloc[0, 0].strftime("%Y-%m-%d")
    except:
        pass
    return None

def check_options(ticker):
    """Check weeklys and bid/ask spreads."""
    has_weeklys = False
    call_spread = None
    put_spread = None
    source_used = "none"
    
    use_tasty = tastytrade_configured()
    try:
        if use_tasty:
            expirations = get_available_expirations_tastytrade(ticker)
            source_used = "tastytrade"
        else:
            expirations = get_available_expirations(ticker)
            source_used = "yfinance"
            
        if not expirations and use_tasty:
            # Fallback
            expirations = get_available_expirations(ticker)
            source_used = "yfinance (fallback)"
            
        if expirations:
            # Check for weeklys (any non-third-Friday)
            for exp in expirations:
                dt = datetime.strptime(exp, "%Y-%m-%d").date()
                if not is_third_friday(dt):
                    has_weeklys = True
                    break
                    
            # Get chain for nearest expiration
            nearest_exp = expirations[0]
            if "tastytrade" in source_used:
                chain_result = load_option_chain_tastytrade(ticker, (nearest_exp,))
            else:
                chain_result = load_option_chain(ticker, (nearest_exp,))
                
            chain = chain_result.chain
            if not chain.empty and chain_result.spot_price:
                # Find ATM Call and Put
                spot = chain_result.spot_price
                chain['dist'] = (chain['strike'] - spot).abs()
                
                calls = chain[chain['option_type'] == 'call']
                if not calls.empty:
                    atm_call = calls.loc[calls['dist'].idxmin()]
                    if atm_call['ask'] > 0 and atm_call['bid'] > 0:
                        call_spread = float(atm_call['ask'] - atm_call['bid'])
                        
                puts = chain[chain['option_type'] == 'put']
                if not puts.empty:
                    atm_put = puts.loc[puts['dist'].idxmin()]
                    if atm_put['ask'] > 0 and atm_put['bid'] > 0:
                        put_spread = float(atm_put['ask'] - atm_put['bid'])
                        
    except Exception as e:
        logger.warning(f"Options check failed for {ticker}: {e}")
        
    return has_weeklys, call_spread, put_spread, source_used

def write_progress(status: dict):
    prog_file = PROJECT_ROOT / "data" / "master_analysis_progress.json"
    with open(prog_file, "w") as f:
        json.dump(status, f)

def main():
    logger.info("Starting Master Ticker Analysis Process...")
    db_path = initialize_database()
    db_path_liq = get_db_path("options_liquidity")
    
    tickers = get_all_tickers()
    total = len(tickers)
    logger.info(f"Total unique tickers to process: {total}")
    
    write_progress({"status": "running", "total": total, "current": 0, "ticker": "Starting..."})
    
    results_master = []
    results_tech = []
    
    for i, ticker in enumerate(tickers):
        logger.info(f"Processing {ticker} ({i+1}/{total})")
        write_progress({"status": "running", "total": total, "current": i, "ticker": ticker})
        
        df = fetch_price_history(ticker, db_path_liq)
        if df.empty:
            logger.warning(f"Skipping {ticker} - No price history.")
            continue
            
        # 1. Technical Indicators
        scan_results = run_indicators_scan(df, "daily")
        add_results = calculate_additional_indicators(df)
        
        if not scan_results:
            continue
            
        # 2. Options Data
        has_weeklys, call_spread, put_spread, source = check_options(ticker)
        
        # 3. Earnings Data
        earnings_date = fetch_earnings_date(ticker)
        
        price = scan_results.get("price", 0.0)
        vwap = scan_results.get("vwap", 0.0)
        above_vwap = bool(price > vwap)
        
        now_str = datetime.now().isoformat()
        
        results_master.append((
            ticker, now_str, price,
            scan_results.get("fdts_signal"),
            scan_results.get("cloud_signal"),
            scan_results.get("macd_signal"),
            scan_results.get("wpr_signal"),
            scan_results.get("darvas_signal"),
            has_weeklys, above_vwap,
            call_spread, put_spread, source, earnings_date
        ))
        
        results_tech.append((
            ticker,
            scan_results.get("ma20"), scan_results.get("ma50"), scan_results.get("ma200"),
            add_results.get("rsi14"), add_results.get("adx14"),
            scan_results.get("macd_value"), scan_results.get("macd_hist"),
            scan_results.get("atr14"),
            add_results.get("bollinger_upper"), add_results.get("bollinger_lower"),
            scan_results.get("darvas_upper"), scan_results.get("darvas_lower"),
            scan_results.get("regression_upper"), scan_results.get("regression_lower"),
            vwap, add_results.get("cvd"),
            scan_results.get("ichimoku_span_a"), scan_results.get("ichimoku_span_b")
        ))
        
        # Batch insert every 20 tickers
        if len(results_master) >= 20 or i == total - 1:
            with sqlite3.connect(db_path) as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO master_analysis 
                    (ticker, last_processed, price, fdts_signal, cloud_signal, macd_signal, wpr_signal, darvas_signal, has_weeklys, above_vwap, call_spread, put_spread, options_source, next_earnings_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, results_master)
                
                conn.executemany("""
                    INSERT OR REPLACE INTO technical_indicators
                    (ticker, ma20, ma50, ma200, rsi14, adx14, macd_value, macd_hist, atr14, bollinger_upper, bollinger_lower, darvas_upper, darvas_lower, regression_upper, regression_lower, vwap, cvd, ichimoku_span_a, ichimoku_span_b)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, results_tech)
                conn.commit()
            results_master.clear()
            results_tech.clear()

    write_progress({"status": "completed", "total": total, "current": total, "ticker": "Done"})
    logger.info("Master Ticker Analysis Process Completed Successfully!")

if __name__ == "__main__":
    main()
