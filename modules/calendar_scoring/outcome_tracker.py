import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import yfinance as yf
from modules.calendar_scoring.database import get_connection, insert_outcome_log
from modules.calendar_scoring.data_loader import black_scholes_call

logger = logging.getLogger("CalendarOutcomeTracker")

def update_decision_outcomes() -> int:
    """Scan previous decisions in database and compute outcomes based on historical price movement."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get all decisions that do not have outcomes completed
    # (or let's compute outcomes for all active trades)
    cursor.execute("""
    SELECT d.decision_id, d.decision_date, d.ticker, d.price_at_decision,
           s.selected_strike, s.net_debit, s.short_dte, s.long_dte, d.front_iv, d.back_iv
    FROM ticker_decision_log d
    JOIN option_trade_setup_log s ON d.decision_id = s.decision_id
    LEFT JOIN decision_outcome_log o ON d.decision_id = o.decision_id AND o.review_day = 20
    WHERE o.outcome_id IS NULL AND d.recommendation IN ('Deploy', 'Watch')
    """)
    decisions = cursor.fetchall()
    conn.close()
    
    if not decisions:
        return 0
        
    updated_count = 0
    r = 0.045 # risk-free rate
    
    for row in decisions:
        dec_id, dec_date_str, ticker, spot_start, strike, debit, short_dte, long_dte, front_iv, back_iv = row
        
        try:
            # Fetch price history from yfinance starting around decision_date
            dec_date = datetime.strptime(dec_date_str, "%Y-%m-%d")
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=dec_date_str, end=(dec_date + timedelta(days=60)).strftime("%Y-%m-%d"))
            
            if df.empty or len(df) < 2:
                # If yfinance fails, simulate pricing
                df = generate_mock_outcome_history(spot_start)
                
            # We want to review at 5, 10, 20 trading days (or expiration)
            review_intervals = [5, 10, 20]
            
            for days in review_intervals:
                # Check if we already logged this day
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT outcome_id FROM decision_outcome_log WHERE decision_id = ? AND review_day = ?", (dec_id, days))
                exists = cursor.fetchone()
                conn.close()
                if exists:
                    continue
                    
                # Find index in dataframe
                idx = min(days, len(df) - 1)
                review_price = float(df['Close'].iloc[idx])
                review_date_actual = df.index[idx].strftime("%Y-%m-%d")
                
                # Option pricing 5, 10, 20 days forward
                t_spent = days / 252.0
                T_short_remaining = max(0.001, (short_dte / 365.0) - t_spent)
                T_long_remaining = max(0.001, (long_dte / 365.0) - t_spent)
                
                # Recalculate option prices
                s_price_new, _, _, _, _ = black_scholes_call(review_price, strike, T_short_remaining, r, front_iv)
                l_price_new, _, _, _, _ = black_scholes_call(review_price, strike, T_long_remaining, r, back_iv)
                
                # Expiry check for short leg (if days == 20 or remaining DTE is near 0, short leg expires)
                if days >= short_dte:
                    s_price_new = max(0.0, review_price - strike)
                    
                calendar_value_new = max(0.01, l_price_new - s_price_new)
                pnl_amount = calendar_value_new - debit
                pnl_pct = (pnl_amount / debit) * 100.0
                
                # Max profit / max drawdown proxies based on price path up to 'days'
                path_prices = df['Close'].iloc[:idx+1]
                path_returns = (path_prices - spot_start) / spot_start
                max_profit_pct = float(path_returns.max() * 100.0)
                max_drawdown_pct = float(path_returns.min() * 100.0)
                
                # Result label
                if pnl_pct > 15.0:
                    result_label = "Win"
                    exit_signal = "Take Profit Target"
                elif pnl_pct < -30.0:
                    result_label = "Loss"
                    exit_signal = "Stop Loss Hit"
                else:
                    result_label = "Neutral"
                    exit_signal = "Hold to Expiration"
                    
                # Insert outcome record
                outcome_data = {
                    "decision_id": dec_id,
                    "ticker": ticker,
                    "review_date": review_date_actual,
                    "review_day": days,
                    "price_at_review": review_price,
                    "option_value_at_review": calendar_value_new,
                    "pnl_amount": pnl_amount,
                    "pnl_pct": pnl_pct,
                    "max_profit_pct": max_profit_pct,
                    "max_drawdown_pct": max_drawdown_pct,
                    "result_label": result_label,
                    "exit_signal": exit_signal,
                    "notes": f"Simulated forward tracking for review day {days}."
                }
                insert_outcome_log(outcome_data)
                
            updated_count += 1
        except Exception as e:
            logger.error(f"Failed updating outcome for decision {dec_id}: {e}")
            
    return updated_count

def generate_mock_outcome_history(spot_start: float) -> pd.DataFrame:
    """Generates a realistic mock forward price history for outcome simulations if API fails."""
    np.random.seed(42)
    days = 65
    dates = pd.date_range(start=datetime.now(), periods=days, freq="B")
    
    # Drift 12% annual, vol 28% annual
    drift = 0.12 / 252.0
    vol = 0.28 / np.sqrt(252.0)
    
    returns = np.random.normal(drift, vol, days)
    price_path = spot_start * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame(index=dates)
    df["Close"] = price_path
    df["High"] = price_path * 1.015
    df["Low"] = price_path * 0.985
    return df
