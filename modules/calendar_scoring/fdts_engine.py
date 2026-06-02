import logging
import numpy as np
import pandas as pd
from modules.calendar_scoring.technical_indicators import calculate_fdts_ha_signal, calculate_rsi

logger = logging.getLogger("CalendarFDTSEngine")

def calculate_fdts_signal(ticker_df: pd.DataFrame, period: int = 20) -> dict:
    """Calculate Heikin-Ashi and Triple EMA (TEMA) deviation FDTS signal and trend score."""
    try:
        # Get Heikin-Ashi + TEMA deviation signal ("Buy", "Sell", "No Trade")
        raw_sig = calculate_fdts_ha_signal(ticker_df, period)
        
        # Map "No Trade" to "Neutral" to maintain compatibility with SQLite database schema and backtesting PnL
        signal = "Neutral" if raw_sig == "No Trade" else raw_sig
        
        # Set base score matching the signal
        if signal == "Buy":
            base_score = 90.0
        elif signal == "Sell":
            base_score = 30.0
        else:
            base_score = 60.0
            
        # Get RSI technical modifier if available
        rsi_val = 50.0
        if not ticker_df.empty and len(ticker_df) >= 15:
            try:
                rsi_series = calculate_rsi(ticker_df["Close"])
                if not rsi_series.empty and not np.isnan(rsi_series.iloc[-1]):
                    rsi_val = float(rsi_series.iloc[-1])
            except Exception as e:
                logger.debug(f"Could not calculate RSI for FDTS modifier: {e}")
                
        # Slight adjustment based on RSI relative to neutral 50
        rsi_factor = (rsi_val - 50.0) * 0.1
        final_score = min(100.0, max(0.0, base_score + rsi_factor))
        
        return {
            "signal": signal,
            "score": round(final_score, 1),
            "state_description": f"Trend: {signal} (Score: {final_score:.1f})"
        }
    except Exception as e:
        logger.error(f"Error calculating FDTS signal: {e}")
        return {
            "signal": "Neutral",
            "score": 50.0,
            "state_description": "FDTS signal calculation failed."
        }
