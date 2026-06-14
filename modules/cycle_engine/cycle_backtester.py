import numpy as np
import pandas as pd
from datetime import datetime
from modules.cycle_engine.cycle_phase_engine import calculate_cycle_phase

def run_historical_backtest(df: pd.DataFrame, dominant_cycle_days: float, scan_periods: int = 60) -> list[dict]:
    """
    Backtests historical cycle signals generated on previous dates.
    For each historical date, it fits the cycle and measures forward return,
    MFE, and MAE across 5, 10, 20, 40, and 60-day horizons.
    """
    results = []
    n = len(df)
    if n < 40:
        return []

    # Horizons to evaluate
    horizons = [5, 10, 20, 40, 60]
    
    # Run backtest scans at intervals to avoid heavy overlap and maintain speed
    # We scan the last 'scan_periods' days, evaluating at steps of 5 trading days
    step = 5
    start_idx = max(30, n - scan_periods - 60) # Ensure we have past data for fitting
    
    for idx in range(start_idx, n - 5, step):
        signal_date_str = df.index[idx].strftime("%Y-%m-%d") if isinstance(df.index[idx], datetime) else str(df.index[idx])
        sub_df = df.iloc[:idx + 1] # Historical data up to that point
        close_entry = df["Close"].iloc[idx]
        
        # 1. Fit cycle phase as of that historical date
        try:
            from modules.cycle_engine.dominant_cycle_detector import detrend_series
            # Use past 125 days for fitting
            fit_prices = sub_df["Close"].iloc[-min(125, len(sub_df)):]
            phase_info = calculate_cycle_phase(fit_prices, dominant_cycle_days)
            direction = phase_info["cycle_direction"]
            phase_label = phase_info["phase_label"]
            phase_pct = phase_info["cycle_phase_pct"]
        except Exception:
            continue
            
        # 2. Track forward performance for each horizon
        for h in horizons:
            if idx + h >= n:
                continue
                
            forward_prices = df["Close"].iloc[idx + 1 : idx + h + 1]
            close_exit = df["Close"].iloc[idx + h]
            
            fwd_return = ((close_exit - close_entry) / close_entry) * 100.0
            
            # Max Favorable / Adverse Excursions
            max_p = forward_prices.max()
            min_p = forward_prices.min()
            
            mfe = ((max_p - close_entry) / close_entry) * 100.0
            mae = ((min_p - close_entry) / close_entry) * 100.0
            
            # Win definition: Return is positive in rising cycle, negative in falling cycle
            if direction == "rising":
                win_flag = 1 if fwd_return > 0 else 0
                pnl_estimate = fwd_return
                expected_direction = "Rising"
            else:
                win_flag = 1 if fwd_return < 0 else 0
                pnl_estimate = -fwd_return
                expected_direction = "Falling"
                
            results.append({
                "signal_date": signal_date_str,
                "forecast_horizon_days": h,
                "expected_direction": expected_direction,
                "actual_return": round(fwd_return, 2),
                "max_favorable_excursion": round(mfe, 2),
                "max_adverse_excursion": round(mae, 2),
                "win_flag": win_flag,
                "pnl_estimate": round(pnl_estimate, 2),
                "phase_pct": phase_pct,
                "phase_label": phase_label
            })
            
    return results

def compute_backtest_summary(results: list[dict]) -> dict:
    """Computes aggregate win rate, average returns, and excursion statistics from backtest runs."""
    if not results:
        return {
            "win_rate": 0.0,
            "avg_return": 0.0,
            "avg_drawdown": 0.0,
            "best_phase": "N/A",
            "worst_phase": "N/A",
            "horizon_stats": {}
        }
        
    df = pd.DataFrame(results)
    
    # Overall statistics
    win_rate = float(df["win_flag"].mean() * 100.0)
    avg_return = float(df["actual_return"].mean())
    avg_drawdown = float(df["max_adverse_excursion"].mean())
    
    # Best and worst phases based on win rate
    phase_stats = df.groupby("phase_label")["win_flag"].agg(["mean", "count"])
    phase_stats = phase_stats[phase_stats["count"] >= 2] # Filter noise
    
    if not phase_stats.empty:
        best_phase = phase_stats["mean"].idxmax()
        worst_phase = phase_stats["mean"].idxmin()
    else:
        best_phase = "N/A"
        worst_phase = "N/A"
        
    # Stats by horizon
    horizon_stats = {}
    for h in df["forecast_horizon_days"].unique():
        h_df = df[df["forecast_horizon_days"] == h]
        horizon_stats[int(h)] = {
            "win_rate": round(float(h_df["win_flag"].mean() * 100.0), 1),
            "avg_return": round(float(h_df["actual_return"].mean()), 2),
            "avg_mae": round(float(h_df["max_adverse_excursion"].mean()), 2),
            "avg_mfe": round(float(h_df["max_favorable_excursion"].mean()), 2),
            "count": len(h_df)
        }
        
    return {
        "win_rate": round(win_rate, 1),
        "avg_return": round(avg_return, 2),
        "avg_drawdown": round(avg_drawdown, 2),
        "best_phase": best_phase,
        "worst_phase": worst_phase,
        "horizon_stats": horizon_stats
    }
