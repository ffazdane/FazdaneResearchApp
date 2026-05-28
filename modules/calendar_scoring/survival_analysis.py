import logging
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("CalendarSurvivalAnalysis")

def extract_trend_durations(df: pd.DataFrame) -> list:
    """
    Extract durations of historical trends from daily price data.
    A trend starts when the price crosses above the 50-day EMA and
    ends when the price closes below the 50-day EMA for 2 consecutive days.
    Returns:
        list of dicts: [{'duration': days, 'censored': bool}]
    """
    if df.empty or len(df) < 60:
        return []
        
    close = df['Close']
    ema_50 = close.ewm(span=50, adjust=False).mean()
    
    in_trend = False
    trend_start_idx = None
    below_count = 0
    durations = []
    
    for i in range(1, len(df)):
        price = close.iloc[i]
        ema = ema_50.iloc[i]
        
        if not in_trend:
            # Check for crossover above 50 EMA
            if price > ema and close.iloc[i-1] <= ema_50.iloc[i-1]:
                in_trend = True
                trend_start_idx = i
                below_count = 0
        else:
            # We are in an active trend. Check if we close below EMA
            if price < ema:
                below_count += 1
                if below_count >= 2:
                    # Trend ended (death event)
                    duration = i - trend_start_idx
                    durations.append({"duration": duration, "censored": False})
                    in_trend = False
            else:
                below_count = 0
                
    # If trend is still active at the end of the history (censored event)
    if in_trend:
        duration = len(df) - 1 - trend_start_idx
        durations.append({"duration": duration, "censored": True})
        
    return durations

def fit_kaplan_meier(durations: list) -> dict:
    """
    Fit a Kaplan-Meier survival estimator to trend durations.
    Returns:
        dict: Plottable curve coordinate mapping {day: survival_probability}
    """
    if not durations:
        # Default fallback curve if no data
        return {d: round(0.95 ** (d / 5.0), 3) for d in range(0, 101, 5)}
        
    df_dur = pd.DataFrame(durations)
    # Sort by duration
    df_dur = df_dur.sort_values(by="duration").reset_index(drop=True)
    
    unique_durations = sorted(df_dur["duration"].unique())
    
    S = 1.0
    survival_curve = {0: 1.0}
    
    for t in unique_durations:
        # Count deaths (completed trends) at duration t
        deaths = len(df_dur[(df_dur["duration"] == t) & (df_dur["censored"] == False)])
        # Count censored at duration t
        censored = len(df_dur[(df_dur["duration"] == t) & (df_dur["censored"] == True)])
        # Total trends at risk at duration >= t
        at_risk = len(df_dur[df_dur["duration"] >= t])
        
        if at_risk > 0:
            p = 1.0 - (deaths / at_risk)
            S *= p
            survival_curve[int(t)] = round(float(S), 4)
            
    # Fill in step intervals for day 0 to 120
    complete_curve = {}
    current_s = 1.0
    for day in range(0, 121):
        if day in survival_curve:
            current_s = survival_curve[day]
        complete_curve[day] = current_s
        
    return complete_curve

def analyze_trend_survival(ticker_symbol: str) -> dict:
    """
    Perform survival analysis on the ticker's historical trend durations.
    Returns:
        dict: {
            'curve': {day: prob},
            'current_age': days,
            'survival_prob_20d': percentage,
            'warning': bool
        }
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Fetch 3 years of daily data to get enough cycles
        df = ticker.history(period="3y")
        
        if df.empty or len(df) < 100:
            raise ValueError(f"Insufficient history for {ticker_symbol}")
            
        # 1. Extract historical durations
        durations = extract_trend_durations(df)
        
        # 2. Fit Kaplan-Meier
        curve = fit_kaplan_meier(durations)
        
        # 3. Calculate current trend age
        close = df['Close']
        ema_50 = close.ewm(span=50, adjust=False).mean()
        
        current_age = 0
        if close.iloc[-1] > ema_50.iloc[-1]:
            # Count back consecutive days where price remained in trend
            below_count = 0
            for i in range(len(df) - 1, 0, -1):
                if close.iloc[i] > ema_50.iloc[i]:
                    current_age += 1
                    below_count = 0
                else:
                    below_count += 1
                    if below_count >= 2:
                        break
                        
        # 4. Compute conditional survival probability P(Age >= Current + 20 | Age >= Current)
        T = min(100, current_age) # cap to avoid index out of bounds
        S_T = curve.get(T, 0.05)
        S_T_plus_20 = curve.get(T + 20, 0.0)
        
        if S_T > 0:
            cond_prob = S_T_plus_20 / S_T
        else:
            cond_prob = 0.0
            
        cond_prob_pct = round(float(cond_prob) * 100.0, 1)
        warning = cond_prob_pct < 60.0
        
        return {
            "success": True,
            "curve": curve,
            "current_age": current_age,
            "survival_prob_20d": cond_prob_pct,
            "warning": warning
        }
        
    except Exception as e:
        logger.warning(f"Survival analysis failed for {ticker_symbol}: {e}. Returning fallback estimate.")
        # Generates fallback flat curve
        curve = {d: round(0.95 ** (d / 5.0), 3) for d in range(0, 121)}
        return {
            "success": False,
            "curve": curve,
            "current_age": 10,
            "survival_prob_20d": 72.5,
            "warning": False
        }

def analyze_trend_survival_with_df(ticker_symbol: str, df: pd.DataFrame) -> dict:
    """
    Perform survival analysis on the ticker's trend durations using an existing DataFrame.
    Saves an extra API call.
    """
    try:
        if df.empty or len(df) < 60:
            raise ValueError(f"Insufficient history in DataFrame for {ticker_symbol}")
            
        durations = extract_trend_durations(df)
        curve = fit_kaplan_meier(durations)
        
        close = df['Close']
        ema_50 = close.ewm(span=50, adjust=False).mean()
        
        current_age = 0
        if close.iloc[-1] > ema_50.iloc[-1]:
            below_count = 0
            for i in range(len(df) - 1, 0, -1):
                if close.iloc[i] > ema_50.iloc[i]:
                    current_age += 1
                    below_count = 0
                else:
                    below_count += 1
                    if below_count >= 2:
                        break
                        
        T = min(100, current_age)
        S_T = curve.get(T, 0.05)
        S_T_plus_20 = curve.get(T + 20, 0.0)
        
        if S_T > 0:
            cond_prob = S_T_plus_20 / S_T
        else:
            cond_prob = 0.0
            
        cond_prob_pct = round(float(cond_prob) * 100.0, 1)
        warning = cond_prob_pct < 60.0
        
        return {
            "success": True,
            "curve": curve,
            "current_age": current_age,
            "survival_prob_20d": cond_prob_pct,
            "warning": warning
        }
    except Exception as e:
        logger.warning(f"Offline survival analysis failed for {ticker_symbol}: {e}")
        curve = {d: round(0.95 ** (d / 5.0), 3) for d in range(0, 121)}
        return {
            "success": False,
            "curve": curve,
            "current_age": 10,
            "survival_prob_20d": 72.5,
            "warning": False
        }
