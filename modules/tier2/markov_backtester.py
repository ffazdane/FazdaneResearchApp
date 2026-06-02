import numpy as np
import pandas as pd
from datetime import datetime

def run_walk_forward_backtest(df: pd.DataFrame, initial_train_pct: float = 0.3) -> dict:
    """
    Run a walk-forward backtest of the Markov signal to predict next-day states 
    and evaluate a simple regime trading strategy.
    
    Strategy:
    - Long if Markov Signal >= 0.20
    - Flat if Markov Signal <= -0.20
    - Flat otherwise
    """
    df = df.copy().sort_values("trade_date").reset_index(drop=True)
    n_samples = len(df)
    
    if n_samples < 100:
        return {
            "total_return": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0,
            "win_rate": 0.0, "prediction_accuracy": 0.0,
            "bull_precision": 0.0, "bear_precision": 0.0, "sideways_precision": 0.0,
            "equity_curve": [1.0] * max(n_samples, 1),
            "dates": df["trade_date"].tolist()
        }
        
    start_idx = int(n_samples * initial_train_pct)
    
    # States mapping
    state_list = ["BULL", "SIDEWAYS", "BEAR"]
    state_to_idx = {s: i for i, s in enumerate(state_list)}
    
    # Initialize transition counts matrix (Laplace smoothing)
    counts = np.ones((3, 3)) * 0.1
    
    # Populate initial training counts
    for t in range(1, start_idx):
        s_curr = df.loc[t-1, "price_state"]
        s_next = df.loc[t, "price_state"]
        if s_curr in state_to_idx and s_next in state_to_idx:
            counts[state_to_idx[s_curr], state_to_idx[s_next]] += 1.0
            
    predictions = []
    actuals = []
    signals = []
    
    strat_returns = []
    benchmark_returns = []
    
    for t in range(start_idx, n_samples - 1):
        s_curr = df.loc[t, "price_state"]
        s_curr_idx = state_to_idx.get(s_curr)
        
        # Calculate transition probabilities for this day based on history
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P = counts / row_sums
        
        # Predict next state (most probable transition from current state)
        if s_curr_idx is not None:
            pred_idx = np.argmax(P[s_curr_idx])
            pred_state = state_list[pred_idx]
            
            # Markov Signal: P(BULL next) - P(BEAR next)
            markov_sig = P[s_curr_idx, state_to_idx["BULL"]] - P[s_curr_idx, state_to_idx["BEAR"]]
        else:
            pred_state = "SIDEWAYS"
            markov_sig = 0.0
            
        predictions.append(pred_state)
        signals.append(markov_sig)
        
        # Compare with actual next state
        s_next = df.loc[t+1, "price_state"]
        actuals.append(s_next)
        
        # Strategy Return simulation
        next_return = df.loc[t+1, "daily_return"]
        if pd.isna(next_return):
            next_return = 0.0
            
        # Strategy action based on signal score
        action = 1.0 if markov_sig >= 0.20 else -1.0 if markov_sig <= -0.50 else 0.0
        
        strat_returns.append(action * next_return)
        benchmark_returns.append(next_return)
        
        # Update transition counts for next step (no lookahead bias)
        s_next_idx = state_to_idx.get(s_next)
        if s_curr_idx is not None and s_next_idx is not None:
            counts[s_curr_idx, s_next_idx] += 1.0

    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Accuracy
    correct = (predictions == actuals)
    accuracy = np.mean(correct) if len(correct) > 0 else 0.0
    
    # Precision per state
    precisions = {}
    for state in state_list:
        tp = np.sum((predictions == state) & (actuals == state))
        fp = np.sum((predictions == state) & (actuals != state))
        precisions[state] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
    # Strategy PnL performance
    strat_returns = np.array(strat_returns)
    equity_curve = np.cumprod(1.0 + strat_returns)
    total_return = float(equity_curve[-1] - 1.0) if len(equity_curve) > 0 else 0.0
    
    # Sharpe Ratio (annualized)
    daily_rf = 0.0
    excess_returns = strat_returns - daily_rf
    std_dev = np.std(excess_returns)
    sharpe = float(np.mean(excess_returns) / std_dev * np.sqrt(252)) if std_dev > 0 else 0.0
    
    # Max Drawdown
    peak = 1.0
    max_dd = 0.0
    curr_equity = 1.0
    for r in strat_returns:
        curr_equity *= (1.0 + r)
        if curr_equity > peak:
            peak = curr_equity
        dd = (peak - curr_equity) / peak
        if dd > max_dd:
            max_dd = dd
            
    # Win Rate (percentage of positive return days)
    wins = np.sum(strat_returns > 0)
    trades = np.sum(strat_returns != 0)
    win_rate = float(wins / trades) if trades > 0 else 0.0
    
    dates = df.loc[start_idx:n_samples-2, "trade_date"].tolist()
    
    return {
        "total_return": round(total_return * 100, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "win_rate": round(win_rate * 100, 2),
        "prediction_accuracy": round(accuracy * 100, 2),
        "bull_precision": round(precisions["BULL"] * 100, 2),
        "bear_precision": round(precisions["BEAR"] * 100, 2),
        "sideways_precision": round(precisions["SIDEWAYS"] * 100, 2),
        "equity_curve": list(equity_curve),
        "dates": dates
    }
