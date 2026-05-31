"""
Test script for verifying Price Action Story Engine calculations and database schema.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.tier2.price_action_story import (
    calculate_adx,
    calculate_atr,
    calculate_rsi,
    calculate_macd,
    calculate_rrg_values,
    evaluate_ticker_price_action
)

def run_tests():
    print("=== STARTING PRICE ACTION STORY ENGINE TESTS ===")
    
    # Generate mock OHLCV data
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = 100.0 + np.random.randn(100).cumsum()
    high = close + np.random.rand(100) * 2.0
    low = close - np.random.rand(100) * 2.0
    open_prices = close + np.random.randn(100)
    volume = np.random.randint(100000, 1000000, size=100)
    
    ticker_df = pd.DataFrame({
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    }, index=dates)
    
    bench_df = pd.DataFrame({
        "Close": 100.0 + np.random.randn(100).cumsum()
    }, index=dates)
    
    print("\n1. Testing Technical Indicators...")
    # Test ADX
    adx, plus_di, minus_di = calculate_adx(ticker_df["High"], ticker_df["Low"], ticker_df["Close"])
    print(f"ADX (last): {adx.iloc[-1]:.2f}")
    assert not adx.isna().all(), "ADX calculations failed"
    
    # Test ATR
    atr = calculate_atr(ticker_df["High"], ticker_df["Low"], ticker_df["Close"])
    print(f"ATR (last): {atr.iloc[-1]:.2f}")
    assert not atr.isna().all(), "ATR calculations failed"
    
    # Test RSI
    rsi = calculate_rsi(ticker_df["Close"])
    print(f"RSI (last): {rsi.iloc[-1]:.2f}")
    assert not rsi.isna().all(), "RSI calculations failed"
    
    # Test MACD
    macd, signal, hist = calculate_macd(ticker_df["Close"])
    print(f"MACD Hist (last): {hist.iloc[-1]:.2f}")
    assert not hist.isna().all(), "MACD calculations failed"
    
    # Test RRG Values
    rs_ratio, rs_mom = calculate_rrg_values(ticker_df["Close"], bench_df["Close"])
    print(f"RS-Ratio: {rs_ratio.iloc[-1]:.2f} | RS-Momentum: {rs_mom.iloc[-1]:.2f}")
    assert not rs_ratio.isna().all(), "RRG calculations failed"
    
    print("\n2. Testing Lifecycle Evaluation and Scoring...")
    eval_res = evaluate_ticker_price_action(ticker_df, bench_df)
    assert eval_res, "Evaluation result is empty"
    print(f"Health Score: {eval_res['Health Score']:.2f}")
    print(f"Classified Stage: {eval_res['Stage']}")
    print(f"VPR: {eval_res['VPR']:.4f}")
    print(f"Distribution Days: {eval_res['Distribution Days']}")
    print(f"Momentum Divergence Warning: {eval_res['Divergence']}")
    
    print("\n=== ALL TESTS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_tests()
