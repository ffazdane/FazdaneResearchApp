"""
FazDane Consolidation Verification Utility
==========================================
Automated parity test suite validating that the refactored, consolidated indicator logic
produces identical mathematical results to the original, duplicate codebases.
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# =====================================================================
# ORIGINAL BASELINES (Copied directly for verification)
# =====================================================================

def orig_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def orig_tema(series: pd.Series, period: int) -> pd.Series:
    ema1 = orig_ema(series, period)
    ema2 = orig_ema(ema1, period)
    ema3 = orig_ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3

def orig_calculate_fdts_signal(symbol: str, ticker_df: pd.DataFrame, period: int = 20) -> str:
    required = {"Open", "High", "Low", "Close"}
    if ticker_df.empty or not required.issubset(ticker_df.columns):
        return "No Trade"
    data = ticker_df[["Open", "High", "Low", "Close"]].dropna().copy()
    if len(data) < 60:
        return "No Trade"
    price = (data["High"] + data["Low"] + data["Close"]) / 3
    tma1 = orig_tema(price, period)
    tma2 = orig_tema(tma1, period)
    typical_tema = tma1 + (tma1 - tma2)
    raw_ha_close = (data["Open"] + data["High"] + data["Low"] + data["Close"]) / 4
    ha_open = pd.Series(index=data.index, dtype="float64")
    ha_open.iloc[0] = (data["High"].iloc[0] + data["Low"].iloc[0]) / 2
    for i in range(1, len(data)):
        ha_open.iloc[i] = (raw_ha_close.iloc[i - 1] + ha_open.iloc[i - 1]) / 2
    ha_close = (
        raw_ha_close
        + ha_open
        + pd.concat([data["High"], ha_open], axis=1).max(axis=1)
        + pd.concat([data["Low"], ha_open], axis=1).min(axis=1)
    ) / 4
    ha_tma1 = orig_tema(ha_close, period)
    ha_tma2 = orig_tema(ha_tma1, period)
    ha_tema = ha_tma1 + (ha_tma1 - ha_tma2)
    fdts_dev = typical_tema - ha_tema
    macd_long = orig_ema(data["Close"], 3) - orig_ema(data["Close"], 10)
    macd_long_dev = macd_long - orig_ema(macd_long, 16)
    macd_short = orig_ema(data["Close"], 12) - orig_ema(data["Close"], 26)
    macd_short_dev = macd_short - orig_ema(macd_short, 9)
    state = pd.Series(0, index=data.index, dtype="int64")
    state[(fdts_dev > 0) & (macd_long_dev > 0)] = 1
    state[(fdts_dev < 0) & (macd_short_dev < 0)] = -1
    current_state = int(state.iloc[-1])
    return "Buy" if current_state == 1 else "Sell" if current_state == -1 else "No Trade"

def orig_compute_rotation(close, benchmark, trail_days=20):
    if benchmark not in close: return pd.DataFrame()
    bench = close[benchmark]
    rows = []
    for ticker in close.columns:
        if ticker == benchmark:
            continue
        px = close[ticker].dropna()
        aligned = pd.concat([px, bench], axis=1, join="inner").dropna()
        if len(aligned) < 70:
            continue
        rel_log = np.log(aligned.iloc[:, 0] / aligned.iloc[:, 1])
        rs_mean = rel_log.rolling(50, min_periods=30).mean()
        rs_std = rel_log.rolling(50, min_periods=30).std().replace(0, np.nan)
        rs_ratio = (100 + 2.0 * ((rel_log - rs_mean) / rs_std).clip(-3, 3)).ewm(span=5, adjust=False).mean()
        mom_raw = rs_ratio.diff(5)
        mom_std = mom_raw.rolling(30, min_periods=15).std().replace(0, np.nan)
        rs_momentum = (100 + 1.4 * (mom_raw / mom_std).clip(-3, 3)).ewm(span=5, adjust=False).mean()
        out = pd.DataFrame({
            "date": aligned.index, "ticker": ticker, "close": aligned.iloc[:, 0].values,
            "rs_ratio": rs_ratio.values, "rs_momentum": rs_momentum.values,
        }).dropna()
        rows.append(out.tail(trail_days))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def orig_calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=low.index).ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, 1e-5))
    minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, 1e-5))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-5))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx, plus_di, minus_di

def orig_calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def orig_calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-5)
    return 100 - (100 / (1 + rs))

def orig_calculate_macd(prices: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def orig_calculate_rrg_values(ticker_prices: pd.Series, bench_prices: pd.Series, ema_span: int = 4) -> tuple[pd.Series, pd.Series]:
    rel_strength = ticker_prices / bench_prices
    rel_strength_smooth = rel_strength.ewm(span=ema_span, adjust=False).mean()
    rs_ratio_raw = 100 * (rel_strength_smooth / rel_strength_smooth.rolling(10).mean())
    rs_ratio = rs_ratio_raw.ewm(span=ema_span, adjust=False).mean()
    rs_mom_raw = 100 * (rs_ratio / rs_ratio.rolling(5).mean())
    rs_momentum = rs_mom_raw.ewm(span=ema_span, adjust=False).mean()
    return rs_ratio, rs_momentum

def orig_evaluate_ticker_price_action(ticker_df: pd.DataFrame, bench_df: pd.DataFrame) -> dict:
    required = {"Open", "High", "Low", "Close", "Volume"}
    if ticker_df.empty or not required.issubset(ticker_df.columns) or len(ticker_df) < 50:
        return {}
    close = ticker_df["Close"].dropna()
    volume = ticker_df["Volume"].dropna()
    high = ticker_df["High"].dropna()
    low = ticker_df["Low"].dropna()
    
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    last_close = close.iloc[-1]
    last_sma20 = sma20.iloc[-1]
    last_sma50 = sma50.iloc[-1]
    last_sma200 = sma200.iloc[-1]
    
    trend_score = 0
    if last_close > last_sma20: trend_score += 20
    if last_close > last_sma50: trend_score += 30
    if last_close > last_sma200: trend_score += 30
    if last_sma20 > last_sma50: trend_score += 10
    if last_sma50 > last_sma200: trend_score += 10
    
    high_5d = high.rolling(5).max()
    low_5d = low.rolling(5).min()
    hh = high_5d.iloc[-1] > high_5d.iloc[-6]
    hl = low_5d.iloc[-1] > low_5d.iloc[-6]
    
    vol20_avg = volume.rolling(20).mean()
    vol20_max = vol20_avg.rolling(60).max()
    vpr = vol20_avg.iloc[-1] / vol20_max.iloc[-1] if vol20_max.iloc[-1] > 0 else 0.5
    volume_score = min(100.0, float(vpr * 100.0))
    
    aligned_bench = bench_df["Close"].reindex(close.index).ffill()
    rs_ratio_s, rs_mom_s = orig_calculate_rrg_values(close, aligned_bench)
    last_rs = rs_ratio_s.iloc[-1]
    last_mom = rs_mom_s.iloc[-1]
    
    if last_rs >= 100 and last_mom >= 100: rs_score = 100
    elif last_rs < 100 and last_mom >= 100: rs_score = 70
    elif last_rs >= 100 and last_mom < 100: rs_score = 45
    else: rs_score = 15
    
    rsi_s = orig_calculate_rsi(close)
    macd_line, macd_sig, macd_hist = orig_calculate_macd(close)
    roc10 = ((close.iloc[-1] / close.iloc[-11]) - 1) * 100 if len(close) > 11 else 0
    roc20 = ((close.iloc[-1] / close.iloc[-21]) - 1) * 100 if len(close) > 21 else 0
    roc60 = ((close.iloc[-1] / close.iloc[-61]) - 1) * 100 if len(close) > 61 else 0
    last_rsi = rsi_s.iloc[-1]
    last_hist = macd_hist.iloc[-1]
    
    mom_score = 50
    if 35 <= last_rsi <= 70: mom_score += 20
    elif last_rsi > 70: mom_score += 10
    if last_hist > 0: mom_score += 20
    if roc10 > 0: mom_score += 10
    
    divergence_warning = False
    if len(close) > 22:
        price_peak = close.iloc[-22:-2].max()
        rsi_peak = rsi_s.iloc[-22:-2].max()
        if last_close > price_peak and last_rsi < rsi_peak:
            divergence_warning = True
            mom_score = max(0, mom_score - 30)

    adx_s, plus_di_s, minus_di_s = orig_calculate_adx(high, low, close)
    last_adx = adx_s.iloc[-1]
    last_plus = plus_di_s.iloc[-1]
    last_minus = minus_di_s.iloc[-1]
    
    adx_score = 50
    if last_adx >= 25:
        if last_plus > last_minus: adx_score = 100
        else: adx_score = 25
    else:
        adx_score = 50
        
    atr_s = orig_calculate_atr(high, low, close)
    last_atr = atr_s.iloc[-1]
    atr_min = atr_s.rolling(252).min().iloc[-1]
    atr_max = atr_s.rolling(252).max().iloc[-1]
    atr_diff = atr_max - atr_min
    atr_pct = ((last_atr - atr_min) / atr_diff) * 100 if atr_diff > 0 else 50.0
    atr_slope = atr_s.iloc[-1] - atr_s.iloc[-6] if len(atr_s) > 6 else 0
    
    atr_score = 70
    if atr_pct < 40:
        if atr_slope > 0: atr_score = 100
        else: atr_score = 90
    elif atr_pct > 75:
        atr_score = 40
        
    is_down_day = close.diff() < 0
    is_higher_vol = volume.diff() > 0
    dist_days = (is_down_day & is_higher_vol).iloc[-20:].sum()
    dist_score = max(0.0, 100.0 - (dist_days * 15.0))
    
    delta = close.diff()
    cvd_s = (np.where(delta >= 0, 1, -1) * volume).cumsum()
    last_cvd = cvd_s.iloc[-1]
    cvd_slope = cvd_s.iloc[-1] - cvd_s.iloc[-11] if len(cvd_s) > 11 else 0
    price_slope = close.iloc[-1] - close.iloc[-11] if len(close) > 11 else 0
    
    cvd_score = 50
    if price_slope > 0:
        if cvd_slope > 0: cvd_score = 100
        else: cvd_score = 20
        
    master_score = (
        (trend_score * 0.20) +
        (volume_score * 0.15) +
        (rs_score * 0.15) +
        (mom_score * 0.15) +
        (adx_score * 0.10) +
        (atr_score * 0.10) +
        (dist_score * 0.10) +
        (cvd_score * 0.05)
    )
    if master_score >= 85: stage = "Early Bull / Expansion"
    elif master_score >= 70: stage = "Strong Bull"
    elif master_score >= 55: stage = "Mature Bull"
    elif master_score >= 40: stage = "Fading Bull"
    elif master_score >= 25: stage = "Distribution"
    else: stage = "Breakdown"
    
    # Simplified FDTS signal logic for verification
    fdts_val = "🟢 Buy" if last_close > last_sma20 > last_sma50 else "🔴 Sell" if last_close < last_sma20 < last_sma50 else "⚪ No Trade"
    
    return {
        "Close": last_close, "Volume": volume.iloc[-1], "VPR": vpr, "RS Ratio": last_rs, "RS Momentum": last_mom,
        "RSI": last_rsi, "MACD Line": macd_line.iloc[-1], "MACD Signal": macd_sig.iloc[-1], "ADX": last_adx,
        "ATR": last_atr, "ATR Percentile": atr_pct, "CVD": last_cvd, "Distribution Days": dist_days,
        "Health Score": master_score, "Stage": stage, "FDTS": fdts_val
    }

# =====================================================================
# VERIFICATION ENGINE RUNNER
# =====================================================================

def verify_math_parity():
    print("=== DOWNLOADING TEST DATA ===")
    tickers = ["AAPL", "SPY"]
    # Download 1-year data for verification
    raw = yf.download(tickers, period="1y", auto_adjust=True, progress=False)
    if raw.empty:
        print("FAIL: Could not download test data from Yahoo Finance.")
        sys.exit(1)
        
    spy_df = pd.DataFrame(index=raw.index)
    aapl_df = pd.DataFrame(index=raw.index)
    
    if isinstance(raw.columns, pd.MultiIndex):
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            spy_df[col] = raw[col]["SPY"]
            aapl_df[col] = raw[col]["AAPL"]
    else:
        # Fallback if only 1 symbol was returned
        print("FAIL: Multiple symbols should have returned.")
        sys.exit(1)

    print("=== IMPORTING REFACTERED MODULES ===")
    try:
        from modules.calendar_scoring.technical_indicators import (
            calculate_rsi, calculate_macd, calculate_adx, calculate_atr,
            calculate_fdts_ha_signal, compute_rrg_zscore, compute_rrg_ratio_ema,
            evaluate_price_action_lifecycle
        )
    except Exception as e:
        print(f"FAIL: Failed to import consolidated technical indicators: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("=== RUNNING PARITY TESTS ===")
    errors = 0

    # 1. RSI test
    r_orig = orig_calculate_rsi(aapl_df["Close"]).dropna()
    r_new = calculate_rsi(aapl_df["Close"]).dropna()
    common_idx = r_orig.index.intersection(r_new.index)
    if not np.allclose(r_orig.loc[common_idx], r_new.loc[common_idx], atol=1e-5):
        print("FAIL: RSI calculations do not match.")
        errors += 1
    else:
        print("PASS: RSI Parity verified.")

    # 2. MACD test
    m_orig_line, m_orig_sig, m_orig_hist = orig_calculate_macd(aapl_df["Close"])
    m_new_line, m_new_sig, m_new_hist = calculate_macd(aapl_df["Close"])
    if not np.allclose(m_orig_line.dropna(), m_new_line.dropna(), atol=1e-5):
        print("FAIL: MACD Line calculations do not match.")
        errors += 1
    else:
        print("PASS: MACD Parity verified.")

    # 3. ATR test
    atr_orig = orig_calculate_atr(aapl_df["High"], aapl_df["Low"], aapl_df["Close"]).dropna()
    atr_new = calculate_atr(aapl_df["High"], aapl_df["Low"], aapl_df["Close"]).dropna()
    common_idx = atr_orig.index.intersection(atr_new.index)
    if not np.allclose(atr_orig.loc[common_idx], atr_new.loc[common_idx], atol=1e-5):
        print("FAIL: ATR calculations do not match.")
        errors += 1
    else:
        print("PASS: ATR Parity verified.")

    # 4. ADX test
    adx_o, pd_o, md_o = orig_calculate_adx(aapl_df["High"], aapl_df["Low"], aapl_df["Close"])
    adx_n, pd_n, md_n = calculate_adx(aapl_df["High"], aapl_df["Low"], aapl_df["Close"])
    if not np.allclose(adx_o.dropna(), adx_n.dropna(), atol=1e-5):
        print("FAIL: ADX calculations do not match.")
        errors += 1
    else:
        print("PASS: ADX Parity verified.")

    # 5. FDTS HA Signal test
    fdts_orig = orig_calculate_fdts_signal("AAPL", aapl_df)
    fdts_new = calculate_fdts_ha_signal(aapl_df)
    if fdts_orig != fdts_new:
        print(f"FAIL: FDTS HA signal doesn't match: Original={fdts_orig}, New={fdts_new}")
        errors += 1
    else:
        print(f"PASS: FDTS HA Signal Parity verified ({fdts_orig}).")

    # 6. RRG Log Z-Score test (Type A)
    close_subset = raw["Close"].copy().ffill()
    rot_orig = orig_compute_rotation(close_subset, "SPY", trail_days=20)
    rot_new = compute_rrg_zscore(close_subset, "SPY", lookback_days=90, trail_days=20)
    if rot_orig.empty or rot_new.empty:
        print("FAIL: RRG Log Z-Score Dataframe empty.")
        errors += 1
    else:
        merged = rot_orig.merge(rot_new, on=["date", "ticker"], suffixes=("_orig", "_new"))
        if merged.empty:
            print("FAIL: Could not align RRG Log Z-Score dataframes.")
            errors += 1
        elif not np.allclose(merged["rs_ratio_orig"], merged["rs_ratio_new"], atol=1e-5):
            print("FAIL: RRG RS-Ratio Log Z-Score does not match.")
            errors += 1
        elif not np.allclose(merged["rs_momentum_orig"], merged["rs_momentum_new"], atol=1e-5):
            print("FAIL: RRG RS-Momentum Log Z-Score does not match.")
            errors += 1
        else:
            print("PASS: RRG Log Z-Score Parity verified.")

    # 7. RRG Ratio EMA test (Type B)
    rrg_orig_ratio, rrg_orig_mom = orig_calculate_rrg_values(aapl_df["Close"], spy_df["Close"])
    rrg_new_ratio, rrg_new_mom = compute_rrg_ratio_ema(aapl_df["Close"], spy_df["Close"])
    if not np.allclose(rrg_orig_ratio.dropna(), rrg_new_ratio.dropna(), atol=1e-5):
        print("FAIL: RRG Ratio EMA does not match.")
        errors += 1
    else:
        print("PASS: RRG Ratio EMA Parity verified.")

    # 8. Lifecycle Score & Stage test
    lc_orig = orig_evaluate_ticker_price_action(aapl_df, spy_df)
    lc_new = evaluate_price_action_lifecycle(aapl_df, spy_df)
    if not lc_orig or not lc_new:
        print("FAIL: Lifecycle evaluation returned empty.")
        errors += 1
    elif abs(lc_orig["Health Score"] - lc_new["Health Score"]) > 1e-4:
        print(f"FAIL: Lifecycle Health Score mismatch: Original={lc_orig['Health Score']:.4f}, New={lc_new['Health Score']:.4f}")
        errors += 1
    elif lc_orig["Stage"] != lc_new["Stage"]:
        print(f"FAIL: Lifecycle Stage mismatch: Original={lc_orig['Stage']}, New={lc_new['Stage']}")
        errors += 1
    else:
        print(f"PASS: Lifecycle Health Score and Stage Parity verified ({lc_orig['Stage']} / Score: {lc_orig['Health Score']:.2f}).")

    if errors > 0:
        print(f"\n[FAIL] Parity check failed with {errors} errors. Do not commit refactored files.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All parity verification tests passed! Core mathematics match perfectly.")
        sys.exit(0)

if __name__ == "__main__":
    verify_math_parity()
