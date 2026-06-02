"""
FazDane Core Technical Indicators and Analytics Library
======================================================
Unified implementation of standard technical indicators, advanced RRG rotation,
FDTS trend signals, and price action lifecycle classification.
"""

import numpy as np
import pandas as pd

# =====================================================================
# Basic Smoothing Helpers
# =====================================================================

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA)."""
    return series.ewm(span=span, adjust=False).mean()

def _tema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Triple Exponential Moving Average (TEMA)."""
    ema1 = _ema(series, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3

# =====================================================================
# Core Indicators
# =====================================================================

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-5)
    return 100 - (100 / (1 + rs))

def calculate_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Moving Average Convergence Divergence (MACD)."""
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Average Directional Index (ADX) and +/- Directional Indicators."""
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

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# =====================================================================
# RRG Rotational Coordinates
# =====================================================================

def compute_rrg_zscore(close: pd.DataFrame, benchmark: str, lookback_days: int = 90, trail_days: int = 20) -> pd.DataFrame:
    """Classical Log Z-Score RRG (Relative Rotation Graph) Model (used in Tier 1 Matrix)."""
    if benchmark not in close:
        return pd.DataFrame()
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

def compute_rrg_ratio_ema(ticker_prices: pd.Series, bench_prices: pd.Series, ema_span: int = 4) -> tuple[pd.Series, pd.Series]:
    """Smoothed Price Ratio RRG Model (used in Tier 2 Price Action Story)."""
    rel_strength = ticker_prices / bench_prices
    rel_strength_smooth = rel_strength.ewm(span=ema_span, adjust=False).mean()
    
    rs_ratio_raw = 100 * (rel_strength_smooth / rel_strength_smooth.rolling(10).mean())
    rs_ratio = rs_ratio_raw.ewm(span=ema_span, adjust=False).mean()
    
    rs_mom_raw = 100 * (rs_ratio / rs_ratio.rolling(5).mean())
    rs_momentum = rs_mom_raw.ewm(span=ema_span, adjust=False).mean()
    return rs_ratio, rs_momentum

def compute_rrg_ratio_sma(close_df: pd.DataFrame, benchmark_ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SMA Price Ratio RRG Model (used in Tier 2 Universe Intelligence)."""
    if benchmark_ticker not in close_df.columns:
        bench_series = close_df.mean(axis=1)
    else:
        bench_series = close_df[benchmark_ticker]

    rs_ratio_df = pd.DataFrame(index=close_df.index)
    rs_mom_df = pd.DataFrame(index=close_df.index)

    for col in close_df.columns:
        ratio = close_df[col] / bench_series
        ratio_sma = ratio.rolling(100, min_periods=30).mean()
        rs_ratio = 100 * (ratio / ratio_sma)
        
        rs_ratio_sma = rs_ratio.rolling(20, min_periods=5).mean()
        rs_mom = 100 * (rs_ratio / rs_ratio_sma)
        
        rs_ratio_df[col] = rs_ratio
        rs_mom_df[col] = rs_mom
        
    return rs_ratio_df.dropna(how="all"), rs_mom_df.dropna(how="all")

# =====================================================================
# FDTS Signals
# =====================================================================

def calculate_fdts_ha_signal(ticker_df: pd.DataFrame, period: int = 20) -> str:
    """Calculate Heikin-Ashi and Triple EMA (TEMA) deviation FDTS signal."""
    if ticker_df.empty:
        return "No Trade"

    # Normalize column names to title case (e.g. open -> Open) to handle database vs yfinance differences
    data_df = ticker_df.copy()
    data_df.columns = [col.capitalize() for col in data_df.columns]

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(data_df.columns):
        return "No Trade"

    data = data_df[["Open", "High", "Low", "Close"]].dropna().copy()
    if len(data) < 60:
        return "No Trade"

    price = (data["High"] + data["Low"] + data["Close"]) / 3
    tma1 = _tema(price, period)
    tma2 = _tema(tma1, period)
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

    ha_tma1 = _tema(ha_close, period)
    ha_tma2 = _tema(ha_tma1, period)
    ha_tema = ha_tma1 + (ha_tma1 - ha_tma2)
    fdts_dev = typical_tema - ha_tema

    macd_long = _ema(data["Close"], 3) - _ema(data["Close"], 10)
    macd_long_dev = macd_long - _ema(macd_long, 16)
    macd_short = _ema(data["Close"], 12) - _ema(data["Close"], 26)
    macd_short_dev = macd_short - _ema(macd_short, 9)

    state = pd.Series(0, index=data.index, dtype="int64")
    state[(fdts_dev > 0) & (macd_long_dev > 0)] = 1
    state[(fdts_dev < 0) & (macd_short_dev < 0)] = -1

    current_state = int(state.iloc[-1])
    return "Buy" if current_state == 1 else "Sell" if current_state == -1 else "No Trade"

def format_fdts_signal(sig: str) -> str:
    """Format raw FDTS signal string into display format with emojis."""
    return {"Buy": "🟢 Buy", "Sell": "🔴 Sell", "No Trade": "⚪ No Trade", "Neutral": "⚪ Neutral"}.get(sig, "⚪ No Trade")

# =====================================================================
# Price Action Lifecycle Analysis
# =====================================================================

def evaluate_price_action_lifecycle(ticker_df: pd.DataFrame, bench_df: pd.DataFrame) -> dict:
    """Calculate 8 Price Action story metrics and score the ticker state (Tier 2 Story Engine)."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    if ticker_df.empty or not required.issubset(ticker_df.columns) or len(ticker_df) < 50:
        return {}

    close = ticker_df["Close"].dropna()
    volume = ticker_df["Volume"].dropna()
    high = ticker_df["High"].dropna()
    low = ticker_df["Low"].dropna()
    
    # 1. Trend Structure (20% Weight)
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
    
    # 2. Volume Participation (15% Weight)
    vol20_avg = volume.rolling(20).mean()
    vol20_max = vol20_avg.rolling(60).max()
    vpr = vol20_avg.iloc[-1] / vol20_max.iloc[-1] if vol20_max.iloc[-1] > 0 else 0.5
    volume_score = min(100.0, float(vpr * 100.0))
    
    # 3. Relative Strength (15% Weight)
    aligned_bench = bench_df["Close"].reindex(close.index).ffill()
    rs_ratio_s, rs_mom_s = compute_rrg_ratio_ema(close, aligned_bench)
    last_rs = rs_ratio_s.iloc[-1]
    last_mom = rs_mom_s.iloc[-1]
    
    if last_rs >= 100 and last_mom >= 100: rs_score = 100
    elif last_rs < 100 and last_mom >= 100: rs_score = 70
    elif last_rs >= 100 and last_mom < 100: rs_score = 45
    else: rs_score = 15
    
    # 4. Momentum (15% Weight)
    rsi_s = calculate_rsi(close)
    macd_line, macd_sig, macd_hist = calculate_macd(close)
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

    # 5. ADX Trend Strength (10% Weight)
    adx_s, plus_di_s, minus_di_s = calculate_adx(high, low, close)
    last_adx = adx_s.iloc[-1]
    last_plus = plus_di_s.iloc[-1]
    last_minus = minus_di_s.iloc[-1]
    
    adx_score = 50
    if last_adx >= 25:
        if last_plus > last_minus: adx_score = 100
        else: adx_score = 25
    else:
        adx_score = 50
        
    # 6. ATR Volatility Setup (10% Weight)
    atr_s = calculate_atr(high, low, close)
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
        
    # 7. Distribution Risk (10% Weight)
    is_down_day = close.diff() < 0
    is_higher_vol = volume.diff() > 0
    dist_days = (is_down_day & is_higher_vol).iloc[-20:].sum()
    dist_score = max(0.0, 100.0 - (dist_days * 15.0))
    
    # 8. CVD Confirmation (5% Weight)
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
    
    # FDTS signal (with emoji formatted for price_action_story dashboard display)
    fdts_raw = calculate_fdts_ha_signal(ticker_df)
    fdts_val = format_fdts_signal(fdts_raw)
    
    return {
        "Close": last_close, "Volume": volume.iloc[-1], "VPR": vpr, "RS Ratio": last_rs, "RS Momentum": last_mom,
        "RSI": last_rsi, "MACD Line": macd_line.iloc[-1], "MACD Signal": macd_sig.iloc[-1], "ADX": last_adx,
        "ATR": last_atr, "ATR Percentile": atr_pct, "CVD": last_cvd, "Distribution Days": dist_days,
        "Health Score": master_score, "Stage": stage, "FDTS": fdts_val, "HH": hh, "HL": hl,
        "Divergence": divergence_warning, "CVD Slope": cvd_slope, "CVD Score": cvd_score,
        "Trend Score": trend_score, "Vol Score": volume_score, "RS Score": rs_score, "Mom Score": mom_score,
        "ADX Score": adx_score, "ATR Score": atr_score, "Dist Score": dist_score,
        "ROC10": roc10, "ROC20": roc20, "ROC60": roc60
    }
