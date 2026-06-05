import numpy as np
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("TradeIndicators")

# ══════════════════════════════════════════════════════════════════════
# BASIC SMOOTHING & TECHNICAL HELPERS
# ══════════════════════════════════════════════════════════════════════

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_tema(series: pd.Series, period: int) -> pd.Series:
    ema1 = calculate_ema(series, period)
    ema2 = calculate_ema(ema1, period)
    ema3 = calculate_ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    
    plus_dm = pd.Series(np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0), index=df.index)
    
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    
    plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, 1e-5))
    minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, 1e-5))
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-5))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

# ══════════════════════════════════════════════════════════════════════
# INDICATOR IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════

def calculate_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = calculate_ema(series, 12)
    ema26 = calculate_ema(series, 26)
    macd_line = ema12 - ema26
    signal_line = calculate_ema(macd_line, 9)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_fdts(df: pd.DataFrame, period: int = 20) -> tuple[pd.Series, pd.Series]:
    """Calculate Heikin-Ashi and Triple EMA (TEMA) deviation FDTS signal and raw delta."""
    if len(df) < 50:
        zeros = pd.Series(0.0, index=df.index)
        signals = pd.Series("Neutral", index=df.index)
        return zeros, signals
        
    price = (df['High'] + df['Low'] + df['Close']) / 3
    tma1 = calculate_tema(price, period)
    tma2 = calculate_tema(tma1, period)
    typical_tema = tma1 + (tma1 - tma2)

    raw_ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = pd.Series(index=df.index, dtype="float64")
    ha_open.iloc[0] = (df['High'].iloc[0] + df['Low'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (raw_ha_close.iloc[i - 1] + ha_open.iloc[i - 1]) / 2

    ha_close = (
        raw_ha_close
        + ha_open
        + pd.concat([df['High'], ha_open], axis=1).max(axis=1)
        + pd.concat([df['Low'], ha_open], axis=1).min(axis=1)
    ) / 4

    ha_tma1 = calculate_tema(ha_close, period)
    ha_tma2 = calculate_tema(ha_tma1, period)
    ha_tema = ha_tma1 + (ha_tma1 - ha_tma2)
    fdts_delta = typical_tema - ha_tema

    macd_long = calculate_ema(df['Close'], 3) - calculate_ema(df['Close'], 10)
    macd_long_dev = macd_long - calculate_ema(macd_long, 16)
    macd_short = calculate_ema(df['Close'], 12) - calculate_ema(df['Close'], 26)
    macd_short_dev = macd_short - calculate_ema(macd_short, 9)

    signals = pd.Series("Neutral", index=df.index)
    signals[(fdts_delta > 0) & (macd_long_dev > 0)] = "Buy"
    signals[(fdts_delta < 0) & (macd_short_dev < 0)] = "Sell"
    
    return fdts_delta, signals

def calculate_darvas_box(df: pd.DataFrame, period: int = 5) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Darvas Box top/bottom levels and breakout signals sequentially."""
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    n = len(df)
    
    box_tops = np.zeros(n)
    box_bottoms = np.zeros(n)
    signals = ["Neutral"] * n
    
    if n < 20:
        return pd.Series(highs, index=df.index), pd.Series(lows, index=df.index), pd.Series(signals, index=df.index)
        
    state = 0 # 0 = scanning for top, 1 = scanning for bottom, 2 = box active
    box_top = highs[0]
    box_bottom = lows[0]
    
    top_candidate = highs[0]
    top_candidate_idx = 0
    
    for i in range(1, n):
        # 1. Scanning for top
        if state == 0:
            if highs[i] > top_candidate:
                top_candidate = highs[i]
                top_candidate_idx = i
            elif i - top_candidate_idx >= period:
                # Top holds for period days
                box_top = top_candidate
                state = 1
                bottom_candidate = lows[i]
                bottom_candidate_idx = i
        
        # 2. Scanning for bottom
        elif state == 1:
            if highs[i] > box_top:
                # Invalidated top, reset search
                top_candidate = highs[i]
                top_candidate_idx = i
                state = 0
            elif lows[i] < bottom_candidate:
                bottom_candidate = lows[i]
                bottom_candidate_idx = i
            elif i - bottom_candidate_idx >= period:
                # Bottom holds for period days
                box_bottom = bottom_candidate
                state = 2
                
        # 3. Box Active
        elif state == 2:
            if closes[i] > box_top:
                signals[i] = "Breakout"
                # Reset search for new box
                top_candidate = highs[i]
                top_candidate_idx = i
                state = 0
            elif closes[i] < box_bottom:
                signals[i] = "Breakdown"
                # Reset search for new box
                top_candidate = highs[i]
                top_candidate_idx = i
                state = 0
                
        box_tops[i] = box_top
        box_bottoms[i] = box_bottom
        
    return pd.Series(box_tops, index=df.index), pd.Series(box_bottoms, index=df.index), pd.Series(signals, index=df.index)

def calculate_linear_regression_channel(close_series: pd.Series, lookback: int = 50, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate linear regression channel middle, upper, and lower bands."""
    n = len(close_series)
    mid = pd.Series(index=close_series.index, dtype='float64')
    upper = pd.Series(index=close_series.index, dtype='float64')
    lower = pd.Series(index=close_series.index, dtype='float64')
    
    if n < lookback:
        # Fallback to simple moving averages
        ma = close_series.rolling(window=max(2, n)).mean()
        std = close_series.rolling(window=max(2, n)).std()
        return ma, ma + std_mult * std, ma - std_mult * std
        
    # Standard linear regression fit
    x = np.arange(lookback)
    for i in range(lookback - 1, n):
        y = close_series.iloc[i - lookback + 1: i + 1].values
        slope, intercept = np.polyfit(x, y, 1)
        fitted_val = slope * (lookback - 1) + intercept
        mid.iloc[i] = fitted_val
        
        residuals = y - (slope * x + intercept)
        std_dev = np.std(residuals)
        upper.iloc[i] = fitted_val + std_mult * std_dev
        lower.iloc[i] = fitted_val - std_mult * std_dev
        
    # Backfill first elements
    mid.iloc[:lookback-1] = mid.iloc[lookback-1]
    upper.iloc[:lookback-1] = upper.iloc[lookback-1]
    lower.iloc[:lookback-1] = lower.iloc[lookback-1]
    
    return mid, upper, lower

def calculate_wpr(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series]:
    """Calculate Williams %R and its signals."""
    high_max = df['High'].rolling(window=period).max()
    low_min = df['Low'].rolling(window=period).min()
    wpr = (high_max - df['Close']) / (high_max - low_min + 1e-8) * -100
    
    signals = pd.Series("Neutral", index=df.index)
    signals[wpr > -20] = "Overbought"
    signals[wpr < -80] = "Oversold"
    
    return wpr, signals

def calculate_ichimoku(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Ichimoku Tenkan, Kijun, and Spans A and B."""
    high_9 = df['High'].rolling(9).max()
    low_9 = df['Low'].rolling(9).min()
    tenkan = (high_9 + low_9) / 2
    
    high_26 = df['High'].rolling(26).max()
    low_26 = df['Low'].rolling(26).min()
    kijun = (high_26 + low_26) / 2
    
    span_a = ((tenkan + kijun) / 2).shift(26)
    
    high_52 = df['High'].rolling(52).max()
    low_52 = df['Low'].rolling(52).min()
    span_b = ((high_52 + low_52) / 2).shift(26)
    
    # Fill NaNs
    tenkan = tenkan.ffill().fillna(df['Close'])
    kijun = kijun.ffill().fillna(df['Close'])
    span_a = span_a.ffill().fillna(df['Close'])
    span_b = span_b.ffill().fillna(df['Close'])
    
    return tenkan, span_a, span_b

def calculate_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate typical price VWAP proxy for daily/hourly charts."""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_vol = tp * df['Volume']
    vwap = tp_vol.rolling(period).sum() / (df['Volume'].rolling(period).sum() + 1e-8)
    return vwap.ffill().fillna(df['Close'])

def calculate_market_forecast(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Calculate three-timeframe Stochastic composite (Market Forecast) and its signals."""
    def stoch(p):
        h = df['High'].rolling(p).max()
        l = df['Low'].rolling(p).min()
        return (df['Close'] - l) / (h - l + 1e-8) * 100
        
    s14 = stoch(14)
    s30 = stoch(30)
    s80 = stoch(80)
    
    composite = (s14 + s30 + s80) / 3
    
    signals = pd.Series("Neutral", index=df.index)
    signals[composite > 80] = "Overbought"
    signals[composite < 20] = "Oversold"
    
    return composite, signals

# ══════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS & ENGINES
# ══════════════════════════════════════════════════════════════════════

def run_indicators_scan(df: pd.DataFrame, timeframe: str) -> dict:
    """Run full indicator calculations on a dataframe for a specific timeframe."""
    if df.empty or len(df) < 20:
        return {}
        
    ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
    ma50 = float(df['Close'].rolling(50).mean().iloc[-1])
    ma200 = float(df['Close'].rolling(200).mean().iloc[-1]) if len(df) >= 200 else float(df['Close'].mean())
    
    vwap = float(calculate_vwap(df).iloc[-1])
    
    fdts_delta, fdts_sig_series = calculate_fdts(df)
    fdts_d = float(fdts_delta.iloc[-1])
    fdts_s = str(fdts_sig_series.iloc[-1])
    
    macd_val, macd_avg, macd_hist = calculate_macd(df['Close'])
    macd_v = float(macd_val.iloc[-1])
    macd_a = float(macd_avg.iloc[-1])
    macd_h = float(macd_hist.iloc[-1])
    macd_s = "Bullish" if macd_h > 0 else "Bearish"
    
    wpr_val, wpr_sig_series = calculate_wpr(df)
    wpr_v = float(wpr_val.iloc[-1])
    wpr_s = str(wpr_sig_series.iloc[-1])
    
    darvas_up, darvas_lo, darvas_sig_series = calculate_darvas_box(df)
    darvas_u = float(darvas_up.iloc[-1])
    darvas_l = float(darvas_lo.iloc[-1])
    darvas_s = str(darvas_sig_series.iloc[-1])
    
    reg_mid, reg_up, reg_lo = calculate_linear_regression_channel(df['Close'])
    reg_m = float(reg_mid.iloc[-1])
    reg_u = float(reg_up.iloc[-1])
    reg_l = float(reg_lo.iloc[-1])
    
    tenkan, span_a, span_b = calculate_ichimoku(df)
    ich_a = float(span_a.iloc[-1])
    ich_b = float(span_b.iloc[-1])
    spot = float(df['Close'].iloc[-1])
    if spot > ich_a and spot > ich_b:
        cloud_s = "Bullish"
    elif spot < ich_a and spot < ich_b:
        cloud_s = "Bearish"
    else:
        cloud_s = "Neutral"
        
    atr = float(calculate_atr(df).iloc[-1])
    
    return {
        "timeframe": timeframe,
        "price": spot,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "vwap": vwap,
        "fdts_delta": fdts_d,
        "fdts_signal": fdts_s,
        "macd_value": macd_v,
        "macd_avg": macd_a,
        "macd_hist": macd_h,
        "macd_signal": macd_s,
        "wpr_value": wpr_v,
        "wpr_signal": wpr_s,
        "darvas_upper": darvas_u,
        "darvas_lower": darvas_l,
        "darvas_signal": darvas_s,
        "regression_upper": reg_u,
        "regression_middle": reg_m,
        "regression_lower": reg_l,
        "ichimoku_span_a": ich_a,
        "ichimoku_span_b": ich_b,
        "cloud_signal": cloud_s,
        "atr14": atr
    }

def run_trend_engine(daily_indicators: dict) -> str:
    """Determine trend state: Strong Bull, Bullish Pullback, Bearish, Sideways, or Transition."""
    price = daily_indicators["price"]
    ma20 = daily_indicators["ma20"]
    ma50 = daily_indicators["ma50"]
    ma200 = daily_indicators["ma200"]
    
    if price > ma20 > ma50 > ma200:
        return "Strong Bull Trend"
    elif price > ma50 > ma200 and price < ma20:
        return "Bullish Pullback"
    elif price < ma20 < ma50 < ma200:
        return "Bearish Breakdown"
    elif price < ma50 < ma200 and price > ma20:
        return "Bearish Pullback in Downtrend"
    elif ma20 > ma50 > ma200 and price > ma200 and (price < ma50 or ma20 < ma50):
        return "Late Trend / Overextended"
    elif abs(price - ma50) / ma50 < 0.03 and abs(ma20 - ma50) / ma50 < 0.02:
        return "Sideways / Range"
    else:
        return "Transition / Mixed"

def run_regime_engine(df: pd.DataFrame, vix_value: float) -> str:
    """Classify regime: Trending, Mean Reverting, Volatile, or Compressed."""
    if df.empty or len(df) < 30:
        return "Mean Reverting"
        
    adx = calculate_adx(df).iloc[-1]
    
    # Calculate Bollinger Band Width relative compression
    closes = df['Close']
    sma20 = closes.rolling(20).mean()
    rstd20 = closes.rolling(20).std()
    bb_width = (rstd20 * 4) / sma20
    
    # Check percentile of BB Width over last 252 days
    bb_width_pct = 50.0
    if len(bb_width) > 50:
        rolling_min = bb_width.rolling(252, min_periods=30).min()
        rolling_max = bb_width.rolling(252, min_periods=30).max()
        bb_width_pct = (bb_width.iloc[-1] - rolling_min.iloc[-1]) / (rolling_max.iloc[-1] - rolling_min.iloc[-1] + 1e-8) * 100
        
    if vix_value > 22.0 or bb_width_pct > 80.0:
        return "Volatile"
    elif bb_width_pct < 20.0:
        return "Compressed"
    elif adx > 25.0:
        return "Trending"
    else:
        return "Mean Reverting"

def run_forecast_engine(spot_price: float, iv: float, daily_indicators: dict, trend_state: str, regime_state: str) -> tuple[str, float, float]:
    """Calculate expected 40-day path and blended expected range boundaries."""
    # 1. Probability Cone expected move
    expected_move_40d = spot_price * iv * np.sqrt(40 / 252.0)
    cone_low = spot_price - expected_move_40d
    cone_high = spot_price + expected_move_40d
    
    # 2. ATR Projection
    atr = daily_indicators["atr14"]
    atr_low = spot_price - atr * np.sqrt(40)
    atr_high = spot_price + atr * np.sqrt(40)
    
    # 3. Regression Channels
    reg_low = daily_indicators["regression_lower"]
    reg_high = daily_indicators["regression_upper"]
    
    # 4. Darvas levels
    darvas_low = daily_indicators["darvas_lower"]
    darvas_high = daily_indicators["darvas_upper"]
    
    # Blended Range calculations
    # Weights: Cone (40%), Regression (30%), Darvas (20%), ATR (10%)
    expected_low  = float(0.40 * cone_low  + 0.30 * reg_low  + 0.20 * darvas_low  + 0.10 * atr_low)
    expected_high = float(0.40 * cone_high + 0.30 * reg_high + 0.20 * darvas_high + 0.10 * atr_high)
    # Clamp: ensure blended targets never cross spot (no inverted range)
    expected_high = max(expected_high, spot_price)
    expected_low  = min(expected_low,  spot_price)
    
    # Expected 40-day path classification
    if "Strong Bull" in trend_state and regime_state == "Trending":
        path = "directional_bullish"
    elif ("Bullish Pullback" in trend_state) or (regime_state == "Mean Reverting" and "Bull" in trend_state):
        path = "mean_reversion_up"
    elif "Bull" in trend_state and regime_state == "Compressed":
        path = "sideways_bullish"
    elif "Bear" in trend_state and regime_state == "Trending":
        path = "directional_bearish"
    elif "Bear" in trend_state and regime_state in ("Mean Reverting", "Compressed"):
        path = "sideways_bearish"
    elif regime_state == "Volatile":
        path = "volatile_expansion"
    elif trend_state in ("Sideways / Range", "Transition / Mixed") or regime_state == "Compressed":
        path = "sideways_range"
    else:
        path = "unclear"
        
    return path, expected_low, expected_high
