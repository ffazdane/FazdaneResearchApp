# Ticker Technical Analysis Documentation

Based on a review of the application's modules across the different tiers, here is a comprehensive breakdown of the technical analysis indicators and methods being used for evaluating a ticker.

## 1. Moving Averages & Trend
*   **Simple Moving Averages (SMA):** Primarily 20-day, 50-day, and 200-day SMAs. The system heavily uses these to determine market regime, trend direction (e.g., 20/50 SMA relationship), and distances from these SMAs.
*   **Exponential Moving Averages (EMA):** 9, 20, 21, 50, and 200 EMAs. These are used for trend classification, support/resistance in pullback strategies, and EMA crossovers.
*   **Triple EMA (TEMA):** Used alongside Heikin-Ashi price action to calculate the proprietary **FDTS** (FazDane Trade Signal) deviation and trend score.

## 2. Momentum & Oscillators
*   **Relative Strength Index (RSI):** The standard 14-period RSI is used pervasively across almost all modules (Tier 1 through Tier 3) to measure momentum and overbought/oversold conditions.
*   **Moving Average Convergence Divergence (MACD):** Standard MACD (12, 26, 9) including the MACD Line, Signal Line, and Histogram. It is often combined with the FDTS signal (e.g., "FDTS + MACD Trade Signal").
*   **Average Directional Index (ADX):** 14-period ADX along with +DI and -DI directional indicators. Used heavily in the Calendar Scoring and Elliott Wave modules to quantify trend strength versus range-bound/consolidating environments.

## 3. Volatility & Price Channels
*   **Average True Range (ATR):** The 14-period ATR is utilized for volatility normalization, expected move calculations, ATR-scaled pivot points (for Elliott Wave swing detection), and risk projections.
*   **Bollinger Bands & Squeeze:** Standard Bollinger Bands are calculated to detect volatility compression ("Bollinger Squeeze") in the Universe Intelligence module.
*   **Darvas Box:** Upper and lower Darvas band calculations are used in the Trade Recommendation engine for breakout setups.
*   **Linear Regression Channels:** Used in the Trade Recommendation engine to define invalidation rules and projected ranges alongside ATR.

## 4. Volume & Support/Resistance
*   **Volume Weighted Average Price (VWAP):** A cumulative typical-price VWAP is used extensively for intraday/daily support levels, trend deviation, and as key KPIs.
*   **Anchored VWAP (AVWAP):** Utilized specifically in Tier 1's Ticker Pullback Strategy as a dynamic support level.
*   **Cumulative Volume Delta (CVD):** Calculated in the Universe Intelligence and Price Action modules to assess buying/selling pressure inside the volume.
*   **Ichimoku Cloud:** Used in the Market Regime and Trade Recommendation modules to define structural support and resistance levels.

## Summary by Module Tiers
*   **Tier 1 (Scans & Liquidity):** Focuses on EMA pullbacks (9, 21, AVWAP), RSI, ADX, and the proprietary FDTS+MACD signals for setups.
*   **Tier 2 (Regime & Universe Intel):** Focuses heavily on SMA/EMA cross relations, Bollinger Squeezes, CVD, Ichimoku support, and clustering stocks by RSI and Beta.
*   **Tier 3 (Elliott Wave):** Focuses on adaptive ATR-scaled pivots for swing detection and uses ADX/RSI to filter consolidating vs. trending waves.
*   **Tier 4 (Volatility Engine):** Focuses on deviations from short-term EMAs (20/50) and Expected Move calculations based on Implied Volatility.
