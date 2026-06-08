import numpy as np
import pandas as pd
import logging
import yfinance as yf
from datetime import datetime, timedelta

logger = logging.getLogger("CalendarScoringEngine")

# ══════════════════════════════════════════════════════════════════════
# OPTIONAL SKLEARN IMPORTS (graceful fallback if not yet installed)
# ══════════════════════════════════════════════════════════════════════

try:
    from sklearn.decomposition import PCA as _PCA
    from sklearn.cluster import KMeans as _KMeans
    from sklearn.preprocessing import StandardScaler as _StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn not installed. PCA and K-Means scoring will use enhanced "
        "rule-based fallbacks. Run: pip install scikit-learn"
    )

# ══════════════════════════════════════════════════════════════════════
# COMPONENT SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def calculate_trend_score(spot_price: float, ema_20: float, ema_50: float, ema_200: float, adx_14: float) -> float:
    """Evaluate trend strength and alignment (Preferred: Uptrend)."""
    score = 50.0

    # Check EMA alignment
    if spot_price > ema_20 > ema_50 > ema_200:
        score = 85.0
        # ADX trend strength bonus
        if adx_14 > 25.0:
            score += 10.0
        elif adx_14 < 18.0:
            score -= 5.0
    elif spot_price > ema_50 > ema_200:
        score = 75.0
        if adx_14 > 20.0:
            score += 5.0
    elif spot_price > ema_200:
        score = 60.0
    else:  # below 200 EMA
        score = 30.0

    return min(100.0, max(0.0, score))


def calculate_option_structure_score(front_iv: float, back_iv: float) -> float:
    """Evaluate volatility structure (Preferred: back_iv > front_iv - stable term structure)."""
    iv_diff = back_iv - front_iv

    # Contango is good for calendars: we buy back_iv, sell front_iv.
    # Normal contango term structure (back_iv > front_iv) gets high score
    if iv_diff > 0.05:
        score = 95.0
    elif iv_diff > 0.0:
        score = 85.0
    elif iv_diff > -0.03:
        score = 65.0
    else:  # Backwardation (front_iv significantly higher than back_iv) is risky
        score = 35.0

    return score


def calculate_volatility_score(iv_rank: float, iv_percentile: float) -> float:
    """Evaluate IV levels (Preferred: low to moderate IV rank to avoid collapse)."""
    # Ideal IV Rank is low-to-mid range (e.g. 15 to 55) for stable/expanding IV potential.
    # High IV rank (>70-80) carries crush risk.
    if 15.0 <= iv_rank <= 55.0:
        score = 90.0
    elif iv_rank < 15.0:
        score = 75.0  # low but stable
    elif iv_rank <= 75.0:
        score = 60.0  # elevated
    else:
        score = 30.0  # very high crush risk

    # blend in percentile
    final_score = (score * 0.7) + (iv_percentile * 0.3)

    # Apply penalty if the broad Volatility Engine detects HIGH or EXTREME fragility risk
    try:
        from modules.tier4.volatility_risk_api import get_current_volatility_risk
        vol_risk = get_current_volatility_risk()
        if vol_risk:
            regime = vol_risk.get("risk_regime", "LOW")
            if regime == "EXTREME":
                final_score -= 25.0
            elif regime == "HIGH":
                final_score -= 15.0
    except Exception:
        pass

    return min(100.0, max(0.0, final_score))


def calculate_fdts_score(fdts_signal_score: float) -> float:
    """Score matching the custom FDTS signal value."""
    return fdts_signal_score


# ══════════════════════════════════════════════════════════════════════
# PCA RELATIVE STRENGTH SCORE
# ══════════════════════════════════════════════════════════════════════

def calculate_pca_score(
    ticker: str,
    df_history: pd.DataFrame,
    benchmark_returns: pd.DataFrame = None
) -> float:
    """
    Calculate PCA Relative Strength Score.

    When scikit-learn is available and benchmark returns are provided, this fits a
    2-component PCA on the benchmark factor returns (SPY + QQQ), extracts the
    first principal component (market factor), then computes the ticker's:
      - Beta to PC1 (market sensitivity)
      - Residual Alpha (excess return unexplained by PC1)
      - Alpha Sharpe ratio (risk-adjusted alpha)

    Tickers with strong, consistent positive alpha relative to the market factor
    receive high scores — these are genuine relative-strength leaders.

    Falls back to an enhanced multi-timeframe Sharpe momentum proxy if sklearn or
    benchmark data are unavailable.

    Args:
        ticker: Ticker symbol (used for logging).
        df_history: 1-year price history DataFrame from fetch_technical_data().
        benchmark_returns: Optional DataFrame with columns ['SPY', 'QQQ'] of daily
                           returns. Pre-fetch with fetch_benchmark_data() for speed.

    Returns:
        float: Score in range [25, 98].
    """
    if df_history.empty or len(df_history) < 40:
        return 70.0

    try:
        ticker_returns = df_history['Close'].pct_change().dropna()

        # ── Real PCA path (sklearn + benchmark data available) ──────────────
        if _SKLEARN_AVAILABLE and benchmark_returns is not None and not benchmark_returns.empty:
            combined = pd.concat(
                [ticker_returns.rename("ticker"), benchmark_returns], axis=1
            ).dropna().tail(60)

            if len(combined) >= 30:
                factor_cols = [c for c in benchmark_returns.columns if c in combined.columns]
                factor_data = combined[factor_cols].values

                # Fit PCA on benchmark factors
                n_comp = min(2, len(factor_cols))
                pca = _PCA(n_components=n_comp)
                pca.fit(factor_data)
                pc1 = pca.components_[0]  # weights of PC1

                # Reconstruct PC1 factor time series
                pc1_returns = pd.Series(factor_data @ pc1, index=combined.index)

                # Beta via covariance
                cov_matrix = np.cov(combined['ticker'].values, pc1_returns.values)
                beta_to_pc1 = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-10)

                # Alpha = ticker return minus beta × market factor
                alpha_series = combined['ticker'] - beta_to_pc1 * pc1_returns
                alpha_mean = alpha_series.mean()
                alpha_std = alpha_series.std()
                alpha_sharpe = alpha_mean / (alpha_std + 1e-8) * np.sqrt(252)

                # Map alpha_sharpe [-3, +3] → score [0, 100], centred at 50
                score = 50.0 + (alpha_sharpe * 15.0)

                # Market leadership bonus/penalty
                if beta_to_pc1 > 1.0 and alpha_mean > 0:
                    score += 8.0   # High-beta leader — premium position
                elif beta_to_pc1 < 0.3:
                    score -= 8.0   # Decorrelated from market factor

                return float(np.clip(score, 25.0, 98.0))

        # ── Enhanced fallback: multi-timeframe Sharpe momentum proxy ────────
        if len(ticker_returns) < 20:
            return 70.0

        r60 = ticker_returns.tail(60)
        r20 = ticker_returns.tail(20)

        mom_20 = r20.mean()
        mom_60 = r60.mean()
        vol_20 = r20.std()
        vol_60 = r60.std()

        sharpe_20 = mom_20 / (vol_20 + 1e-8) * np.sqrt(252)
        sharpe_60 = mom_60 / (vol_60 + 1e-8) * np.sqrt(252)

        composite_sharpe = sharpe_20 * 0.60 + sharpe_60 * 0.40
        score = 50.0 + (composite_sharpe * 15.0)
        return float(np.clip(score, 30.0, 95.0))

    except Exception as e:
        logger.warning(f"PCA score calculation failed for {ticker}: {e}")
        return 72.0


# ══════════════════════════════════════════════════════════════════════
# CLUSTER CLASSIFICATION SCORE
# ══════════════════════════════════════════════════════════════════════

def calculate_cluster_score(ticker: str, df_history: pd.DataFrame) -> tuple[float, str]:
    """
    Classify the ticker into a market phase cluster using rules-based heuristics
    on rolling technical features (distance from EMAs, momentum, and ADX).

    This is the Phase 1 rules-based implementation. The real K-Means machine
    learning clustering model is deferred to Phase 2.

    Returns:
        tuple[float, str]: (score, label) where label ∈
            {"Early Trend", "Mid Trend", "Consolidating", "Overextended"}
    """
    if df_history.empty or len(df_history) < 20:
        return 70.0, "Early Trend"

    try:
        close = df_history['Close']
        returns = close.pct_change()

        # Calculate EMA distance & momentum
        ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

        dist_20 = (close.iloc[-1] - ema_20) / (ema_20 + 1e-8)
        dist_50 = (close.iloc[-1] - ema_50) / (ema_50 + 1e-8)
        momentum_20 = (close.iloc[-1] / (close.iloc[-20] if len(close) >= 20 else close.iloc[0])) - 1.0

        # ADX trend strength proxy
        adx_proxy = returns.abs().rolling(14).mean().iloc[-1] / (returns.rolling(14).std().iloc[-1] + 1e-8)

        # Heuristic rules to assign cluster labels
        if dist_50 > 0.08 or (dist_50 > 0.05 and momentum_20 > 0.08):
            label = "Overextended"
        elif momentum_20 > 0.025 and adx_proxy > 0.6 and dist_50 > 0.01:
            label = "Mid Trend"
        elif momentum_20 > 0.005 and dist_50 >= 0.0:
            label = "Early Trend"
        else:
            label = "Consolidating"

        score_map = {
            "Early Trend":   95.0,
            "Mid Trend":     90.0,
            "Consolidating": 75.0,
            "Overextended":  60.0,
        }
        return score_map.get(label, 75.0), label

    except Exception as e:
        logger.warning(f"Cluster score failed for {ticker}: {e}")
        return 70.0, "Early Trend"


# ══════════════════════════════════════════════════════════════════════
# LEADING / LAGGING RELATIVE STRENGTH SCORE
# ══════════════════════════════════════════════════════════════════════

def calculate_leading_lagging_score(
    ticker: str,
    df_history: pd.DataFrame,
    spy_df: pd.DataFrame = None
) -> tuple[float, str]:
    """
    Multi-timeframe relative strength vs SPY benchmark.

    Computes the ticker's excess return over SPY across three horizons:
      - 5-day  (20% weight) — short-term price leadership
      - 20-day (40% weight) — intermediate trend leadership
      - 60-day (40% weight) — structural relative strength

    A consistency bonus/penalty is applied when all three timeframes
    agree on direction (all outperforming or all underperforming SPY).

    Args:
        ticker: Ticker symbol (used for logging).
        df_history: 1-year price history from fetch_technical_data().
        spy_df: Optional pre-fetched SPY history DataFrame (avoids repeated
                API calls when scanning multiple tickers). Falls back to a
                live yfinance fetch if not provided.

    Returns:
        tuple[float, str]: (score, state) where state ∈
            {"Strong Leader", "Leading", "Neutral", "Lagging", "Strong Lagger"}
    """
    if df_history.empty:
        return 75.0, "Leading"

    try:
        close = df_history['Close']

        # Resolve SPY reference
        spy_close = None
        if spy_df is not None and not spy_df.empty:
            spy_close = spy_df['Close']
        else:
            try:
                spy_raw = yf.Ticker("SPY").history(period="1y")
                if not spy_raw.empty:
                    spy_close = spy_raw['Close']
            except Exception:
                pass

        # Compute excess returns across three timeframes
        timeframes = [(5, 0.20), (20, 0.40), (60, 0.40)]
        diffs = {}

        for days, weight in timeframes:
            if len(close) > days:
                ticker_ret = (close.iloc[-1] / close.iloc[-days]) - 1.0
                if spy_close is not None and len(spy_close) > days:
                    spy_ret = (spy_close.iloc[-1] / spy_close.iloc[-days]) - 1.0
                    diffs[days] = (ticker_ret - spy_ret, weight)
                else:
                    # No SPY reference: use raw return, offset by a modest neutral proxy
                    diffs[days] = (ticker_ret - 0.01, weight)
            else:
                diffs[days] = (0.0, weight)

        # Weighted composite excess return
        composite = sum(diff * w for diff, w in diffs.values())

        # Consistency flags
        all_positive = all(diff > 0.0 for diff, _ in diffs.values())
        all_negative = all(diff < 0.0 for diff, _ in diffs.values())

        # State classification
        if composite >= 0.05:
            state, score = "Strong Leader", 95.0
        elif composite >= 0.01:
            state, score = "Leading", 85.0
        elif composite >= -0.01:
            state, score = "Neutral", 70.0
        elif composite >= -0.04:
            state, score = "Lagging", 55.0
        else:
            state, score = "Strong Lagger", 35.0

        # ±5 consistency adjustment
        if all_positive:
            score = min(100.0, score + 5.0)
        elif all_negative:
            score = max(0.0, score - 5.0)

        return score, state

    except Exception as e:
        logger.warning(f"Leading/lagging score failed for {ticker}: {e}")
        return 75.0, "Leading"


# ══════════════════════════════════════════════════════════════════════
# LIQUIDITY SCORE
# ══════════════════════════════════════════════════════════════════════

def calculate_liquidity_score(bid_ask_spread_pct: float, avg_option_volume: float) -> float:
    """Evaluate option chain liquidity (Preferred: tight spreads and high volume)."""
    score = 50.0

    # Spread evaluation
    if bid_ask_spread_pct <= 0.01:
        score += 30.0
    elif bid_ask_spread_pct <= 0.03:
        score += 20.0
    elif bid_ask_spread_pct <= 0.07:
        score += 10.0
    else:
        score -= 10.0

    # Volume/OI evaluation
    if avg_option_volume > 1000:
        score += 20.0
    elif avg_option_volume > 200:
        score += 10.0

    return min(100.0, max(0.0, score))


# ══════════════════════════════════════════════════════════════════════
# EVENT RISK SCORE
# ══════════════════════════════════════════════════════════════════════

def calculate_event_risk_score(earnings_date_str: str, short_dte: int) -> tuple[float, int]:
    """Calculate event risk score based on earnings proximity (Preferred: earnings outside DTE)."""
    if not earnings_date_str:
        return 95.0, 0  # No earnings date found, low risk

    try:
        earn_date = datetime.strptime(earnings_date_str, "%Y-%m-%d")
        today = datetime.now()
        days_to_earn = (earn_date - today).days

        # If earnings occur during our option trade (especially before short leg expiry)
        if 0 <= days_to_earn <= (short_dte + 5):
            # Extremely high event risk (implied vol crush, stock gap risk)
            return 20.0, 1
        elif 0 <= days_to_earn <= 45:
            # Moderate event risk (occurs between short and long expiry)
            return 60.0, 0
        else:
            return 95.0, 0
    except Exception:
        return 90.0, 0


# ══════════════════════════════════════════════════════════════════════
# INSTITUTIONAL FLOW SCORE
# ══════════════════════════════════════════════════════════════════════

def calculate_institutional_flow_score(
    ticker: str,
    option_chain_data: dict = None
) -> float:
    """
    Estimate institutional options flow activity score.

    Uses yfinance data to detect four institutional signatures:

    1. **Relative Volume Spike** — last session volume vs 3-month average.
       Institutional accumulation typically shows as 1.5–3× average volume.

    2. **Call/Put Volume Ratio** — dominant call buying vs put buying in the
       front-month options chain. High call dominance (>1.3×) signals bullish
       institutional positioning.

    3. **OI vs Volume Divergence** — when today's volume exceeds a high fraction
       of open interest, new positions are being opened (fresh institutional money).

    4. **Front-Month IV Skew** — the spread between top-volume put IV and call IV.
       Low/negative skew (call IV ≈ put IV or call > put) reflects bullish flow;
       high put skew reflects hedging or fear.

    Falls back to a muted neutral range [40–65] when live option chain data is
    unavailable (synthetic chain mode).

    Args:
        ticker: Ticker symbol.
        option_chain_data: Option chain dict from fetch_option_chain_data().
                           Used to determine the front-month expiry and whether
                           the chain is synthetic.

    Returns:
        float: Score in range [0, 100]. 50 = neutral.
    """
    score = 50.0  # Neutral baseline

    try:
        ticker_obj = yf.Ticker(ticker)

        # ── Feature 1: Relative Volume Spike ──────────────────────────────────
        try:
            fast_info = ticker_obj.fast_info
            avg_vol  = getattr(fast_info, 'three_month_average_volume', None)
            last_vol = getattr(fast_info, 'last_volume', None)

            if avg_vol and last_vol and avg_vol > 0:
                vol_ratio = last_vol / avg_vol
                if vol_ratio >= 2.0:
                    score += 20.0   # Significant institutional volume spike
                elif vol_ratio >= 1.5:
                    score += 12.0
                elif vol_ratio >= 1.2:
                    score += 6.0
                elif vol_ratio < 0.5:
                    score -= 10.0   # Unusually low activity
        except Exception:
            pass

        # ── Features 2–4: Live Options Chain Flow Analysis ────────────────────
        is_synthetic  = True
        short_expiry  = None

        if option_chain_data:
            is_synthetic = option_chain_data.get('is_synthetic', True)
            short_expiry = option_chain_data.get('short_expiry')

        if not is_synthetic and short_expiry:
            try:
                available_exps = ticker_obj.options
                if short_expiry in available_exps:
                    chain = ticker_obj.option_chain(short_expiry)
                    calls = chain.calls
                    puts  = chain.puts

                    if not calls.empty and not puts.empty:
                        # Feature 2: Call/Put Volume Ratio
                        total_call_vol = calls['volume'].fillna(0).sum()
                        total_put_vol  = puts['volume'].fillna(0).sum()

                        if total_put_vol > 10 and total_call_vol > 10:
                            cp_ratio = total_call_vol / total_put_vol
                            if cp_ratio >= 2.0:
                                score += 18.0   # Dominant bullish call flow
                            elif cp_ratio >= 1.3:
                                score += 10.0
                            elif cp_ratio <= 0.5:
                                score -= 20.0   # Strong put dominance = bearish
                            elif cp_ratio < 0.7:
                                score -= 12.0

                        # Feature 3: OI vs Volume divergence (fresh positioning)
                        total_call_oi = calls['openInterest'].fillna(0).sum()
                        if total_call_oi > 0 and total_call_vol > 0:
                            fresh_ratio = total_call_vol / (total_call_oi + 1.0)
                            if fresh_ratio > 0.25:   # >25% of OI = new positions today
                                score += 8.0
                            elif fresh_ratio > 0.10:
                                score += 4.0

                        # Feature 4: Front-month IV skew
                        if 'impliedVolatility' in calls.columns and 'impliedVolatility' in puts.columns:
                            top_calls = calls[calls['volume'].fillna(0) > 0].sort_values(
                                'volume', ascending=False
                            ).head(5)
                            top_puts = puts[puts['volume'].fillna(0) > 0].sort_values(
                                'volume', ascending=False
                            ).head(5)

                            if not top_calls.empty and not top_puts.empty:
                                call_iv = top_calls['impliedVolatility'].mean()
                                put_iv  = top_puts['impliedVolatility'].mean()

                                if call_iv > 0 and put_iv > 0:
                                    skew = put_iv - call_iv
                                    if skew < 0.01:     # Tight or inverted skew = bullish
                                        score += 8.0
                                    elif skew > 0.10:   # High put premium = hedging/fear
                                        score -= 12.0
                                    elif skew > 0.06:
                                        score -= 6.0

            except Exception as e:
                logger.debug(f"Options chain flow analysis skipped for {ticker}: {e}")

        elif is_synthetic:
            # Synthetic chain: no live flow signal — clamp to muted neutral range
            score = float(np.clip(score, 40.0, 65.0))

    except Exception as e:
        logger.warning(f"Institutional flow score error for {ticker}: {e}")

    return float(np.clip(score, 0.0, 100.0))


# ══════════════════════════════════════════════════════════════════════
# HARD FILTERS
# ══════════════════════════════════════════════════════════════════════

def apply_hard_filters(ticker: str, tech_data: dict, option_setup: dict, fdts_signal: str) -> list[str]:
    """Verify ticker eligibility against hard criteria. Returns reasons for exclusion, if any."""
    exclusions = []

    # 1. FDTS == Sell
    if fdts_signal == "Sell":
        exclusions.append("FDTS Signal is Sell")

    # 2. Bid/Ask spread > 7%
    spread = option_setup.get("bid_ask_spread_pct", 1.0)
    if spread > 0.07:
        exclusions.append(f"Bid/Ask Spread too wide ({spread*100:.1f}%)")

    # 3. Average option volume too low
    vol = option_setup.get("avg_option_volume", 0)
    if vol < 5:
        exclusions.append(f"Low Option Volume ({vol:.0f})")

    # 4. Open interest too low
    oi = option_setup.get("avg_open_interest", 0)
    if oi < 50:
        exclusions.append(f"Low Open Interest ({oi:.0f})")

    # 5. Earnings inside trade window (before short-leg expiration)
    earn_date_str = tech_data.get("earnings_date")
    if earn_date_str:
        try:
            earn_date = datetime.strptime(earn_date_str, "%Y-%m-%d")
            days_to_earn = (earn_date - datetime.now()).days
            if 0 <= days_to_earn <= option_setup.get("short_dte", 20):
                exclusions.append(f"Earnings Inside Trade Window ({days_to_earn} days to earnings)")
        except Exception:
            pass

    # 6. IV Rank > 80
    iv_rank = tech_data.get("iv_rank", 0.0)
    if iv_rank > 80.0:
        exclusions.append(f"IV Rank too high ({iv_rank:.1f})")

    # 7. ADX too weak
    adx = tech_data.get("adx_14", 20.0)
    if adx < 12.0:
        exclusions.append(f"Trend ADX too weak ({adx:.1f})")

    # 8. Price below 50 EMA and 200 EMA
    spot   = tech_data.get("spot_price", 0.0)
    ema_50 = tech_data.get("ema_50", 0.0)
    ema_200 = tech_data.get("ema_200", 0.0)
    if spot < ema_50 and spot < ema_200:
        exclusions.append("Price below both 50 EMA and 200 EMA")

    # 9. Missing back/front expiry or legs
    if not option_setup.get("short_expiry") or not option_setup.get("long_expiry"):
        exclusions.append("Missing front or back month option legs")

    return exclusions
