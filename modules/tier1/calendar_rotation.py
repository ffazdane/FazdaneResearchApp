"""
FazDane Analytics — Tier 1
Calendar Option Strategy Rotation Matrix
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import yfinance as yf
import sqlite3
from datetime import datetime, date, timedelta
import logging
from modules.base_module import FazDaneModule
from utils.universe_manager import render_universe_manager
from utils.persistence import get_db_path
from utils.formatting import calculate_strength_pct, format_strength_meter, format_strength_meter_html
from modules.calendar_scoring.database import insert_decision_log, insert_option_setup
from modules.calendar_scoring.data_loader import fetch_option_chain_data, black_scholes_call
from modules.calendar_scoring.trade_setup_engine import select_calendar_setup
from modules.calendar_scoring.config import MODEL_VERSION

logger = logging.getLogger("CalendarRotation")

# Global Parameters
LOOKBACK_DAYS = 90
TRAIL_DAYS = 20
PLOT_TOP_N = 18
TRAIL_SMOOTH_WINDOW = 4

UNIVERSES = {
    "Calendar Candidates": {
        "tickers": {
            "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF", "IWM": "Russell 2000 ETF",
            "DIA": "Dow Jones ETF", "GLD": "Gold ETF", "NVDA": "Nvidia",
            "TSLA": "Tesla", "AAPL": "Apple", "MSFT": "Microsoft",
            "AMZN": "Amazon", "META": "Meta Platforms", "GOOGL": "Alphabet",
            "AVGO": "Broadcom", "AMD": "Advanced Micro Devices", "NFLX": "Netflix",
            "INTC": "Intel", "QCOM": "Qualcomm", "CSCO": "Cisco",
            "AMAT": "Applied Materials", "COIN": "Coinbase", "HOOD": "Robinhood",
            "PLTR": "Palantir", "IBM": "IBM", "CRM": "Salesforce",
            "ADBE": "Adobe", "ORCL": "Oracle", "CRWD": "CrowdStrike",
            "JPM": "JPMorgan Chase", "GS": "Goldman Sachs", "UNH": "UnitedHealth",
            "LLY": "Eli Lilly", "COST": "Costco", "HD": "Home Depot",
            "BA": "Boeing", "CAT": "Caterpillar"
        },
        "benchmark": "SPY",
    },
    "SPX Sectors": {
        "tickers": {
            "XLC": "Communication Services", "XLY": "Consumer Discretionary",
            "XLP": "Consumer Staples", "XLE": "Energy", "XLF": "Financials",
            "XLV": "Health Care", "XLI": "Industrials", "XLB": "Materials",
            "XLRE": "Real Estate", "XLK": "Technology", "XLU": "Utilities"
        },
        "benchmark": "SPY",
    },
    "MAG 7": {
        "tickers": {
            "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia",
            "AMZN": "Amazon", "META": "Meta Platforms", "GOOGL": "Alphabet",
            "TSLA": "Tesla"
        },
        "benchmark": "QQQ",
    },
    "Leading ETFs": {
        "tickers": {
            "QQQ": "Nasdaq 100 ETF", "SPY": "S&P 500 ETF", "IWM": "Russell 2000 ETF",
            "DIA": "Dow Jones ETF", "SMH": "Semiconductor ETF", "XLK": "Technology ETF",
            "XLF": "Financial ETF", "XLE": "Energy ETF", "GLD": "Gold ETF",
            "SLV": "Silver ETF", "TLT": "Long Bond ETF", "HYG": "High Yield Bond ETF"
        },
        "benchmark": "SPY",
    },
    "Custom Tickers": {
        "tickers": {},
        "benchmark": "SPY",
    }
}

def configure_universe(selected_universe):
    uni = UNIVERSES[selected_universe]
    bench = uni["benchmark"]
    candidates = sorted(set([t for t in uni["tickers"].keys() if t != bench]))
    return {
        "selected_universe": selected_universe, "universe": uni, "benchmark": bench,
        "ticker_names": uni["tickers"],
        "tickers": list(uni["tickers"].keys()), "candidates": candidates,
    }

from modules.calendar_scoring.technical_indicators import (
    calculate_fdts_ha_signal,
    compute_rrg_zscore,
    calculate_atr
)

def calculate_fdts_signal(symbol: str, ticker_df: pd.DataFrame, period: int = 20) -> str:
    """Calculate the FDTS + MACD Trade Signal (Buy/No Trade/Sell)."""
    return calculate_fdts_ha_signal(ticker_df, period)

def extract_ticker_df(raw, symbol):
    if raw.empty:
        return pd.DataFrame()
    ticker_df = pd.DataFrame(index=raw.index)
    if isinstance(raw.columns, pd.MultiIndex):
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in raw and symbol in raw[col].columns:
                ticker_df[col] = raw[col][symbol]
    else:
        # If single symbol was downloaded
        ticker_df = raw.copy()
        if "Adj Close" in ticker_df.columns and "Close" not in ticker_df.columns:
            ticker_df["Close"] = ticker_df["Adj Close"]
    return ticker_df.dropna(how="all")

def download_price_data(tickers, benchmark, period="6mo"):
    symbols = sorted(set(list(tickers) + [benchmark]))
    raw = yf.download(symbols, period=period, auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": symbols[0]})
    return close.dropna(how="all"), volume.reindex(close.index).fillna(0), raw

def compute_rotation(close, benchmark, trail_days=TRAIL_DAYS):
    return compute_rrg_zscore(close, benchmark, lookback_days=LOOKBACK_DAYS, trail_days=trail_days)

def compute_price_features(close, volume, benchmark):
    features = []
    bench_ret_20 = close[benchmark].pct_change(20).iloc[-1] if benchmark in close else np.nan
    for ticker in close.columns:
        if ticker == benchmark:
            continue
        s = close[ticker].dropna()
        if len(s) < 50:
            continue
        v = volume[ticker].reindex(s.index).fillna(0)
        vol_ratio = v.iloc[-1] / max(v.rolling(20).mean().iloc[-1], 1)
        true_range = pd.concat([s.diff().abs(), (s - s.shift()).abs()], axis=1).max(axis=1)
        atr20 = true_range.rolling(20).mean().iloc[-1]
        trend_score = 0
        trend_score += 20 if s.iloc[-1] > s.ewm(span=8).mean().iloc[-1] else 0
        trend_score += 20 if s.ewm(span=8).mean().iloc[-1] > s.ewm(span=21).mean().iloc[-1] else 0
        trend_score += 15 if s.ewm(span=21).mean().iloc[-1] > s.ewm(span=21).mean().iloc[-6] else 0
        trend_score += 15 if s.iloc[-1] > s.rolling(50).mean().iloc[-1] else 0
        trend_score += 15 if s.iloc[-1] >= 0.97 * s.rolling(20).max().iloc[-1] else 0
        v_last = float(v.iloc[-1]) if not v.empty else 0
        trend_score += 15 if vol_ratio >= 1.2 else 0
        strength_pct = calculate_strength_pct(s)
        features.append({
            "ticker": ticker, "spot": float(s.iloc[-1]), "atr20": float(atr20),
            "option_oi": int(max(v_last * 0.005, 500)),
            "option_volume": int(max(v_last * 0.001, 100)),
            "trend_score": min(float(trend_score), 100.0),
            "rel_strength_20": float(s.pct_change(20).iloc[-1] - bench_ret_20),
            "strength_pct": strength_pct
        })
    return pd.DataFrame(features)

def add_scores(df, rotation_latest):
    if df.empty or rotation_latest.empty: return pd.DataFrame()
    df = df.merge(rotation_latest[["ticker", "rs_ratio", "rs_momentum"]], on="ticker", how="left")
    df["option_liquidity_score"] = np.clip(60 + (df["option_volume"] / df["option_volume"].max()) * 40, 60, 100).fillna(60)
    df["calendar_score"] = (0.4 * df["trend_score"] + 0.3 * df["rs_ratio"] + 0.3 * df["rs_momentum"]).clip(0, 100)
    df["target_strike"] = (df["spot"] * 1.03).round(1)
    df["distance_atr"] = 1.1
    df["stage"] = "Stage 3 Active"
    df["quality"] = np.where(df["calendar_score"] >= 75, "Best", "Watch")
    return df.sort_values("calendar_score", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_universe(univ_name, tickers, benchmark):
    ticker_list = list(tickers)
    candidates = sorted(set([t for t in ticker_list if t != benchmark]))
    ctx = {
        "selected_universe": univ_name,
        "benchmark": benchmark,
        "ticker_names": {t: t for t in ticker_list},
        "tickers": ticker_list,
        "candidates": candidates,
    }
    close, volume, raw = download_price_data(ctx["candidates"], ctx["benchmark"])
    if close.empty: return None
    rotation = compute_rotation(close, ctx["benchmark"])
    price_feats = compute_price_features(close, volume, ctx["benchmark"])
    if rotation.empty or price_feats.empty: return None
    latest_rot = rotation.sort_values("date").groupby("ticker").tail(1)
    
    # Calculate FDTS signals for all candidates
    fdts_signals = {}
    for ticker in ctx["candidates"]:
        ticker_df = extract_ticker_df(raw, ticker)
        sig = calculate_fdts_signal(ticker, ticker_df)
        fdts_signals[ticker] = {"Buy": "🟢 Buy", "Sell": "🔴 Sell", "No Trade": "⚪ No Trade"}.get(sig, "⚪ No Trade")

    final_scores = add_scores(price_feats, latest_rot)
    final_scores["fdts_signal"] = final_scores["ticker"].map(fdts_signals).fillna("⚪ No Trade")
    final_scores["universe"] = univ_name
    return {
        "context": ctx, "rotation": rotation, "scores": final_scores,
        "close": close, "volume": volume
    }

def format_fdts_emoji(sig: str) -> str:
    """Format FDTS signals to standard visual emoji bullet points."""
    sig_clean = str(sig).replace("🟢", "").replace("🔴", "").replace("⚪", "").strip()
    return {"Buy": "🟢 Buy", "Sell": "🔴 Sell", "Neutral": "⚪ Neutral", "No Trade": "⚪ No Trade"}.get(sig_clean, f"⚪ {sig_clean}")

@st.cache_data(ttl=1800, show_spinner=False)
def compute_live_market_metrics(tickers: tuple) -> dict[str, dict]:
    """
    Download OHLC data for all tickers in one batch and compute fresh FDTS signals,
    15-day price strength percentages, current price, net change, and ATR.
    Cached for 30 minutes. Always reflects the current market state.
    Returns dict: ticker -> {
        "fdts_signal": str, "strength_pct": float | None,
        "spot": float | None, "net_change_val": float | None,
        "net_change_pct": float | None, "atr": float | None
    }
    """
    metrics = {}
    if not tickers:
        return metrics
    try:
        raw = yf.download(
            list(tickers), period="6mo", auto_adjust=True,
            progress=False, threads=True
        )
        if raw.empty:
            return metrics
        for ticker in tickers:
            try:
                ticker_df = extract_ticker_df(raw, ticker)
                if ticker_df.empty:
                    metrics[ticker] = {
                        "fdts_signal": "No Trade", "strength_pct": None,
                        "spot": None, "net_change_val": None, "net_change_pct": None, "atr": None,
                        "hv_30": 0.30
                    }
                    continue
                
                sig = "No Trade"
                if len(ticker_df) >= 60:
                    sig = calculate_fdts_ha_signal(ticker_df)
                    if sig not in ("Buy", "Sell", "No Trade"):
                        sig = "No Trade"
                
                strength_pct = calculate_strength_pct(ticker_df)
                
                spot = float(ticker_df["Close"].iloc[-1])
                prev_close = float(ticker_df["Close"].iloc[-2]) if len(ticker_df) >= 2 else spot
                net_change_val = spot - prev_close
                net_change_pct = (net_change_val / prev_close) * 100 if prev_close != 0 else 0.0
                
                atr_series = calculate_atr(ticker_df["High"], ticker_df["Low"], ticker_df["Close"], period=14)
                atr = float(atr_series.iloc[-1]) if not atr_series.dropna().empty else None
                
                try:
                    hist_vol_30 = float(ticker_df['Close'].pct_change().rolling(30).std().iloc[-1] * np.sqrt(252))
                    if np.isnan(hist_vol_30) or hist_vol_30 <= 0:
                        hist_vol_30 = 0.30
                except Exception:
                    hist_vol_30 = 0.30
                
                metrics[ticker] = {
                    "fdts_signal": sig,
                    "strength_pct": strength_pct,
                    "spot": spot,
                    "net_change_val": net_change_val,
                    "net_change_pct": net_change_pct,
                    "atr": atr,
                    "hv_30": hist_vol_30
                }
            except Exception:
                metrics[ticker] = {
                    "fdts_signal": "No Trade", "strength_pct": None,
                    "spot": None, "net_change_val": None, "net_change_pct": None, "atr": None,
                    "hv_30": 0.30
                }
    except Exception as e:
        logger.warning(f"compute_live_market_metrics batch download failed: {e}")
    return metrics

def query_options_liquidity_store(tickers: list[str]) -> dict[str, dict]:
    summary = {}
    try:
        from utils.options_liquidity_store import DB_PATH as ol_db_path
        if not ol_db_path.exists():
            return {}
        with sqlite3.connect(ol_db_path) as conn:
            placeholders = ",".join("?" for _ in tickers)
            query = f"""
                SELECT symbol, total_volume, total_open_interest, avg_iv_pct, median_spread_pct,
                       contract_count, call_volume, put_volume
                FROM ol_symbol_snapshot_summary t1
                WHERE symbol IN ({placeholders})
                  AND scan_ts = (
                      SELECT MAX(scan_ts)
                      FROM ol_symbol_snapshot_summary t2
                      WHERE t2.symbol = t1.symbol
                  )
            """
            rows = conn.execute(query, tickers).fetchall()
            for r in rows:
                summary[r[0]] = {
                    "total_volume": r[1],
                    "total_oi": r[2],
                    "avg_iv": r[3],
                    "median_spread_pct": r[4],
                    "contract_count": r[5],
                    "call_volume": r[6] or 0.0,
                    "put_volume": r[7] or 0.0,
                }
    except Exception as e:
        logger.warning(f"Could not read options liquidity database: {e}")
    return summary

def query_earnings_calendar_store(tickers: list[str]) -> dict[str, str]:
    dates = {}
    try:
        from utils.earnings_calendar_store import DB_PATH as ec_db_path
        if not ec_db_path.exists():
            return {}
        with sqlite3.connect(ec_db_path) as conn:
            placeholders = ",".join("?" for _ in tickers)
            today_str = datetime.today().strftime("%Y-%m-%d")
            query = f"""
                SELECT ticker, MIN(date)
                FROM ec_earnings_events
                WHERE ticker IN ({placeholders}) AND date >= ?
                GROUP BY ticker
            """
            rows = conn.execute(query, [*tickers, today_str]).fetchall()
            for r in rows:
                dates[r[0]] = r[1]
    except Exception as e:
        logger.warning(f"Could not read earnings calendar database: {e}")
    return dates

def load_consolidated_recommendations(tickers: list[str]) -> pd.DataFrame:
    """Query SQLite databases to construct the consolidated ticker matrix."""
    cs_db = get_db_path("calendar_scoring")
    pa_db = get_db_path("price_action_story")
    
    tickers_upper = [t.strip().upper() for t in tickers]
    merged = pd.DataFrame({"ticker": tickers_upper})
    
    df_cs = pd.DataFrame(columns=[
        "ticker", "earnings_date", "fdts_signal", "cs_rec", "cs_score", "cs_reason",
        "trend_score", "option_structure_score", "volatility_score", "pca_score",
        "cluster_score", "leading_lagging_score", "liquidity_score", "event_risk_score",
        "price_at_decision", "atr_14", "rsi_14", "adx_14", "ema_20", "ema_50", "ema_200",
        "iv_rank", "iv_percentile", "event_risk_flag"
    ])
    df_mre = pd.DataFrame(columns=["ticker", "mre_rec", "mre_sig", "current_state", "bear_prob_1d", "stickiness_score", "mre_fdts", "realized_vol"])
    df_pa = pd.DataFrame(columns=["ticker", "pa_rec", "pa_score", "atr", "atr_min", "atr_max", "volume"])
    
    if cs_db.exists():
        try:
            with sqlite3.connect(cs_db) as conn:
                # Latest Calendar Scoring recommendation for each ticker
                q_cs = """
                    SELECT ticker, earnings_date, fdts_signal, recommendation as cs_rec, final_score as cs_score, reason_summary as cs_reason,
                           trend_score, option_structure_score, volatility_score, pca_score, cluster_score, leading_lagging_score,
                           liquidity_score, event_risk_score, price_at_decision, atr_14, rsi_14, adx_14, ema_20, ema_50, ema_200,
                           iv_rank, iv_percentile, event_risk_flag
                    FROM ticker_decision_log t1
                    WHERE decision_id = (
                        SELECT MAX(decision_id) FROM ticker_decision_log t2 WHERE t2.ticker = t1.ticker
                    )
                """
                df_cs = pd.read_sql_query(q_cs, conn)
                
                # Latest Regime (Markov) Forecast for each ticker along with state and realized vol
                q_mre = """
                    SELECT f.ticker, f.final_action as mre_rec, f.markov_signal as mre_sig,
                           f.current_state, f.bear_prob_1d, f.stickiness_score, f.fdts_signal as mre_fdts,
                           (SELECT close_price FROM markov_daily_state s WHERE s.ticker = f.ticker ORDER BY trade_date DESC LIMIT 1) as close_price,
                           (SELECT realized_vol_20d FROM markov_daily_state s WHERE s.ticker = f.ticker ORDER BY trade_date DESC LIMIT 1) as realized_vol
                    FROM markov_forecast f
                    WHERE f.as_of_date = (
                        SELECT MAX(as_of_date) FROM markov_forecast t2 WHERE t2.ticker = f.ticker
                    )
                """
                df_mre = pd.read_sql_query(q_mre, conn)
        except Exception as e:
            logger.error(f"Error querying calendar_scoring database: {e}")
            
    if pa_db.exists():
        try:
            with sqlite3.connect(pa_db) as conn:
                # Latest Price Action Stage for each ticker along with rolling min/max ATR and volume
                q_pa = """
                    SELECT t1.ticker, t1.stage as pa_rec, t1.health_score as pa_score, t1.atr, t1.volume,
                           (SELECT MIN(atr) FROM ticker_stage_history s2 WHERE s2.ticker = t1.ticker) as atr_min,
                           (SELECT MAX(atr) FROM ticker_stage_history s2 WHERE s2.ticker = t1.ticker) as atr_max
                    FROM ticker_stage_history t1
                    WHERE t1.scan_ts = (
                        SELECT MAX(scan_ts) FROM ticker_stage_history t2 WHERE t2.ticker = t1.ticker
                    )
                """
                df_pa = pd.read_sql_query(q_pa, conn)
        except Exception as e:
            logger.error(f"Error querying price_action_story database: {e}")
            
    for df in [df_cs, df_mre, df_pa]:
        if not df.empty and "ticker" in df.columns:
            df["ticker"] = df["ticker"].str.strip().str.upper()
            
    merged = merged.merge(df_cs, on="ticker", how="left")
    merged = merged.merge(df_mre, on="ticker", how="left")
    merged = merged.merge(df_pa, on="ticker", how="left")
    
    # Fill N/As
    merged["earnings_date"] = merged["earnings_date"].fillna("N/A")
    merged["fdts_signal"] = merged["fdts_signal"].fillna("Neutral")  # will be overridden below
    merged["cs_rec"] = merged["cs_rec"].fillna("N/A")
    merged["cs_score"] = merged["cs_score"].fillna(0.0)
    merged["cs_reason"] = merged["cs_reason"].fillna("No detailed reason logged")
    merged["mre_rec"] = merged["mre_rec"].fillna("N/A")
    merged["mre_sig"] = merged["mre_sig"].fillna(0.0)
    merged["pa_rec"] = merged["pa_rec"].fillna("N/A")
    merged["pa_score"] = merged["pa_score"].fillna(0.0)
    for col, default_val in [
        ("trend_score", 80.0), ("option_structure_score", 80.0), ("volatility_score", 80.0),
        ("pca_score", 80.0), ("cluster_score", 80.0), ("leading_lagging_score", 80.0),
        ("liquidity_score", 80.0), ("event_risk_score", 80.0), ("rsi_14", 55.0),
        ("adx_14", 22.0), ("ema_20", 0.0), ("ema_50", 0.0), ("ema_200", 0.0),
        ("iv_rank", 30.0), ("iv_percentile", 30.0), ("event_risk_flag", 0)
    ]:
        if col not in merged.columns:
            merged[col] = default_val
        else:
            merged[col] = merged[col].fillna(default_val)

    # --- LIVE Market Metrics override: always recalculate from fresh price data ---
    # The DB value can be stale (from a previous engine run). Live calculation
    # guarantees the signals shown match the actual current market state.
    live_metrics = compute_live_market_metrics(tuple(tickers_upper))
    merged["fdts_signal"] = merged["ticker"].map(
        lambda t: live_metrics.get(t, {}).get("fdts_signal", "No Trade")
    )
    merged["strength_pct"] = merged["ticker"].map(
        lambda t: live_metrics.get(t, {}).get("strength_pct")
    )
    
    def get_spot(row):
        t = row["ticker"]
        val = live_metrics.get(t, {}).get("spot")
        if pd.notnull(val):
            return val
        val_db = row.get("close_price")
        if pd.notnull(val_db):
            return val_db
        return None

    def get_atr(row):
        t = row["ticker"]
        val = live_metrics.get(t, {}).get("atr")
        if pd.notnull(val):
            return val
        val_db = row.get("atr")
        if pd.notnull(val_db):
            return val_db
        return None

    merged["spot"] = merged.apply(get_spot, axis=1)
    merged["net_change_val"] = merged["ticker"].map(
        lambda t: live_metrics.get(t, {}).get("net_change_val") if pd.notnull(live_metrics.get(t, {}).get("net_change_val")) else None
    )
    merged["net_change_pct"] = merged["ticker"].map(
        lambda t: live_metrics.get(t, {}).get("net_change_pct") if pd.notnull(live_metrics.get(t, {}).get("net_change_pct")) else None
    )
    merged["atr"] = merged.apply(get_atr, axis=1)
    merged["hv_30"] = merged["ticker"].map(
        lambda t: live_metrics.get(t, {}).get("hv_30", 0.30)
    )

    opt_summary = query_options_liquidity_store(tickers_upper)
    earnings_dates = query_earnings_calendar_store(tickers_upper)
    
    # Process Price Action & Markov Regime display recommendations
    pa_display_recs = []
    mre_display_recs = []
    
    atr_pct_list = []
    days_to_earnings_list = []
    liq_list = []
    spread_pct_list = []
    opt_volume_list = []
    opt_oi_list = []
    opt_call_vol_list = []
    opt_put_vol_list = []
    opt_pcr_list = []
    
    for _, row in merged.iterrows():
        # Calculate ATR pct
        atr = row.get("atr") or 0.0
        atr_min = row.get("atr_min") or 0.0
        atr_max = row.get("atr_max") or 0.0
        atr_diff = atr_max - atr_min
        atr_pct = ((atr - atr_min) / atr_diff) * 100 if atr_diff > 0 else 50.0
        atr_pct_list.append(atr_pct)
        
        # Retrieve options metrics
        sym = row["ticker"]
        opt_data = opt_summary.get(sym) or {}
        spread_pct = opt_data.get("median_spread_pct")
        if spread_pct is None:
            spread_pct = 2.5
        total_volume = opt_data.get("total_volume", 0)
        total_oi = opt_data.get("total_oi", 0)
        call_vol = opt_data.get("call_volume", 0.0) or 0.0
        put_vol = opt_data.get("put_volume", 0.0) or 0.0
        pcr = put_vol / total_volume if total_volume > 0 else 0.0
        
        if opt_data:
            liq = "High" if total_volume > 1000 else "Medium"
        else:
            liq = "High" if row.get("volume", 0) > 2000000 else "Medium" if row.get("volume", 0) > 500000 else "Low"
            
        liq_list.append(liq)
        spread_pct_list.append(spread_pct)
        opt_volume_list.append(total_volume)
        opt_oi_list.append(total_oi)
        opt_call_vol_list.append(call_vol)
        opt_put_vol_list.append(put_vol)
        opt_pcr_list.append(pcr)
            
        # Retrieve earnings
        earn_date_str = earnings_dates.get(sym, "None")
        days_to_earnings = 999
        if earn_date_str != "None":
            try:
                earn_date = datetime.strptime(earn_date_str, "%Y-%m-%d")
                days_to_earnings = (earn_date - datetime.today()).days
            except Exception:
                pass
        days_to_earnings_list.append(days_to_earnings)
        
        # Run rules
        pa_stage = row["pa_rec"]
        if pa_stage in ["Early Bull / Expansion", "Strong Bull"]:
            if atr_pct < 50 and days_to_earnings > 15 and liq != "Low" and spread_pct <= 2.5:
                pa_display = "🟢 Deploy Calendar"
            elif days_to_earnings <= 15:
                pa_display = "🟡 Watch (Earnings Risk)"
            elif atr_pct > 70:
                pa_display = "🔴 Avoid (Extended Vol)"
            else:
                pa_display = "🟢 Deploy Calendar (Watch Spread)"
        else:
            pa_display = pa_stage
            
        pa_display_recs.append(pa_display)
        
        # --- 2. Markov Regime Setup Confirmation (Calendar Setup Column) ---
        fdts_signal = row.get("mre_fdts", "Neutral")
        current_state = row.get("current_state", "SIDEWAYS")
        bear_prob_1d = row.get("bear_prob_1d", 0.0)
        stickiness = row.get("stickiness_score", 0.0)
        realized_vol = row.get("realized_vol", 0.0)
        
        is_calendar_candidate = (
            fdts_signal == "Buy" and
            current_state in ["BULL", "SIDEWAYS"] and
            bear_prob_1d < 0.30 and
            stickiness > 0.60 and
            realized_vol < 0.45
        )
        mre_display = "✅ Yes" if is_calendar_candidate else "❌ No"
        mre_display_recs.append(mre_display)
        
    merged["pa_display_rec"] = pa_display_recs
    merged["mre_display_rec"] = mre_display_recs
    merged["atr_pct"] = atr_pct_list
    merged["days_to_earnings"] = days_to_earnings_list
    merged["options_liq"] = liq_list
    merged["options_spread"] = spread_pct_list
    merged["options_volume"] = opt_volume_list
    merged["options_oi"] = opt_oi_list
    merged["options_call_volume"] = opt_call_vol_list
    merged["options_put_volume"] = opt_put_vol_list
    merged["options_pcr"] = opt_pcr_list
    
    # --- 3. Options Liquidity Put/Call Bias ---
    ol_bias_list = []
    for _, row in merged.iterrows():
        sym = row["ticker"]
        opt_data = opt_summary.get(sym) or {}
        call_vol = opt_data.get("call_volume", 0.0) or 0.0
        put_vol  = opt_data.get("put_volume",  0.0) or 0.0
        total    = call_vol + put_vol
        if total == 0:
            ol_bias_list.append("⚪ No Data")
        else:
            put_ratio = put_vol / total
            if put_ratio > 0.60:
                ol_bias_list.append("🔴 Put Heavy")
            elif put_ratio > 0.50:
                ol_bias_list.append("🟡 Slight Put")
            elif put_ratio < 0.40:
                ol_bias_list.append("🟢 Call Heavy")
            else:
                ol_bias_list.append("⚪ Balanced")
    merged["ol_bias"] = ol_bias_list
    
    # Map earnings risk category
    def get_earnings_risk(days):
        if pd.isnull(days):
            return "No Color (No Risk / > 40d)"
        try:
            d = float(days)
            if 0 <= d <= 20:
                return "🔴 Red (Earnings <= 20d)"
            elif d <= 40:
                return "🟡 Yellow (Earnings 21-40d)"
        except (ValueError, TypeError):
            pass
        return "No Color (No Risk / > 40d)"
    
    merged["earnings_risk_category"] = merged["days_to_earnings"].apply(get_earnings_risk)
    
    return merged


def build_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean, pre-formatted display DataFrame for st.dataframe.
    All values are plain text with emoji indicators for color/status cues.
    """
    rows = []
    for _, row in df.iterrows():
        ticker = row["ticker"]

        # ── Price & Net Change ──
        spot = row.get("spot")
        spot_str = f"${spot:,.2f}" if pd.notnull(spot) else "N/A"

        nc_val = row.get("net_change_val")
        nc_pct = row.get("net_change_pct")
        try:
            nc_v = float(nc_val)
            nc_p = float(nc_pct)
            if nc_v > 0:
                nc_str = f"+${abs(nc_v):,.2f} (+{abs(nc_p):.2f}%)"
            elif nc_v < 0:
                nc_str = f"-${abs(nc_v):,.2f} (-{abs(nc_p):.2f}%)"
            else:
                nc_str = "$0.00 (0.00%)"
        except (TypeError, ValueError):
            nc_str = "N/A"
            nc_v = 0

        # ── ATR ──
        atr = row.get("atr")
        atr_str = f"${atr:,.2f}" if pd.notnull(atr) else "N/A"

        # ── Strength ──
        st_val, _ = format_strength_meter(row.get("strength_pct"))

        # ── Earnings Date ──
        ed_val = row.get("earnings_date") or "N/A"
        if ed_val != "N/A":
            try:
                days_diff = (datetime.strptime(ed_val, "%Y-%m-%d").date() - datetime.now().date()).days
                if 0 <= days_diff <= 20:
                    ed_val = f"🔴 {ed_val}"
                elif days_diff <= 40:
                    ed_val = f"🟡 {ed_val}"
            except Exception:
                pass

        # ── FDTS Signal (already has emoji from format_fdts_emoji) ──
        fdts_str = format_fdts_emoji(row.get("fdts_signal", "No Trade"))

        # ── Calendar Scoring Engine ──
        cs_rec = row.get("cs_rec", "N/A")
        cs_score = row.get("cs_score", 0.0)
        _cs_icon = {"Deploy": "🟢", "Watch": "🟡", "Monitor": "🔵", "Avoid": "🔴", "Filtered": "⚫"}
        cs_str = f"{_cs_icon.get(cs_rec, '⚪')} {cs_rec}"

        # ── Price Action — plain label only; Styler background provides the colour cue ──
        pa_val = str(row.get("pa_display_rec", "N/A"))

        # ── Regime Calendar Setup ──
        mre_val = row.get("mre_display_rec", "❌ No")

        # ── Options Flow Bias ──
        ol_str = str(row.get("ol_bias", "⚪ No Data"))

        rows.append({
            "Ticker":         ticker,
            "Price":          spot_str,
            "Net Change":     nc_str,
            "ATR":            atr_str,
            "Strength":       st_val,
            "Earnings Date":  ed_val,
            "FDTS Signal":    fdts_str,
            "Cal. Score":     cs_str,
            "Price Action":   pa_val,
            "Regime Setup":   str(mre_val),
            "Options Bias":   ol_str,
            # raw for sorting
            "_nc_val":        nc_v,
        })

    return pd.DataFrame(rows)


def _apply_matrix_styles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a same-shape DataFrame of CSS style strings for each cell.
    Used by Styler.apply(axis=None) to colour the st.dataframe.
    """
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    for i in df.index:
        row = df.loc[i]

        # ── Ticker ──
        styles.at[i, "Ticker"] = "font-weight: 700; color: #60a5fa"

        # ── Net Change ── green / red text
        nc = str(row.get("Net Change", ""))
        if nc.startswith("+"):
            styles.at[i, "Net Change"] = "color: #22c55e; font-weight: 700"
        elif nc.startswith("-"):
            styles.at[i, "Net Change"] = "color: #ef4444; font-weight: 700"

        # ── Strength ── teal / red / orange, enlarged
        st_v = str(row.get("Strength", ""))
        if "▲" in st_v:
            styles.at[i, "Strength"] = "color: #00D4AA; font-weight: 700; font-size: 22px; text-align: center"
        elif "▼" in st_v:
            styles.at[i, "Strength"] = "color: #FF4B4B; font-weight: 700; font-size: 22px; text-align: center"
        elif "▶" in st_v:
            styles.at[i, "Strength"] = "color: #FFA421; font-weight: 700; font-size: 22px; text-align: center"

        # ── Earnings Date ── red / amber background
        ed = str(row.get("Earnings Date", ""))
        if "🔴" in ed:
            styles.at[i, "Earnings Date"] = (
                "background-color: rgba(220,38,38,0.28); color: #ef4444; font-weight: 700"
            )
        elif "🟡" in ed:
            styles.at[i, "Earnings Date"] = (
                "background-color: rgba(255,184,0,0.22); color: #ffb800; font-weight: 700"
            )

        # ── FDTS Signal ──
        fdts = str(row.get("FDTS Signal", ""))
        if "Buy" in fdts:
            styles.at[i, "FDTS Signal"] = "color: #22c55e; font-weight: 700"
        elif "Sell" in fdts:
            styles.at[i, "FDTS Signal"] = "color: #ef4444; font-weight: 700"
        else:
            styles.at[i, "FDTS Signal"] = "color: #94a3b8"

        # ── Calendar Scoring Engine ── coloured badge fill
        cs = str(row.get("Cal. Score", ""))
        if "🟢" in cs:
            styles.at[i, "Cal. Score"] = (
                "background-color: rgba(58,181,74,0.28); color: #3ab54a; font-weight: 700"
            )
        elif "🟡" in cs:
            styles.at[i, "Cal. Score"] = (
                "background-color: rgba(255,184,0,0.22); color: #ffb800; font-weight: 700"
            )
        elif "🔵" in cs:
            styles.at[i, "Cal. Score"] = (
                "background-color: rgba(2,132,199,0.22); color: #38bdf8"
            )
        elif "🔴" in cs:
            styles.at[i, "Cal. Score"] = (
                "background-color: rgba(220,38,38,0.22); color: #ef4444"
            )
        elif "⚫" in cs:
            styles.at[i, "Cal. Score"] = (
                "background-color: rgba(100,116,139,0.18); color: #64748b; text-decoration: line-through"
            )

        # ── Price Action — match on text since we stripped emojis from the cell ──
        pa = str(row.get("Price Action", "")).lower()
        if "deploy" in pa:
            styles.at[i, "Price Action"] = (
                "background-color: rgba(34,197,94,0.25); color: #22c55e; font-weight: 700"
            )
        elif "watch" in pa or "mature" in pa:
            styles.at[i, "Price Action"] = (
                "background-color: rgba(255,184,0,0.20); color: #ffb800; font-weight: 700"
            )
        elif "avoid" in pa or "fading" in pa or "distribution" in pa or "breakdown" in pa:
            styles.at[i, "Price Action"] = (
                "background-color: rgba(220,38,38,0.22); color: #ef4444; font-weight: 700"
            )
        elif "early" in pa or "strong" in pa:
            styles.at[i, "Price Action"] = (
                "background-color: rgba(34,197,94,0.18); color: #4ade80; font-weight: 700"
            )

        # ── Regime Setup ──
        mre = str(row.get("Regime Setup", ""))
        if "✅" in mre:
            styles.at[i, "Regime Setup"] = (
                "background-color: rgba(34,197,94,0.28); color: #22c55e; font-weight: 700"
            )
        else:
            styles.at[i, "Regime Setup"] = (
                "background-color: rgba(220,38,38,0.20); color: #f87171"
            )

        # ── Options Bias ──
        ol = str(row.get("Options Bias", ""))
        if "Put Heavy" in ol:
            styles.at[i, "Options Bias"] = (
                "background-color: rgba(220,38,38,0.28); color: #ef4444; font-weight: 700"
            )
        elif "Slight Put" in ol:
            styles.at[i, "Options Bias"] = (
                "background-color: rgba(255,184,0,0.20); color: #ffb800; font-weight: 700"
            )
        elif "Call Heavy" in ol:
            styles.at[i, "Options Bias"] = (
                "background-color: rgba(34,197,94,0.25); color: #22c55e; font-weight: 700"
            )
        elif "Balanced" in ol:
            styles.at[i, "Options Bias"] = "color: #94a3b8"

    return styles


def _render_detail_panel(row: pd.Series) -> None:
    """Renders a 4-engine diagnostic card for the selected ticker."""
    ticker = row.get("ticker", "N/A")
    spot   = row.get("spot", None)
    spot_str = f"${spot:,.2f}" if pd.notnull(spot) else "N/A"

    st.markdown(f"### 📊 {ticker} — Diagnostic Breakdown  ·  {spot_str}")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    # ── Engine 1: Calendar Scoring ──
    with c1:
        cs_rec   = row.get("cs_rec", "N/A")
        cs_score = row.get("cs_score", 0.0)
        cs_reason = row.get("cs_reason") or "No reason logged"
        _icon = {"Deploy": "🟢", "Watch": "🟡", "Monitor": "🔵", "Avoid": "🔴", "Filtered": "⚫"}
        st.markdown(f"**📅 Calendar Scoring Engine**")
        st.markdown(f"{_icon.get(cs_rec,'⚪')} **{cs_rec}** — Score: `{cs_score:.0f}/100`")
        st.caption(f"Reason: {cs_reason}")

    # ── Engine 2: Price Action ──
    with c2:
        pa_rec   = row.get("pa_rec", "N/A")
        pa_disp  = row.get("pa_display_rec", "N/A")
        pa_score = row.get("pa_score", 0.0)
        atr_pct  = row.get("atr_pct", 0.0)
        dte      = row.get("days_to_earnings", 999)
        opt_liq  = row.get("options_liq", "N/A")
        opt_sprd = row.get("options_spread", 0.0)
        st.markdown("**📈 Price Action Story Engine**")
        st.markdown(f"**{pa_disp}** — Score: `{pa_score:.0f}/100`")
        st.caption(
            f"ATR %ile: {atr_pct:.1f}%  |  "
            f"Days to Earnings: {'N/A' if dte >= 999 else f'{dte}d'}  |  "
            f"Liquidity: {opt_liq}  |  "
            f"Bid-Ask: {opt_sprd:.2f}%"
        )

    # ── Engine 3: Regime Intelligence ──
    with c3:
        mre_val  = row.get("mre_display_rec", "❌ No")
        state    = row.get("current_state", "N/A")
        bear1d   = row.get("bear_prob_1d", 0.0)
        sticky   = row.get("stickiness_score", 0.0)
        rv       = row.get("realized_vol", 0.0)
        mre_fdts = row.get("mre_fdts", "Neutral")
        st.markdown("**🧠 Regime Intelligence Dashboard**")
        st.markdown(f"**{mre_val}** — State: `{state}`")
        st.caption(
            f"Bear Prob (1d): {bear1d:.2f}  |  "
            f"Stickiness: {sticky:.2f}  |  "
            f"Realized Vol: {rv:.2f}  |  "
            f"FDTS: {mre_fdts}"
        )

    # ── Engine 4: Options Flow ──
    with c4:
        ol_bias  = row.get("ol_bias", "N/A")
        opt_vol  = row.get("options_volume", 0)
        opt_oi   = row.get("options_oi", 0)
        call_vol = row.get("options_call_volume", 0.0)
        put_vol  = row.get("options_put_volume", 0.0)
        pcr      = row.get("options_pcr", 0.0)
        st.markdown("**📊 Options Flow Bias**")
        st.markdown(f"**{ol_bias}** — PCR: `{pcr:.2f}`")
        st.caption(
            f"Total Vol: {opt_vol:,}  |  OI: {opt_oi:,}  |  "
            f"Calls: {call_vol:,.0f}  |  Puts: {put_vol:,.0f}"
        )

    st.divider()

    # ── Option Calendar Spread Leg Calculator & Deployment ──
    if spot is not None:
        st.markdown("### 🛠️ Options Look-and-Deploy Spread Setup")
        
        # Options controls
        col_ctrl1, col_ctrl2 = st.columns([1, 1])
        with col_ctrl1:
            force_synthetic = st.checkbox(
                "Force Synthetic Option Chain Fallback",
                value=False,
                help="Use generated synthetic options chains if live market data is slow or unavailable."
            )
        with col_ctrl2:
            target_delta_val = st.slider(
                "Target Short Leg Delta",
                min_value=0.10,
                max_value=0.50,
                value=0.25,
                step=0.05,
                help="The delta used to pick the short option leg. Default is 0.25 delta."
            )
            
        # Get HV30
        hv_30_val = row.get("hv_30", 0.30)
        
        # Load and select optimal legs
        with st.spinner("Calculating optimal calendar spread legs..."):
            option_chain = fetch_option_chain_data(ticker, spot, use_synthetic=force_synthetic)
            setup = select_calendar_setup(ticker, option_chain, spot, hv_30_val, target_delta=target_delta_val)
            
        if not setup:
            st.warning("⚠️ Could not load option chain or select optimal calendar legs. Check internet connectivity or toggle synthetic fallback.")
        else:
            # Let's display the legs in a clean UI
            st.markdown(f"**🎯 Strategy Selected: {setup['strategy_type']}**")
            
            c_leg1, c_leg2, c_leg3 = st.columns(3)
            with c_leg1:
                st.markdown("##### 🔴 Short Front Leg (Sell)")
                st.markdown(f"**Expiry**: `{setup['short_expiry']}` (DTE: `{setup['short_dte']}`)")
                st.markdown(f"**Strike**: `${setup['selected_strike']:.1f}`")
                st.markdown(f"**Bid/Ask**: `${setup['short_bid']:.2f}` / `${setup['short_ask']:.2f}`")
                st.markdown(f"**Mid Price**: `${setup['short_mid']:.2f}`")
            with c_leg2:
                st.markdown("##### 🟢 Long Back Leg (Buy)")
                st.markdown(f"**Expiry**: `{setup['long_expiry']}` (DTE: `{setup['long_dte']}`)")
                st.markdown(f"**Strike**: `${setup['selected_strike']:.1f}`")
                st.markdown(f"**Bid/Ask**: `${setup['long_bid']:.2f}` / `${setup['long_ask']:.2f}`")
                st.markdown(f"**Mid Price**: `${setup['long_mid']:.2f}`")
            with c_leg3:
                st.markdown("##### 💰 Spread Costs & Parameters")
                st.markdown(f"**Net Debit (Cost/Risk)**: `${setup['net_debit']:.2f}`")
                st.markdown(f"**Max Potential Loss**: `${setup['max_risk']:.2f}`")
                st.markdown(f"**Bid-Ask Spread**: `{setup['bid_ask_spread_pct']*100:.2f}%`")
                st.markdown(f"**Breakevens**: `${setup['breakeven_low']:.2f}` to `${setup['breakeven_high']:.2f}`")
                
            st.markdown("##### 📐 Consolidated Greeks")
            cg1, cg2, cg3, cg4 = st.columns(4)
            with cg1:
                st.metric("Net Delta", f"{setup['setup_delta']:.4f}")
            with cg2:
                st.metric("Net Gamma", f"{setup['setup_gamma']:.4f}")
            with cg3:
                st.metric("Net Theta", f"{setup['setup_theta']:.4f}", help="Positive is beneficial (Theta collection)")
            with cg4:
                st.metric("Net Vega", f"{setup['setup_vega']:.4f}", help="Positive is beneficial (Long Volatility)")
                
            st.divider()
            
            # Payoff chart and trade plan columns
            col_chart, col_plan = st.columns([4, 3])
            
            with col_chart:
                st.markdown("##### 📈 Spread Payoff Profile (Front-Month Expiration)")
                prices = np.linspace(spot * 0.85, spot * 1.15, 80)
                payoff = []
                r = 0.045
                for p in prices:
                    T_back_rem = (setup["long_dte"] - setup["short_dte"]) / 365.0
                    back_val, _, _, _, _ = black_scholes_call(p, setup["selected_strike"], T_back_rem, r, setup["back_iv"])
                    short_val = max(0.0, p - setup["selected_strike"])
                    spread_val = back_val - short_val
                    pnl = spread_val - setup["net_debit"]
                    payoff.append(pnl)
                    
                fig = go.Figure()
                # Draw PnL area
                fig.add_trace(go.Scatter(
                    x=prices, y=payoff, name="PnL at Front Expiry",
                    line=dict(color='#00D4AA', width=3),
                    fill='tozeroy', fillcolor='rgba(0, 212, 170, 0.08)'
                ))
                # Add spot marker line
                fig.add_vline(x=spot, line_dash="dash", line_color="#ffb800", annotation_text="Current Spot", annotation_position="top left")
                # Add break-even horizontal line
                fig.add_hline(y=0.0, line_color="#64748b", line_width=1)
                
                fig.update_layout(
                    xaxis_title="Stock Price",
                    yaxis_title="Profit / Loss ($)",
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E2E8F0"),
                    xaxis=dict(gridcolor="#1e3a5f", showgrid=True),
                    yaxis=dict(gridcolor="#1e3a5f", showgrid=True)
                )
                st.plotly_chart(fig, use_container_width=True, theme=None)
                
            with col_plan:
                st.markdown("##### 📋 Copyable Trade Plan")
                # Trade plan text block
                trade_plan_text = f"""=== FAZDANE RESEARCH SYSTEM - CALENDAR TRADE PLAN ===
Ticker: {ticker}
Current Spot: ${spot:,.2f}
Strategy: Bullish Calendar Spread
Option Structure: Sell {setup['short_expiry']} (DTE {setup['short_dte']}) ${setup['selected_strike']:.1f} Call / Buy {setup['long_expiry']} (DTE {setup['long_dte']}) ${setup['selected_strike']:.1f} Call
Target Strike: ${setup['selected_strike']:.2f}
Est. Net Debit (Max Risk): ${setup['net_debit']:.2f}
Net Greeks:
  - Delta: {setup['setup_delta']:.4f}
  - Gamma: {setup['setup_gamma']:.4f}
  - Theta: {setup['setup_theta']:.4f}
  - Vega:  {setup['setup_vega']:.4f}
Est. Breakeven Low: ${setup['breakeven_low']:.2f}
Est. Breakeven High: ${setup['breakeven_high']:.2f}
Rules:
  - Entry Trigger: Pullback confirmation near 20MA or 1H reversion to mean.
  - Invalidation: Close below support or stop loss hit.
  - Profit Target: Exit at 30% - 50% of debit paid (${setup['net_debit']*0.3:.2f} - ${setup['net_debit']*0.5:.2f} gain).
  - Stop Loss: Exit if spread value loses 35% - 50% of debit paid (${setup['net_debit']*0.35:.2f} - ${setup['net_debit']*0.5:.2f} loss).
"""
                st.code(trade_plan_text, language="markdown")
                
            st.divider()
            
            # Action button for deployment
            deploy_btn_col, _ = st.columns([2, 3])
            with deploy_btn_col:
                if st.button("🚀 Log & Deploy Calendar Spread Setup", use_container_width=True, type="primary"):
                    try:
                        # Log decision & option setup to paper database
                        now = datetime.now()
                        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                        today_str = now.strftime("%Y-%m-%d")
                        
                        # Find next earnings date if available
                        earn_date = row.get("earnings_date")
                        if earn_date == "N/A" or not earn_date:
                            earn_date = (now + timedelta(days=45)).strftime("%Y-%m-%d")
                            
                        decision_data = {
                            "decision_datetime": now_str,
                            "decision_date": today_str,
                            "ticker": ticker,
                            "strategy_type": "Bullish Calendar Spread",
                            "recommendation": "Deploy",
                            "rank_today": 1,
                            "final_score": float(row.get("cs_score", 80.0)),
                            "market_regime": row.get("current_state", "SIDEWAYS") + " State",
                            "fdts_signal": row.get("fdts_signal", "Neutral"),
                            "fdts_score": float(row.get("cs_score", 80.0)),
                            "trend_score": float(row.get("trend_score", 80.0)),
                            "option_structure_score": float(row.get("option_liquidity_score", 80.0)),
                            "volatility_score": 80.0,
                            "pca_score": 80.0,
                            "cluster_score": 80.0,
                            "leading_lagging_score": float(row.get("strength_pct", 80.0) or 80.0),
                            "liquidity_score": float(row.get("option_liquidity_score", 80.0)),
                            "event_risk_score": 80.0,
                            "institutional_flow_score": 0.0,
                            "cluster_label": row.get("pa_display_rec", "Stage 3 Active"),
                            "leading_lagging_state": row.get("pa_display_rec", "Stage 3 Active"),
                            "price_at_decision": float(spot),
                            "atr_14": float(row.get("atr", spot * 0.02) or spot * 0.02),
                            "rsi_14": float(row.get("rsi_14", 55.0) or 55.0),
                            "adx_14": float(row.get("adx_14", 22.0) or 22.0),
                            "ema_20": float(row.get("ema_20", spot) or spot),
                            "ema_50": float(row.get("ema_50", spot) or spot),
                            "ema_200": float(row.get("ema_200", spot) or spot),
                            "iv_rank": float(row.get("iv_rank", 30.0) or 30.0),
                            "iv_percentile": float(row.get("iv_percentile", 30.0) or 30.0),
                            "front_iv": float(setup.get("front_iv", 0.30)),
                            "back_iv": float(setup.get("back_iv", 0.32)),
                            "iv_term_structure": float(setup.get("back_iv", 0.32) - setup.get("front_iv", 0.30)),
                            "avg_option_volume": float(setup.get("avg_option_volume", 1000.0)),
                            "avg_open_interest": float(setup.get("avg_open_interest", 5000.0)),
                            "bid_ask_spread_pct": float(setup.get("bid_ask_spread_pct", 0.015)),
                            "earnings_date": earn_date,
                            "event_risk_flag": 0,
                            "reason_summary": f"Manually deployed from Consolidated Strategy Matrix. PA: {row.get('pa_display_rec')}. Regime: {row.get('mre_display_rec')}",
                            "model_version": MODEL_VERSION,
                            "ml_predicted_return": 0.0
                        }
                        
                        decision_id = insert_decision_log(decision_data)
                        
                        if decision_id:
                            setup_data = {
                                "decision_id": decision_id,
                                "ticker": ticker,
                                "strategy_type": "Bullish Calendar Spread",
                                "short_dte": int(setup["short_dte"]),
                                "long_dte": int(setup["long_dte"]),
                                "target_delta": float(setup["target_delta"]),
                                "short_expiry": setup["short_expiry"],
                                "long_expiry": setup["long_expiry"],
                                "selected_strike": float(setup["selected_strike"]),
                                "short_bid": float(setup["short_bid"]),
                                "short_ask": float(setup["short_ask"]),
                                "short_mid": float(setup["short_mid"]),
                                "long_bid": float(setup["long_bid"]),
                                "long_ask": float(setup["long_ask"]),
                                "long_mid": float(setup["long_mid"]),
                                "net_debit": float(setup["net_debit"]),
                                "max_risk": float(setup["max_risk"]),
                                "setup_delta": float(setup["setup_delta"]),
                                "setup_gamma": float(setup["setup_gamma"]),
                                "setup_theta": float(setup["setup_theta"]),
                                "setup_vega": float(setup["setup_vega"]),
                                "breakeven_low": float(setup["breakeven_low"]),
                                "breakeven_high": float(setup["breakeven_high"])
                            }
                            insert_option_setup(setup_data)
                            st.success(f"🚀 Calendar Spread for **{ticker}** logged and deployed successfully to SQLite (decision_id: {decision_id}, strike: ${setup['selected_strike']:.1f})!")
                        else:
                            st.error("Failed to insert decision record into the database.")
                    except Exception as ex:
                        st.error(f"Error deploying trade setup: {ex}")


def render_matrix_native(filtered_df: pd.DataFrame) -> None:
    """
    Render the consolidated strategy matrix using Streamlit's native st.dataframe
    with a pandas Styler for cell-level coloring and row-click detail panel.
    """
    display_df = build_display_df(filtered_df)

    show_cols = [
        "Ticker", "Price", "Net Change", "ATR", "Strength",
        "Earnings Date", "FDTS Signal", "Cal. Score", "Price Action",
        "Regime Setup", "Options Bias",
    ]
    df_show = display_df[show_cols].reset_index(drop=True)

    styled = df_show.style.apply(_apply_matrix_styles, axis=None)

    # Fixed height: 15 rows visible (15 × 35px + 48px header)
    table_height = 573

    st.caption("💡 Click any row to see the full 4-engine diagnostic breakdown below the table.")

    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=table_height,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Ticker": st.column_config.TextColumn(
                "Ticker", width="small", help="Ticker symbol",
            ),
            "Price": st.column_config.TextColumn(
                "Price", width="small", help="Current spot price",
            ),
            "Net Change": st.column_config.TextColumn(
                "Net Change", width="medium",
                help="Daily price change — green = up, red = down",
            ),
            "ATR": st.column_config.TextColumn(
                "ATR (14d)", width="small",
                help="14-day Average True Range",
            ),
            "Strength": st.column_config.TextColumn(
                "Strength", width="small",
                help="15-day price strength  ▲ Uptrend | ▶ Sideways | ▼ Downtrend",
            ),
            "Earnings Date": st.column_config.TextColumn(
                "Earnings Date", width="medium",
                help="🔴 = within 20 days (risk)  🟡 = 21–40 days (caution)",
            ),
            "FDTS Signal": st.column_config.TextColumn(
                "FDTS Signal", width="small",
                help="FazDane Trend Signal — Buy | Sell | No Trade",
            ),
            "Cal. Score": st.column_config.TextColumn(
                "Calendar Scoring Engine", width="medium",
                help="🟢 Deploy  🟡 Watch  🔵 Monitor  🔴 Avoid  ⚫ Filtered",
            ),
            "Price Action": st.column_config.TextColumn(
                "Price Action (Action)", width="large",
                help="Price Action Story Engine stage — click row for full breakdown",
            ),
            "Regime Setup": st.column_config.TextColumn(
                "Regime Setup", width="small",
                help="✅ Yes = regime approves calendar  ❌ No = unfavorable regime",
            ),
            "Options Bias": st.column_config.TextColumn(
                "Options Flow Bias", width="medium",
                help="Options flow bias — Call Heavy | Put Heavy | Balanced",
            ),
        },
    )

    # ── Row-click detail panel ──
    selected_rows = []
    try:
        selected_rows = event.selection.rows
    except Exception:
        pass

    if selected_rows:
        idx = selected_rows[0]
        if idx < len(filtered_df):
            selected_raw = filtered_df.reset_index(drop=True).iloc[idx]
            _render_detail_panel(selected_raw)


class CalendarRotationModule(FazDaneModule):
    MODULE_NAME = "Calendar Strategy Matrix"
    MODULE_ICON = "📅"
    MODULE_DESCRIPTION = "Consolidated Triple-Engine Scanning Strategy Matrix and Rotation Dashboard"
    TIER = 1
    SOURCE_NOTEBOOK = "05-SPX Sector Rotation / RRG-Style Visualization.ipynb"
    CACHE_TTL = 3600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance", "calendar_scoring_sqlite", "price_action_story_sqlite"]


    def render_sidebar(self):
        st.markdown("**Watchlist**")
        self.universe_name, self.tickers, self.benchmark = render_universe_manager(
            key_prefix="cal_strategy_matrix",
            show_benchmark=True,
            label="Select Universe:",
        )
        st.caption(f"{len(self.tickers)} symbols selected.")
        
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("**Triple-Engine Execution**")
        run_all_scans = st.button("🚀 Run Quad-Engine Scan", use_container_width=True, type="primary")
        
        if run_all_scans:
            st.session_state["run_all_scans_triggered"] = True

    def render_main(self):
        self.render_section_header(
            "📅 Calendar Strategy Matrix & Rotation Dashboard",
            "Consolidated Ticker Recommendations from Multi-Factor Scanning Engines"
        )
        
        if not self.tickers:
            st.warning("⚠️ Please select a ticker universe in the sidebar to begin.")
            return

        # ── Trigger Triple-Engine Scan On-Demand ─────────────────────────────
        if st.session_state.pop("run_all_scans_triggered", False):
            self._execute_triple_engine_scans()

        # ── Render Page Navigation Tabs ──────────────────────────────────────
        tab_matrix, tab_rrg = st.tabs([
            "🔍 Consolidated Strategy Matrix",
            "📊 RRG Rotation Matrix"
        ])
        
        # ── TAB 1: Consolidated Strategy Matrix ──────────────────────────────
        with tab_matrix:
            df = load_consolidated_recommendations(self.tickers)
            
            # Check if any database records exist
            has_data = not df.empty and not (df["cs_rec"].eq("N/A") & df["pa_rec"].eq("N/A") & df["mre_rec"].eq("N/A")).all()
            
            if not has_data:
                st.warning("⚠️ No processed engine results found in database for the selected universe. You must run a fresh triple scan.")
                if st.button("🚀 Execute Triple-Engine Scan Now", key="cal_strategy_first_run_btn", use_container_width=True, type="primary"):
                    st.session_state["run_all_scans_triggered"] = True
                    st.rerun()
                return

            # Display KPI Summary Cards
            cs_deploy_watch = len(df[df["cs_rec"].isin(["Deploy", "Watch"])])
            pa_calendar = len(df[df["pa_display_rec"].str.contains("Deploy Calendar", na=False)])
            mre_calendar_yes = len(df[df["mre_display_rec"] == "✅ Yes"])
            ol_put_heavy = len(df[df["ol_bias"].str.contains("Put", na=False)])
            
            metrics = {
                "Total Universe Tickers": (len(df), None, ""),
                "Scoring Engine Deploy/Watch": (cs_deploy_watch, None, f" / {len(df)}"),
                "PA Calendar Spreads": (pa_calendar, None, f" / {len(df)}"),
                "Regime Calendar Setup ✅": (mre_calendar_yes, None, f" / {len(df)}"),
                "⚠️ Put-Heavy Tickers": (ol_put_heavy, None, f" / {len(df)}")
            }
            self.render_metrics_row(metrics)
            st.write("")

            # Filter Options Block
            with st.expander("🔍 Interactive Recommendations Filters", expanded=True):
                col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                with col_f1:
                    all_cs_recs = sorted(list(df["cs_rec"].unique()))
                    sel_cs_recs = st.multiselect(
                        "Scoring Recommendation",
                        options=all_cs_recs,
                        default=all_cs_recs,
                        key="cal_filter_cs"
                    )
                with col_f2:
                    all_pa_display_recs = sorted(list(df["pa_display_rec"].unique()))
                    sel_pa_display_recs = st.multiselect(
                        "Price Action (Action)",
                        options=all_pa_display_recs,
                        default=all_pa_display_recs,
                        key="cal_filter_pa"
                    )
                with col_f3:
                    all_mre_display_recs = sorted(list(df["mre_display_rec"].unique()))
                    sel_mre_display_recs = st.multiselect(
                        "Regime Calendar Setup",
                        options=all_mre_display_recs,
                        default=all_mre_display_recs,
                        key="cal_filter_mre"
                    )
                with col_f4:
                    all_fdts_sigs = sorted(list(df["fdts_signal"].unique()))
                    sel_fdts_sigs = st.multiselect(
                        "FDTS Signal",
                        options=all_fdts_sigs,
                        default=all_fdts_sigs,
                        key="cal_filter_fdts"
                    )
                col_f5, col_f6 = st.columns(2)
                with col_f5:
                    all_ol_bias = sorted(list(df["ol_bias"].unique()))
                    sel_ol_bias = st.multiselect(
                        "Options Flow Bias",
                        options=all_ol_bias,
                        default=all_ol_bias,
                        key="cal_filter_ol_bias"
                    )
                with col_f6:
                    all_earnings_risk = sorted(list(df["earnings_risk_category"].unique()))
                    sel_earnings_risk = st.multiselect(
                        "Earnings Date Risk (Color Category)",
                        options=all_earnings_risk,
                        default=all_earnings_risk,
                        key="cal_filter_earnings_risk"
                    )

            # Apply filters
            filtered_df = df[
                df["cs_rec"].isin(sel_cs_recs) &
                df["pa_display_rec"].isin(sel_pa_display_recs) &
                df["mre_display_rec"].isin(sel_mre_display_recs) &
                df["fdts_signal"].isin(sel_fdts_sigs) &
                df["ol_bias"].isin(sel_ol_bias) &
                df["earnings_risk_category"].isin(sel_earnings_risk)
            ]
            
            if filtered_df.empty:
                st.info("No tickers match the active recommendation filters.")
            else:
                tickers_list = sorted(list(filtered_df["ticker"].unique()))
                tickers_str = ", ".join(tickers_list)
                st.markdown(
                    f"<div style='font-size:14px; font-weight:600; color:#94a3b8; margin-top:8px; margin-bottom:4px;'>"
                    f"📋 Copy Filtered Tickers ({len(tickers_list)} symbols)</div>", 
                    unsafe_allow_html=True
                )
                st.code(tickers_str, language="text")
                st.write("")
                render_matrix_native(filtered_df)

        # ── TAB 2: RRG Rotation Matrix (Backed up visualizations) ────────────
        with tab_rrg:
            with st.spinner(f"Analyzing {self.universe_name} and calculating rotation scores..."):
                res = analyze_universe(
                    self.universe_name,
                    tuple(self.tickers),
                    self.benchmark,
                )
                
            if res:
                combined_scores = res["scores"].copy()
                combined_scores["calendar_score_normalized"] = (
                    combined_scores["calendar_score"] - combined_scores["calendar_score"].mean()
                ) / (combined_scores["calendar_score"].std() + 1e-8)
                combined_scores = combined_scores.sort_values("calendar_score", ascending=False).reset_index(drop=True)
                
                self._render_dashboard({self.universe_name: res}, combined_scores)
                self._render_top_candidates(combined_scores)
                self._render_universe_summary({self.universe_name: res}, combined_scores)
                self._render_interpretation_guide()
            else:
                st.error("Failed to compute RRG rotation data for the selected universe.")

    def _execute_triple_engine_scans(self):
        """Execute calculations and scoring on demand sequentially across all 4 scanning engines."""
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        try:
            # 🎬 ENGINE 1: Calendar Opportunity Scoring Engine
            status_text.write("### 🎬 Starting Engine 1/4: Calendar Opportunity Scoring Engine...")
            from modules.calendar_scoring.dashboard import CalendarOpportunityScoringModule
            scoring_module = CalendarOpportunityScoringModule()
            scoring_module.execute_engine_scan(
                universe_name=self.universe_name,
                tickers=self.tickers,
                rerun=False,
                progress_bar=progress_bar,
                status_text=status_text
            )
            
            # 🎬 ENGINE 2: Price Action Story Engine
            status_text.write("### 🎬 Starting Engine 2/4: Price Action Story Engine...")
            from modules.tier2.price_action_story import PriceActionStoryModule
            pa_module = PriceActionStoryModule()
            pa_module.execute_price_action_scan(
                universe_name=self.universe_name,
                tickers=self.tickers,
                benchmark=self.benchmark,
                lookback_days=252,
                rerun=False,
                progress_bar=progress_bar,
                status_text=status_text
            )
            
            # 🎬 ENGINE 3: Regime Intelligence Dashboard
            status_text.write("### 🎬 Starting Engine 3/4: Regime Intelligence Dashboard...")
            from modules.tier2.markov_regime_engine import MarkovRegimeEngineModule
            regime_module = MarkovRegimeEngineModule()
            regime_module.execute_regime_scan(
                universe_name=self.universe_name,
                symbols=self.tickers,
                lookback_years=5,
                n_states=3,
                rerun=False,
                progress_bar=progress_bar,
                status_text=status_text
            )
            
            # 🎬 ENGINE 4: Options Liquidity Discovery (Put/Call Flow)
            status_text.write("### 🎬 Starting Engine 4/4: Options Liquidity Discovery...")
            from modules.tier1.options_liquidity import OptionsLiquidityModule
            ol_module = OptionsLiquidityModule()
            ol_module.execute_options_liquidity_scan(
                tickers=self.tickers,
                progress_bar=progress_bar,
                status_text=status_text
            )
            
            # Success notification
            progress_bar.empty()
            status_text.empty()
            st.success("🎉 **Quad-Engine Scan Completed!** All 4 scanning engines successfully ran and saved results to SQLite databases.")
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error during Quad-Engine scan execution: {e}")
            logger.error("Quad-Engine scan error", exc_info=True)
            
        st.rerun()

    def _render_dashboard(self, MULTI_RESULTS, combined_scores):
        universe_colors = {
            "Calendar Candidates": "#3B82F6",
            "SPX Sectors": "#10B981",
            "MAG 7": "#F59E0B",
            "Leading ETFs": "#EF4444"
        }
        palette = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4", "#EC4899", "#84CC16"]
        for idx, univ_name in enumerate(MULTI_RESULTS.keys()):
            universe_colors.setdefault(univ_name, palette[idx % len(palette)])

        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "<b>Q1: Rotation Matrix</b> (RS Ratio vs Momentum)",
                "<b>Q2: Trend Strength</b> (Top 12 by Score)",
                "<b>Q3: Quality Assessment</b> (Score vs Spot Price)",
                "<b>Q4: Liquidity Heatmap</b> (Top 15 Candidates)"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ],
            vertical_spacing=0.14,
            horizontal_spacing=0.10
        )

        # ----- QUADRANT 1: ROTATION MATRIX -----
        for univ_name, color in universe_colors.items():
            if univ_name not in MULTI_RESULTS:
                continue
            scores_top = MULTI_RESULTS[univ_name]["scores"].head(PLOT_TOP_N)
            if len(scores_top) == 0:
                continue

            fig.add_trace(
                go.Scatter(
                    x=scores_top["rs_ratio"],
                    y=scores_top["rs_momentum"],
                    mode="markers+text",
                    name=univ_name,
                    text=scores_top["ticker"],
                    textposition="top center",
                    textfont=dict(size=9, color="white"),
                    marker=dict(size=10, color=color, opacity=0.75, line=dict(color="white", width=1)),
                    customdata=scores_top["fdts_signal"],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        f"Universe: {univ_name}<br>"
                        "RS Ratio: %{x:.2f}<br>"
                        "RS Momentum: %{y:.2f}<br>"
                        "FDTS Signal: %{customdata}<br>"
                        "<extra></extra>"
                    ),
                    legendgroup="rotation"
                ),
                row=1, col=1
            )

        fig.add_hline(y=100, line_color="#94A3B8", line_width=1, line_dash="dash", row=1, col=1)
        fig.add_vline(x=100, line_color="#94A3B8", line_width=1, line_dash="dash", row=1, col=1)

        # ----- QUADRANT 2: TREND STRENGTH BARS -----
        top_scores = combined_scores.head(12).copy()
        bar_colors = [universe_colors.get(u, "#3B82F6") for u in top_scores["universe"]]

        fig.add_trace(
            go.Bar(
                x=top_scores["ticker"],
                y=top_scores["trend_score"],
                name="Trend Score",
                marker=dict(color=bar_colors, line=dict(color="white", width=1)),
                text=top_scores["trend_score"].round(0).astype(int),
                textposition="outside",
                customdata=top_scores["fdts_signal"],
                hovertemplate="<b>%{x}</b><br>Trend Score: %{y:.1f}<br>FDTS Signal: %{customdata}<br><extra></extra>",
                showlegend=False
            ),
            row=1, col=2
        )

        # ----- QUADRANT 3: QUALITY SCATTER -----
        for quality_type, q_color in [("Best", "#10B981"), ("Watch", "#F59E0B")]:
            subset = combined_scores[combined_scores["quality"] == quality_type]
            if len(subset) == 0: continue

            fig.add_trace(
                go.Scatter(
                    x=subset["calendar_score"],
                    y=subset["spot"],
                    mode="markers+text",
                    name=f"{quality_type} Quality",
                    text=subset["ticker"],
                    textposition="top center",
                    textfont=dict(size=8, color="white"),
                    marker=dict(size=12, color=q_color, opacity=0.8, line=dict(color="white", width=1)),
                    customdata=subset["fdts_signal"],
                    hovertemplate=(
                        "<b>%{text}</b><br>Score: %{x:.1f}<br>Spot: $%{y:.2f}<br>"
                        f"Quality: {quality_type}<br>FDTS Signal: %{{customdata}}<br><extra></extra>"
                    ),
                    legendgroup="quality"
                ),
                row=2, col=1
            )

        fig.add_vline(x=75, line_color="#10B981", line_width=1, line_dash="dot", row=2, col=1)

        # ----- QUADRANT 4: LIQUIDITY HEATMAP -----
        heatmap_data = combined_scores.head(15).copy()
        if not heatmap_data.empty:
            z_data = np.array([
                heatmap_data["option_liquidity_score"].values,
                (heatmap_data["option_oi"] / heatmap_data["option_oi"].max() * 100).values,
                (heatmap_data["option_volume"] / heatmap_data["option_volume"].max() * 100).values
            ])

            fig.add_trace(
                go.Heatmap(
                    z=z_data,
                    x=heatmap_data["ticker"],
                    y=["Liquidity Score", "Option OI", "Option Volume"],
                    colorscale=[[0, "#1E293B"], [0.2, "#3B82F6"], [0.6, "#10B981"], [1.0, "#F59E0B"]],
                    showscale=True,
                    customdata=np.tile(heatmap_data["fdts_signal"].values, (3, 1)),
                    hovertemplate="<b>%{x}</b><br>%{y}: %{z:.0f}<br>FDTS Signal: %{customdata}<br><extra></extra>",
                    colorbar=dict(x=1.02, len=0.4, y=0.22, thickness=15, tickfont=dict(color="white", size=10))
                ),
                row=2, col=2
            )

        # ----- DASHBOARD LAYOUT -----
        fig.update_layout(
            title=dict(
                text="<b>4-Quadrant Multi-Universe Dashboard</b><br><sub style='color:#94A3B8'>Comparative Relative Strength & Momentum Analysis</sub>",
                x=0.5, xanchor="center", font=dict(size=18, color="#E2E8F0")
            ),
            paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e",
            font=dict(family="Inter, sans-serif", size=11, color="#E2E8F0"),
            height=850,
            showlegend=True,
            legend=dict(
                orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.08, 
                bgcolor="rgba(13,27,46,0.8)", bordercolor="#1e3a5f", borderwidth=1,
                font=dict(color="#E2E8F0")
            ),
            margin=dict(l=40, r=140, t=80, b=40)
        )

        fig.update_xaxes(title_text="RS Ratio %", row=1, col=1, gridcolor="#1e3a5f", zeroline=False)
        fig.update_yaxes(title_text="RS Momentum %", row=1, col=1, gridcolor="#1e3a5f", zeroline=False)
        fig.update_xaxes(title_text="Ticker", row=1, col=2, gridcolor="#1e3a5f")
        fig.update_yaxes(title_text="Trend Score (0-100)", row=1, col=2, gridcolor="#1e3a5f", range=[0, 110])
        fig.update_xaxes(title_text="Calendar Score", row=2, col=1, gridcolor="#1e3a5f")
        fig.update_yaxes(title_text="Spot Price ($)", row=2, col=1, gridcolor="#1e3a5f")
        fig.update_xaxes(title_text="Ticker", row=2, col=2)

        # Adjust subplot title colors for dark theme
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=14, color="#e2e8f0")

        st.plotly_chart(fig, use_container_width=True, theme=None)

    def _render_top_candidates(self, combined_scores):
        st.markdown("### 🏆 Top Rotation Candidates")
        
        display_df = combined_scores.head(15).copy()
        display_df["Strength"] = display_df["strength_pct"].apply(lambda p: format_strength_meter(p)[0])
        
        display_cols = [
            "ticker", "universe", "quality", "fdts_signal", "Strength", "calendar_score", "trend_score",
            "rs_ratio", "rs_momentum", "spot", "target_strike", "option_liquidity_score"
        ]
        
        def highlight_strength(val):
            if val == '▲':
                return 'color: #00D4AA; font-weight: bold; text-align: center;'
            elif val == '▼':
                return 'color: #FF4B4B; font-weight: bold; text-align: center;'
            elif val == '▶':
                return 'color: #FFA421; font-weight: bold; text-align: center;'
            return ''
            
        st.dataframe(
            display_df[display_cols].style.map(highlight_strength, subset=["Strength"]),
            use_container_width=True,
            column_config={
                "fdts_signal": st.column_config.TextColumn("FDTS Signal", width="small"),
                "Strength": st.column_config.TextColumn("Strength", width="small"),
                "calendar_score": st.column_config.NumberColumn("Cal Score", format="%.1f"),
                "trend_score": st.column_config.NumberColumn("Trend Score", format="%.1f"),
                "rs_ratio": st.column_config.NumberColumn("RS Ratio", format="%.1f"),
                "rs_momentum": st.column_config.NumberColumn("RS Mom", format="%.1f"),
                "spot": st.column_config.NumberColumn("Spot Price", format="$%.2f"),
                "target_strike": st.column_config.NumberColumn("Target Strike", format="$%.2f"),
                "option_liquidity_score": st.column_config.NumberColumn("Liquidity", format="%.0f"),
                "quality": st.column_config.TextColumn("Quality")
            }
        )

    def _render_universe_summary(self, MULTI_RESULTS, combined_scores):
        st.markdown("### 📊 Universe Comparison Summary")
        summary_rows = []
        for univ_name in MULTI_RESULTS.keys():
            u_scores = combined_scores[combined_scores["universe"] == univ_name]
            if len(u_scores) == 0: continue
            summary_rows.append({
                "Universe": univ_name,
                "Tickers": len(u_scores),
                "Avg Score": u_scores["calendar_score"].mean(),
                "Top Score": u_scores["calendar_score"].max(),
                "Best Count": (u_scores["quality"] == "Best").sum(),
                "Watch Count": (u_scores["quality"] == "Watch").sum(),
                "Avg Trend": u_scores["trend_score"].mean(),
                "Avg RS Ratio": u_scores["rs_ratio"].mean(),
                "Avg Momentum": u_scores["rs_momentum"].mean(),
                "Top Ticker": u_scores.iloc[0]["ticker"] if len(u_scores) > 0 else "N/A"
            })

        summary_df = pd.DataFrame(summary_rows).set_index("Universe")
        st.dataframe(
            summary_df,
            use_container_width=True,
            column_config={
                "Avg Score": st.column_config.NumberColumn(format="%.1f"),
                "Top Score": st.column_config.NumberColumn(format="%.1f"),
                "Avg Trend": st.column_config.NumberColumn(format="%.1f"),
                "Avg RS Ratio": st.column_config.NumberColumn(format="%.1f"),
                "Avg Momentum": st.column_config.NumberColumn(format="%.1f"),
            }
        )

    def _render_interpretation_guide(self):
        st.markdown("### 📍 4-Quadrant Interpretation Guide")
        
        c1, c2 = st.columns(2)
        with c1:
            st.success("**↗ UPPER RIGHT (>100, >100) | LEADING**\n\nStrong relative strength + accelerating momentum. Top calendar spread candidates.")
            st.error("**↙ LOWER LEFT (<100, <100) | LAGGING**\n\nBoth weakening. Avoid for new calendar spread positions.")
        with c2:
            st.info("**↖ UPPER LEFT (<100, >100) | IMPROVING**\n\nMomentum accelerating but RS still lagging. Monitor for catch-up.")
            st.warning("**↘ LOWER RIGHT (>100, <100) | WEAKENING**\n\nStrong RS but momentum fading. Caution on new entries.")

        st.markdown("#### Scoring Components")
        st.markdown("- **Trend Score (40%)**: EMA alignment, MA crossovers, recent highs, volume confirmation\n- **RS Ratio (30%)**: Relative strength vs benchmark (100 = parity)\n- **RS Momentum (30%)**: Rate of change in relative strength\n- **Calendar Score**: Weighted composite. Score ≥ 75 = 'Best' candidate ✨")

        st.markdown("#### Quality Badges")
        st.markdown("- **✨ Best (Score ≥ 75)**: Ready for calendar spread execution. Primary candidates.\n- **👁️ Watch (Score < 75)**: Monitor closely. Wait for signal improvement.")