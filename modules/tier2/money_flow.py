"""
FazDane Analytics — Tier 2
Multi-Timeframe Money Flow Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.transforms import blended_transform_factory
from datetime import datetime, timedelta
import logging
from modules.base_module import FazDaneModule
from utils.universe_manager import render_universe_manager

logger = logging.getLogger("MoneyFlow")

import json
import os

CONFIG_PATH = os.path.join("config", "asset_lists.json")

def load_asset_sets():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading asset sets: {e}")
    return {}

def save_asset_sets(asset_sets):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(asset_sets, f, indent=4)

CONFIG = {
    'Daily':   {'interval': '1d',  'fmt': '%Y-%m-%d', 'days_mult': 2,   'h_factor': 0.45},
    'Weekly':  {'interval': '1wk', 'fmt': '%Y-%m-%d', 'days_mult': 10,  'h_factor': 0.55},
    'Monthly': {'interval': '1mo', 'fmt': '%Y-%m',    'days_mult': 35,  'h_factor': 0.65},
    'Yearly':  {'interval': '1mo', 'fmt': '%Y',       'days_mult': 400, 'h_factor': 0.85}
}

def get_text_color(val, bg_color):
    if bg_color is None:
        return 'black'
    r, g, b, _ = bg_color
    luminance = 0.299*r + 0.587*g + 0.114*b
    if luminance < 0.55:
        return 'white'
    if val < 0:
        return '#8B0000'
    return 'black'

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_money_flow_data(tickers, interval, days_mult):
    tickers = list(tickers)
    if not tickers:
        return pd.DataFrame()

    # Always fetch the max possible periods (500) so that changing lookback uses cache instantly
    end_date = datetime.now()
    start_date = end_date - timedelta(days=500 * days_mult)
    
    data = yf.download(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
        threads=True
    )
    
    if data.empty:
        return pd.DataFrame()
        
    return data

def calculate_fdts_macd(df, period=20, 
                        macdFastLong=3, macdSlowLong=10, macdSignalLong=16,
                        macdFastShort=12, macdSlowShort=26, macdSignalShort=9):
    def exp_average(series, span):
        return series.ewm(span=span, adjust=False).mean()

    open_p = df['Open']
    high_p = df['High']
    low_p = df['Low']
    close_p = df['Close']
    
    price = (high_p + low_p + close_p) / 3
    
    ema1 = exp_average(price, period)
    ema2 = exp_average(ema1, period)
    ema3 = exp_average(ema2, period)
    tma1 = 3 * ema1 - 3 * ema2 + ema3
    
    tma1_ema1 = exp_average(tma1, period)
    tma1_ema2 = exp_average(tma1_ema1, period)
    tma1_ema3 = exp_average(tma1_ema2, period)
    tma2 = 3 * tma1_ema1 - 3 * tma1_ema2 + tma1_ema3
    
    typical_tema = tma1 + (tma1 - tma2)
    
    # Heikin-Ashi
    ha_open = pd.Series(np.nan, index=df.index)
    
    first_valid = close_p.first_valid_index()
    if first_valid is not None:
        idx = df.index.get_loc(first_valid)
        hl2 = (high_p.iloc[idx] + low_p.iloc[idx]) / 2
        ha_open.iloc[idx] = hl2
    
    ohlc4 = (open_p + high_p + low_p + close_p) / 4
    if first_valid is not None:
        for i in range(idx + 1, len(df)):
            ha_open.iloc[i] = (ohlc4.iloc[i-1] + ha_open.iloc[i-1]) / 2
        
    ha_close = (ohlc4 + ha_open + np.maximum(high_p, ha_open) + np.minimum(low_p, ha_open)) / 4
    
    ha_ema1 = exp_average(ha_close, period)
    ha_ema2 = exp_average(ha_ema1, period)
    ha_ema3 = exp_average(ha_ema2, period)
    ha_tma1 = 3 * ha_ema1 - 3 * ha_ema2 + ha_ema3
    
    ha_tma1_ema1 = exp_average(ha_tma1, period)
    ha_tma1_ema2 = exp_average(ha_tma1_ema1, period)
    ha_tma1_ema3 = exp_average(ha_tma1_ema2, period)
    ha_tma2 = 3 * ha_tma1_ema1 - 3 * ha_tma1_ema2 + ha_tma1_ema3
    
    ha_tema = ha_tma1 + (ha_tma1 - ha_tma2)
    fdts_dev = typical_tema - ha_tema
    
    macd_long = exp_average(close_p, macdFastLong) - exp_average(close_p, macdSlowLong)
    macd_long_dev = macd_long - exp_average(macd_long, macdSignalLong)
    
    macd_short = exp_average(close_p, macdFastShort) - exp_average(close_p, macdSlowShort)
    macd_short_dev = macd_short - exp_average(macd_short, macdSignalShort)
    
    buy_signal = (fdts_dev > 0) & (macd_long_dev > 0)
    sell_signal = (fdts_dev < 0) & (macd_short_dev < 0)
    
    state = pd.Series(0, index=df.index)
    state[close_p.isna()] = np.nan
    state[buy_signal] = 1
    state[sell_signal] = -1
    
    state_changed = state != state.shift(1)
    state_changed[state.isna()] = False
    
    group_id = state_changed.cumsum()
    group_id = group_id.where(~state.isna())
    days_in_state = group_id.groupby(group_id).cumcount() + 1
    
    return state, days_in_state

def get_upcoming_earnings(tickers, days=40):
    try:
        from utils.earnings_calendar_store import DB_PATH as ec_db_path
        import sqlite3
        from datetime import datetime, timedelta
        import yfinance as yf
        import concurrent.futures
        
        if not tickers:
            return {}
            
        today_date = datetime.now()
        today_str = today_date.strftime("%Y-%m-%d")
        future_str = (today_date + timedelta(days=days)).strftime("%Y-%m-%d")
        
        result = {}
        missing_tickers = list(tickers)
        
        if ec_db_path.exists():
            placeholders = ",".join("?" for _ in tickers)
            with sqlite3.connect(ec_db_path) as conn:
                rows = conn.execute(
                    f"""
                    SELECT ticker, MIN(date) 
                    FROM ec_earnings_events 
                    WHERE ticker IN ({placeholders}) AND date >= ? AND date <= ?
                    GROUP BY ticker
                    """,
                    [*tickers, today_str, future_str]
                ).fetchall()
                for row in rows:
                    result[row[0]] = row[1]
                    if row[0] in missing_tickers:
                        missing_tickers.remove(row[0])
                        
        # Fallback to yfinance for missing tickers
        if missing_tickers:
            def fetch_yf(t):
                try:
                    cal = yf.Ticker(t).calendar
                    if isinstance(cal, dict) and "Earnings Date" in cal:
                        ed = cal.get("Earnings Date")
                        if isinstance(ed, list) and len(ed) > 0:
                            ed_val = ed[0].strftime("%Y-%m-%d")
                            if today_str <= ed_val <= future_str:
                                return t, ed_val
                    elif hasattr(cal, 'columns') and "Earnings Date" in cal.columns:
                        ed_val = cal["Earnings Date"].iloc[0].strftime("%Y-%m-%d")
                        if today_str <= ed_val <= future_str:
                            return t, ed_val
                except Exception:
                    pass
                return t, None

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                for t, ed_val in executor.map(fetch_yf, missing_tickers):
                    if ed_val:
                        result[t] = ed_val

        return result
    except Exception as e:
        import logging
        logging.getLogger("FazDaneApp").error(f"Failed to fetch earnings dates: {e}")
        return {}

class MoneyFlowModule(FazDaneModule):
    MODULE_NAME = "Multi-Timeframe Money Flow"
    MODULE_ICON = "🔥"
    MODULE_DESCRIPTION = "Heatmap of Cumulative Returns Across Multiple Timeframes"
    TIER = 2
    SOURCE_NOTEBOOK = "01-Heat Map with Cumulative Total v05142025.ipynb"
    CACHE_TTL = 3600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def render_sidebar(self):
        st.markdown("**Ticker Universe**")
        self.mf_universe_name, self.mf_tickers, _ = render_universe_manager(
            key_prefix="mf",
            show_benchmark=False,
            label="Asset List:"
        )
        st.caption(f"{len(self.mf_tickers)} tickers selected from {self.mf_universe_name}.")

        self.timeframe = st.selectbox("Interval:", options=['Daily', 'Weekly', 'Monthly', 'Yearly'], index=0)
        self.lookback = st.number_input("Lookback Periods:", min_value=1, max_value=500, value=10)
        self.sort_order = st.selectbox("Sort Order:", options=['Total Cumulative %', 'FDTS Signal & Days', 'Ticker Rank'], index=0)
        
        st.markdown("**View Filters**")
        self.filter_type = st.selectbox(
            "View Filter:",
            options=['Ranked Pagination', 'Bottom 10', 'Perf Range (Custom %)', 'Show All (Sorted)'],
            index=0
        )
        
        if self.filter_type == 'Ranked Pagination':
            self.rank_count = st.number_input("Count:", min_value=1, value=15, step=15)
            max_start_rank = max(1, len(self.mf_tickers))
            self.rank_start = st.number_input("Start Rank:", min_value=1, max_value=max_start_rank, value=1, step=self.rank_count)
            self.range_limits = None
        elif self.filter_type == 'Perf Range (Custom %)':
            self.range_limits = st.slider("Range %:", min_value=-100.0, max_value=500.0, value=(5.0, 30.0), step=1.0)
        else:
            self.range_limits = None
            
        if st.button("🔄 Refresh Data", width="stretch"):
            fetch_money_flow_data.clear()
            st.rerun()

    def render_main(self):
        self.render_section_header("🔥 Multi-Timeframe Money Flow Dashboard", "Analyze comparative momentum and capital rotation.")
        
        cfg = CONFIG[self.timeframe]
        
        # Use tickers from universe manager
        initial_tickers = tuple(self.mf_tickers)
        
        if not initial_tickers:
            st.warning("⚠️ Please provide at least one valid ticker symbol.")
            return
            
        with st.spinner(f"Fetching {self.timeframe} data for {len(initial_tickers)} tickers..."):
            full_data = fetch_money_flow_data(
                tickers=initial_tickers, 
                interval=cfg['interval'], 
                days_mult=cfg['days_mult']
            )
            
        if full_data.empty:
            st.warning("⚠️ No data found for the selected parameters.")
            return
            
        is_multi = isinstance(full_data.columns, pd.MultiIndex)
        
        # Calculate FDTS for each ticker
        fdts_results = {}
        for ticker in initial_tickers:
            try:
                if is_multi:
                    df = full_data.xs(ticker, level=1, axis=1)
                else:
                    df = full_data  # Only one ticker, columns are Open, High, Low, Close
                    
                state_series, days_series = calculate_fdts_macd(df)
                if not state_series.empty:
                    fdts_results[ticker] = {
                        'state': state_series.iloc[-1],
                        'days': days_series.iloc[-1]
                    }
            except Exception as e:
                logger.error(f"Error calculating FDTS for {ticker}: {e}")
                
        if is_multi:
            close_data = full_data['Close'] if 'Close' in full_data else full_data['Adj Close']
        else:
            close_data = full_data['Close'] if 'Close' in full_data else full_data['Adj Close']
            close_data = pd.DataFrame(close_data, columns=[initial_tickers[0]])
            
        if self.timeframe == 'Yearly':
            close_data = close_data.resample('YE').last()
            
        returns = close_data.pct_change().dropna(how='all') * 100
        period_returns = returns.tail(self.lookback).fillna(0)
        if period_returns.empty:
            st.warning("Not enough return history for the selected universe and lookback.")
            return
        
        # True Cumulative
        cumulative = ((period_returns / 100 + 1).prod() - 1) * 100
        
        # Sorting
        if self.sort_order == 'FDTS Signal & Days':
            scores = {}
            for ticker, cum in cumulative.items():
                if ticker in fdts_results:
                    stt = fdts_results[ticker]['state']
                    dys = fdts_results[ticker]['days']
                    if stt == 1:
                        score = 10000 - dys  # Buy, ascending days (lowest days at top, so highest score = 10000 - 1 = 9999)
                    elif stt == 0:
                        score = 0 + dys      # Neutral, descending days (most days at top)
                    else:
                        score = -10000 - dys # Sell, ascending days (lowest days at top of sells = highest score among sells)
                else:
                    score = -20000
                scores[ticker] = score
            sorted_all = pd.Series(scores).sort_values(ascending=False)
        elif self.sort_order == 'Ticker Rank':
            mf_tickers_list = list(self.mf_tickers)
            valid_tickers = [t for t in mf_tickers_list if t in cumulative.index]
            sorted_all = cumulative.loc[valid_tickers]
        else:
            sorted_all = cumulative.sort_values(ascending=False)
        
        # Filtering
        if self.filter_type == 'Ranked Pagination':
            start_idx = self.rank_start - 1
            if start_idx >= len(sorted_all):
                start_idx = 0
                self.rank_start = 1
                
            end_idx = start_idx + self.rank_count
            tickers = sorted_all.iloc[start_idx:end_idx].index.tolist()
            if not tickers:
                st.warning(f"No tickers found at rank {self.rank_start}. Max rank is {len(sorted_all)}.")
                return
            header_filter = f"RANKS {self.rank_start} TO {start_idx + len(tickers)}"
        elif self.filter_type == 'Bottom 10':
            tickers = sorted_all.tail(10).index.tolist()
            header_filter = "BOTTOM 10"
        elif self.filter_type == 'Perf Range (Custom %)':
            low_limit, high_limit = self.range_limits
            mask = (sorted_all >= low_limit) & (sorted_all <= high_limit)
            tickers = sorted_all[mask].index.tolist()
            header_filter = f"PERF RANGE ({low_limit}% to {high_limit}%)"
        else:
            tickers = sorted_all.index.tolist()
            header_filter = "ALL ASSETS (SORTED)"
            
        if not tickers:
            st.warning("⚠️ No tickers match the current filter criteria.")
            return
            
        tickers = [t for t in tickers if t in period_returns.columns]
        plot_data = period_returns[tickers]
        
        if self.timeframe == 'Yearly':
            plot_data.index = plot_data.index.year.astype(str)
        else:
            plot_data.index = plot_data.index.strftime(cfg['fmt'])
            
        cum_footer = cumulative.loc[tickers].to_frame().T
        cum_footer.index = ['TOTAL CUMULATIVE %']
        final_df = pd.concat([plot_data, cum_footer]).fillna(0)
        
        # Add earnings indication
        upcoming_earnings = get_upcoming_earnings(tickers, days=40)
        final_df.columns = [f"{t} ☎️" if t in upcoming_earnings else t for t in tickers]
        final_df.columns.name = None  # Remove 'Ticker' label for cleaner top axis
        
        # -------- PLOT --------
        cfg = CONFIG[self.timeframe]
        calc_height = max(8, (len(final_df) + 2) * cfg.get('h_factor', 0.45))
        fig, ax = plt.subplots(figsize=(16, calc_height))

        # Explicitly set the background color of the figure and axes to transparent
        # so it blends with the Streamlit theme, but keep the core plot colors intact
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        n_rows = len(final_df)
        n_tickers = len(tickers)
        num_cells = n_rows * n_tickers
        
        show_annotations = num_cells <= 1500

        sns.heatmap(
            final_df,
            annot=show_annotations,
            cmap='RdYlGn',
            center=0,
            fmt=".1f" if show_annotations else "",
            linewidths=.5,
            cbar_kws={'shrink': 0.25},
            ax=ax
        )

        if show_annotations:
            facecolors = ax.collections[0].get_facecolors()
            for i, text in enumerate(ax.texts):
                try:
                    val_str = text.get_text().replace('%', '')
                    val = float(val_str)
                    text.set_text(f"{val:.1f}%")
                    bg_color = facecolors[i] if i < len(facecolors) else None
                    text.set_color(get_text_color(val, bg_color))
                    if abs(val) >= 5 or val < 0:
                        text.set_weight('bold')
                    if i >= (n_rows - 1) * n_tickers:
                        text.set_weight('bold')
                        text.set_fontsize(12)
                except Exception:
                    continue

        ax.axhline(n_rows - 1, color='#222222', linewidth=3)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.xticks(rotation=0, ha='center', fontweight='bold', color='#111111', fontsize=11)
        plt.yticks(color='#333333', fontsize=10)
        
        # Color the xtick labels red if they have upcoming earnings
        for tick_label in ax.get_xticklabels():
            if '☎' in tick_label.get_text():
                tick_label.set_color("red")

        mf_tickers_list = list(self.mf_tickers)
        
        blend_labels = blended_transform_factory(ax.transAxes, ax.transData)
        y_rank = n_rows + 0.5
        y_fdts = n_rows + 1.2
        y_dots = n_rows + 1.8

        ax.text(
            -0.01, y_rank, 'Ticker Ranks', 
            transform=blend_labels, 
            ha='right', va='center', 
            fontweight='bold', fontsize=10, color='#444444'
        )
        
        ax.text(
            -0.01, y_fdts, 'FDTS Signal (Days)', 
            transform=blend_labels, 
            ha='right', va='center', 
            fontweight='bold', fontsize=10, color='#444444'
        )

        SIDEWAYS_THRESHOLD = 1.0

        for i, ticker in enumerate(tickers):
            # Draw Ticker Rank
            rank_str = str(mf_tickers_list.index(ticker) + 1) if ticker in mf_tickers_list else '?'
            ax.text(
                i + 0.5, y_rank,
                rank_str,
                transform=ax.transData,
                ha='center', va='center',
                fontsize=11, fontweight='bold', color='black',
                clip_on=False
            )

            # Draw FDTS Signal
            if 'fdts_results' in locals() and ticker in fdts_results:
                stt = fdts_results[ticker]['state']
                dys = fdts_results[ticker]['days']
                if pd.isna(stt):
                    txt = "-"
                    tcolor = 'black'
                elif stt == 1:
                    txt = f"B({int(dys)})"
                    tcolor = '#008800'
                elif stt == -1:
                    txt = f"S({int(dys)})"
                    tcolor = '#BB0000'
                else:
                    txt = f"N({int(dys)})"
                    tcolor = '#888800'
            else:
                txt = "-"
                tcolor = 'black'
                
            ax.text(
                i + 0.5, y_fdts,
                txt,
                transform=ax.transData,
                ha='center', va='center',
                fontsize=10, fontweight='bold', color=tcolor,
                clip_on=False
            )

            # Draw Dots
            cum_val = cumulative.get(ticker, 0)
            if cum_val > SIDEWAYS_THRESHOLD:
                dot_color = '#00BB00'
            elif cum_val < -SIDEWAYS_THRESHOLD:
                dot_color = '#EE2222'
            else:
                dot_color = '#DDAA00'

            ax.text(
                i + 0.5, y_dots,
                '●',
                transform=ax.transData,
                ha='center', va='center',
                fontsize=14, color=dot_color,
                clip_on=False
            )


        # Increase bottom margin slightly to fit the new manual labels
        plt.subplots_adjust(bottom=0.25, top=0.9)

        universe_name = getattr(self, "mf_universe_name", "Selected Universe")
        title_str = f'{self.timeframe.upper()} {header_filter}: {universe_name.upper()}'
        plt.title(title_str, fontsize=20, fontweight='900', pad=80, color='#111111')
        
        plt.text(
            0.5, 1.12, 'Copyright © FazDane Analytics | Research & Trading Intelligence Platform',
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=10, fontstyle='italic',
            color='#666666'
        )

        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### 📋 Export Tickers")
        st.markdown("**Current View**")
        st.code(", ".join(tickers), language="text")
        
        st.markdown("**Full Sorted List**")
        st.code(", ".join(sorted_all.index.tolist()), language="text")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 25**")
            st.code(", ".join(sorted_all.head(25).index.tolist()), language="text")
        with col2:
            st.markdown("**Bottom 25**")
            st.code(", ".join(sorted_all.tail(25).index.tolist()), language="text")
