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
def fetch_money_flow_data(tickers, interval, days_mult, lookback):
    tickers = list(tickers)
    if not tickers:
        return pd.DataFrame()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback * days_mult)
    
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
        
    close_data = data['Close'] if 'Close' in data else data['Adj Close']
    
    # If there's only one ticker, it returns a Series. Convert to DataFrame.
    if isinstance(close_data, pd.Series):
        close_data = pd.DataFrame(close_data, columns=[tickers[0]])
        
    return close_data

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
        
        st.markdown("**View Filters**")
        self.filter_type = st.selectbox(
            "View Filter:",
            options=['Top 15', 'Bottom 10', 'Perf Range (Custom %)', 'Show All (Sorted)'],
            index=0
        )
        
        if self.filter_type == 'Perf Range (Custom %)':
            self.range_limits = st.slider("Range %:", min_value=-100.0, max_value=500.0, value=(5.0, 30.0), step=1.0)
        else:
            self.range_limits = None
            
        if st.button("🔄 Refresh Data", use_container_width=True):
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
            close_data = fetch_money_flow_data(
                tickers=initial_tickers, 
                interval=cfg['interval'], 
                days_mult=cfg['days_mult'], 
                lookback=self.lookback
            )
            
        if close_data.empty:
            st.warning("⚠️ No data found for the selected parameters.")
            return
            
        if self.timeframe == 'Yearly':
            close_data = close_data.resample('YE').last()
            
        returns = close_data.pct_change().dropna(how='all') * 100
        period_returns = returns.tail(self.lookback).fillna(0)
        if period_returns.empty:
            st.warning("Not enough return history for the selected universe and lookback.")
            return
        
        # True Cumulative
        cumulative = ((period_returns / 100 + 1).prod() - 1) * 100
        sorted_all = cumulative.sort_values(ascending=False)
        
        # Filtering
        if self.filter_type == 'Top 15':
            tickers = sorted_all.head(15).index.tolist()
            header_filter = "TOP 15"
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
        
        # -------- PLOT --------
        cfg = CONFIG[self.timeframe]
        calc_height = max(8, (len(final_df) + 2) * cfg.get('h_factor', 0.45))
        fig, ax = plt.subplots(figsize=(16, calc_height))

        # Explicitly set the background color of the figure and axes to transparent
        # so it blends with the Streamlit theme, but keep the core plot colors intact
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        sns.heatmap(
            final_df,
            annot=True,
            cmap='RdYlGn',
            center=0,
            fmt=".1f",
            linewidths=.5,
            cbar_kws={'shrink': 0.25},
            ax=ax
        )

        n_rows = len(final_df)
        n_tickers = len(tickers)

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

        ax.axhline(n_rows - 1, color='black', linewidth=4)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.xticks(rotation=0, ha='center', fontweight='bold', color='black')
        plt.yticks(color='black')

        ax_bottom = ax.secondary_xaxis('bottom')
        ax_bottom.set_xticks(range(len(tickers)))
        ax_bottom.set_xticklabels(tickers, fontweight='bold', fontsize=12, color='black')

        SIDEWAYS_THRESHOLD = 1.0
        blend = blended_transform_factory(ax.transData, ax.transAxes)

        for i, ticker in enumerate(tickers):
            cum_val = cumulative.get(ticker, 0)
            if cum_val > SIDEWAYS_THRESHOLD:
                dot_color = '#00BB00'
            elif cum_val < -SIDEWAYS_THRESHOLD:
                dot_color = '#EE2222'
            else:
                dot_color = '#DDAA00'

            ax.text(
                i + 0.5, -0.075,
                '●',
                transform=blend,
                ha='center', va='center',
                fontsize=14, color=dot_color,
                clip_on=False
            )

        for x_frac, label, color in [
            (0.20, '● Uptrend',   '#00BB00'),
            (0.50, '● Sideways',  '#DDAA00'),
            (0.80, '● Downtrend', '#EE2222'),
        ]:
            ax.text(
                x_frac, -0.115,
                label,
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color=color, clip_on=False
            )

        plt.subplots_adjust(bottom=0.25, top=0.9)

        universe_name = getattr(self, "mf_universe_name", "Selected Universe")
        title_str = f'{self.timeframe.upper()} {header_filter}: {universe_name.upper()}'
        plt.title(title_str, fontsize=18, fontweight='bold', pad=60, color='black')
        
        plt.text(
            1.0, -0.155, 'Copyright (c) FazDane Analytics | Research & Trading Intelligence Platform',
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=14, fontweight='bold',
            color='#444444', style='italic'
        )

        st.pyplot(fig)
