import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from modules.base_module import FazDaneModule
from utils.universe_manager import render_universe_manager, get_ticker_names
from utils.backtest_store import get_historical_data, save_backtest_result, get_latest_backtest_result, get_latest_option_spreads

class FDTSBacktestModule(FazDaneModule):
    MODULE_NAME = "FDTS Regime Backtester"
    MODULE_ICON = "📈"
    TIER = 5

    def __init__(self):
        super().__init__()
        self.period = 20
        self.macd_fast_long = 3
        self.macd_slow_long = 10
        self.macd_signal_long = 16
        self.macd_fast_short = 12
        self.macd_slow_short = 26
        self.macd_signal_short = 9

    def render_sidebar(self):
        pass

    def render_main(self):
        st.markdown(
            """
            <div style="background:linear-gradient(135deg, rgba(26,58,143,0.3) 0%, rgba(58,181,74,0.1) 100%);
                        border:1px solid #1e3a5f; border-left:4px solid #3ab54a;
                        border-radius:12px; padding:18px 22px; margin-bottom:24px;">
                <div style="color:#3ab54a;font-size:20px;font-weight:700;margin-bottom:4px;">FDTS Regime Backtester</div>
                <div style="color:#94a3b8;font-size:14px;">Backtest FDTS + Dual MACD + Ichimoku regimes (24-Month History)</div>
            </div>
            """, unsafe_allow_html=True
        )

        tab_single, tab_bulk = st.tabs(["Single Ticker Analysis", "Bulk Universe Scanner"])
        
        with tab_single:
            self._render_single_tab()
            
        with tab_bulk:
            self._render_bulk_tab()

    def _render_single_tab(self):
        col1, col2 = st.columns([1, 1])
        with col1:
            universe_name, tickers, _ = render_universe_manager(key_prefix="backtest_univ")
        with col2:
            ticker_names = get_ticker_names(universe_name)
            selected_ticker = st.selectbox(
                "Select Ticker",
                options=tickers,
                format_func=lambda x: f"{x} - {ticker_names.get(x, x)}" if ticker_names.get(x) else x
            )

        st.divider()

        kpis, df_last, last_run_timestamp = get_latest_backtest_result(selected_ticker)
        
        col_run1, col_run2 = st.columns([1, 3])
        with col_run1:
            run_btn = st.button("Run Backtest", type="primary", use_container_width=True)
        with col_run2:
            if last_run_timestamp:
                try:
                    dt = datetime.fromisoformat(last_run_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    st.info(f"Last backtest run: {dt}. Click 'Run Backtest' to force update.")
                except:
                    st.info("Previous backtest data available.")
            else:
                st.info("No previous backtest data found for this ticker.")
                
        if run_btn:
            with st.spinner(f"Running backtest for {selected_ticker}..."):
                df = get_historical_data(selected_ticker, months=24)
                if df.empty or len(df) < 50:
                    st.error("Not enough historical data to run backtest.")
                    return
                    
                df, kpis = self._run_backtest(df)
                save_backtest_result(selected_ticker, kpis, df)
                st.success("Backtest completed successfully!")
                
                # Update current view
                df_last = df

        if df_last is not None and kpis is not None:
            self._render_dashboard(selected_ticker, kpis, df_last)

    def _render_bulk_tab(self):
        st.markdown("### Bulk Universe Backtest")
        st.markdown("Run the backtest on an entire universe and segment the results by Market Regime.")
        
        col_univ, col_filt, col_btn = st.columns([2, 1, 1])
        with col_univ:
            universe_name, tickers, _ = render_universe_manager(key_prefix="bulk_univ")
        with col_filt:
            max_spread = st.number_input("Max Option Spread ($)", value=0.50, step=0.10)
        with col_btn:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            run_bulk_btn = st.button("Run Bulk Backtest", type="primary", use_container_width=True)
            
        if run_bulk_btn:
            if not tickers:
                st.warning("No tickers in selected universe.")
                return
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            bulk_results = []
            
            for i, ticker in enumerate(tickers):
                status_text.text(f"Processing {ticker} ({i+1}/{len(tickers)})...")
                df = get_historical_data(ticker, months=24)
                if not df.empty and len(df) >= 50:
                    df, kpis = self._run_backtest(df)
                    save_backtest_result(ticker, kpis, df)
                    
                    bulk_results.append({
                        "Ticker": ticker,
                        "Regime": kpis.get("Current Regime", "N/A"),
                        "Current VIX": kpis.get("Current VIX", 0.0),
                        "Days In Regime": df['DaysInState'].iloc[-1] if not df.empty else 0,
                        "Max Points": kpis.get(f"Max Points ({'Buy' if kpis.get('Current Regime') == 'BUY' else 'Sell'})", 0.0)
                    })
                
                progress_bar.progress((i + 1) / len(tickers))
                
            status_text.text("Fetching Option Spreads from Database...")
            spreads = get_latest_option_spreads(tickers)
            
            for row in bulk_results:
                tick = row["Ticker"]
                row["Call Spread"] = spreads.get(tick, {}).get("call_spread", 0.0)
                row["Put Spread"] = spreads.get(tick, {}).get("put_spread", 0.0)
                row["Max Spread"] = max(row["Call Spread"], row["Put Spread"])
                
            status_text.empty()
            progress_bar.empty()
            st.session_state['bulk_backtest_results'] = bulk_results
            
        if 'bulk_backtest_results' in st.session_state:
            results_df = pd.DataFrame(st.session_state['bulk_backtest_results'])
            
            # Apply Filter
            filtered_df = results_df[results_df['Max Spread'] <= max_spread]
            
            st.markdown(f"**Showing {len(filtered_df)} out of {len(results_df)} tickers with Option Spread <= ${max_spread}**")
            
            buy_df = filtered_df[filtered_df['Regime'] == 'BUY'].drop(columns=['Regime', 'Max Spread'])
            sell_df = filtered_df[filtered_df['Regime'] == 'SELL'].drop(columns=['Regime', 'Max Spread'])
            side_df = filtered_df[filtered_df['Regime'] == 'NO TRADE'].drop(columns=['Regime', 'Max Spread'])
            
            with st.expander(f"🟢 Current BUY Tickers ({len(buy_df)})", expanded=True):
                if not buy_df.empty:
                    st.dataframe(buy_df.sort_values(by="Max Points", ascending=False), use_container_width=True, hide_index=True)
                    st.code(", ".join(buy_df['Ticker'].tolist()))
                else:
                    st.info("No BUY tickers match the criteria.")
                    
            with st.expander(f"🔴 Current SELL Tickers ({len(sell_df)})", expanded=True):
                if not sell_df.empty:
                    st.dataframe(sell_df.sort_values(by="Max Points", ascending=False), use_container_width=True, hide_index=True)
                    st.code(", ".join(sell_df['Ticker'].tolist()))
                else:
                    st.info("No SELL tickers match the criteria.")
                    
            with st.expander(f"🟡 Current NO TRADE Tickers ({len(side_df)})", expanded=True):
                if not side_df.empty:
                    st.dataframe(side_df.sort_values(by="Days In Regime", ascending=False), use_container_width=True, hide_index=True)
                    st.code(", ".join(side_df['Ticker'].tolist()))
                else:
                    st.info("No NO TRADE tickers match the criteria.")

    def _run_backtest(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        df = df.copy()
        
        # 1. Price
        df['Price'] = (df['high'] + df['low'] + df['close']) / 3
        
        def ema(series, span):
            return series.ewm(span=span, adjust=False).mean()
        
        # 2. FDTS
        TMA1 = 3 * ema(df['Price'], self.period) - 3 * ema(ema(df['Price'], self.period), self.period) + ema(ema(ema(df['Price'], self.period), self.period), self.period)
        TMA2 = 3 * ema(TMA1, self.period) - 3 * ema(ema(TMA1, self.period), self.period) + ema(ema(ema(TMA1, self.period), self.period), self.period)
        TypicalTEMA = TMA1 + (TMA1 - TMA2)
        
        # Heikin-Ashi
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_open = np.zeros(len(df))
        ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
        
        df['haOpen'] = ha_open
        df['haClose'] = (ha_close + df['haOpen'] + df[['high', 'haOpen']].max(axis=1) + df[['low', 'haOpen']].min(axis=1)) / 4
        
        HATMA1 = 3 * ema(df['haClose'], self.period) - 3 * ema(ema(df['haClose'], self.period), self.period) + ema(ema(ema(df['haClose'], self.period), self.period), self.period)
        HATMA2 = 3 * ema(HATMA1, self.period) - 3 * ema(ema(HATMA1, self.period), self.period) + ema(ema(ema(HATMA1, self.period), self.period), self.period)
        HATEMA = HATMA1 + (HATMA1 - HATMA2)
        
        df['FDTS_Dev'] = TypicalTEMA - HATEMA
        
        # 3. MACD
        macdLong = ema(df['close'], self.macd_fast_long) - ema(df['close'], self.macd_slow_long)
        df['macdLongDev'] = macdLong - ema(macdLong, self.macd_signal_long)
        
        macdShort = ema(df['close'], self.macd_fast_short) - ema(df['close'], self.macd_slow_short)
        df['macdShortDev'] = macdShort - ema(macdShort, self.macd_signal_short)
        
        # 4. Ichimoku
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        tenkan = (high_9 + low_9) / 2
        
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        kijun = (high_26 + low_26) / 2
        
        span_a = ((tenkan + kijun) / 2).shift(26)
        
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        span_b = ((high_52 + low_52) / 2).shift(26)
        
        df['CloudMax'] = pd.concat([span_a, span_b], axis=1).max(axis=1)
        df['CloudMin'] = pd.concat([span_a, span_b], axis=1).min(axis=1)
        
        # 5. Signals
        buy_cond = (df['FDTS_Dev'] > 0) & (df['macdLongDev'] > 0) & (df['close'] > df['CloudMax'])
        sell_cond = (df['FDTS_Dev'] < 0) & (df['macdShortDev'] < 0) & (df['close'] < df['CloudMin'])
        
        df['State'] = 0
        df.loc[buy_cond, 'State'] = 1
        df.loc[sell_cond, 'State'] = -1
        
        # 6. Regime Tracking
        df['State_Changed'] = df['State'] != df['State'].shift(1)
        
        # We need to iteratively track Days, Entry Price, Delta
        days = np.zeros(len(df))
        entry = np.zeros(len(df))
        points = np.zeros(len(df))
        
        curr_state = 0
        curr_days = 0
        curr_entry = 0.0
        
        for i in range(len(df)):
            if pd.isna(df['close'].iloc[i]):
                continue
                
            state = df['State'].iloc[i]
            
            if df['State_Changed'].iloc[i]:
                curr_days = 1
                curr_entry = df['close'].iloc[i]
            else:
                curr_days += 1
                
            days[i] = curr_days
            entry[i] = curr_entry
            
            if state == 1:
                points[i] = df['close'].iloc[i] - curr_entry
            elif state == -1:
                points[i] = curr_entry - df['close'].iloc[i]
            else:
                points[i] = 0.0
                
        df['DaysInState'] = days
        df['EntryPrice'] = entry
        df['PointsEarned'] = points
        
        # Map state to string for display
        state_map = {1: 'BUY', -1: 'SELL', 0: 'NO TRADE'}
        df['Regime'] = df['State'].map(state_map)
        
        # Add VIX Data
        vix_df = get_historical_data('^VIX', months=24)
        if not vix_df.empty:
            vix_df = vix_df[['date', 'close']].rename(columns={'close': 'VIX_Close'})
            df = pd.merge(df, vix_df, on='date', how='left')
            # Forward fill VIX if there are missing days
            df['VIX_Close'] = df['VIX_Close'].ffill()
        else:
            df['VIX_Close'] = np.nan
        
        # Calculate true average trade durations
        df['Trade_ID'] = df['State_Changed'].cumsum()
        trade_lengths = df.groupby(['State', 'Trade_ID'])['DaysInState'].max().reset_index()
        avg_buy_days = trade_lengths[trade_lengths['State'] == 1]['DaysInState'].mean()
        avg_sell_days = trade_lengths[trade_lengths['State'] == -1]['DaysInState'].mean()
        
        # KPIs
        buy_periods = df[df['State'] == 1]
        sell_periods = df[df['State'] == -1]
        sideways_periods = df[df['State'] == 0]
        
        avg_buy_vix = buy_periods['VIX_Close'].mean() if not buy_periods.empty else np.nan
        avg_sell_vix = sell_periods['VIX_Close'].mean() if not sell_periods.empty else np.nan
        avg_side_vix = sideways_periods['VIX_Close'].mean() if not sideways_periods.empty else np.nan
        
        total_days = len(df)
        kpis = {
            "Total Days": total_days,
            "Bullish Days": len(buy_periods),
            "Bearish Days": len(sell_periods),
            "Sideways Days": len(sideways_periods),
            "Bullish %": round(len(buy_periods) / total_days * 100, 1) if total_days > 0 else 0,
            "Bearish %": round(len(sell_periods) / total_days * 100, 1) if total_days > 0 else 0,
            "Sideways %": round(len(sideways_periods) / total_days * 100, 1) if total_days > 0 else 0,
            "Max Points (Buy)": round(buy_periods['PointsEarned'].max(), 2) if not buy_periods.empty else 0,
            "Max Points (Sell)": round(sell_periods['PointsEarned'].max(), 2) if not sell_periods.empty else 0,
            "Average Trade Days (Buy)": round(avg_buy_days, 1) if not pd.isna(avg_buy_days) else 0,
            "Average Trade Days (Sell)": round(avg_sell_days, 1) if not pd.isna(avg_sell_days) else 0,
            "Avg VIX in Bullish": round(avg_buy_vix, 2) if not pd.isna(avg_buy_vix) else 0,
            "Avg VIX in Bearish": round(avg_sell_vix, 2) if not pd.isna(avg_sell_vix) else 0,
            "Avg VIX in Sideways": round(avg_side_vix, 2) if not pd.isna(avg_side_vix) else 0,
            "Current VIX": round(df['VIX_Close'].iloc[-1], 2) if not pd.isna(df['VIX_Close'].iloc[-1]) else 0,
            "Current Regime": df['Regime'].iloc[-1]
        }
        
        return df, kpis

    def _render_dashboard(self, ticker: str, kpis: dict, df: pd.DataFrame):
        st.markdown("### Backtest Results")
        
        # KPI Cards
        cols = st.columns(4)
        cols[0].metric("Market Condition", "Bullish", f"{kpis['Bullish Days']} Days ({kpis['Bullish %']}%)")
        cols[1].metric("Market Condition", "Bearish", f"{kpis['Bearish Days']} Days ({kpis['Bearish %']}%)")
        cols[2].metric("Market Condition", "Sideways", f"{kpis['Sideways Days']} Days ({kpis['Sideways %']}%)")
        cols[3].metric("Total Days Analyzed", str(kpis['Total Days']))
        
        cols2 = st.columns(4)
        cols2[0].metric("Max Points (Buy)", str(kpis['Max Points (Buy)']))
        cols2[1].metric("Max Points (Sell)", str(kpis['Max Points (Sell)']))
        cols2[2].metric("Avg Trade Length (Buy)", f"{kpis['Average Trade Days (Buy)']} Days")
        cols2[3].metric("Avg Trade Length (Sell)", f"{kpis['Average Trade Days (Sell)']} Days")

        st.divider()
        
        st.markdown("### VIX Correlation & Trade Recommendations")
        vix_cols = st.columns(4)
        vix_cols[0].metric("Avg VIX (Bullish)", str(kpis['Avg VIX in Bullish']))
        vix_cols[1].metric("Avg VIX (Bearish)", str(kpis['Avg VIX in Bearish']))
        vix_cols[2].metric("Avg VIX (Sideways)", str(kpis['Avg VIX in Sideways']))
        vix_cols[3].metric("Current VIX", str(kpis['Current VIX']))
        
        # Recommendations
        st.markdown("#### Strategic Insights")
        curr_regime = kpis["Current Regime"]
        curr_vix = kpis["Current VIX"]
        avg_side = kpis['Sideways %']
        
        rec = ""
        if curr_regime == "NO TRADE":
            if curr_vix > 20:
                rec = f"<b>Current State:</b> {curr_regime} with Elevated VIX ({curr_vix}).<br><b>Recommendation:</b> Iron Condors or wide neutral premium selling. The market is choppy but premiums are rich."
            else:
                rec = f"<b>Current State:</b> {curr_regime} with Low VIX ({curr_vix}).<br><b>Recommendation:</b> Calendar Spreads (e.g. 20/40 days 25 delta). This ticker spends {avg_side}% of its time sideways, making it an excellent candidate for neutral, long-vega calendars while IV is low."
        elif curr_regime == "BUY":
            if curr_vix > 20:
                rec = f"<b>Current State:</b> Bullish with Elevated VIX ({curr_vix}).<br><b>Recommendation:</b> Put Credit Spreads. Capitalize on the upward trend while selling the inflated downside premium."
            else:
                rec = f"<b>Current State:</b> Bullish with Low VIX ({curr_vix}).<br><b>Recommendation:</b> Call Debit Spreads or Long Calls. Volatility is cheap and momentum is upward."
        elif curr_regime == "SELL":
            if curr_vix > 20:
                rec = f"<b>Current State:</b> Bearish with Elevated VIX ({curr_vix}).<br><b>Recommendation:</b> Bear Call Credit Spreads. Volatility is already expanded; better to sell premium on the rallies than buy expensive puts."
            else:
                rec = f"<b>Current State:</b> Bearish with Low VIX ({curr_vix}).<br><b>Recommendation:</b> Put Debit Spreads. Good time to buy downside protection or speculate short while VIX is relatively low."

        color_map = {
            "BUY": ("rgba(58, 181, 74, 0.15)", "#3ab54a"),
            "SELL": ("rgba(239, 68, 68, 0.15)", "#ef4444"),
            "NO TRADE": ("rgba(234, 179, 8, 0.15)", "#eab308")
        }
        bg_color, border_color = color_map.get(curr_regime, ("rgba(255,255,255,0.1)", "#ffffff"))
        
        st.markdown(
            f'''
            <div style="background-color: {bg_color}; border-left: 4px solid {border_color}; padding: 16px; border-radius: 8px; margin-top: 10px;">
                <span style="font-size: 15px; line-height: 1.5;">{rec}</span>
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.divider()

        # Plotly Chart
        st.markdown("### Historical Regimes & Price Action")
        fig = go.Figure()
        
        # Main price line
        fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Close Price', line=dict(color='#94a3b8', width=1.5)))
        
        # Highlight Regimes
        buy_df = df[df['State'] == 1]
        sell_df = df[df['State'] == -1]
        
        fig.add_trace(go.Scatter(x=buy_df['date'], y=buy_df['close'], mode='markers', name='Buy Regime', marker=dict(color='#3ab54a', size=6, opacity=0.8)))
        fig.add_trace(go.Scatter(x=sell_df['date'], y=sell_df['close'], mode='markers', name='Sell Regime', marker=dict(color='#ef4444', size=6, opacity=0.8)))
        
        fig.update_layout(
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        
        # Download Data
        st.markdown("### Export Daily History")
        export_df = df[['date', 'open', 'high', 'low', 'close', 'Regime', 'DaysInState', 'EntryPrice', 'PointsEarned']].copy()
        
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Historical Indicator Data as CSV",
            data=csv,
            file_name=f"{ticker}_backtest_results.csv",
            mime="text/csv",
        )
        
        with st.expander("Preview Data"):
            st.dataframe(export_df.tail(20), use_container_width=True)

