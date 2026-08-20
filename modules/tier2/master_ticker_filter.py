import streamlit as st
import pandas as pd
import sqlite3
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from modules.base_module import FazDaneModule
from utils.persistence import get_db_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class MasterTickerFilterModule(FazDaneModule):
    MODULE_NAME = "Master Ticker Universe Filter"
    MODULE_ICON = "🔍"
    MODULE_DESCRIPTION = "Filter deduplicated universe by technicals."
    TIER = 2

    def render_sidebar(self):
        st.markdown("**Filters**")
        self.min_rsi = st.slider("Min RSI", 0, 100, 0)
        self.max_rsi = st.slider("Max RSI", 0, 100, 100)
        self.macd_sig = st.selectbox("MACD Signal", ["Any", "Bullish", "Bearish"])
        self.cloud_sig = st.selectbox("Cloud Signal", ["Any", "Bullish", "Bearish", "Neutral"])
        self.fdts_sig = st.multiselect("FDTS Signal", ["Buy", "Sell", "Neutral"], default=["Buy", "Sell", "Neutral"])
        
        self.max_call_spread = st.number_input("Max Call Spread", min_value=0.0, max_value=50.0, value=50.0, step=0.1)
        self.min_price = st.number_input("Price Above", min_value=0.0, value=0.0, step=1.0)
        
        self.above_vwap = st.checkbox("Above VWAP", value=False)
        self.has_weeklys = st.checkbox("Has Weeklys", value=False)
        if st.button("Apply Filters", width="stretch"):
            pass

    def render_main(self):
        st.title(f"{self.MODULE_ICON} {self.MODULE_NAME}")
        
        # Check progress
        prog_file = PROJECT_ROOT / "data" / "master_analysis_progress.json"
        
        st.markdown("### Master Analysis Status")
        
        db_path = get_db_path("master_ticker_analysis")
        last_run_dt = "Unknown"
        if db_path.exists():
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT MAX(last_processed) FROM master_analysis")
                    res = cursor.fetchone()
                    if res and res[0]:
                        dt = datetime.fromisoformat(res[0])
                        last_run_dt = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
                
        col1, col2 = st.columns([2, 1])
        with col1:
            if prog_file.exists():
                try:
                    with open(prog_file, "r") as f:
                        status = json.load(f)
                    if status.get("status") == "running":
                        pct = int((status["current"] / max(status["total"], 1)) * 100)
                        st.info(f"Background Process Running: {pct}% ({status['current']}/{status['total']}) - Processing: {status['ticker']}")
                        if st.button("Refresh Status"):
                            st.rerun()
                    else:
                        st.success(f"Background Process Completed. Last processed: {last_run_dt}")
                except:
                    st.warning("Could not read progress file.")
            else:
                st.info("No master analysis run detected.")
                
        with col2:
            if st.button("Run Master Analysis", type="primary", use_container_width=True):
                script_path = PROJECT_ROOT / "scripts" / "run_master_analysis.py"
                try:
                    subprocess.Popen([sys.executable, str(script_path)])
                    st.success("Started background process!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start: {e}")
                    
        st.divider()
        if not db_path.exists():
            st.warning("Master database not found. Please run the analysis first.")
            return
            
        with sqlite3.connect(db_path) as conn:
            query = """
                SELECT m.ticker, m.price, m.next_earnings_date, m.fdts_signal, m.macd_signal, m.cloud_signal, m.above_vwap, m.has_weeklys, 
                       m.call_spread, m.put_spread, m.options_source, 
                       t.ma20, t.ma50, t.ma200, t.rsi14, t.adx14, t.macd_value, t.macd_hist, t.atr14, 
                       t.bollinger_upper, t.bollinger_lower, t.darvas_upper, t.darvas_lower, 
                       t.regression_upper, t.regression_lower, t.vwap, t.cvd, t.ichimoku_span_a, t.ichimoku_span_b
                FROM master_analysis m
                LEFT JOIN technical_indicators t ON m.ticker = t.ticker
                WHERE (t.rsi14 >= ? AND t.rsi14 <= ? OR t.rsi14 IS NULL)
            """
            params = [self.min_rsi, self.max_rsi]
            
            if self.macd_sig != "Any":
                query += " AND m.macd_signal = ?"
                params.append(self.macd_sig)

            if self.cloud_sig != "Any":
                query += " AND m.cloud_signal = ?"
                params.append(self.cloud_sig)

            if self.max_call_spread < 50.0:
                query += " AND (m.call_spread <= ? AND m.call_spread IS NOT NULL)"
                params.append(self.max_call_spread)
                
            if self.min_price > 0.0:
                query += " AND m.price > ?"
                params.append(self.min_price)
                
            if self.fdts_sig:
                placeholders = ','.join(['?']*len(self.fdts_sig))
                query += f" AND m.fdts_signal IN ({placeholders})"
                params.extend(self.fdts_sig)
                
            if self.above_vwap:
                query += " AND m.above_vwap = 1"
                
            if self.has_weeklys:
                query += " AND m.has_weeklys = 1"
                
            df = pd.read_sql_query(query, conn, params=params)
            
        st.markdown(f"**Results:** {len(df)} tickers found.")
        if not df.empty:
            col_a, col_b = st.columns([1, 1])
            with col_a:
                tickers_str = ",".join(df["ticker"].tolist())
                with st.expander("Copy Tickers (Comma Delimited)"):
                    st.code(tickers_str, language="text")
            with col_b:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Export to CSV (Excel)",
                    data=csv,
                    file_name="master_ticker_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        st.dataframe(df, use_container_width=True, hide_index=True, height=1200)
