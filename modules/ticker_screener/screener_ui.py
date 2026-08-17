import streamlit as st
import pandas as pd
from modules.base_module import FazDaneModule
from utils.universe_manager import load_universes
from modules.ticker_screener.screener_engine import check_volume_and_options

class TickerScreenerModule(FazDaneModule):
    MODULE_NAME = "High Liquidity Ticker Screener"
    MODULE_ICON = "⚡"
    MODULE_DESCRIPTION = "Scan for tickers with >5M average volume and >$1.50 option liquidity."
    TIER = 2

    def render_sidebar(self):
        st.markdown("**Screener Configuration**")
        
        universes = load_universes()
        universe_names = [name for name in universes.keys() if not name.startswith("__")]
        
        self.selected_universes = st.multiselect(
            "Select Universes to Scan",
            options=universe_names,
            default=universe_names[:1] if universe_names else None
        )
        
        self.min_volume = st.number_input("Minimum Average Volume", value=5000000, step=1000000)
        self.min_premium = st.number_input("Minimum Option Premium ($)", value=1.50, step=0.50)
        self.max_spread = st.number_input("Maximum Bid/Ask Spread (%)", value=10.0, step=1.0)
        
        if st.button("Run Screen", width="stretch", type="primary"):
            st.session_state["run_liquidity_screen"] = True

    def render_main(self):
        st.title(f"{self.MODULE_ICON} {self.MODULE_NAME}")
        st.markdown(self.MODULE_DESCRIPTION)
        st.divider()
        
        if "screener_results" not in st.session_state:
            st.session_state["screener_results"] = None
            
        if st.session_state.get("run_liquidity_screen"):
            st.session_state["run_liquidity_screen"] = False
            
            if not getattr(self, "selected_universes", None):
                st.warning("Please select at least one universe from the sidebar.")
                return
                
            universes = load_universes()
            all_tickers = set()
            for u in self.selected_universes:
                for t in universes[u].get("tickers", []):
                    all_tickers.add(t)
                    
            tickers = sorted(list(all_tickers))
            
            if not tickers:
                st.warning("No tickers found in selected universes.")
                return
                
            st.info(f"Scanning {len(tickers)} tickers. This may take a moment as it fetches real-time options data.")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for i, ticker in enumerate(tickers):
                status_text.text(f"Scanning {ticker} ({i+1}/{len(tickers)})...")
                
                # In a real heavy-duty app this might be a background task, 
                # but for a targeted universe Streamlit can handle it synchronously.
                res = check_volume_and_options(ticker, min_volume=self.min_volume, min_premium=self.min_premium, max_spread_pct=self.max_spread)
                
                if res:
                    results.append(res)
                        
                progress_bar.progress((i + 1) / len(tickers))
                
            status_text.text("Scan complete!")
            st.session_state["screener_results"] = results
            st.rerun()

        results = st.session_state.get("screener_results")
        
        if results is not None:
            if len(results) == 0:
                st.warning("No tickers met the liquidity and volume criteria.")
            else:
                df = pd.DataFrame(results)
                st.success(f"Found {len(df)} highly liquid tickers.")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    tickers_str = ",".join(df["Ticker"].tolist())
                    with st.expander("Copy Tickers"):
                        st.code(tickers_str, language="text")
                with col2:
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Excel (CSV)",
                        data=csv,
                        file_name=f"liquidity_screener_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                st.dataframe(df, use_container_width=True, hide_index=True)
