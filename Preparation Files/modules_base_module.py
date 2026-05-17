"""
FazDane Base Module Class
Template for all module implementations
"""

from abc import ABC, abstractmethod
import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import Any, Dict, Optional

class FazDaneModule(ABC):
    """
    Abstract base class for all FazDane modules
    
    Every module must inherit from this class and implement:
    - MODULE_NAME: Display name
    - MODULE_ICON: Emoji icon
    - TIER: Priority tier (1-4)
    - render_sidebar(): Control panel
    - render_main(): Main content area
    """
    
    # ─────────────────────────────────────────────────────────────────────
    # MODULE METADATA (Override in subclass)
    # ─────────────────────────────────────────────────────────────────────
    
    MODULE_NAME: str = "Base Module"
    MODULE_ICON: str = "📊"
    MODULE_DESCRIPTION: str = "Base module description"
    TIER: int = 1
    SOURCE_NOTEBOOK: str = "Unknown"
    
    # ─────────────────────────────────────────────────────────────────────
    # CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────
    
    REQUIRES_LIVE_DATA: bool = False
    CACHE_TTL: int = 3600  # 1 hour
    REFRESH_INTERVAL: int = 60  # seconds
    DATA_SOURCES: list = []  # List of data sources (e.g., 'yfinance', 'api', 'csv')
    DEPENDENCIES: list = []  # List of required packages
    
    def __init__(self):
        """Initialize module"""
        self.logger = self._setup_logger()
        self.data_cache: Dict[str, Any] = {}
        self.last_refresh: Dict[str, datetime] = {}
        self.module_params: Dict[str, Any] = {}
        
        logger.info(f"Initialized {self.MODULE_NAME} module")
    
    # ═════════════════════════════════════════════════════════════════════
    # ABSTRACT METHODS (Must implement in subclass)
    # ═════════════════════════════════════════════════════════════════════
    
    @abstractmethod
    def render_sidebar(self):
        """
        Render sidebar control panel
        
        Example:
        --------
        st.markdown("### Search Parameters")
        self.symbol = st.text_input("Symbol", value="SPY")
        self.period = st.selectbox("Period", ["1D", "1W", "1M"])
        """
        pass
    
    @abstractmethod
    def render_main(self):
        """
        Render main content area
        
        Example:
        --------
        st.markdown("## Main Content")
        data = self.fetch_data(self.symbol)
        st.dataframe(data)
        """
        pass
    
    # ═════════════════════════════════════════════════════════════════════
    # PUBLIC INTERFACE
    # ═════════════════════════════════════════════════════════════════════
    
    def run(self):
        """Main entry point - Execute module"""
        
        # Set page config
        st.set_page_config(
            page_title=f"{self.MODULE_ICON} {self.MODULE_NAME}",
            page_icon=self.MODULE_ICON,
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # ─────────────────────────────────────────────────────────────────
        # SIDEBAR
        # ─────────────────────────────────────────────────────────────────
        with st.sidebar:
            st.markdown(f"## {self.MODULE_ICON} {self.MODULE_NAME}")
            st.markdown(f"*{self.MODULE_DESCRIPTION}*")
            st.markdown(f"**Source**: {self.SOURCE_NOTEBOOK}")
            st.markdown(f"**Tier**: {self.TIER}")
            
            st.divider()
            
            # Render module-specific sidebar
            self.render_sidebar()
            
            st.divider()
            
            # Refresh controls
            st.markdown("### Data Controls")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh", use_container_width=True, help="Fetch fresh data"):
                    st.cache_data.clear()
                    st.rerun()
            
            with col2:
                if st.button("⚙️ Settings", use_container_width=True, help="Module settings"):
                    st.session_state.show_module_settings = True
        
        # ─────────────────────────────────────────────────────────────────
        # MAIN CONTENT
        # ─────────────────────────────────────────────────────────────────
        
        try:
            self.render_main()
        except Exception as e:
            self.logger.error(f"Error rendering {self.MODULE_NAME}: {str(e)}")
            st.error(f"❌ Error loading module: {str(e)}")
            st.info("💡 Try refreshing the page or adjusting your parameters")
    
    # ═════════════════════════════════════════════════════════════════════
    # DATA FETCHING WITH CACHING
    # ═════════════════════════════════════════════════════════════════════
    
    @st.cache_data(ttl=3600)
    def fetch_data(self, source: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch data with caching
        
        Parameters
        ----------
        source : str
            Data source ('yfinance', 'api', 'csv', etc.)
        **kwargs
            Source-specific parameters
        
        Returns
        -------
        pd.DataFrame or None
        """
        try:
            self.logger.info(f"Fetching data from {source} with params: {kwargs}")
            
            if source == "yfinance":
                return self._fetch_yfinance(**kwargs)
            elif source == "csv":
                return self._fetch_csv(**kwargs)
            elif source == "api":
                return self._fetch_api(**kwargs)
            else:
                raise ValueError(f"Unknown source: {source}")
        
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            st.error(f"Failed to fetch data: {str(e)}")
            return None
    
    # ═════════════════════════════════════════════════════════════════════
    # DATA SOURCE IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════
    
    def _fetch_yfinance(self, **kwargs) -> pd.DataFrame:
        """Fetch data from yfinance"""
        import yfinance as yf
        
        symbol = kwargs.get('symbol', 'SPY')
        start = kwargs.get('start')
        end = kwargs.get('end')
        interval = kwargs.get('interval', '1d')
        
        data = yf.download(symbol, start=start, end=end, interval=interval)
        self.logger.info(f"Downloaded {symbol}: {len(data)} rows")
        
        return data
    
    def _fetch_csv(self, **kwargs) -> pd.DataFrame:
        """Fetch data from CSV file"""
        filepath = kwargs.get('filepath')
        
        if not filepath:
            raise ValueError("filepath required for CSV source")
        
        data = pd.read_csv(filepath)
        self.logger.info(f"Loaded CSV: {len(data)} rows")
        
        return data
    
    def _fetch_api(self, **kwargs) -> pd.DataFrame:
        """Fetch data from API"""
        # Implement per module
        raise NotImplementedError("API fetch must be implemented in subclass")
    
    # ═════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═════════════════════════════════════════════════════════════════════
    
    def _setup_logger(self) -> logging.Logger:
        """Setup module logger"""
        logger = logging.getLogger(self.MODULE_NAME)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(f"logs/{self.MODULE_NAME}.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def display_metrics(self, metrics: Dict[str, tuple]):
        """
        Display metric cards
        
        Parameters
        ----------
        metrics : Dict[str, tuple]
            Format: {"Label": (value, delta, unit)}
        
        Example
        -------
        self.display_metrics({
            "Total Volume": (1234567, 120000, ""),
            "IV Rank": (67, 5, "%"),
            "Daily Return": (0.45, -0.2, "%"),
        })
        """
        cols = st.columns(len(metrics))
        
        for idx, (label, (value, delta, unit)) in enumerate(metrics.items()):
            with cols[idx]:
                st.metric(
                    label=label,
                    value=f"{value:,.2f}{unit}" if isinstance(value, (int, float)) else value,
                    delta=f"{delta:+.2f}{unit}" if delta else None
                )
    
    def display_dataframe(self, df: pd.DataFrame, title: str = "Data", **kwargs):
        """
        Display dataframe with formatting
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to display
        title : str
            Chart title
        **kwargs
            Additional st.dataframe parameters
        """
        st.markdown(f"### {title}")
        st.dataframe(df, use_container_width=True, **kwargs)
    
    def display_chart(self, fig, title: str = "Chart"):
        """
        Display plotly chart
        
        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Chart to display
        title : str
            Chart title
        """
        st.markdown(f"### {title}")
        st.plotly_chart(fig, use_container_width=True)
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate module parameters
        
        Override in subclass for custom validation
        """
        return True
    
    def get_module_info(self) -> Dict[str, Any]:
        """Get module metadata"""
        return {
            "name": self.MODULE_NAME,
            "icon": self.MODULE_ICON,
            "description": self.MODULE_DESCRIPTION,
            "tier": self.TIER,
            "source": self.SOURCE_NOTEBOOK,
            "cache_ttl": self.CACHE_TTL,
            "data_sources": self.DATA_SOURCES,
            "dependencies": self.DEPENDENCIES,
        }

# ═════════════════════════════════════════════════════════════════════════
# LOGGER FOR BASE CLASS
# ═════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("FazDaneModule")
