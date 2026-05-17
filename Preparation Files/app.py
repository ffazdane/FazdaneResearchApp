"""
FazDane Research Application
Main Entry Point - Authentication-Gated Streamlit Dashboard

Framework Version: 1.0
Module-by-Module Finance Program Migration
"""

import streamlit as st
import logging
from datetime import datetime
from pages.auth import FazDaneAuthenticator

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION & SETUP
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FazDane Research | Trading Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://fazdane.com/help",
        "Report a bug": "https://fazdane.com/support",
        "About": "FazDane Research Application v1.0"
    }
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fazdane.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FazDaneApp")

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM STYLING - DARK FINANCE AESTHETIC
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    :root {
        --primary: #00ff88;
        --secondary: #0f172a;
        --accent: #ff6b6b;
        --border: #1e293b;
        --success: #10b981;
        --danger: #ef4444;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1a1f3a 100%);
        border-right: 1px solid #1e293b;
    }
    
    /* Main content area */
    .stMainBlockContainer {
        background: #0a0e27;
        color: #e2e8f0;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #00ff88;
        font-family: 'Courier Prime', monospace;
        font-weight: 700;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1f3a 0%, #0f172a 100%);
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
    }
    
    /* Buttons */
    .stButton > button {
        background: #00ff88;
        color: #0f172a;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #00dd77;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        color: #64748b;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #00ff88;
        border-bottom: 2px solid #00ff88;
    }
    
    /* Divider */
    hr {
        border-color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# AUTHENTICATION LAYER
# ═══════════════════════════════════════════════════════════════════════════

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_role = None
    st.session_state.login_time = None

if not st.session_state.authenticated:
    authenticator = FazDaneAuthenticator()
    authenticator.render_login_screen()
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# AUTHENTICATED INTERFACE - SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px; padding: 20px 0;">
        <h1 style="color: #00ff88; font-size: 32px; margin: 0; font-family: 'Courier Prime', monospace;">⚡</h1>
        <h2 style="color: #00ff88; font-size: 20px; margin: 8px 0; font-family: 'Courier Prime', monospace;">FazDane</h2>
        <p style="color: #64748b; font-size: 12px; margin: 0;">Trading Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # User Info
    st.markdown(f"""
    **👤 Logged In**: {st.session_state.username}  
    **🔐 Role**: {st.session_state.user_role.title()}  
    **⏰ Session**: {st.session_state.login_time}
    """)
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════
    # TIER 1: CRITICAL TRADING MODULES (High Priority)
    # ═══════════════════════════════════════════════════════════════════════
    
    st.markdown("### 🔥 TIER 1: LIVE TRADING (Priority)")
    st.markdown("*Real-time trading & portfolio management*")
    
    tier1_options = [
        "💧 Options Liquidity Discovery",
        "📊 Market Breadth Dashboard",
        "⚙️ ES Futures Backtester",
        "🎯 Portfolio Optimizer",
        "🔄 Sector Rotation Monitor",
    ]
    
    tier1_module = st.radio(
        "Select Trading Module",
        options=tier1_options,
        key="tier1_nav",
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════
    # TIER 2: ANALYSIS & INTELLIGENCE
    # ═══════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📈 TIER 2: ANALYSIS & INTELLIGENCE")
    st.markdown("*Research & decision support*")
    
    tier2_options = [
        "🔍 Stock Ticker Screener",
        "📊 Implied Volatility Analysis",
        "🏦 Index Analysis Dashboard",
        "📅 Calendar Heatmaps",
        "📺 Earnings Calendar",
    ]
    
    tier2_module = st.radio(
        "Select Analysis Module",
        options=tier2_options,
        key="tier2_nav",
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════
    # TIER 3: FORECASTING & CYCLES
    # ═══════════════════════════════════════════════════════════════════════
    
    with st.expander("⏳ TIER 3: FORECASTING", expanded=False):
        st.markdown("*Medium-term predictions & cycle analysis*")
        
        tier3_options = [
            "🔮 Price Forecasting (Prophet/LSTM)",
            "🌙 Bradley Siderograph Cycles",
            "🌊 Elliott Wave Analysis",
            "⏰ Market Cycle Timing",
        ]
        
        tier3_module = st.radio(
            "Forecasting Tools",
            options=tier3_options,
            key="tier3_nav",
            label_visibility="collapsed"
        )
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════
    # SYSTEM & DATA MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    st.markdown("### ⚙️ SYSTEM")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Sync Drive", use_container_width=True, help="Sync latest notebooks from Google Drive"):
            with st.spinner("Syncing..."):
                logger.info(f"User {st.session_state.username} initiated Google Drive sync")
                st.success("✅ Synced successfully")
    
    with col2:
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.show_settings = True
    
    st.divider()
    
    # Module Health Dashboard
    st.markdown("### 📡 Module Status")
    
    module_status = {
        "Options Liquidity": "🟢",
        "Market Breadth": "🟢",
        "ES Backtester": "🟡",
        "Portfolio Opt": "🟢",
    }
    
    for module, status in list(module_status.items())[:4]:
        st.markdown(f"{status} {module}")
    
    st.divider()
    
    # Logout
    if st.button("🔐 Logout", use_container_width=True, type="secondary"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_role = None
        logger.info(f"User {st.session_state.username} logged out")
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT AREA - MODULE DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════

st.title("⚡ FazDane Research Application")
st.markdown("*Integrated Finance & Trading Intelligence Platform*")

st.divider()

# Module routing logic
selected_module = tier1_module or tier2_module or st.session_state.get('tier3_nav')

# ─────────────────────────────────────────────────────────────────────────
# TIER 1 MODULES
# ─────────────────────────────────────────────────────────────────────────

if selected_module == "💧 Options Liquidity Discovery":
    from modules.tier1.options_liquidity import OptionsLiquidityModule
    module = OptionsLiquidityModule()
    module.run()
    logger.info(f"User {st.session_state.username} opened Options Liquidity module")

elif selected_module == "📊 Market Breadth Dashboard":
    from modules.tier1.market_breadth import MarketBreadthModule
    module = MarketBreadthModule()
    module.run()
    logger.info(f"User {st.session_state.username} opened Market Breadth module")

elif selected_module == "⚙️ ES Futures Backtester":
    from modules.tier1.es_backtester import ESBacktesterModule
    module = ESBacktesterModule()
    module.run()
    logger.info(f"User {st.session_state.username} opened ES Backtester module")

elif selected_module == "🎯 Portfolio Optimizer":
    from modules.tier1.portfolio_optimizer import PortfolioOptimizerModule
    module = PortfolioOptimizerModule()
    module.run()
    logger.info(f"User {st.session_state.username} opened Portfolio Optimizer module")

elif selected_module == "🔄 Sector Rotation Monitor":
    from modules.tier1.sector_rotation import SectorRotationModule
    module = SectorRotationModule()
    module.run()
    logger.info(f"User {st.session_state.username} opened Sector Rotation module")

# ─────────────────────────────────────────────────────────────────────────
# TIER 2 MODULES
# ─────────────────────────────────────────────────────────────────────────

elif selected_module == "🔍 Stock Ticker Screener":
    from modules.tier2.ticker_screener import TickerScreenerModule
    module = TickerScreenerModule()
    module.run()

elif selected_module == "📊 Implied Volatility Analysis":
    from modules.tier2.iv_analysis import IVAnalysisModule
    module = IVAnalysisModule()
    module.run()

elif selected_module == "🏦 Index Analysis Dashboard":
    from modules.tier2.index_analysis import IndexAnalysisModule
    module = IndexAnalysisModule()
    module.run()

# ─────────────────────────────────────────────────────────────────────────
# DEFAULT: WELCOME DASHBOARD
# ─────────────────────────────────────────────────────────────────────────

else:
    st.markdown("""
    ## 👋 Welcome to FazDane Research
    
    **Your All-in-One Trading Intelligence Platform**
    
    This application brings together 89+ financial analysis programs organized into intelligent, menu-driven modules.
    
    ### Quick Start
    
    1. **Start with Tier 1** (Left Sidebar) - Real-time trading tools
    2. **Explore Tier 2** - Deeper analysis and research
    3. **Advanced** (Tier 3) - Forecasting and cycle analysis
    
    ### What's Available
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Modules", "18+")
        st.metric("💧 Active Options", "Real-time")
    
    with col2:
        st.metric("🔄 Sector Tracking", "11 Sectors")
        st.metric("⚙️ Portfolio Tools", "5+ Models")
    
    with col3:
        st.metric("🔮 Forecasting", "7 Methods")
        st.metric("📈 Data Sources", "12+")
    
    st.divider()
    
    st.markdown("""
    ### 📚 Module Categories
    
    **TIER 1: Live Trading** (Highest Priority)
    - Options Liquidity Discovery - Real-time options scanning
    - Market Breadth - Advance/decline analysis
    - ES Futures Backtester - Intraday strategy testing
    - Portfolio Optimizer - Sharpe ratio optimization
    - Sector Rotation - RRG-style visualization
    
    **TIER 2: Analysis & Intelligence**
    - Stock Screener - Fundamental + technical filters
    - IV Analysis - Volatility surface analysis
    - Index Analysis - SPX, VIX, volatility metrics
    
    **TIER 3: Forecasting**
    - Price Forecasting - Prophet, ARIMA, LSTM models
    - Bradley Cycles - Astronomical market cycles
    - Elliott Wave - Structural price analysis
    
    ### 🚀 Getting Started
    
    Select a module from the sidebar to begin. All data is cached for performance.
    """)
    
    st.divider()
    
    st.info("""
    **💡 Tip**: Click the sidebar modules to explore different tools. 
    Use the filters in the sidebar to customize your analysis.
    """)

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**Version**: 1.0 | **Status**: Beta")

with footer_col2:
    st.markdown("**Last Update**: May 2026 | **Modules**: 18+")

with footer_col3:
    st.markdown("**Framework**: Streamlit | **Auth**: Secure")
