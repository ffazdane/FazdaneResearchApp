"""
FazDane Analytics — Main Application Entry Point
Authentication-gated Streamlit dashboard.
"""

import streamlit as st
import logging
import os
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FazDane Analytics | Trading Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "FazDane Analytics v1.0 — Research & Trading Intelligence Platform",
    },
)

# ══════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/fazdane.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("FazDaneApp")


def launch_module(module_name: str, tier: int) -> None:
    st.session_state.pop("active_menu_tier", None)
    st.session_state["pending_nav"] = {"module": module_name, "tier": tier}
    st.rerun()


def render_home_module_button(label: str, module_name: str, tier: int, key: str) -> None:
    if st.button(label, key=key, use_container_width=True):
        launch_module(module_name, tier)

# ══════════════════════════════════════════════════════════════════════
# =====================================================================
# STREAMLIT DEFAULT DARK THEME
# Keep app-wide styling minimal so Streamlit's native dark UI shows through.
# =====================================================================

st.markdown(
    """
    <style>
        .stApp { background: #0e1117; }
        [data-testid="stSidebar"] { background: #0e1117; }
    </style>
    """,
    unsafe_allow_html=True,
)
# =====================================================================
# AUTHENTICATION CHECK
# ══════════════════════════════════════════════════════════════════════

from pages.auth import FazDaneAuthenticator, logout, get_current_user

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    authenticator = FazDaneAuthenticator()
    authenticator.render_login_screen()
    st.stop()

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — Navigation
# ══════════════════════════════════════════════════════════════════════

user = get_current_user()

pending_nav = st.session_state.pop("pending_nav", None)
if pending_nav:
    action = pending_nav.get("action")
    module_name = pending_nav.get("module")
    tier = pending_nav.get("tier")
    if action == "home":
        st.session_state["tier1_module_nav"] = "⚪ Select Live Trading Module..."
        st.session_state["tier2_nav"] = "⚪ Select Analysis Module..."
        st.session_state["tier3_nav"] = "⚪ Select Forecasting Module..."
        st.session_state.pop("active_menu_tier", None)
    elif action == "clear_tier" and tier == 1:
        st.session_state["tier1_module_nav"] = "⚪ Select Live Trading Module..."
        st.session_state["active_menu_tier"] = 1
    elif action == "clear_tier" and tier == 2:
        st.session_state["tier2_nav"] = "⚪ Select Analysis Module..."
        st.session_state["active_menu_tier"] = 2
    elif action == "clear_tier" and tier == 3:
        st.session_state["tier3_nav"] = "⚪ Select Forecasting Module..."
        st.session_state["active_menu_tier"] = 3
    elif tier == 1:
        st.session_state.pop("active_menu_tier", None)
        st.session_state["tier1_module_nav"] = module_name
        st.session_state["tier2_nav"] = "⚪ Select Analysis Module..."
        st.session_state["tier3_nav"] = "⚪ Select Forecasting Module..."
    elif tier == 2:
        st.session_state.pop("active_menu_tier", None)
        st.session_state["tier1_module_nav"] = "⚪ Select Live Trading Module..."
        st.session_state["tier2_nav"] = module_name
        st.session_state["tier3_nav"] = "⚪ Select Forecasting Module..."
    elif tier == 3:
        st.session_state.pop("active_menu_tier", None)
        st.session_state["tier1_module_nav"] = "⚪ Select Live Trading Module..."
        st.session_state["tier2_nav"] = "⚪ Select Analysis Module..."
        st.session_state["tier3_nav"] = module_name

with st.sidebar:
    # Logo
    try:
        st.image("assets/logo.png", use_container_width=True)
    except Exception:
        st.markdown(
            "<h2 style='color:#3ab54a;text-align:center;'>FazDane Analytics</h2>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # User badge
    role_color = "#3ab54a" if user["role"] == "admin" else "#93c5fd"
    st.markdown(
        f"""
        <div style="
            background:rgba(26,58,143,0.2);
            border:1px solid #1e3a5f;
            border-radius:8px;
            padding:10px 14px;
            margin-bottom:12px;
        ">
            <div style="color:#e2e8f0;font-weight:600;font-size:14px;">👤 {user['display_name']}</div>
            <div style="color:{role_color};font-size:11px;text-transform:uppercase;letter-spacing:1px;">{user['role']}</div>
            <div style="color:#475569;font-size:11px;margin-top:2px;">Since {user['login_time']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Tier 1: Live Trading ──────────────────────────────────────────
    if st.button("🏠 Home Dashboard", use_container_width=True, key="home_dashboard_nav"):
        st.session_state["pending_nav"] = {"action": "home"}
        st.rerun()

    with st.expander("🔥 Live Trading", expanded=False):
        st.markdown(
        "<div style='color:#3ab54a;font-size:12px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;'>🔥 Live Trading</div>",
        unsafe_allow_html=True,
        )

        tier1_options = [
        "⚪ Select Live Trading Module...",
        "💧 Options Liquidity Discovery",
        "📊 Market Breadth Dashboard",
        "📅 Calendar Strategy Matrix",
        "🦅 Iron Condor Analyzer",
        "⚙️ ES Pivot Analysis",
        "🔄 Sector Rotation Monitor",
        ]
        tier1_sel = st.radio(
            "tier1", tier1_options, key="tier1_module_nav", label_visibility="collapsed"
        )

    st.divider()

    # ── Tier 2: Analysis ──────────────────────────────────────────────
    with st.expander("📈 Analysis & Intelligence", expanded=False):
        tier2_options = [
            "⚪ Select Analysis Module...",
            "🔥 Multi-Timeframe Money Flow",
            "🗓️ Market Structure Heatmap",
            "🧮 Correlation Matrix",
            "📺 Earnings Calendar",
            "📄 Equity Income Statement",
            "📅 Equity / Index Seasonality",
            "📰 Stock Sentiment Analysis",
        ]
        tier2_sel = st.radio(
            "tier2", tier2_options, key="tier2_nav", label_visibility="collapsed"
        )

    # ── Tier 3: Forecasting ───────────────────────────────────────────
    with st.expander("🔮 Forecasting & Cycles", expanded=False):
        tier3_options = [
            "⚪ Select Forecasting Module...",
            "🌙 Bradley Siderograph",
            "🌊 Elliott Wave Analysis",
        ]
        tier3_sel = st.radio(
            "tier3", tier3_options, key="tier3_nav", label_visibility="collapsed"
        )

    st.divider()

    # ── Logout ────────────────────────────────────────────────────────
    if st.button("🔐 Logout", use_container_width=True, type="secondary"):
        logger.info(f"User {user['username']} logged out")
        logout()

# ══════════════════════════════════════════════════════════════════════
# MODULE DISPATCHER
# ══════════════════════════════════════════════════════════════════════

# Determine which module is active — tier1 takes precedence
if tier1_sel and tier1_sel != "⚪ Select Live Trading Module...":
    active_module = tier1_sel
elif st.session_state.get("tier2_nav") and st.session_state.get("tier2_nav") != tier2_options[0]:
    active_module = st.session_state.get("tier2_nav")
elif st.session_state.get("tier3_nav") and st.session_state.get("tier3_nav") != tier3_options[0]:
    active_module = st.session_state.get("tier3_nav")
elif st.session_state.get("active_menu_tier"):
    active_module = f"__MENU_TIER_{st.session_state.get('active_menu_tier')}__"
else:
    active_module = "🏠 Home Dashboard"

# ── Route to module ───────────────────────────────────────────────────

if active_module.startswith("__MENU_TIER_"):
    menu_tier = int(st.session_state.get("active_menu_tier", 1))
    menu_config = {
        1: ("Live Trading", "Tier 1 execution tools and real-time market dashboards", tier1_options[1:]),
        2: ("Analysis & Intelligence", "Research modules for cross-market, fundamental, seasonal, and sentiment work", tier2_options[1:]),
        3: ("Forecasting & Cycles", "Cycle and forecasting modules for turning-window analysis", tier3_options[1:]),
    }

    try:
        st.image("assets/logo.png", width=240)
    except Exception:
        st.title("FazDane Analytics")

    title, subtitle, options = menu_config.get(menu_tier, ("Module Menu", "Choose a module to open.", []))
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(26,58,143,0.30) 0%, rgba(58,181,74,0.10) 100%);
            border: 1px solid #1e3a5f;
            border-left: 4px solid #3ab54a;
            border-radius: 12px;
            padding: 18px 22px;
            margin: 8px 0 18px 0;
        ">
            <div style="color:#3ab54a;font-size:20px;font-weight:700;margin-bottom:4px;">{title}</div>
            <div style="color:#94a3b8;font-size:14px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    switch_cols = st.columns(3)
    for tier_id, (switch_label, _, _) in menu_config.items():
        with switch_cols[tier_id - 1]:
            button_type = "primary" if menu_tier == tier_id else "secondary"
            if st.button(switch_label, key=f"module_menu_switch_{tier_id}", use_container_width=True, type=button_type):
                st.session_state["active_menu_tier"] = tier_id
                st.rerun()

    st.divider()
    st.markdown("### Select Module")
    cols = st.columns(2)
    for index, option in enumerate(options):
        with cols[index % 2]:
            if st.button(option, key=f"back_menu_{menu_tier}_{index}", use_container_width=True):
                launch_module(option, menu_tier)

elif active_module == "💧 Options Liquidity Discovery":
    from modules.tier1.options_liquidity import OptionsLiquidityModule
    module = OptionsLiquidityModule()
    module.run()
    logger.info(f"User {user['username']} → Options Liquidity Discovery")

elif active_module == "📊 Market Breadth Dashboard":
    from modules.tier1.market_breadth import MarketBreadthModule
    module = MarketBreadthModule()
    module.run()
    logger.info(f"User {user['username']} → Market Breadth Dashboard")

elif active_module == "⚙️ ES Pivot Analysis":
    from modules.tier1.es_pivot_analysis import ESPivotAnalysisModule
    module = ESPivotAnalysisModule()
    module.run()
    logger.info(f"User {user['username']} → ES Pivot Analysis")

elif active_module == "🔄 Sector Rotation Monitor":
    from modules.tier1.sector_rotation import SectorRotationModule
    module = SectorRotationModule()
    module.run()
    logger.info(f"User {user['username']} → Sector Rotation Monitor")

elif active_module == "📅 Calendar Strategy Matrix":
    from modules.tier1.calendar_rotation import CalendarRotationModule
    module = CalendarRotationModule()
    module.run()
    logger.info(f"User {user['username']} → Calendar Strategy Matrix")

elif active_module == "🦅 Iron Condor Analyzer":
    from modules.tier1.iron_condor import IronCondorModule
    module = IronCondorModule()
    module.run()
    logger.info(f"User {user['username']} → Iron Condor Analyzer")

elif active_module == "🔥 Multi-Timeframe Money Flow":
    from modules.tier2.money_flow import MoneyFlowModule
    module = MoneyFlowModule()
    module.run()
    logger.info(f"User {user['username']} → Money Flow")

elif active_module == "🗓️ Market Structure Heatmap":
    from modules.tier2.market_structure import MarketStructureModule
    module = MarketStructureModule()
    module.run()
    logger.info(f"User {user['username']} → Market Structure")

elif active_module == "🧮 Correlation Matrix":
    from modules.tier2.correlation_matrix import CorrelationMatrixModule
    module = CorrelationMatrixModule()
    module.run()
    logger.info(f"User {user['username']} → Correlation Matrix")

elif active_module == "📺 Earnings Calendar":
    from modules.tier2.earnings_calendar import EarningsCalendarModule
    module = EarningsCalendarModule()
    module.run()
    logger.info(f"User {user['username']} → Earnings Calendar")

elif active_module == "📄 Equity Income Statement":
    from modules.tier2.equity_income_statement import EquityIncomeStatementModule
    module = EquityIncomeStatementModule()
    module.run()
    logger.info(f"User {user['username']} → Equity Income Statement")

elif active_module == "📅 Equity / Index Seasonality":
    from modules.tier2.seasonality_analysis import SeasonalityAnalysisModule
    module = SeasonalityAnalysisModule()
    module.run()
    logger.info(f"User {user['username']} → Equity / Index Seasonality")

elif active_module == "📰 Stock Sentiment Analysis":
    from modules.tier2.stock_sentiment import StockSentimentModule
    module = StockSentimentModule()
    module.run()
    logger.info(f"User {user['username']} → Stock Sentiment Analysis")

elif "Bradley Siderograph" in active_module:
    from modules.tier3.bradley_siderograph import BradleySiderographModule
    module = BradleySiderographModule()
    module.run()
    logger.info(f"User {user['username']} → Bradley Siderograph")

elif "Elliott Wave Analysis" in active_module:
    from modules.tier3.elliott_wave_analysis import ElliottWaveAnalysisModule
    module = ElliottWaveAnalysisModule()
    module.run()
    logger.info(f"User {user['username']} → Elliott Wave Analysis")

elif active_module in tier2_options:
    st.info(f"{active_module} — Analysis module coming in Weeks 3–4")

elif active_module in tier3_options:
    st.info(f"{active_module} — Forecasting module coming in Weeks 5–6")

else:
    # ══════════════════════════════════════════════════════════════════
    # HOME DASHBOARD
    # ══════════════════════════════════════════════════════════════════
    try:
        st.image("assets/logo.png", width=280)
    except Exception:
        st.title("FazDane Analytics")

    st.markdown(
        "<p style='color:#64748b;font-size:16px;margin-top:-8px;'>Research & Trading Intelligence Platform</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Welcome card
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(26,58,143,0.3) 0%, rgba(58,181,74,0.1) 100%);
            border: 1px solid #1e3a5f;
            border-left: 4px solid #3ab54a;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
        ">
            <div style="color:#3ab54a;font-size:18px;font-weight:700;margin-bottom:6px;">
                Welcome back, {user['display_name']}! 👋
            </div>
            <div style="color:#94a3b8;font-size:14px;">
                Your personal finance research platform is ready. Select a module from the sidebar to begin.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    macro_module_tabs = [
        {
            "label": "Live Trading",
            "items": [
                {"label": "💧 Options Liquidity Discovery", "module": "💧 Options Liquidity Discovery", "tier": 1, "key": "macro_options_liquidity"},
                {"label": "📊 Market Breadth Dashboard", "module": "📊 Market Breadth Dashboard", "tier": 1, "key": "macro_market_breadth"},
                {"label": "🔄 Sector Rotation Monitor", "module": "🔄 Sector Rotation Monitor", "tier": 1, "key": "macro_sector_rotation"},
                {"label": "📅 Calendar Strategy Matrix", "module": "📅 Calendar Strategy Matrix", "tier": 1, "key": "macro_calendar_strategy"},
                {"label": "🦅 Iron Condor Analyzer", "module": "🦅 Iron Condor Analyzer", "tier": 1, "key": "macro_iron_condor"},
                {"label": "⚙️ ES Pivot Analysis", "module": "⚙️ ES Pivot Analysis", "tier": 1, "key": "macro_es_pivot"},
            ],
        },
        {
            "label": "Analysis & Intelligence",
            "items": [
                {"label": "🔥 Multi-Timeframe Money Flow", "module": "🔥 Multi-Timeframe Money Flow", "tier": 2, "key": "macro_money_flow"},
                {"label": "🗓️ Market Structure Heatmap", "module": "🗓️ Market Structure Heatmap", "tier": 2, "key": "macro_market_structure"},
                {"label": "🧮 Correlation Matrix", "module": "🧮 Correlation Matrix", "tier": 2, "key": "macro_correlation"},
                {"label": "📺 Earnings Calendar", "module": "📺 Earnings Calendar", "tier": 2, "key": "macro_earnings"},
                {"label": "📄 Equity Income Statement", "module": "📄 Equity Income Statement", "tier": 2, "key": "macro_income_statement"},
                {"label": "📅 Equity / Index Seasonality", "module": "📅 Equity / Index Seasonality", "tier": 2, "key": "macro_seasonality"},
                {"label": "📰 Stock Sentiment Analysis", "module": "📰 Stock Sentiment Analysis", "tier": 2, "key": "macro_sentiment"},
            ],
        },
        {
            "label": "Forecasting",
            "items": [
                {"label": "🌙 Bradley Siderograph", "module": "🌙 Bradley Siderograph", "tier": 3, "key": "macro_bradley"},
                {"label": "🌊 Elliott Wave Analysis", "module": "🌊 Elliott Wave Analysis", "tier": 3, "key": "macro_elliott"},
            ],
        },
    ]

    from modules.tier2.macro_intelligence import render_macro_dashboard
    render_macro_dashboard(show_download=True, module_tabs=macro_module_tabs, launch_callback=launch_module)

    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#334155;font-size:12px;'>FazDane Analytics v1.0 · © 2026 All Rights Reserved · Built on Streamlit</p>",
        unsafe_allow_html=True,
    )
