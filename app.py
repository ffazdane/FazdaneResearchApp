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

# ══════════════════════════════════════════════════════════════════════
# GLOBAL CSS — FazDane Analytics Brand
# Colors: navy #0d1b2e, blue #1a3a8f, green #3ab54a
# ══════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Courier+Prime:wght@400;700&display=swap" rel="stylesheet">
    <style>
        /* Base */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* Background */
        .stApp { background: #0d1b2e; color: #e2e8f0; }

        /* ALL text */
        p, li, span, label, small,
        .stMarkdown p, .stMarkdown li, .stMarkdown span { color: #e2e8f0; }

        /* Widget labels */
        .stTextInput label, .stTextArea label, .stSelectbox label,
        .stMultiSelect label, .stSlider label, .stRadio label,
        .stCheckbox label, .stNumberInput label, .stDateInput label,
        [data-testid="stWidgetLabel"] p {
            color: #cbd5e1 !important; font-size: 13px !important; font-weight: 500 !important;
        }

        /* Radio / checkbox text */
        .stRadio div[role="radiogroup"] label p,
        .stRadio div[role="radiogroup"] label span,
        .stCheckbox label p { color: #e2e8f0 !important; }

        /* Selectbox */
        .stSelectbox [data-baseweb="select"] span,
        .stSelectbox [data-baseweb="select"] div,
        [data-baseweb="select"] span {
            color: #e2e8f0 !important; background-color: rgba(21,40,71,0.9) !important;
        }

        /* Multiselect tags */
        .stMultiSelect [data-baseweb="tag"] {
            background: rgba(26,58,143,0.75) !important;
            border: 1px solid #3ab54a !important;
            border-radius: 6px !important;
            padding: 4px 8px !important;
        }
        .stMultiSelect [data-baseweb="tag"] span {
            color: #ffffff !important;
            font-size: 13px !important;
            font-weight: 600 !important;
        }
        .stMultiSelect [data-baseweb="tag"] [data-testid="stMultiSelectClearButton"],
        .stMultiSelect [data-baseweb="tag"] button,
        .stMultiSelect [data-baseweb="tag"] svg { color: #94a3b8 !important; fill: #94a3b8 !important; }
        .stMultiSelect [data-baseweb="select"] span,
        .stMultiSelect [data-baseweb="select"] div { color: #e2e8f0 !important; }
        .stMultiSelect [data-baseweb="select"] input { color: #e2e8f0 !important; }

        /* Dropdown option lists */
        [data-baseweb="popover"] li, [data-baseweb="menu"] li,
        [role="option"], [role="listbox"] li {
            color: #e2e8f0 !important; background: #152847 !important;
        }
        [data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover,
        [role="option"]:hover { background: rgba(58,181,74,0.15) !important; }

        /* Text input & textarea */
        .stTextInput > div > div > input, .stTextArea textarea, .stNumberInput input {
            background: rgba(21,40,71,0.9) !important; border-color: #1e3a5f !important;
            color: #e2e8f0 !important; border-radius: 8px !important;
        }
        .stTextInput > div > div > input::placeholder,
        .stTextArea textarea::placeholder,
        .stNumberInput input::placeholder { color: #475569 !important; }

        /* Slider */
        .stSlider > div > div > div > div { background: #3ab54a !important; }
        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"] { color: #94a3b8 !important; }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1b2e 0%, #152847 60%, #0d1b2e 100%);
            border-right: 1px solid #1e3a5f;
        }
        [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
        [data-testid="stSidebar"] .stRadio label { padding: 6px 8px; border-radius: 6px; transition: background 0.2s; }
        [data-testid="stSidebar"] .stRadio label:hover { background: rgba(58,181,74,0.12) !important; }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #94a3b8 !important; }

        /* Main content */
        .stMainBlockContainer { background: #0d1b2e !important; padding-top: 1.5rem; }

        /* Headings */
        h1, h2, h3 { font-family: 'Courier Prime', monospace !important; color: #3ab54a !important; }
        h4, h5, h6 { color: #93c5fd !important; }

        /* Metrics */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #152847 0%, #0d1b2e 100%);
            border: 1px solid #1e3a5f; border-radius: 10px; padding: 16px 20px;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        [data-testid="metric-container"]:hover { border-color: #3ab54a; box-shadow: 0 0 16px rgba(58,181,74,0.12); }
        [data-testid="metric-container"] [data-testid="stMetricLabel"] p {
            color: #94a3b8 !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 1px;
        }
        [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 24px !important; font-weight: 700; }
        [data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 13px !important; }

        /* Buttons */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1a3a8f 0%, #3ab54a 100%);
            color: #ffffff !important; font-weight: 600; border: none;
            border-radius: 8px; padding: 10px 20px; transition: all 0.25s ease;
            box-shadow: 0 2px 8px rgba(58,181,74,0.2);
        }
        .stButton > button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(58,181,74,0.35); }
        .stButton > button[kind="secondary"] {
            background: rgba(26,58,143,0.25); color: #93c5fd !important;
            border: 1px solid #1e3a5f; border-radius: 8px; transition: all 0.2s;
        }
        .stButton > button[kind="secondary"]:hover { background: rgba(26,58,143,0.45); border-color: #3ab54a; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(21,40,71,0.6); border-radius: 10px; padding: 4px; gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px !important; color: #94a3b8 !important;
            padding: 8px 20px !important; font-weight: 500; transition: all 0.2s;
        }
        .stTabs [data-baseweb="tab"] p { color: #94a3b8 !important; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: rgba(58,181,74,0.15) !important; color: #3ab54a !important; border-bottom: 2px solid #3ab54a !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] p { color: #3ab54a !important; }

        /* Dataframe */
        [data-testid="stDataFrameContainer"] { border: 1px solid #1e3a5f; border-radius: 10px; }
        [data-testid="stDataFrameContainer"] th { background: #152847 !important; color: #94a3b8 !important; }
        [data-testid="stDataFrameContainer"] td { color: #e2e8f0 !important; }

        /* Misc */
        .stMarkdown em, .stCaption { color: #94a3b8 !important; }
        hr { border-color: #1e3a5f !important; }
        .stAlert { border-radius: 10px !important; }
        .stAlert p, .stAlert span { color: #e2e8f0 !important; }
        [data-testid="stExpander"] summary p { color: #93c5fd !important; }
        .stSpinner p { color: #94a3b8 !important; }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0d1b2e; }
        ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #3ab54a; }
    
        /* ── Sidebar multiselect & selectbox — remove white box ─────── */
        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"],
        [data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
            background-color: rgba(13, 27, 46, 0.0) !important;
            border-color: #1e3a5f !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
            background-color: #152847 !important;
            border-color: #1e3a5f !important;
        }

        /* ── All multiselect containers — transparent bg in sidebar ─── */
        [data-testid="stSidebar"] div[data-baseweb="select"] {
            background: transparent !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: rgba(21, 40, 71, 0.6) !important;
            border: 1px solid #1e3a5f !important;
            border-radius: 8px !important;
        }

        /* ── Selectbox dropdown in sidebar ──────────────────────────── */
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
            background: #152847 !important;
            border: 1px solid #1e3a5f !important;
            border-radius: 8px !important;
            color: #e2e8f0 !important;
        }

    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
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
    st.markdown(
        "<div style='color:#3ab54a;font-size:12px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;'>🔥 Tier 1 — Live Trading</div>",
        unsafe_allow_html=True,
    )

    tier1_options = [
        "🏠 Home Dashboard",
        "💧 Options Liquidity Discovery",
        "📊 Market Breadth Dashboard",
        "📅 Calendar Strategy Matrix",
        "🦅 Iron Condor Analyzer",
        "⚙️ ES Pivot Analysis",
        "🔄 Sector Rotation Monitor",
    ]
    tier1_sel = st.radio(
        "tier1", tier1_options, key="tier1_nav", label_visibility="collapsed"
    )

    st.divider()

    # ── Tier 2: Analysis ──────────────────────────────────────────────
    with st.expander("📈 Tier 2 — Analysis & Intelligence", expanded=False):
        tier2_options = [
            "⚪ Select Tier 2 Module...",
            "🔥 Multi-Timeframe Money Flow",
            "🗓️ Market Structure Heatmap",
            "🔍 Stock Ticker Screener",
            "📉 Implied Volatility Analysis",
            "🏦 Index Analysis Dashboard",
            "📅 Calendar Heatmaps",
            "📺 Earnings Calendar",
        ]
        tier2_sel = st.radio(
            "tier2", tier2_options, key="tier2_nav", label_visibility="collapsed"
        )

    # ── Tier 3: Forecasting ───────────────────────────────────────────
    with st.expander("🔮 Tier 3 — Forecasting & Cycles", expanded=False):
        tier3_options = [
            "⚪ Select Tier 3 Module...",
            "🔮 SPX Price Forecasting",
            "🌙 Bradley Siderograph",
            "🌊 Elliott Wave Analysis",
            "⏰ Market Cycle Timing",
        ]
        tier3_sel = st.radio(
            "tier3", tier3_options, key="tier3_nav", label_visibility="collapsed"
        )

    st.divider()

    # ── Module Status ─────────────────────────────────────────────────
    st.markdown(
        "<div style='color:#64748b;font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>📡 Module Status</div>",
        unsafe_allow_html=True,
    )
    module_statuses = [
        ("💧 Options Liquidity", "🟢 Live"),
        ("📊 Market Breadth", "🟢 Live"),
        ("🔄 Sector Rot.", "🟢 Live"),
        ("📅 Calendar Strat.", "🟢 Live"),
        ("🦅 Iron Condor", "🟢 Live"),
        ("⚙️ ES Pivot Analys.", "🟢 Live"),
        ("🔥 Money Flow", "🟢 Live"),
        ("🗓️ Market Structure", "🟢 Live"),
    ]
    for name, status in module_statuses:
        st.markdown(
            f"<div style='font-size:12px;color:#64748b;padding:2px 0;'>{name} <span style='float:right;'>{status}</span></div>",
            unsafe_allow_html=True,
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
if tier1_sel and tier1_sel != "🏠 Home Dashboard":
    active_module = tier1_sel
elif st.session_state.get("tier2_nav") and st.session_state.get("tier2_nav") != tier2_options[0]:
    active_module = st.session_state.get("tier2_nav")
elif st.session_state.get("tier3_nav") and st.session_state.get("tier3_nav") != tier3_options[0]:
    active_module = st.session_state.get("tier3_nav")
else:
    active_module = "🏠 Home Dashboard"

# ── Route to module ───────────────────────────────────────────────────

if active_module == "💧 Options Liquidity Discovery":
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

elif active_module in tier2_options:
    st.info(f"{active_module} — Tier 2 module coming in Weeks 3–4")

elif active_module in tier3_options:
    st.info(f"{active_module} — Tier 3 module coming in Weeks 5–6")

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

    # Platform stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Total Modules", "18+", help="Across Tier 1–4")
    with col2:
        st.metric("🟢 Live Now", "8", help="Options Liquidity, Market Breadth, Sector Rotation, Calendar Matrix, Iron Condor, ES Pivot, Money Flow, Market Structure")
    with col3:
        st.metric("🔄 Tier 1 Progress", "100%", help="6 of 6 Tier 1 modules complete")
    with col4:
        st.metric("📊 Data Sources", "yfinance", help="More integrations coming")

    st.divider()

    # Module grid
    st.markdown(
        "<h3 style='color:#3ab54a;'>📦 Module Roadmap</h3>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            <div style="
                background:rgba(21,40,71,0.6);
                border:1px solid #1e3a5f;
                border-radius:12px;
                padding:20px;
                margin-bottom:12px;
            ">
                <div style="color:#3ab54a;font-weight:700;font-size:14px;margin-bottom:12px;">🔥 TIER 1 — Live Trading</div>
                <div style="font-size:13px;line-height:1.9;color:#94a3b8;">
                    🟢 <span style="color:#e2e8f0;">💧 Options Liquidity Discovery</span><br>
                    🟢 <span style="color:#e2e8f0;">📊 Market Breadth Dashboard</span><br>
                    🟢 <span style="color:#e2e8f0;">🔄 Sector Rotation Monitor</span><br>
                    🟢 <span style="color:#e2e8f0;">📅 Calendar Strategy Matrix</span><br>
                    🟢 <span style="color:#e2e8f0;">🦅 Iron Condor Analyzer</span><br>
                    🟢 <span style="color:#e2e8f0;">⚙️ ES Pivot Analysis</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="
                background:rgba(21,40,71,0.6);
                border:1px solid #1e3a5f;
                border-radius:12px;
                padding:20px;
            ">
                <div style="color:#93c5fd;font-weight:700;font-size:14px;margin-bottom:12px;">🔮 TIER 3 — Forecasting</div>
                <div style="font-size:13px;line-height:1.9;color:#475569;">
                    ⚪ 🔮 SPX Price Forecasting<br>
                    ⚪ 🌙 Bradley Siderograph<br>
                    ⚪ 🌊 Elliott Wave Analysis<br>
                    ⚪ ⏰ Market Cycle Timing
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div style="
                background:rgba(21,40,71,0.6);
                border:1px solid #1e3a5f;
                border-radius:12px;
                padding:20px;
                margin-bottom:12px;
            ">
                <div style="color:#93c5fd;font-weight:700;font-size:14px;margin-bottom:12px;">📈 TIER 2 — Analysis</div>
                <div style="font-size:13px;line-height:1.9;color:#475569;">
                    🟢 <span style="color:#e2e8f0;">🔥 Multi-Timeframe Money Flow</span><br>
                    🟢 <span style="color:#e2e8f0;">🗓️ Market Structure Heatmap</span><br>
                    ⚪ 🔍 Stock Ticker Screener<br>
                    ⚪ 📉 Implied Volatility Analysis<br>
                    ⚪ 🏦 Index Analysis Dashboard<br>
                    ⚪ 📅 Calendar Heatmaps<br>
                    ⚪ 📺 Earnings Calendar
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="
                background:rgba(21,40,71,0.6);
                border:1px solid #1e3a5f;
                border-radius:12px;
                padding:20px;
            ">
                <div style="color:#475569;font-weight:700;font-size:14px;margin-bottom:12px;">🧪 TIER 4 — Specialized</div>
                <div style="font-size:13px;line-height:1.9;color:#334155;">
                    ⚪ 🌍 Macro Intelligence<br>
                    ⚪ 🌕 Moon Cycle Analysis<br>
                    ⚪ 🦅 Iron Condor Analyzer<br>
                    ⚪ 📐 Volume Profile Analysis
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#334155;font-size:12px;'>FazDane Analytics v1.0 · © 2026 All Rights Reserved · Built on Streamlit</p>",
        unsafe_allow_html=True,
    )
