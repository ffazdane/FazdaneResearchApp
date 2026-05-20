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


def refresh_live_data() -> None:
    """Clear cached data and common module result state, then reload."""
    st.cache_data.clear()
    for key in [
        "ol_results",
        "ol_iv_ranks",
        "ol_last_snapshot",
        "ol_snapshot_error",
        "ol_active_data_source",
    ]:
        st.session_state.pop(key, None)
    st.rerun()


def back_to_current_menu() -> None:
    """Return from the active module to the matching module menu."""
    tier1_default = "⚪ Select Live Trading Module..."
    tier2_default = "⚪ Select Analysis Module..."
    tier3_default = "⚪ Select Forecasting Module..."

    if st.session_state.get("tier1_module_nav") and st.session_state.get("tier1_module_nav") != tier1_default:
        st.session_state["pending_nav"] = {"action": "clear_tier", "tier": 1}
    elif st.session_state.get("tier2_nav") and st.session_state.get("tier2_nav") != tier2_default:
        st.session_state["pending_nav"] = {"action": "clear_tier", "tier": 2}
    elif st.session_state.get("tier3_nav") and st.session_state.get("tier3_nav") != tier3_default:
        st.session_state["pending_nav"] = {"action": "clear_tier", "tier": 3}
    else:
        st.session_state["active_menu_tier"] = st.session_state.get("active_menu_tier", 1)
    st.rerun()

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

        /* Hide Streamlit header/banner chrome */
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu,
        footer {
            visibility: hidden !important;
            height: 0 !important;
            min-height: 0 !important;
            display: none !important;
        }

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
        [data-testid="stSidebar"] .stButton > button {
            min-height: 36px;
            padding: 7px 10px !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            line-height: 1.25 !important;
        }
        [data-testid="stSidebar"] [data-testid="column"] .stButton > button {
            min-height: 34px;
            padding: 6px 4px !important;
            font-size: 12px !important;
            line-height: 1.15 !important;
            white-space: nowrap;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            min-height: 34px;
            padding: 7px 8px !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
            font-size: 13px !important;
            font-weight: 600 !important;
            line-height: 1.25 !important;
        }
        [data-testid="stSidebar"] .stRadio label {
            padding: 5px 7px;
            border-radius: 6px;
            transition: background 0.2s;
            min-height: 30px;
        }
        [data-testid="stSidebar"] .stRadio label p,
        [data-testid="stSidebar"] .stRadio label span {
            font-size: 12.5px !important;
            font-weight: 500 !important;
            line-height: 1.25 !important;
        }
        [data-testid="stSidebar"] .stRadio label:hover { background: rgba(58,181,74,0.12) !important; }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #94a3b8 !important; }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: rgba(13,27,46,0.24);
            border: 1px solid rgba(30,58,95,0.72);
            border-radius: 10px;
            margin-bottom: 8px;
            overflow: hidden;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
            background: rgba(26,58,143,0.18) !important;
        }

        /* Main content */
        .stMainBlockContainer { background: #0d1b2e !important; padding-top: 0.75rem; }

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
        [data-testid="stDataFrameContainer"] {
            background: #0b1628 !important;
            border: 1px solid #1e3a5f;
            border-radius: 10px;
            overflow: hidden;
        }
        [data-testid="stDataFrameContainer"] *,
        [data-testid="stDataFrame"] *,
        .stDataFrame *,
        .stTable * {
            color: #e2e8f0 !important;
        }
        [data-testid="stDataFrameContainer"] > div,
        [data-testid="stDataFrameContainer"] section,
        [data-testid="stDataFrame"] > div,
        .stDataFrame > div {
            background-color: #0b1628 !important;
        }
        [data-testid="stDataFrameContainer"] canvas {
            background-color: transparent !important;
        }
        [data-testid="stDataFrameContainer"] th,
        [data-testid="stDataFrameContainer"] thead tr,
        [data-testid="stDataFrame"] th,
        .stDataFrame th,
        .stTable th {
            background: #152847 !important;
            color: #94a3b8 !important;
        }
        [data-testid="stDataFrameContainer"] td,
        [data-testid="stDataFrame"] td,
        .stDataFrame td,
        .stTable td {
            background: #0b1628 !important;
            color: #e2e8f0 !important;
            border-color: #1e3a5f !important;
        }
        [data-testid="stDataFrameContainer"] tbody tr:nth-child(even) td,
        [data-testid="stDataFrame"] tbody tr:nth-child(even) td,
        .stDataFrame tbody tr:nth-child(even) td,
        .stTable tbody tr:nth-child(even) td {
            background: #0d1b2e !important;
        }

        /* Pandas Styler / markdown tables */
        table {
            background: #0b1628 !important;
            color: #e2e8f0 !important;
            border-color: #1e3a5f !important;
        }
        table th {
            background: #152847 !important;
            color: #94a3b8 !important;
            border-color: #1e3a5f !important;
        }
        table td {
            background: #0b1628 !important;
            color: #e2e8f0 !important;
            border-color: #1e3a5f !important;
        }

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

    st.markdown(
        """
        <div style="color:#94a3b8;font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;margin:2px 0 8px 0;">
            Workspace
        </div>
        """,
        unsafe_allow_html=True,
    )
    top_nav_col1, top_nav_col2, top_nav_col3 = st.columns(3)
    with top_nav_col1:
        if st.button("Home", use_container_width=True, key="home_dashboard_top_nav"):
            st.session_state["pending_nav"] = {"action": "home"}
            st.rerun()
    with top_nav_col2:
        if st.button("Menu", use_container_width=True, key="back_to_menu_top_nav"):
            back_to_current_menu()
    with top_nav_col3:
        if st.button("Refresh", use_container_width=True, key="refresh_data_nav"):
            refresh_live_data()

    role_color = "#3ab54a" if user["role"] == "admin" else "#93c5fd"
    st.markdown(
        f"""
        <div style="
            background:rgba(26,58,143,0.2);
            border:1px solid #1e3a5f;
            border-radius:8px;
            padding:10px 14px;
            margin:10px 0 12px 0;
        ">
            <div style="color:#e2e8f0;font-weight:600;font-size:14px;">{user['display_name']}</div>
            <div style="color:{role_color};font-size:11px;text-transform:uppercase;letter-spacing:1px;">{user['role']}</div>
            <div style="color:#475569;font-size:11px;margin-top:2px;">Since {user['login_time']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        """
        <div style="color:#3ab54a;font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;margin:2px 0 8px 0;">
            Modules
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("🔥 Live Trading", expanded=True):
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
