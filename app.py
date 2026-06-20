"""
Research & Trading Intelligence Platform  Main Application Entry Point
Authentication-gated Streamlit dashboard.
"""

import streamlit as st
import logging
import os
from datetime import datetime
from utils.version import VERSION

#
# PAGE CONFIG  (must be first Streamlit call)
#

st.set_page_config(
    page_title="Research & Trading Intelligence Platform",
    page_icon="FD",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": f"Research & Trading Intelligence Platform {VERSION}",
    },
)

#
# THEMES CONFIGURATION
#

THEMES = {
    "Classic Navy (Default)": {
        "bg_color": "#0d1b2e",
        "sidebar_bg": "linear-gradient(180deg, #0d1b2e 0%, #152847 60%, #0d1b2e 100%)",
        "sidebar_bg_solid": "#152847",
        "card_bg": "linear-gradient(135deg, #152847 0%, #0d1b2e 100%)",
        "card_bg_solid": "#152847",
        "widget_bg": "rgba(21, 40, 71, 0.9)",
        "text_color": "#e2e8f0",
        "text_muted": "#94a3b8",
        "border_color": "#1e3a5f",
        "accent_color": "#3ab54a",
        "accent_hover": "rgba(58, 181, 74, 0.15)",
        "tab_bg": "rgba(21, 40, 71, 0.6)",
        "table_bg": "#0b1628",
        "table_even_bg": "#0d1b2e",
        "input_border": "#1e3a5f",
        "plotly_template": "plotly_dark",
        "color_scheme": "dark"
    },
    "Obsidian / Onyx": {
        "bg_color": "#0f0f11",
        "sidebar_bg": "linear-gradient(180deg, #0f0f11 0%, #1c1c1f 60%, #0f0f11 100%)",
        "sidebar_bg_solid": "#1c1c1f",
        "card_bg": "linear-gradient(135deg, #1c1c1f 0%, #0f0f11 100%)",
        "card_bg_solid": "#1c1c1f",
        "widget_bg": "rgba(28, 28, 31, 0.9)",
        "text_color": "#f3f4f6",
        "text_muted": "#9ca3af",
        "border_color": "#2d2d34",
        "accent_color": "#8b5cf6",
        "accent_hover": "rgba(139, 92, 246, 0.15)",
        "tab_bg": "rgba(28, 28, 31, 0.6)",
        "table_bg": "#141416",
        "table_even_bg": "#0f0f11",
        "input_border": "#2d2d34",
        "plotly_template": "plotly_dark",
        "color_scheme": "dark"
    },
    "Cyberpunk Gold / Amber": {
        "bg_color": "#120e16",
        "sidebar_bg": "linear-gradient(180deg, #120e16 0%, #1f1a24 60%, #120e16 100%)",
        "sidebar_bg_solid": "#1f1a24",
        "card_bg": "linear-gradient(135deg, #1f1a24 0%, #120e16 100%)",
        "card_bg_solid": "#1f1a24",
        "widget_bg": "rgba(31, 26, 36, 0.9)",
        "text_color": "#f8f6f9",
        "text_muted": "#a098a6",
        "border_color": "#3d2d4c",
        "accent_color": "#ffb800",
        "accent_hover": "rgba(255, 184, 0, 0.15)",
        "tab_bg": "rgba(31, 26, 36, 0.6)",
        "table_bg": "#18141d",
        "table_even_bg": "#120e16",
        "input_border": "#3d2d4c",
        "plotly_template": "plotly_dark",
        "color_scheme": "dark"
    },
    "Emerald Forest": {
        "bg_color": "#0a1410",
        "sidebar_bg": "linear-gradient(180deg, #0a1410 0%, #12241e 60%, #0a1410 100%)",
        "sidebar_bg_solid": "#12241e",
        "card_bg": "linear-gradient(135deg, #12241e 0%, #0a1410 100%)",
        "card_bg_solid": "#12241e",
        "widget_bg": "rgba(18, 36, 30, 0.9)",
        "text_color": "#ecfdf5",
        "text_muted": "#a7f3d0",
        "border_color": "#1b3d32",
        "accent_color": "#10b981",
        "accent_hover": "rgba(16, 185, 129, 0.15)",
        "tab_bg": "rgba(18, 36, 30, 0.6)",
        "table_bg": "#0d1f19",
        "table_even_bg": "#0a1410",
        "input_border": "#1b3d32",
        "plotly_template": "plotly_dark",
        "color_scheme": "dark"
    },
    "Nordic Frost (Light Mode)": {
        "bg_color": "#f3f7fa",
        "sidebar_bg": "linear-gradient(180deg, #e4ecf2 0%, #f3f7fa 100%)",
        "sidebar_bg_solid": "#e4ecf2",
        "card_bg": "linear-gradient(135deg, #ffffff 0%, #f3f7fa 100%)",
        "card_bg_solid": "#ffffff",
        "widget_bg": "rgba(255, 255, 255, 0.9)",
        "text_color": "#1e293b",
        "text_muted": "#64748b",
        "border_color": "#cbd5e1",
        "accent_color": "#0284c7",
        "accent_hover": "rgba(2, 132, 199, 0.15)",
        "tab_bg": "rgba(228, 236, 242, 0.6)",
        "table_bg": "#ffffff",
        "table_even_bg": "#f8fafc",
        "input_border": "#cbd5e1",
        "plotly_template": "plotly_white",
        "color_scheme": "light"
    }
}

if "app_theme_selection" not in st.session_state:
    st.session_state["app_theme_selection"] = "Classic Navy (Default)"

theme_colors = THEMES[st.session_state["app_theme_selection"]]
st.session_state["theme_colors"] = theme_colors

#
# PLOTLY THEMING
# 1) A custom "fazdane" template carries the active theme's colors, so any
#    figure that doesn't set explicit colors is themed automatically
#    (pio.templates.default).
# 2) A small monkey-patch swaps the legacy hard-coded #0d1b2e backgrounds that
#    some modules still set explicitly. Remove it once no module hard-codes
#    that color. Originals are stored as class attributes so they survive
#    Streamlit reruns and never point at the patched version (recursion).
#
try:
    import copy as _copy
    import plotly.io as pio
    import plotly.graph_objects as go

    _base_template = theme_colors.get("plotly_template", "plotly_dark")
    _tpl = _copy.deepcopy(pio.templates[_base_template])
    _tpl.layout.paper_bgcolor = theme_colors["bg_color"]
    _tpl.layout.plot_bgcolor = theme_colors["bg_color"]
    _tpl.layout.font.color = theme_colors["text_color"]
    pio.templates["fazdane"] = _tpl
    pio.templates.default = "fazdane"

    if not hasattr(go.Figure, "_true_original_init"):
        # First time only: stash the real originals on the class itself
        go.Figure._true_original_init = go.Figure.__init__
        go.Figure._true_original_update_layout = go.Figure.update_layout

        def custom_init(self, *args, **kwargs):
            # Always call via the class attribute, never via a closure variable
            go.Figure._true_original_init(self, *args, **kwargs)
            try:
                theme = st.session_state.get("theme_colors")
                if theme and hasattr(self, "layout") and self.layout is not None:
                    bg_color = theme.get("bg_color")
                    self.layout.template = "fazdane"
                    if self.layout.paper_bgcolor == "#0d1b2e":
                        self.layout.paper_bgcolor = bg_color
                    if self.layout.plot_bgcolor == "#0d1b2e":
                        self.layout.plot_bgcolor = bg_color
            except Exception:
                logging.getLogger("FazDaneApp").debug(
                    "Plotly figure theming skipped", exc_info=True
                )

        def custom_update_layout(self, *args, **kwargs):
            try:
                theme = st.session_state.get("theme_colors")
                if theme:
                    bg_color = theme.get("bg_color")
                    if "template" in kwargs:
                        kwargs["template"] = "fazdane"
                    for key in ["paper_bgcolor", "plot_bgcolor"]:
                        if key in kwargs and kwargs[key] == "#0d1b2e":
                            kwargs[key] = bg_color
                    for arg in args:
                        if isinstance(arg, dict):
                            if arg.get("template") is not None:
                                arg["template"] = "fazdane"
                            for key in ["paper_bgcolor", "plot_bgcolor"]:
                                if arg.get(key) == "#0d1b2e":
                                    arg[key] = bg_color
            except Exception:
                logging.getLogger("FazDaneApp").debug(
                    "Plotly update_layout theming skipped", exc_info=True
                )
            return go.Figure._true_original_update_layout(self, *args, **kwargs)

        go.Figure.__init__ = custom_init
        go.Figure.update_layout = custom_update_layout
except Exception:
    logging.getLogger("FazDaneApp").warning(
        "Plotly theming setup failed", exc_info=True
    )


#
# LOGGING SETUP
#

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s  %(message)s",
    handlers=[
        logging.FileHandler("logs/fazdane.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("FazDaneApp")

TIER1_DEFAULT = "Select Live Trading Module..."
TIER2_DEFAULT = "Select Analysis Module..."
TIER3_DEFAULT = "Select Forecasting Module..."
TIER4_DEFAULT = "Select Volatility Module..."

#
# MODULE REGISTRY  single source of truth for every module:
# display name -> (import path, class name, tier).
# Sidebar radios, tier menus, and the home dashboard tabs are all generated
# from this dict, and the dispatcher imports lazily from it.
#
MODULE_REGISTRY = {
    # Tier 1  Live Trading
    "Search Module": ("modules.tier1.search_module", "SearchModule", 1),
    "Market Breadth Dashboard": ("modules.tier1.market_breadth", "MarketBreadthModule", 1),
    "Trade Intelligence Engine": ("modules.trade_recommendation.engine", "TradeRecommendationEngineModule", 1),
    "Calendar Strategy Matrix": ("modules.tier1.calendar_rotation", "CalendarRotationModule", 1),
    "Iron Condor Analyzer": ("modules.tier1.iron_condor", "IronCondorModule", 1),
    "ES Pivot Analysis": ("modules.tier1.es_pivot_analysis", "ESPivotAnalysisModule", 1),
    "Sector Rotation Monitor": ("modules.tier1.sector_rotation", "SectorRotationModule", 1),
    # Tier 2  Analysis & Intelligence
    "Universe Intelligence System": ("modules.tier2.universe_intelligence", "UniverseIntelligenceModule", 2),
    "Portfolio Module": ("modules.tier2.portfolio_module", "PortfolioModule", 2),
    "Multi-Timeframe Money Flow": ("modules.tier2.money_flow", "MoneyFlowModule", 2),
    "Market Trend Analysis": ("modules.tier2.market_trend_analysis", "MarketTrendAnalysisModule", 2),
    "Market Structure Heatmap": ("modules.tier2.market_structure", "MarketStructureModule", 2),
    "Correlation Matrix": ("modules.tier2.correlation_matrix", "CorrelationMatrixModule", 2),
    "Earnings Calendar": ("modules.tier2.earnings_calendar", "EarningsCalendarModule", 2),
    "Equity Income Statement": ("modules.tier2.equity_income_statement", "EquityIncomeStatementModule", 2),
    "Equity / Index Seasonality": ("modules.tier2.seasonality_analysis", "SeasonalityAnalysisModule", 2),
    "Stock Sentiment Analysis": ("modules.tier2.stock_sentiment", "StockSentimentModule", 2),
    "Social Stock Stories": ("modules.tier2.social_stock_stories", "SocialStockStoriesModule", 2),
    "Calendar Opportunity Scoring Engine": ("modules.calendar_scoring.dashboard", "CalendarOpportunityScoringModule", 2),
    "Price Action Story Engine": ("modules.tier2.price_action_story", "PriceActionStoryModule", 2),
    "Regime Intelligence Dashboard": ("modules.tier2.markov_regime_engine", "MarkovRegimeEngineModule", 2),
    # Tier 3  Forecasting & Cycles
    "Bradley Siderograph": ("modules.tier3.bradley_siderograph", "BradleySiderographModule", 3),
    "Elliott Wave Analysis": ("modules.tier3.elliott_wave_analysis", "ElliottWaveAnalysisModule", 3),
    "Cycle Analysis Engine": ("modules.cycle_engine.cycle_dashboard", "CycleAnalysisDashboardModule", 3),
    # Tier 4  Volatility
    "Volatility Strategy Engine": ("modules.tier4.volatility_engine", "VolatilityEngineModule", 4),
    "Gamma Flip Line Module": ("modules.tier4.gamma_flip.gamma_dashboard", "GammaFlipLineModule", 4),
}


def modules_for_tier(tier: int) -> list:
    """Ordered module names for a tier (order = registry insertion order)."""
    return [name for name, (_, _, t) in MODULE_REGISTRY.items() if t == tier]


def run_module(name: str) -> None:
    """Lazily import and run a registered module with uniform error handling."""
    import importlib
    import traceback

    module_path, class_name, _tier = MODULE_REGISTRY[name]
    try:
        module_cls = getattr(importlib.import_module(module_path), class_name)
        module_cls().run()
        logger.info(f"Module opened: {name}")
    except Exception as exc:
        st.error(f"Failed to load {name}: {exc}")
        st.code(traceback.format_exc())


def launch_module(module_name: str, tier: int) -> None:
    st.session_state.pop("active_menu_tier", None)
    st.session_state["pending_nav"] = {"module": module_name, "tier": tier}
    st.rerun()


def select_sidebar_tier(tier: int) -> None:
    """Keep sidebar radios mutually exclusive so every module link opens."""
    if tier == 1 and st.session_state.get("tier1_module_nav") != TIER1_DEFAULT:
        st.session_state["tier2_nav"] = TIER2_DEFAULT
        st.session_state["tier3_nav"] = TIER3_DEFAULT
        st.session_state["tier4_nav"] = TIER4_DEFAULT
        st.session_state.pop("active_menu_tier", None)
    elif tier == 2 and st.session_state.get("tier2_nav") != TIER2_DEFAULT:
        st.session_state["tier1_module_nav"] = TIER1_DEFAULT
        st.session_state["tier3_nav"] = TIER3_DEFAULT
        st.session_state["tier4_nav"] = TIER4_DEFAULT
        st.session_state.pop("active_menu_tier", None)
    elif tier == 3 and st.session_state.get("tier3_nav") != TIER3_DEFAULT:
        st.session_state["tier1_module_nav"] = TIER1_DEFAULT
        st.session_state["tier2_nav"] = TIER2_DEFAULT
        st.session_state["tier4_nav"] = TIER4_DEFAULT
        st.session_state.pop("active_menu_tier", None)
    elif tier == 4 and st.session_state.get("tier4_nav") != TIER4_DEFAULT:
        st.session_state["tier1_module_nav"] = TIER1_DEFAULT
        st.session_state["tier2_nav"] = TIER2_DEFAULT
        st.session_state["tier3_nav"] = TIER3_DEFAULT
        st.session_state.pop("active_menu_tier", None)


def render_home_module_button(label: str, module_name: str, tier: int, key: str) -> None:
    if st.button(label, key=key, width="stretch"):
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
        "pp_last_saved_hash",
        "pp_last_save_info",
    ]:
        st.session_state.pop(key, None)
    st.rerun()


def back_to_current_menu() -> None:
    """Return from the active module to the matching module menu."""
    if st.session_state.get("tier1_module_nav") and st.session_state.get("tier1_module_nav") != TIER1_DEFAULT:
        st.session_state["pending_nav"] = {"action": "clear_tier", "tier": 1}
    elif st.session_state.get("tier2_nav") and st.session_state.get("tier2_nav") != TIER2_DEFAULT:
        st.session_state["pending_nav"] = {"action": "clear_tier", "tier": 2}
    elif st.session_state.get("tier3_nav") and st.session_state.get("tier3_nav") != TIER3_DEFAULT:
        st.session_state["pending_nav"] = {"action": "clear_tier", "tier": 3}
    elif st.session_state.get("tier4_nav") and st.session_state.get("tier4_nav") != TIER4_DEFAULT:
        st.session_state["pending_nav"] = {"action": "clear_tier", "tier": 4}
    else:
        st.session_state["active_menu_tier"] = st.session_state.get("active_menu_tier", 1)
    st.rerun()

#
# GLOBAL CSS  Research & Trading Intelligence Platform Brand
# Colors: navy #0d1b2e, blue #1a3a8f, green #3ab54a
# Built once per theme and cached across reruns (st.cache_data).
#

@st.cache_data(show_spinner=False)
def build_css(theme_name: str) -> str:
    theme_colors = THEMES[theme_name]
    return f"""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Courier+Prime:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            color-scheme: {theme_colors['color_scheme']} !important;
            --bg-color: {theme_colors['bg_color']};
            --sidebar-bg: {theme_colors['sidebar_bg']};
            --sidebar-bg-solid: {theme_colors['sidebar_bg_solid']};
            --card-bg: {theme_colors['card_bg']};
            --card-bg-solid: {theme_colors['card_bg_solid']};
            --widget-bg: {theme_colors['widget_bg']};
            --text-color: {theme_colors['text_color']};
            --text-muted: {theme_colors['text_muted']};
            --border-color: {theme_colors['border_color']};
            --accent-color: {theme_colors['accent_color']};
            --accent-hover: {theme_colors['accent_hover']};
            --tab-bg: {theme_colors['tab_bg']};
            --table-bg: {theme_colors['table_bg']};
            --table-even-bg: {theme_colors['table_even_bg']};
            --input-border: {theme_colors['input_border']};

            /* Override Streamlit native theme variables on root for canvas dataframes & UI widgets */
            --primary-color: {theme_colors['accent_color']} !important;
            --background-color: {theme_colors['bg_color']} !important;
            --secondary-background-color: {theme_colors['sidebar_bg_solid']} !important;
            --text-color: {theme_colors['text_color']} !important;

            /* Override Streamlit prefixed native variables */
            --st-color-background: {theme_colors['bg_color']} !important;
            --st-color-text: {theme_colors['text_color']} !important;
            --st-color-primary: {theme_colors['accent_color']} !important;
            --st-color-secondary-background: {theme_colors['sidebar_bg_solid']} !important;
        }}

        /* Base */
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

        /* Background */
        .stApp {{ background: var(--bg-color); color: var(--text-color); }}

        /* Make default Streamlit header bar transparent and click-through */
        header[data-testid="stHeader"] {{
            background-color: transparent !important;
            border-bottom: none !important;
            height: 40px !important;
            pointer-events: none !important;
        }}
        header[data-testid="stHeader"] * {{
            pointer-events: auto !important;
        }}

        /* Enable pointer-events on the header for mobile/tablet screens to fix the iOS sidebar expand click bug */
        @media (max-width: 768px) {{
            header[data-testid="stHeader"] {{
                pointer-events: auto !important;
            }}
        }}

        /* Force-show the sidebar collapse/expand button and ensure it is clickable and visible */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stSidebarCollapseButton"] svg,
        [data-testid="stExpandSidebarButton"],
        [data-testid="stExpandSidebarButton"] button,
        [data-testid="stExpandSidebarButton"] svg,
        [data-testid="collapsedControl"],
        [data-testid="collapsedControl"] button,
        [data-testid="collapsedControl"] svg {{
            opacity: 1 !important;
            visibility: visible !important;
            display: inline-flex !important;
            pointer-events: auto !important;
            color: var(--text-color) !important;
            fill: var(--text-color) !important;
            z-index: 999999 !important;
        }}

        /* Hide decorative bar, toolbar, hamburger menu, deploy button, and footer */
        [data-testid="stDecoration"],
        [data-testid="stToolbar"],
        [data-testid="stHeaderDeployButton"],
        #MainMenu,
        footer {{
            visibility: hidden !important;
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
        }}

        /* ALL text */
        p, li, span, label, small,
        .stMarkdown p, .stMarkdown li, .stMarkdown span {{ color: var(--text-color); }}

        /* Widget labels */
        .stTextInput label, .stTextArea label, .stSelectbox label,
        .stMultiSelect label, .stSlider label, .stRadio label,
        .stCheckbox label, .stNumberInput label, .stDateInput label,
        [data-testid="stWidgetLabel"] p {{
            color: var(--text-muted) !important; font-size: 13px !important; font-weight: 500 !important;
        }}

        /* Radio / checkbox text */
        .stRadio div[role="radiogroup"] label p,
        .stRadio div[role="radiogroup"] label span,
        .stCheckbox label p {{ color: var(--text-color) !important; }}
        .stRadio div[role="radiogroup"] label {{
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
            min-height: 28px !important;
        }}
        .stRadio div[role="radiogroup"] label > div:first-child {{
            margin-top: 0 !important;
            align-self: center !important;
        }}
        .stRadio div[role="radiogroup"] label p {{
            margin: 0 !important;
            line-height: 1.25 !important;
        }}

        /* Selectbox */
        .stSelectbox [data-baseweb="select"] span,
        .stSelectbox [data-baseweb="select"] div,
        [data-baseweb="select"] span {{
            color: var(--text-color) !important; background-color: var(--widget-bg) !important;
        }}

        /* Multiselect tags */
        .stMultiSelect [data-baseweb="tag"] {{
            background: rgba(26,58,143,0.75) !important;
            border: 1px solid var(--accent-color) !important;
            border-radius: 6px !important;
            padding: 4px 8px !important;
        }}
        .stMultiSelect [data-baseweb="tag"] span {{
            color: #ffffff !important;
            font-size: 13px !important;
            font-weight: 600 !important;
        }}
        .stMultiSelect [data-baseweb="tag"] [data-testid="stMultiSelectClearButton"],
        .stMultiSelect [data-baseweb="tag"] button,
        .stMultiSelect [data-baseweb="tag"] svg {{ color: var(--text-muted) !important; fill: var(--text-muted) !important; }}
        .stMultiSelect [data-baseweb="select"] span,
        .stMultiSelect [data-baseweb="select"] div {{ color: var(--text-color) !important; }}
        .stMultiSelect [data-baseweb="select"] input {{ color: var(--text-color) !important; }}

        /* Dropdown option lists */
        [data-baseweb="popover"] li, [data-baseweb="menu"] li,
        [role="option"], [role="listbox"] li {{
            color: var(--text-color) !important; background: var(--sidebar-bg-solid) !important;
        }}
        [data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover,
        [role="option"]:hover {{ background: var(--accent-hover) !important; }}

        /* Text input & textarea */
        .stTextInput > div > div > input, .stTextArea textarea, .stNumberInput input {{
            background: var(--widget-bg) !important; border-color: var(--input-border) !important;
            color: var(--text-color) !important; border-radius: 8px !important;
        }}
        .stTextInput > div > div > input::placeholder,
        .stTextArea textarea::placeholder,
        .stNumberInput input::placeholder {{ color: var(--text-muted) !important; }}

        /* Slider */
        .stSlider > div > div > div > div {{ background: var(--accent-color) !important; }}
        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"] {{ color: var(--text-muted) !important; }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
        }}
        /* NOTE: the old [data-sidebar-collapsed] force-open hack was removed.
           That attribute does not exist in Streamlit 1.55, so it was dead CSS.
           Sidebar auto-expand is now handled by the JS watchdog injected below. */

        /* Hide Streamlit's automatic multipage nav (pages/auth.py would otherwise
           appear as a clickable "auth" page at the top of the sidebar) */
        [data-testid="stSidebarNav"] {{
            display: none !important;
        }}
        [data-testid="stSidebar"] * {{ color: var(--text-color) !important; }}
        [data-testid="stSidebar"] .stButton > button {{
            min-height: 36px;
            padding: 7px 10px !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            line-height: 1.25 !important;
        }}
        [data-testid="stSidebar"] [data-testid="column"] .stButton > button {{
            min-height: 34px;
            padding: 6px 4px !important;
            font-size: 12px !important;
            line-height: 1.15 !important;
            white-space: nowrap;
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {{
            min-height: 34px;
            padding: 7px 8px !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
            font-size: 13px !important;
            font-weight: 600 !important;
            line-height: 1.25 !important;
        }}
        [data-testid="stSidebar"] .stRadio label {{
            padding: 5px 7px;
            border-radius: 6px;
            transition: background 0.2s;
            min-height: 30px;
        }}
        [data-testid="stSidebar"] .stRadio label p,
        [data-testid="stSidebar"] .stRadio label span {{
            font-size: 12.5px !important;
            font-weight: 500 !important;
            line-height: 1.25 !important;
        }}
        [data-testid="stSidebar"] .stRadio label:hover {{ background: var(--accent-hover) !important; }}
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color: var(--text-muted) !important; }}
        [data-testid="stSidebar"] [data-testid="stExpander"] {{
            background: rgba(13,27,46,0.24);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            margin-bottom: 8px;
            overflow: hidden;
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{
            background: var(--accent-hover) !important;
        }}

        /* Main content */
        .stMainBlockContainer {{ background: var(--bg-color) !important; padding-top: 0.75rem; }}

        /* Headings */
        h1, h2, h3 {{ font-family: 'Courier Prime', monospace !important; color: var(--accent-color) !important; }}
        h4, h5, h6 {{ color: var(--text-muted) !important; }}

        /* Metrics */
        [data-testid="metric-container"] {{
            background: var(--card-bg);
            border: 1px solid var(--border-color); border-radius: 10px; padding: 16px 20px;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}
        [data-testid="metric-container"]:hover {{ border-color: var(--accent-color); box-shadow: 0 0 16px var(--accent-hover); }}
        [data-testid="metric-container"] [data-testid="stMetricLabel"] p {{
            color: var(--text-muted) !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 1px;
        }}
        [data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: var(--text-color) !important; font-size: 24px !important; font-weight: 700; }}
        [data-testid="metric-container"] [data-testid="stMetricDelta"] {{ font-size: 13px !important; }}

        /* Buttons */
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, var(--sidebar-bg-solid) 0%, var(--accent-color) 100%);
            color: #ffffff !important; font-weight: 600; border: none;
            border-radius: 8px; padding: 10px 20px; transition: all 0.25s ease;
            box-shadow: 0 2px 8px var(--accent-hover);
        }}
        .stButton > button[kind="primary"]:hover {{ transform: translateY(-1px); box-shadow: 0 4px 20px var(--accent-hover); }}
        .stButton > button[kind="secondary"] {{
            background: rgba(26,58,143,0.25); color: var(--text-color) !important;
            border: 1px solid var(--border-color); border-radius: 8px; transition: all 0.2s;
        }}
        .stButton > button[kind="secondary"]:hover {{ background: var(--accent-hover); border-color: var(--accent-color); }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background: var(--tab-bg); border-radius: 10px; padding: 4px; gap: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px !important; color: var(--text-muted) !important;
            padding: 8px 20px !important; font-weight: 500; transition: all 0.2s;
        }}
        .stTabs [data-baseweb="tab"] p {{ color: var(--text-muted) !important; }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            background: var(--accent-hover) !important; color: var(--accent-color) !important; border-bottom: 2px solid var(--accent-color) !important;
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] p {{ color: var(--accent-color) !important; }}

        /* Dataframe */
        [data-testid="stDataFrameContainer"] {{
            background: var(--table-bg) !important;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            overflow: hidden;
        }}
        [data-testid="stDataFrameContainer"] *,
        [data-testid="stDataFrame"] *,
        .stDataFrame *,
        .stTable * {{
            color: var(--text-color) !important;
        }}
        [data-testid="stDataFrameContainer"] > div,
        [data-testid="stDataFrameContainer"] section,
        [data-testid="stDataFrame"] > div,
        .stDataFrame > div {{
            background-color: var(--table-bg) !important;
        }}
        [data-testid="stDataFrameContainer"] canvas {{
            background-color: transparent !important;
        }}
        [data-testid="stDataFrameContainer"] th,
        [data-testid="stDataFrameContainer"] thead tr,
        [data-testid="stDataFrame"] th,
        .stDataFrame th,
        .stTable th {{
            background: var(--sidebar-bg-solid) !important;
            color: var(--text-muted) !important;
        }}
        [data-testid="stDataFrameContainer"] td,
        [data-testid="stDataFrame"] td,
        .stDataFrame td,
        .stTable td {{
            background: var(--table-bg) !important;
            color: var(--text-color) !important;
            border-color: var(--border-color) !important;
        }}
        [data-testid="stDataFrameContainer"] tbody tr:nth-child(even) td,
        [data-testid="stDataFrame"] tbody tr:nth-child(even) td,
        .stDataFrame tbody tr:nth-child(even) td,
        .stTable tbody tr:nth-child(even) td {{
            background: var(--table-even-bg) !important;
        }}

        /* Pandas Styler / markdown tables */
        table {{
            background: var(--table-bg) !important;
            color: var(--text-color) !important;
            border-color: var(--border-color) !important;
        }}
        table th {{
            background: var(--sidebar-bg-solid) !important;
            color: var(--text-muted) !important;
            border-color: var(--border-color) !important;
        }}
        table td {{
            background: var(--table-bg) !important;
            color: var(--text-color) !important;
            border-color: var(--border-color) !important;
        }}

        /* Misc */
        .stMarkdown em, .stCaption {{ color: var(--text-muted) !important; }}
        hr {{ border-color: var(--border-color) !important; }}
        .stAlert {{ border-radius: 10px !important; }}
        .stAlert p, .stAlert span {{ color: var(--text-color) !important; }}
        [data-testid="stExpander"] summary p {{ color: var(--accent-color) !important; }}
        .stSpinner p {{ color: var(--text-muted) !important; }}

        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-color); }}
        ::-webkit-scrollbar-thumb {{ background: var(--border-color); border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--accent-color); }}

        /*  Sidebar multiselect & selectbox  remove white box  */
        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"],
        [data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {{
            background-color: rgba(13, 27, 46, 0.0) !important;
            border-color: var(--border-color) !important;
        }}
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {{
            background-color: var(--sidebar-bg-solid) !important;
            border-color: var(--border-color) !important;
        }}

        /*  All multiselect containers  transparent bg in sidebar  */
        [data-testid="stSidebar"] div[data-baseweb="select"] {{
            background: transparent !important;
        }}
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background: var(--tab-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }}

        /*  Selectbox dropdown in sidebar  */
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
            background: var(--sidebar-bg-solid) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            color: var(--text-color) !important;
        }}

    </style>
    """


st.markdown(build_css(st.session_state["app_theme_selection"]), unsafe_allow_html=True)

#
# SIDEBAR AUTO-EXPAND WATCHDOG
#
# Bug: Streamlit >= 1.3x persists the sidebar collapsed/expanded state in the
# browser's localStorage (key "stSidebarCollapsed-<appId>"), and that saved
# value OVERRIDES initial_sidebar_state="expanded" on every page load.
# Once the sidebar gets collapsed once (e.g. auto-collapse in a narrow
# preview iframe, or an accidental click), the app keeps opening with the
# sidebar hidden until the user manually reloads/expands it.
#
# Fix: a tiny script that (1) clears the sticky "collapsed" flag from
# localStorage and (2) for a few seconds after load, clicks Streamlit's own
# expand control if the sidebar is still collapsed. Clicking the real button
# keeps Streamlit's internal state consistent (no CSS layout hacks needed).
#
import streamlit.components.v1 as _components

_components.html(
    """
    <script>
    (function () {
        try {
            var doc = window.parent.document;
            var ls = window.parent.localStorage;
            // 1) Neutralize the sticky "collapsed" state Streamlit saves per app
            for (var i = 0; i < ls.length; i++) {
                var k = ls.key(i);
                if (k && k.indexOf("stSidebarCollapsed") === 0 && ls.getItem(k) === "true") {
                    ls.setItem(k, "false");
                }
            }
            // 2) Watchdog: if the sidebar is collapsed or missing, click the
            //    expand control. Retry for ~10s to survive slow first renders
            //    inside preview iframes, then stop.
            var attempts = 0;
            var timer = setInterval(function () {
                attempts += 1;
                var sidebar = doc.querySelector('section[data-testid="stSidebar"]');
                var expanded = sidebar &&
                    sidebar.getAttribute("aria-expanded") !== "false" &&
                    sidebar.offsetWidth > 50;
                if (expanded || attempts > 40) { clearInterval(timer); return; }
                var btn = doc.querySelector('[data-testid="stExpandSidebarButton"] button') ||
                          doc.querySelector('[data-testid="stExpandSidebarButton"]');
                if (btn) { btn.click(); }
            }, 250);

            // 3) Mobile Sidebar Button Injection
            if (!doc.getElementById('mobile-sidebar-btn')) {
                var mobileBtn = doc.createElement('button');
                mobileBtn.id = 'mobile-sidebar-btn';
                mobileBtn.innerHTML = '☰';
                mobileBtn.style.cssText = 'display: none; position: fixed; bottom: 20px; right: 20px; background-color: #3ab54a; color: white; border: none; border-radius: 50%; width: 56px; height: 56px; font-size: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); z-index: 9999999; cursor: pointer; text-align: center; line-height: 56px; transition: all 0.2s ease;';
                
                mobileBtn.onclick = function() {
                    var expandBtn = doc.querySelector('[data-testid="stExpandSidebarButton"] button') || 
                                    doc.querySelector('[data-testid="stExpandSidebarButton"]') || 
                                    doc.querySelector('[data-testid="collapsedControl"] button') ||
                                    doc.querySelector('[data-testid="collapsedControl"] svg') ||
                                    doc.querySelector('[data-testid="collapsedControl"]');
                    if(expandBtn) { 
                        expandBtn.click(); 
                    }
                };
                
                doc.body.appendChild(mobileBtn);
                
                function checkSize() {
                    if (doc.body.clientWidth <= 768) {
                        mobileBtn.style.display = 'block';
                    } else {
                        mobileBtn.style.display = 'none';
                    }
                }
                doc.defaultView.addEventListener('resize', checkSize);
                checkSize();
            }
        } catch (e) { /* sandboxed iframe or storage blocked - ignore */ }
    })();
    </script>
    """,
    height=0,
)

#
# AUTHENTICATION CHECK
#

from views.auth import FazDaneAuthenticator, logout, get_current_user

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    # Renders the sidebar with branding and system status when logged out
    with st.sidebar:
        # Logo
        try:
            st.image("assets/logo.png", width="stretch")
        except Exception:
            st.markdown(
                "<h2 style='color:#3ab54a;text-align:center;'>Research & Trading Intelligence Platform</h2>",
                unsafe_allow_html=True,
            )
        
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div style="
                background: rgba(26,58,143,0.15);
                border: 1px solid #1e3a5f;
                border-radius: 8px;
                padding: 12px 14px;
                margin-bottom: 16px;
            ">
                <div style="color:#3ab54a;font-weight:700;font-size:14px;margin-bottom:4px;">SYSTEM STATUS</div>
                <div style="color:#e2e8f0;font-size:12px;">📊 Broad VIX: Connected</div>
                <div style="color:#e2e8f0;font-size:12px;">🗄️ Database: Active</div>
                <div style="color:#e2e8f0;font-size:12px;">⚡ API Engine: Online</div>
            </div>
            <div style="text-align:center;color:#64748b;font-size:11px;margin-top:20px;">
                Please authenticate in the main panel to access workspace modules.
            </div>
            """,
            unsafe_allow_html=True
        )
    
    authenticator = FazDaneAuthenticator()
    authenticator.render_login_screen()
    st.stop()

# Initialize databases from cloud backup on startup
if "db_initialized" not in st.session_state:
    try:
        from utils.persistence import restore_all_databases, initialize_volatility_cache_tables

        # force=False: databases already present on disk are used as-is, so the
        # blocking cloud download is skipped on startup. Use the sidebar's
        # database control panel for a forced restore from cloud.
        with st.spinner("Checking local databases..."):
            restored, failed = restore_all_databases(force=False)
            if restored:
                st.session_state["db_restore_msg"] = f"Restored: {', '.join(restored)}"
            if failed:
                st.session_state["db_restore_err"] = f"Failed: {', '.join(failed)}"
        # Ensure volatility cache tables exist after databases are restored
        initialize_volatility_cache_tables()
        
        # Ensure trade recommendation database and tables exist on startup
        try:
            from modules.trade_recommendation.database import create_tables as create_trade_tables
            create_trade_tables()
        except Exception as e:
            logger.warning(f"Failed to initialize trade recommendation database: {e}")
    except ImportError as exc:
        logger.exception("Database persistence import failed")
        st.session_state["db_restore_err"] = f"Database persistence unavailable: {exc}"
    except Exception as exc:
        logger.exception("Database persistence initialization failed")
        st.session_state["db_restore_err"] = f"Database initialization skipped: {exc}"
    st.session_state["db_initialized"] = True

#
# SIDEBAR  Navigation
#

user = get_current_user()

pending_nav = st.session_state.pop("pending_nav", None)
if pending_nav:
    action = pending_nav.get("action")
    module_name = pending_nav.get("module")
    tier = pending_nav.get("tier")
    if action == "home":
        st.session_state["tier1_module_nav"] = TIER1_DEFAULT
        st.session_state["tier2_nav"] = TIER2_DEFAULT
        st.session_state["tier3_nav"] = TIER3_DEFAULT
        st.session_state["tier4_nav"] = TIER4_DEFAULT
        st.session_state.pop("active_menu_tier", None)
    elif action == "clear_tier" and tier == 1:
        st.session_state["tier1_module_nav"] = TIER1_DEFAULT
        st.session_state["active_menu_tier"] = 1
    elif action == "clear_tier" and tier == 2:
        st.session_state["tier2_nav"] = TIER2_DEFAULT
        st.session_state["active_menu_tier"] = 2
    elif action == "clear_tier" and tier == 3:
        st.session_state["tier3_nav"] = TIER3_DEFAULT
        st.session_state["active_menu_tier"] = 3
    elif action == "clear_tier" and tier == 4:
        st.session_state["tier4_nav"] = TIER4_DEFAULT
        st.session_state["active_menu_tier"] = 4
    elif tier == 1:
        st.session_state.pop("active_menu_tier", None)
        st.session_state["tier1_module_nav"] = module_name
        st.session_state["tier2_nav"] = TIER2_DEFAULT
        st.session_state["tier3_nav"] = TIER3_DEFAULT
        st.session_state["tier4_nav"] = TIER4_DEFAULT
    elif tier == 2:
        st.session_state.pop("active_menu_tier", None)
        st.session_state["tier1_module_nav"] = TIER1_DEFAULT
        st.session_state["tier2_nav"] = module_name
        st.session_state["tier3_nav"] = TIER3_DEFAULT
        st.session_state["tier4_nav"] = TIER4_DEFAULT
    elif tier == 3:
        st.session_state.pop("active_menu_tier", None)
        st.session_state["tier1_module_nav"] = TIER1_DEFAULT
        st.session_state["tier2_nav"] = TIER2_DEFAULT
        st.session_state["tier3_nav"] = module_name
        st.session_state["tier4_nav"] = TIER4_DEFAULT
    elif tier == 4:
        st.session_state.pop("active_menu_tier", None)
        st.session_state["tier1_module_nav"] = TIER1_DEFAULT
        st.session_state["tier2_nav"] = TIER2_DEFAULT
        st.session_state["tier3_nav"] = TIER3_DEFAULT
        st.session_state["tier4_nav"] = module_name

with st.sidebar:
    # Logo
    try:
        st.image("assets/logo.png", width="stretch")
    except Exception:
        st.markdown(
            "<h2 style='color:#3ab54a;text-align:center;'>Research & Trading Intelligence Platform</h2>",
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
        if st.button("Home", width="stretch", key="home_dashboard_top_nav"):
            st.session_state["pending_nav"] = {"action": "home"}
            st.rerun()
    with top_nav_col2:
        if st.button("Menu", width="stretch", key="back_to_menu_top_nav"):
            back_to_current_menu()
    with top_nav_col3:
        if st.button("Refresh", width="stretch", key="refresh_data_nav"):
            refresh_live_data()

    # Theme Selector Widget
    st.markdown(
        """
        <div style="color:#94a3b8;font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;margin:12px 0 6px 0;">
            Theme / Display
        </div>
        """,
        unsafe_allow_html=True,
    )
    theme_names = list(THEMES.keys())
    selected_theme = st.selectbox(
        "Theme",
        options=theme_names,
        index=theme_names.index(st.session_state.get("app_theme_selection", "Classic Navy (Default)")),
        key="app_theme_selection_widget",
        label_visibility="collapsed"
    )
    if selected_theme != st.session_state.get("app_theme_selection"):
        st.session_state["app_theme_selection"] = selected_theme
        st.session_state["theme_colors"] = THEMES[selected_theme]
        st.rerun()

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

    restore_msg = st.session_state.get("db_restore_msg")
    restore_err = st.session_state.get("db_restore_err")
    if restore_msg:
        st.caption(f"🗄️ {restore_msg}")
    if restore_err:
        st.caption(f"⚠️ {restore_err}")

    # Database manual control and logs panel
    try:
        from utils.persistence import render_db_control_panel
        render_db_control_panel()
    except ImportError as exc:
        st.caption(f"Database controls unavailable: {exc}")
    except Exception as exc:
        st.caption(f"Database controls unavailable: {exc}")

    st.divider()

    st.markdown(
        """
        <div style="color:#3ab54a;font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;margin:2px 0 8px 0;">
            Modules
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Live Trading", expanded=True):
        st.markdown(
        "<div style='color:#3ab54a;font-size:12px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;'> Live Trading</div>",
        unsafe_allow_html=True,
        )

        tier1_options = [TIER1_DEFAULT] + modules_for_tier(1)

        tier1_sel = st.radio(
            "tier1",
            tier1_options,
            key="tier1_module_nav",
            label_visibility="collapsed",
            on_change=select_sidebar_tier,
            args=(1,),
        )

    st.divider()

    #  Tier 2: Analysis
    with st.expander("Analysis & Intelligence", expanded=False):
        tier2_options = [TIER2_DEFAULT] + modules_for_tier(2)
        tier2_sel = st.radio(
            "tier2",
            tier2_options,
            key="tier2_nav",
            label_visibility="collapsed",
            on_change=select_sidebar_tier,
            args=(2,),
        )

    #  Tier 3: Forecasting
    with st.expander("Forecasting & Cycles", expanded=False):
        tier3_options = [TIER3_DEFAULT] + modules_for_tier(3)
        tier3_sel = st.radio(
            "tier3",
            tier3_options,
            key="tier3_nav",
            label_visibility="collapsed",
            on_change=select_sidebar_tier,
            args=(3,),
        )

    #  Tier 4: Volatility
    with st.expander("Volatility", expanded=False):
        tier4_options = [TIER4_DEFAULT] + modules_for_tier(4)
        tier4_sel = st.radio(
            "tier4",
            tier4_options,
            key="tier4_nav",
            label_visibility="collapsed",
            on_change=select_sidebar_tier,
            args=(4,),
        )

    st.divider()

    #  Logout
    if st.button("Logout", width="stretch", type="secondary"):
        logger.info(f"User {user['username']} logged out")
        logout()

    st.markdown(
        f"<div style='text-align:center;color:#475569;font-size:11px;margin-top:12px;'>Research & Trading Intelligence Platform {VERSION}</div>",
        unsafe_allow_html=True
    )

#
# MODULE DISPATCHER
#

# Determine which module is active  tier1 takes precedence
if tier1_sel and tier1_sel != TIER1_DEFAULT:
    active_module = tier1_sel
elif st.session_state.get("tier2_nav") and st.session_state.get("tier2_nav") != tier2_options[0]:
    active_module = st.session_state.get("tier2_nav")
elif st.session_state.get("tier3_nav") and st.session_state.get("tier3_nav") != tier3_options[0]:
    active_module = st.session_state.get("tier3_nav")
elif st.session_state.get("tier4_nav") and st.session_state.get("tier4_nav") != tier4_options[0]:
    active_module = st.session_state.get("tier4_nav")
elif st.session_state.get("active_menu_tier"):
    active_module = f"__MENU_TIER_{st.session_state.get('active_menu_tier')}__"
else:
    active_module = "Home Dashboard"

# NOTE: the old "Loading module..." intermediate rerun was removed  it added
# a full extra script execution to every navigation click. Streamlit already
# clears the previous frame on rerun.

#  Route to module

if active_module.startswith("__MENU_TIER_"):
    menu_tier = int(st.session_state.get("active_menu_tier", 1))
    menu_config = {
        1: ("Live Trading", "Tier 1 execution tools and real-time market dashboards", tier1_options[1:]),
        2: ("Analysis & Intelligence", "Research modules for cross-market, fundamental, seasonal, and sentiment work", tier2_options[1:]),
        3: ("Forecasting & Cycles", "Cycle and forecasting modules for turning-window analysis", tier3_options[1:]),
        4: ("Volatility", "Dealer gamma, implied-volatility, and premium-selling regime dashboards", tier4_options[1:]),
    }

    try:
        st.image("assets/logo.png", width=240)
    except Exception:
        st.title("Research & Trading Intelligence Platform")

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

    switch_cols = st.columns(4)
    for tier_id, (switch_label, _, _) in menu_config.items():
        with switch_cols[tier_id - 1]:
            button_type = "primary" if menu_tier == tier_id else "secondary"
            if st.button(switch_label, key=f"module_menu_switch_{tier_id}", width="stretch", type=button_type):
                st.session_state["active_menu_tier"] = tier_id
                st.rerun()

    st.divider()
    st.markdown("### Select Module")
    cols = st.columns(2)
    for index, option in enumerate(options):
        with cols[index % 2]:
            if st.button(option, key=f"back_menu_{menu_tier}_{index}", width="stretch"):
                launch_module(option, menu_tier)

elif active_module in MODULE_REGISTRY:
    run_module(active_module)

elif active_module in tier2_options:
    st.info(f"{active_module}  Analysis module coming in Weeks 34")

elif active_module in tier3_options:
    st.info(f"{active_module}  Forecasting module coming in Weeks 56")

elif active_module in tier4_options:
    st.info(f"{active_module}  Volatility module coming soon")

else:
    #
    # HOME DASHBOARD
    #
    try:
        st.image("assets/logo.png", width=280)
    except Exception:
        st.title("Research & Trading Intelligence Platform")

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
                Welcome back, {user['display_name']}!
            </div>
            <div style="color:#94a3b8;font-size:14px;">
                Your personal finance research platform is ready. Select a module from the sidebar to begin.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Home-dashboard tabs are generated from MODULE_REGISTRY (single source
    # of truth) instead of a duplicated hand-maintained list.
    import re as _re

    def _macro_key(name: str) -> str:
        return "macro_" + _re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    macro_module_tabs = [
        {
            "label": tab_label,
            "items": [
                {"label": m, "module": m, "tier": tier_id, "key": _macro_key(m)}
                for m in modules_for_tier(tier_id)
            ],
        }
        for tier_id, tab_label in [
            (1, "Live Trading"),
            (2, "Analysis & Intelligence"),
            (3, "Forecasting"),
            (4, "Volatility"),
        ]
    ]

    from modules.tier2.macro_intelligence import render_macro_dashboard
    render_macro_dashboard(show_download=True, module_tabs=macro_module_tabs, launch_callback=launch_module)

    st.divider()
    st.markdown(
        f"<p style='text-align:center;color:#334155;font-size:12px;'>Copyright (c) FazDane Analytics | Research & Trading Intelligence Platform {VERSION} · 2026 All Rights Reserved  Built on Streamlit</p>",
        unsafe_allow_html=True,
    )
