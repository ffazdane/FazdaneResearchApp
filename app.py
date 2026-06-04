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
# PLOTLY THEME MONKEY-PATCHING
#
try:
    import plotly.graph_objects as go
    
    _original_init = go.Figure.__init__
    _original_update_layout = go.Figure.update_layout

    def custom_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        theme = st.session_state.get("theme_colors")
        if theme and hasattr(self, "layout") and self.layout is not None:
            bg_color = theme.get("bg_color")
            template = theme.get("plotly_template", "plotly_dark")
            self.layout.template = template
            if self.layout.paper_bgcolor == "#0d1b2e":
                self.layout.paper_bgcolor = bg_color
            if self.layout.plot_bgcolor == "#0d1b2e":
                self.layout.plot_bgcolor = bg_color

    def custom_update_layout(self, *args, **kwargs):
        theme = st.session_state.get("theme_colors")
        if theme:
            bg_color = theme.get("bg_color")
            template = theme.get("plotly_template", "plotly_dark")
            
            if "template" in kwargs:
                kwargs["template"] = template
                
            for key in ["paper_bgcolor", "plot_bgcolor"]:
                if key in kwargs and kwargs[key] == "#0d1b2e":
                    kwargs[key] = bg_color
                    
            for arg in args:
                if isinstance(arg, dict):
                    if arg.get("template") is not None:
                        arg["template"] = template
                    for key in ["paper_bgcolor", "plot_bgcolor"]:
                        if arg.get(key) == "#0d1b2e":
                            arg[key] = bg_color
                            
        return _original_update_layout(self, *args, **kwargs)

    go.Figure.__init__ = custom_init
    go.Figure.update_layout = custom_update_layout
except Exception as e:
    pass


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
#

st.markdown(
    f"""
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

        /* Permanently hide sidebar collapse and expand controls */
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"] {{
            opacity: 0 !important;
            pointer-events: none !important;
            position: absolute !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
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
    <script>
        (function() {{
            function expandSidebar() {{
                const sidebar = document.querySelector('[data-testid="stSidebar"]');
                const expandButton = document.querySelector('[data-testid="collapsedControl"] button');
                if (sidebar && sidebar.getAttribute('data-collapsed') === 'true' && expandButton) {{
                    expandButton.click();
                }}
            }}
            expandSidebar();
            const interval = setInterval(expandSidebar, 100);
            setTimeout(() => clearInterval(interval), 3000);
        }})();
    </script>
    """,
    unsafe_allow_html=True,
)

#
# AUTHENTICATION CHECK
#

from pages.auth import FazDaneAuthenticator, logout, get_current_user

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    authenticator = FazDaneAuthenticator()
    authenticator.render_login_screen()
    st.stop()

# Initialize databases from cloud backup on startup
if "db_initialized" not in st.session_state:
    try:
        from utils.persistence import restore_all_databases, initialize_volatility_cache_tables

        with st.spinner("Restoring databases from cloud..."):
            restored, failed = restore_all_databases(force=True)
            if restored:
                st.session_state["db_restore_msg"] = f"Restored: {', '.join(restored)}"
            if failed:
                st.session_state["db_restore_err"] = f"Failed: {', '.join(failed)}"
        # Ensure volatility cache tables exist after databases are restored
        initialize_volatility_cache_tables()
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
        st.image("assets/logo.png", use_container_width=True)
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
        if st.button("Home", use_container_width=True, key="home_dashboard_top_nav"):
            st.session_state["pending_nav"] = {"action": "home"}
            st.rerun()
    with top_nav_col2:
        if st.button("Menu", use_container_width=True, key="back_to_menu_top_nav"):
            back_to_current_menu()
    with top_nav_col3:
        if st.button("Refresh", use_container_width=True, key="refresh_data_nav"):
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

        tier1_options = [
            TIER1_DEFAULT,
            "Search Module",
            "Market Breadth Dashboard",
            "Calendar Strategy Matrix",
            "Iron Condor Analyzer",
            "ES Pivot Analysis",
            "Sector Rotation Monitor",
        ]

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
        tier2_options = [
            TIER2_DEFAULT,
            "Universe Intelligence System",
            "Portfolio Module",
            "Multi-Timeframe Money Flow",
            "Market Structure Heatmap",
            "Correlation Matrix",
            "Earnings Calendar",
            "Equity Income Statement",
            "Equity / Index Seasonality",
            "Stock Sentiment Analysis",
            "Social Stock Stories",
            "Calendar Opportunity Scoring Engine",
            "Price Action Story Engine",
            "Regime Intelligence Dashboard",
        ]
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
        tier3_options = [
            TIER3_DEFAULT,
            "Bradley Siderograph",
            "Elliott Wave Analysis",
        ]
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
        tier4_options = [
            TIER4_DEFAULT,
            "Volatility Strategy Engine",
            "Gamma Flip Line Module",
        ]
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
    if st.button("Logout", use_container_width=True, type="secondary"):
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

# Detect module transitions to clear previous visualizations immediately
if "last_active_module" not in st.session_state:
    st.session_state["last_active_module"] = active_module

if active_module != st.session_state["last_active_module"]:
    st.session_state["last_active_module"] = active_module
    st.markdown(
        """
        <div style="text-align:center; padding:100px 20px;">
            <h2 style='color:#3ab54a; font-family:Inter,sans-serif;'>Loading module...</h2>
            <div style='color:#64748b; font-size:14px; margin-top:10px;'>Initializing components and clearing resources</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.rerun()

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

elif active_module == "Search Module":
    from modules.tier1.search_module import SearchModule
    module = SearchModule()
    module.run()
    logger.info(f"User {user['username']} -> Search Module")

elif active_module == "Market Breadth Dashboard":
    from modules.tier1.market_breadth import MarketBreadthModule
    module = MarketBreadthModule()
    module.run()
    logger.info(f"Market Breadth Dashboard")

elif active_module == "ES Pivot Analysis":
    from modules.tier1.es_pivot_analysis import ESPivotAnalysisModule
    module = ESPivotAnalysisModule()
    module.run()
    logger.info(f"ES Pivot Analysis")

elif active_module == "Sector Rotation Monitor":
    from modules.tier1.sector_rotation import SectorRotationModule
    module = SectorRotationModule()
    module.run()
    logger.info(f"Sector Rotation Monitor")

elif active_module == "Calendar Strategy Matrix":
    from modules.tier1.calendar_rotation import CalendarRotationModule
    module = CalendarRotationModule()
    module.run()
    logger.info(f"Calendar Strategy Matrix")

elif active_module == "Iron Condor Analyzer":
    from modules.tier1.iron_condor import IronCondorModule
    module = IronCondorModule()
    module.run()
    logger.info(f"Iron Condor Analyzer")

elif active_module == "Multi-Timeframe Money Flow":
    from modules.tier2.money_flow import MoneyFlowModule
    module = MoneyFlowModule()
    module.run()
    logger.info(f"User {user['username']}  Money Flow")

elif active_module == "Portfolio Module":
    from modules.tier2.portfolio_module import PortfolioModule
    module = PortfolioModule()
    module.run()
    logger.info(f"User {user['username']} -> Portfolio Module")

elif active_module == "Market Structure Heatmap":
    from modules.tier2.market_structure import MarketStructureModule
    module = MarketStructureModule()
    module.run()
    logger.info(f"User {user['username']}  Market Structure")

elif active_module == "Correlation Matrix":
    from modules.tier2.correlation_matrix import CorrelationMatrixModule
    module = CorrelationMatrixModule()
    module.run()
    logger.info(f"Correlation Matrix")

elif active_module == "Earnings Calendar":
    from modules.tier2.earnings_calendar import EarningsCalendarModule
    module = EarningsCalendarModule()
    module.run()
    logger.info(f"Earnings Calendar")

elif active_module == "Equity Income Statement":
    from modules.tier2.equity_income_statement import EquityIncomeStatementModule
    module = EquityIncomeStatementModule()
    module.run()
    logger.info(f"Equity Income Statement")

elif active_module == "Equity / Index Seasonality":
    from modules.tier2.seasonality_analysis import SeasonalityAnalysisModule
    module = SeasonalityAnalysisModule()
    module.run()
    logger.info(f"Equity / Index Seasonality")

elif active_module == "Stock Sentiment Analysis":
    from modules.tier2.stock_sentiment import StockSentimentModule
    module = StockSentimentModule()
    module.run()
    logger.info(f"Stock Sentiment Analysis")

elif active_module == "Social Stock Stories":
    from modules.tier2.social_stock_stories import SocialStockStoriesModule
    module = SocialStockStoriesModule()
    module.run()
    logger.info("Social Stock Stories")

elif active_module == "Universe Intelligence System":
    try:
        from modules.tier2.universe_intelligence import UniverseIntelligenceModule
        module = UniverseIntelligenceModule()
        module.run()
        logger.info("Universe Intelligence System")
    except Exception as e:
        import traceback
        st.error(f"Failed to load Universe Intelligence System: {e}")
        st.code(traceback.format_exc())

elif active_module == "Calendar Opportunity Scoring Engine":
    try:
        from modules.calendar_scoring.dashboard import CalendarOpportunityScoringModule
        module = CalendarOpportunityScoringModule()
        module.run()
        logger.info("Calendar Opportunity Scoring Engine")
    except Exception as e:
        import traceback
        st.error(f"Failed to load Calendar Opportunity Scoring Engine: {e}")
        st.code(traceback.format_exc())

elif active_module == "Price Action Story Engine":
    try:
        from modules.tier2.price_action_story import PriceActionStoryModule
        module = PriceActionStoryModule()
        module.run()
        logger.info("Price Action Story Engine")
    except Exception as e:
        import traceback
        st.error(f"Failed to load Price Action Story Engine: {e}")
        st.code(traceback.format_exc())

elif active_module == "Regime Intelligence Dashboard":
    try:
        from modules.tier2.markov_regime_engine import MarkovRegimeEngineModule
        module = MarkovRegimeEngineModule()
        module.run()
        logger.info("Regime Intelligence Dashboard")
    except Exception as e:
        import traceback
        st.error(f"Failed to load Regime Intelligence Dashboard: {e}")
        st.code(traceback.format_exc())

elif "Bradley Siderograph" in active_module:
    from modules.tier3.bradley_siderograph import BradleySiderographModule
    module = BradleySiderographModule()
    module.run()
    logger.info(f"Bradley Siderograph")

elif "Elliott Wave Analysis" in active_module:
    from modules.tier3.elliott_wave_analysis import ElliottWaveAnalysisModule
    module = ElliottWaveAnalysisModule()
    module.run()
    logger.info(f"Elliott Wave Analysis")

elif active_module == "Volatility Strategy Engine":
    from modules.tier4.volatility_engine import VolatilityEngineModule
    module = VolatilityEngineModule()
    module.run()
    logger.info("Volatility Strategy Engine")

elif active_module == "Gamma Flip Line Module":
    from modules.tier4.gamma_flip.gamma_dashboard import GammaFlipLineModule
    module = GammaFlipLineModule()
    module.run()
    logger.info("Gamma Flip Line Module")

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

    macro_module_tabs = [
        {
            "label": "Live Trading",
            "items": [
                {"label": "Search Module", "module": "Search Module", "tier": 1, "key": "macro_search_module"},
                {"label": "Market Breadth Dashboard", "module": "Market Breadth Dashboard", "tier": 1, "key": "macro_market_breadth"},
                {"label": "Sector Rotation Monitor", "module": "Sector Rotation Monitor", "tier": 1, "key": "macro_sector_rotation"},
                {"label": "Calendar Strategy Matrix", "module": "Calendar Strategy Matrix", "tier": 1, "key": "macro_calendar_strategy"},
                {"label": "Iron Condor Analyzer", "module": "Iron Condor Analyzer", "tier": 1, "key": "macro_iron_condor"},
                {"label": "ES Pivot Analysis", "module": "ES Pivot Analysis", "tier": 1, "key": "macro_es_pivot"},
            ],
        },
        {
            "label": "Analysis & Intelligence",
            "items": [
                {"label": "Universe Intelligence System", "module": "Universe Intelligence System", "tier": 2, "key": "macro_universe_intelligence"},
                {"label": "Portfolio Module", "module": "Portfolio Module", "tier": 2, "key": "macro_portfolio_module"},
                {"label": "Multi-Timeframe Money Flow", "module": "Multi-Timeframe Money Flow", "tier": 2, "key": "macro_money_flow"},
                {"label": "Market Structure Heatmap", "module": "Market Structure Heatmap", "tier": 2, "key": "macro_market_structure"},
                {"label": "Correlation Matrix", "module": "Correlation Matrix", "tier": 2, "key": "macro_correlation"},
                {"label": "Earnings Calendar", "module": "Earnings Calendar", "tier": 2, "key": "macro_earnings"},
                {"label": "Equity Income Statement", "module": "Equity Income Statement", "tier": 2, "key": "macro_income_statement"},
                {"label": "Equity / Index Seasonality", "module": "Equity / Index Seasonality", "tier": 2, "key": "macro_seasonality"},
                {"label": "Stock Sentiment Analysis", "module": "Stock Sentiment Analysis", "tier": 2, "key": "macro_sentiment"},
                {"label": "Social Stock Stories", "module": "Social Stock Stories", "tier": 2, "key": "macro_social_stories"},
                {"label": "Calendar Opportunity Scoring Engine", "module": "Calendar Opportunity Scoring Engine", "tier": 2, "key": "macro_calendar_scoring"},
                {"label": "Price Action Story Engine", "module": "Price Action Story Engine", "tier": 2, "key": "macro_price_action_story"},
                {"label": "Regime Intelligence Dashboard", "module": "Regime Intelligence Dashboard", "tier": 2, "key": "macro_regime_intelligence"},
            ],
        },
        {
            "label": "Forecasting",
            "items": [
                {"label": "Bradley Siderograph", "module": "Bradley Siderograph", "tier": 3, "key": "macro_bradley"},
                {"label": "Elliott Wave Analysis", "module": "Elliott Wave Analysis", "tier": 3, "key": "macro_elliott"},
            ],
        },
        {
            "label": "Volatility",
            "items": [
                {"label": "Volatility Strategy Engine", "module": "Volatility Strategy Engine", "tier": 4, "key": "macro_volatility_engine"},
                {"label": "Gamma Flip Line Module", "module": "Gamma Flip Line Module", "tier": 4, "key": "macro_gamma_flip"},
            ],
        },
    ]

    from modules.tier2.macro_intelligence import render_macro_dashboard
    render_macro_dashboard(show_download=True, module_tabs=macro_module_tabs, launch_callback=launch_module)

    st.divider()
    st.markdown(
        f"<p style='text-align:center;color:#334155;font-size:12px;'>Copyright (c) FazDane Analytics | Research & Trading Intelligence Platform {VERSION} · 2026 All Rights Reserved  Built on Streamlit</p>",
        unsafe_allow_html=True,
    )
