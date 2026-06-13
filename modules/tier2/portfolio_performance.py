"""
FazDane Analytics - Tier 2
Portfolio Performance
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from modules.base_module import FazDaneModule
from utils.tastytrade_provider import (
    TastytradeProviderError,
    fetch_market_data_by_type,
    load_config,
)
from utils.portfolio_performance_store import (
    get_database_status,
    get_latest_portfolio_details,
    get_latest_portfolio_positions,
    get_portfolio_history,
    get_recent_portfolio_snapshots,
    parse_schwab_position_details_csv,
    parse_schwab_positions_csv,
    save_portfolio_snapshot,
    summarize_positions,
    clean_ticker_for_lookup,
    parse_uploaded_files,
    get_broker_dot,
    format_ticker_for_display,
    classify_option_strategy,
    save_portfolio_log,
    delete_portfolio_log,
    get_portfolio_logs,
    get_portfolio_log_images,
)
from utils.universe_manager import update_fazdane_portfolio_universe


BRAND = {
    "bg": "#0d1b2e",
    "panel": "#152847",
    "grid": "#1e3a5f",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "green": "#3ab54a",
    "red": "#ef4444",
    "blue": "#93c5fd",
    "yellow": "#facc15",
    "purple": "#a78bfa",
}


def fmt_money(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def fmt_num(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):,.2f}"


def style_figure(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=BRAND["bg"],
        plot_bgcolor=BRAND["panel"],
        font=dict(color=BRAND["text"], family="Inter"),
        margin=dict(l=24, r=24, t=56, b=64),
        legend=dict(
            bgcolor="rgba(21,40,71,0.82)",
            bordercolor=BRAND["grid"],
            borderwidth=1,
            font=dict(color=BRAND["text"]),
        ),
        xaxis=dict(
            gridcolor=BRAND["grid"],
            zerolinecolor=BRAND["muted"],
            tickfont=dict(color=BRAND["text"]),
            automargin=True,
        ),
        yaxis=dict(
            gridcolor=BRAND["grid"],
            zerolinecolor=BRAND["muted"],
            tickfont=dict(color=BRAND["text"]),
            automargin=True,
        ),
    )
    return fig


def metric_delta(value: float, suffix: str = "") -> str:
    return f"{value:+,.2f}{suffix}"


@st.cache_data(ttl=180, show_spinner=False)
def fetch_daily_net_change(symbols: tuple[str, ...], source_preference: str) -> pd.DataFrame:
    """Fetch current daily price change for portfolio tickers."""
    clean_symbols = tuple(dict.fromkeys(clean_ticker_for_lookup(symbol) for symbol in symbols if str(symbol).strip()))
    if not clean_symbols:
        return pd.DataFrame()

    frames = []
    source_label = source_preference.lower()
    use_tasty = source_label in {"tastytrade first", "tastytrade only"}
    use_yahoo = source_label in {"tastytrade first", "yfinance only"}

    if use_tasty:
        tasty_df = _fetch_tastytrade_daily_change(clean_symbols)
        if not tasty_df.empty:
            frames.append(tasty_df)
        if source_label == "tastytrade only":
            return _dedupe_daily_change(frames)

    existing = set(pd.concat(frames)["ticker"]) if frames else set()
    missing_symbols = tuple(symbol for symbol in clean_symbols if symbol not in existing)
    if use_yahoo and missing_symbols:
        yahoo_df = _fetch_yfinance_daily_change(missing_symbols)
        if not yahoo_df.empty:
            frames.append(yahoo_df)

    return _dedupe_daily_change(frames)


def _fetch_tastytrade_daily_change(symbols: tuple[str, ...]) -> pd.DataFrame:
    try:
        config = load_config()
        if not config.is_configured:
            return pd.DataFrame()
        quotes = fetch_market_data_by_type(equities=symbols, config=config)
    except (TastytradeProviderError, Exception):
        return pd.DataFrame()

    if quotes.empty:
        return pd.DataFrame()

    rows = []
    for _, quote in quotes.iterrows():
        ticker = str(quote.get("market_symbol") or "").upper()
        last = _first_numeric(quote, ["last_price", "mark", "ask", "bid"])
        previous = _first_numeric(quote, ["prev_close", "close"])
        net_change = _first_numeric(quote, ["net_change"])
        pct_change = _first_numeric(quote, ["percent_change"])
        if previous is None and last is not None and net_change is not None:
            previous = float(last) - float(net_change)
        if pct_change is not None and abs(pct_change) <= 1:
            pct_change = pct_change * 100
        if not ticker or last is None or previous is None or previous == 0:
            continue
        if pct_change is None:
            pct_change = (last / previous - 1) * 100
        price_change = net_change if net_change is not None else last - previous
        rows.append(
            {
                "ticker": ticker,
                "last_price": float(last),
                "previous_close": float(previous),
                "price_change": float(price_change),
                "pct_change": float(pct_change),
                "market_source": "Tastytrade",
            }
        )
    return pd.DataFrame(rows)


def _fetch_yfinance_daily_change(symbols: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            last = _first_attr_or_key(info, ["last_price", "lastPrice", "regular_market_price", "regularMarketPrice"])
            previous = _first_attr_or_key(
                info,
                ["previous_close", "previousClose", "regular_market_previous_close", "regularMarketPreviousClose"],
            )

            if not last or not previous:
                hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
                closes = hist["Close"].dropna() if not hist.empty and "Close" in hist.columns else pd.Series(dtype=float)
                if len(closes) >= 2:
                    last = float(closes.iloc[-1])
                    previous = float(closes.iloc[-2])

            if last is None or previous is None or float(previous) == 0:
                continue

            last_float = float(last)
            previous_float = float(previous)
            rows.append(
                {
                    "ticker": symbol,
                    "last_price": last_float,
                    "previous_close": previous_float,
                    "price_change": last_float - previous_float,
                    "pct_change": (last_float / previous_float - 1) * 100,
                    "market_source": "yfinance",
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def _dedupe_daily_change(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return combined
    return combined.drop_duplicates("ticker", keep="first")


def _first_numeric(row: pd.Series, columns: list[str]) -> float | None:
    for column in columns:
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            return float(value)
    return None


def _first_attr_or_key(value: object, names: list[str]) -> float | None:
    for name in names:
        try:
            found = getattr(value, name)
        except Exception:
            found = None
        if found is None:
            try:
                found = value.get(name)  # type: ignore[attr-defined]
            except Exception:
                found = None
        try:
            if found is not None:
                return float(found)
        except (TypeError, ValueError):
            continue
    return None


class PortfolioPerformanceModule(FazDaneModule):
    MODULE_NAME = "Portfolio Performance"
    MODULE_ICON = "PF"
    MODULE_DESCRIPTION = "Schwab position snapshots, risk exposure, and saved daily portfolio progress"
    TIER = 2
    SOURCE_NOTEBOOK = "FazDane Analytics Portfolio Dashboard Generator"
    CACHE_TTL = 300
    REQUIRES_LIVE_DATA = False
    DATA_SOURCES = ["Schwab Position Statement CSV", "Local SQLite"]

    def render_sidebar(self):
        st.markdown("**Snapshot Source**")
        self.uploaded_files = st.file_uploader(
            "Upload Broker CSVs (Schwab / Tastytrade)",
            type=["csv"],
            accept_multiple_files=True,
            key="pp_csv_uploads",
        )
        self.load_latest_saved = st.checkbox(
            "Use latest saved snapshot when no file is uploaded",
            value=True,
            key="pp_use_latest",
        )
        self.auto_save = st.checkbox(
            "Save parsed upload to daily database",
            value=True,
            key="pp_auto_save",
        )

        if st.button("🔄 Sync 'FazDane Portfolio'", key="pp_sync_universe_btn", width="stretch"):
            pos, det, meta, label = self._load_active_snapshot()
            if not pos.empty:
                raw_tickers = pos["ticker"].apply(clean_ticker_for_lookup).unique().tolist()
                update_fazdane_portfolio_universe(raw_tickers)
                st.success(f"Updated 'FazDane Portfolio' universe with {len(raw_tickers)} tickers!")
            else:
                st.warning("No active portfolio positions loaded to sync.")

        st.markdown("**View Settings**")
        self.top_n = st.slider("Rows per leaderboard", min_value=3, max_value=15, value=7, key="pp_top_n")
        self.history_days = st.selectbox("History Window", [30, 60, 90, 180, 365], index=2, key="pp_hist_days")
        self.delta_caution = st.number_input(
            "High Delta Caution Level",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
            key="pp_delta_caution",
        )
        self.market_data_source = st.selectbox(
            "Daily Net Change Source",
            ["yfinance Only", "Tastytrade First", "Tastytrade Only"],
            index=0,
            key="pp_daily_change_source",
        )

        if st.button("Refresh Current Pricing", width="stretch", type="primary", key="pp_refresh_prices"):
            fetch_daily_net_change.clear()
            st.rerun()

        if st.button("Refresh Portfolio View", width="stretch", key="pp_refresh"):
            st.session_state.pop("pp_last_saved_hash", None)
            fetch_daily_net_change.clear()
            st.rerun()

        self._render_database_status()

    def render_main(self):
        self.render_section_header(
            "Portfolio Performance",
            "Upload Schwab positions, save daily snapshots, and monitor P/L, theta, and delta concentration.",
        )
        st.markdown("<div style='font-size:13px;color:#888;margin-bottom:12px;'>Source: 🔴 Tastytrade | 🔵 Schwab</div>", unsafe_allow_html=True)

        positions, details, metadata, source_label = self._load_active_snapshot()
        
        # Auto-update the "FazDane Portfolio" universe on new statement upload
        if self.uploaded_files and not positions.empty:
            file_hash = metadata.get("file_sha256")
            if file_hash and st.session_state.get("pp_last_synced_universe_hash") != file_hash:
                raw_tickers = positions["ticker"].apply(clean_ticker_for_lookup).unique().tolist()
                update_fazdane_portfolio_universe(raw_tickers)
                st.session_state["pp_last_synced_universe_hash"] = file_hash

        if positions.empty:
            self._render_welcome()
            self._render_history()
            return

        saved_info = None
        if self.uploaded_files and self.auto_save:
            saved_info = self._save_uploaded_snapshot_once(positions, details, metadata)

        totals = summarize_positions(positions)
        self._render_source_banner(metadata, source_label, saved_info)
        self._render_metrics(totals, positions)

        tab_overview, tab_risk, tab_history, tab_value_tracker, tab_data, tab_details, tab_logs = st.tabs(
            ["Overview", "Risk Map", "Daily History", "Value & Delta Tracker", "Position Data", "Raw Details", "Portfolio Logs"]
        )
        with tab_overview:
            self._render_overview(positions, totals)
        with tab_risk:
            self._render_risk_map(positions)
        with tab_history:
            self._render_history()
        with tab_value_tracker:
            self._render_value_tracker()
        with tab_data:
            self._render_data_tab(positions, metadata)
        with tab_details:
            self._render_details_tab(details, metadata)
        with tab_logs:
            self._render_logs_tab(metadata)

    def _render_logs_tab(self, metadata: dict):
        st.markdown("### Portfolio Daily Logs")
        
        import base64
        from st_img_pastebutton import paste
        
        # Load logs from DB
        logs_df = get_portfolio_logs()
           # Handle edit state loading
        edit_id = st.session_state.get("pp_edit_log_id")
        log_to_edit = None
        if edit_id and not logs_df.empty:
            matching = logs_df[logs_df["log_id"] == edit_id]
            if not matching.empty:
                log_to_edit = matching.iloc[0]
                
        # Initialize the current images list in session state
        if "pp_current_images" not in st.session_state or st.session_state.get("pp_loaded_edit_id") != edit_id:
            st.session_state["pp_loaded_edit_id"] = edit_id
            if edit_id and not logs_df.empty and log_to_edit is not None:
                images = []
                if "image_data" in log_to_edit and log_to_edit["image_data"] is not None:
                    if isinstance(log_to_edit["image_data"], bytes) and len(log_to_edit["image_data"]) > 0:
                        images.append(log_to_edit["image_data"])
                try:
                    db_images = get_portfolio_log_images().get(edit_id, [])
                    images.extend(db_images)
                except Exception:
                    pass
                st.session_state["pp_current_images"] = images
            else:
                st.session_state["pp_current_images"] = []
        
        # 1. Input Form
        form_title = "✍️ Edit Portfolio Log Entry" if log_to_edit is not None else "✍️ Add Daily Portfolio Log Entry"
        with st.container(border=True):
            st.markdown(f"**{form_title}**")
            
            # Default values
            if log_to_edit is not None:
                default_date = datetime.strptime(log_to_edit["log_date"], "%Y-%m-%d")
                default_category = log_to_edit["category"]
                default_content = log_to_edit["content"]
                default_snippet = log_to_edit["snippet"] if "snippet" in log_to_edit and log_to_edit["snippet"] is not None else ""
            else:
                active_date_str = metadata.get("snapshot_date")
                if active_date_str:
                    try:
                        default_date = datetime.strptime(active_date_str, "%Y-%m-%d")
                    except Exception:
                        default_date = datetime.today()
                else:
                    default_date = datetime.today()
                default_category = "What Happened"
                default_content = ""
                default_snippet = ""
                
            col_date, col_cat = st.columns(2)
            with col_date:
                log_date = st.date_input("Target Log Date", value=default_date, key="pp_log_date_input")
            with col_cat:
                categories = ["What Happened", "What Can Improve", "Market Context", "Action Items", "General Notes"]
                cat_idx = categories.index(default_category) if default_category in categories else 0
                category = st.selectbox("Log Category", options=categories, index=cat_idx, key="pp_log_category_input")
                
            content = st.text_area("What's on your mind? (Supports Markdown)", value=default_content, height=120, key="pp_log_content_input")
            snippet = st.text_area("Code/Data Snippet (Optional, e.g. terminal output, JSON, trade detail)", value=default_snippet, height=100, key="pp_log_snippet_input")
            
            # Use dynamic keys for uploader/paste to allow resetting
            paste_counter = st.session_state.get("pp_paste_counter", 0)
            uploaded_image = st.file_uploader("Upload Screenshot File (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], key=f"pp_log_image_upload_{paste_counter}")
            
            st.markdown("<p style='font-size:13px;color:#94a3b8;margin-bottom:4px;'>Or paste screenshot directly from clipboard:</p>", unsafe_allow_html=True)
            pasted_image_data = paste(label="📋 Click to Paste Clipboard Image", key=f"pp_log_image_paste_{paste_counter}")
            
            # Check if new image uploaded
            if uploaded_image is not None:
                image_bytes = uploaded_image.read()
                st.session_state["pp_current_images"].append(image_bytes)
                st.session_state["pp_paste_counter"] = paste_counter + 1
                st.rerun()
                
            # Check if new image pasted
            if pasted_image_data is not None:
                try:
                    header, encoded = pasted_image_data.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                    st.session_state["pp_current_images"].append(image_bytes)
                    st.session_state["pp_paste_counter"] = paste_counter + 1
                    st.rerun()
                except Exception:
                    pass
            
            # Display attached screenshots
            current_images = st.session_state.get("pp_current_images", [])
            if current_images:
                st.markdown("<p style='font-size:14px;font-weight:bold;margin-top:10px;margin-bottom:6px;'>Attached Screenshots:</p>", unsafe_allow_html=True)
                for idx, img_bytes in enumerate(current_images):
                    col_img, col_act = st.columns([5, 1])
                    with col_img:
                        st.image(img_bytes, caption=f"Screenshot #{idx + 1}", width=250)
                    with col_act:
                        st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)
                        if st.button("🗑️ Remove", key=f"pp_remove_img_{idx}"):
                            st.session_state["pp_current_images"].pop(idx)
                            st.rerun()
            
            sub_col1, sub_col2 = st.columns([1, 6])
            with sub_col1:
                submit_button = st.button("Save Log Entry", type="primary", key="pp_save_log_btn")
            with sub_col2:
                if log_to_edit is not None:
                    cancel_button = st.button("Cancel Edit", key="pp_cancel_edit_btn")
                    if cancel_button:
                        st.session_state.pop("pp_edit_log_id", None)
                        st.session_state.pop("pp_loaded_edit_id", None)
                        st.session_state.pop("pp_current_images", None)
                        st.session_state["pp_paste_counter"] = paste_counter + 1
                        st.rerun()
                        
            if submit_button:
                if not content.strip():
                    st.error("Log content cannot be empty.")
                else:
                    log_date_str = log_date.strftime("%Y-%m-%d")
                    run_id = metadata.get("run_id") if log_to_edit is None else log_to_edit["run_id"]
                    
                    save_portfolio_log(
                        log_date=log_date_str,
                        category=category,
                        content=content,
                        log_id=edit_id,
                        run_id=run_id,
                        image_data=None,
                        clear_image=False,
                        snippet=snippet if snippet.strip() else None,
                        images_list=current_images,
                    )
                    st.session_state.pop("pp_edit_log_id", None)
                    st.session_state.pop("pp_loaded_edit_id", None)
                    st.session_state.pop("pp_current_images", None)
                    st.session_state["pp_paste_counter"] = paste_counter + 1
                    st.success("Portfolio log saved successfully!")
                    st.rerun()

        # 2. Insights counters
        if not logs_df.empty:
            total_logs = len(logs_df)
            improvements = len(logs_df[logs_df["category"] == "What Can Improve"])
            happened = len(logs_df[logs_df["category"] == "What Happened"])
            
            try:
                current_month_str = datetime.today().strftime("%Y-%m")
                logs_this_month = len(logs_df[logs_df["log_date"].str.startswith(current_month_str)])
            except Exception:
                logs_this_month = 0
                
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Logs", f"{total_logs}")
            c2.metric("What Happened", f"{happened}")
            c3.metric("What Can Improve", f"{improvements}", delta="Opportunities" if improvements > 0 else None, delta_color="normal")
            c4.metric("Logs This Month", f"{logs_this_month}")
            
        st.markdown("---")
        
        # 3. Filtering & View
        st.markdown("### Log Timeline & Reporting")
        
        if logs_df.empty:
            st.info("No logs saved yet. Add a log entry above to populate your timeline!")
            return
            
        col_search, col_filter = st.columns([2, 1])
        with col_search:
            search_query = st.text_input("🔍 Search logs...", placeholder="Type to search...", key="pp_logs_search")
        with col_filter:
            cat_filter = st.multiselect("Filter by Category", options=["What Happened", "What Can Improve", "Market Context", "Action Items", "General Notes"], key="pp_logs_cat_filter")
            
        view_period = st.radio("Group/Report logs by:", ["Day", "Week", "Month", "Year"], horizontal=True, key="pp_logs_period")
        
        # Get all images for logs timeline
        all_images_dict = {}
        try:
            all_images_dict = get_portfolio_log_images()
        except Exception:
            pass
            
        # Apply search and category filters
        filtered_df = logs_df.copy()
        if search_query:
            filtered_df = filtered_df[
                filtered_df["content"].str.contains(search_query, case=False) |
                filtered_df["category"].str.contains(search_query, case=False)
            ]
        if cat_filter:
            filtered_df = filtered_df[filtered_df["category"].isin(cat_filter)]
            
        if filtered_df.empty:
            st.warning("No logs match the current search or filter criteria.")
            return

        filtered_df["log_date_dt"] = pd.to_datetime(filtered_df["log_date"])
        
        cat_styles = {
            "What Happened": ("#3ab54a", "rgba(58, 181, 74, 0.12)"),
            "What Can Improve": ("#ef4444", "rgba(239, 68, 68, 0.12)"),
            "Market Context": ("#facc15", "rgba(250, 204, 21, 0.12)"),
            "Action Items": ("#a78bfa", "rgba(167, 139, 250, 0.12)"),
            "General Notes": ("#93c5fd", "rgba(147, 197, 253, 0.12)"),
        }
        
        if view_period == "Day":
            grouped = filtered_df.groupby("log_date", sort=False)
            for date_str, group in grouped:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                st.markdown(f"#### 📅 {dt.strftime('%A, %B %d, %Y')}")
                for _, row in group.iterrows():
                    self._render_log_item_card(row, cat_styles, all_images_dict=all_images_dict)
                    
        elif view_period == "Week":
            filtered_df["week_start"] = filtered_df["log_date_dt"].dt.to_period("W").dt.start_time
            grouped = filtered_df.groupby("week_start", sort=False)
            for week_start, group in grouped:
                week_end = week_start + pd.Timedelta(days=6)
                st.markdown(f"#### 🗓️ Week of {week_start.strftime('%B %d, %Y')} to {week_end.strftime('%B %d, %Y')}")
                for _, row in group.iterrows():
                    self._render_log_item_card(row, cat_styles, show_date=True, all_images_dict=all_images_dict)
                    
        elif view_period == "Month":
            filtered_df["year_month"] = filtered_df["log_date_dt"].dt.to_period("M")
            grouped = filtered_df.groupby("year_month", sort=False)
            for month_period, group in grouped:
                st.markdown(f"#### 📅 {month_period.start_time.strftime('%B %Y')}")
                for _, row in group.iterrows():
                    self._render_log_item_card(row, cat_styles, show_date=True, all_images_dict=all_images_dict)
                    
        elif view_period == "Year":
            filtered_df["year"] = filtered_df["log_date_dt"].dt.to_period("Y")
            grouped = filtered_df.groupby("year", sort=False)
            for year_period, group in grouped:
                st.markdown(f"#### 🗓️ Year {year_period.start_time.strftime('%Y')}")
                for _, row in group.iterrows():
                    self._render_log_item_card(row, cat_styles, show_date=True, all_images_dict=all_images_dict)

    def _render_log_item_card(self, row: pd.Series, cat_styles: dict, show_date: bool = False, all_images_dict: dict | None = None):
        cat = row["category"]
        color, bg = cat_styles.get(cat, ("#e2e8f0", "rgba(226, 232, 240, 0.15)"))
        
        date_badge = f"<span style='background-color:#1e3a5f;color:#e2e8f0;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;margin-right:8px;'>{row['log_date']}</span>" if show_date else ""
        run_badge = f"<span style='background-color:#1e3a5f;color:#94a3b8;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;margin-right:8px;'>Run: {row['run_id']}</span>" if row["run_id"] else ""
        
        st.markdown(
            f"""<div style="background-color:#152847; border:1px solid #1e3a5f; border-left:4px solid {color}; border-radius:8px; padding:12px 16px; margin-bottom:10px;">
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:6px;">
<div>
{date_badge}
<span style="background-color:{bg}; color:{color}; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; border:1px solid {color};">
{cat.upper()}
</span>
{run_badge}
</div>
<span style="color:#94a3b8; font-size:11px;">Updated: {row['updated_at'][:16].replace('T', ' ')}</span>
</div>
<div style="color:#e2e8f0; font-size:13.5px; line-height:1.6; white-space:pre-wrap; margin-bottom:10px;">{row['content']}</div>
</div>""",
            unsafe_allow_html=True,
        )
        
        # Render legacy + multiple images
        log_images = []
        if "image_data" in row and row["image_data"] is not None:
            if isinstance(row["image_data"], bytes) and len(row["image_data"]) > 0:
                log_images.append(row["image_data"])
                
        if all_images_dict is not None:
            db_images = all_images_dict.get(row["log_id"], [])
        else:
            try:
                db_images = get_portfolio_log_images().get(row["log_id"], [])
            except Exception:
                db_images = []
        log_images.extend(db_images)
        
        for img in log_images:
            st.image(img, width="stretch")
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            
        # Check if there is snippet data and render it
        if "snippet" in row and row["snippet"] is not None and str(row["snippet"]).strip() != "":
            st.code(row["snippet"])
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        
        col_btn1, col_btn2, _ = st.columns([1, 1, 15])
        with col_btn1:
            if st.button("Edit", key=f"pp_edit_btn_{row['log_id']}"):
                st.session_state["pp_edit_log_id"] = row["log_id"]
                st.rerun()
        with col_btn2:
            if st.button("Delete", key=f"pp_delete_btn_{row['log_id']}"):
                delete_portfolio_log(row["log_id"])
                st.success("Log entry deleted.")
                st.rerun()


    def _load_active_snapshot(self) -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
        positions, details, metadata, label = pd.DataFrame(), pd.DataFrame(), {}, "No snapshot"
        if self.uploaded_files:
            file_tuples = [(f.getvalue(), f.name) for f in self.uploaded_files]
            positions, details, metadata = parse_uploaded_files(file_tuples)
            label = "Uploaded broker CSV(s)"
        elif self.load_latest_saved:
            latest_positions, latest_metadata = get_latest_portfolio_positions()
            if latest_metadata:
                latest_details, _ = get_latest_portfolio_details()
                positions, details, metadata = latest_positions, latest_details, latest_metadata
                label = "Latest saved snapshot"

        if not positions.empty and "ticker" in positions.columns:
            positions["ticker"] = positions["ticker"].apply(format_ticker_for_display)
        if not details.empty and "underlying" in details.columns:
            details["underlying"] = details["underlying"].apply(format_ticker_for_display)

        # On-the-fly strategy classification for loaded Tastytrade database positions
        if not positions.empty and not details.empty:
            tasty_mask = positions["ticker"].str.contains("🔴") & positions["account_group"].astype(str).str.contains("5WT|TASTY", case=False)
            if tasty_mask.any():
                option_legs = details[details["row_type"] == "option_leg"]
                if not option_legs.empty:
                    strategies = {}
                    for und, group in option_legs.groupby("underlying"):
                        strategies[und] = classify_option_strategy(group)
                    positions.loc[tasty_mask, "account_group"] = positions.loc[tasty_mask, "ticker"].map(strategies).fillna("Position Basket")

        return positions, details, metadata, label

    def _save_uploaded_snapshot_once(self, positions: pd.DataFrame, details: pd.DataFrame, metadata: dict) -> dict | None:
        file_hash = metadata.get("file_sha256")
        last_hash = st.session_state.get("pp_last_saved_hash")
        if file_hash and last_hash == file_hash:
            return st.session_state.get("pp_last_save_info")

        try:
            saved = save_portfolio_snapshot(positions, metadata, details=details)
            st.session_state["pp_last_saved_hash"] = file_hash
            st.session_state["pp_last_save_info"] = saved
            return saved
        except Exception as exc:
            st.warning(f"Snapshot parsed, but database save failed: {exc}")
            return None

    def _render_source_banner(self, metadata: dict, source_label: str, saved_info: dict | None):
        source_file = metadata.get("source_file", "Unknown file")
        snapshot_ts = metadata.get("snapshot_ts", "")
        saved_text = ""
        if saved_info:
            saved_text = f"<div style='color:{BRAND['green']};font-size:12px;margin-top:6px;'>Saved run {saved_info['run_id']}</div>"

        st.markdown(
            f"""
            <div style="
                background:linear-gradient(135deg, rgba(26,58,143,0.28), rgba(58,181,74,0.08));
                border:1px solid {BRAND['grid']};
                border-left:4px solid {BRAND['green']};
                border-radius:10px;
                padding:14px 18px;
                margin:8px 0 16px 0;
            ">
                <div style="color:{BRAND['green']};font-size:15px;font-weight:700;">{source_label}</div>
                <div style="color:{BRAND['text']};font-size:13px;margin-top:4px;">{source_file}</div>
                <div style="color:{BRAND['muted']};font-size:12px;margin-top:3px;">Snapshot time: {snapshot_ts}</div>
                {saved_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_metrics(self, totals: dict[str, float], positions: pd.DataFrame):
        winners = int((positions["pl_day"] > 0).sum())
        losers = int((positions["pl_day"] < 0).sum())
        win_rate = winners / max(winners + losers, 1) * 100

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("P/L Open", fmt_money(totals["total_pl_open"]), fmt_money(totals["positive_open"]))
        m2.metric("P/L Day", fmt_money(totals["total_pl_day"]), f"{win_rate:.0f}% day win rate")
        m3.metric("Theta", fmt_num(totals["total_theta"]), metric_delta(totals["total_theta"]))
        m4.metric("Delta", fmt_num(totals["total_delta"]), "Net exposure")
        m5.metric("Tickers", f"{len(positions):,}", f"{winners} up / {losers} down")

    def _render_overview(self, positions: pd.DataFrame, totals: dict[str, float]):
        left, right = st.columns([1.35, 1.0])
        with left:
            st.markdown("### P/L by Ticker")
            value_col = st.radio(
                "Performance Field",
                ["pl_open", "pl_day"],
                horizontal=True,
                format_func=lambda value: "P/L Open" if value == "pl_open" else "P/L Day",
                key="pp_perf_field",
                label_visibility="collapsed",
            )
            chart_df = positions.sort_values(value_col, ascending=True).copy()
            chart_df["color"] = chart_df[value_col].map(lambda value: "Gain" if value >= 0 else "Loss")
            fig = px.bar(
                chart_df,
                x=value_col,
                y="ticker",
                orientation="h",
                color="color",
                color_discrete_map={"Gain": BRAND["green"], "Loss": BRAND["red"]},
                hover_data=["theta", "delta", "market_value", "account_group"],
                labels={value_col: "Dollars", "ticker": "Ticker"},
            )
            fig.add_vline(x=0, line_width=1, line_color=BRAND["muted"])
            fig.update_traces(texttemplate="%{x:,.0f}", textposition="outside", cliponaxis=False)
            style_figure(fig, height=max(380, 28 * len(chart_df) + 80))
            fig.update_layout(
                yaxis=dict(
                    anchor="free",
                    position=0.0,
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    title=None,
                ),
                margin=dict(l=24, r=24, t=56, b=85),
            )
            st.plotly_chart(fig, width="stretch", theme=None)
            self._render_daily_net_change_chart(positions)

        with right:
            self._render_insights(positions, totals)

        st.markdown("### Exposure Leaders")
        c1, c2 = st.columns(2)
        with c1:
            theta_df = positions.sort_values("theta", ascending=False).head(self.top_n)
            fig = px.bar(
                theta_df,
                x="ticker",
                y="theta",
                color="theta",
                color_continuous_scale=[[0, BRAND["red"]], [0.5, BRAND["yellow"]], [1, BRAND["green"]]],
                labels={"theta": "Theta", "ticker": "Ticker"},
                title="Theta Generators",
            )
            style_figure(fig, height=330)
            st.plotly_chart(fig, width="stretch", theme=None)

        with c2:
            delta_df = positions.reindex(positions["delta"].abs().sort_values(ascending=False).index).head(self.top_n)
            fig = px.bar(
                delta_df,
                x="ticker",
                y="delta",
                color="delta",
                color_continuous_scale=[[0, BRAND["red"]], [0.5, BRAND["yellow"]], [1, BRAND["blue"]]],
                labels={"delta": "Delta", "ticker": "Ticker"},
                title="Largest Delta Exposures",
            )
            fig.add_hline(y=self.delta_caution, line_dash="dash", line_color=BRAND["yellow"])
            style_figure(fig, height=330)
            st.plotly_chart(fig, width="stretch", theme=None)

    def _render_daily_net_change_chart(self, positions: pd.DataFrame):
        st.markdown("### Daily Net Change by Portfolio Ticker")
        symbols = tuple(positions["ticker"].dropna().astype(str).str.upper().unique())
        with st.spinner("Fetching daily market change for portfolio tickers..."):
            market = fetch_daily_net_change(symbols, self.market_data_source)

        if market.empty:
            st.info("Daily market change is unavailable from the selected source. Try Tastytrade First or yfinance Only.")
            return

        quantity = positions[["ticker", "quantity", "market_value"]].copy()
        quantity["ticker"] = quantity["ticker"].astype(str).str.upper()
        quantity["base_ticker"] = quantity["ticker"].apply(clean_ticker_for_lookup)
        daily = market.merge(quantity, left_on="ticker", right_on="base_ticker", how="left", suffixes=("_market", ""))
        daily["ticker"] = daily["ticker"].fillna(daily["ticker_market"])
        daily["quantity"] = pd.to_numeric(daily["quantity"], errors="coerce").fillna(0)
        daily["market_value"] = pd.to_numeric(daily["market_value"], errors="coerce").fillna(0)
        daily["estimated_dollar_change"] = daily["quantity"] * daily["price_change"]
        fallback = daily["market_value"] * daily["pct_change"] / 100
        daily["estimated_dollar_change"] = daily["estimated_dollar_change"].where(
            daily["estimated_dollar_change"].abs() > 0,
            fallback,
        )
        daily = daily.sort_values("pct_change", ascending=True)
        daily["direction"] = daily["pct_change"].map(lambda value: "Up" if value >= 0 else "Down")
        daily["pct_label"] = daily["pct_change"].map(lambda value: f"{value:+.2f}%")

        fig = px.bar(
            daily,
            x="pct_change",
            y="ticker",
            text="pct_label",
            orientation="h",
            color="direction",
            color_discrete_map={"Up": BRAND["green"], "Down": BRAND["red"]},
            hover_data={
                "last_price": ":$.2f",
                "previous_close": ":$.2f",
                "price_change": ":$.2f",
                "pct_change": ":.2f",
                "pct_label": False,
                "estimated_dollar_change": ":$.2f",
                "market_source": True,
                "direction": False,
            },
            labels={"pct_change": "Daily Change (%)", "ticker": "Ticker"},
            title="Market-Sourced Daily Price Performance",
        )
        fig.add_vline(x=0, line_width=1, line_color=BRAND["muted"])
        style_figure(fig, height=max(360, 26 * len(daily) + 96))
        fig.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False)
        fig.update_xaxes(ticksuffix="%", tickformat=".2f", separatethousands=False)
        fig.update_layout(
            yaxis=dict(
                anchor="free",
                position=0.0,
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                title=None,
            ),
            margin=dict(l=24, r=24, t=56, b=85),
        )
        st.plotly_chart(fig, width="stretch", theme=None)

        total_estimate = daily["estimated_dollar_change"].sum()
        source_mix = ", ".join(sorted(daily["market_source"].dropna().unique()))
        st.caption(
            f"Estimated net dollar move from displayed quantities: {fmt_money(total_estimate)}. "
            f"Source: {source_mix}."
        )
        if st.button("Refresh Current Pricing", key="pp_inline_refresh_prices", width="stretch"):
            fetch_daily_net_change.clear()
            st.rerun()

    def _render_insights(self, positions: pd.DataFrame, totals: dict[str, float]):
        top_open = positions.sort_values("pl_open", ascending=False).head(self.top_n)
        open_drags = positions.sort_values("pl_open").head(self.top_n)
        day_leaders = positions.sort_values("pl_day", ascending=False).head(self.top_n)
        watch = self._build_watchlist(positions)

        st.markdown("### Insights")
        self._leaderboard("Top Open Winners", top_open, "pl_open", BRAND["green"], money=True)
        self._leaderboard("Largest Open Drags", open_drags, "pl_open", BRAND["red"], money=True)
        self._leaderboard("Day Leaders", day_leaders, "pl_day", BRAND["green"], money=True)

        watch_text = ", ".join(watch) if watch else "No concentrated caution names in this snapshot."
        st.markdown(
            f"""
            <div style="
                background:rgba(21,40,71,0.70);
                border:1px solid {BRAND['grid']};
                border-radius:8px;
                padding:14px 16px;
                margin-top:10px;
            ">
                <div style="color:{BRAND['yellow']};font-weight:700;font-size:13px;">Watchlist / Caution</div>
                <div style="color:{BRAND['muted']};font-size:13px;line-height:1.55;margin-top:6px;">
                    {watch_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _leaderboard(self, title: str, df: pd.DataFrame, column: str, color: str, money: bool = False):
        rows = []
        for _, row in df.iterrows():
            val = row[column]
            val_color = BRAND["green"] if val >= 0 else BRAND["red"]
            value = fmt_money(val) if money else f"{val:,.2f}"
            rows.append(
                f"<div style='display:flex;justify-content:space-between;gap:16px;margin:4px 0;'>"
                f"<span style='color:{BRAND['text']};font-weight:600;'>{row['ticker']}</span>"
                f"<span style='color:{val_color};font-weight:700;'>{value}</span>"
                f"</div>"
            )
        st.markdown(
            f"""
            <div style="
                background:rgba(21,40,71,0.55);
                border:1px solid {BRAND['grid']};
                border-radius:8px;
                padding:12px 14px;
                margin-bottom:10px;
            ">
                <div style="color:{color};font-weight:700;font-size:13px;margin-bottom:6px;">{title}</div>
                {''.join(rows)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_risk_map(self, positions: pd.DataFrame):
        st.markdown("### Delta vs Theta Risk Map")
        risk = positions.copy()
        risk["bubble_size"] = risk["pl_open"].abs().clip(lower=25)
        risk["quadrant"] = risk.apply(self._risk_quadrant, axis=1)

        fig = px.scatter(
            risk,
            x="delta",
            y="theta",
            size="bubble_size",
            color="quadrant",
            text="ticker",
            hover_data=["pl_open", "pl_day", "market_value", "account_group"],
            color_discrete_map={
                "Constructive": BRAND["green"],
                "Directional": BRAND["blue"],
                "Decay Drag": BRAND["yellow"],
                "Pressure": BRAND["red"],
            },
            labels={"delta": "Delta", "theta": "Theta"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color=BRAND["muted"])
        fig.add_vline(x=self.delta_caution, line_dash="dash", line_color=BRAND["yellow"])
        fig.update_traces(textposition="top center", marker=dict(opacity=0.78, line=dict(width=1, color="#e2e8f0")))
        style_figure(fig, height=520)
        st.plotly_chart(fig, width="stretch", theme=None)

        st.markdown("### Caution Candidates")
        caution = positions.loc[positions["ticker"].isin(self._build_watchlist(positions))].copy()
        if caution.empty:
            st.info("No names matched the current caution rules.")
        else:
            self._render_position_table(caution.sort_values(["delta", "theta"], ascending=[False, True]))

    def _render_history(self):
        st.markdown("### Saved Daily Progress")
        history = get_portfolio_history(days=self.history_days)
        recent = get_recent_portfolio_snapshots(limit=15)
        if history.empty:
            st.info("No saved portfolio snapshots yet. Upload a Schwab CSV and keep database save enabled to start the daily progress record.")
            return

        h1, h2, h3 = st.columns(3)
        latest = history.iloc[-1]
        previous = history.iloc[-2] if len(history) > 1 else None
        h1.metric(
            "Latest Open P/L",
            fmt_money(latest["total_pl_open"]),
            fmt_money(latest["total_pl_open"] - previous["total_pl_open"]) if previous is not None else None,
        )
        h2.metric(
            "Latest Day P/L",
            fmt_money(latest["total_pl_day"]),
            fmt_money(latest["total_pl_day"] - previous["total_pl_day"]) if previous is not None else None,
        )
        h3.metric(
            "Net Theta / Delta",
            f"{latest['total_theta']:,.1f} / {latest['total_delta']:,.1f}",
            "Saved snapshot totals",
        )

        chart = history.copy()
        chart["snapshot_ts"] = pd.to_datetime(chart["snapshot_ts"], errors="coerce")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart["snapshot_ts"],
            y=chart["total_pl_open"],
            mode="lines+markers",
            name="P/L Open",
            line=dict(color=BRAND["green"], width=3),
        ))
        fig.add_trace(go.Bar(
            x=chart["snapshot_ts"],
            y=chart["total_pl_day"],
            name="P/L Day",
            marker_color=chart["total_pl_day"].map(lambda value: BRAND["green"] if value >= 0 else BRAND["red"]),
            opacity=0.58,
            yaxis="y2",
        ))
        fig.update_layout(
            title="Portfolio Progress by Saved Snapshot",
            yaxis=dict(title="P/L Open", gridcolor=BRAND["grid"]),
            yaxis2=dict(title="P/L Day", overlaying="y", side="right", showgrid=False),
            barmode="overlay",
        )
        style_figure(fig, height=420)
        st.plotly_chart(fig, width="stretch", theme=None)

        if not recent.empty:
            st.markdown("### Recent Saved Snapshots")
            st.dataframe(
                recent,
                width="stretch",
                hide_index=True,
                height=460,
                column_config={
                    "snapshot_ts": st.column_config.TextColumn("Snapshot Time"),
                    "snapshot_date": st.column_config.TextColumn("Date"),
                    "source_file": st.column_config.TextColumn("Source"),
                    "row_count": st.column_config.NumberColumn("Tickers", format="%d"),
                    "total_pl_open": st.column_config.NumberColumn("P/L Open", format="$%.2f"),
                    "total_pl_day": st.column_config.NumberColumn("P/L Day", format="$%.2f"),
                    "total_theta": st.column_config.NumberColumn("Theta", format="%.2f"),
                    "total_delta": st.column_config.NumberColumn("Delta", format="%.2f"),
                },
            )

    def _render_value_tracker(self):
        st.markdown("### Portfolio Value & Daily Delta Progress")
        
        # Load snapshot history
        history = get_portfolio_history(days=self.history_days)
        if history.empty:
            st.info("No saved portfolio snapshots yet. Upload statements to populate the database daily history.")
            return

        # Prepare base dataframe
        df = history.copy()
        df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce")
        df = df.sort_values("snapshot_ts").reset_index(drop=True)

        # Ensure we have the total_market_value column
        if "total_market_value" not in df.columns:
            st.warning("Total Portfolio Value history is unavailable in the database. Ensure statements have market value information.")
            return

        # Controls row
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            group_mode = st.radio(
                "Grouping Mode:",
                ["Clean Daily (One per day)", "Show All Runs"],
                index=0,
                horizontal=True,
                key="pp_vt_group_mode"
            )
        with col_c2:
            delta_metric = st.selectbox(
                "Delta Chart Metric:",
                ["Daily Value Change (Difference between consecutive loads)", "Statement Daily P/L (Broker reported)"],
                index=0,
                key="pp_vt_delta_metric"
            )

        # Apply grouping if selected
        if group_mode == "Clean Daily (One per day)":
            chart_df = df.groupby("snapshot_date").last().reset_index()
            chart_df["snapshot_ts"] = pd.to_datetime(chart_df["snapshot_ts"])
            chart_df = chart_df.sort_values("snapshot_date").reset_index(drop=True)
        else:
            chart_df = df.copy()

        # Compute consecutive value difference delta
        chart_df["val_delta"] = chart_df["total_market_value"].diff()

        # Latest metrics
        latest = chart_df.iloc[-1]
        prev = chart_df.iloc[-2] if len(chart_df) > 1 else None

        # Display Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        
        # 1. Total Value
        val_delta_str = None
        if prev is not None:
            change = latest["total_market_value"] - prev["total_market_value"]
            val_delta_str = f"${change:+,.2f}"
        m1.metric("Portfolio Value", f"${latest['total_market_value']:,.2f}", delta=val_delta_str)

        # 2. Daily Delta (Value Change)
        latest_val_delta = latest["val_delta"] if pd.notna(latest["val_delta"]) else 0.0
        m2.metric(
            "Latest Value Delta (+/-)",
            f"${latest_val_delta:+,.2f}",
            delta="Difference from last load"
        )

        # 3. Statement Daily P/L
        m3.metric(
            "Statement Daily P/L",
            fmt_money(latest["total_pl_day"]),
            delta=None
        )

        # 4. Period High / Low
        max_val = chart_df["total_market_value"].max()
        min_val = chart_df["total_market_value"].min()
        m4.metric("Period High / Low", f"${max_val:,.0f}", delta=f"Low: ${min_val:,.0f}", delta_color="off")

        # --- Chart 1: Portfolio Value Progress (Line + Shaded Area) ---
        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(
            x=chart_df["snapshot_ts"],
            y=chart_df["total_market_value"],
            mode="lines+markers",
            name="Portfolio Value",
            line=dict(color=BRAND["blue"], width=3),
            fill="tozeroy",
            fillcolor="rgba(147, 197, 253, 0.08)",
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d %H:%M}<br><b>Portfolio Value:</b> $%{y:,.2f}<extra></extra>"
        ))
        fig_val.update_layout(
            title="Portfolio Value Progress",
            xaxis=dict(title="Date / Time", gridcolor=BRAND["grid"]),
            yaxis=dict(title="Portfolio Value ($)", gridcolor=BRAND["grid"], tickformat="$,.2f"),
        )
        style_figure(fig_val, height=400)
        st.plotly_chart(fig_val, width="stretch", theme=None)

        # --- Chart 2: Daily Delta (Bar Chart) ---
        fig_delta = go.Figure()
        
        # Select target metric column
        if delta_metric.startswith("Daily Value Change"):
            delta_col = "val_delta"
            chart_title = "Daily Value Delta (+/-)"
            y_axis_title = "Value Change ($)"
        else:
            delta_col = "total_pl_day"
            chart_title = "Statement Daily P/L"
            y_axis_title = "P/L ($)"

        # Fill NaNs with 0.0 for delta plotting
        plot_df = chart_df.copy()
        plot_df[delta_col] = plot_df[delta_col].fillna(0.0)

        # Set bar colors dynamically (green for positive, red for negative)
        bar_colors = plot_df[delta_col].map(lambda val: BRAND["green"] if val >= 0 else BRAND["red"]).tolist()

        fig_delta.add_trace(go.Bar(
            x=plot_df["snapshot_ts"],
            y=plot_df[delta_col],
            name="Daily Delta",
            marker_color=bar_colors,
            opacity=0.85,
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d %H:%M}<br><b>Delta:</b> $%{y:+,.2f}<extra></extra>"
        ))
        fig_delta.update_layout(
            title=chart_title,
            xaxis=dict(title="Date / Time", gridcolor=BRAND["grid"]),
            yaxis=dict(title=y_axis_title, gridcolor=BRAND["grid"], tickformat="$,.2f"),
        )
        fig_delta.add_hline(y=0, line_width=1, line_color=BRAND["muted"])
        style_figure(fig_delta, height=350)
        st.plotly_chart(fig_delta, width="stretch", theme=None)

    def _render_database_status(self):
        status = get_database_status()
        with st.expander("Database Storage", expanded=False):
            st.caption(status["db_path"])
            if status.get("warning"):
                st.warning(status["warning"])
            elif status["configured_env_path"]:
                st.success("Using PORTFOLIO_PERFORMANCE_DB_PATH.")
            else:
                st.info("Using the default local development database path.")

            c1, c2 = st.columns(2)
            c1.metric("Saved Runs", f"{status['run_count']:,}")
            c2.metric("Saved Positions", f"{status['position_count']:,}")
            if status.get("latest_snapshot_ts"):
                st.caption(
                    f"Latest snapshot: {status['latest_snapshot_ts']} "
                    f"from {status.get('latest_source_file') or 'unknown source'}"
                )

    def _render_data_tab(self, positions: pd.DataFrame, metadata: dict):
        st.markdown("### Parsed Position Snapshot")
        self._render_position_table(positions)
        csv_bytes = positions.to_csv(index=False).encode("utf-8")
        date_part = str(metadata.get("snapshot_date") or datetime.now().date())
        st.download_button(
            "Download Parsed Snapshot CSV",
            data=csv_bytes,
            file_name=f"portfolio_performance_{date_part}.csv",
            mime="text/csv",
            width="stretch",
        )

    def _render_details_tab(self, details: pd.DataFrame, metadata: dict):
        st.markdown("### Raw Broker Position Details")
        if details.empty:
            st.info("No raw detail rows were saved for this snapshot. Re-upload the Schwab CSV to persist option legs.")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detail Rows", f"{len(details):,}")
        c2.metric("Option Legs", f"{int((details['row_type'] == 'option_leg').sum()):,}")
        c3.metric("Strategies", f"{int((details['row_type'] == 'strategy').sum()):,}")
        c4.metric("Tickers", f"{details['underlying'].dropna().nunique():,}")

        row_types = sorted(details["row_type"].dropna().unique().tolist())
        selected_types = st.multiselect(
            "Detail Row Types",
            row_types,
            default=[row_type for row_type in ["ticker_summary", "strategy", "option_leg"] if row_type in row_types],
            key="pp_detail_row_types",
        )
        underlyings = sorted(details["underlying"].dropna().unique().tolist())
        selected_underlying = st.selectbox("Underlying", ["All"] + underlyings, key="pp_detail_underlying")

        view = details.copy()
        if selected_types:
            view = view[view["row_type"].isin(selected_types)]
        if selected_underlying != "All":
            view = view[view["underlying"] == selected_underlying]

        view["Source"] = view["underlying"].apply(get_broker_dot)
        view["underlying"] = view["underlying"].apply(clean_ticker_for_lookup)

        display_cols = [
            "Source",
            "account_group",
            "underlying",
            "row_type",
            "strategy",
            "instrument",
            "side",
            "quantity",
            "days",
            "expiration",
            "strike",
            "call_put",
            "trade_price",
            "mark_price",
            "delta",
            "theta",
            "gamma",
            "vega",
            "pl_open",
            "pl_day",
        ]
        available = [column for column in display_cols if column in view.columns]
        st.dataframe(
            view[available],
            width="stretch",
            hide_index=True,
            height=min(820, max(460, 34 * (len(view) + 1))),
            column_config={
                "Source": st.column_config.TextColumn("Source", width="small"),
                "quantity": st.column_config.NumberColumn("Contracts", format="%.0f"),
                "trade_price": st.column_config.NumberColumn("Trade Price", format="$%.3f"),
                "mark_price": st.column_config.NumberColumn("Mark", format="$%.3f"),
                "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                "delta": st.column_config.NumberColumn("Delta", format="%.2f"),
                "theta": st.column_config.NumberColumn("Theta", format="%.2f"),
                "gamma": st.column_config.NumberColumn("Gamma", format="%.2f"),
                "vega": st.column_config.NumberColumn("Vega", format="%.2f"),
                "pl_open": st.column_config.NumberColumn("P/L Open", format="$%.2f"),
                "pl_day": st.column_config.NumberColumn("P/L Day", format="$%.2f"),
            },
        )

        csv_bytes = view.to_csv(index=False).encode("utf-8")
        date_part = str(metadata.get("snapshot_date") or datetime.now().date())
        st.download_button(
            "Download Raw Detail Rows CSV",
            data=csv_bytes,
            file_name=f"portfolio_details_{date_part}.csv",
            mime="text/csv",
            width="stretch",
        )

    def _render_position_table(self, positions: pd.DataFrame):
        positions = positions.copy()
        positions["Source"] = positions["ticker"].apply(get_broker_dot)
        positions["ticker"] = positions["ticker"].apply(clean_ticker_for_lookup)
        display_cols = [
            "Source",
            "ticker",
            "account_group",
            "quantity",
            "mark_price",
            "market_value",
            "pl_open",
            "pl_day",
            "theta",
            "delta",
            "bp_effect",
        ]
        available = [column for column in display_cols if column in positions.columns]
        st.dataframe(
            positions[available],
            width="stretch",
            hide_index=True,
            height=min(820, max(460, 38 * (len(positions) + 1))),
            column_config={
                "Source": st.column_config.TextColumn("Source", width="small"),
                "ticker": st.column_config.TextColumn("Ticker"),
                "account_group": st.column_config.TextColumn("Group"),
                "quantity": st.column_config.NumberColumn("Qty", format="%.2f"),
                "mark_price": st.column_config.NumberColumn("Mark", format="$%.2f"),
                "market_value": st.column_config.NumberColumn("Market Value", format="$%.2f"),
                "pl_open": st.column_config.NumberColumn("P/L Open", format="$%.2f"),
                "pl_day": st.column_config.NumberColumn("P/L Day", format="$%.2f"),
                "theta": st.column_config.NumberColumn("Theta", format="%.2f"),
                "delta": st.column_config.NumberColumn("Delta", format="%.2f"),
                "bp_effect": st.column_config.NumberColumn("BP Effect", format="$%.2f"),
            },
        )

    def _build_watchlist(self, positions: pd.DataFrame) -> list[str]:
        if positions.empty:
            return []
        delta_threshold = max(self.delta_caution, float(positions["delta"].quantile(0.75)))
        risk = positions.copy()
        risk["risk_score"] = (
            risk["pl_open"].lt(0).astype(int)
            + risk["pl_day"].lt(0).astype(int)
            + risk["theta"].lt(0).astype(int)
            + risk["delta"].gt(delta_threshold).astype(int)
        )
        return (
            risk[risk["risk_score"] > 0]
            .sort_values(["risk_score", "delta"], ascending=False)
            .head(self.top_n)["ticker"]
            .tolist()
        )

    @staticmethod
    def _risk_quadrant(row: pd.Series) -> str:
        if row["pl_open"] < 0 and row["theta"] < 0:
            return "Pressure"
        if row["theta"] < 0:
            return "Decay Drag"
        if row["delta"] > 0 and row["pl_open"] >= 0:
            return "Constructive"
        return "Directional"

    def _render_welcome(self):
        st.markdown(
            f"""
            <div style="
                background:linear-gradient(135deg, rgba(26,58,143,0.24), rgba(58,181,74,0.08));
                border:1px solid {BRAND['grid']};
                border-left:4px solid {BRAND['green']};
                border-radius:10px;
                padding:22px 24px;
                margin:12px 0 22px 0;
            ">
                <div style="color:{BRAND['green']};font-size:18px;font-weight:700;margin-bottom:8px;">Portfolio Performance</div>
                <div style="color:{BRAND['muted']};font-size:14px;line-height:1.7;">
                    Upload a Schwab Position Statement CSV from the sidebar. The module parses ticker-level
                    positions, builds performance and risk views, and can save each upload as a local daily snapshot.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
