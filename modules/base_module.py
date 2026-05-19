"""
FazDane Base Module
Abstract base class that every module inherits from.
Provides: caching, logging, data fetching, shared UI helpers.
"""

from abc import ABC, abstractmethod
import streamlit as st
import pandas as pd
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional


# ══════════════════════════════════════════════════════════════════════
# BASE MODULE CLASS
# ══════════════════════════════════════════════════════════════════════

class FazDaneModule(ABC):
    """
    Abstract base class for all FazDane Analytics modules.

    Subclasses MUST define:
        MODULE_NAME   : str  — display name
        MODULE_ICON   : str  — emoji icon
        TIER          : int  — priority tier (1–4)

    Subclasses MUST implement:
        render_sidebar() — sidebar controls
        render_main()    — main content area
    """

    # ── Module Metadata (override in subclass) ────────────────────────
    MODULE_NAME: str = "Base Module"
    MODULE_ICON: str = "📊"
    MODULE_DESCRIPTION: str = ""
    TIER: int = 1
    SOURCE_NOTEBOOK: str = "Unknown"

    # ── Configuration ─────────────────────────────────────────────────
    REQUIRES_LIVE_DATA: bool = False
    CACHE_TTL: int = 3600          # seconds
    DATA_SOURCES: list = []

    # ══════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ══════════════════════════════════════════════════════════════════

    def __init__(self):
        self.logger = self._setup_logger()
        self.data_cache: Dict[str, Any] = {}
        self.module_params: Dict[str, Any] = {}
        self.logger.info(f"Module initialised: {self.MODULE_NAME}")

    # ── Entry point (called by app.py dispatcher) ─────────────────────
    def run(self):
        """Execute the module — renders sidebar then main content."""
        # NOTE: set_page_config is owned by app.py — do NOT call it here

        with st.sidebar:
            st.markdown(
                f"""
                <div style='
                    background:rgba(26,58,143,0.15);
                    border:1px solid #1e3a5f;
                    border-radius:10px;
                    padding:14px 16px;
                    margin-bottom:12px;
                '>
                    <div style='font-size:22px;margin-bottom:4px;'>{self.MODULE_ICON}</div>
                    <div style='color:#3ab54a;font-weight:700;font-size:15px;'>{self.MODULE_NAME}</div>
                    <div style='color:#64748b;font-size:11px;margin-top:4px;'>Tier {self.TIER}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if self.MODULE_DESCRIPTION:
                st.caption(self.MODULE_DESCRIPTION)

            st.divider()
            self.render_sidebar()
            st.divider()

            # Navigation controls
            st.markdown(
                "<div style='color:#64748b;font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;'>Navigation</div>",
                unsafe_allow_html=True,
            )
            nav_col1, nav_col2 = st.columns(2)
            with nav_col1:
                if st.button(self._back_button_label(), use_container_width=True, help="Return to the module menu"):
                    self._clear_current_tier_selection()
                    st.rerun()
            with nav_col2:
                if st.button("Home", use_container_width=True, help="Return to dashboard"):
                    self._clear_all_navigation()
                    st.rerun()

            if st.button("Refresh Data", use_container_width=True, help="Clear cache and reload this module"):
                st.cache_data.clear()
                st.rerun()

        # Main content
        try:
            self.render_main()
        except Exception as e:
            self.logger.error(f"Render error in {self.MODULE_NAME}: {e}", exc_info=True)
            st.error(f"❌ Error loading module: {e}")
            st.info("💡 Try refreshing the page or adjusting parameters in the sidebar.")

    # ══════════════════════════════════════════════════════════════════
    # ABSTRACT METHODS
    # ══════════════════════════════════════════════════════════════════

    @abstractmethod
    def render_sidebar(self):
        """Render sidebar control panel (filters, parameters, buttons)."""
        pass

    @abstractmethod
    def render_main(self):
        """Render main content area (charts, tables, metrics)."""
        pass

    # ══════════════════════════════════════════════════════════════════
    # DATA FETCHING
    # ══════════════════════════════════════════════════════════════════

    def _fetch_yfinance(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from yfinance with basic error handling."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(**kwargs)
            self.logger.info(f"yfinance: {symbol} → {len(data)} rows")
            return data
        except Exception as e:
            self.logger.error(f"yfinance fetch failed ({symbol}): {e}")
            return None

    # ══════════════════════════════════════════════════════════════════
    # SHARED UI HELPERS
    # ══════════════════════════════════════════════════════════════════

    def render_metrics_row(self, metrics: Dict[str, tuple]):
        """
        Render a horizontal row of st.metric cards.

        metrics format: { "Label": (value, delta, suffix) }
        Example:
            { "Total Opportunities": (42, 5, ""), "Avg IV": (67.3, None, "%") }
        """
        cols = st.columns(len(metrics))
        for idx, (label, (value, delta, suffix)) in enumerate(metrics.items()):
            with cols[idx]:
                formatted_val = f"{value:,.2f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"
                formatted_delta = f"{delta:+.2f}{suffix}" if isinstance(delta, (int, float)) and delta is not None else None
                st.metric(label=label, value=formatted_val, delta=formatted_delta)

    def render_section_header(self, title: str, subtitle: str = ""):
        """Render a branded section header."""
        st.markdown(
            f"""
            <div style='margin-bottom:8px;'>
                <span style='color:#3ab54a;font-size:20px;font-weight:700;'>{title}</span>
                {"<br><span style='color:#64748b;font-size:13px;'>" + subtitle + "</span>" if subtitle else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

    def get_module_info(self) -> Dict[str, Any]:
        return {
            "name": self.MODULE_NAME,
            "icon": self.MODULE_ICON,
            "description": self.MODULE_DESCRIPTION,
            "tier": self.TIER,
            "source": self.SOURCE_NOTEBOOK,
            "cache_ttl": self.CACHE_TTL,
            "data_sources": self.DATA_SOURCES,
        }

    # ══════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _back_button_label(self) -> str:
        labels = {
            1: "Back to Live",
            2: "Back to Analysis",
            3: "Back to Forecast",
            4: "Back to Menu",
        }
        return labels.get(self.TIER, "Back")

    def _clear_current_tier_selection(self) -> None:
        st.session_state["pending_nav"] = {"action": "clear_tier", "tier": self.TIER}
        st.session_state.active_module = None

    def _clear_all_navigation(self) -> None:
        st.session_state["pending_nav"] = {"action": "home"}
        st.session_state.active_module = None

    def _setup_logger(self) -> logging.Logger:
        log = logging.getLogger(self.MODULE_NAME)
        log.setLevel(logging.INFO)
        if not log.handlers:
            os.makedirs("logs", exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.MODULE_NAME).strip("_")
            handler = logging.FileHandler(os.path.join("logs", f"{safe_name}.log"))
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
            )
            log.addHandler(handler)
        return log
