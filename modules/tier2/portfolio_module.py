"""Unified Portfolio Module.

Combines Portfolio Performance tracking and Portfolio Risk Management
into a single module interface.
"""

from __future__ import annotations

import streamlit as st
from modules.base_module import FazDaneModule
from modules.tier2.portfolio_performance import PortfolioPerformanceModule
from modules.tier2.portfolio_risk_management import PortfolioRiskManagementModule


class PortfolioModule(FazDaneModule):
    MODULE_NAME = "Portfolio Module"
    MODULE_ICON = "💼"
    MODULE_DESCRIPTION = "Unified portfolio management module combining performance tracking and risk management."
    TIER = 2

    def __init__(self):
        super().__init__()
        self.portfolio_performance = PortfolioPerformanceModule()
        self.portfolio_risk = PortfolioRiskManagementModule()

    def render_sidebar(self):
        st.markdown("**Portfolio Mode**")
        portfolio_mode = st.selectbox(
            "Select Portfolio Tool",
            options=["Portfolio Performance", "Risk Management"],
            key="pm_portfolio_mode",
            label_visibility="collapsed"
        )
        st.divider()

        if portfolio_mode == "Portfolio Performance":
            self.portfolio_performance.render_sidebar()
        elif portfolio_mode == "Risk Management":
            self.portfolio_risk.render_sidebar()

    def render_main(self):
        portfolio_mode = st.session_state.get("pm_portfolio_mode", "Portfolio Performance")

        if portfolio_mode == "Portfolio Performance":
            self.portfolio_performance.render_main()
        elif portfolio_mode == "Risk Management":
            self.portfolio_risk.render_main()
