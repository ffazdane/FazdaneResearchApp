"""Unified Search Module.

Combines Option Search Universe, Ticker Pullback Strategy, and Options Liquidity Discovery
into a single module interface.
"""

from __future__ import annotations

import streamlit as st
from modules.base_module import FazDaneModule
from modules.tier1.option_search import OptionSearchModule
from modules.tier1.ticker_pullback_strategy import TickerPullbackStrategyModule
from modules.tier1.options_liquidity import OptionsLiquidityModule


class SearchModule(FazDaneModule):
    MODULE_NAME = "Search Module"
    MODULE_ICON = "🔍"
    MODULE_DESCRIPTION = "Unified search engine combining Option Search Universe, Ticker Pullback Strategy, and Options Liquidity Discovery."
    TIER = 1

    def __init__(self):
        super().__init__()
        self.option_search = OptionSearchModule()
        self.pullback_strategy = TickerPullbackStrategyModule()
        self.liquidity_discovery = OptionsLiquidityModule()

    def render_sidebar(self):
        st.markdown("**Search Mode**")
        search_mode = st.selectbox(
            "Select Search Tool",
            options=["Option Search Universe", "Ticker Pullback Strategy", "Options Liquidity Discovery"],
            key="sm_search_mode",
            label_visibility="collapsed"
        )
        st.divider()

        if search_mode == "Option Search Universe":
            self.option_search.render_sidebar()
        elif search_mode == "Ticker Pullback Strategy":
            self.pullback_strategy.render_sidebar()
        elif search_mode == "Options Liquidity Discovery":
            self.liquidity_discovery.render_sidebar()

    def render_main(self):
        search_mode = st.session_state.get("sm_search_mode", "Option Search Universe")

        if search_mode == "Option Search Universe":
            self.option_search.render_main()
        elif search_mode == "Ticker Pullback Strategy":
            self.pullback_strategy.render_main()
        elif search_mode == "Options Liquidity Discovery":
            self.liquidity_discovery.render_main()
