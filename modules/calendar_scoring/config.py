# Calendar Opportunity Scoring Engine - Configurations

# Default Scoring Weights (sum to 100%)
DEFAULT_WEIGHTS = {
    "trend_weight": 0.20,
    "option_structure_weight": 0.20,
    "volatility_weight": 0.15,
    "fdts_weight": 0.15,
    "pca_weight": 0.10,
    "cluster_weight": 0.10,
    "leading_lagging_weight": 0.05,
    "liquidity_weight": 0.03,
    "event_risk_weight": 0.02,
    "institutional_flow_weight": 0.00,  # Labeled as Phase 2
}

# Core Strategy Assumptions
STRATEGY_CONFIG = {
    "strategy_type": "Bullish Calendar Spread",
    "short_dte_target": 20,
    "long_dte_target": 40,
    "target_delta": 0.25,
    "min_dte_short": 12,
    "max_dte_short": 28,
    "min_dte_long": 30,
    "max_dte_long": 50,
}

# Hard Filter Settings
HARD_FILTERS = {
    "max_bid_ask_spread_pct": 0.07,  # 7%
    "min_option_volume": 100,
    "min_open_interest": 500,
    "min_adx": 15,
}

# Model Metadata
MODEL_VERSION = "Phase 1 - MVP v2.02"
