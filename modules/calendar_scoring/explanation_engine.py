import logging

logger = logging.getLogger("CalendarExplanationEngine")

def generate_llm_prompt(cand: dict) -> str:
    """Generate a clean, copyable text prompt for a ChatGPT/LLM to explain the calendar opportunity decision."""
    setup = cand.get("option_setup", {})
    
    setup_str = (
        f"{setup.get('short_expiry')} & {setup.get('long_expiry')} Call Calendar "
        f"at strike ${setup.get('selected_strike', 0.0):.2f} (Net Debit: ${setup.get('net_debit', 0.0):.2f})"
        if setup else "No option setup generated."
    )
    
    prompt = f"""You are an options strategy analyst.

Explain why this ticker was ranked as a {cand.get('recommendation', 'Deploy')} candidate for a bullish calendar spread.

Use the structured data below:
Ticker: {cand.get('ticker')}
Final Score: {cand.get('final_score', 0.0):.1f}
FDTS: {cand.get('fdts_signal')} (Score: {cand.get('fdts_score', 0.0):.1f})
Trend Score: {cand.get('trend_score', 0.0):.1f}
Option Structure Score: {cand.get('option_structure_score', 0.0):.1f}
Volatility Score: {cand.get('volatility_score', 0.0):.1f}
PCA Score: {cand.get('pca_score', 0.0):.1f}
Cluster: {cand.get('cluster_label')} (Score: {cand.get('cluster_score', 0.0):.1f})
Leading/Lagging: {cand.get('leading_lagging_state')} (Score: {cand.get('leading_lagging_score', 0.0):.1f})
Liquidity: {cand.get('liquidity_score', 0.0):.1f}
Event Risk: {cand.get('event_risk_score', 0.0):.1f}
Selected Calendar Setup: {setup_str}
Market Regime: {cand.get('market_regime')}

Explain:
1. Why this ticker is attractive
2. What risk to monitor
3. Why this is suitable for a 20/40 DTE 25-delta calendar
4. What would invalidate the trade"""
    
    return prompt
