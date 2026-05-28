import pandas as pd
import numpy as np
import logging
from modules.calendar_scoring.data_loader import black_scholes_call

logger = logging.getLogger("CalendarOptionChainEngine")

def select_optimal_calendar_legs(option_data: dict, spot_price: float, hv_30: float, target_delta: float = 0.25) -> dict:
    """Filter and select the optimal calendar spread legs for a candidate."""
    try:
        short_calls = option_data["short_calls"]
        long_calls = option_data["long_calls"]
        short_dte = option_data["short_dte"]
        long_dte = option_data["long_dte"]
        
        r = 0.045  # 4.5% risk-free rate
        
        # Calculate Delta for all short strikes if not already present or if empty/NaN
        if "delta" not in short_calls.columns or short_calls["delta"].isnull().all():
            short_deltas = []
            for _, row in short_calls.iterrows():
                strike = row["strike"]
                iv = row.get("impliedVolatility", hv_30)
                if np.isnan(iv) or iv <= 0:
                    iv = hv_30
                _, delta, _, _, _ = black_scholes_call(spot_price, strike, short_dte / 365.0, r, iv)
                short_deltas.append(delta)
            short_calls = short_calls.copy()
            short_calls["delta"] = short_deltas
            
        # Select strike closest to target delta (0.25) in short calls
        short_calls["delta_diff"] = (short_calls["delta"] - target_delta).abs()
        best_short = short_calls.sort_values(by="delta_diff").iloc[0]
        selected_strike = best_short["strike"]
        
        # Find matching strike in long calls
        matching_long_rows = long_calls[long_calls["strike"] == selected_strike]
        if matching_long_rows.empty:
            # Sort strikes in long calls and find closest to selected_strike
            long_calls = long_calls.copy()
            long_calls["strike_diff"] = (long_calls["strike"] - selected_strike).abs()
            best_long = long_calls.sort_values(by="strike_diff").iloc[0]
            selected_strike = best_long["strike"]
            matching_long_rows = long_calls[long_calls["strike"] == selected_strike]
            
        best_long = matching_long_rows.iloc[0]
        
        # Recalculate options Greeks for both legs to ensure complete accuracy
        iv_short = best_short.get("impliedVolatility", hv_30)
        iv_long = best_long.get("impliedVolatility", hv_30 + 0.02)
        if np.isnan(iv_short) or iv_short <= 0: iv_short = hv_30
        if np.isnan(iv_long) or iv_long <= 0: iv_long = hv_30 + 0.02
        
        price_s, delta_s, gamma_s, theta_s, vega_s = black_scholes_call(spot_price, selected_strike, short_dte / 365.0, r, iv_short)
        price_l, delta_l, gamma_l, theta_l, vega_l = black_scholes_call(spot_price, selected_strike, long_dte / 365.0, r, iv_long)
        
        # Bid/Ask and Mid calculations
        short_bid = float(best_short.get("bid", price_s * 0.95))
        short_ask = float(best_short.get("ask", price_s * 1.05))
        short_mid = (short_bid + short_ask) / 2.0
        
        long_bid = float(best_long.get("bid", price_l * 0.95))
        long_ask = float(best_long.get("ask", price_l * 1.05))
        long_mid = (long_bid + long_ask) / 2.0
        
        # Bid/Ask spread %
        bid_ask_spread_pct = ((short_ask - short_bid) / (short_mid + 1e-8))
        
        # Combined Greeks for the calendar (Long Leg minus Short Leg)
        setup_delta = delta_l - delta_s
        setup_gamma = gamma_l - gamma_s
        setup_theta = theta_l - theta_s  # Should be positive (theta collection)
        setup_vega = vega_l - vega_s    # Should be positive (long volatility exposure)
        
        return {
            "short_expiry": option_data["short_expiry"],
            "long_expiry": option_data["long_expiry"],
            "short_dte": short_dte,
            "long_dte": long_dte,
            "selected_strike": float(selected_strike),
            "short_bid": short_bid,
            "short_ask": short_ask,
            "short_mid": short_mid,
            "long_bid": long_bid,
            "long_ask": long_ask,
            "long_mid": long_mid,
            "front_iv": float(iv_short),
            "back_iv": float(iv_long),
            "bid_ask_spread_pct": float(bid_ask_spread_pct),
            "avg_option_volume": float((best_short.get("volume", 0) + best_long.get("volume", 0)) / 2.0),
            "avg_open_interest": float((best_short.get("openInterest", 0) + best_long.get("openInterest", 0)) / 2.0),
            "setup_delta": float(setup_delta),
            "setup_gamma": float(setup_gamma),
            "setup_theta": float(setup_theta),
            "setup_vega": float(setup_vega),
            "net_debit": float(long_mid - short_mid),
            "max_risk": float(long_mid - short_mid)
        }
    except Exception as e:
        logger.error(f"Error selecting calendar legs: {e}")
        return {}
