import logging

logger = logging.getLogger("CalendarFDTSEngine")

def calculate_fdts_signal(spot_price: float, ema_20: float, ema_50: float, ema_200: float, rsi_14: float) -> dict:
    """Calculate the custom FDTS signal and trend score for option calendar selection."""
    try:
        # Determine trend alignment
        bull_alignment = (spot_price > ema_20) and (ema_20 > ema_50) and (ema_50 > ema_200)
        early_trend = (spot_price > ema_20) and (spot_price > ema_50) and not (ema_20 > ema_50)
        bear_alignment = (spot_price < ema_50) and (ema_50 < ema_200)
        
        # Base calculations for score
        if bull_alignment:
            signal = "Buy"
            # RSI adds/subtracts score points
            if rsi_14 > 75:  # overbought, might consolidate
                base_score = 85.0
            elif rsi_14 < 45: # oversold but in uptrend? unlikely, but let's score 80
                base_score = 80.0
            else:
                base_score = 92.0
        elif early_trend:
            signal = "Buy"
            base_score = 82.0
        elif bear_alignment:
            signal = "Sell"
            base_score = 30.0
        else:
            signal = "Neutral"
            base_score = 60.0
            
        # Add technical modifier points
        rsi_factor = (rsi_14 - 50.0) * 0.1  # slightly adjusts score based on neutral rsi
        final_score = min(100.0, max(0.0, base_score + rsi_factor))
        
        return {
            "signal": signal,
            "score": round(final_score, 1),
            "state_description": f"Trend: {signal} (Score: {final_score:.1f})"
        }
    except Exception as e:
        logger.error(f"Error calculating FDTS signal: {e}")
        return {
            "signal": "Neutral",
            "score": 50.0,
            "state_description": "FDTS signal calculation failed."
        }
