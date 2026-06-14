from datetime import datetime, date, timedelta

def add_trading_days(start_date: date, days: int) -> date:
    """Add a specified number of trading days, skipping weekends."""
    curr = start_date
    step = 1 if days >= 0 else -1
    for _ in range(abs(days)):
        curr += timedelta(days=step)
        while curr.weekday() >= 5:  # Skip Saturday and Sunday
            curr += timedelta(days=step)
    return curr

def calculate_turning_points(
    as_of_date: date,
    days_to_peak: int,
    days_to_bottom: int,
    cycle_strength: float,
    method_agreement: float,
    current_price: float = None,
    support: float = None,
    resistance: float = None,
    vix_stable: bool = True,
    event_risk_score: float = 20.0,
    days_to_earnings: int = None
) -> dict:
    """
    Projects the next peak and bottom windows, and adjusts confidence scores
    based on macro events, support/resistance, VIX stability, and method agreement.
    """
    # 1. Project target dates
    peak_date = add_trading_days(as_of_date, days_to_peak)
    bottom_date = add_trading_days(as_of_date, days_to_bottom)

    # 2. Project windows (+/- 2 trading days)
    peak_window_start = add_trading_days(peak_date, -2)
    peak_window_end = add_trading_days(peak_date, 2)
    
    bottom_window_start = add_trading_days(bottom_date, -2)
    bottom_window_end = add_trading_days(bottom_date, 2)

    # 3. Base confidence is cycle strength
    base_conf = cycle_strength

    # 4. Apply confidence adjusters
    peak_adj = 0.0
    bottom_adj = 0.0

    # Proximity to support/resistance (increases confidence)
    if current_price is not None:
        if resistance is not None and current_price >= resistance * 0.98:
            peak_adj += 10.0  # Near resistance, peak more likely
        if support is not None and current_price <= support * 1.02:
            bottom_adj += 10.0  # Near support, bottom more likely

    # Method agreement (increases confidence)
    if method_agreement > 75.0:
        peak_adj += 5.0
        bottom_adj += 5.0
    elif method_agreement < 40.0:
        peak_adj -= 10.0
        bottom_adj -= 10.0

    # VIX stability
    if vix_stable:
        peak_adj += 5.0
        bottom_adj += 5.0
    else:
        peak_adj -= 15.0
        bottom_adj -= 15.0

    # Event risk score (decreases confidence if high)
    if event_risk_score > 60.0:
        peak_adj -= 10.0
        bottom_adj -= 10.0
    if event_risk_score > 80.0:
        peak_adj -= 15.0
        bottom_adj -= 15.0

    # Earnings proximity (decreases confidence if extremely close)
    if days_to_earnings is not None:
        if days_to_earnings < 7:
            peak_adj -= 15.0
            bottom_adj -= 15.0

    # Combine and clamp scores between 10% and 95%
    peak_confidence = max(min(base_conf + peak_adj, 95.0), 10.0)
    bottom_confidence = max(min(base_conf + bottom_adj, 95.0), 10.0)

    return {
        "next_peak_date": peak_date.strftime("%Y-%m-%d"),
        "next_bottom_date": bottom_date.strftime("%Y-%m-%d"),
        "next_peak_window": [peak_window_start.strftime("%Y-%m-%d"), peak_window_end.strftime("%Y-%m-%d")],
        "next_bottom_window": [bottom_window_start.strftime("%Y-%m-%d"), bottom_window_end.strftime("%Y-%m-%d")],
        "peak_confidence": round(peak_confidence, 1),
        "bottom_confidence": round(bottom_confidence, 1)
    }
