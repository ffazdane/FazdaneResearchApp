import sqlite3
from datetime import datetime, date, timedelta
from modules.cycle_engine.cycle_database import save_events, load_events
from utils.earnings_calendar_store import load_earnings_events

def seed_macro_calendar_if_empty(as_of_date: date):
    """Seed the macro calendar with standard 2026 dates if empty."""
    # Let's check if we have any events in database
    events = load_events((as_of_date - timedelta(days=30)).strftime("%Y-%m-%d"), (as_of_date + timedelta(days=90)).strftime("%Y-%m-%d"))
    if len(events) > 0:
        return
        
    # Seed 2026 major events
    # 2026 FOMC Meetings (standard 8 sessions)
    fomc_dates = [
        "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-11-05", "2026-12-16"
    ]
    
    # 2026 CPI Release Dates (approximate monthly schedules)
    cpi_dates = [
        "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-10",
        "2026-05-13", "2026-06-10", "2026-07-15", "2026-08-12",
        "2026-09-11", "2026-10-14", "2026-11-12", "2026-12-10"
    ]
    
    # OPEX Dates (Third Friday of every month)
    opex_dates = [
        "2026-01-16", "2026-02-20", "2026-03-20", "2026-04-17",
        "2026-05-15", "2026-06-19", "2026-07-17", "2026-08-21",
        "2026-09-18", "2026-10-16", "2026-11-20", "2026-12-18"
    ]
    
    rows = []
    for d_str in fomc_dates:
        rows.append({"event_date": d_str, "event_type": "FOMC", "event_name": "FOMC Rate Decision", "expected_impact": "HIGH", "source": "Fed"})
    for d_str in cpi_dates:
        rows.append({"event_date": d_str, "event_type": "CPI", "event_name": "Consumer Price Index Inflation", "expected_impact": "HIGH", "source": "BLS"})
    for d_str in opex_dates:
        rows.append({"event_date": d_str, "event_type": "OPEX", "event_name": "Monthly Options Expiration", "expected_impact": "MEDIUM", "source": "CBOE"})
        
    save_events(rows)

def calculate_liquidity_event_risk(ticker: str, as_of_date: date) -> dict:
    """
    Evaluates macro calendar events and stock earnings proximity to compute
    an event risk score (0-100) and position-size modifier.
    """
    seed_macro_calendar_if_empty(as_of_date)
    
    # Query events from database for next 30 days
    end_date = as_of_date + timedelta(days=30)
    db_events = load_events(as_of_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    
    # Load earnings events for the ticker
    earnings_date_val = None
    try:
        ec_df = load_earnings_events([ticker], as_of_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if not ec_df.empty:
            earnings_date_val = datetime.strptime(ec_df["Date"].iloc[0], "%Y-%m-%d").date()
    except Exception:
        pass
        
    event_risks = []
    
    # 1. Evaluate stock earnings
    if earnings_date_val is not None:
        days_to_earn = (earnings_date_val - as_of_date).days
        event_risks.append({
            "name": "Earnings Announcement",
            "type": "Earnings",
            "days": days_to_earn,
            "weight": 60.0 if days_to_earn <= 5 else 30.0 if days_to_earn <= 10 else 10.0
        })
        
    # 2. Evaluate macro events
    for ev in db_events:
        ev_date = datetime.strptime(ev["event_date"], "%Y-%m-%d").date()
        days_to_ev = (ev_date - as_of_date).days
        if days_to_ev < 0:
            continue
            
        weight = 0.0
        if ev["event_type"] == "FOMC":
            weight = 40.0 if days_to_ev <= 2 else 20.0 if days_to_ev <= 5 else 5.0
        elif ev["event_type"] == "CPI":
            weight = 30.0 if days_to_ev <= 1 else 15.0 if days_to_ev <= 3 else 5.0
        elif ev["event_type"] == "OPEX":
            weight = 15.0 if days_to_ev <= 2 else 5.0
            
        event_risks.append({
            "name": ev["event_name"],
            "type": ev["event_type"],
            "days": days_to_ev,
            "weight": weight
        })
        
    # Sort risks by closeness and score
    if event_risks:
        event_risks = sorted(event_risks, key=lambda x: x["days"])
        nearest = event_risks[0]
        nearest_event = nearest["name"]
        days_to_event = nearest["days"]
        
        # Risk score is a cumulative sum clamped to 100
        risk_score = min(sum(e["weight"] for e in event_risks[:3]), 100.0)
    else:
        nearest_event = "None"
        days_to_event = 99
        risk_score = 0.0

    # 3. Determine status label
    if any(e["type"] in ["FOMC", "CPI"] and e["days"] <= 3 for e in event_risks):
        liq_status = "Negative Liquidity Window"
    elif any(e["type"] == "OPEX" and e["days"] <= 5 for e in event_risks):
        liq_status = "Options Pinning/Chop week"
    else:
        liq_status = "Neutral"

    # 4. Sizing modifier
    if risk_score >= 80.0:
        modifier = 0.0     # Avoid new trades
    elif risk_score >= 60.0:
        modifier = 0.50    # Cut position size in half
    elif risk_score >= 30.0:
        modifier = 0.75    # Reduce size slightly
    else:
        modifier = 1.0     # Full size allowed

    return {
        "event_risk_score": round(risk_score, 1),
        "nearest_event": nearest_event,
        "days_to_event": days_to_event,
        "liquidity_cycle_status": liq_status,
        "trade_size_modifier": modifier,
        "earnings_date": earnings_date_val.strftime("%Y-%m-%d") if earnings_date_val else None
    }
