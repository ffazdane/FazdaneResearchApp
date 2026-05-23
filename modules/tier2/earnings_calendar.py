"""
FazDane Analytics - Tier 2
Earnings Calendar
"""

import calendar
from collections import defaultdict
from datetime import date, datetime
from html import escape

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

from modules.base_module import FazDaneModule
from utils.earnings_calendar_store import (
    get_coverage_sample,
    get_database_summary,
    get_recent_events,
    get_saved_tickers,
    load_earnings_events,
    load_market_earnings,
    mark_market_dates,
    mark_ticker_coverage,
    missing_market_dates,
    missing_ticker_coverage,
    save_earnings_events,
)
from utils.universe_manager import get_tickers, render_universe_manager


def normalize_earnings_dates(value) -> list[pd.Timestamp]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, pd.Series, pd.DatetimeIndex)):
        raw_values = list(value)
    else:
        raw_values = [value]

    dates = []
    for raw in raw_values:
        try:
            ts = pd.Timestamp(raw)
        except Exception:
            continue
        if not pd.isna(ts):
            dates.append(ts)
    return dates


def earnings_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    today = pd.Timestamp(datetime.now().date())
    return today - pd.Timedelta(days=10), today + pd.Timedelta(days=30)


def date_only_timestamp(value) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def calendar_fallback_rows(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    try:
        ticker_obj = yf.Ticker(ticker)
        cal = ticker_obj.calendar
    except Exception:
        return []

    if cal is None:
        return []

    earnings_dates = []
    eps_estimate = None
    if isinstance(cal, dict):
        earnings_dates = normalize_earnings_dates(cal.get("Earnings Date"))
        eps_estimate = cal.get("Earnings Average")
    elif isinstance(cal, pd.DataFrame) and not cal.empty:
        if "Earnings Date" in cal.index:
            earnings_dates = normalize_earnings_dates(cal.loc["Earnings Date"].dropna().tolist())
        elif "Earnings Date" in cal.columns:
            earnings_dates = normalize_earnings_dates(cal["Earnings Date"].dropna().tolist())
        if "Earnings Average" in cal.index:
            eps_values = cal.loc["Earnings Average"].dropna()
            eps_estimate = eps_values.iloc[0] if len(eps_values) else None
        elif "Earnings Average" in cal.columns:
            eps_values = cal["Earnings Average"].dropna()
            eps_estimate = eps_values.iloc[0] if len(eps_values) else None

    rows = []
    for ts in earnings_dates:
        event_day = date_only_timestamp(ts)
        if event_day is None or not start <= event_day <= end:
            continue
        rows.append(
            {
                "Date": ts.strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Time": "TBD" if isinstance(ts.date(), date) and ts.hour == 0 and ts.minute == 0 else ts.strftime("%I:%M %p %Z"),
                "EPS Estimate": eps_estimate,
                "Reported EPS": None,
                "Surprise %": None,
                "Source": "yfinance calendar",
            }
        )
    return rows


def parse_eps_forecast(value):
    if value is None:
        return None
    text = str(value).replace("$", "").replace(",", "").strip()
    if text in {"", "N/A", "n/a", "--"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def nasdaq_time_label(value: str) -> str:
    text = str(value or "").lower()
    if "after" in text:
        return "After Market Close"
    if "pre" in text or "before" in text:
        return "Before Market Open"
    if "during" in text:
        return "During Market"
    return "TBD"


def observed_market_holiday(day: pd.Timestamp) -> pd.Timestamp:
    if day.weekday() == 5:
        return day - pd.Timedelta(days=1)
    if day.weekday() == 6:
        return day + pd.Timedelta(days=1)
    return day


def easter_date(year: int) -> pd.Timestamp:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def nth_weekday(year: int, month: int, weekday: int, nth: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + (nth - 1) * 7)


def last_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    last = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    offset = (last.weekday() - weekday) % 7
    return last - pd.Timedelta(days=offset)


def market_holidays(year: int) -> set[pd.Timestamp]:
    holidays = {
        observed_market_holiday(pd.Timestamp(year=year, month=1, day=1)),
        nth_weekday(year, 1, 0, 3),
        nth_weekday(year, 2, 0, 3),
        easter_date(year) - pd.Timedelta(days=2),
        last_weekday(year, 5, 0),
        observed_market_holiday(pd.Timestamp(year=year, month=6, day=19)),
        observed_market_holiday(pd.Timestamp(year=year, month=7, day=4)),
        nth_weekday(year, 9, 0, 1),
        nth_weekday(year, 11, 3, 4),
        observed_market_holiday(pd.Timestamp(year=year, month=12, day=25)),
    }
    return {holiday.normalize() for holiday in holidays}


def is_market_trading_day(day: pd.Timestamp) -> bool:
    day = day.normalize()
    if day.weekday() >= 5:
        return False
    years = {day.year - 1, day.year, day.year + 1}
    holidays = set().union(*(market_holidays(year) for year in years))
    return day not in holidays


def next_market_trading_days(count: int, start_day: pd.Timestamp | None = None) -> list[pd.Timestamp]:
    day = (start_day or pd.Timestamp(datetime.now().date())).normalize()
    trading_days = []
    while len(trading_days) < count:
        if is_market_trading_day(day):
            trading_days.append(day)
        day += pd.Timedelta(days=1)
    return trading_days


def nasdaq_calendar_fallback_rows(tickers: tuple[str, ...], start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    wanted = {ticker.upper() for ticker in tickers if ticker and not ticker.startswith("^") and "=" not in ticker}
    if not wanted:
        return []

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/market-activity/earnings",
    }
    rows = []
    for day in pd.date_range(start=start, end=end, freq="D"):
        url = f"https://api.nasdaq.com/api/calendar/earnings?date={day.strftime('%Y-%m-%d')}"
        try:
            response = requests.get(url, headers=headers, timeout=12)
            if response.status_code != 200:
                continue
            payload = response.json()
            earnings_rows = payload.get("data", {}).get("rows") or []
        except Exception:
            continue

        for item in earnings_rows:
            symbol = str(item.get("symbol", "")).upper()
            if symbol not in wanted:
                continue
            rows.append(
                {
                    "Date": day.strftime("%Y-%m-%d"),
                    "Ticker": symbol,
                    "Time": nasdaq_time_label(item.get("time")),
                    "EPS Estimate": parse_eps_forecast(item.get("epsForecast")),
                    "Reported EPS": None,
                    "Surprise %": None,
                    "Source": "Nasdaq earnings calendar",
                }
            )
    return rows


def nasdaq_earnings_rows_for_date(day: pd.Timestamp) -> list[dict]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/market-activity/earnings",
    }
    url = f"https://api.nasdaq.com/api/calendar/earnings?date={day.strftime('%Y-%m-%d')}"
    try:
        response = requests.get(url, headers=headers, timeout=12)
        if response.status_code != 200:
            return []
        payload = response.json()
        return payload.get("data", {}).get("rows") or []
    except Exception:
        return []


def safe_price_lookup(symbol: str) -> float | None:
    try:
        ticker = yf.Ticker(symbol)
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            for key in ("last_price", "regular_market_price", "previous_close"):
                try:
                    value = fast_info.get(key) if hasattr(fast_info, "get") else getattr(fast_info, key, None)
                except Exception:
                    value = None
                if value is not None and not pd.isna(value):
                    return float(value)

        history = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if history is not None and not history.empty and "Close" in history:
            close = history["Close"].dropna()
            if not close.empty:
                return float(close.iloc[-1])
    except Exception:
        return None
    return None


def batch_latest_prices(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}

    prices = {}
    try:
        data = yf.download(
            tickers=" ".join(symbols),
            period="5d",
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True,
            auto_adjust=False,
        )
        if data is None or data.empty:
            return prices

        if isinstance(data.columns, pd.MultiIndex):
            for symbol in symbols:
                if symbol not in data.columns.get_level_values(0):
                    continue
                close = data[symbol].get("Close")
                if close is None:
                    continue
                close = close.dropna()
                if not close.empty:
                    prices[symbol] = float(close.iloc[-1])
        elif "Close" in data:
            close = data["Close"].dropna()
            if len(symbols) == 1 and not close.empty:
                prices[symbols[0]] = float(close.iloc[-1])
    except Exception:
        return prices

    return prices


def fetch_next_week_earnings_by_price(min_price: float) -> pd.DataFrame:
    trading_days = next_market_trading_days(7)
    date_keys = [day.strftime("%Y-%m-%d") for day in trading_days]
    missing_dates = set(missing_market_dates(date_keys))
    candidates = []
    seen = set()

    for day in trading_days:
        date_key = day.strftime("%Y-%m-%d")
        if date_key not in missing_dates:
            continue
        for item in nasdaq_earnings_rows_for_date(day):
            symbol = str(item.get("symbol", "")).upper().strip()
            row_key = (date_key, symbol)
            if not symbol or row_key in seen or symbol.startswith("^") or "=" in symbol:
                continue
            seen.add(row_key)
            candidates.append(
                {
                    "Date": date_key,
                    "Ticker": symbol,
                    "Name": str(item.get("name") or item.get("companyName") or "").strip(),
                    "Time": nasdaq_time_label(item.get("time")),
                    "Source": "Nasdaq earnings calendar",
                }
            )

    if missing_dates:
        price_map = batch_latest_prices(sorted({row["Ticker"] for row in candidates}))
        market_rows = []
        for candidate in candidates:
            symbol = candidate["Ticker"]
            price = price_map.get(symbol)
            if price is None:
                price = safe_price_lookup(symbol)
            market_rows.append({**candidate, "Price": price})

        save_earnings_events(market_rows, scope="market_next_7")
        mark_market_dates(sorted(missing_dates), scope="market_next_7")

    records = load_market_earnings(date_keys, min_price, scope="market_next_7")
    if records.empty:
        return records
    return records.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def build_next_week_earnings_html(records: pd.DataFrame, min_price: float) -> str:
    cards = []

    for day in next_market_trading_days(7):
        date_key = day.strftime("%Y-%m-%d")
        day_records = records[records["Date"] == date_key] if not records.empty else pd.DataFrame()
        weekday = day.strftime("%a")
        label = f"{day.strftime('%b')} {day.day}"

        if day_records.empty:
            table_body = "<tr><td class='empty-row' colspan='5'>No matches</td></tr>"
        else:
            table_rows = []
            for _, row in day_records.iterrows():
                name = str(row.get("Name") or "").strip()
                short_name = name if len(name) <= 34 else f"{name[:31]}..."
                price = row.get("Price")
                price_text = "" if pd.isna(price) else f"${float(price):,.2f}"
                table_rows.append(
                    f"""
                    <tr>
                        <td class='ticker-cell'>{escape(str(row.get("Ticker", "")))}</td>
                        <td title='{escape(name)}'>{escape(short_name)}</td>
                        <td>{escape(str(row.get("Date", "")))}</td>
                        <td>{escape(str(row.get("Time", "")))}</td>
                        <td class='price-cell'>{escape(price_text)}</td>
                    </tr>
                    """
                )
            table_body = "".join(table_rows)

        cards.append(
            f"""
            <section class='next-card'>
                <div class='next-card-head'>
                    <div>
                        <div class='weekday'>{escape(weekday)}</div>
                        <div class='date-label'>{escape(label)}</div>
                    </div>
                    <span>{len(day_records)}</span>
                </div>
                <div class='table-wrap'>
                    <table>
                        <thead>
                            <tr>
                                <th>Ticker</th>
                                <th>Name</th>
                                <th>Date</th>
                                <th>Time</th>
                                <th>Price</th>
                            </tr>
                        </thead>
                        <tbody>{table_body}</tbody>
                    </table>
                </div>
            </section>
            """
        )

    return f"""
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: #0d1b2e;
            font-family: Inter, Arial, sans-serif;
        }}
        .next-week {{
            color: #cbd5e1;
        }}
        .next-week h3 {{
            margin: 0 0 6px;
            color: #3ab54a;
            font-family: 'Courier Prime', monospace;
            font-size: 20px;
        }}
        .next-week .subhead {{
            margin: 0 0 14px;
            color: #94a3b8;
            font-size: 13px;
        }}
        .next-grid {{
            display: grid;
            grid-template-columns: repeat(7, minmax(210px, 1fr));
            gap: 10px;
            overflow-x: auto;
            padding-bottom: 6px;
        }}
        .next-card {{
            min-width: 210px;
            border: 1px solid #1e3a5f;
            background: rgba(21, 40, 71, 0.48);
            border-radius: 8px;
            overflow: hidden;
        }}
        .next-card-head {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            background: #152847;
            border-bottom: 1px solid #1e3a5f;
            padding: 9px 10px;
        }}
        .weekday {{
            color: #94a3b8;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .date-label {{
            color: #f8fafc;
            font-size: 14px;
            font-weight: 800;
        }}
        .next-card-head span {{
            color: #0d1b2e;
            background: #facc15;
            border-radius: 999px;
            padding: 2px 7px;
            font-size: 11px;
            font-weight: 800;
        }}
        .table-wrap {{
            max-height: 260px;
            overflow: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }}
        th, td {{
            border-bottom: 1px solid rgba(30, 58, 95, 0.8);
            padding: 7px 8px;
            text-align: left;
            vertical-align: top;
            color: #cbd5e1;
            font-size: 11px;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }}
        th {{
            position: sticky;
            top: 0;
            z-index: 1;
            background: #10213a;
            color: #94a3b8;
            font-size: 10px;
            text-transform: uppercase;
        }}
        .ticker-cell {{
            color: #f87171;
            font-weight: 800;
        }}
        .price-cell {{
            color: #86efac;
            font-weight: 700;
        }}
        .empty-row {{
            color: #64748b;
            text-align: center;
            padding: 28px 8px;
        }}
    </style>
    <div class='next-week'>
        <h3>Next 7 Trading Days Earnings</h3>
        <p class='subhead'>Whole-market companies reporting earnings on market trading days with latest price at or above ${min_price:,.2f}</p>
        <div class='next-grid'>{''.join(cards)}</div>
    </div>
    """


def fetch_earnings_dates(tickers: tuple[str, ...]) -> tuple[dict, pd.DataFrame, list[str]]:
    start, end = earnings_window()
    limit = 8
    window_dates = [day.strftime("%Y-%m-%d") for day in pd.date_range(start=start, end=end, freq="D")]
    requested_tickers = tuple(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip())
    tickers_to_fetch = tuple(missing_ticker_coverage(requested_tickers, window_dates, scope="universe"))
    fetched_rows = []
    failures = []
    seen = set()
    tickers_with_rows = set()

    def add_row(row: dict) -> None:
        key = (row["Date"], row["Ticker"])
        if key in seen:
            return
        seen.add(key)
        tickers_with_rows.add(row["Ticker"].upper())
        fetched_rows.append(row)

    for ticker in tickers_to_fetch:
        try:
            ticker_rows_before = len(fetched_rows)
            data = yf.Ticker(ticker).get_earnings_dates(limit=limit)
            if data is not None and not data.empty:
                for dt, values in data.iterrows():
                    ts = pd.Timestamp(dt)
                    event_day = date_only_timestamp(ts)
                    if event_day is None or not start <= event_day <= end:
                        continue

                    add_row(
                        {
                            "Date": event_day.strftime("%Y-%m-%d"),
                            "Ticker": ticker,
                            "Time": ts.strftime("%I:%M %p %Z"),
                            "EPS Estimate": values.get("EPS Estimate", None),
                            "Reported EPS": values.get("Reported EPS", None),
                            "Surprise %": values.get("Surprise(%)", None),
                            "Source": "yfinance earnings_dates",
                        }
                    )

            if len(fetched_rows) == ticker_rows_before:
                for row in calendar_fallback_rows(ticker, start, end):
                    add_row(row)
        except Exception:
            fallback_rows = calendar_fallback_rows(ticker, start, end)
            if fallback_rows:
                for row in fallback_rows:
                    add_row(row)
            else:
                failures.append(ticker)

    missing_tickers = tuple(ticker for ticker in tickers_to_fetch if ticker.upper() not in tickers_with_rows)
    if missing_tickers:
        for row in nasdaq_calendar_fallback_rows(missing_tickers, start, end):
            add_row(row)
        failures = [ticker for ticker in failures if ticker.upper() not in tickers_with_rows]

    if tickers_to_fetch:
        save_earnings_events(fetched_rows, scope="universe")
        mark_ticker_coverage(tickers_to_fetch, window_dates, scope="universe")

    records = load_earnings_events(requested_tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), scope="universe")
    earnings_map = defaultdict(list)
    if not records.empty:
        for _, row in records.iterrows():
            ticker = row["Ticker"]
            date_key = row["Date"]
            if ticker not in earnings_map[date_key]:
                earnings_map[date_key].append(ticker)

    clean_map = {date: sorted(symbols) for date, symbols in sorted(earnings_map.items())}
    if not records.empty:
        records = records.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    return clean_map, records, failures


def months_from_earnings_map(earnings_map: dict) -> list[tuple[int, int]]:
    months = set()
    for date_key in earnings_map:
        try:
            ts = pd.Timestamp(date_key)
        except Exception:
            continue
        months.add((ts.year, ts.month))
    return sorted(months)


def build_month_calendar_html(year: int, month: int, earnings_map: dict) -> str:
    month_name = calendar.month_name[month]
    weeks = calendar.monthcalendar(year, month)
    today_key = datetime.now().strftime("%Y-%m-%d")

    header_cells = "".join(f"<th>{day}</th>" for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    body_rows = []

    for week in weeks:
        cells = []
        for day in week:
            if day == 0:
                cells.append("<td class='empty'></td>")
                continue

            date_key = f"{year}-{month:02d}-{day:02d}"
            tickers = earnings_map.get(date_key, [])
            ticker_html = "".join(f"<div class='ticker'>{escape(ticker)}</div>" for ticker in tickers)
            today_class = " today" if date_key == today_key else ""
            count_badge = f"<span class='count'>{len(tickers)}</span>" if tickers else ""
            cells.append(
                f"""
                <td>
                    <div class='day-card{today_class}'>
                        <div class='day-head'><span>{day}</span>{count_badge}</div>
                        <div class='ticker-scroll'>{ticker_html}</div>
                    </div>
                </td>
                """
            )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: #0d1b2e;
            font-family: Inter, Arial, sans-serif;
        }}
        .earnings-month {{
            margin: 14px 0 26px;
        }}
        .earnings-month h3 {{
            margin: 0 0 10px;
            color: #3ab54a;
            font-family: 'Courier Prime', monospace;
            font-size: 20px;
        }}
        .earnings-calendar {{
            width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            background: #0d1b2e;
            border: 1px solid #1e3a5f;
        }}
        .earnings-calendar th {{
            background: #152847;
            color: #94a3b8;
            padding: 10px 6px;
            text-align: center;
            border: 1px solid #1e3a5f;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .earnings-calendar td {{
            border: 1px solid #1e3a5f;
            vertical-align: top;
            height: 124px;
            padding: 0;
            background: rgba(21, 40, 71, 0.35);
        }}
        .earnings-calendar td.empty {{
            background: rgba(13, 27, 46, 0.45);
        }}
        .day-card {{
            min-height: 124px;
            padding: 7px;
            box-sizing: border-box;
        }}
        .day-card.today {{
            background: rgba(26, 58, 143, 0.28);
            outline: 2px solid #3ab54a;
            outline-offset: -2px;
        }}
        .day-head {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #cbd5e1;
            font-weight: 700;
            font-size: 13px;
            margin-bottom: 5px;
        }}
        .count {{
            color: #0d1b2e;
            background: #facc15;
            border-radius: 999px;
            padding: 1px 6px;
            font-size: 10px;
        }}
        .ticker-scroll {{
            max-height: 88px;
            overflow-y: auto;
            scrollbar-width: thin;
            padding-right: 2px;
        }}
        .ticker {{
            color: #f87171;
            font-size: 12px;
            font-weight: 700;
            line-height: 1.25;
            margin-bottom: 3px;
            white-space: nowrap;
        }}
    </style>
    <div class='earnings-month'>
        <h3>{month_name} {year}</h3>
        <table class='earnings-calendar'>
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
        </table>
    </div>
    """


def month_calendar_height(year: int, month: int) -> int:
    week_count = len(calendar.monthcalendar(year, month))
    return 92 + (week_count * 124)


def render_database_status():
    summary = get_database_summary()
    with st.expander("Earnings Database Status", expanded=False):
        st.caption(summary["db_path"])
        if summary.get("warning"):
            st.warning(summary["warning"])
        elif summary.get("configured_env_path"):
            st.success("Using EARNINGS_CALENDAR_DB_PATH.")
        else:
            st.info("Using the default local development database path.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Saved Events", f"{summary['event_count']:,}")
        c2.metric("Saved Tickers", f"{summary['ticker_count']:,}")
        c3.metric("Ticker Coverage Rows", f"{summary['ticker_coverage_count']:,}")
        c4.metric("Market Days Covered", f"{summary['market_coverage_count']:,}")

        if summary.get("latest_fetch"):
            st.caption(f"Latest database write: {summary['latest_fetch']}")

        tab_tickers, tab_events, tab_coverage = st.tabs(["Tickers", "Recent Events", "Coverage"])
        with tab_tickers:
            saved_tickers = get_saved_tickers()
            if saved_tickers.empty:
                st.info("No saved earnings tickers yet.")
            else:
                st.dataframe(saved_tickers, use_container_width=True, hide_index=True)

        with tab_events:
            recent_events = get_recent_events(limit=100)
            if recent_events.empty:
                st.info("No saved earnings events yet.")
            else:
                st.dataframe(recent_events, use_container_width=True, hide_index=True)

        with tab_coverage:
            coverage = get_coverage_sample(limit=100)
            if coverage.empty:
                st.info("No ticker coverage rows yet.")
            else:
                st.dataframe(coverage, use_container_width=True, hide_index=True)


class EarningsCalendarModule(FazDaneModule):
    MODULE_NAME = "Earnings Calendar"
    MODULE_ICON = "📺"
    MODULE_DESCRIPTION = "Upcoming and historical earnings dates by selected ticker universe"
    TIER = 2
    SOURCE_NOTEBOOK = "Colab Earnings Calendar"
    CACHE_TTL = 21600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def render_sidebar(self):
        st.markdown("**Earnings Universe**")
        self.universe_name, tickers, _ = render_universe_manager(
            key_prefix="earnings",
            show_benchmark=False,
            label="Ticker Universe:",
        )
        self.tickers = tickers
        st.caption(f"{len(self.tickers)} tickers selected from {self.universe_name}.")

        st.markdown("**Calendar Settings**")
        self.window_start, self.window_end = earnings_window()
        st.caption(
            f"Rolling window: {self.window_start.strftime('%Y-%m-%d')} to {self.window_end.strftime('%Y-%m-%d')}."
        )
        self.show_table = st.checkbox("Show detail table", value=True, key="earnings_show_table")
        st.markdown("**Next 7 Days Filter**")
        self.next_week_min_price = float(
            st.number_input(
                "Minimum stock price:",
                min_value=0.0,
                max_value=10000.0,
                value=100.0,
                step=5.0,
                key="earnings_next_week_min_price",
            )
        )

        if st.button("Refresh Earnings", use_container_width=True, type="primary", key="earnings_refresh"):
            st.cache_data.clear()
            st.rerun()

    def render_next_week_panel(self):
        with st.spinner("Building next 7 days earnings price filter..."):
            next_week_records = fetch_next_week_earnings_by_price(self.next_week_min_price)
        components.html(
            build_next_week_earnings_html(next_week_records, self.next_week_min_price),
            height=390,
            scrolling=False,
        )

    def render_main(self):
        self.render_section_header(
            "📺 Earnings Calendar",
            "Calendar view of earnings dates from the selected ticker universe",
        )

        if not self.tickers:
            st.warning("Select or create a ticker universe to build the earnings calendar.")
            self.render_next_week_panel()
            return

        with st.spinner(f"Fetching rolling earnings window for {len(self.tickers)} tickers..."):
            earnings_map, records, failures = fetch_earnings_dates(tuple(self.tickers))
        portfolio_tickers = get_tickers("FazDane Portfolio")
        portfolio_records = pd.DataFrame()
        portfolio_failures = []
        if portfolio_tickers:
            with st.spinner(f"Fetching FazDane Portfolio earnings for {len(portfolio_tickers)} tickers..."):
                _, portfolio_records, portfolio_failures = fetch_earnings_dates(tuple(portfolio_tickers))

        total_events = sum(len(tickers) for tickers in earnings_map.values())
        active_months = months_from_earnings_map(earnings_map)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Universe", self.universe_name)
        c2.metric("Tickers", str(len(self.tickers)))
        c3.metric("Earnings Events", str(total_events))
        c4.metric("Window", "10D back / 30D ahead")
        render_database_status()

        if failures:
            with st.expander(f"Tickers skipped by yfinance ({len(failures)})", expanded=False):
                st.write(", ".join(sorted(failures)))

        if not earnings_map:
            st.warning("No earnings dates were returned for this universe in the rolling window.")
            self.render_next_week_panel()
            return

        for year, month in active_months:
            components.html(
                build_month_calendar_html(year, month, earnings_map),
                height=month_calendar_height(year, month),
                scrolling=False,
            )

        self.render_next_week_panel()

        st.markdown("### FazDane Portfolio Earnings")
        if portfolio_records.empty:
            st.info("No FazDane Portfolio earnings dates were returned for the rolling window.")
        else:
            portfolio_display = portfolio_records.copy()
            portfolio_display["Portfolio"] = "FazDane Portfolio"
            portfolio_display = portfolio_display[
                ["Portfolio", "Date", "Ticker", "Time", "EPS Estimate", "Reported EPS", "Surprise %"]
            ]
            st.dataframe(portfolio_display, use_container_width=True, hide_index=True)
            st.download_button(
                "Download FazDane Portfolio Earnings CSV",
                data=portfolio_display.to_csv(index=False),
                file_name="fazdane_portfolio_earnings_rolling_window.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if portfolio_failures:
            with st.expander(f"FazDane Portfolio tickers skipped by yfinance ({len(portfolio_failures)})", expanded=False):
                st.write(", ".join(sorted(portfolio_failures)))

        if self.show_table and not records.empty:
            st.markdown("### Earnings Detail")
            st.dataframe(records, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Earnings CSV",
                data=records.to_csv(index=False),
                file_name=f"earnings_{self.universe_name.replace(' ', '_').lower()}_rolling_window.csv",
                mime="text/csv",
                use_container_width=True,
            )
