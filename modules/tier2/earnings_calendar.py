"""
FazDane Analytics - Tier 2
Earnings Calendar
"""

import calendar
from collections import defaultdict
from datetime import date, datetime
from html import escape

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

from modules.base_module import FazDaneModule
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


def calendar_fallback_rows(ticker: str, year: int) -> list[dict]:
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
        if ts.year != year:
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


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_earnings_dates(tickers: tuple[str, ...], year: int, limit: int) -> tuple[dict, pd.DataFrame, list[str]]:
    earnings_map = defaultdict(list)
    rows = []
    failures = []
    seen = set()

    def add_row(row: dict) -> None:
        key = (row["Date"], row["Ticker"])
        if key in seen:
            return
        seen.add(key)
        if row["Ticker"] not in earnings_map[row["Date"]]:
            earnings_map[row["Date"]].append(row["Ticker"])
        rows.append(row)

    for ticker in tickers:
        try:
            ticker_rows_before = len(rows)
            data = yf.Ticker(ticker).get_earnings_dates(limit=limit)
            if data is not None and not data.empty:
                for dt, values in data.iterrows():
                    ts = pd.Timestamp(dt)
                    if pd.isna(ts) or ts.year != year:
                        continue

                    add_row(
                        {
                            "Date": ts.strftime("%Y-%m-%d"),
                            "Ticker": ticker,
                            "Time": ts.strftime("%I:%M %p %Z"),
                            "EPS Estimate": values.get("EPS Estimate", None),
                            "Reported EPS": values.get("Reported EPS", None),
                            "Surprise %": values.get("Surprise(%)", None),
                            "Source": "yfinance earnings_dates",
                        }
                    )

            if len(rows) == ticker_rows_before:
                for row in calendar_fallback_rows(ticker, year):
                    add_row(row)
        except Exception:
            fallback_rows = calendar_fallback_rows(ticker, year)
            if fallback_rows:
                for row in fallback_rows:
                    add_row(row)
            else:
                failures.append(ticker)

    clean_map = {date: sorted(symbols) for date, symbols in sorted(earnings_map.items())}
    records = pd.DataFrame(rows)
    if not records.empty:
        records = records.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    return clean_map, records, failures


def month_has_data(year: int, month: int, earnings_map: dict) -> bool:
    prefix = f"{year}-{month:02d}-"
    return any(date.startswith(prefix) for date in earnings_map)


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
        current_year = datetime.now().year
        self.year = int(
            st.number_input(
                "Earnings Year:",
                min_value=current_year - 2,
                max_value=current_year + 2,
                value=current_year,
                step=1,
                key="earnings_year",
            )
        )
        self.limit = int(
            st.slider(
                "Dates to request per ticker:",
                min_value=4,
                max_value=40,
                value=20,
                step=2,
                key="earnings_limit",
            )
        )
        self.month_filter = st.selectbox(
            "Months to Display:",
            ["Current month", "Only months with earnings", "All months"],
            index=0,
            key="earnings_month_filter",
        )
        current_month_index = datetime.now().month - 1
        self.selected_month = st.selectbox(
            "Jump to Month:",
            list(calendar.month_name)[1:],
            index=current_month_index,
            key="earnings_selected_month",
        )
        self.show_table = st.checkbox("Show detail table", value=True, key="earnings_show_table")

        if st.button("Refresh Earnings", use_container_width=True, type="primary", key="earnings_refresh"):
            fetch_earnings_dates.clear()
            st.rerun()

    def render_main(self):
        self.render_section_header(
            "📺 Earnings Calendar",
            "Calendar view of earnings dates from the selected ticker universe",
        )

        if not self.tickers:
            st.warning("Select or create a ticker universe to build the earnings calendar.")
            return

        with st.spinner(f"Fetching earnings dates for {len(self.tickers)} tickers..."):
            earnings_map, records, failures = fetch_earnings_dates(tuple(self.tickers), self.year, self.limit)
        portfolio_tickers = get_tickers("FazDane Portfolio")
        portfolio_records = pd.DataFrame()
        portfolio_failures = []
        if portfolio_tickers:
            with st.spinner(f"Fetching FazDane Portfolio earnings for {len(portfolio_tickers)} tickers..."):
                _, portfolio_records, portfolio_failures = fetch_earnings_dates(
                    tuple(portfolio_tickers),
                    self.year,
                    self.limit,
                )

        total_events = sum(len(tickers) for tickers in earnings_map.values())
        active_months = [m for m in range(1, 13) if month_has_data(self.year, m, earnings_map)]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Universe", self.universe_name)
        c2.metric("Tickers", str(len(self.tickers)))
        c3.metric("Earnings Events", str(total_events))
        c4.metric("Months With Data", str(len(active_months)))

        if failures:
            with st.expander(f"Tickers skipped by yfinance ({len(failures)})", expanded=False):
                st.write(", ".join(sorted(failures)))

        if not earnings_map:
            st.warning("No earnings dates were returned for this universe and year.")
            return

        if self.month_filter == "Current month":
            months = [list(calendar.month_name).index(self.selected_month)]
        elif self.month_filter == "All months":
            months = range(1, 13)
        else:
            months = active_months

        for month in months:
            components.html(
                build_month_calendar_html(self.year, month, earnings_map),
                height=month_calendar_height(self.year, month),
                scrolling=False,
            )

        st.markdown("### FazDane Portfolio Earnings")
        if portfolio_records.empty:
            st.info("No FazDane Portfolio earnings dates were returned for the selected year.")
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
                file_name=f"fazdane_portfolio_earnings_{self.year}.csv",
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
                file_name=f"earnings_{self.universe_name.replace(' ', '_').lower()}_{self.year}.csv",
                mime="text/csv",
                use_container_width=True,
            )
