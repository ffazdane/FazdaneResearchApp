"""
FazDane Analytics - Tier 2
Equity / Index Seasonality Analysis
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar

from modules.base_module import FazDaneModule
from utils.universe_manager import format_ticker_display, get_ticker_names, get_universe_names, render_universe_manager


MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

TICKER_ALIASES = {
    "SPX": "^GSPC",
    "NDX": "^NDX",
    "RUT": "^RUT",
}


def normalize_symbol(symbol: str) -> str:
    clean = symbol.strip().upper()
    return TICKER_ALIASES.get(clean, clean)


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_seasonality_data(ticker: str, years: int) -> pd.DataFrame:
    import sqlite3
    from utils.persistence import get_db_path
    
    symbol = ticker.strip().upper()
    aliases = {"^GSPC": "SPX", "SPX": "SPX", "^NDX": "NDX", "NDX": "NDX", "^RUT": "RUT", "RUT": "RUT", "^VIX": "VIX", "VIX": "VIX", "^DJI": "DJI", "DJI": "DJI"}
    resolved_symbol = aliases.get(symbol, symbol)
    
    # Try reading from SQLite first
    try:
        db_path = get_db_path("options_liquidity")
        if db_path.exists():
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM daily_prices WHERE symbol = ?", (resolved_symbol,))
                if cursor.fetchone()[0] > 0:
                    query = "SELECT date as Date, close as Close FROM daily_prices WHERE symbol = ? ORDER BY date"
                    df = pd.read_sql_query(query, conn)
                    if not df.empty:
                        df["Date"] = pd.to_datetime(df["Date"])
                        df["Close"] = pd.to_numeric(df["Close"])
                        df["DailyPctChange"] = df["Close"].pct_change() * 100
                        df = df.dropna(subset=["DailyPctChange"]).reset_index(drop=True)
                        df["DateStr"] = df["Date"].dt.strftime("%Y-%m-%d")
                        df["Year"] = df["Date"].dt.year
                        df["MonthNum"] = df["Date"].dt.month
                        df["Month"] = pd.Categorical(df["Date"].dt.month_name(), categories=MONTH_ORDER, ordered=True)
                        df["DayName"] = df["Date"].dt.day_name()
                        
                        # Filter by lookback years
                        end_date = datetime.today()
                        start_date = end_date - timedelta(days=int(years * 365.25) + 10)
                        df = df[df["Date"] >= start_date].reset_index(drop=True)
                        return df
    except Exception as e:
        pass

    # Fallback to yfinance
    symbol = normalize_symbol(ticker)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=int(years * 365.25) + 10)
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join(str(part) for part in col if part).strip() for col in data.columns]

    close_cols = [col for col in data.columns if "Close" in str(col)]
    if not close_cols:
        return pd.DataFrame()

    close_col = close_cols[0]
    df = data[[close_col]].copy()
    df.columns = ["Close"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["DailyPctChange"] = df["Close"].pct_change() * 100
    df = df.dropna(subset=["DailyPctChange"]).reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df["DateStr"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["Year"] = df["Date"].dt.year
    df["MonthNum"] = df["Date"].dt.month
    df["Month"] = pd.Categorical(df["Date"].dt.month_name(), categories=MONTH_ORDER, ordered=True)
    df["DayName"] = df["Date"].dt.day_name()
    return df.sort_values("Date").reset_index(drop=True)



def monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby("Month", observed=False).agg(
        Avg_DailyPctChange=("DailyPctChange", "mean"),
        Median_DailyPctChange=("DailyPctChange", "median"),
        Max_DailyPctChange=("DailyPctChange", "max"),
        Min_DailyPctChange=("DailyPctChange", "min"),
        Volatility=("DailyPctChange", "std"),
        Count_Positive=("DailyPctChange", lambda x: (x >= 0).sum()),
        Count_Negative=("DailyPctChange", lambda x: (x < 0).sum()),
        Total_Days=("DailyPctChange", "count"),
    ).reset_index()
    stats["Pct_Positive"] = np.where(
        stats["Total_Days"] > 0,
        100 * stats["Count_Positive"] / stats["Total_Days"],
        0,
    )
    return stats.sort_values("Month")


def event_stats(events: pd.DataFrame) -> pd.Series:
    if events.empty:
        return pd.Series(
            {"Events": 0, "Avg %": 0.0, "Median %": 0.0, "Win Rate": 0.0, "Best %": 0.0, "Worst %": 0.0}
        )
    returns = events["ReturnPct"].dropna()
    return pd.Series(
        {
            "Events": len(returns),
            "Avg %": returns.mean(),
            "Median %": returns.median(),
            "Win Rate": (returns > 0).mean() * 100 if len(returns) else 0,
            "Best %": returns.max() if len(returns) else 0,
            "Worst %": returns.min() if len(returns) else 0,
        }
    )


def grouped_return_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    stats = df.groupby(group_col, observed=False).agg(
        Avg_ReturnPct=("DailyPctChange", "mean"),
        Median_ReturnPct=("DailyPctChange", "median"),
        Win_Rate=("DailyPctChange", lambda x: (x > 0).mean() * 100 if len(x) else 0),
        Best_ReturnPct=("DailyPctChange", "max"),
        Worst_ReturnPct=("DailyPctChange", "min"),
        Events=("DailyPctChange", "count"),
    ).reset_index()
    return stats


def election_cycle_label(year: int) -> str:
    cycle = int(year) % 4
    if cycle == 0:
        return "Election Year"
    if cycle == 1:
        return "Post-Election Year"
    if cycle == 2:
        return "Midterm Year"
    return "Pre-Election Year"


def annual_election_cycle_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    yearly = (
        df.sort_values("Date")
        .groupby("Year", as_index=False)
        .agg(
            StartDate=("Date", "first"),
            EndDate=("Date", "last"),
            StartClose=("Close", "first"),
            EndClose=("Close", "last"),
            TradingDays=("DailyPctChange", "count"),
        )
    )
    if yearly.empty:
        return pd.DataFrame(), pd.DataFrame()

    yearly["YearReturnPct"] = (yearly["EndClose"] / yearly["StartClose"] - 1) * 100
    yearly["ElectionCycle"] = yearly["Year"].map(election_cycle_label)
    yearly = yearly.sort_values("Year", ascending=False).reset_index(drop=True)

    summary = yearly.groupby("ElectionCycle", as_index=False).agg(
        Avg_YearReturnPct=("YearReturnPct", "mean"),
        Median_YearReturnPct=("YearReturnPct", "median"),
        Win_Rate=("YearReturnPct", lambda x: (x > 0).mean() * 100 if len(x) else 0),
        Best_YearReturnPct=("YearReturnPct", "max"),
        Worst_YearReturnPct=("YearReturnPct", "min"),
        Years=("YearReturnPct", "count"),
    )
    order = ["Election Year", "Post-Election Year", "Midterm Year", "Pre-Election Year"]
    summary["ElectionCycle"] = pd.Categorical(summary["ElectionCycle"], categories=order, ordered=True)
    summary = summary.sort_values("ElectionCycle").reset_index(drop=True)
    return yearly, summary


def first_last_trading_days(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = df.copy()
    monthly["Period"] = monthly["Date"].dt.to_period("M")
    first = monthly.groupby("Period", as_index=False).first()
    last = monthly.groupby("Period", as_index=False).last()
    first["Event"] = "First Trading Day"
    last["Event"] = "Last Trading Day"
    for frame in (first, last):
        frame["ReturnPct"] = frame["DailyPctChange"]
        frame["DateStr"] = frame["Date"].dt.strftime("%Y-%m-%d")
        frame["Month"] = frame["Date"].dt.month_name()
    return first, last


def first_last_monthly_stats(first_days: pd.DataFrame, last_days: pd.DataFrame) -> pd.DataFrame:
    events = pd.concat([first_days, last_days], ignore_index=True)
    if events.empty:
        return pd.DataFrame()

    stats = events.groupby(["Month", "Event"], observed=False).agg(
        Avg_ReturnPct=("ReturnPct", "mean"),
        Median_ReturnPct=("ReturnPct", "median"),
        Win_Rate=("ReturnPct", lambda x: (x > 0).mean() * 100 if len(x) else 0),
        Best_ReturnPct=("ReturnPct", "max"),
        Worst_ReturnPct=("ReturnPct", "min"),
        Events=("ReturnPct", "count"),
    ).reset_index()
    stats["Month"] = pd.Categorical(stats["Month"], categories=MONTH_ORDER, ordered=True)
    return stats.sort_values(["Month", "Event"])


def holiday_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cal = USFederalHolidayCalendar()
    start = start - pd.Timedelta(days=7)
    end = end + pd.Timedelta(days=7)
    try:
        holidays = cal.holidays(start=start, end=end, return_name=True)
        rows = [{"Date": pd.Timestamp(date), "Holiday": name} for date, name in holidays.items()]
    except TypeError:
        rows = []
        for rule in cal.rules:
            for date in rule.dates(start, end):
                rows.append({"Date": pd.Timestamp(date), "Holiday": rule.name})
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def detect_long_weekends(df: pd.DataFrame) -> pd.DataFrame:
    dates = df["Date"].sort_values().reset_index(drop=True)
    holidays = holiday_calendar(dates.min(), dates.max())
    events = []
    for i in range(len(dates) - 1):
        before = dates.iloc[i]
        after = dates.iloc[i + 1]
        gap_days = (after - before).days
        if gap_days < 3:
            continue
        holiday_hits = holidays[(holidays["Date"] > before) & (holidays["Date"] < after)]
        if gap_days < 4 and holiday_hits.empty:
            continue
        holiday_name = holiday_hits["Holiday"].iloc[0] if not holiday_hits.empty else "Market Long Weekend"
        events.append(
            {
                "Holiday": holiday_name,
                "LastTradingDayBefore": before,
                "FirstTradingDayAfter": after,
                "GapDays": gap_days,
            }
        )
    return pd.DataFrame(events)


def long_weekend_windows(df: pd.DataFrame, events: pd.DataFrame, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    indexed = df.set_index("Date")
    rows = []
    summaries = []

    for _, event in events.iterrows():
        before_date = event["LastTradingDayBefore"]
        after_date = event["FirstTradingDayAfter"]
        before_idx = df.index[df["Date"] == before_date]
        after_idx = df.index[df["Date"] == after_date]
        if len(before_idx) == 0 or len(after_idx) == 0:
            continue

        before_pos = int(before_idx[0])
        after_pos = int(after_idx[0])
        for offset in range(-window, window + 1):
            pos = before_pos + offset if offset <= 0 else after_pos + offset - 1
            if pos < 0 or pos >= len(df):
                continue
            row = df.iloc[pos]
            rows.append(
                {
                    "Holiday": event["Holiday"],
                    "AnchorDate": before_date.strftime("%Y-%m-%d"),
                    "TradingDayOffset": offset,
                    "Date": row["Date"],
                    "DateStr": row["DateStr"],
                    "ReturnPct": row["DailyPctChange"],
                    "Close": row["Close"],
                }
            )

        pre_start = max(before_pos - window + 1, 0)
        pre = indexed.loc[df.iloc[pre_start]["Date"]:before_date]
        post_end = min(after_pos + window - 1, len(df) - 1)
        post = indexed.loc[after_date:df.iloc[post_end]["Date"]]
        pre_return = (pre["Close"].iloc[-1] / pre["Close"].iloc[0] - 1) * 100 if len(pre) > 1 else np.nan
        post_return = (post["Close"].iloc[-1] / post["Close"].iloc[0] - 1) * 100 if len(post) > 1 else np.nan
        summaries.append(
            {
                "Holiday": event["Holiday"],
                "Before Date": before_date.strftime("%Y-%m-%d"),
                "After Date": after_date.strftime("%Y-%m-%d"),
                f"{window}D Before Return %": pre_return,
                f"{window}D After Return %": post_return,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(summaries)


def plot_monthly_scatter(df: pd.DataFrame, ticker: str):
    chart_df = df.sort_values("Month")
    fig = px.scatter(
        chart_df,
        x="Month",
        y="DailyPctChange",
        color=np.where(chart_df["DailyPctChange"] >= 0, "Positive", "Negative"),
        color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444"},
        custom_data=["DateStr", "DailyPctChange", "Close"],
        category_orders={"Month": MONTH_ORDER},
        title=f"Daily % Change by Month - {ticker}",
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Date: %{customdata[0]}<br>Daily Change: %{customdata[1]:.2f}%<br>Close: %{customdata[2]:.2f}<extra></extra>"
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", height=520)
    return fig


def plot_monthly_bars(stats: pd.DataFrame):
    fig = px.bar(
        stats,
        x="Month",
        y="Avg_DailyPctChange",
        color="Avg_DailyPctChange",
        color_continuous_scale=["#ef4444", "#facc15", "#22c55e"],
        category_orders={"Month": MONTH_ORDER},
        title="Average Daily % Change by Month",
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", height=420)
    return fig


def plot_pos_neg(stats: pd.DataFrame):
    melt = stats[["Month", "Count_Positive", "Count_Negative"]].melt(
        id_vars="Month", var_name="Type", value_name="Count"
    )
    fig = px.bar(
        melt,
        x="Month",
        y="Count",
        color="Type",
        barmode="stack",
        color_discrete_map={"Count_Positive": "#22c55e", "Count_Negative": "#ef4444"},
        category_orders={"Month": MONTH_ORDER},
        title="Positive vs Negative Days by Month",
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", height=420)
    return fig


class SeasonalityAnalysisModule(FazDaneModule):
    MODULE_NAME = "Equity / Index Seasonality"
    MODULE_ICON = "📅"
    MODULE_DESCRIPTION = "Seasonality, first/last trading day, and long-weekend anomaly analytics"
    TIER = 2
    SOURCE_NOTEBOOK = "Colab Daily Percentage Change Seasonal Scatter"
    CACHE_TTL = 21600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def render_sidebar(self):
        self._default_index_universe()
        st.markdown("**Seasonality Universe**")
        self.universe_name, tickers, _ = render_universe_manager(
            key_prefix="seasonality",
            show_benchmark=False,
            label="Ticker Universe:",
        )
        ticker_names = get_ticker_names(self.universe_name)
        self.tickers = tickers
        if self.tickers:
            default = self.tickers.index("^GSPC") if "^GSPC" in self.tickers else 0
            if st.session_state.get("seasonality_ticker") not in self.tickers:
                st.session_state["seasonality_ticker"] = self.tickers[default]
            self.ticker = st.selectbox(
                "Ticker / Index:",
                self.tickers,
                index=default,
                key="seasonality_ticker",
                format_func=lambda ticker: format_ticker_display(ticker, ticker_names),
            )
        else:
            self.ticker = ""

        self.lookback_years = int(st.slider("Lookback Years:", 3, 15, 7, key="seasonality_years"))
        self.long_weekend_window = 5
        st.caption("Holiday window: 5 trading days before and after")
        self.show_raw = st.checkbox("Show event detail tables", value=True, key="seasonality_raw")

        if st.button("Refresh Seasonality", width="stretch", type="primary", key="seasonality_refresh"):
            fetch_seasonality_data.clear()
            st.rerun()

    def render_main(self):
        self.render_section_header(
            "📅 Equity / Index Seasonality",
            "Monthly return behavior, first/last trading day effects, and long-weekend anomaly windows",
        )

        if not self.ticker:
            st.warning("Select or create a ticker universe, then choose one ticker or index.")
            return

        symbol = normalize_symbol(self.ticker)
        with st.spinner(f"Fetching {self.lookback_years} years of daily data for {symbol}..."):
            df = fetch_seasonality_data(symbol, self.lookback_years)

        if df.empty:
            st.warning(f"No daily close data returned for {symbol}.")
            return

        stats = monthly_stats(df)
        first_days, last_days = first_last_trading_days(df)
        edge_monthly = first_last_monthly_stats(first_days, last_days)
        long_weekends = detect_long_weekends(df)
        lw_rows, lw_summary = long_weekend_windows(df, long_weekends, self.long_weekend_window)
        if not lw_summary.empty:
            lw_summary = lw_summary.sort_values("Before Date", ascending=False).reset_index(drop=True)
        if not lw_rows.empty:
            lw_rows = lw_rows.sort_values(["AnchorDate", "TradingDayOffset"], ascending=[False, True]).reset_index(drop=True)

        best_month = stats.loc[stats["Avg_DailyPctChange"].idxmax()]
        worst_month = stats.loc[stats["Avg_DailyPctChange"].idxmin()]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", symbol)
        c2.metric("Trading Days", f"{len(df):,}")
        c3.metric("Best Month", f"{best_month['Month']}", f"{best_month['Avg_DailyPctChange']:.2f}% avg")
        c4.metric("Worst Month", f"{worst_month['Month']}", f"{worst_month['Avg_DailyPctChange']:.2f}% avg")

        tab_month, tab_cycles, tab_edge, tab_weekend, tab_dist, tab_data = st.tabs(
            ["Monthly", "Calendar Cycles", "First/Last Day", "Long Weekends", "Distribution Study", "Data"]
        )

        with tab_month:
            st.plotly_chart(plot_monthly_scatter(df, symbol), width="stretch")
            left, right = st.columns(2)
            with left:
                st.plotly_chart(plot_monthly_bars(stats), width="stretch")
            with right:
                fig_box = px.box(
                    df,
                    x="Month",
                    y="DailyPctChange",
                    points="all",
                    category_orders={"Month": MONTH_ORDER},
                    title="Daily % Change Distribution by Month",
                )
                fig_box.update_layout(template="plotly_dark", paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", height=420)
                st.plotly_chart(fig_box, width="stretch")
            st.plotly_chart(plot_pos_neg(stats), width="stretch")
            st.dataframe(stats.round(2), width="stretch", hide_index=True)

        with tab_cycles:
            cycle_df = df.copy()
            cycle_df["Weekday"] = pd.Categorical(cycle_df["DayName"], categories=WEEKDAY_ORDER, ordered=True)
            cycle_df["ISOWeek"] = cycle_df["Date"].dt.isocalendar().week.astype(int)
            cycle_df["Quarter"] = "Q" + cycle_df["Date"].dt.quarter.astype(str)

            weekday_stats = grouped_return_stats(cycle_df, "Weekday").sort_values("Weekday")
            current_week = int(pd.Timestamp.today().isocalendar().week)
            weekly_stats = grouped_return_stats(cycle_df[cycle_df["ISOWeek"] <= 52], "ISOWeek").sort_values("ISOWeek")
            weekly_stats["Current"] = np.where(weekly_stats["ISOWeek"] == current_week, "Current Week", "Other Weeks")

            # Calculate Historical, Last Year, and Current Year parameters
            current_year = int(cycle_df["Year"].max())
            last_year = current_year - 1
            has_multiple_years = (cycle_df["Year"].nunique() >= 2) and (last_year in cycle_df["Year"].values)

            if has_multiple_years:
                hist_df = cycle_df[(cycle_df["Year"] < last_year) & (cycle_df["ISOWeek"] <= 52)]
                ly_df = cycle_df[(cycle_df["Year"] == last_year) & (cycle_df["ISOWeek"] <= 52)]
                cy_df = cycle_df[(cycle_df["Year"] == current_year) & (cycle_df["ISOWeek"] <= 52)]

                hist_weekly = grouped_return_stats(hist_df, "ISOWeek").sort_values("ISOWeek") if not hist_df.empty else pd.DataFrame(columns=["ISOWeek", "Avg_ReturnPct", "Win_Rate", "Events"])
                ly_weekly = grouped_return_stats(ly_df, "ISOWeek").sort_values("ISOWeek") if not ly_df.empty else pd.DataFrame(columns=["ISOWeek", "Avg_ReturnPct", "Win_Rate", "Events"])
                cy_weekly = grouped_return_stats(cy_df, "ISOWeek").sort_values("ISOWeek") if not cy_df.empty else pd.DataFrame(columns=["ISOWeek", "Avg_ReturnPct", "Win_Rate", "Events"])

                # Align to ISOWeeks 1-52
                weeks_df = pd.DataFrame({"ISOWeek": range(1, 53)})
                hist_aligned = pd.merge(weeks_df, hist_weekly, on="ISOWeek", how="left")
                ly_aligned = pd.merge(weeks_df, ly_weekly, on="ISOWeek", how="left")
                cy_aligned = pd.merge(weeks_df, cy_weekly, on="ISOWeek", how="left")

                # Calculate Trend Seasonality Alignment KPIs
                merge_hist = pd.merge(hist_weekly[["ISOWeek", "Avg_ReturnPct"]], cy_weekly[["ISOWeek", "Avg_ReturnPct"]], on="ISOWeek", suffixes=("_hist", "_cy")).dropna()
                if not merge_hist.empty:
                    merge_hist["Match"] = np.sign(merge_hist["Avg_ReturnPct_hist"]) == np.sign(merge_hist["Avg_ReturnPct_cy"])
                    hist_matches = int(merge_hist["Match"].sum())
                    hist_total = len(merge_hist)
                    hist_alignment_pct = (hist_matches / hist_total) * 100
                    hist_corr = merge_hist["Avg_ReturnPct_hist"].corr(merge_hist["Avg_ReturnPct_cy"])
                else:
                    hist_alignment_pct = 0.0
                    hist_corr = 0.0
                    hist_matches = 0
                    hist_total = 0

                merge_ly = pd.merge(ly_weekly[["ISOWeek", "Avg_ReturnPct"]], cy_weekly[["ISOWeek", "Avg_ReturnPct"]], on="ISOWeek", suffixes=("_ly", "_cy")).dropna()
                if not merge_ly.empty:
                    merge_ly["Match"] = np.sign(merge_ly["Avg_ReturnPct_ly"]) == np.sign(merge_ly["Avg_ReturnPct_cy"])
                    ly_matches = int(merge_ly["Match"].sum())
                    ly_total = len(merge_ly)
                    ly_alignment_pct = (ly_matches / ly_total) * 100
                    ly_corr = merge_ly["Avg_ReturnPct_ly"].corr(merge_ly["Avg_ReturnPct_cy"])
                else:
                    ly_alignment_pct = 0.0
                    ly_corr = 0.0
                    ly_matches = 0
                    ly_total = 0

                curr_wk_hist_match = None
                curr_wk_ly_match = None
                if current_week in merge_hist["ISOWeek"].values:
                    curr_wk_hist_match = bool(merge_hist[merge_hist["ISOWeek"] == current_week]["Match"].values[0])
                if current_week in merge_ly["ISOWeek"].values:
                    curr_wk_ly_match = bool(merge_ly[merge_ly["ISOWeek"] == current_week]["Match"].values[0])

            monthly_perf = stats[[
                "Month", "Avg_DailyPctChange", "Median_DailyPctChange", "Pct_Positive",
                "Max_DailyPctChange", "Min_DailyPctChange", "Total_Days",
            ]].copy()
            monthly_perf.rename(
                columns={
                    "Avg_DailyPctChange": "Avg_ReturnPct",
                    "Median_DailyPctChange": "Median_ReturnPct",
                    "Pct_Positive": "Win_Rate",
                    "Max_DailyPctChange": "Best_ReturnPct",
                    "Min_DailyPctChange": "Worst_ReturnPct",
                    "Total_Days": "Events",
                },
                inplace=True,
            )

            yearly_cycle, cycle_summary = annual_election_cycle_stats(cycle_df)
            quarter_stats = grouped_return_stats(cycle_df, "Quarter").sort_values("Quarter")
            sub_tabs = st.tabs([
                "Weekday Performance",
                "Weekly Performance",
                "Monthly Performance",
                "Election & Quarter Cycles"
            ])

            with sub_tabs[0]:
                st.markdown("### Average Day Performance")
                fig_weekday = px.bar(
                    weekday_stats,
                    x="Weekday",
                    y="Avg_ReturnPct",
                    color="Avg_ReturnPct",
                    color_continuous_scale=["#ef4444", "#facc15", "#22c55e"],
                    category_orders={"Weekday": WEEKDAY_ORDER},
                    hover_data={"Median_ReturnPct": ":.2f", "Win_Rate": ":.1f", "Events": True},
                    title="Average Daily Return by Weekday",
                )
                fig_weekday.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
                fig_weekday.update_layout(template="plotly_dark", paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", height=380)
                st.plotly_chart(fig_weekday, width="stretch")
                st.dataframe(weekday_stats.round(2), width="stretch", hide_index=True)

            with sub_tabs[1]:
                st.markdown("### Average Week Performance")
                if has_multiple_years:
                    st.markdown("#### 📊 Trend Seasonality Alignment KPIs")
                    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
                    
                    with col_kpi1:
                        hist_corr_str = f"{hist_corr:+.2f}" if not pd.isna(hist_corr) else "N/A"
                        st.metric(
                            label="Historical Seasonality Alignment",
                            value=f"{hist_alignment_pct:.1f}%",
                            delta=f"Pearson Corr: {hist_corr_str}",
                            help="Percentage of weeks this year where the return direction (up/down) matches the long-term historical average (prior to last year)."
                        )
                        st.caption(f"{hist_matches} of {hist_total} weeks matched")
                        
                    with col_kpi2:
                        ly_corr_str = f"{ly_corr:+.2f}" if not pd.isna(ly_corr) else "N/A"
                        st.metric(
                            label=f"Prior Year ({last_year}) Alignment",
                            value=f"{ly_alignment_pct:.1f}%",
                            delta=f"Pearson Corr: {ly_corr_str}",
                            help=f"Percentage of weeks this year where the return direction (up/down) matches the previous year ({last_year})."
                        )
                        st.caption(f"{ly_matches} of {ly_total} weeks matched")
                        
                    with col_kpi3:
                        # Current Week status summary vs Hist and Last Year
                        if curr_wk_hist_match is not None and curr_wk_ly_match is not None:
                            status_label = "In Sync" if (curr_wk_hist_match and curr_wk_ly_match) else "Mixed" if (curr_wk_hist_match or curr_wk_ly_match) else "Out of Sync"
                            delta_label = []
                            if curr_wk_hist_match:
                                delta_label.append("Hist Avg")
                            if curr_wk_ly_match:
                                delta_label.append(str(last_year))
                            delta_str = "Aligns with " + " & ".join(delta_label) if delta_label else "Opposite of both"
                        else:
                            status_label = "No Active Week"
                            delta_str = ""
                        st.metric(
                            label=f"Current Week ({current_week}) Trend",
                            value=status_label,
                            delta=delta_str,
                            help="Tells whether the current week's market direction matches the long-term historical trend, the prior year's trend, or neither."
                        )
                    st.markdown("---")
                    quarters = [
                        ("Q1 (Weeks 1-13)", 1, 13),
                        ("Q2 (Weeks 14-26)", 14, 26),
                        ("Q3 (Weeks 27-39)", 27, 39),
                        ("Q4 (Weeks 40-52+)", 40, 53)
                    ]
                    
                    for q_label, start_wk, end_wk in quarters:
                        ly_q = ly_aligned[(ly_aligned["ISOWeek"] >= start_wk) & (ly_aligned["ISOWeek"] <= end_wk)]
                        cy_q = cy_aligned[(cy_aligned["ISOWeek"] >= start_wk) & (cy_aligned["ISOWeek"] <= end_wk)]
                        
                        if ly_q.empty and cy_q.empty:
                            continue
                            
                        fig_q = go.Figure()
                        
                        # Last Year (Solid Slate Grey)
                        fig_q.add_trace(go.Bar(
                            x=ly_q["ISOWeek"],
                            y=ly_q["Avg_ReturnPct"],
                            name=f"Last Year ({last_year})",
                            marker_color="#4b5563",
                            customdata=np.stack([
                                ly_q["Win_Rate"].fillna(0),
                                ly_q["Events"].fillna(0).astype(int)
                            ], axis=-1),
                            hovertemplate=f"<b>Week %{{x}} ({last_year})</b><br>Avg Return: %{{y:.2f}}%<br>Win Rate: %{{customdata[0]:.1f}}%<br>Days: %{{customdata[1]}}<extra></extra>"
                        ))

                        # Current Year (Rich Red/Green)
                        cy_q_colors = ["#059669" if (not pd.isna(r) and r >= 0) else "#dc2626" if (not pd.isna(r) and r < 0) else "rgba(0,0,0,0)" for r in cy_q["Avg_ReturnPct"]]
                        fig_q.add_trace(go.Bar(
                            x=cy_q["ISOWeek"],
                            y=cy_q["Avg_ReturnPct"],
                            name=f"Current Year ({current_year})",
                            marker_color=cy_q_colors,
                            customdata=np.stack([
                                cy_q["Win_Rate"].fillna(0),
                                cy_q["Events"].fillna(0).astype(int)
                            ], axis=-1),
                            hovertemplate=f"<b>Week %{{x}} ({current_year})</b><br>Avg Return: %{{y:.2f}}%<br>Win Rate: %{{customdata[0]:.1f}}%<br>Days: %{{customdata[1]}}<extra></extra>"
                        ))

                        # Add current week vertical line if the data goes up to the current calendar year and falls within this quarter
                        today_year = pd.Timestamp.today().year
                        if current_year == today_year and start_wk <= current_week <= end_wk:
                            fig_q.add_vline(x=current_week, line_dash="dash", line_color="#38bdf8")

                        fig_q.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
                        fig_q.update_layout(
                            barmode="group",
                            template="plotly_dark",
                            paper_bgcolor="#0d1b2e",
                            plot_bgcolor="#0d1b2e",
                            height=350,
                            title=f"{q_label} Performance ({last_year} vs {current_year})",
                            xaxis=dict(title="ISO Week", tickmode="linear", tick0=start_wk, dtick=1),
                            yaxis=dict(title="Avg Daily Return (%)")
                        )
                        st.plotly_chart(fig_q, width="stretch")

                    df_table = pd.DataFrame({
                        "ISOWeek": range(1, 53),
                        f"{last_year} Return %": ly_aligned["Avg_ReturnPct"].round(2),
                        f"{last_year} Win Rate %": ly_aligned["Win_Rate"].round(1),
                        f"{last_year} Days": ly_aligned["Events"].fillna(0).astype(int),
                        f"{current_year} Return %": cy_aligned["Avg_ReturnPct"].round(2),
                        f"{current_year} Win Rate %": cy_aligned["Win_Rate"].round(1),
                        f"{current_year} Days": cy_aligned["Events"].fillna(0).astype(int),
                    })
                    st.dataframe(df_table, width="stretch", hide_index=True)
                else:
                    fig_week = px.bar(
                        weekly_stats,
                        x="ISOWeek",
                        y="Avg_ReturnPct",
                        color="Current",
                        color_discrete_map={"Current Week": "#38bdf8", "Other Weeks": "#64748b"},
                        hover_data={"Median_ReturnPct": ":.2f", "Win_Rate": ":.1f", "Events": True},
                        title=f"Average Daily Return by ISO Week (Current Week: {current_week})",
                    )
                    if current_week <= 52:
                        fig_week.add_vline(x=current_week, line_dash="dash", line_color="#38bdf8")
                    fig_week.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
                    fig_week.update_layout(template="plotly_dark", paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", height=440)
                    st.plotly_chart(fig_week, width="stretch")
                    st.dataframe(weekly_stats.drop(columns=["Current"]).round(2), width="stretch", hide_index=True)

            with sub_tabs[2]:
                st.markdown("### Average Month Performance")
                fig_month_perf = px.bar(
                    monthly_perf,
                    x="Month",
                    y="Avg_ReturnPct",
                    color="Avg_ReturnPct",
                    color_continuous_scale=["#ef4444", "#facc15", "#22c55e"],
                    category_orders={"Month": MONTH_ORDER},
                    hover_data={"Median_ReturnPct": ":.2f", "Win_Rate": ":.1f", "Events": True},
                    title="Average Daily Return by Month",
                )
                fig_month_perf.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
                fig_month_perf.update_layout(template="plotly_dark", paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", height=420)
                st.plotly_chart(fig_month_perf, width="stretch")
                st.dataframe(monthly_perf.round(2), width="stretch", hide_index=True)

            with sub_tabs[3]:
                st.markdown("### Yearly Analysis by Election Cycle")
                if cycle_summary.empty:
                    st.info("Not enough yearly data to calculate election-cycle analysis.")
                else:
                    fig_cycle = px.bar(
                        cycle_summary,
                        x="ElectionCycle",
                        y="Avg_YearReturnPct",
                        color="Avg_YearReturnPct",
                        color_continuous_scale=["#ef4444", "#facc15", "#22c55e"],
                        hover_data={"Median_YearReturnPct": ":.2f", "Win_Rate": ":.1f", "Years": True},
                        title="Average Annual Return by US Presidential Election Cycle",
                    )
                    fig_cycle.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
                    fig_cycle.update_layout(template="plotly_dark", paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", height=420)
                    st.plotly_chart(fig_cycle, width="stretch")
                    st.dataframe(cycle_summary.round(2), width="stretch", hide_index=True)
                    st.markdown("#### Year-by-Year Returns")
                    st.dataframe(
                        yearly_cycle[["Year", "ElectionCycle", "YearReturnPct", "TradingDays", "StartDate", "EndDate"]].round(2),
                        width="stretch",
                        hide_index=True,
                    )

                st.markdown("### Additional Useful Seasonality Reads")
                extra_left, extra_right = st.columns(2)
                with extra_left:
                    st.markdown("#### Quarter Performance")
                    fig_quarter = px.bar(
                        quarter_stats,
                        x="Quarter",
                        y="Avg_ReturnPct",
                        color="Avg_ReturnPct",
                        color_continuous_scale=["#ef4444", "#facc15", "#22c55e"],
                        hover_data={"Median_ReturnPct": ":.2f", "Win_Rate": ":.1f", "Events": True},
                        title="Average Daily Return by Quarter",
                    )
                    fig_quarter.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
                    fig_quarter.update_layout(template="plotly_dark", paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", height=360)
                    st.plotly_chart(fig_quarter, width="stretch")
                with extra_right:
                    st.markdown("#### Strongest / Weakest Calendar Buckets")
                    leaders = pd.DataFrame(
                        [
                            {"Bucket": "Best Weekday", "Value": str(weekday_stats.sort_values("Avg_ReturnPct", ascending=False).iloc[0]["Weekday"]), "Avg %": weekday_stats["Avg_ReturnPct"].max()},
                            {"Bucket": "Worst Weekday", "Value": str(weekday_stats.sort_values("Avg_ReturnPct", ascending=True).iloc[0]["Weekday"]), "Avg %": weekday_stats["Avg_ReturnPct"].min()},
                            {"Bucket": "Best ISO Week", "Value": str(int(weekly_stats.sort_values("Avg_ReturnPct", ascending=False).iloc[0]["ISOWeek"])), "Avg %": weekly_stats["Avg_ReturnPct"].max()},
                            {"Bucket": "Worst ISO Week", "Value": str(int(weekly_stats.sort_values("Avg_ReturnPct", ascending=True).iloc[0]["ISOWeek"])), "Avg %": weekly_stats["Avg_ReturnPct"].min()},
                            {"Bucket": "Best Month", "Value": str(monthly_perf.sort_values("Avg_ReturnPct", ascending=False).iloc[0]["Month"]), "Avg %": monthly_perf["Avg_ReturnPct"].max()},
                            {"Bucket": "Worst Month", "Value": str(monthly_perf.sort_values("Avg_ReturnPct", ascending=True).iloc[0]["Month"]), "Avg %": monthly_perf["Avg_ReturnPct"].min()},
                        ]
                    )
                    st.dataframe(leaders.round(2), width="stretch", hide_index=True)

        with tab_edge:
            edge_summary = pd.DataFrame(
                [
                    {"Event": "First Trading Day", **event_stats(first_days).to_dict()},
                    {"Event": "Last Trading Day", **event_stats(last_days).to_dict()},
                ]
            )
            st.dataframe(edge_summary.round(2), width="stretch", hide_index=True)
            edge_df = pd.concat([first_days, last_days], ignore_index=True)

            if not edge_monthly.empty:
                st.markdown("### Month-by-Month First vs Last Trading Day")
                fig_edge_month = px.bar(
                    edge_monthly,
                    x="Month",
                    y="Avg_ReturnPct",
                    color="Event",
                    barmode="group",
                    category_orders={"Month": MONTH_ORDER},
                    title="Average First/Last Trading Day Return by Month",
                    hover_data={
                        "Avg_ReturnPct": ":.2f",
                        "Median_ReturnPct": ":.2f",
                        "Win_Rate": ":.1f",
                        "Events": True,
                    },
                )
                fig_edge_month.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
                fig_edge_month.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0d1b2e",
                    plot_bgcolor="#0d1b2e",
                    height=480,
                )
                st.plotly_chart(fig_edge_month, width="stretch")

                st.dataframe(edge_monthly.round(2), width="stretch", hide_index=True)

                best_first = edge_monthly[edge_monthly["Event"] == "First Trading Day"].sort_values("Avg_ReturnPct", ascending=False).head(1)
                best_last = edge_monthly[edge_monthly["Event"] == "Last Trading Day"].sort_values("Avg_ReturnPct", ascending=False).head(1)
                if not best_first.empty and not best_last.empty:
                    c1, c2 = st.columns(2)
                    c1.metric(
                        "Best First-Day Month",
                        str(best_first.iloc[0]["Month"]),
                        f"{best_first.iloc[0]['Avg_ReturnPct']:.2f}% avg",
                    )
                    c2.metric(
                        "Best Last-Day Month",
                        str(best_last.iloc[0]["Month"]),
                        f"{best_last.iloc[0]['Avg_ReturnPct']:.2f}% avg",
                    )

            fig_edge = px.box(
                edge_df,
                x="Event",
                y="ReturnPct",
                points="all",
                color="Event",
                title="First vs Last Trading Day Monthly Return Distribution",
            )
            fig_edge.update_layout(template="plotly_dark", paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", height=500)
            st.plotly_chart(fig_edge, width="stretch")

        with tab_weekend:
            if lw_rows.empty:
                st.info("No long-weekend gaps were detected in the selected data.")
            else:
                holiday_counts = (
                    lw_summary.groupby("Holiday", as_index=False)
                    .agg(
                        Events=("Holiday", "size"),
                        Latest_Before_Date=("Before Date", "max"),
                    )
                    .sort_values(["Latest_Before_Date", "Holiday"], ascending=[False, True])
                    .rename(columns={"Latest_Before_Date": "Latest Before Date"})
                )

                holiday_options = ["All Holidays"] + holiday_counts["Holiday"].tolist()
                if st.session_state.get("seasonality_holiday_filter") not in holiday_options:
                    st.session_state["seasonality_holiday_filter"] = "All Holidays"
                selected_holiday = st.selectbox(
                    "Holiday",
                    holiday_options,
                    key="seasonality_holiday_filter",
                    help="Filter the average 5-trading-day before/after chart by holiday.",
                )

                chart_rows = lw_rows.copy()
                chart_summary = lw_summary.copy()
                title_holiday = "All Long Weekends"
                if selected_holiday != "All Holidays":
                    chart_rows = chart_rows[chart_rows["Holiday"] == selected_holiday]
                    chart_summary = chart_summary[chart_summary["Holiday"] == selected_holiday]
                    title_holiday = selected_holiday

                avg_by_offset = chart_rows.groupby("TradingDayOffset", as_index=False).agg(
                    ReturnPct=("ReturnPct", "mean"),
                    Events=("ReturnPct", "count"),
                )
                fig_lw = px.bar(
                    avg_by_offset,
                    x="TradingDayOffset",
                    y="ReturnPct",
                    color="ReturnPct",
                    color_continuous_scale=["#ef4444", "#facc15", "#22c55e"],
                    hover_data={"Events": True, "ReturnPct": ":.2f"},
                    title=f"Average Daily Return 5 Trading Days Before/After - {title_holiday}",
                )
                fig_lw.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
                fig_lw.update_layout(template="plotly_dark", paper_bgcolor="#0d1b2e", plot_bgcolor="#0d1b2e", height=480)
                st.plotly_chart(fig_lw, width="stretch")
                st.dataframe(chart_summary.round(2), width="stretch", hide_index=True)

        with tab_dist:
            st.markdown("### Return Distribution & Normal Fit Study")
            st.markdown(
                "Compare the return distributions and fitted Normal curves of daily returns "
                "across different seasonal cycles and buckets."
            )
            
            # Category selection
            cycle_cat = st.selectbox(
                "Select Seasonal Cycle Category:",
                ["Month", "Weekday", "US Presidential Election Cycle", "First vs Last Trading Day"],
                key="season_dist_cycle_cat"
            )
            
            # Prepare options based on category
            if cycle_cat == "Month":
                bucket_options = MONTH_ORDER
                df_temp = df.copy()
                df_temp["Bucket"] = df_temp["Month"].astype(str)
                default_a, default_b = "January", "December"
            elif cycle_cat == "Weekday":
                bucket_options = WEEKDAY_ORDER
                df_temp = df.copy()
                df_temp["Bucket"] = df_temp["DayName"].astype(str)
                default_a, default_b = "Monday", "Friday"
            elif cycle_cat == "US Presidential Election Cycle":
                bucket_options = ["Election Year", "Post-Election Year", "Midterm Year", "Pre-Election Year"]
                df_temp = df.copy()
                df_temp["Bucket"] = df_temp["Year"].map(election_cycle_label)
                default_a, default_b = "Election Year", "Midterm Year"
            else:  # First vs Last Trading Day
                bucket_options = ["First Trading Day", "Last Trading Day"]
                # Build first/last day events dataframe
                first_events = first_days.copy()
                last_events = last_days.copy()
                first_events["Bucket"] = "First Trading Day"
                first_events["ReturnVal"] = first_events["ReturnPct"]
                last_events["Bucket"] = "Last Trading Day"
                last_events["ReturnVal"] = last_events["ReturnPct"]
                df_temp = pd.concat([first_events[["Bucket", "ReturnVal"]], last_events[["Bucket", "ReturnVal"]]], ignore_index=True)
                default_a, default_b = "First Trading Day", "Last Trading Day"
                
            # Selection of specific buckets to compare
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                bucket_a = st.selectbox("Compare Bucket A:", bucket_options, index=bucket_options.index(default_a), key="season_dist_bucket_a")
            with col_b2:
                bucket_b = st.selectbox("Compare Bucket B:", bucket_options, index=bucket_options.index(default_b), key="season_dist_bucket_b")
                
            # Extract returns series
            if cycle_cat == "First vs Last Trading Day":
                series_a = df_temp.loc[df_temp["Bucket"] == bucket_a, "ReturnVal"].dropna()
                series_b = df_temp.loc[df_temp["Bucket"] == bucket_b, "ReturnVal"].dropna()
            else:
                series_a = df_temp.loc[df_temp["Bucket"] == bucket_a, "DailyPctChange"].dropna()
                series_b = df_temp.loc[df_temp["Bucket"] == bucket_b, "DailyPctChange"].dropna()
                
            if series_a.empty or series_b.empty:
                st.warning("Insufficient data in the selected range to run the distribution study for these buckets.")
            else:
                # Dashboard Cards showing occurrences and percentages
                c_card1, c_card2, c_card3 = st.columns(3)
                total_days = len(df)
                c_card1.metric("Total Trading Days", f"{total_days:,}")
                c_card2.metric(f"{bucket_a} Occurrences", f"{len(series_a):,} days", f"{len(series_a) / total_days * 100:.1f}% of total")
                c_card3.metric(f"{bucket_b} Occurrences", f"{len(series_b):,} days", f"{len(series_b) / total_days * 100:.1f}% of total")
                st.markdown("<br>", unsafe_allow_html=True)
                
                import scipy.stats as stats
                
                # Calculate parameters
                mu_a, std_a = series_a.mean(), series_a.std()
                mu_b, std_b = series_b.mean(), series_b.std()
                
                skew_a, kurt_a = stats.skew(series_a), stats.kurtosis(series_a)
                skew_b, kurt_b = stats.skew(series_b), stats.kurtosis(series_b)
                
                jb_a_stat, jb_a_p = stats.jarque_bera(series_a) if len(series_a) > 2 else (0, 1.0)
                jb_b_stat, jb_b_p = stats.jarque_bera(series_b) if len(series_b) > 2 else (0, 1.0)
                
                # Generate curves
                x_min_s = min(series_a.min(), series_b.min())
                x_max_s = max(series_a.max(), series_b.max())
                x_range_s = np.linspace(x_min_s, x_max_s, 300)
                
                pdf_a = stats.norm.pdf(x_range_s, mu_a, std_a)
                pdf_b = stats.norm.pdf(x_range_s, mu_b, std_b)
                
                # Calculate common bins for visual alignment and raw occurrence count tooltips
                combined_s = np.concatenate([series_a, series_b])
                counts_all_s, bin_edges_s = np.histogram(combined_s, bins=50)
                counts_a, _ = np.histogram(series_a, bins=bin_edges_s)
                counts_b, _ = np.histogram(series_b, bins=bin_edges_s)
                
                bin_centers_s = (bin_edges_s[:-1] + bin_edges_s[1:]) / 2
                bin_width_s = bin_edges_s[1] - bin_edges_s[0]
                
                density_a = counts_a / (counts_a.sum() * bin_width_s) if counts_a.sum() > 0 else counts_a
                density_b = counts_b / (counts_b.sum() * bin_width_s) if counts_b.sum() > 0 else counts_b
                
                # Plotly Chart
                fig_dist_s = go.Figure()
                
                # Bucket A empirical bar (acting as Histogram)
                fig_dist_s.add_trace(go.Bar(
                    x=bin_centers_s,
                    y=density_a,
                    width=[bin_width_s] * len(bin_centers_s),
                    name=f"{bucket_a} Empirical (Hist)",
                    marker_color='rgba(58, 181, 74, 0.4)',
                    customdata=counts_a,
                    hovertemplate="Bin Range: %{x:.2f}%<br>Density: %{y:.4f}<br>Occurrences: %{customdata} days<extra></extra>"
                ))
                
                # Bucket B empirical bar (acting as Histogram)
                fig_dist_s.add_trace(go.Bar(
                    x=bin_centers_s,
                    y=density_b,
                    width=[bin_width_s] * len(bin_centers_s),
                    name=f"{bucket_b} Empirical (Hist)",
                    marker_color='rgba(30, 58, 95, 0.4)',
                    customdata=counts_b,
                    hovertemplate="Bin Range: %{x:.2f}%<br>Density: %{y:.4f}<br>Occurrences: %{customdata} days<extra></extra>"
                ))
                
                # Bucket A fitted normal curve
                fig_dist_s.add_trace(go.Scatter(
                    x=x_range_s,
                    y=pdf_a,
                    mode='lines',
                    name=f"{bucket_a} Fitted Normal (μ={mu_a:.3f}%, σ={std_a:.3f}%)",
                    line=dict(color='#3ab54a', width=2.5)
                ))
                
                # Bucket B fitted normal curve
                fig_dist_s.add_trace(go.Scatter(
                    x=x_range_s,
                    y=pdf_b,
                    mode='lines',
                    name=f"{bucket_b} Fitted Normal (μ={mu_b:.3f}%, σ={std_b:.3f}%)",
                    line=dict(color='#93c5fd', width=2.5)
                ))
                
                fig_dist_s.update_layout(
                    title=f"Return Distribution Overlay & Normal Curves: {bucket_a} vs {bucket_b}",
                    xaxis=dict(title="Daily Return (%)", showgrid=False, color="#8B9CB6"),
                    yaxis=dict(title="Probability Density", showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#8B9CB6"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CDD5E0")),
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=420,
                    barmode='overlay'
                )
                st.plotly_chart(fig_dist_s, width="stretch")
                
                # Metrics Table
                st.markdown("#### Distribution Statistics Comparison")
                
                dist_stats_table = {
                    "Parameter": [
                        "Average Return (Mean)",
                        "Realized Volatility (Std Dev)",
                        "Skewness (Asymmetry)",
                        "Kurtosis (Fat Tails / Tail Risk)",
                        "Jarque-Bera Test p-value",
                        "Normality Conclusion"
                    ],
                    f"{bucket_a}": [
                        f"{mu_a:.3f}%",
                        f"{std_a:.3f}%",
                        f"{skew_a:.3f}",
                        f"{kurt_a:.3f}",
                        f"{jb_a_p:.4e}",
                        "Rejected (Non-Normal)" if jb_a_p < 0.05 else "Accepted (Normal)"
                    ],
                    f"{bucket_b}": [
                        f"{mu_b:.3f}%",
                        f"{std_b:.3f}%",
                        f"{skew_b:.3f}",
                        f"{kurt_b:.3f}",
                        f"{jb_b_p:.4e}",
                        "Rejected (Non-Normal)" if jb_b_p < 0.05 else "Accepted (Normal)"
                    ]
                }
                st.dataframe(pd.DataFrame(dist_stats_table), width="stretch", hide_index=True)
                
                # Empirical Win Rates and Shock Probabilities
                st.markdown("#### 🎯 Empirical Return Probabilities & Edge Comparison")
                
                c_se1, c_se2 = st.columns(2)
                
                win_a = (series_a > 0).mean() * 100
                win_b = (series_b > 0).mean() * 100
                
                up1_a = (series_a > 1).mean() * 100
                up1_b = (series_b > 1).mean() * 100
                
                down1_a = (series_a < -1).mean() * 100
                down1_b = (series_b < -1).mean() * 100
                
                down2_a = (series_a < -2).mean() * 100
                down2_b = (series_b < -2).mean() * 100
                
                with c_se1:
                    st.markdown(f"##### 🟢 Positive Outcome Probabilities ({bucket_a} vs {bucket_b})")
                    st.write(f"**Win Rate (Daily Return > 0%):**")
                    st.write(f"- {bucket_a}: **{win_a:.2f}%** | {bucket_b}: **{win_b:.2f}%**")
                    st.write(f"**Strong Trend Day (Daily Return > +1.00%):**")
                    st.write(f"- {bucket_a}: **{up1_a:.2f}%** | {bucket_b}: **{up1_b:.2f}%**")
                    
                    if abs(win_a - win_b) > 2.0:
                        better_bucket = bucket_a if win_a > win_b else bucket_b
                        w_diff = abs(win_a - win_b)
                        st.success(f"👉 **{better_bucket}** has a **{w_diff:.1f}% higher Win Rate** compared to the other bucket.")
                    else:
                        st.info("👉 Win rates are comparable (within 2% deviation).")
                        
                with c_se2:
                    st.markdown(f"##### 🔴 Negative Downside Outcome Probabilities ({bucket_a} vs {bucket_b})")
                    st.write(f"**Moderate Down Day (Daily Return < -1.00%):**")
                    st.write(f"- {bucket_a}: **{down1_a:.2f}%** | {bucket_b}: **{down1_b:.2f}%**")
                    st.write(f"**Severe Sell-off Day (Daily Return < -2.00%):**")
                    st.write(f"- {bucket_a}: **{down2_a:.2f}%** | {bucket_b}: **{down2_b:.2f}%**")
                    
                    riskier_bucket = bucket_a if down2_a > down2_b else bucket_b
                    r_diff = abs(down2_a - down2_b)
                    if r_diff > 0.5:
                        st.warning(f"👉 **{riskier_bucket}** exhibits higher tail risk, with **{r_diff:.2f}% more frequent** daily sell-offs exceeding -2.00%.")
                    else:
                        st.info("👉 Downside tail probabilities are comparable.")
                        
                st.markdown(
                    f"""
                    <div style="background:rgba(58, 181, 74, 0.04);border:1px solid rgba(58, 181, 74, 0.15);border-radius:8px;padding:14px;margin-top:16px;">
                        <p style="color:#3ab54a;font-weight:700;font-size:0.85rem;margin:0 0 6px 0;">📊 Distribution Insight for {symbol}:</p>
                        <ul style="color:#8B9CB6;font-size:0.78rem;margin:0;padding-left:16px;">
                            <li>Comparing <strong>{bucket_a}</strong> (mean: <strong>{mu_a:.3f}%</strong>) vs <strong>{bucket_b}</strong> (mean: <strong>{mu_b:.3f}%</strong>) reveals the mathematical return shift that drives seasonality edges.</li>
                            <li>Both datasets show a Jarque-Bera p-value near <code>0.00e+00</code>. This highlights that daily stock/index index returns are <strong>leptokurtic (fat-tailed)</strong>. Outlier movements are significantly more frequent than a fitted Normal curve predicts.</li>
                            <li>Realized Volatility (Std Dev) for <strong>{bucket_a}</strong> is <strong>{std_a:.3f}%</strong> compared to <strong>{std_b:.3f}%</strong> for <strong>{bucket_b}</strong>, highlighting which period carries more trading variance.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with tab_data:
            if self.show_raw:
                st.markdown("### First Trading Day Events")
                st.dataframe(first_days[["DateStr", "Month", "ReturnPct", "Close"]].round(2), width="stretch", hide_index=True)
                st.markdown("### Last Trading Day Events")
                st.dataframe(last_days[["DateStr", "Month", "ReturnPct", "Close"]].round(2), width="stretch", hide_index=True)
                st.markdown("### Long Weekend Event Rows")
                display_cols = ["Holiday", "AnchorDate", "TradingDayOffset", "DateStr", "ReturnPct", "Close"]
                if not lw_rows.empty:
                    st.dataframe(lw_rows[display_cols].round(2), width="stretch", hide_index=True)
                else:
                    st.dataframe(lw_rows, width="stretch", hide_index=True)
            st.download_button(
                "Download Daily Seasonality CSV",
                data=df.to_csv(index=False),
                file_name=f"seasonality_{symbol.replace('^', '').lower()}_{self.lookback_years}y.csv",
                mime="text/csv",
                width="stretch",
            )

    def _default_index_universe(self):
        key = "seasonality_sel"
        target = "Index Universe"
        if target in get_universe_names() and key not in st.session_state:
            st.session_state[key] = target
