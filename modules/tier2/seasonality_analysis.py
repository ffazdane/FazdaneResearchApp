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

        if st.button("Refresh Seasonality", use_container_width=True, type="primary", key="seasonality_refresh"):
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

        tab_month, tab_cycles, tab_edge, tab_weekend, tab_data = st.tabs(
            ["Monthly", "Calendar Cycles", "First/Last Day", "Long Weekends", "Data"]
        )

        with tab_month:
            st.plotly_chart(plot_monthly_scatter(df, symbol), use_container_width=True)
            left, right = st.columns(2)
            with left:
                st.plotly_chart(plot_monthly_bars(stats), use_container_width=True)
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
                st.plotly_chart(fig_box, use_container_width=True)
            st.plotly_chart(plot_pos_neg(stats), use_container_width=True)
            st.dataframe(stats.round(2), use_container_width=True, hide_index=True)

        with tab_cycles:
            cycle_df = df.copy()
            cycle_df["Weekday"] = pd.Categorical(cycle_df["DayName"], categories=WEEKDAY_ORDER, ordered=True)
            cycle_df["ISOWeek"] = cycle_df["Date"].dt.isocalendar().week.astype(int)
            cycle_df["Quarter"] = "Q" + cycle_df["Date"].dt.quarter.astype(str)

            weekday_stats = grouped_return_stats(cycle_df, "Weekday").sort_values("Weekday")
            current_week = int(pd.Timestamp.today().isocalendar().week)
            weekly_stats = grouped_return_stats(cycle_df[cycle_df["ISOWeek"] <= 52], "ISOWeek").sort_values("ISOWeek")
            weekly_stats["Current"] = np.where(weekly_stats["ISOWeek"] == current_week, "Current Week", "Other Weeks")

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
            st.plotly_chart(fig_weekday, use_container_width=True)
            st.dataframe(weekday_stats.round(2), use_container_width=True, hide_index=True)

            st.markdown("### Average Week Performance")
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
            st.plotly_chart(fig_week, use_container_width=True)
            st.dataframe(weekly_stats.drop(columns=["Current"]).round(2), use_container_width=True, hide_index=True)

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
            st.plotly_chart(fig_month_perf, use_container_width=True)
            st.dataframe(monthly_perf.round(2), use_container_width=True, hide_index=True)

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
                st.plotly_chart(fig_cycle, use_container_width=True)
                st.dataframe(cycle_summary.round(2), use_container_width=True, hide_index=True)
                st.markdown("#### Year-by-Year Returns")
                st.dataframe(
                    yearly_cycle[["Year", "ElectionCycle", "YearReturnPct", "TradingDays", "StartDate", "EndDate"]].round(2),
                    use_container_width=True,
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
                st.plotly_chart(fig_quarter, use_container_width=True)
            with extra_right:
                st.markdown("#### Strongest / Weakest Calendar Buckets")
                leaders = pd.DataFrame(
                    [
                        {"Bucket": "Best Weekday", "Value": str(weekday_stats.sort_values("Avg_ReturnPct", ascending=False).iloc[0]["Weekday"]), "Avg %": weekday_stats["Avg_ReturnPct"].max()},
                        {"Bucket": "Worst Weekday", "Value": str(weekday_stats.sort_values("Avg_ReturnPct", ascending=True).iloc[0]["Weekday"]), "Avg %": weekday_stats["Avg_ReturnPct"].min()},
                        {"Bucket": "Best ISO Week", "Value": int(weekly_stats.sort_values("Avg_ReturnPct", ascending=False).iloc[0]["ISOWeek"]), "Avg %": weekly_stats["Avg_ReturnPct"].max()},
                        {"Bucket": "Worst ISO Week", "Value": int(weekly_stats.sort_values("Avg_ReturnPct", ascending=True).iloc[0]["ISOWeek"]), "Avg %": weekly_stats["Avg_ReturnPct"].min()},
                        {"Bucket": "Best Month", "Value": str(monthly_perf.sort_values("Avg_ReturnPct", ascending=False).iloc[0]["Month"]), "Avg %": monthly_perf["Avg_ReturnPct"].max()},
                        {"Bucket": "Worst Month", "Value": str(monthly_perf.sort_values("Avg_ReturnPct", ascending=True).iloc[0]["Month"]), "Avg %": monthly_perf["Avg_ReturnPct"].min()},
                    ]
                )
                st.dataframe(leaders.round(2), use_container_width=True, hide_index=True)

        with tab_edge:
            edge_summary = pd.DataFrame(
                [
                    {"Event": "First Trading Day", **event_stats(first_days).to_dict()},
                    {"Event": "Last Trading Day", **event_stats(last_days).to_dict()},
                ]
            )
            st.dataframe(edge_summary.round(2), use_container_width=True, hide_index=True)
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
                st.plotly_chart(fig_edge_month, use_container_width=True)

                st.dataframe(edge_monthly.round(2), use_container_width=True, hide_index=True)

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
            st.plotly_chart(fig_edge, use_container_width=True)

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
                st.plotly_chart(fig_lw, use_container_width=True)
                st.dataframe(chart_summary.round(2), use_container_width=True, hide_index=True)

        with tab_data:
            if self.show_raw:
                st.markdown("### First Trading Day Events")
                st.dataframe(first_days[["DateStr", "Month", "ReturnPct", "Close"]].round(2), use_container_width=True, hide_index=True)
                st.markdown("### Last Trading Day Events")
                st.dataframe(last_days[["DateStr", "Month", "ReturnPct", "Close"]].round(2), use_container_width=True, hide_index=True)
                st.markdown("### Long Weekend Event Rows")
                st.dataframe(lw_rows.round(2), use_container_width=True, hide_index=True)
            st.download_button(
                "Download Daily Seasonality CSV",
                data=df.to_csv(index=False),
                file_name=f"seasonality_{symbol.replace('^', '').lower()}_{self.lookback_years}y.csv",
                mime="text/csv",
                use_container_width=True,
            )

    def _default_index_universe(self):
        key = "seasonality_sel"
        target = "Index Universe"
        if target in get_universe_names() and key not in st.session_state:
            st.session_state[key] = target
