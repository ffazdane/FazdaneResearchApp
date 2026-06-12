"""
FazDane Analytics — Tier 2
Market Structure Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import logging
from modules.base_module import FazDaneModule
from utils.universe_manager import format_ticker_display, get_ticker_names, render_universe_manager

logger = logging.getLogger("MarketStructure")

DOW_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
WEEK_ORDER = ["Week1","Week2","Week3","Week4","Week5"]
MONTH_ORDER = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
VIEW_OPTIONS = [
    "Average Return",
    "Contribution",
    "T-Stat",
    "Positive Probability",
    "Return Attribution"
]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_history(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    if "Close" in df.columns:
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
    else:
        s = df.iloc[:, 0]
        
    return s

def week_of_month(date):
    d = date.day
    if d <= 7: return "Week1"
    if d <= 14: return "Week2"
    if d <= 21: return "Week3"
    if d <= 28: return "Week4"
    return "Week5"

def calendar_week_of_month(date):
    first_day = date.replace(day=1)
    return f"Week{((date.day + first_day.weekday() - 1) // 7) + 1}"

def compounded(series):
    s = series.dropna()
    if len(s) == 0: return np.nan
    return (1 + s).prod() - 1

def t_stat(series):
    s = series.dropna()
    if len(s) < 2: return np.nan
    mean = s.mean()
    std = s.std(ddof=1)
    if std == 0: return np.nan
    return mean / (std / np.sqrt(len(s)))

def positive_prob(series):
    s = series.dropna()
    if len(s) == 0: return np.nan
    return (s > 0).mean() * 100

def build_matrix(df, segment_col, segment_order, view):
    if view == "Average Return":
        mat = df.pivot_table(index="MonthName", columns=segment_col, values="Return", aggfunc="mean").reindex(index=MONTH_ORDER, columns=segment_order) * 100
    elif view == "Contribution":
        mean = df.pivot_table(index="MonthName", columns=segment_col, values="Return", aggfunc="mean")
        count = df.pivot_table(index="MonthName", columns=segment_col, values="Return", aggfunc="count")
        mat = (mean * count * 100).reindex(index=MONTH_ORDER, columns=segment_order)
    elif view == "T-Stat":
        mat = df.groupby(["MonthName", segment_col])["Return"].apply(t_stat).unstack(segment_col).reindex(index=MONTH_ORDER, columns=segment_order)
    elif view == "Positive Probability":
        mat = df.groupby(["MonthName", segment_col])["Return"].apply(positive_prob).unstack(segment_col).reindex(index=MONTH_ORDER, columns=segment_order)
    elif view == "Return Attribution":
        mat = df.groupby(["MonthName", segment_col])["Return"].apply(lambda s: compounded(s) * 100).unstack(segment_col).reindex(index=MONTH_ORDER, columns=segment_order)
    else:
        mat = pd.DataFrame()
    return mat

def month_total(df):
    return df.groupby("MonthName")["Return"].apply(compounded) * 100

def segment_total(df, segment_col, order):
    return df.groupby(segment_col)["Return"].apply(compounded).reindex(order) * 100

def year_total(df):
    return compounded(df["Return"]) * 100

class MarketStructureModule(FazDaneModule):
    MODULE_NAME = "Market Structure Heatmap"
    MODULE_ICON = "🗓️"
    MODULE_DESCRIPTION = "Month × Segment Analytics Engine for Market Seasonality"
    TIER = 2
    SOURCE_NOTEBOOK = "02-Weekday Ticker Heatmap.ipynb"
    CACHE_TTL = 3600
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["yfinance"]

    def render_sidebar(self):
        current_year = datetime.now().year

        st.markdown("**Ticker Universe**")
        universe_name, tickers_list, _ = render_universe_manager(
            key_prefix="ms",
            show_benchmark=False,
            label="Asset List:"
        )
        ticker_names = get_ticker_names(universe_name)

        # Single asset selector from the universe
        if tickers_list:
            self.symbol = st.selectbox(
                "Select Asset:",
                options=tickers_list,
                index=0,
                format_func=lambda ticker: format_ticker_display(ticker, ticker_names),
            )
            self.asset_label = format_ticker_display(self.symbol, ticker_names)
        else:
            self.symbol = st.text_input("Enter Ticker Symbol:", value="SPY").strip().upper()
            self.asset_label = self.symbol
        
        self.year = st.selectbox("Year:", options=list(range(current_year, current_year - 20, -1)), index=0)
        
        self.segment_mode = st.selectbox("Segment:", options=["Weekday", "Week of Month"], index=0)
        self.view = st.selectbox("View:", options=VIEW_OPTIONS, index=0)
        
        if st.button("🔄 Generate Report", width="stretch", type="primary"):
            st.rerun()

    def render_main(self):
        self.render_section_header("🗓️ Market Structure Dashboard", "Analyze asset seasonality, returns, and structural flow patterns by weekday and month.")
        
        # Determine current date bounds
        current_year = datetime.now().year
        
        start = f"{self.year}-01-01"
        if self.year == current_year:
            end = datetime.now().strftime("%Y-%m-%d")
        else:
            end = f"{self.year}-12-31"
        
        with st.spinner(f"Fetching data for {self.symbol} ({self.year})..."):
            close_data = fetch_market_history(self.symbol, start, end)
            
        if close_data.empty:
            st.warning(f"⚠️ No data found for {self.symbol} in {self.year}.")
            return
            
        df = pd.DataFrame({"Date": close_data.index, "Close": close_data.values})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df["Return"] = df["Close"].pct_change()
        
        df["Day"] = df["Date"].dt.day_name()
        df["MonthName"] = df["Date"].dt.month_name()
        df["WeekOfMonth"] = df["Date"].apply(week_of_month)
        
        df = df[df["Day"].isin(DOW_ORDER)]
        df = df.dropna(subset=["Return"])
        
        if df.empty:
            st.warning("⚠️ Not enough trading data to compute returns for this year.")
            return

        if self.segment_mode == "Weekday":
            seg_col = "Day"
            seg_order = DOW_ORDER
        else:
            seg_col = "WeekOfMonth"
            seg_order = WEEK_ORDER

        matrix = build_matrix(df, seg_col, seg_order, self.view)
        m_total = month_total(df).reindex(MONTH_ORDER)
        s_total = segment_total(df, seg_col, seg_order)
        y_total = year_total(df)
        
        pivot = matrix.copy()
        pivot["Month Total (Comp %)"] = m_total
        
        bottom = list(s_total.values) + [y_total]
        pivot.loc["Year Total (Comp %)"] = bottom
        
        # Plotly rendering
        z_vals = pivot.values.astype(float)
        
        # Create text array
        value_suffix = "" if self.view == "T-Stat" else "%"
        text_array = []
        for i in range(pivot.shape[0]):
            row_texts = []
            for j in range(pivot.shape[1]):
                val = z_vals[i, j]
                if np.isfinite(val):
                    row_texts.append(f"{val:.2f}{value_suffix}")
                else:
                    row_texts.append("")
            text_array.append(row_texts)
            
        fig = go.Figure(data=go.Heatmap(
            z=z_vals,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            text=text_array,
            texttemplate="%{text}",
            colorscale='RdYlGn',
            zmid=0,
            showscale=True,
            colorbar=dict(title=self.view, thickness=15),
            hoverinfo="x+y+z",
            hovertemplate="<b>%{y} - %{x}</b><br>Value: %{text}<extra></extra>",
            xgap=1,
            ygap=1,
        ))
        fig.update_traces(textfont=dict(size=16, color="#111827", family="Arial Black"))
        
        calc_height = max(500, (len(pivot) + 1) * 45)
        
        fig.update_layout(
            title=f"{self.asset_label} – {self.view} (Month × {self.segment_mode}) – {self.year}",
            title_font=dict(size=20, color="#3ab54a", family="Arial Black"),
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#0d1b2e",
            font=dict(color="#e2e8f0", size=16, family="Arial Black"),
            height=calc_height,
            margin=dict(l=0, r=0, t=110, b=40),
            xaxis=dict(side="top", gridcolor="#1e3a5f", tickfont=dict(size=15, color="#e2e8f0", family="Arial Black")),
            yaxis=dict(autorange="reversed", gridcolor="#1e3a5f", tickfont=dict(size=15, color="#94a3b8", family="Arial Black"))
        )
        
        # Add separating lines
        fig.add_hline(y=len(pivot) - 1.5, line_width=2, line_color="#cbd5e1")
        fig.add_vline(x=len(seg_order) - 0.5, line_width=2, line_color="#cbd5e1")
        
        st.plotly_chart(fig, width="stretch")

        st.markdown("### Monthly Drill Down")
        available_months = [month for month in MONTH_ORDER if month in df["MonthName"].unique()]
        default_month_index = max(len(available_months) - 1, 0)
        selected_month = st.selectbox(
            "Month:",
            options=available_months,
            index=default_month_index,
            key=f"market_structure_drilldown_month_{self.symbol}_{self.year}",
        )

        drilldown = df[df["MonthName"] == selected_month].copy()
        drilldown["ReturnPct"] = drilldown["Return"] * 100
        drilldown["DayOfMonth"] = drilldown["Date"].dt.day
        drilldown["DateLabel"] = drilldown["Date"].dt.strftime("%b %d")
        drilldown["CalendarWeek"] = drilldown["Date"].apply(calendar_week_of_month)
        drilldown = drilldown.sort_values("Date")

        calendar_week_order = [f"Week{i}" for i in range(1, 7)]
        active_weeks = [week for week in calendar_week_order if week in drilldown["CalendarWeek"].unique()]
        daily_matrix = (
            drilldown.pivot_table(index="CalendarWeek", columns="Day", values="ReturnPct", aggfunc="first")
            .reindex(index=active_weeks, columns=DOW_ORDER)
        )
        date_matrix = (
            drilldown.pivot_table(index="CalendarWeek", columns="Day", values="DateLabel", aggfunc="first")
            .reindex(index=active_weeks, columns=DOW_ORDER)
        )
        daily_matrix["Total %"] = (
            drilldown.groupby("CalendarWeek")["Return"]
            .apply(compounded)
            .mul(100)
            .reindex(active_weeks)
        )
        weekday_summary = matrix if seg_order == DOW_ORDER else build_matrix(df, "Day", DOW_ORDER, self.view)
        total_row = weekday_summary.reindex(index=[selected_month], columns=DOW_ORDER).iloc[0]
        total_row["Total %"] = m_total.reindex([selected_month]).iloc[0]
        daily_matrix.loc["Total %"] = total_row
        row_labels = [week.replace("Week", "Week ") for week in active_weeks] + ["Total %"]
        daily_matrix.index = row_labels
        date_matrix.index = row_labels[:-1]
        daily_text = daily_matrix.copy().astype(object)
        for row_label in row_labels:
            for column_label in daily_matrix.columns:
                value = daily_matrix.loc[row_label, column_label]
                if not np.isfinite(value):
                    daily_text.loc[row_label, column_label] = ""
                elif row_label == "Total %" or column_label == "Total %":
                    daily_text.loc[row_label, column_label] = f"Total {value:.2f}{value_suffix}"
                else:
                    date_label = date_matrix.loc[row_label, column_label]
                    daily_text.loc[row_label, column_label] = f"{date_label} {value:+.2f}%"

        drilldown_fig = go.Figure(
            data=go.Heatmap(
                z=daily_matrix.values,
                x=daily_matrix.columns.tolist(),
                y=daily_matrix.index.tolist(),
                text=daily_text.values,
                texttemplate="",
                colorscale="RdYlGn",
                zmid=0,
                showscale=True,
                colorbar=dict(title="Daily Return %", thickness=15),
                hovertemplate="<b>%{y} - %{x}</b><br>Return: %{text}<extra></extra>",
                xgap=1,
                ygap=1,
            )
        )
        for row_label in daily_matrix.index:
            for column_label in daily_matrix.columns:
                value = daily_matrix.loc[row_label, column_label]
                if not np.isfinite(value):
                    continue
                drilldown_fig.add_annotation(
                    x=column_label,
                    y=row_label,
                    text=daily_text.loc[row_label, column_label],
                    showarrow=False,
                    font=dict(size=16, color="#111827", family="Arial Black"),
                )
        drilldown_fig.update_layout(
            title=f"{self.asset_label} - Monthly Drill Down by Weekday - {selected_month} {self.year}",
            title_font=dict(size=20, color="#3ab54a", family="Arial Black"),
            paper_bgcolor="#0d1b2e",
            plot_bgcolor="#0d1b2e",
            font=dict(color="#e2e8f0", size=16, family="Arial Black"),
            height=max(520, 72 * (len(daily_matrix.index) + 1)),
            margin=dict(l=0, r=0, t=135, b=80),
            xaxis=dict(side="top", tickangle=0, gridcolor="#1e3a5f", tickfont=dict(size=15, color="#e2e8f0", family="Arial Black")),
            yaxis=dict(autorange="reversed", gridcolor="#1e3a5f", tickfont=dict(size=15, color="#94a3b8", family="Arial Black")),
        )
        drilldown_fig.add_hline(y=len(daily_matrix.index) - 1.5, line_width=4, line_color="#000000")
        drilldown_fig.add_vline(x=len(DOW_ORDER) - 0.5, line_width=4, line_color="#000000")

        st.plotly_chart(drilldown_fig, width="stretch")

        month_total_value = m_total.reindex([selected_month]).iloc[0]
        best_day = drilldown.loc[drilldown["ReturnPct"].idxmax()]
        worst_day = drilldown.loc[drilldown["ReturnPct"].idxmin()]
        positive_days = int((drilldown["ReturnPct"] > 0).sum())
        negative_days = int((drilldown["ReturnPct"] < 0).sum())
        best_week = (
            drilldown.groupby("CalendarWeek")["Return"]
            .apply(compounded)
            .mul(100)
            .idxmax()
        )
        best_day_label = f"{best_day['Date'].strftime('%b')} {best_day['Date'].day}"
        worst_day_label = f"{worst_day['Date'].strftime('%b')} {worst_day['Date'].day}"
        best_week_date = drilldown.loc[drilldown["CalendarWeek"] == best_week, "Date"].min()
        best_week_start = f"{best_week_date.strftime('%b')} {best_week_date.day}"
        total_color = "#86efac" if month_total_value >= 0 else "#fb7185"

        st.markdown(
            f"""
            <div style="
                display: grid;
                grid-template-columns: 1.35fr repeat(5, minmax(120px, 1fr));
                gap: 14px;
                margin-top: 18px;
                padding: 14px;
                border: 1px solid #1e3a5f;
                border-radius: 10px;
                background: #071426;
            ">
                <div style="
                    min-height: 124px;
                    padding: 22px;
                    border: 2px solid #22c55e;
                    border-radius: 9px;
                    background: #052317;
                    color: #e2e8f0;
                    box-shadow: inset 0 0 24px rgba(34, 197, 94, 0.14);
                ">
                    <div style="font-size: 20px; font-weight: 800;">{self.symbol} {selected_month} {self.year}</div>
                    <div style="font-size: 52px; line-height: 1.1; font-weight: 900; color: {total_color};">{month_total_value:+.2f}%</div>
                </div>
                <div style="min-height: 124px; padding: 18px; border: 1px solid #263f63; border-radius: 9px; background: #0a1b2f; color: #e2e8f0;">
                    <div style="font-size: 17px; font-weight: 800;">Best Day</div>
                    <div style="font-size: 28px; font-weight: 900; color: #86efac;">{best_day_label}</div>
                    <div style="font-size: 25px; font-weight: 900; color: #86efac;">{best_day["ReturnPct"]:+.2f}%</div>
                </div>
                <div style="min-height: 124px; padding: 18px; border: 1px solid #263f63; border-radius: 9px; background: #0a1b2f; color: #e2e8f0;">
                    <div style="font-size: 17px; font-weight: 800;">Worst Day</div>
                    <div style="font-size: 28px; font-weight: 900; color: #fb7185;">{worst_day_label}</div>
                    <div style="font-size: 25px; font-weight: 900; color: #fb7185;">{worst_day["ReturnPct"]:+.2f}%</div>
                </div>
                <div style="min-height: 124px; padding: 18px; border: 1px solid #263f63; border-radius: 9px; background: #0a1b2f; color: #e2e8f0;">
                    <div style="font-size: 17px; font-weight: 800;">Positive Days</div>
                    <div style="font-size: 54px; line-height: 1.15; font-weight: 900; color: #86efac;">{positive_days}</div>
                </div>
                <div style="min-height: 124px; padding: 18px; border: 1px solid #263f63; border-radius: 9px; background: #0a1b2f; color: #e2e8f0;">
                    <div style="font-size: 17px; font-weight: 800;">Negative Days</div>
                    <div style="font-size: 54px; line-height: 1.15; font-weight: 900; color: #fb7185;">{negative_days}</div>
                </div>
                <div style="min-height: 124px; padding: 18px; border: 1px solid #263f63; border-radius: 9px; background: #0a1b2f; color: #e2e8f0;">
                    <div style="font-size: 17px; font-weight: 800;">Best Week</div>
                    <div style="font-size: 25px; line-height: 1.25; font-weight: 900; color: #38bdf8;">Week of<br>{best_week_start}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
