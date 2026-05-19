"""
FazDane Analytics - Tier 3
Bradley Siderograph vs SPX with inverse overlay.
"""

from datetime import date, datetime, timedelta
from importlib.util import find_spec

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.signal import find_peaks, savgol_filter

from modules.base_module import FazDaneModule


ASPECTS = [0, 60, 90, 120, 180]
DEFAULT_ORB = 15
ASPECT_SIGN = {60: 1, 90: -1, 120: 1, 180: -1}
CONJ_SIGN = {
    frozenset(["Venus", "Mars"]): 1,
    frozenset(["Venus", "Jupiter"]): 1,
    frozenset(["Venus", "Saturn"]): -1,
    frozenset(["Mars", "Jupiter"]): 1,
    frozenset(["Mars", "Saturn"]): -1,
    frozenset(["Jupiter", "Saturn"]): -1,
}
PLANET_PAIRS = [
    ("Venus", "Mars"),
    ("Venus", "Jupiter"),
    ("Venus", "Saturn"),
    ("Mars", "Jupiter"),
    ("Mars", "Saturn"),
    ("Jupiter", "Saturn"),
]


def normalize_series(series: pd.Series) -> pd.Series:
    low = series.min()
    high = series.max()
    if pd.isna(low) or pd.isna(high) or high == low:
        return pd.Series(np.nan, index=series.index)
    return (series - low) / (high - low) * 100


def safe_savgol(values: list[float], requested_window: int, polyorder: int) -> np.ndarray:
    if len(values) < 3:
        return np.array(values, dtype=float)

    window = min(int(requested_window), len(values))
    if window % 2 == 0:
        window -= 1
    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    window = max(window, min_window)
    if window > len(values):
        window = len(values) if len(values) % 2 == 1 else len(values) - 1
    if window <= polyorder:
        return np.array(values, dtype=float)
    return savgol_filter(values, window, polyorder)


@st.cache_resource(show_spinner=False)
def load_planet_kernel():
    from skyfield.api import load

    planets = load("de421.bsp")
    return planets, load.timescale()


@st.cache_data(ttl=86400, show_spinner=False)
def calculate_bradley_siderograph(
    start_date: date,
    end_date: date,
    orb: int,
    smooth_window: int,
    declination_weight: float,
) -> pd.DataFrame:
    planets, ts = load_planet_kernel()

    earth = planets[399]
    bodies = {
        "Venus": planets[299],
        "Mars": planets[499],
        "Jupiter": planets[5],
        "Saturn": planets[6],
    }
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    scores = []
    mid_scores = []
    long_scores = []
    declination_scores = []

    for current_date in dates:
        t = ts.utc(current_date.year, current_date.month, current_date.day)
        positions = {}
        declinations = {}

        for name, body in bodies.items():
            astrometric = earth.at(t).observe(body)
            _, lon, _ = astrometric.ecliptic_latlon()
            positions[name] = lon.degrees

        for name in ("Venus", "Mars"):
            astrometric = earth.at(t).observe(bodies[name])
            _, dec, _ = astrometric.radec()
            declinations[name] = dec.degrees

        mid_score = 0.0
        long_score = 0.0
        for p1, p2 in PLANET_PAIRS:
            distance = abs(positions[p1] - positions[p2])
            if distance > 180:
                distance = 360 - distance

            for aspect in ASPECTS:
                if abs(distance - aspect) <= orb:
                    closeness = 1 - abs(distance - aspect) / orb
                    sign = ASPECT_SIGN[aspect] if aspect != 0 else CONJ_SIGN.get(frozenset([p1, p2]), 1)
                    value = sign * closeness
                    if {p1, p2} == {"Jupiter", "Saturn"}:
                        long_score += value
                    else:
                        mid_score += value
                    break

        declination_score = 0.5 * (declinations["Venus"] + declinations["Mars"])
        composite = mid_score + declination_weight * (long_score + declination_score)
        mid_scores.append(mid_score)
        long_scores.append(long_score)
        declination_scores.append(declination_score)
        scores.append(composite)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Bradley_Raw": scores,
            "Midterm_Aspects": mid_scores,
            "Longterm_Aspects": long_scores,
            "Declination": declination_scores,
        }
    )
    df["Bradley_Score"] = safe_savgol(df["Bradley_Raw"].tolist(), smooth_window, 3)
    df["Bradley_Norm"] = normalize_series(df["Bradley_Score"])
    df["Bradley_Inverse"] = 100 - df["Bradley_Norm"]
    return df


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_spx_data(start_date: date, end_date: date) -> pd.DataFrame:
    data = yf.download("^GSPC", start=start_date, end=end_date + timedelta(days=1), auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    close_col = "Close" if "Close" in data.columns else data.columns[0]
    df = data[[close_col]].copy()
    df.columns = ["Close"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.reset_index().rename(columns={"index": "Date"})
    df["SPX_Norm"] = normalize_series(df["Close"])
    return df


def build_bradley_chart(merged: pd.DataFrame, show_raw: bool, show_inverse: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged["Date"],
            y=merged["Bradley_Norm"],
            mode="lines",
            name="Bradley Siderograph",
            line=dict(color="#3b82f6", width=2.5),
        )
    )
    if show_inverse:
        fig.add_trace(
            go.Scatter(
                x=merged["Date"],
                y=merged["Bradley_Inverse"],
                mode="lines",
                name="Bradley Inverse",
                line=dict(color="#ef4444", width=2, dash="dot"),
            )
        )
    if show_raw:
        fig.add_trace(
            go.Scatter(
                x=merged["Date"],
                y=merged["Bradley_Raw_Norm"],
                mode="lines",
                name="Raw Bradley",
                line=dict(color="#94a3b8", width=1, dash="dash"),
                opacity=0.55,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=merged["Date"],
            y=merged["SPX_Norm"],
            mode="lines",
            name="SPX Normalized",
            line=dict(color="#22c55e", width=2),
            connectgaps=True,
        )
    )
    fig.update_layout(
        title="Bradley Siderograph vs SPX with Inversion",
        xaxis_title="Date",
        yaxis_title="Normalized Score (0-100)",
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#0d1b2e",
        plot_bgcolor="#0d1b2e",
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=24, r=24, t=84, b=36),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)", range=[0, 100])
    return fig


def turning_points(df: pd.DataFrame, prominence: float) -> pd.DataFrame:
    clean = df.dropna(subset=["Bradley_Norm"]).reset_index(drop=True)
    if clean.empty:
        return pd.DataFrame()

    peaks, _ = find_peaks(clean["Bradley_Norm"], prominence=prominence)
    troughs, _ = find_peaks(100 - clean["Bradley_Norm"], prominence=prominence)
    rows = []
    for idx in peaks:
        rows.append({"Date": clean.loc[idx, "Date"], "Type": "Bradley High", "Score": clean.loc[idx, "Bradley_Norm"]})
    for idx in troughs:
        rows.append({"Date": clean.loc[idx, "Date"], "Type": "Bradley Low", "Score": clean.loc[idx, "Bradley_Norm"]})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def paired_correlation(df: pd.DataFrame, left: str, right: str) -> float:
    pairs = df[[left, right]].dropna()
    if len(pairs) < 3:
        return np.nan
    return float(pairs.corr().iloc[0, 1])


class BradleySiderographModule(FazDaneModule):
    MODULE_NAME = "Bradley Siderograph"
    MODULE_ICON = "Cycles"
    MODULE_DESCRIPTION = "Planetary aspect cycle model compared with normalized SPX, including inverse Bradley overlay"
    TIER = 3
    SOURCE_NOTEBOOK = "Forecasting/Cycle Analysis/Bradley Siderograph.ipynb"
    CACHE_TTL = 86400
    REQUIRES_LIVE_DATA = True
    DATA_SOURCES = ["Skyfield de421", "yfinance"]

    def render_sidebar(self):
        today = datetime.today().date()
        default_start = date(2025, 1, 1)
        default_end = date(2027, 12, 31)

        st.markdown("**Bradley Window**")
        self.start_date = st.date_input("Start Date:", value=default_start, key="bradley_start")
        self.end_date = st.date_input("End Date:", value=default_end, key="bradley_end")

        st.markdown("**Model Controls**")
        self.orb = int(st.slider("Aspect Orb:", 5, 25, DEFAULT_ORB, key="bradley_orb"))
        self.smooth_window = int(st.slider("Smoothing Window:", 5, 31, 11, step=2, key="bradley_smooth"))
        self.declination_weight = float(
            st.slider("Declination Weight:", 1.0, 8.0, 5.0, step=0.5, key="bradley_declination_weight")
        )
        self.turn_prominence = float(st.slider("Turning Point Sensitivity:", 2.0, 20.0, 8.0, step=1.0, key="bradley_prom"))

        st.markdown("**Chart Layers**")
        self.show_inverse = st.checkbox("Show inverse Bradley", value=True, key="bradley_show_inverse")
        self.show_raw = st.checkbox("Show raw Bradley", value=False, key="bradley_show_raw")
        self.show_components = st.checkbox("Show component table", value=False, key="bradley_components")

        if self.end_date > today:
            st.caption(f"SPX data is available only through the latest Yahoo Finance session; future Bradley dates remain plotted.")

        if st.button("Refresh Bradley Model", use_container_width=True, type="primary", key="bradley_refresh"):
            calculate_bradley_siderograph.clear()
            fetch_spx_data.clear()
            st.rerun()

    def render_main(self):
        self.render_section_header(
            "Bradley Siderograph vs SPX",
            "Original and inverted Bradley cycle curves normalized against S&P 500 daily closes",
        )

        if find_spec("skyfield") is None:
            st.error("Skyfield is not installed in the Python environment currently running Streamlit.")
            st.code("pip install skyfield jplephem", language="powershell")
            st.info("After installing, restart Streamlit so the Bradley module can load the astronomy kernel.")
            return

        if self.start_date >= self.end_date:
            st.warning("Choose an end date after the start date.")
            return

        with st.spinner("Calculating planetary aspects and SPX comparison..."):
            bradley = calculate_bradley_siderograph(
                self.start_date,
                self.end_date,
                self.orb,
                self.smooth_window,
                self.declination_weight,
            )
            spx = fetch_spx_data(self.start_date, min(self.end_date, datetime.today().date()))

        if bradley.empty:
            st.warning("No Bradley data could be calculated for this date range.")
            return

        bradley["Bradley_Raw_Norm"] = normalize_series(bradley["Bradley_Raw"])
        if spx.empty:
            merged = bradley.copy()
            merged["Close"] = np.nan
            merged["SPX_Norm"] = np.nan
            st.warning("No SPX data returned from Yahoo Finance for the selected range.")
        else:
            merged = pd.merge(bradley, spx[["Date", "Close", "SPX_Norm"]], on="Date", how="left")

        recent = merged.dropna(subset=["SPX_Norm"])
        latest_spx_date = recent["Date"].max() if not recent.empty else None
        latest_bradley = merged.iloc[-1]
        inv_corr = paired_correlation(merged, "Bradley_Inverse", "SPX_Norm")
        direct_corr = paired_correlation(merged, "Bradley_Norm", "SPX_Norm")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bradley Days", f"{len(merged):,}")
        c2.metric("Latest SPX Date", latest_spx_date.strftime("%Y-%m-%d") if latest_spx_date is not None else "N/A")
        c3.metric("Direct Corr", f"{direct_corr:.2f}" if not pd.isna(direct_corr) else "N/A")
        c4.metric("Inverse Corr", f"{inv_corr:.2f}" if not pd.isna(inv_corr) else "N/A")

        st.plotly_chart(build_bradley_chart(merged, self.show_raw, self.show_inverse), use_container_width=True)

        turns = turning_points(merged, self.turn_prominence)
        tab_turns, tab_data, tab_notes = st.tabs(["Turning Points", "Data", "Method"])

        with tab_turns:
            if turns.empty:
                st.info("No Bradley highs/lows met the selected sensitivity threshold.")
            else:
                upcoming = turns[turns["Date"].dt.date >= datetime.today().date()].head(12)
                st.markdown("### Upcoming Bradley Turning Windows")
                if upcoming.empty:
                    st.info("No future turning windows remain in the selected range.")
                else:
                    st.dataframe(upcoming.assign(Date=upcoming["Date"].dt.strftime("%Y-%m-%d")).round(2), use_container_width=True, hide_index=True)
                st.markdown("### All Detected Turning Windows")
                st.dataframe(turns.assign(Date=turns["Date"].dt.strftime("%Y-%m-%d")).round(2), use_container_width=True, hide_index=True)

        with tab_data:
            display_cols = ["Date", "Bradley_Norm", "Bradley_Inverse", "SPX_Norm", "Close"]
            table = merged[display_cols].copy()
            table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(table.round(2), use_container_width=True, hide_index=True)
            st.download_button(
                "Download Bradley vs SPX CSV",
                data=merged.to_csv(index=False),
                file_name=f"bradley_siderograph_spx_{self.start_date}_{self.end_date}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if self.show_components:
                component_cols = ["Date", "Bradley_Raw", "Bradley_Score", "Midterm_Aspects", "Longterm_Aspects", "Declination"]
                components = merged[component_cols].copy()
                components["Date"] = components["Date"].dt.strftime("%Y-%m-%d")
                st.markdown("### Bradley Components")
                st.dataframe(components.round(4), use_container_width=True, hide_index=True)

        with tab_notes:
            st.markdown(
                """
                This module scores Venus, Mars, Jupiter, and Saturn aspects at conjunction, sextile,
                square, trine, and opposition. The daily composite is smoothed, normalized to 0-100,
                and mirrored as `100 - Bradley_Norm` for inverse-cycle comparison against SPX.

                Bradley dates are best treated as potential turning windows, not directional signals.
                Future SPX values are blank until Yahoo Finance has market data for those sessions.
                """
            )
