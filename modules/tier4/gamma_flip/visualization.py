"""Plotly visuals for the Gamma Flip Line / GEX Engine."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


BRAND_GREEN = "#3ab54a"
BRAND_BLUE = "#1a3a8f"
TEXT = "#e2e8f0"
GRID = "rgba(148,163,184,0.18)"


def _base_layout(title: str) -> dict:
    return {
        "title": title,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(13,27,46,0.35)",
        "font": {"color": TEXT},
        "margin": {"l": 40, "r": 24, "t": 54, "b": 42},
        "xaxis": {"gridcolor": GRID},
        "yaxis": {"gridcolor": GRID},
        "legend": {"orientation": "h", "y": 1.02, "x": 0},
    }


def net_gex_by_strike_chart(by_strike: pd.DataFrame, spot: float, gamma_flip: float | None, call_wall: float | None, put_wall: float | None) -> go.Figure:
    fig = go.Figure()
    if not by_strike.empty:
        colors = ["#3ab54a" if value >= 0 else "#ef4444" for value in by_strike["Net GEX"]]
        fig.add_bar(x=by_strike["Strike"], y=by_strike["Net GEX"], marker_color=colors, name="Net GEX")
    fig.add_vline(x=spot, line_color="#f8fafc", line_width=2, annotation_text="Spot")
    if gamma_flip is not None:
        fig.add_vline(x=gamma_flip, line_color="#f59e0b", line_dash="dash", annotation_text="Gamma Flip")
    if call_wall is not None:
        fig.add_vline(x=call_wall, line_color=BRAND_GREEN, line_dash="dot", annotation_text="Call Wall")
    if put_wall is not None:
        fig.add_vline(x=put_wall, line_color="#ef4444", line_dash="dot", annotation_text="Put Wall")
    fig.update_layout(**_base_layout("Net Gamma Exposure by Strike"))
    return fig


def simulated_gex_chart(simulation: pd.DataFrame, spot: float, gamma_flip: float | None) -> go.Figure:
    fig = go.Figure()
    if not simulation.empty:
        fig.add_trace(
            go.Scatter(
                x=simulation["price_level"],
                y=simulation["total_gex"],
                mode="lines",
                name="Simulated Total GEX",
                line={"color": BRAND_GREEN, "width": 3},
            )
        )
    fig.add_hline(y=0, line_color="rgba(226,232,240,0.45)", line_dash="dot")
    fig.add_vline(x=spot, line_color="#f8fafc", line_width=2, annotation_text="Spot")
    if gamma_flip is not None:
        fig.add_vline(x=gamma_flip, line_color="#f59e0b", line_dash="dash", annotation_text="Gamma Flip")
    fig.update_layout(**_base_layout("Simulated Total GEX Across Price Levels"))
    return fig


def expiration_heatmap(gex_rows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not gex_rows.empty:
        heat = gex_rows.groupby(["strike", "expiration"], as_index=False)["signed_gex"].sum()
        pivot = heat.pivot(index="strike", columns="expiration", values="signed_gex").fillna(0)
        fig.add_trace(
            go.Heatmap(
                x=list(pivot.columns),
                y=list(pivot.index),
                z=pivot.values,
                colorscale=[[0, "#ef4444"], [0.5, "#111827"], [1, BRAND_GREEN]],
                colorbar={"title": "GEX"},
            )
        )
    fig.update_layout(**_base_layout("Strike x Expiration GEX Heatmap"))
    return fig

