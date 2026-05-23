"""Plotly visuals for the Gamma Flip Line / GEX Engine."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


BRAND_GREEN = "#3ab54a"
BRAND_BLUE = "#1a3a8f"
TEXT = "#e2e8f0"
GRID = "rgba(148,163,184,0.12)"


def _base_layout(title: str) -> dict:
    return {
        "title": {
            "text": title,
            "font": {"size": 15, "family": "Outfit, Inter, sans-serif", "weight": "bold"}
        },
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(10, 20, 38, 0.4)",
        "font": {"color": TEXT, "family": "Inter, sans-serif"},
        "margin": {"l": 50, "r": 30, "t": 60, "b": 50},
        "xaxis": {
            "gridcolor": GRID,
            "showspikes": True,
            "spikethickness": 1,
            "spikedash": "dot",
            "spikecolor": "rgba(226,232,240,0.3)",
            "spikemode": "across"
        },
        "yaxis": {
            "gridcolor": GRID,
            "showspikes": True,
            "spikethickness": 1,
            "spikedash": "dot",
            "spikecolor": "rgba(226,232,240,0.3)",
            "spikemode": "across"
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)"
        },
        "hovermode": "x unified",
    }


def net_gex_by_strike_chart(
    by_strike: pd.DataFrame,
    spot: float,
    gamma_flip: float | None,
    call_wall: float | None,
    put_wall: float | None
) -> go.Figure:
    fig = go.Figure()
    if not by_strike.empty:
        # Plot Call GEX (Positive green bars)
        fig.add_trace(
            go.Bar(
                x=by_strike["Strike"],
                y=by_strike["Call GEX"],
                name="Call GEX",
                marker_color="rgba(58, 181, 74, 0.6)",
                hovertemplate="Call GEX: %{y:,.0f}<extra></extra>",
            )
        )
        # Plot Put GEX (Negative red bars)
        fig.add_trace(
            go.Bar(
                x=by_strike["Strike"],
                y=by_strike["Put GEX"],
                name="Put GEX",
                marker_color="rgba(239, 68, 68, 0.6)",
                hovertemplate="Put GEX: %{y:,.0f}<extra></extra>",
            )
        )
        # Plot Net GEX (Bold line overlay)
        fig.add_trace(
            go.Scatter(
                x=by_strike["Strike"],
                y=by_strike["Net GEX"],
                name="Net GEX",
                line=dict(color="#38bdf8", width=2.5),
                mode="lines+markers",
                marker=dict(size=4, color="#38bdf8"),
                hovertemplate="Net GEX: %{y:,.0f}<extra></extra>",
            )
        )

    # Indicator vertical lines
    fig.add_vline(
        x=spot,
        line_color="#f8fafc",
        line_width=2,
        annotation_text=" SPOT",
        annotation_position="top right",
        annotation_font=dict(color="#f8fafc", size=10, family="Outfit, sans-serif")
    )
    if gamma_flip is not None:
        fig.add_vline(
            x=gamma_flip,
            line_color="#f59e0b",
            line_dash="dash",
            annotation_text=" FLIP",
            annotation_position="top left",
            annotation_font=dict(color="#f59e0b", size=10, family="Outfit, sans-serif")
        )
    if call_wall is not None:
        fig.add_vline(
            x=call_wall,
            line_color=BRAND_GREEN,
            line_dash="dot",
            annotation_text=" CALL WALL",
            annotation_position="top right",
            annotation_font=dict(color=BRAND_GREEN, size=10, family="Outfit, sans-serif")
        )
    if put_wall is not None:
        fig.add_vline(
            x=put_wall,
            line_color="#ef4444",
            line_dash="dot",
            annotation_text=" PUT WALL",
            annotation_position="top left",
            annotation_font=dict(color="#ef4444", size=10, family="Outfit, sans-serif")
        )

    layout = _base_layout("Gamma Exposure Profile by Strike")
    layout["xaxis"]["title"] = "Strike Price"
    layout["yaxis"]["title"] = "Gamma Exposure ($ / point)"
    fig.update_layout(**layout, barmode="relative")
    return fig


def simulated_gex_chart(simulation: pd.DataFrame, spot: float, gamma_flip: float | None) -> go.Figure:
    fig = go.Figure()
    if not simulation.empty:
        # Shaded Positive Gamma Zone (above 0)
        fig.add_trace(
            go.Scatter(
                x=simulation["price_level"],
                y=simulation["total_gex"].clip(lower=0),
                fill="tozeroy",
                fillcolor="rgba(58, 181, 74, 0.15)",
                line=dict(width=0),
                hoverinfo="skip",
                name="Positive Gamma Regime",
            )
        )
        # Shaded Negative Gamma Zone (below 0)
        fig.add_trace(
            go.Scatter(
                x=simulation["price_level"],
                y=simulation["total_gex"].clip(upper=0),
                fill="tozeroy",
                fillcolor="rgba(239, 68, 68, 0.15)",
                line=dict(width=0),
                hoverinfo="skip",
                name="Negative Gamma Regime",
            )
        )
        # Main Line Trace
        fig.add_trace(
            go.Scatter(
                x=simulation["price_level"],
                y=simulation["total_gex"],
                mode="lines",
                name="Simulated GEX Profile",
                line=dict(color="#f8fafc", width=3.5),
                hovertemplate="Total GEX: %{y:,.0f}<extra></extra>"
            )
        )

    fig.add_hline(y=0, line_color="rgba(226,232,240,0.45)", line_dash="dot")
    fig.add_vline(
        x=spot,
        line_color="#f8fafc",
        line_width=2,
        annotation_text=" SPOT",
        annotation_position="top right",
        annotation_font=dict(color="#f8fafc", size=10, family="Outfit, sans-serif")
    )
    if gamma_flip is not None:
        fig.add_vline(
            x=gamma_flip,
            line_color="#f59e0b",
            line_dash="dash",
            annotation_text=" FLIP",
            annotation_position="top left",
            annotation_font=dict(color="#f59e0b", size=10, family="Outfit, sans-serif")
        )

    layout = _base_layout("Simulated Total GEX across Price Levels")
    layout["xaxis"]["title"] = "Underlying Price Level"
    layout["yaxis"]["title"] = "Total Gamma Exposure ($ / point)"
    fig.update_layout(**layout)
    return fig


def expiration_heatmap(gex_rows: pd.DataFrame, spot: float | None = None) -> go.Figure:
    fig = go.Figure()
    if not gex_rows.empty:
        # Zoom in to strikes around Spot (+/- 10%)
        if spot is not None and spot > 0:
            min_strike = spot * 0.90
            max_strike = spot * 1.10
            filtered_rows = gex_rows[(gex_rows["strike"] >= min_strike) & (gex_rows["strike"] <= max_strike)]
            if filtered_rows.empty:
                filtered_rows = gex_rows
        else:
            filtered_rows = gex_rows

        heat = filtered_rows.groupby(["strike", "expiration"], as_index=False)["signed_gex"].sum()
        if not heat.empty:
            pivot = heat.pivot(index="strike", columns="expiration", values="signed_gex").fillna(0)
            fig.add_trace(
                go.Heatmap(
                    x=list(pivot.columns),
                    y=list(pivot.index),
                    z=pivot.values,
                    colorscale=[
                        [0.0, "rgb(239, 68, 68)"],     # Red
                        [0.5, "rgb(15, 23, 42)"],      # Dark slate blue midpoint (0 GEX)
                        [1.0, "rgb(58, 181, 74)"]      # Green
                    ],
                    zmid=0,
                    colorbar={"title": "GEX ($)"},
                    hovertemplate="Expiration: %{x}<br>Strike: $%{y:.2f}<br>GEX: $%{z:,.0f}<extra></extra>"
                )
            )

    layout = _base_layout("Strike x Expiration GEX Heatmap")
    layout["hovermode"] = "closest"
    layout["yaxis"]["title"] = "Strike Price"
    layout["xaxis"]["title"] = "Expiration Date"
    fig.update_layout(**layout)
    return fig
