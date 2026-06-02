import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def generate_transition_heatmap(matrix: np.ndarray, state_names: list) -> go.Figure:
    """Generate a Plotly heatmap of the Markov state transition probabilities matrix."""
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=state_names,
        y=state_names,
        colorscale="Viridis",
        zmin=0.0,
        zmax=1.0,
        text=[[f"{val:.2%}" for val in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 13, "family": "Inter", "color": "#ffffff"},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title={
            "text": "State Transition Probability Matrix",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "family": "Courier Prime"}
        },
        xaxis={"title": "Next State (To)", "side": "bottom", "gridcolor": "#1e3a5f"},
        yaxis={"title": "Current State (From)", "autorange": "reversed", "gridcolor": "#1e3a5f"},
        paper_bgcolor="#0d1b2e",
        plot_bgcolor="#0d1b2e",
        font={"color": "#e2e8f0", "family": "Inter"},
        height=320,
        margin={"l": 40, "r": 20, "t": 60, "b": 40}
    )
    return fig

def generate_regime_timeline(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Generate a Plotly price chart of the ticker with background bands colored by
    the classified Markov price and volatility regimes.
    """
    df = df.copy().sort_values("trade_date").reset_index(drop=True)
    
    fig = go.Figure()
    
    # Identify spans of regimes to add colored shapes (to prevent huge trace count overhead)
    # Price line
    fig.add_trace(go.Scatter(
        x=df["trade_date"],
        y=df["close"],
        name=ticker,
        line={"color": "#ffffff", "width": 2},
        hovertemplate="<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>"
    ))
    
    # State coloring mappings
    # Green = Bull, Gray = Sideways, Red = Bear
    color_map = {
        "BULL": "rgba(58, 181, 74, 0.12)",
        "SIDEWAYS": "rgba(148, 163, 184, 0.08)",
        "BEAR": "rgba(239, 68, 68, 0.12)"
    }
    
    # Identify contiguous blocks of states
    if not df.empty:
        curr_state = df.loc[0, "price_state"]
        start_date = df.loc[0, "trade_date"]
        
        for idx in range(1, len(df)):
            state = df.loc[idx, "price_state"]
            if state != curr_state or idx == len(df) - 1:
                end_date = df.loc[idx, "trade_date"]
                fig.add_vrect(
                    x0=start_date,
                    x1=end_date,
                    fillcolor=color_map.get(curr_state, "rgba(0,0,0,0)"),
                    opacity=1.0,
                    line_width=0,
                    layer="below"
                )
                start_date = end_date
                curr_state = state
                
        # Highlight HIGH_VOL states with subtle orange dots or vertical marker stripes on top/bottom
        high_vol_df = df[df["volatility_state"] == "HIGH_VOL"]
        if not high_vol_df.empty:
            fig.add_trace(go.Scatter(
                x=high_vol_df["trade_date"],
                y=high_vol_df["close"],
                mode="markers",
                name="High Volatility State",
                marker={"color": "#fb923c", "size": 5, "symbol": "circle-open"},
                hovertemplate="<b>%{x}</b><br>High Volatility Alert<extra></extra>"
            ))
            
    fig.update_layout(
        title={
            "text": f"{ticker} Price & Markov Regime Timeline",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "family": "Courier Prime"}
        },
        xaxis={"gridcolor": "#1e3a5f", "title": "Date"},
        yaxis={"gridcolor": "#1e3a5f", "title": "Close Price ($)"},
        paper_bgcolor="#0d1b2e",
        plot_bgcolor="#0d1b2e",
        font={"color": "#e2e8f0", "family": "Inter"},
        height=380,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
    )
    return fig

def generate_probability_trend(df_forecast: pd.DataFrame, ticker: str) -> go.Figure:
    """Generate a historical line chart of the state probabilities and the Markov Signal."""
    df_forecast = df_forecast.copy().sort_values("as_of_date").reset_index(drop=True)
    
    fig = go.Figure()
    
    # 1. Bull Probability
    fig.add_trace(go.Scatter(
        x=df_forecast["as_of_date"],
        y=df_forecast["bull_prob_1d"],
        name="Bull Prob (1D)",
        line={"color": "#3ab54a", "width": 1.5},
        hovertemplate="Bull Prob: %{y:.1%}<extra></extra>"
    ))
    
    # 2. Bear Probability
    fig.add_trace(go.Scatter(
        x=df_forecast["as_of_date"],
        y=df_forecast["bear_prob_1d"],
        name="Bear Prob (1D)",
        line={"color": "#ef4444", "width": 1.5},
        hovertemplate="Bear Prob: %{y:.1%}<extra></extra>"
    ))
    
    # 3. Sideways Probability
    fig.add_trace(go.Scatter(
        x=df_forecast["as_of_date"],
        y=df_forecast["sideways_prob_1d"],
        name="Sideways Prob (1D)",
        line={"color": "#94a3b8", "width": 1.5},
        hovertemplate="Sideways Prob: %{y:.1%}<extra></extra>"
    ))
    
    # 4. Markov Signal
    fig.add_trace(go.Scatter(
        x=df_forecast["as_of_date"],
        y=df_forecast["markov_signal"],
        name="Markov Signal Score",
        line={"color": "#fb923c", "width": 2, "dash": "dash"},
        hovertemplate="Markov Signal: %{y:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title={
            "text": f"{ticker} Probability Trends & Markov Signal",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "family": "Courier Prime"}
        },
        xaxis={"gridcolor": "#1e3a5f", "title": "Date"},
        yaxis={"gridcolor": "#1e3a5f", "title": "Value / Probability", "tickformat": ".0%"},
        paper_bgcolor="#0d1b2e",
        plot_bgcolor="#0d1b2e",
        font={"color": "#e2e8f0", "family": "Inter"},
        height=380,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
    )
    return fig

def generate_backtest_equity_curve(bt_results: dict, ticker: str) -> go.Figure:
    """Generate a Plotly chart showing the strategy equity curve vs benchmark."""
    dates = bt_results.get("dates", [])
    equity = bt_results.get("equity_curve", [])
    
    fig = go.Figure()
    
    if dates and equity:
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity,
            name="Markov Strategy",
            line={"color": "#3ab54a", "width": 2},
            hovertemplate="Markov Strategy: %{y:.2f}x<extra></extra>"
        ))
        
    fig.update_layout(
        title={
            "text": f"{ticker} Markov Strategy Growth vs Benchmark",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "family": "Courier Prime"}
        },
        xaxis={"gridcolor": "#1e3a5f", "title": "Date"},
        yaxis={"gridcolor": "#1e3a5f", "title": "Growth (Initial = 1.0)"},
        paper_bgcolor="#0d1b2e",
        plot_bgcolor="#0d1b2e",
        font={"color": "#e2e8f0", "family": "Inter"},
        height=350,
        margin={"l": 40, "r": 20, "t": 60, "b": 40}
    )
    return fig
