import plotly.graph_objects as go
import pandas as pd


def _base_layout(title: str, yaxis_title: str = "") -> dict:
    """Shared chart layout settings."""
    return dict(
        title=title,
        template="plotly_white",
        height=500,
        margin=dict(l=60, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title=yaxis_title),
        xaxis=dict(title=""),
        hovermode="x unified",
    )


def chart_rolling_win_pct(df: pd.DataFrame, x_col: str = "trade_num") -> go.Figure:
    """Chart 1: Rolling Win % + Std Dev Bands."""
    fig = go.Figure()
    x = df[x_col]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, w in enumerate([25, 50, 100]):
        col = f"win_pct_{w}"
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df[col], mode="lines",
                name=f"Rolling {w} Win %", line=dict(color=colors[i], width=1),
            ))

    fig.add_trace(go.Scatter(
        x=x, y=df["win_pct_avg"], mode="lines",
        name="Average", line=dict(color="black", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df["win_pct_upper"], mode="lines",
        name="Upper Band", line=dict(color="green", width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df["win_pct_lower"], mode="lines",
        name="Lower Band", line=dict(color="red", width=1, dash="dash"),
    ))

    layout = _base_layout("Rolling Win % + Standard Deviation", "Win %")
    layout["yaxis"]["tickformat"] = ".0%"
    fig.update_layout(**layout)
    return fig


def chart_rolling_gain(df: pd.DataFrame, x_col: str = "trade_num") -> go.Figure:
    """Chart 2: Rolling Avg Gain % + Std Dev Bands."""
    fig = go.Figure()
    x = df[x_col]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, w in enumerate([25, 50, 100, 200]):
        col = f"gain_avg_{w}"
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df[col], mode="lines",
                name=f"Rolling {w} Avg Gain", line=dict(color=colors[i], width=1),
            ))

    fig.add_trace(go.Scatter(
        x=x, y=df["gain_avg"], mode="lines",
        name="Average", line=dict(color="black", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df["gain_upper"], mode="lines",
        name="Upper Band", line=dict(color="green", width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df["gain_lower"], mode="lines",
        name="Lower Band", line=dict(color="red", width=1, dash="dash"),
    ))

    layout = _base_layout("Rolling Avg Gain % + Standard Deviation", "Gain %")
    layout["yaxis"]["tickformat"] = ".1%"
    fig.update_layout(**layout)
    return fig
