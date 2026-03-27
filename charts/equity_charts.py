import plotly.graph_objects as go
import pandas as pd


def _add_live_line(fig: go.Figure, live_date):
    """Add a vertical dashed red line at the live start date."""
    if live_date is not None:
        fig.add_shape(
            type="line", x0=live_date, x1=live_date, y0=0, y1=1,
            yref="paper", line=dict(color="red", dash="dash", width=1),
        )


def _base_layout(title: str, yaxis_title: str = "", log_y: bool = False) -> dict:
    """Shared chart layout settings."""
    return dict(
        title=title,
        template="plotly_white",
        height=500,
        margin=dict(l=60, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title=yaxis_title, type="log" if log_y else "linear"),
        xaxis=dict(title=""),
        hovermode="x unified",
    )


def chart_equity_curve(df: pd.DataFrame, live_date: pd.Timestamp, log_y: bool) -> go.Figure:
    """Chart 1: System Equity Curve with live start date marker."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["equity"], mode="lines",
        name="Equity Curve", line=dict(color="#1f77b4", width=1.5),
    ))
    _add_live_line(fig, live_date)
    if live_date is not None:
        fig.add_annotation(
            x=live_date, y=1, yref="paper", text="Live Start",
            showarrow=False, yanchor="bottom", font=dict(color="red", size=10),
        )
    fig.update_layout(**_base_layout("System Equity Curve", "Equity", log_y))
    return fig


def chart_drawdown(df: pd.DataFrame, live_date=None) -> go.Figure:
    """Chart 2: Drawdown % with rolling avg and percentile bands."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["drawdown"], fill="tozeroy",
        name="Drawdown", line=dict(color="rgba(255,0,0,0.5)", width=0.5),
        fillcolor="rgba(255,0,0,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["dd_avg_whole"], mode="lines",
        name="Avg DD (Whole Period)", line=dict(color="blue", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["dd_upper_pct"], mode="lines",
        name="Upper 5Y DD Percentile", line=dict(color="green", width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["dd_lower_pct"], mode="lines",
        name="Lower 5Y DD Percentile", line=dict(color="orange", width=1, dash="dash"),
    ))
    _add_live_line(fig, live_date)
    fig.update_layout(**_base_layout("System Drawdown", "Drawdown %"))
    fig.update_yaxes(tickformat=".1%")
    return fig


def chart_rolling_cagr(df: pd.DataFrame, live_date=None) -> go.Figure:
    """Chart 3: Rolling CAGR for 1, 3, 5, 7, 10 year windows."""
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, n in enumerate([1, 3, 5, 7, 10]):
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df[f"cagr_{n}y"], mode="lines",
            name=f"{n}Y CAGR", line=dict(color=colors[i], width=1.2),
        ))
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=0.5)
    _add_live_line(fig, live_date)
    fig.update_layout(**_base_layout("Rolling CAGR", "CAGR"))
    fig.update_yaxes(tickformat=".0%")
    return fig


def chart_equity_bollinger(df: pd.DataFrame, log_y: bool = False, live_date=None) -> go.Figure:
    """Chart 4: Equity Curve + Bollinger Bands (on MA average).

    Uses log-space bands when log_y is True for symmetric visual display.
    """
    upper_col = "bb_upper_log" if log_y else "bb_upper"
    lower_col = "bb_lower_log" if log_y else "bb_lower"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["equity"], mode="lines",
        name="Equity Curve", line=dict(color="#1f77b4", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["ma_avg"], mode="lines",
        name="MA Average (25/50/100/200)", line=dict(color="orange", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df[upper_col], mode="lines",
        name="Upper BB", line=dict(color="green", width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df[lower_col], mode="lines",
        name="Lower BB", line=dict(color="red", width=1, dash="dash"),
    ))
    _add_live_line(fig, live_date)
    fig.update_layout(**_base_layout("System Equity Curve + Std Dev Bands", "Equity", log_y))
    return fig


def chart_bb_consecutive(df: pd.DataFrame, live_date=None) -> go.Figure:
    """Chart 5: Consecutive days above/below Bollinger Bands."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["days_above_bb"],
        name="Days Above Upper BB", marker_color="green",
    ))
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["days_below_bb"],
        name="Days Below Lower BB", marker_color="red",
    ))
    _add_live_line(fig, live_date)
    fig.update_layout(**_base_layout("Standard Deviation Bands", "Consecutive Days"))
    fig.update_layout(barmode="relative", bargap=0)
    return fig


def chart_volatility(df: pd.DataFrame, live_date=None) -> go.Figure:
    """Chart 6: Rolling annualized volatility."""
    fig = go.Figure()
    series = [
        ("vol_25", "25-Day Vol", "#1f77b4"),
        ("vol_50", "50-Day Vol", "#ff7f0e"),
        ("vol_100", "100-Day Vol", "#2ca02c"),
        ("vol_200", "200-Day Vol", "#d62728"),
        ("vol_median", "Long-term Median", "#9467bd"),
    ]
    for col, name, color in series:
        dash = "dash" if col == "vol_median" else "solid"
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df[col], mode="lines",
            name=name, line=dict(color=color, width=1.2, dash=dash),
        ))
    _add_live_line(fig, live_date)
    fig.update_layout(**_base_layout("Rolling Volatility (Annualized)", "Volatility"))
    fig.update_yaxes(tickformat=".0%")
    return fig


def chart_best_fit(df: pd.DataFrame, log_y: bool, live_date=None) -> go.Figure:
    """Chart 7: Equity Curve + Best Fit Floor/Ceiling."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["equity"], mode="lines",
        name="Equity Curve", line=dict(color="#1f77b4", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["best_fit"], mode="lines",
        name="Best Fit Line", line=dict(color="black", width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["best_fit_upper"], mode="lines",
        name="Upper Channel", line=dict(color="green", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["best_fit_lower"], mode="lines",
        name="Lower Channel", line=dict(color="red", width=1, dash="dot"),
    ))
    _add_live_line(fig, live_date)
    fig.update_layout(**_base_layout(
        "System Equity Curve + Best Fit Floor / Ceiling", "Equity", log_y,
    ))
    return fig
