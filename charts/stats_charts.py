"""Plotly charts for statistical analysis tab."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as sp_stats


def _base_layout(title: str, yaxis_title: str = "") -> dict:
    return dict(
        title=title,
        template="plotly_white",
        height=450,
        margin=dict(l=60, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title=yaxis_title),
        xaxis=dict(title=""),
        hovermode="x unified",
    )


# ── Scope 1: Health Check Charts ─────────────────────────────────────────────

def chart_return_distribution(returns: pd.Series, mean_val: float,
                               median_val: float) -> go.Figure:
    """Density plot with mean and median vertical lines.

    X-axis is clipped to 1st–99th percentile to keep the chart readable;
    extreme outliers are counted in the histogram but don't stretch the view.
    """
    fig = go.Figure()

    # Clip display range to 1st-99th percentile
    r = returns.dropna().values
    p1, p99 = np.percentile(r, [1, 99])
    x_lo = p1 - 0.1 * abs(p1)
    x_hi = p99 + 0.1 * abs(p99)
    n_outliers = int(((returns < x_lo) | (returns > x_hi)).sum())

    fig.add_trace(go.Histogram(
        x=returns, nbinsx=80, histnorm="probability density",
        marker_color="rgba(31,119,180,0.5)", name="Return Distribution",
    ))
    # KDE overlay (computed on full data, displayed in clipped range)
    if len(r) > 2:
        kde = sp_stats.gaussian_kde(r)
        x_kde = np.linspace(x_lo, x_hi, 200)
        fig.add_trace(go.Scatter(
            x=x_kde, y=kde(x_kde), mode="lines",
            name="KDE", line=dict(color="#1f77b4", width=2),
        ))
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_val:.2%}")
    fig.add_vline(x=median_val, line_dash="dot", line_color="green",
                  annotation_text=f"Median: {median_val:.2%}")
    title = "Trade Return Distribution (Density)"
    if n_outliers > 0:
        title += f"  ({n_outliers} outliers beyond view)"
    layout = _base_layout(title, "Density")
    layout["xaxis"] = dict(title="PctGain", tickformat=".0%",
                           range=[x_lo, x_hi])
    fig.update_layout(**layout)
    return fig


def chart_qq_plot(returns: pd.Series) -> go.Figure:
    """Q-Q plot against normal distribution."""
    r = np.sort(returns.dropna().values)
    n = len(r)
    theoretical = sp_stats.norm.ppf(np.linspace(0.01, 0.99, n))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical, y=r, mode="markers",
        marker=dict(size=3, color="#1f77b4"), name="Trades",
    ))
    # Reference line
    lo, hi = theoretical.min(), theoretical.max()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[r.mean() + r.std() * lo, r.mean() + r.std() * hi],
        mode="lines", line=dict(color="red", dash="dash"), name="Normal Ref",
    ))
    layout = _base_layout("Q-Q Plot (vs Normal Distribution)", "Sample Quantiles")
    layout["xaxis"] = dict(title="Theoretical Quantiles")
    fig.update_layout(**layout)
    return fig


# ── Scope 2: Fragility Charts ────────────────────────────────────────────────

def chart_box_plot(profits: pd.Series, label: str = "Profit") -> go.Figure:
    """Box plot of trade profits/returns showing outliers."""
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=profits, name=label, boxpoints="outliers",
        marker_color="#1f77b4", line_color="#1f77b4",
    ))
    layout = _base_layout(f"Trade {label} — Outlier Box Plot", label)
    fig.update_layout(**layout)
    return fig


def chart_annual_returns(yearly: pd.DataFrame) -> go.Figure:
    """Bar chart of net return by year, highlighting outlier years."""
    fig = go.Figure()
    colors = ["red" if o else "#1f77b4"
              for o in yearly["outlier_year"]]
    fig.add_trace(go.Bar(
        x=yearly["year"], y=yearly["net_return"],
        marker_color=colors, name="Net Return",
    ))

    mu = yearly["net_return"].mean()
    sigma = yearly["net_return"].std()
    fig.add_hline(y=mu, line_dash="dash", line_color="gray",
                  annotation_text=f"Mean: {mu:.2%}")
    if sigma > 0:
        fig.add_hline(y=mu + 2 * sigma, line_dash="dot", line_color="green",
                      annotation_text="+2σ")
        fig.add_hline(y=mu - 2 * sigma, line_dash="dot", line_color="red",
                      annotation_text="-2σ")

    layout = _base_layout("Annual Net Returns", "Net Return")
    layout["yaxis"]["tickformat"] = ".0%"
    layout["xaxis"] = dict(title="Year", dtick=1)
    fig.update_layout(**layout)
    return fig


def chart_equity_drawdown_shading(cum_equity: pd.Series, peak: pd.Series,
                                   in_dd: pd.Series,
                                   dates: pd.Series | None = None) -> go.Figure:
    """Equity curve with drawdown periods shaded."""
    x = dates if dates is not None else cum_equity.index
    fig = go.Figure()

    # Shade DD regions
    dd_starts = []
    in_region = False
    for i in range(len(in_dd)):
        if in_dd.iloc[i] and not in_region:
            dd_starts.append(i)
            in_region = True
        elif not in_dd.iloc[i] and in_region:
            fig.add_vrect(
                x0=x.iloc[dd_starts[-1]], x1=x.iloc[i],
                fillcolor="red", opacity=0.08, line_width=0,
            )
            in_region = False
    if in_region:
        fig.add_vrect(
            x0=x.iloc[dd_starts[-1]], x1=x.iloc[-1],
            fillcolor="red", opacity=0.08, line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=x, y=cum_equity, mode="lines",
        name="Cumulative Equity", line=dict(color="#1f77b4", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=peak, mode="lines",
        name="Peak", line=dict(color="gray", width=1, dash="dot"),
    ))

    layout = _base_layout("Equity Curve — Drawdown Periods Shaded", "Equity (Cumulative)")
    fig.update_layout(**layout)
    return fig
