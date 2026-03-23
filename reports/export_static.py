"""Static HTML report generation using matplotlib.

Produces a lightweight self-contained HTML with PNG chart images (~1-2 MB)
instead of the ~11 MB interactive Plotly version.
"""
import base64
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def _fig_to_base64(fig: plt.Figure, dpi: int = 100) -> str:
    """Render figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _img_tag(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1100px;">'


def _pct_fmt(x, _pos=None):
    return f"{x:.0%}"


def _pct_fmt1(x, _pos=None):
    return f"{x:.1%}"


def _setup_axes(ax, title: str, ylabel: str):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.8)


def _add_live_line(ax, live_date):
    if live_date is not None:
        ax.axvline(live_date, color="red", linestyle="--", linewidth=0.8, alpha=0.7)


# ── Individual chart renderers ────────────────────────────────────────────────

def _chart_equity_curve(df, live_date, log_y):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["Date"], df["equity"], color="#1f77b4", linewidth=1, label="Equity Curve")
    if log_y:
        ax.set_yscale("log")
    _add_live_line(ax, live_date)
    _setup_axes(ax, "System Equity Curve", "Equity")
    return fig


def _chart_drawdown(df, live_date):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(df["Date"], df["drawdown"], 0, color="red", alpha=0.15, label="Drawdown")
    ax.plot(df["Date"], df["drawdown"], color="red", linewidth=0.5, alpha=0.5)
    ax.plot(df["Date"], df["dd_avg_whole"], color="blue", linewidth=1, linestyle=":", label="Avg DD (Whole)")
    if "dd_upper_pct" in df.columns:
        ax.plot(df["Date"], df["dd_upper_pct"], color="green", linewidth=1, linestyle="--", label="Upper Pct")
        ax.plot(df["Date"], df["dd_lower_pct"], color="orange", linewidth=1, linestyle="--", label="Lower Pct")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt1))
    _add_live_line(ax, live_date)
    _setup_axes(ax, "System Drawdown", "Drawdown %")
    return fig


def _chart_rolling_cagr(df, live_date):
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = {"1y": "#1f77b4", "3y": "#ff7f0e", "5y": "#2ca02c", "7y": "#d62728", "10y": "#9467bd"}
    for n, c in colors.items():
        col = f"cagr_{n}"
        if col in df.columns:
            ax.plot(df["Date"], df[col], color=c, linewidth=1, label=f"{n.upper()} CAGR")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    _add_live_line(ax, live_date)
    _setup_axes(ax, "Rolling CAGR", "CAGR")
    return fig


def _chart_bollinger(df, log_y, live_date):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["Date"], df["equity"], color="#1f77b4", linewidth=1, label="Equity Curve")
    ax.plot(df["Date"], df["ma_avg"], color="orange", linewidth=1, label="MA Average")
    upper_col = "bb_upper_log" if log_y else "bb_upper"
    lower_col = "bb_lower_log" if log_y else "bb_lower"
    if upper_col in df.columns:
        ax.plot(df["Date"], df[upper_col], color="green", linewidth=1, linestyle="--", label="Upper Band")
        ax.plot(df["Date"], df[lower_col], color="red", linewidth=1, linestyle="--", label="Lower Band")
    if log_y:
        ax.set_yscale("log")
    _add_live_line(ax, live_date)
    _setup_axes(ax, "Equity Curve + Std Dev Bands", "Equity")
    return fig


def _chart_bb_consecutive(df, live_date):
    fig, ax = plt.subplots(figsize=(11, 4))
    if "days_above_bb" in df.columns:
        ax.bar(df["Date"], df["days_above_bb"], color="green", alpha=0.6, width=2, label="Days Above BB")
        ax.bar(df["Date"], -df["days_below_bb"], color="red", alpha=0.6, width=2, label="Days Below BB")
    _add_live_line(ax, live_date)
    _setup_axes(ax, "Std Dev Bands (Consecutive Days)", "Consecutive Days")
    return fig


def _chart_volatility(df, live_date):
    fig, ax = plt.subplots(figsize=(11, 4))
    traces = [("vol_25", "#1f77b4", "-"), ("vol_50", "#ff7f0e", "-"),
              ("vol_100", "#2ca02c", "-"), ("vol_200", "#d62728", "-"),
              ("vol_median", "#9467bd", "--")]
    names = ["25-Day", "50-Day", "100-Day", "200-Day", "Long-term Median"]
    for (col, c, ls), name in zip(traces, names):
        if col in df.columns:
            ax.plot(df["Date"], df[col], color=c, linewidth=1, linestyle=ls, label=name)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    _add_live_line(ax, live_date)
    _setup_axes(ax, "Rolling Volatility", "Volatility")
    return fig


def _chart_best_fit(df, log_y, live_date):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["Date"], df["equity"], color="#1f77b4", linewidth=1, label="Equity Curve")
    if "best_fit" in df.columns:
        ax.plot(df["Date"], df["best_fit"], color="black", linewidth=1, linestyle="--", label="Best Fit")
        ax.plot(df["Date"], df["best_fit_upper"], color="green", linewidth=1, linestyle=":", label="Upper Channel")
        ax.plot(df["Date"], df["best_fit_lower"], color="red", linewidth=1, linestyle=":", label="Lower Channel")
    if log_y:
        ax.set_yscale("log")
    _add_live_line(ax, live_date)
    _setup_axes(ax, "Best Fit Floor / Ceiling", "Equity")
    return fig


def _chart_rolling_win_pct(df, x_col):
    fig, ax = plt.subplots(figsize=(11, 4))
    x = df[x_col]
    for w, c in [(25, "#1f77b4"), (50, "#ff7f0e"), (100, "#2ca02c")]:
        col = f"win_pct_{w}"
        if col in df.columns:
            ax.plot(x, df[col], color=c, linewidth=1, label=f"Rolling {w}")
    if "win_pct_avg" in df.columns:
        ax.plot(x, df["win_pct_avg"], color="black", linewidth=1.5, label="Average")
    if "win_pct_upper" in df.columns:
        ax.plot(x, df["win_pct_upper"], color="green", linewidth=1, linestyle="--", label="Upper Band")
        ax.plot(x, df["win_pct_lower"], color="red", linewidth=1, linestyle="--", label="Lower Band")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    _setup_axes(ax, "Rolling Win % + Std Dev", "Win %")
    return fig


def _chart_rolling_gain(df, x_col):
    fig, ax = plt.subplots(figsize=(11, 4))
    x = df[x_col]
    for w, c in [(25, "#1f77b4"), (50, "#ff7f0e"), (100, "#2ca02c"), (200, "#d62728")]:
        col = f"gain_avg_{w}"
        if col in df.columns:
            ax.plot(x, df[col], color=c, linewidth=1, label=f"Rolling {w}")
    if "gain_avg" in df.columns:
        ax.plot(x, df["gain_avg"], color="black", linewidth=1.5, label="Average")
    if "gain_upper" in df.columns:
        ax.plot(x, df["gain_upper"], color="green", linewidth=1, linestyle="--", label="Upper Band")
        ax.plot(x, df["gain_lower"], color="red", linewidth=1, linestyle="--", label="Lower Band")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt1))
    _setup_axes(ax, "Rolling Avg Gain % + Std Dev", "Gain %")
    return fig


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _metric_card(label: str, value: str) -> str:
    return (
        f'<div class="metric"><span class="metric-label">{label}</span>'
        f'<span class="metric-value">{value}</span></div>'
    )


def _html_table(df: pd.DataFrame, col_formats: dict | None = None) -> str:
    col_formats = col_formats or {}
    rows = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>\n"
    for _, row in df.iterrows():
        cells = ""
        for col in df.columns:
            fmt = col_formats.get(col)
            text = fmt.format(row[col]) if fmt else str(row[col])
            cells += f"<td>{text}</td>"
        rows += f"<tr>{cells}</tr>\n"
    return f'<table class="data-table">{rows}</table>'


_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px 40px;
       background: #fafafa; color: #333; }
h1 { text-align: center; margin-bottom: 4px; }
.subtitle { text-align: center; color: #888; margin-bottom: 30px; }
h2 { border-bottom: 2px solid #ddd; padding-bottom: 6px; margin-top: 40px; }
h3 { color: #555; margin-top: 24px; }
.metrics-row { display: flex; gap: 20px; margin: 12px 0 16px; }
.metric { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
           padding: 12px 20px; flex: 1; text-align: center; }
.metric-label { display: block; font-size: 0.85em; color: #888; }
.metric-value { display: block; font-size: 1.3em; font-weight: 600; margin-top: 4px; }
.data-table { border-collapse: collapse; margin: 10px 0; }
.data-table th, .data-table td { border: 1px solid #ddd; padding: 6px 14px; text-align: center; }
.data-table th { background: #f0f0f0; font-weight: 600; }
.chart-block { margin: 16px 0; text-align: center; }
@media print { body { padding: 10px; } .chart-block { page-break-inside: avoid; } }
"""


# ── Main entry point ─────────────────────────────────────────────────────────

def generate_static_report(
    eq_view: pd.DataFrame,
    eq_full: pd.DataFrame,
    tr_view: pd.DataFrame | None,
    live_date: pd.Timestamp,
    log_y: bool,
    dd_dist: pd.DataFrame,
    cur_dd: float,
    cur_dd_pct: float,
    pct_above_ceiling: float,
    pct_below_floor: float,
    x_col: str,
) -> str:
    """Generate a lightweight static HTML report with matplotlib chart images."""

    date_min = eq_view["Date"].min().strftime("%Y-%m-%d")
    date_max = eq_view["Date"].max().strftime("%Y-%m-%d")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts: list[str] = []
    parts.append("<h1>Trading System Performance Report</h1>")
    parts.append(
        f'<p class="subtitle">Period: {date_min} &mdash; {date_max} &nbsp;|&nbsp; '
        f"Generated: {generated} &nbsp;|&nbsp; Static Report</p>"
    )

    # ── Equity Section ────────────────────────────────────────────────────────
    parts.append("<h2>Equity Curve Dashboard</h2>")

    chart_funcs = [
        ("System Equity Curve", lambda: _chart_equity_curve(eq_view, live_date, log_y)),
        ("System Drawdown", lambda: _chart_drawdown(eq_view, live_date)),
        ("Rolling CAGR", lambda: _chart_rolling_cagr(eq_view, live_date)),
        ("Equity Curve + Std Dev Bands", lambda: _chart_bollinger(eq_view, log_y, live_date)),
        ("Std Dev Bands (Consecutive Days)", lambda: _chart_bb_consecutive(eq_view, live_date)),
        ("Rolling Volatility", lambda: _chart_volatility(eq_view, live_date)),
        ("Best Fit Floor / Ceiling", lambda: _chart_best_fit(eq_view, log_y, live_date)),
    ]

    for i, (title, make_fig) in enumerate(chart_funcs):
        parts.append(f"<h3>{title}</h3>")
        fig = make_fig()
        parts.append(f'<div class="chart-block">{_img_tag(_fig_to_base64(fig))}</div>')

        if i == 1:  # after drawdown
            parts.append('<div class="metrics-row">')
            parts.append(_metric_card("Current Drawdown", f"{cur_dd:.2%}"))
            parts.append(_metric_card("Current DD Percentile", f"{cur_dd_pct:.1%}"))
            parts.append("</div>")
            parts.append(_html_table(dd_dist, {
                "percentile": "{:.0%}", "drawdown": "{:.4%}",
            }))

        if i == 6:  # after best fit
            parts.append('<div class="metrics-row">')
            parts.append(_metric_card("% Time Above Ceiling", f"{pct_above_ceiling:.1%}"))
            parts.append(_metric_card("% Time Below Floor", f"{pct_below_floor:.1%}"))
            parts.append("</div>")

    # ── Trades Section ────────────────────────────────────────────────────────
    if tr_view is not None:
        parts.append("<h2>Trades Dashboard</h2>")

        tr_charts = [
            ("Rolling Win % + Std Dev", lambda: _chart_rolling_win_pct(tr_view, x_col)),
            ("Rolling Avg Gain % + Std Dev", lambda: _chart_rolling_gain(tr_view, x_col)),
        ]
        for title, make_fig in tr_charts:
            parts.append(f"<h3>{title}</h3>")
            fig = make_fig()
            parts.append(f'<div class="chart-block">{_img_tag(_fig_to_base64(fig))}</div>')

    body = "\n".join(parts)

    return (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        "<title>Trading System Performance Report (Static)</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        "</body>\n</html>"
    )
