import streamlit as st
import pandas as pd

from calculations.equity_calcs import load_equity_curve, compute_all as eq_compute_all
from calculations.equity_calcs import compute_dd_distribution
from calculations.trades_calcs import load_trades, compute_all as tr_compute_all
from charts.equity_charts import (
    chart_equity_curve, chart_drawdown, chart_rolling_cagr,
    chart_equity_bollinger, chart_bb_consecutive, chart_volatility, chart_best_fit,
)
from charts.trades_charts import chart_rolling_win_pct, chart_rolling_gain
from reports.export_html import generate_report as generate_interactive_report
from reports.export_static import generate_static_report

st.set_page_config(page_title="Trading System Dashboard", layout="wide")
st.title("Trading System Performance Dashboard")

# ── Sidebar: Data Source ──────────────────────────────────────────────────────
st.sidebar.header("Data Source")
eq_upload = st.sidebar.file_uploader("Upload Equity Curve CSV", type=["csv"])
tr_upload = st.sidebar.file_uploader("Upload Trades List CSV", type=["csv"])

st.sidebar.divider()

# ── Sidebar: Equity Curve Parameters ─────────────────────────────────────────
st.sidebar.header("Equity Curve Parameters")
live_date = st.sidebar.date_input("System Live Start Date", value=pd.Timestamp("2025-08-25"))
live_date = pd.Timestamp(live_date)
dd_percentile = st.sidebar.slider("Drawdown Percentile", 1, 50, 30)
bb_std = st.sidebar.number_input("Bollinger Band Std Devs", 0.5, 10.0, 2.0, 0.5)
band_pct = st.sidebar.number_input("Best Fit ± %", 1.0, 100.0, 33.0, 1.0)
log_y = st.sidebar.checkbox("Log Scale (Equity Charts)", value=True)
eq_date_placeholder = st.sidebar.container()

st.sidebar.divider()

# ── Sidebar: Trades Parameters ────────────────────────────────────────────────
st.sidebar.header("Trades Parameters")
trade_std = st.sidebar.number_input("Trade Std Devs", 0.5, 10.0, 4.0, 0.5)


# ── Load & Compute ───────────────────────────────────────────────────────────
@st.cache_data
def get_equity_data(upload_bytes, dd_pct, bb, band):
    if upload_bytes is not None:
        import io
        df = load_equity_curve(io.BytesIO(upload_bytes))
    else:
        df = load_equity_curve("data/equity_curve.csv")
    df = eq_compute_all(df, dd_percentile=dd_pct, bb_std=bb, band_pct=band)
    return df


@st.cache_data
def get_trades_data(upload_bytes, t_std):
    if upload_bytes is not None:
        import io
        df = load_trades(io.BytesIO(upload_bytes))
    else:
        df = load_trades("data/trades_list.csv")
    df = tr_compute_all(df, trade_std=t_std)
    return df


eq_bytes = eq_upload.read() if eq_upload else None
tr_bytes = tr_upload.read() if tr_upload else None

eq_df = None
tr_df = None

try:
    eq_df = get_equity_data(eq_bytes, dd_percentile, bb_std, band_pct)
except ValueError as e:
    st.sidebar.warning(f"Equity CSV: {e}")

try:
    tr_df = get_trades_data(tr_bytes, trade_std)
except ValueError as e:
    st.sidebar.warning(f"Trades CSV: {e}")

# ── Date Range (Equity) ───────────────────────────────────────────────────────
if eq_df is not None:
    min_date = eq_df["Date"].min().date()
    max_date = eq_df["Date"].max().date()
    with eq_date_placeholder:
        date_range = st.slider(
            "Equity Curve Period",
            min_value=min_date, max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
        )
        recalc = st.checkbox("Recalculate Selected Period", value=False)
    start_dt = pd.Timestamp(date_range[0])
    end_dt = pd.Timestamp(date_range[1])

    # When recalculate is checked, recompute metrics on the selected slice
    if recalc:
        eq_slice = eq_df[(eq_df["Date"] >= start_dt) & (eq_df["Date"] <= end_dt)].copy()
        eq_slice = eq_slice[["Date", "equity"]].reset_index(drop=True)
        eq_view = eq_compute_all(eq_slice, dd_percentile=dd_percentile, bb_std=bb_std, band_pct=band_pct)
    else:
        eq_view = eq_df[(eq_df["Date"] >= start_dt) & (eq_df["Date"] <= end_dt)]

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_eq, tab_tr = st.tabs(["Equity Curve Dashboard", "Trades Dashboard"])

with tab_eq:
    if eq_df is None:
        st.warning("Equity data not available. Please upload a valid Equity Curve CSV.")
    else:
        st.plotly_chart(chart_equity_curve(eq_view, live_date, log_y), use_container_width=True)
        st.plotly_chart(chart_drawdown(eq_view, live_date), use_container_width=True)

        # Drawdown Percentile Distribution (right below drawdown chart)
        dd_src = eq_view if recalc else eq_df
        dist, cur_dd, cur_dd_pct = compute_dd_distribution(dd_src)
        with st.expander("Drawdown Percentile Distribution"):
            col1, col2 = st.columns(2)
            col1.metric("Current Drawdown", f"{cur_dd:.2%}")
            col2.metric("Current DD Percentile", f"{cur_dd_pct:.1%}")
            st.dataframe(dist.style.format({"percentile": "{:.0%}", "drawdown": "{:.4%}"}))

        st.plotly_chart(chart_rolling_cagr(eq_view, live_date), use_container_width=True)
        st.plotly_chart(chart_equity_bollinger(eq_view, log_y, live_date), use_container_width=True)
        st.plotly_chart(chart_bb_consecutive(eq_view, live_date), use_container_width=True)
        st.plotly_chart(chart_volatility(eq_view, live_date), use_container_width=True)
        st.plotly_chart(chart_best_fit(eq_view, log_y, live_date), use_container_width=True)

        # Best Fit Floor/Ceiling stats
        fit_src = eq_view if recalc else eq_df
        if "pct_above_ceiling" in fit_src.columns:
            last_hist = fit_src.loc[fit_src["above_ceiling"].notna()].iloc[-1]
            col1, col2 = st.columns(2)
            col1.metric("% Time Spent Above Ceiling", f"{last_hist['pct_above_ceiling']:.1%}")
            col2.metric("% Time Spent Below Floor", f"{last_hist['pct_below_floor']:.1%}")


with tab_tr:
    if tr_df is None:
        st.warning("Trades data not available. Please upload a valid Trades List CSV.")
    else:
        total_trades = len(tr_df)
        trade_range = st.slider(
            "Trade Range",
            1, total_trades, (1, total_trades),
        )
        tr_view = tr_df[(tr_df["trade_num"] >= trade_range[0]) & (tr_df["trade_num"] <= trade_range[1])]

        x_col_option = st.radio("X-Axis", ["Trade Number", "Date (Exit)"], horizontal=True)
        x_col = "trade_num" if x_col_option == "Trade Number" else "DateOut"

        st.plotly_chart(chart_rolling_win_pct(tr_view, x_col), use_container_width=True)
        st.plotly_chart(chart_rolling_gain(tr_view, x_col), use_container_width=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("Export")


def _export_metrics():
    """Gather shared export metrics."""
    dd_src = eq_view if recalc else eq_df
    dist, cur_dd, cur_dd_pct = compute_dd_distribution(dd_src)
    fit_src = eq_view if recalc else eq_df
    pct_ac = fit_src["pct_above_ceiling"].iloc[-1] if "pct_above_ceiling" in fit_src.columns else 0.0
    pct_bf = fit_src["pct_below_floor"].iloc[-1] if "pct_below_floor" in fit_src.columns else 0.0
    return dist, cur_dd, cur_dd_pct, pct_ac, pct_bf


if st.sidebar.button("Export Interactive Report", use_container_width=True):
    if eq_df is None:
        st.sidebar.error("No equity data to export.")
    else:
        with st.spinner("Generating interactive report..."):
            eq_figs = [
                chart_equity_curve(eq_view, live_date, log_y),
                chart_drawdown(eq_view, live_date),
                chart_rolling_cagr(eq_view, live_date),
                chart_equity_bollinger(eq_view, log_y, live_date),
                chart_bb_consecutive(eq_view, live_date),
                chart_volatility(eq_view, live_date),
                chart_best_fit(eq_view, log_y, live_date),
            ]
            _tr_view = tr_view if tr_df is not None else None
            _x_col = x_col if tr_df is not None else "trade_num"
            tr_figs = []
            if tr_df is not None:
                tr_figs = [
                    chart_rolling_win_pct(_tr_view, _x_col),
                    chart_rolling_gain(_tr_view, _x_col),
                ]
            dist, cur_dd, cur_dd_pct, pct_ac, pct_bf = _export_metrics()
            html_str = generate_interactive_report(
                eq_view=eq_view, eq_full=eq_df, tr_view=_tr_view,
                live_date=live_date, log_y=log_y,
                dd_dist=dist, cur_dd=cur_dd, cur_dd_pct=cur_dd_pct,
                pct_above_ceiling=pct_ac, pct_below_floor=pct_bf,
                x_col=_x_col, equity_charts=eq_figs, trades_charts=tr_figs,
            )
            import os
            os.makedirs("data", exist_ok=True)
            out_path = os.path.join("data", "dashboard_report_interactive.html")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html_str)
            st.sidebar.success(f"Saved to {out_path}")

if st.sidebar.button("Export Static Report", use_container_width=True):
    if eq_df is None:
        st.sidebar.error("No equity data to export.")
    else:
        with st.spinner("Generating static report..."):
            _tr_view = tr_view if tr_df is not None else None
            _x_col = x_col if tr_df is not None else "trade_num"
            dist, cur_dd, cur_dd_pct, pct_ac, pct_bf = _export_metrics()
            html_str = generate_static_report(
                eq_view=eq_view, eq_full=eq_df, tr_view=_tr_view,
                live_date=live_date, log_y=log_y,
                dd_dist=dist, cur_dd=cur_dd, cur_dd_pct=cur_dd_pct,
                pct_above_ceiling=pct_ac, pct_below_floor=pct_bf,
                x_col=_x_col,
            )
            import os
            os.makedirs("data", exist_ok=True)
            out_path = os.path.join("data", "dashboard_report_static.html")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html_str)
            st.sidebar.success(f"Saved to {out_path}")
