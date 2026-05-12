import streamlit as st
import pandas as pd
import numpy as np

pd.set_option("styler.render.max_elements", 500_000)

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
from calculations.stats_calcs import (
    load_trades_extended, compute_health_check, outlier_stress_test,
    annual_returns, regime_performance, profit_concentration,
    equity_dip_analysis, significance_decay, fragility_verdict,
)
from charts.stats_charts import (
    chart_return_distribution, chart_qq_plot, chart_box_plot,
    chart_annual_returns, chart_equity_drawdown_shading,
)

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
tab_eq, tab_tr, tab_stats, tab_data = st.tabs(["Equity Curve Dashboard", "Trades Dashboard", "Statistical Analysis", "Data Sheet"])

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

with tab_stats:
    st.header("Statistical Analysis — Strategy Health Check & Fragility Test")
    if tr_df is None:
        st.warning("Trades data not available. Please upload a valid Trades List CSV.")
    else:
        # Load extended data (with Profit, MAE, MFE if available)
        @st.cache_data
        def get_stats_data(upload_bytes):
            if upload_bytes is not None:
                import io
                return load_trades_extended(io.BytesIO(upload_bytes))
            return load_trades_extended("data/trades_list.csv")

        try:
            stats_df = get_stats_data(tr_bytes)
        except ValueError as e:
            st.error(f"Cannot run statistical analysis: {e}")
            stats_df = None

        if stats_df is not None:
            returns = stats_df["PctGain"].dropna()

            # ── SCOPE 1: Statistical Health Check ─────────────────────────
            st.subheader("1. Statistical Significance Tests")
            hc = compute_health_check(returns)

            # Summary table
            summary_data = {
                "Metric": ["N Trades", "Win Rate", "Mean Return", "Median Return",
                           "Profit Factor", "Skewness", "Kurtosis", "SEM"],
                "Value": [
                    f"{hc['n_trades']:,}",
                    f"{hc['win_rate']:.2%}",
                    f"{hc['mean_return']:.4%}",
                    f"{hc['median_return']:.4%}",
                    f"{hc['profit_factor']:.2f}",
                    f"{hc['skewness']:.4f}",
                    f"{hc['kurtosis']:.4f}",
                    f"{hc['sem']:.6f}",
                ],
            }
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.caption("Core Metrics")
                st.dataframe(pd.DataFrame(summary_data), hide_index=True)

            # P-value dashboard
            pval_data = {
                "Test": ["T-test (H₀: μ=0)", "Wilcoxon Signed-Rank",
                         "Binomial (H₀: WR=50%)", "Jarque-Bera (Normality)",
                         "Runs Test (Independence)"],
                "Statistic": [
                    f"{hc['t_stat']:.4f}",
                    f"{hc.get('wilcoxon_stat', 'N/A')}",
                    "—",
                    f"{hc['jb_stat']:.2f}",
                    f"{hc['runs']}",
                ],
                "p-value": [
                    f"{hc['t_pval']:.6f}",
                    f"{hc['wilcoxon_pval']:.6f}" if not np.isnan(hc.get('wilcoxon_pval', np.nan)) else "N/A",
                    f"{hc['binom_pval']:.6f}",
                    f"{hc['jb_pval']:.6f}",
                    f"{hc['runs_pval']:.6f}" if not np.isnan(hc.get('runs_pval', np.nan)) else "N/A",
                ],
                "Significant (p<0.05)": [
                    "✓" if hc['t_pval'] < 0.05 else "✗",
                    "✓" if hc.get('wilcoxon_pval', 1) < 0.05 else "✗",
                    "✓" if hc['binom_pval'] < 0.05 else "✗",
                    "✓" if hc['jb_pval'] < 0.05 else "✗",
                    "✓" if hc.get('runs_pval', 1) < 0.05 else "✗",
                ],
            }
            with col_s2:
                st.caption("P-Value Dashboard")
                st.dataframe(pd.DataFrame(pval_data), hide_index=True)

            # Autocorrelation & Bootstrap
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.caption("Autocorrelation")
                acf_data = {"Lag": [1, 2, 5],
                            "ACF": [f"{hc.get(f'acf_lag{l}', 0):.4f}" for l in [1, 2, 5]]}
                st.dataframe(pd.DataFrame(acf_data), hide_index=True)
            with col_a2:
                st.caption("Bootstrap 95% CI for Mean Return")
                st.metric("Mean (Bootstrap)", f"{hc['bootstrap_mean']:.4%}")
                st.metric("95% CI", f"[{hc['ci_lower']:.4%}, {hc['ci_upper']:.4%}]")

            # Distribution charts
            st.subheader("2. Fat-Tail & Distribution Analysis")
            mean_vs = hc['mean_return'] - hc['median_return']
            if abs(mean_vs) > 0.001:
                st.info(f"Mean–Median gap: {mean_vs:.4%}. "
                        f"{'Positive skew suggests the strategy benefits from large outlier wins.' if mean_vs > 0 else 'Negative skew suggests vulnerability to large losses.'}")

            st.plotly_chart(chart_return_distribution(returns, hc['mean_return'], hc['median_return']),
                           use_container_width=True)
            st.plotly_chart(chart_qq_plot(returns), use_container_width=True)

            # ── SCOPE 2: Fragility & Robustness ──────────────────────────
            st.subheader("3. Fragility & Outlier Stress Test")
            stress = outlier_stress_test(stats_df, equity_df=eq_df)
            stress_table = pd.DataFrame({
                "Metric": ["CAGR", "Max Drawdown", "Sharpe Ratio", "Profit Factor", "N Trades"],
                "Pre (Full)": [
                    f"{stress['pre']['cagr']:.2%}" if not np.isnan(stress['pre']['cagr']) else "N/A",
                    f"{stress['pre']['max_dd']:.2%}",
                    f"{stress['pre']['sharpe']:.2f}",
                    f"{stress['pre']['profit_factor']:.2f}",
                    f"{stress['pre']['n_trades']:,}",
                ],
                "Post (Top 5% Removed)": [
                    f"{stress['post']['cagr']:.2%}" if not np.isnan(stress['post']['cagr']) else "N/A",
                    f"{stress['post']['max_dd']:.2%}" if not np.isnan(stress['post']['max_dd']) else "N/A",
                    f"{stress['post']['sharpe']:.2f}",
                    f"{stress['post']['profit_factor']:.2f}",
                    f"{stress['post']['n_trades']:,}",
                ],
            })
            st.dataframe(stress_table, hide_index=True)
            st.caption(f"Removed {stress['removed']} trades above threshold")

            profit_col = "Profit" if "Profit" in stats_df.columns else "PctGain"
            st.plotly_chart(chart_box_plot(stats_df[profit_col], profit_col),
                           use_container_width=True)

            # Temporal analysis
            st.subheader("4. Temporal Bias & Regime Analysis")
            yr = annual_returns(stats_df, equity_df=eq_df)
            st.plotly_chart(chart_annual_returns(yr), use_container_width=True)

            conc = profit_concentration(yr)
            if conc["best_year"]:
                st.metric("Best Year Profit Contribution",
                          f"{conc['contribution_pct']:.1%}",
                          delta=f"Year {conc['best_year']}")

            regime = regime_performance(stats_df, equity_df=eq_df)
            if not regime.empty and regime["n_trades"].sum() > 0:
                st.caption("Regime Performance")
                regime_fmt = regime.copy()
                regime_fmt["net_return"] = regime_fmt["net_return"].apply(
                    lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A")
                regime_fmt["win_rate"] = regime_fmt["win_rate"].apply(
                    lambda x: f"{x:.1%}" if not pd.isna(x) else "N/A")
                regime_fmt["profit_factor"] = regime_fmt["profit_factor"].apply(
                    lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
                st.dataframe(regime_fmt, hide_index=True)
            else:
                st.info("No trades found in predefined crisis periods.")

            # Equity dip analysis
            st.subheader("5. Equity Dip & Recovery Analysis")
            dip = equity_dip_analysis(stats_df, equity_df=eq_df)
            dip_table = pd.DataFrame({
                "Metric": ["N Trades", "Win Rate", "Avg Profit", "Profit Factor", "Avg MAE"],
                "In Drawdown": [
                    f"{dip['in_drawdown']['n']:,}",
                    f"{dip['in_drawdown']['win_rate']:.1%}" if not np.isnan(dip['in_drawdown']['win_rate']) else "N/A",
                    f"{dip['in_drawdown']['avg_profit']:.4%}" if not np.isnan(dip['in_drawdown']['avg_profit']) else "N/A",
                    f"{dip['in_drawdown']['profit_factor']:.2f}" if not np.isnan(dip['in_drawdown']['profit_factor']) else "N/A",
                    f"{dip['in_drawdown']['avg_mae']:.4%}" if not np.isnan(dip['in_drawdown']['avg_mae']) else "N/A",
                ],
                "At Equity Peak": [
                    f"{dip['at_peak']['n']:,}",
                    f"{dip['at_peak']['win_rate']:.1%}" if not np.isnan(dip['at_peak']['win_rate']) else "N/A",
                    f"{dip['at_peak']['avg_profit']:.4%}" if not np.isnan(dip['at_peak']['avg_profit']) else "N/A",
                    f"{dip['at_peak']['profit_factor']:.2f}" if not np.isnan(dip['at_peak']['profit_factor']) else "N/A",
                    f"{dip['at_peak']['avg_mae']:.4%}" if not np.isnan(dip['at_peak']['avg_mae']) else "N/A",
                ],
            })
            st.dataframe(dip_table, hide_index=True)

            # MAE significance callout
            mae_dd = dip['in_drawdown']['avg_mae']
            mae_pk = dip['at_peak']['avg_mae']
            if not np.isnan(mae_dd) and not np.isnan(mae_pk) and mae_dd != 0 and mae_pk != 0:
                mae_diff = mae_dd - mae_pk  # both are negative, more negative = worse
                if mae_diff < -0.005:
                    st.warning(f"⚠ MAE is significantly worse during drawdowns "
                               f"({mae_dd:.2%} vs {mae_pk:.2%}). "
                               f"Trades taken in drawdowns experience deeper adverse moves.")
                elif abs(mae_diff) <= 0.005:
                    st.info(f"MAE is similar in both regimes ({mae_dd:.2%} vs {mae_pk:.2%}). "
                            f"Risk behavior appears consistent regardless of equity state.")

            st.plotly_chart(
                chart_equity_drawdown_shading(
                    dip["cum_equity"], dip["peak_equity"],
                    dip["in_dd_mask"], dip["chart_dates"],
                ),
                use_container_width=True,
            )

            # Significance decay
            st.subheader("6. Statistical Significance Decay")
            sig = significance_decay(returns)
            sig_table = pd.DataFrame({
                "Metric": ["T-Statistic", "P-Value", "Significant (p<0.05)"],
                "Full Sample": [
                    f"{sig['pre_t']:.4f}",
                    f"{sig['pre_p']:.6f}",
                    "✓" if sig['significant_pre'] else "✗",
                ],
                "Top 5% Removed": [
                    f"{sig['post_t']:.4f}",
                    f"{sig['post_p']:.6f}",
                    "✓" if sig['significant_post'] else "✗",
                ],
            })
            st.dataframe(sig_table, hide_index=True)
            if sig['significant_pre'] and not sig['significant_post']:
                st.warning("⚠ Strategy loses statistical significance when top 5% outliers are removed.")
            elif sig['significant_pre'] and sig['significant_post']:
                st.success("Strategy edge remains significant even after removing top outliers.")

            # Final verdict
            st.subheader("7. Fragility Verdict")
            verdict, detail = fragility_verdict(stress, sig, dip, conc)
            if verdict == "FRAGILE":
                st.error(f"🔴 **{verdict}** — {detail}")
            else:
                st.success(f"🟢 **{verdict}** — {detail}")

with tab_data:
    st.header("Data Sheet — Audit & Verification")
    ds_eq, ds_tr = st.tabs(["Equity Calculations", "Trades Calculations"])

    # ── Equity data sheet ─────────────────────────────────────────────────────
    with ds_eq:
        if eq_df is None:
            st.warning("No equity data loaded.")
        else:
            src = eq_view if recalc else eq_df

            # Column grouping for readability
            _eq_groups = {
                "Core": ["Date", "equity", "highest_high", "drawdown"],
                "Drawdown Bands": ["dd_avg_whole", "dd_avg_252", "dd_upper_pct", "dd_lower_pct"],
                "Rolling CAGR": ["cagr_1y", "cagr_3y", "cagr_5y", "cagr_7y", "cagr_10y"],
                "Moving Averages": ["ma_25", "ma_50", "ma_100", "ma_200", "ma_avg"],
                "Bollinger Bands": [
                    "bb_std", "bb_upper", "bb_lower",
                    "bb_std_log", "bb_upper_log", "bb_lower_log",
                    "days_above_bb", "days_below_bb",
                ],
                "Volatility": ["log_return", "vol_25", "vol_50", "vol_100", "vol_200", "vol_median"],
                "Best Fit Channel": [
                    "best_fit", "best_fit_upper", "best_fit_lower",
                    "above_fit", "below_fit",
                    "above_ceiling", "below_floor",
                    "pct_above_ceiling", "pct_below_floor",
                ],
            }

            # Let user pick which groups to display
            all_groups = list(_eq_groups.keys())
            selected_groups = st.multiselect(
                "Column Groups", all_groups, default=all_groups, key="eq_groups",
            )
            show_cols = []
            for g in selected_groups:
                show_cols.extend(c for c in _eq_groups[g] if c in src.columns)
            # Remove duplicates while preserving order
            show_cols = list(dict.fromkeys(show_cols))

            st.caption(f"{len(src):,} rows  ·  {len(show_cols)} columns")

            # Format percentages and large numbers
            _pct = {"drawdown", "dd_avg_whole", "dd_avg_252", "dd_upper_pct", "dd_lower_pct",
                    "cagr_1y", "cagr_3y", "cagr_5y", "cagr_7y", "cagr_10y",
                    "log_return", "pct_above_ceiling", "pct_below_floor"}
            _int = {"days_above_bb", "days_below_bb", "above_fit", "below_fit",
                    "above_ceiling", "below_floor"}
            fmt = {c: "{:.4%}" if c in _pct else "{:.0f}" if c in _int else "{:,.2f}"
                   for c in show_cols if c != "Date"}

            st.dataframe(
                src[show_cols].style.format(fmt, na_rep=""),
                height=600,
                use_container_width=True,
            )

            # CSV download
            csv_bytes = src[show_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Equity Data as CSV",
                csv_bytes,
                file_name="equity_data_sheet.csv",
                mime="text/csv",
            )

    # ── Trades data sheet ─────────────────────────────────────────────────────
    with ds_tr:
        if tr_df is None:
            st.warning("No trades data loaded.")
        else:
            # Split columns into original CSV columns vs computed columns
            _computed_cols = {
                "trade_num", "pct_gain", "bars", "win",
                "win_pct_25", "win_pct_50", "win_pct_100", "win_pct_200",
                "win_pct_avg", "win_pct_std", "win_pct_upper", "win_pct_lower",
                "gain_avg_25", "gain_avg_50", "gain_avg_100", "gain_avg_200",
                "gain_avg", "gain_std", "gain_upper", "gain_lower",
            }
            original_cols = [c for c in tr_df.columns if c not in _computed_cols]

            _tr_groups = {
                "All Columns": original_cols,
                "Rolling Win %": [
                    "win_pct_25", "win_pct_50", "win_pct_100", "win_pct_200",
                    "win_pct_avg", "win_pct_std", "win_pct_upper", "win_pct_lower",
                ],
                "Rolling Avg Gain": [
                    "gain_avg_25", "gain_avg_50", "gain_avg_100", "gain_avg_200",
                    "gain_avg", "gain_std", "gain_upper", "gain_lower",
                ],
            }

            all_tr_groups = list(_tr_groups.keys())
            selected_tr = st.multiselect(
                "Column Groups", all_tr_groups, default=all_tr_groups, key="tr_groups",
            )
            tr_show = []
            for g in selected_tr:
                tr_show.extend(c for c in _tr_groups[g] if c in tr_df.columns)
            tr_show = list(dict.fromkeys(tr_show))

            st.caption(f"{len(tr_df):,} trades  ·  {len(tr_show)} columns")

            tr_pct = {"pct_gain", "PctGain", "PctMFE", "PctMAE", "Fraction",
                      "CommsInPct", "CommsOutPct", "SlipInPct", "SlipOutPct",
                      "VolumePct_In", "VolumePct_Out",
                      "win_pct_25", "win_pct_50", "win_pct_100", "win_pct_200",
                      "win_pct_avg", "win_pct_std", "win_pct_upper", "win_pct_lower",
                      "gain_avg_25", "gain_avg_50", "gain_avg_100", "gain_avg_200",
                      "gain_avg", "gain_std", "gain_upper", "gain_lower"}
            tr_int = {"trade_num", "Trade", "bars", "Bars", "win", "QtyIn", "QtyOut"}
            _date_cols = {"DateIn", "DateOut"}
            _str_cols = {"Strategy", "Symbol", "Side", "Reason", "TimeIn", "TimeOut",
                         "SavedCalcIn", "SavedCalcOut"}
            _skip_fmt = _date_cols | _str_cols
            tr_fmt = {c: "{:.4%}" if c in tr_pct else "{:.0f}" if c in tr_int else "{:,.2f}"
                      for c in tr_show if c not in _skip_fmt}

            st.dataframe(
                tr_df[tr_show].style.format(tr_fmt, na_rep=""),
                height=600,
                use_container_width=True,
            )

            csv_bytes = tr_df[tr_show].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Trades Data as CSV",
                csv_bytes,
                file_name="trades_data_sheet.csv",
                mime="text/csv",
            )

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
