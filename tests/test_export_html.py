"""Unit tests for reports/export_html.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from reports.export_html import (
    _fig_to_html_div,
    _metric_card,
    _html_table,
    generate_report,
)
from calculations.equity_calcs import compute_all as eq_compute_all, compute_dd_distribution
from calculations.trades_calcs import compute_all as tr_compute_all
from charts.equity_charts import (
    chart_equity_curve, chart_drawdown, chart_rolling_cagr,
    chart_equity_bollinger, chart_bb_consecutive, chart_volatility, chart_best_fit,
)
from charts.trades_charts import chart_rolling_win_pct, chart_rolling_gain


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_fig():
    return go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="test"))


@pytest.fixture
def equity_df():
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    equity = 100_000 * np.cumprod(
        1 + np.random.default_rng(42).normal(0.0003, 0.01, len(dates))
    )
    return eq_compute_all(pd.DataFrame({"Date": dates, "equity": equity}))


@pytest.fixture
def trades_df():
    rng = np.random.default_rng(42)
    n = 250
    df = pd.DataFrame({
        "Trade": range(1, n + 1),
        "DateIn": pd.date_range("2020-01-01", periods=n, freq="B"),
        "DateOut": pd.date_range("2020-01-02", periods=n, freq="B"),
        "Bars": rng.integers(1, 20, n),
        "PctGain": rng.normal(0.002, 0.02, n),
    })
    return tr_compute_all(df)


LIVE_DATE = pd.Timestamp("2020-10-01")


@pytest.fixture
def full_report_args(equity_df, trades_df):
    """Build all arguments needed for generate_report."""
    dist, cur_dd, cur_dd_pct = compute_dd_distribution(equity_df)
    eq_figs = [
        chart_equity_curve(equity_df, LIVE_DATE, True),
        chart_drawdown(equity_df, LIVE_DATE),
        chart_rolling_cagr(equity_df, LIVE_DATE),
        chart_equity_bollinger(equity_df, True, LIVE_DATE),
        chart_bb_consecutive(equity_df, LIVE_DATE),
        chart_volatility(equity_df, LIVE_DATE),
        chart_best_fit(equity_df, True, LIVE_DATE),
    ]
    tr_figs = [
        chart_rolling_win_pct(trades_df, "trade_num"),
        chart_rolling_gain(trades_df, "trade_num"),
    ]
    return dict(
        eq_view=equity_df, eq_full=equity_df, tr_view=trades_df,
        live_date=LIVE_DATE, log_y=True,
        dd_dist=dist, cur_dd=cur_dd, cur_dd_pct=cur_dd_pct,
        pct_above_ceiling=equity_df["pct_above_ceiling"].iloc[-1],
        pct_below_floor=equity_df["pct_below_floor"].iloc[-1],
        x_col="trade_num", equity_charts=eq_figs, trades_charts=tr_figs,
    )


# ── _fig_to_html_div ─────────────────────────────────────────────────────────

class TestFigToHtmlDiv:
    def test_returns_string(self, simple_fig):
        html = _fig_to_html_div(simple_fig)
        assert isinstance(html, str)

    def test_contains_plotly_div(self, simple_fig):
        html = _fig_to_html_div(simple_fig)
        assert "<div" in html

    def test_contains_newplot(self, simple_fig):
        html = _fig_to_html_div(simple_fig)
        assert "Plotly.newPlot" in html or "Plotly.react" in html

    def test_no_plotlyjs_by_default(self, simple_fig):
        html = _fig_to_html_div(simple_fig, include_js=False)
        # Should not contain the full plotly.js library
        assert "plotly.min.js" not in html.lower()

    def test_includes_plotlyjs_when_requested(self, simple_fig):
        html = _fig_to_html_div(simple_fig, include_js=True)
        # Should contain the plotly.js library (large chunk of JS)
        assert len(html) > 100_000  # plotly.js is several MB


# ── _metric_card ──────────────────────────────────────────────────────────────

class TestMetricCard:
    def test_returns_html_string(self):
        result = _metric_card("Test Label", "42%")
        assert isinstance(result, str)

    def test_contains_label(self):
        result = _metric_card("My Label", "99")
        assert "My Label" in result

    def test_contains_value(self):
        result = _metric_card("Label", "12.5%")
        assert "12.5%" in result

    def test_has_metric_class(self):
        result = _metric_card("L", "V")
        assert 'class="metric"' in result


# ── _html_table ───────────────────────────────────────────────────────────────

class TestHtmlTable:
    def test_returns_html_string(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _html_table(df)
        assert isinstance(result, str)

    def test_contains_headers(self):
        df = pd.DataFrame({"Col1": [10], "Col2": [20]})
        result = _html_table(df)
        assert "<th>Col1</th>" in result
        assert "<th>Col2</th>" in result

    def test_contains_data_rows(self):
        df = pd.DataFrame({"x": [100, 200]})
        result = _html_table(df)
        assert "<td>100</td>" in result
        assert "<td>200</td>" in result

    def test_applies_column_formats(self):
        df = pd.DataFrame({"pct": [0.1234], "val": [5]})
        result = _html_table(df, col_formats={"pct": "{:.1%}"})
        assert "12.3%" in result

    def test_has_table_class(self):
        df = pd.DataFrame({"a": [1]})
        result = _html_table(df)
        assert 'class="data-table"' in result


# ── generate_report ───────────────────────────────────────────────────────────

class TestGenerateReport:
    def test_returns_string(self, full_report_args):
        html = generate_report(**full_report_args)
        assert isinstance(html, str)

    def test_is_valid_html(self, full_report_args):
        html = generate_report(**full_report_args)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_title(self, full_report_args):
        html = generate_report(**full_report_args)
        assert "Trading System Performance Report" in html

    def test_contains_date_period(self, full_report_args):
        html = generate_report(**full_report_args)
        assert "2020-01-01" in html  # start date
        assert "Period:" in html

    def test_contains_plotlyjs(self, full_report_args):
        html = generate_report(**full_report_args)
        # plotly.js must be embedded inline for self-contained report
        assert "Plotly.newPlot" in html or "Plotly.react" in html

    def test_contains_all_equity_chart_sections(self, full_report_args):
        html = generate_report(**full_report_args)
        for name in [
            "System Equity Curve", "System Drawdown", "Rolling CAGR",
            "Std Dev Bands", "Rolling Volatility", "Best Fit Floor",
        ]:
            assert name in html

    def test_contains_trades_section(self, full_report_args):
        html = generate_report(**full_report_args)
        assert "Trades Dashboard" in html
        assert "Rolling Win %" in html
        assert "Rolling Avg Gain %" in html

    def test_contains_dd_metrics(self, full_report_args):
        html = generate_report(**full_report_args)
        assert "Current Drawdown" in html
        assert "Current DD Percentile" in html

    def test_contains_ceiling_floor_metrics(self, full_report_args):
        html = generate_report(**full_report_args)
        assert "% Time Above Ceiling" in html
        assert "% Time Below Floor" in html

    def test_contains_dd_distribution_table(self, full_report_args):
        html = generate_report(**full_report_args)
        assert "data-table" in html

    def test_has_nine_chart_blocks(self, full_report_args):
        html = generate_report(**full_report_args)
        assert html.count('class="chart-block"') == 9

    def test_css_embedded(self, full_report_args):
        html = generate_report(**full_report_args)
        assert "<style>" in html
        assert ".metric" in html

    def test_no_trades_section_when_empty(self, full_report_args):
        full_report_args["trades_charts"] = []
        html = generate_report(**full_report_args)
        assert "Trades Dashboard" not in html
        assert html.count('class="chart-block"') == 7

    def test_report_is_self_contained(self, full_report_args):
        """Report should be large (>1MB) because plotly.js is bundled inline."""
        html = generate_report(**full_report_args)
        assert len(html) > 1_000_000
