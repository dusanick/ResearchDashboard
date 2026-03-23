"""Unit tests for reports/export_static.py"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from reports.export_static import (
    _fig_to_base64,
    _img_tag,
    _metric_card,
    _html_table,
    generate_static_report,
)
from calculations.equity_calcs import compute_all as eq_compute_all, compute_dd_distribution
from calculations.trades_calcs import compute_all as tr_compute_all


# ── Fixtures ──────────────────────────────────────────────────────────────────

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
    dist, cur_dd, cur_dd_pct = compute_dd_distribution(equity_df)
    return dict(
        eq_view=equity_df, eq_full=equity_df, tr_view=trades_df,
        live_date=LIVE_DATE, log_y=True,
        dd_dist=dist, cur_dd=cur_dd, cur_dd_pct=cur_dd_pct,
        pct_above_ceiling=equity_df["pct_above_ceiling"].iloc[-1],
        pct_below_floor=equity_df["pct_below_floor"].iloc[-1],
        x_col="trade_num",
    )


# ── _fig_to_base64 ───────────────────────────────────────────────────────────

class TestFigToBase64:
    def test_returns_string(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        result = _fig_to_base64(fig)
        assert isinstance(result, str)

    def test_is_valid_base64(self):
        import base64
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        result = _fig_to_base64(fig)
        decoded = base64.b64decode(result)
        # PNG magic bytes
        assert decoded[:4] == b"\x89PNG"

    def test_closes_figure(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        before = len(plt.get_fignums())
        _fig_to_base64(fig)
        after = len(plt.get_fignums())
        assert after < before


# ── _img_tag ──────────────────────────────────────────────────────────────────

class TestImgTag:
    def test_returns_img_element(self):
        result = _img_tag("AAAA")
        assert result.startswith("<img")
        assert "data:image/png;base64,AAAA" in result


# ── _metric_card ──────────────────────────────────────────────────────────────

class TestMetricCard:
    def test_contains_label_and_value(self):
        result = _metric_card("Test", "42%")
        assert "Test" in result
        assert "42%" in result

    def test_has_metric_class(self):
        result = _metric_card("L", "V")
        assert 'class="metric"' in result


# ── _html_table ───────────────────────────────────────────────────────────────

class TestHtmlTable:
    def test_contains_headers_and_data(self):
        df = pd.DataFrame({"Col1": [10], "Col2": [20]})
        result = _html_table(df)
        assert "<th>Col1</th>" in result
        assert "<td>10</td>" in result

    def test_applies_formats(self):
        df = pd.DataFrame({"pct": [0.123]})
        result = _html_table(df, col_formats={"pct": "{:.1%}"})
        assert "12.3%" in result


# ── generate_static_report ────────────────────────────────────────────────────

class TestGenerateStaticReport:
    def test_returns_string(self, full_report_args):
        html = generate_static_report(**full_report_args)
        assert isinstance(html, str)

    def test_is_valid_html(self, full_report_args):
        html = generate_static_report(**full_report_args)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_title(self, full_report_args):
        html = generate_static_report(**full_report_args)
        assert "Trading System Performance Report" in html

    def test_contains_static_label(self, full_report_args):
        html = generate_static_report(**full_report_args)
        assert "Static Report" in html

    def test_contains_date_period(self, full_report_args):
        html = generate_static_report(**full_report_args)
        assert "2020-01-01" in html

    def test_contains_png_images(self, full_report_args):
        html = generate_static_report(**full_report_args)
        assert "data:image/png;base64," in html

    def test_has_nine_chart_blocks(self, full_report_args):
        html = generate_static_report(**full_report_args)
        assert html.count('class="chart-block"') == 9

    def test_contains_dd_metrics(self, full_report_args):
        html = generate_static_report(**full_report_args)
        assert "Current Drawdown" in html
        assert "Current DD Percentile" in html

    def test_contains_ceiling_floor_metrics(self, full_report_args):
        html = generate_static_report(**full_report_args)
        assert "% Time Above Ceiling" in html
        assert "% Time Below Floor" in html

    def test_contains_trades_section(self, full_report_args):
        html = generate_static_report(**full_report_args)
        assert "Trades Dashboard" in html

    def test_no_trades_when_none(self, full_report_args):
        full_report_args["tr_view"] = None
        html = generate_static_report(**full_report_args)
        assert "Trades Dashboard" not in html
        assert html.count('class="chart-block"') == 7

    def test_no_plotlyjs(self, full_report_args):
        """Static report must NOT contain plotly.js (that's the whole point)."""
        html = generate_static_report(**full_report_args)
        assert "Plotly.newPlot" not in html
        assert "plotly.min.js" not in html.lower()

    def test_smaller_than_interactive(self, full_report_args):
        """Static report should be significantly smaller than 5 MB."""
        html = generate_static_report(**full_report_args)
        size_mb = len(html.encode("utf-8")) / (1024 * 1024)
        assert size_mb < 5

    def test_css_embedded(self, full_report_args):
        html = generate_static_report(**full_report_args)
        assert "<style>" in html
        assert ".metric" in html
