"""Unit tests for charts/equity_charts.py and charts/trades_charts.py

These tests verify that chart functions return valid Plotly Figures
with expected traces, layout, and don't crash on edge cases.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from charts.equity_charts import (
    chart_equity_curve, chart_drawdown, chart_rolling_cagr,
    chart_equity_bollinger, chart_bb_consecutive, chart_volatility, chart_best_fit,
)
from charts.trades_charts import chart_rolling_win_pct, chart_rolling_gain
from calculations.equity_calcs import compute_all as eq_compute_all
from calculations.trades_calcs import compute_all as tr_compute_all


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def equity_df():
    """Computed equity DataFrame with all columns."""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    equity = 100_000 * np.cumprod(1 + np.random.default_rng(42).normal(0.0003, 0.01, len(dates)))
    df = pd.DataFrame({"Date": dates, "equity": equity})
    return eq_compute_all(df)


@pytest.fixture
def trades_df():
    """Computed trades DataFrame with all columns."""
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


# ── Equity Charts ─────────────────────────────────────────────────────────────

class TestChartEquityCurve:
    def test_returns_figure(self, equity_df):
        fig = chart_equity_curve(equity_df, LIVE_DATE, log_y=False)
        assert isinstance(fig, go.Figure)

    def test_log_scale_layout(self, equity_df):
        fig = chart_equity_curve(equity_df, LIVE_DATE, log_y=True)
        assert fig.layout.yaxis.type == "log"

    def test_linear_scale_layout(self, equity_df):
        fig = chart_equity_curve(equity_df, LIVE_DATE, log_y=False)
        assert fig.layout.yaxis.type == "linear"

    def test_has_traces(self, equity_df):
        fig = chart_equity_curve(equity_df, LIVE_DATE, log_y=False)
        assert len(fig.data) >= 1

    def test_no_live_date(self, equity_df):
        fig = chart_equity_curve(equity_df, None, log_y=False)
        assert isinstance(fig, go.Figure)


class TestChartDrawdown:
    def test_returns_figure(self, equity_df):
        fig = chart_drawdown(equity_df, LIVE_DATE)
        assert isinstance(fig, go.Figure)

    def test_has_multiple_traces(self, equity_df):
        fig = chart_drawdown(equity_df, LIVE_DATE)
        assert len(fig.data) >= 3  # drawdown, avg, percentile bands


class TestChartRollingCagr:
    def test_returns_figure(self, equity_df):
        fig = chart_rolling_cagr(equity_df, LIVE_DATE)
        assert isinstance(fig, go.Figure)

    def test_has_cagr_traces(self, equity_df):
        fig = chart_rolling_cagr(equity_df, LIVE_DATE)
        assert len(fig.data) >= 5  # 1/3/5/7/10 yr


class TestChartEquityBollinger:
    def test_returns_figure(self, equity_df):
        fig = chart_equity_bollinger(equity_df, log_y=False, live_date=LIVE_DATE)
        assert isinstance(fig, go.Figure)

    def test_uses_log_bands_on_log_scale(self, equity_df):
        fig = chart_equity_bollinger(equity_df, log_y=True, live_date=LIVE_DATE)
        # Should have 4 traces: equity, ma_avg, upper, lower
        assert len(fig.data) == 4

    def test_has_four_traces(self, equity_df):
        fig = chart_equity_bollinger(equity_df, log_y=False, live_date=LIVE_DATE)
        assert len(fig.data) == 4


class TestChartBBConsecutive:
    def test_returns_figure(self, equity_df):
        fig = chart_bb_consecutive(equity_df, LIVE_DATE)
        assert isinstance(fig, go.Figure)

    def test_has_bar_traces(self, equity_df):
        fig = chart_bb_consecutive(equity_df, LIVE_DATE)
        assert any(isinstance(t, go.Bar) for t in fig.data)


class TestChartVolatility:
    def test_returns_figure(self, equity_df):
        fig = chart_volatility(equity_df, LIVE_DATE)
        assert isinstance(fig, go.Figure)

    def test_has_five_traces(self, equity_df):
        fig = chart_volatility(equity_df, LIVE_DATE)
        assert len(fig.data) == 5


class TestChartBestFit:
    def test_returns_figure(self, equity_df):
        fig = chart_best_fit(equity_df, log_y=True, live_date=LIVE_DATE)
        assert isinstance(fig, go.Figure)

    def test_has_four_traces(self, equity_df):
        fig = chart_best_fit(equity_df, log_y=False, live_date=LIVE_DATE)
        assert len(fig.data) == 4  # equity, best_fit, upper, lower


# ── Trades Charts ─────────────────────────────────────────────────────────────

class TestChartRollingWinPct:
    def test_returns_figure(self, trades_df):
        fig = chart_rolling_win_pct(trades_df, "trade_num")
        assert isinstance(fig, go.Figure)

    def test_has_traces(self, trades_df):
        fig = chart_rolling_win_pct(trades_df, "trade_num")
        assert len(fig.data) >= 4

    def test_date_x_axis(self, trades_df):
        fig = chart_rolling_win_pct(trades_df, "DateOut")
        assert isinstance(fig, go.Figure)


class TestChartRollingGain:
    def test_returns_figure(self, trades_df):
        fig = chart_rolling_gain(trades_df, "trade_num")
        assert isinstance(fig, go.Figure)

    def test_has_traces(self, trades_df):
        fig = chart_rolling_gain(trades_df, "trade_num")
        assert len(fig.data) >= 5


# ── Layout Tests ──────────────────────────────────────────────────────────────

class TestLayoutProperties:
    def test_top_margin_is_80(self, equity_df):
        fig = chart_equity_curve(equity_df, LIVE_DATE, log_y=False)
        assert fig.layout.margin.t == 80

    def test_legend_horizontal(self, equity_df):
        fig = chart_drawdown(equity_df)
        assert fig.layout.legend.orientation == "h"
