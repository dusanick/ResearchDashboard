"""Unit tests for calculations/equity_calcs.py"""
import io
import numpy as np
import pandas as pd
import pytest

from calculations.equity_calcs import (
    EQUITY_REQUIRED_COLS,
    load_equity_curve,
    compute_drawdown,
    compute_dd_percentile_bands,
    compute_dd_distribution,
    compute_rolling_cagr,
    compute_moving_averages,
    compute_bollinger_bands,
    compute_volatility,
    compute_best_fit,
    compute_all,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_equity_df():
    """Small equity DataFrame for fast deterministic tests."""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    equity = 100_000 * np.cumprod(1 + np.random.default_rng(42).normal(0.0003, 0.01, len(dates)))
    return pd.DataFrame({"Date": dates, "equity": equity})


@pytest.fixture
def monotonic_equity_df():
    """Monotonically increasing equity — zero drawdown."""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    equity = np.linspace(100_000, 200_000, len(dates))
    return pd.DataFrame({"Date": dates, "equity": equity})


@pytest.fixture
def csv_good(tmp_path):
    """Valid equity CSV file."""
    p = tmp_path / "eq.csv"
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=10, freq="B").strftime("%Y-%m-%d"),
        "System Equity Curve": np.linspace(100, 200, 10),
    })
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def csv_missing_col(tmp_path):
    """CSV missing the equity column."""
    p = tmp_path / "bad.csv"
    pd.DataFrame({"Date": ["2020-01-01"], "Wrong": [100]}).to_csv(p, index=False)
    return str(p)


@pytest.fixture
def csv_extra_cols(tmp_path):
    """CSV with extra columns that should be ignored."""
    p = tmp_path / "extra.csv"
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=5, freq="B").strftime("%Y-%m-%d"),
        "System Equity Curve": [100, 110, 105, 115, 120],
        "ExtraCol": [1, 2, 3, 4, 5],
        "AnotherCol": ["a", "b", "c", "d", "e"],
    })
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def csv_mixed_dates(tmp_path):
    """CSV with mixed ISO and Excel serial number dates."""
    p = tmp_path / "mixed.csv"
    rows = [
        {"Date": "2020-01-01", "System Equity Curve": 100},
        {"Date": "2020-01-02", "System Equity Curve": 105},
        {"Date": "43833", "System Equity Curve": 110},  # 2020-01-03 as serial
    ]
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


# ── load_equity_curve ─────────────────────────────────────────────────────────

class TestLoadEquityCurve:
    def test_loads_valid_csv(self, csv_good):
        df = load_equity_curve(csv_good)
        assert "Date" in df.columns
        assert "equity" in df.columns
        assert len(df) == 10

    def test_rejects_missing_columns(self, csv_missing_col):
        with pytest.raises(ValueError, match="Missing required equity columns"):
            load_equity_curve(csv_missing_col)

    def test_keeps_only_required_columns(self, csv_extra_cols):
        df = load_equity_curve(csv_extra_cols)
        assert "ExtraCol" not in df.columns
        assert "AnotherCol" not in df.columns
        assert list(df.columns) == ["Date", "equity"]

    def test_handles_mixed_date_formats(self, csv_mixed_dates):
        df = load_equity_curve(csv_mixed_dates)
        assert len(df) == 3
        assert df["Date"].isna().sum() == 0

    def test_sorts_by_date(self, csv_good):
        df = load_equity_curve(csv_good)
        assert df["Date"].is_monotonic_increasing

    def test_column_order_independent(self, tmp_path):
        """Columns in any order should work."""
        p = tmp_path / "reversed.csv"
        pd.DataFrame({
            "System Equity Curve": [100, 200],
            "Date": ["2020-01-01", "2020-01-02"],
        }).to_csv(p, index=False)
        df = load_equity_curve(str(p))
        assert len(df) == 2


# ── compute_drawdown ──────────────────────────────────────────────────────────

class TestComputeDrawdown:
    def test_output_columns(self, simple_equity_df):
        df = compute_drawdown(simple_equity_df)
        for col in ["highest_high", "drawdown", "dd_avg_whole", "dd_avg_252"]:
            assert col in df.columns

    def test_drawdown_is_non_positive(self, simple_equity_df):
        df = compute_drawdown(simple_equity_df)
        assert (df["drawdown"] <= 0).all()

    def test_zero_drawdown_on_monotonic(self, monotonic_equity_df):
        df = compute_drawdown(monotonic_equity_df)
        assert (df["drawdown"] == 0).all()

    def test_highest_high_is_cummax(self, simple_equity_df):
        df = compute_drawdown(simple_equity_df)
        np.testing.assert_array_equal(df["highest_high"], df["equity"].cummax())

    def test_dd_avg_252_needs_252_rows(self, simple_equity_df):
        df = compute_drawdown(simple_equity_df)
        assert df["dd_avg_252"].iloc[:251].isna().all()
        assert df["dd_avg_252"].iloc[251:].notna().all()


# ── compute_dd_percentile_bands ───────────────────────────────────────────────

class TestComputeDDPercentileBands:
    def test_output_columns(self, simple_equity_df):
        df = compute_drawdown(simple_equity_df)
        df = compute_dd_percentile_bands(df, 30)
        assert "dd_upper_pct" in df.columns
        assert "dd_lower_pct" in df.columns

    def test_upper_less_negative_than_lower(self, simple_equity_df):
        df = compute_drawdown(simple_equity_df)
        df = compute_dd_percentile_bands(df, 30)
        valid = df.dropna(subset=["dd_upper_pct", "dd_lower_pct"])
        # DD values are negative; upper pct (30th) is more negative than lower (70th)
        assert (valid["dd_upper_pct"] <= valid["dd_lower_pct"]).all()


# ── compute_dd_distribution ───────────────────────────────────────────────────

class TestComputeDDDistribution:
    def test_returns_tuple_of_three(self, simple_equity_df):
        df = compute_drawdown(simple_equity_df)
        result = compute_dd_distribution(df)
        assert len(result) == 3
        dist, cur_dd, cur_pct = result
        assert isinstance(dist, pd.DataFrame)
        assert isinstance(cur_dd, float)
        assert isinstance(cur_pct, float)

    def test_distribution_table_shape(self, simple_equity_df):
        df = compute_drawdown(simple_equity_df)
        dist, _, _ = compute_dd_distribution(df)
        assert len(dist) == 18  # 10% to 95% in steps of 5%
        assert "percentile" in dist.columns
        assert "drawdown" in dist.columns

    def test_current_dd_is_last_value(self, simple_equity_df):
        df = compute_drawdown(simple_equity_df)
        _, cur_dd, _ = compute_dd_distribution(df)
        assert cur_dd == df["drawdown"].iloc[-1]

    def test_percentile_is_from_table(self, simple_equity_df):
        df = compute_drawdown(simple_equity_df)
        dist, cur_dd, cur_pct = compute_dd_distribution(df)
        # cur_pct should be one of the table percentiles (or 0.0)
        valid_pcts = set(dist["percentile"].tolist()) | {0.0}
        assert cur_pct in valid_pcts


# ── compute_rolling_cagr ─────────────────────────────────────────────────────

class TestComputeRollingCagr:
    def test_output_columns(self, simple_equity_df):
        df = compute_rolling_cagr(simple_equity_df)
        for n in [1, 3, 5, 7, 10]:
            assert f"cagr_{n}y" in df.columns

    def test_nan_before_lookback(self, simple_equity_df):
        df = compute_rolling_cagr(simple_equity_df)
        # 300 rows, 1yr = 252, so first 251 should be NaN for 1Y
        assert df["cagr_1y"].iloc[:251].isna().all()

    def test_positive_cagr_for_growth(self):
        """Steady growth should give positive CAGR."""
        dates = pd.date_range("2018-01-01", periods=260, freq="B")
        equity = 100_000 * (1.0004 ** np.arange(len(dates)))
        df = pd.DataFrame({"Date": dates, "equity": equity})
        df = compute_rolling_cagr(df)
        valid = df["cagr_1y"].dropna()
        assert (valid > 0).all()


# ── compute_moving_averages ───────────────────────────────────────────────────

class TestComputeMovingAverages:
    def test_output_columns(self, simple_equity_df):
        df = compute_moving_averages(simple_equity_df)
        for w in [25, 50, 100, 200]:
            assert f"ma_{w}" in df.columns
        assert "ma_avg" in df.columns

    def test_ma_avg_starts_from_row_25(self, simple_equity_df):
        df = compute_moving_averages(simple_equity_df)
        # ma_avg uses skipna=True so it starts when ma_25 starts (row 24)
        assert df["ma_avg"].iloc[:24].isna().all()
        assert df["ma_avg"].iloc[24:].notna().all()

    def test_ma_25_is_rolling_mean(self, simple_equity_df):
        df = compute_moving_averages(simple_equity_df)
        expected = simple_equity_df["equity"].rolling(25, min_periods=25).mean()
        pd.testing.assert_series_equal(df["ma_25"], expected, check_names=False)


# ── compute_bollinger_bands ───────────────────────────────────────────────────

class TestComputeBollingerBands:
    def test_output_columns(self, simple_equity_df):
        df = compute_moving_averages(simple_equity_df)
        df = compute_bollinger_bands(df, 2.0)
        for col in ["bb_std", "bb_upper", "bb_lower", "bb_upper_log", "bb_lower_log",
                     "days_above_bb", "days_below_bb"]:
            assert col in df.columns

    def test_upper_above_lower(self, simple_equity_df):
        df = compute_moving_averages(simple_equity_df)
        df = compute_bollinger_bands(df, 2.0)
        valid = df.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_log_bands_always_positive(self, simple_equity_df):
        df = compute_moving_averages(simple_equity_df)
        df = compute_bollinger_bands(df, 3.0)
        valid = df["bb_lower_log"].dropna()
        assert (valid > 0).all()

    def test_consecutive_days_reset(self):
        """Consecutive counter should reset to 0 when condition breaks."""
        dates = pd.date_range("2020-01-01", periods=250, freq="B")
        # Alternating above/below pattern
        equity = np.where(np.arange(250) % 4 < 2, 120_000, 80_000).astype(float)
        df = pd.DataFrame({"Date": dates, "equity": equity})
        df = compute_moving_averages(df)
        df = compute_bollinger_bands(df, 0.5)
        # There should be zeros in the consecutive columns (resets)
        assert (df["days_above_bb"] == 0).any()


# ── compute_volatility ────────────────────────────────────────────────────────

class TestComputeVolatility:
    def test_output_columns(self, simple_equity_df):
        df = compute_volatility(simple_equity_df)
        for col in ["log_return", "vol_25", "vol_50", "vol_100", "vol_200", "vol_median"]:
            assert col in df.columns

    def test_volatility_non_negative(self, simple_equity_df):
        df = compute_volatility(simple_equity_df)
        for w in [25, 50, 100, 200]:
            valid = df[f"vol_{w}"].dropna()
            assert (valid >= 0).all()

    def test_log_return_first_is_nan(self, simple_equity_df):
        df = compute_volatility(simple_equity_df)
        assert pd.isna(df["log_return"].iloc[0])


# ── compute_best_fit ──────────────────────────────────────────────────────────

class TestComputeBestFit:
    def test_output_columns(self, simple_equity_df):
        df = compute_best_fit(simple_equity_df, 33.0)
        for col in ["best_fit", "best_fit_upper", "best_fit_lower",
                     "above_fit", "below_fit",
                     "above_ceiling", "below_floor",
                     "pct_above_ceiling", "pct_below_floor"]:
            assert col in df.columns

    def test_upper_above_lower_channel(self, simple_equity_df):
        df = compute_best_fit(simple_equity_df, 33.0)
        valid = df.dropna(subset=["best_fit_upper", "best_fit_lower"])
        assert (valid["best_fit_upper"] >= valid["best_fit_lower"]).all()

    def test_best_fit_positive(self, simple_equity_df):
        df = compute_best_fit(simple_equity_df, 33.0)
        valid = df["best_fit"].dropna()
        assert (valid > 0).all()

    def test_pct_above_ceiling_bounded(self, simple_equity_df):
        df = compute_best_fit(simple_equity_df, 33.0)
        valid = df["pct_above_ceiling"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_short_data_returns_nan(self):
        df = pd.DataFrame({"Date": [pd.Timestamp("2020-01-01")], "equity": [100]})
        df = compute_best_fit(df, 33.0)
        assert df["best_fit"].isna().all()


# ── compute_all ───────────────────────────────────────────────────────────────

class TestComputeAll:
    def test_runs_without_error(self, simple_equity_df):
        df = compute_all(simple_equity_df)
        assert len(df) == 300

    def test_all_expected_columns_present(self, simple_equity_df):
        df = compute_all(simple_equity_df)
        expected = ["equity", "drawdown", "ma_avg", "bb_upper", "bb_lower",
                    "vol_25", "best_fit", "cagr_1y"]
        for col in expected:
            assert col in df.columns
