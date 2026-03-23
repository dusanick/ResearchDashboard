"""Unit tests for calculations/trades_calcs.py"""
import numpy as np
import pandas as pd
import pytest

from calculations.trades_calcs import (
    TRADES_REQUIRED_COLS,
    load_trades,
    compute_base_columns,
    compute_rolling_win_pct,
    compute_rolling_gain,
    compute_all,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def csv_good(tmp_path):
    """Valid trades CSV with required columns."""
    p = tmp_path / "trades.csv"
    rng = np.random.default_rng(42)
    n = 250
    df = pd.DataFrame({
        "Trade": range(1, n + 1),
        "DateIn": pd.date_range("2020-01-01", periods=n, freq="B").strftime("%Y-%m-%d"),
        "DateOut": pd.date_range("2020-01-02", periods=n, freq="B").strftime("%Y-%m-%d"),
        "Bars": rng.integers(1, 20, n),
        "PctGain": rng.normal(0.002, 0.02, n),
    })
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def csv_missing_col(tmp_path):
    """CSV missing PctGain."""
    p = tmp_path / "bad.csv"
    pd.DataFrame({
        "Trade": [1], "DateIn": ["2020-01-01"], "DateOut": ["2020-01-02"], "Bars": [5],
    }).to_csv(p, index=False)
    return str(p)


@pytest.fixture
def csv_extra_cols(tmp_path):
    """CSV with extra columns."""
    p = tmp_path / "extra.csv"
    pd.DataFrame({
        "Trade": [1, 2], "DateIn": ["2020-01-01", "2020-01-02"],
        "DateOut": ["2020-01-02", "2020-01-03"], "Bars": [5, 3],
        "PctGain": [0.01, -0.005], "Profit": [100, -50], "Symbol": ["AAPL", "MSFT"],
    }).to_csv(p, index=False)
    return str(p)


@pytest.fixture
def trades_df():
    """In-memory trades DataFrame for calculation tests."""
    rng = np.random.default_rng(42)
    n = 250
    return pd.DataFrame({
        "Trade": range(1, n + 1),
        "DateIn": pd.date_range("2020-01-01", periods=n, freq="B"),
        "DateOut": pd.date_range("2020-01-02", periods=n, freq="B"),
        "Bars": rng.integers(1, 20, n),
        "PctGain": rng.normal(0.002, 0.02, n),
    })


# ── load_trades ───────────────────────────────────────────────────────────────

class TestLoadTrades:
    def test_loads_valid_csv(self, csv_good):
        df = load_trades(csv_good)
        assert len(df) == 250
        for col in TRADES_REQUIRED_COLS:
            assert col in df.columns

    def test_rejects_missing_columns(self, csv_missing_col):
        with pytest.raises(ValueError, match="Missing required trades columns"):
            load_trades(csv_missing_col)

    def test_keeps_only_required_columns(self, csv_extra_cols):
        df = load_trades(csv_extra_cols)
        assert "Profit" not in df.columns
        assert "Symbol" not in df.columns

    def test_dates_are_datetime(self, csv_good):
        df = load_trades(csv_good)
        assert pd.api.types.is_datetime64_any_dtype(df["DateIn"])
        assert pd.api.types.is_datetime64_any_dtype(df["DateOut"])

    def test_column_order_independent(self, tmp_path):
        """Columns in reversed order should work."""
        p = tmp_path / "rev.csv"
        pd.DataFrame({
            "PctGain": [0.01], "Bars": [5], "DateOut": ["2020-01-02"],
            "DateIn": ["2020-01-01"], "Trade": [1],
        }).to_csv(p, index=False)
        df = load_trades(str(p))
        assert len(df) == 1


# ── compute_base_columns ─────────────────────────────────────────────────────

class TestComputeBaseColumns:
    def test_output_columns(self, trades_df):
        df = compute_base_columns(trades_df)
        for col in ["trade_num", "pct_gain", "bars", "win"]:
            assert col in df.columns

    def test_win_flag_binary(self, trades_df):
        df = compute_base_columns(trades_df)
        assert set(df["win"].unique()).issubset({0, 1})

    def test_win_when_positive(self, trades_df):
        df = compute_base_columns(trades_df)
        positive = df[df["pct_gain"] > 0]
        assert (positive["win"] == 1).all()

    def test_loss_when_non_positive(self, trades_df):
        df = compute_base_columns(trades_df)
        non_positive = df[df["pct_gain"] <= 0]
        assert (non_positive["win"] == 0).all()

    def test_does_not_mutate_input(self, trades_df):
        original_cols = list(trades_df.columns)
        compute_base_columns(trades_df)
        assert list(trades_df.columns) == original_cols


# ── compute_rolling_win_pct ───────────────────────────────────────────────────

class TestComputeRollingWinPct:
    def test_output_columns(self, trades_df):
        df = compute_base_columns(trades_df)
        df = compute_rolling_win_pct(df, 4.0)
        for col in ["win_pct_25", "win_pct_50", "win_pct_100", "win_pct_200",
                     "win_pct_avg", "win_pct_std", "win_pct_upper", "win_pct_lower"]:
            assert col in df.columns

    def test_win_pct_bounded_0_1(self, trades_df):
        df = compute_base_columns(trades_df)
        df = compute_rolling_win_pct(df, 4.0)
        for w in [25, 50, 100, 200]:
            valid = df[f"win_pct_{w}"].dropna()
            assert (valid >= 0).all() and (valid <= 1).all()

    def test_nan_before_window(self, trades_df):
        df = compute_base_columns(trades_df)
        df = compute_rolling_win_pct(df, 4.0)
        assert df["win_pct_25"].iloc[:24].isna().all()

    def test_all_wins_gives_100pct(self):
        """All winning trades should give 100% win rate."""
        df = pd.DataFrame({
            "Trade": range(1, 31), "DateIn": pd.date_range("2020-01-01", periods=30, freq="B"),
            "DateOut": pd.date_range("2020-01-02", periods=30, freq="B"),
            "Bars": [5] * 30, "PctGain": [0.01] * 30,
        })
        df = compute_base_columns(df)
        df = compute_rolling_win_pct(df, 4.0)
        valid = df["win_pct_25"].dropna()
        assert (valid == 1.0).all()


# ── compute_rolling_gain ──────────────────────────────────────────────────────

class TestComputeRollingGain:
    def test_output_columns(self, trades_df):
        df = compute_base_columns(trades_df)
        df = compute_rolling_gain(df, 4.0)
        for col in ["gain_avg_25", "gain_avg_50", "gain_avg_100", "gain_avg_200",
                     "gain_avg", "gain_std", "gain_upper", "gain_lower"]:
            assert col in df.columns

    def test_upper_above_lower(self, trades_df):
        df = compute_base_columns(trades_df)
        df = compute_rolling_gain(df, 4.0)
        valid = df.dropna(subset=["gain_upper", "gain_lower"])
        assert (valid["gain_upper"] >= valid["gain_lower"]).all()

    def test_constant_gain(self):
        """Constant gain should give zero std and equal bands."""
        df = pd.DataFrame({
            "Trade": range(1, 230), "DateIn": pd.date_range("2020-01-01", periods=229, freq="B"),
            "DateOut": pd.date_range("2020-01-02", periods=229, freq="B"),
            "Bars": [5] * 229, "PctGain": [0.01] * 229,
        })
        df = compute_base_columns(df)
        df = compute_rolling_gain(df, 4.0)
        valid_std = df["gain_std"].dropna()
        np.testing.assert_allclose(valid_std.values, 0, atol=1e-12)


# ── compute_all ───────────────────────────────────────────────────────────────

class TestComputeAll:
    def test_runs_without_error(self, trades_df):
        df = compute_all(trades_df)
        assert len(df) == 250

    def test_all_expected_columns(self, trades_df):
        df = compute_all(trades_df)
        expected = ["trade_num", "pct_gain", "win", "win_pct_avg", "gain_avg"]
        for col in expected:
            assert col in df.columns
