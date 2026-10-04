import numpy as np
import pandas as pd
import pytest

from pinnse import Normalization, Denormalization


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": rng.normal(loc=5.0, scale=2.0, size=64),
            "b": rng.uniform(low=-3.0, high=3.0, size=64),
            "c": rng.exponential(scale=1.5, size=64),
        }
    )


def test_min_max_round_trip(sample_df):
    norm, metrics = Normalization.min_max(sample_df)
    back = Denormalization.min_max(norm, metrics)
    np.testing.assert_allclose(back.to_numpy(), sample_df.to_numpy(), rtol=1e-6)


def test_max_abs_round_trip(sample_df):
    norm, metrics = Normalization.max_abs(sample_df)
    back = Denormalization.max_abs(norm, metrics)
    np.testing.assert_allclose(back.to_numpy(), sample_df.to_numpy(), rtol=1e-6)


def test_mean_norm_round_trip(sample_df):
    norm, metrics = Normalization.mean_norm(sample_df)
    back = Denormalization.mean_abs(norm, metrics)
    np.testing.assert_allclose(back.to_numpy(), sample_df.to_numpy(), rtol=1e-6)


def test_z_score_round_trip(sample_df):
    """z_score forward must be (x - mean)/std so that Denormalization.z_norm inverts it."""
    norm, metrics = Normalization.z_score(sample_df)
    back = Denormalization.z_norm(norm, metrics)
    np.testing.assert_allclose(back.to_numpy(), sample_df.to_numpy(), rtol=1e-6)


def test_z_score_produces_unit_variance(sample_df):
    """Sanity check: standardized data should have ~zero mean and ~unit std per column."""
    norm, _ = Normalization.z_score(sample_df)
    for col in sample_df.columns:
        assert abs(norm[col].mean()) < 1e-6
        assert norm[col].std() == pytest.approx(1.0, rel=1e-6)


def test_min_max_subset(sample_df):
    norm, metrics = Normalization.min_max(sample_df, normalize_cols=["a"])
    assert "a" in metrics
    assert "b" not in metrics
    np.testing.assert_allclose(norm["b"].to_numpy(), sample_df["b"].to_numpy())
