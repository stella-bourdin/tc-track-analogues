"""Tests for tc_track_analogues core functions."""

import numpy as np
import pandas as pd
import pytest

from tc_track_analogues import find_analogues, compute_attribution, track_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_track(lon_start, lon_end, lat_start, lat_end, n=20):
    return pd.DataFrame({
        "lon": np.linspace(lon_start, lon_end, n),
        "lat": np.linspace(lat_start, lat_end, n),
    })


def _make_ensemble(n_members, base_track, noise_scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return [
        pd.DataFrame({
            "lon": base_track["lon"].values + rng.normal(0, noise_scale, len(base_track)),
            "lat": base_track["lat"].values + rng.normal(0, noise_scale, len(base_track)),
        })
        for _ in range(n_members)
    ]


# ---------------------------------------------------------------------------
# track_distance
# ---------------------------------------------------------------------------

class TestTrackDistance:
    def test_identical_tracks_zero(self):
        track = _make_track(-60, -90, 15, 30)
        assert track_distance(track, track) == pytest.approx(0.0, abs=1e-6)

    def test_parallel_tracks_nonzero(self):
        track_a = _make_track(-60, -90, 15, 30)
        track_b = _make_track(-60, -90, 20, 35)  # shifted north
        dist = track_distance(track_a, track_b)
        assert dist > 0

    def test_hausdorff_method(self):
        track_a = _make_track(-60, -90, 15, 30)
        track_b = _make_track(-60, -90, 20, 35)
        dist = track_distance(track_a, track_b, method="hausdorff")
        assert dist > 0

    def test_unknown_method_raises(self):
        track = _make_track(-60, -90, 15, 30)
        with pytest.raises(ValueError, match="Unknown method"):
            track_distance(track, track, method="unknown")

    def test_symmetry(self):
        track_a = _make_track(-60, -90, 15, 30)
        track_b = _make_track(-55, -85, 18, 28)
        assert track_distance(track_a, track_b) == pytest.approx(
            track_distance(track_b, track_a), rel=1e-6
        )


# ---------------------------------------------------------------------------
# find_analogues
# ---------------------------------------------------------------------------

class TestFindAnalogues:
    def setup_method(self):
        self.target = _make_track(-60, -90, 15, 30)
        # First member is identical to target, rest are noisy
        self.ensemble = [self.target.copy()] + _make_ensemble(49, self.target, noise_scale=3.0)

    def test_returns_dataframe(self):
        result = find_analogues(self.target, self.ensemble, n_analogues=5)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self):
        result = find_analogues(self.target, self.ensemble, n_analogues=5)
        assert "member" in result.columns
        assert "distance" in result.columns

    def test_sorted_by_distance(self):
        result = find_analogues(self.target, self.ensemble, n_analogues=10)
        assert (result["distance"].diff().dropna() >= 0).all()

    def test_n_analogues_respected(self):
        n = 7
        result = find_analogues(self.target, self.ensemble, n_analogues=n)
        assert len(result) == n

    def test_identical_member_is_top(self):
        result = find_analogues(self.target, self.ensemble, n_analogues=1)
        # Member 0 is identical, so distance should be ~0
        assert result.iloc[0]["distance"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_attribution
# ---------------------------------------------------------------------------

class TestComputeAttribution:
    def setup_method(self):
        self.target = _make_track(-60, -90, 15, 30)
        self.factual = _make_ensemble(50, self.target, noise_scale=2.0, seed=1)
        shifted = _make_track(-60, -90, 10, 25)  # shifted away from target
        self.counterfactual = _make_ensemble(50, shifted, noise_scale=2.0, seed=2)

    def test_returns_dict(self):
        result = compute_attribution(
            self.factual, self.counterfactual, self.target, n_analogues=5
        )
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = compute_attribution(
            self.factual, self.counterfactual, self.target, n_analogues=5
        )
        for key in ("factual_analogues", "counterfactual_analogues",
                    "probability_ratio", "mean_distance_factual",
                    "mean_distance_counterfactual"):
            assert key in result

    def test_probability_ratio_nonnegative(self):
        result = compute_attribution(
            self.factual, self.counterfactual, self.target, n_analogues=5
        )
        assert result["probability_ratio"] >= 0

    def test_mean_distances_nonnegative(self):
        result = compute_attribution(
            self.factual, self.counterfactual, self.target, n_analogues=5
        )
        assert result["mean_distance_factual"] >= 0
        assert result["mean_distance_counterfactual"] >= 0

    def test_factual_closer_than_counterfactual(self):
        result = compute_attribution(
            self.factual, self.counterfactual, self.target, n_analogues=10
        )
        # Factual ensemble is centred on target; counterfactual is shifted away
        assert result["mean_distance_factual"] < result["mean_distance_counterfactual"]
