"""
Core functions for finding TC track analogues and computing attribution.

A *track* is represented as a :class:`pandas.DataFrame` with at least the
columns ``lon``, ``lat`` (and optionally ``time``).  An *ensemble* is a
list of such DataFrames, one per member.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import Sequence


# ---------------------------------------------------------------------------
# Distance metric
# ---------------------------------------------------------------------------

def track_distance(
    track_a: pd.DataFrame,
    track_b: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    method: str = "frechet",
) -> float:
    """Return the distance between two TC tracks.

    The tracks are resampled to the same number of equally-spaced points
    before comparison so that tracks of different lengths can be compared.

    Parameters
    ----------
    track_a, track_b:
        DataFrames with at least *lon_col* and *lat_col* columns.
    lon_col, lat_col:
        Column names for longitude and latitude.
    method:
        ``"frechet"`` (default) – discrete Fréchet distance;
        ``"hausdorff"`` – Hausdorff distance.

    Returns
    -------
    float
        Non-negative distance value in degrees.
    """
    pts_a = _resample_track(track_a[[lon_col, lat_col]].values)
    pts_b = _resample_track(track_b[[lon_col, lat_col]].values)

    if method == "hausdorff":
        return _hausdorff(pts_a, pts_b)
    elif method == "frechet":
        return _discrete_frechet(pts_a, pts_b)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'frechet' or 'hausdorff'.")


def _resample_track(pts: np.ndarray, n: int = 100) -> np.ndarray:
    """Resample a track to *n* equidistant points."""
    if len(pts) < 2:
        return np.tile(pts[0], (n, 1))
    dists = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = dists[-1]
    if total == 0:
        return np.tile(pts[0], (n, 1))
    s = np.linspace(0, total, n)
    resampled = np.column_stack([
        np.interp(s, dists, pts[:, i]) for i in range(pts.shape[1])
    ])
    return resampled


def _hausdorff(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Hausdorff distance between two point sets."""
    dist_matrix = cdist(pts_a, pts_b)
    return max(dist_matrix.min(axis=1).max(), dist_matrix.min(axis=0).max())


def _discrete_frechet(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Discrete Fréchet distance between two ordered point sequences."""
    n, m = len(pts_a), len(pts_b)
    ca = np.full((n, m), -1.0)

    def _c(i: int, j: int) -> float:
        if ca[i, j] >= 0:
            return ca[i, j]
        d = float(np.linalg.norm(pts_a[i] - pts_b[j]))
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(_c(0, j - 1), d)
        elif j == 0:
            ca[i, j] = max(_c(i - 1, 0), d)
        else:
            ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), d)
        return ca[i, j]

    return _c(n - 1, m - 1)


# ---------------------------------------------------------------------------
# Finding analogues
# ---------------------------------------------------------------------------

def find_analogues(
    target_track: pd.DataFrame,
    ensemble: Sequence[pd.DataFrame],
    n_analogues: int = 10,
    lon_col: str = "lon",
    lat_col: str = "lat",
    method: str = "frechet",
) -> pd.DataFrame:
    """Find the closest analogues to *target_track* within *ensemble*.

    Parameters
    ----------
    target_track:
        The reference track to match against.
    ensemble:
        A sequence of candidate tracks to search through.
    n_analogues:
        Maximum number of analogues to return.
    lon_col, lat_col:
        Column names for longitude and latitude.
    method:
        Distance method passed to :func:`track_distance`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ``member`` (index into *ensemble*),
        ``distance``, sorted by ascending distance.
    """
    distances = [
        track_distance(target_track, track, lon_col=lon_col, lat_col=lat_col, method=method)
        for track in ensemble
    ]
    result = pd.DataFrame({"member": range(len(ensemble)), "distance": distances})
    result = result.sort_values("distance").reset_index(drop=True)
    return result.head(n_analogues)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def compute_attribution(
    factual_ensemble: Sequence[pd.DataFrame],
    counterfactual_ensemble: Sequence[pd.DataFrame],
    target_track: pd.DataFrame,
    n_analogues: int = 10,
    lon_col: str = "lon",
    lat_col: str = "lat",
    method: str = "frechet",
) -> dict:
    """Compute an attribution metric using the track-analogue method.

    Finds the *n_analogues* closest tracks in both the factual and
    counterfactual ensembles and returns a summary of their distances,
    which can be used to infer the effect of climate change on the
    likelihood of the target track.

    Parameters
    ----------
    factual_ensemble:
        Ensemble of tracks representing present-day (factual) climate.
    counterfactual_ensemble:
        Ensemble of tracks representing the counterfactual climate
        (e.g., pre-industrial).
    target_track:
        The observed TC track to attribute.
    n_analogues:
        Number of closest analogues to use from each ensemble.
    lon_col, lat_col:
        Column names for longitude and latitude.
    method:
        Distance method passed to :func:`track_distance`.

    Returns
    -------
    dict
        Dictionary with keys:

        ``factual_analogues``
            :class:`~pandas.DataFrame` of closest factual analogues.
        ``counterfactual_analogues``
            :class:`~pandas.DataFrame` of closest counterfactual analogues.
        ``probability_ratio``
            Ratio of the fraction of analogues within a threshold
            distance in the factual vs. counterfactual ensemble.
        ``mean_distance_factual``
            Mean distance of the *n_analogues* closest factual members.
        ``mean_distance_counterfactual``
            Mean distance of the *n_analogues* closest counterfactual members.
    """
    factual_analogues = find_analogues(
        target_track, factual_ensemble, n_analogues=n_analogues,
        lon_col=lon_col, lat_col=lat_col, method=method,
    )
    counterfactual_analogues = find_analogues(
        target_track, counterfactual_ensemble, n_analogues=n_analogues,
        lon_col=lon_col, lat_col=lat_col, method=method,
    )

    threshold = factual_analogues["distance"].max()

    p_factual = (factual_analogues["distance"] <= threshold).sum() / len(factual_ensemble)
    p_counterfactual = (
        (counterfactual_analogues["distance"] <= threshold).sum() / len(counterfactual_ensemble)
    )

    probability_ratio = p_factual / p_counterfactual if p_counterfactual > 0 else np.inf

    return {
        "factual_analogues": factual_analogues,
        "counterfactual_analogues": counterfactual_analogues,
        "probability_ratio": probability_ratio,
        "mean_distance_factual": factual_analogues["distance"].mean(),
        "mean_distance_counterfactual": counterfactual_analogues["distance"].mean(),
    }
