"""
Plotting helpers for TC track analogues.

These functions require ``matplotlib`` and optionally ``cartopy``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_tracks(
    tracks: Sequence[pd.DataFrame],
    target_track: pd.DataFrame | None = None,
    lon_col: str = "lon",
    lat_col: str = "lat",
    ax=None,
    track_kwargs: dict | None = None,
    target_kwargs: dict | None = None,
):
    """Plot a collection of TC tracks on a map.

    Parameters
    ----------
    tracks:
        Sequence of track DataFrames to plot as thin lines.
    target_track:
        Optional reference track plotted with a thicker line.
    lon_col, lat_col:
        Column names for longitude and latitude.
    ax:
        Matplotlib axes to plot onto.  A new figure is created if *None*.
    track_kwargs:
        Extra keyword arguments forwarded to the ensemble track ``plot`` call.
    target_kwargs:
        Extra keyword arguments forwarded to the target track ``plot`` call.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    track_kwargs = {"color": "steelblue", "alpha": 0.4, "linewidth": 0.8, **(track_kwargs or {})}
    target_kwargs = {"color": "black", "linewidth": 2.5, "label": "Target", **(target_kwargs or {})}

    for track in tracks:
        ax.plot(track[lon_col], track[lat_col], **track_kwargs)

    if target_track is not None:
        ax.plot(target_track[lon_col], target_track[lat_col], **target_kwargs)
        ax.legend()

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return ax


def plot_analogues(
    target_track: pd.DataFrame,
    factual_analogues_df: pd.DataFrame,
    factual_ensemble: Sequence[pd.DataFrame],
    counterfactual_analogues_df: pd.DataFrame | None = None,
    counterfactual_ensemble: Sequence[pd.DataFrame] | None = None,
    lon_col: str = "lon",
    lat_col: str = "lat",
):
    """Plot the best analogues from the factual (and optionally counterfactual) ensemble.

    Parameters
    ----------
    target_track:
        The observed reference track.
    factual_analogues_df:
        Output of :func:`~tc_track_analogues.find_analogues` for the factual ensemble.
    factual_ensemble:
        The full factual ensemble (to look up member tracks by index).
    counterfactual_analogues_df:
        Output of :func:`~tc_track_analogues.find_analogues` for the counterfactual
        ensemble (optional).
    counterfactual_ensemble:
        The full counterfactual ensemble (optional).
    lon_col, lat_col:
        Column names for longitude and latitude.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_panels = 2 if counterfactual_analogues_df is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), squeeze=False)

    def _draw(ax, analogues_df, ensemble, title):
        for _, row in analogues_df.iterrows():
            t = ensemble[int(row["member"])]
            ax.plot(t[lon_col], t[lat_col], color="steelblue", alpha=0.6, linewidth=1)
        ax.plot(target_track[lon_col], target_track[lat_col], color="black", linewidth=2.5, label="Target")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()

    _draw(axes[0][0], factual_analogues_df, factual_ensemble, "Factual analogues")

    if counterfactual_analogues_df is not None and counterfactual_ensemble is not None:
        _draw(axes[0][1], counterfactual_analogues_df, counterfactual_ensemble, "Counterfactual analogues")

    fig.tight_layout()
    return fig
