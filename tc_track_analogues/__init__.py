"""
tc-track-analogues
==================
TC attribution using track analogues as per Bourdin et al. 2025.

Core workflow
-------------
1. Load observed and counterfactual TC track ensembles.
2. Find analogues between the two ensembles using :func:`find_analogues`.
3. Compute an attribution metric with :func:`compute_attribution`.
4. Visualise results with the helpers in :mod:`tc_track_analogues.plot`.
"""

from .analogues import find_analogues, compute_attribution, track_distance
from .target import load_target_track_hourly, extract_target_window_before_landfall, plot_target_case
from . import plot

__all__ = [
    "find_analogues",
    "compute_attribution",
    "track_distance",
    "plot",
    "load_target_track_hourly", 
    "extract_target_window_before_landfall", 
    "plot_target_case",
]

__version__ = "0.1.0"
