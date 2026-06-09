"""
tc-track-analogues
==================
TC attribution using track analogues as per Bourdin et al. 2025.
"""

from .target import (
    load_target_track_hourly,
    extract_target_window_before_landfall,
    plot_target_case,
)
from .catalogues import (
    load_ibtracs_catalogue,
    load_CHAZ_catalogue,
    load_MIT_catalogue,
    load_SEAS520C_catalogue,
)
from .analogues import add_dist_from_target_landfall, find_analogues, plot_analogues
from .utils import flag_periods, interp_time
from .comparison import (
    plot_diff_in_freq,
    plot_diff_in_intensity,
    plot_summary_diff_wind,
    plot_diff_in_seasonality,
)

__all__ = [
    "load_target_track_hourly",
    "extract_target_window_before_landfall",
    "plot_target_case",
    "load_ibtracs_catalogue",
    "load_CHAZ_catalogue",
    "load_MIT_catalogue",
    "load_SEAS520C_catalogue",
    "add_dist_from_target_landfall",
    "find_analogues",
    "flag_periods",
    "interp_time",
    "plot_analogues",
    "plot_diff_in_freq",
    "plot_diff_in_intensity",
    "plot_summary_diff_wind",
    "plot_diff_in_seasonality",
]

__version__ = "0.1.0"
