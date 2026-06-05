"""
tc-track-analogues
==================
TC attribution using track analogues as per Bourdin et al. 2025.
"""

from .target import load_target_track_hourly, extract_target_window_before_landfall, plot_target_case
from .catalogues import load_ibtracs_catalogue, load_CHAZ_catalogue, load_MIT_catalogue, load_SEAS520C_catalogue

__all__ = [
    "load_target_track_hourly", 
    "extract_target_window_before_landfall", 
    "plot_target_case",
    "load_ibtracs_catalogue",
    "load_CHAZ_catalogue",
    "load_MIT_catalogue", 
    "load_SEAS520C_catalogue"
]

__version__ = "0.1.0"
