import numpy as np
import huracanpy
import pandas as pd
import xarray as xr
import calendar


def get_month_doy_edges_labels():
    month_days = np.array([calendar.monthrange(2001, m)[1] for m in range(1, 13)])
    bin_edges_doy = np.concatenate([[0], np.cumsum(month_days)])
    bin_edges_angle = bin_edges_doy * 2 * np.pi / 365

    month_centers_doy = (bin_edges_doy[:-1] + bin_edges_doy[1:]) / 2
    tick_positions = month_centers_doy * 2 * np.pi / 365
    month_labels = [calendar.month_abbr[m] for m in range(1, 13)]
    return bin_edges_doy, bin_edges_angle, tick_positions, month_labels


def interp_time(
    tracks,
    track_ids,
    freq="1h",
):

    new_tracks = []

    iterator = np.unique(track_ids.values)

    for tid in iterator:
        t = huracanpy.sel_id(tracks, track_ids, tid)
        T = pd.Series(
            pd.date_range(t.time.min().values, t.time.max().values, freq=freq)
        )
        T = T.to_xarray().rename("T").rename({"index": "time"})
        T["time"] = T.astype(str).str.slice(0, 19).astype(np.datetime64)
        new_tracks.append(
            t.set_coords("time")
            .swap_dims({"record": "time"})
            .interp(time=T.time)
            .swap_dims({"time": "record"})
            .reset_coords("time")
        )
    return xr.concat(new_tracks, dim="record")


def flag_periods(year_series, period_boundaries):
    period = np.where(
        year_series.between(period_boundaries[0], period_boundaries[1]),
        "CF",
        "nan",
    )
    period = np.where(
        year_series.between(period_boundaries[2], period_boundaries[3]),
        "F",
        period,
    )
    return period
