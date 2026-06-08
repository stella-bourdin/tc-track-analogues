import numpy as np
from haversine import haversine, Unit
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def add_dist_from_target_landfall(
    catalogue,
    lf_lon,
    lf_lat,
):
    """
    For all points in a catalogue, add the distance to h_l defined by its longitude and latitude
    """
    tracks_coords = np.concatenate([[catalogue.lat], [catalogue.lon]]).T
    X = [lf_lat, lf_lon]  # Landfall coords
    dist = [
        haversine(c, [float(X[0]), float(X[1])], unit=Unit.KILOMETERS)
        for c in tracks_coords
    ]
    return xr.DataArray(dist, dims="record")


def extract_track_window(track, closest_pt, time_window=24):
    """
    Applies the window to the track
    """
    time_delta = track.time.values - closest_pt.time.values
    mask_window = (time_delta > -np.timedelta64(time_window, "h")) & (
        time_delta <= np.timedelta64(0, "h")
    )
    return track[mask_window]


# Distance functions for analogue computation
def dist_haversine(A, B):
    A_coords = np.concatenate([[A.lat], [A.lon]]).T
    B_coords = np.concatenate([[B.lat], [B.lon]]).T
    return [haversine(a, b, unit=Unit.DEGREES) for a, b in zip(A_coords, B_coords)]


def find_analogues(
    target_window, catalogue, exclusion_dist=1000, d_max=1.0, time_window=24
):
    """
    Function to get the analogues of a given case

    name: name of the target case
    data: cataogue of tracks
    target_window: 1-day trajectory of the target case before landfall
    intensity_col: list of column names that will be kept in the output
    exclusion_dist: Points that are further away than this distance from the target cases' landfall will not be considered at all. Used for performance.
    DIStarget_window_MAX: d_max in the paper
    CF_MIN
    """

    # Prepare input
    data_df = catalogue.to_dataframe()
    groups = data_df.groupby("track_id")

    # Select points within exclusion_dist of the landfall
    pts_within_exclusion_dist = data_df[data_df["dist2target"] < exclusion_dist]
    # Keep the closest point for each track_id
    closest_pt_per_track = (
        pts_within_exclusion_dist.sort_values("dist2target").groupby("track_id").first()
    )

    # Treat each track: extract window, compute analogue distance
    analogues = closest_pt_per_track.assign(analogue_dist=np.nan)
    for sid in closest_pt_per_track.index.values:
        closest_pt = closest_pt_per_track[
            closest_pt_per_track.index == sid
        ]  # The point being treated
        track = groups.get_group(sid)[
            [
                "track_id",
                "lon",
                "lat",
                "time",
            ]
        ]  # Track in which this point is
        # Check that the track is long enough
        if (
            track.time.min()
            >= (closest_pt.time - np.timedelta64(time_window, "h")).values
        ):
            analogues = analogues.drop(sid)  # Drop if track is too short
        else:
            # Extract the track over the window
            track_window = extract_track_window(track, closest_pt, time_window)
            # Compute the distance to the target
            analogue_dist = np.mean(dist_haversine(target_window, track_window))
            if analogue_dist > d_max:
                # If too far, remove the point
                analogues = analogues.drop(sid)
            else:
                # If close enough, store the analogue distance
                analogues.loc[sid, "analogue_dist"] = analogue_dist

    return analogues


def flag_periods(analogues, parameters):
    """
    Given a set of analogues and parameters containing the period coubndaries, 
    add a column identifying which analogues belong to the counter-factual or factual period
    """
    for c in analogues:
        analogues[c] = analogues[c].assign(
            period=np.where(
                analogues[c].time.dt.year.between(
                    parameters.dates.loc[c][0], parameters.dates.loc[c][1]
                ),
                "CF",
                "nan",
            )
        )
        analogues[c] = analogues[c].assign(
            period=np.where(
                analogues[c].time.dt.year.between(
                    parameters.dates.loc[c][2], parameters.dates.loc[c][3]
                ),
                "F",
                analogues[c].period,
            )
        )
    return analogues


def plot_analogues(
    analogues, catalogues, target, target_window, landfall, color_cf, color_f, pad=10
):
    """
    Given a set of analogues from different periods and catalogues, 
    plots the analogues along the target track.
    """
    fig, axs = plt.subplots(
        2,
        len(catalogues),
        figsize=(3 * len(catalogues), 6),
        subplot_kw=dict(projection=ccrs.PlateCarree()),
        squeeze=False,
    )

    for ax in axs.flatten():
        ax.coastlines()
        ax.plot(target.lon, target.lat, color="r", linewidth=1)
        ax.plot(target_window.lon, target_window.lat, color="r", linewidth=5)
        ax.scatter(landfall.lon, landfall.lat, color="w", edgecolor="r", zorder=10)
        ax.set_extent([
            landfall.lon - pad, landfall.lon + pad,
            landfall.lat - pad, landfall.lat + pad,
        ])

    rows = [("CF", 0, color_cf), ("F", 1, color_f)]

    for j, c in enumerate(catalogues):
        groups = catalogues[c].groupby("track_id")
        for period, row, color in rows:
            tids = analogues[c][analogues[c].period == period].index.values
            for tid in tids:
                t = groups.get_group(tid)
                lf = analogues[c].loc[tid]
                axs[row, j].plot(t.lon, t.lat, color=color, linewidth=1, alpha=0.5)
                axs[row, j].scatter(lf.lon, lf.lat, edgecolor=color, color="w", zorder=9)
            prefix = f"{c}\n" if row == 0 else ""
            axs[row, j].set_title(f"{prefix}{period} – {len(tids)}")

    fig.tight_layout()
    return fig, axs