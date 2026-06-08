"""
This module contains the functions relative to the target identification and plotting.
"""

from datetime import datetime
import huracanpy
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs

def _ibtracs_subset_choice(year, basin):
    """
    Chooses which ibtracs subset to load depending on the year of the cyclone
    """
    current_year = datetime.now().year
    if year >= current_year - 3:
        return "last3years"
    else:
        return basin

def load_target_track_hourly(name, season, basin, use_cache = True):
    """
    Loads the track of the target case and interpolate it to 1-hourly

    Set use_cache to False if you want to overwrite the cached target track data.
    """
    cache_file = "../data/cache/target_"+name+"_"+str(int(season))+"_"+basin+".pkl"
    if os.file.exists(cache_file) & use_cache:
        with open(cache_file, "rb") as f:
            return pkl.load(f)
    else:
        # Load ibtracs
        ib = huracanpy.load(source = "ibtracs", ibtracs_subset = _ibtracs_subset_choice(season, basin))
        # Select the target
        target = ib.where((ib.name == name) & (ib.time.dt.year == season), drop = True)
        target = target[["time", "lon", "lat", "usa_wind", "usa_pres", "track_id"]]
        # Interpolate to 1-hourly
        T = pd.Series(pd.date_range(target.time.min().values, target.time.max().values, freq='1h'))
        T = T.to_xarray().rename("T").rename({"index":"time"})
        T["time"] = T.astype(str).str.slice(0,19).astype(np.datetime64)
        target_1h = target.set_coords("time").swap_dims({"record": "time"}).interp(
            time=T.time
        )#.swap_dims({"time": "record"}).reset_coords("time")
        # Compute SSHS category
        target_1h["usa_sshs"] = huracanpy.tc.saffir_simpson_category(target_1h.usa_wind, wind_units="knots")
        with open(cache_file, "wb") as f:
            pkl.dump(target_1h, f)
        return target_1h

def extract_target_window_before_landfall(target, landfall_time, time_window = 24):
    """
    Applies the window to the target track
    """
    landfall_time = np.datetime64(landfall_time)
    time_delta = [t - landfall_time for t in target.time.values]
    mask_window = (time_delta > -np.timedelta64(time_window, 'h')) & \
                (time_delta <= np.timedelta64(0, 'h'))
    mask_window = xr.DataArray(mask_window, dims = "time", coords = {"time":target.time})
    return target.where(mask_window, drop = True)

def plot_target_case(target_1h, target_window, landfall, name):
    """
    Nice plot of the target case
    """
    
    fig, ax = plt.subplots(subplot_kw = dict(projection = ccrs.PlateCarree()), figsize = (8, 4))
    
    cmap = mpl.colormaps["RdBu_r"]
    cmaplist = [cmap(i) for i in [(128-30+25)//2, 25, 128+30, (225+128+30)//2, 225]]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, 5)
    
    # Plot full track
    lc = huracanpy.plot.fancyline(
        target_1h.lon,
        target_1h.lat,
        'k',
        vmin=-1.5, vmax=5.5,
        linewidths = target_1h.usa_wind,
        wrange = (0,4), wmin=0, wmax=150,
        ax=ax,
    )
    
    # Plot 1day before landfall
    lc = huracanpy.plot.fancyline(
        target_window.lon,
        target_window.lat,
        target_window.usa_sshs,
        vmin=0.5, vmax=5.5,
        linewidths = 7,
        ax=ax,
        cmap = cmap, 
    )

    # Highlight landfall
    plt.scatter(landfall.lon, landfall.lat, 
                transform = ccrs.PlateCarree(), 
                color = 'w', s = 200, zorder = 50, edgecolor = 'k', linewidth = 3)
    
    plt.colorbar(lc, extend="neither", label = "SSHS category")
    ax.coastlines(alpha = .5)
    ax.gridlines(draw_labels=["left", "bottom"])
    ax.set_title(name)