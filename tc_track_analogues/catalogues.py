import huracanpy
import xarray as xr
import numpy as np

from .utils import interp_time


def load_ibtracs_catalogue(
    basin, filename=None, update=True, wind_var="usa_wind", pres_var="usa_pres"
):
    """
    Loads the ibtracs catalogue.
    If update = True, it will download the latest version of ibtracs online,
        and save this version as a csv in the data folder.
    If update = False, it will load the last saved csv version in the data/ folder.
    """
    # Default filename
    if filename is None:
        filename = "../data/ibtracs_" + basin + ".csv"
    if update:  # If update, load from internet and save to filename
        ib = huracanpy.load(
            source="ibtracs",
            ibtracs_subset=basin,
        )
        ib.hrcn.save(filename)
    else:  # If not update, load from saved file
        ib = huracanpy.load(filename)
    # Prepare the catalogue from the ibtracs file
    ## Restrict variables
    ib = ib[
        ["track_id", "name", "season", "lon", "lat", "time", wind_var, pres_var]
    ].rename({wind_var: "wind", pres_var: "pres"})
    ## Convert winds from kn to m/s
    ib["wind"] = ib.wind / 1.94384
    ## Keep longitude between -180 and 180
    ib["lon"] = xr.DataArray(
        np.where(ib.lon > 180, ib.lon - 360, ib.lon), dims="record"
    )
    ## Interpolate to 1-hourly
    ib = interp_time(ib, ib.track_id, freq="1h").reset_coords()
    return ib


def load_CHAZ_catalogue(filename="../data/CHAZ.nc"):
    ds = xr.open_dataset(filename)
    varlist = list(ds.variables.keys())
    if "stormID" in varlist:
        ds = ds.drop_vars("stormID")
    if "lifelength" in varlist:
        ds = ds.drop_vars("lifelength")
    ds["wind"] = ds.wind / 1.94384  # kn to m/s
    return ds.rename({"sid": "track_id"})


def load_MIT_catalogue(filename="../data/MIT.nc"):
    return xr.open_dataset(filename).rename({"sid": "track_id"})


def load_SEAS520C_catalogue(filename="../data/SEAS5-20C.nc"):
    return xr.open_dataset(filename).rename({"sid": "track_id"})
