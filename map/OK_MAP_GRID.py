# ─── Imports ─────────────────────────────────────────────────────────────────────
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
import xesmf as xe
import rasterio
from rasterio.plot import show
from rasterio.warp import reproject, calculate_default_transform, Resampling

# ─── Function Definitions ────────────────────────────────────────────────────────

def convert_dataset_lon_360_to_180(ds, lon_name="longitude"):
    """
    Convert longitude coordinates from 0..360 to -180..180 in xarray Dataset.
    Sorts dataset by longitude if it is a dimension coordinate.
    """
    lon = ds[lon_name]
    lon_180 = ((lon + 180) % 360) - 180
    ds = ds.assign_coords({lon_name: lon_180})

    if lon_name in ds.dims:
        ds = ds.sortby(lon_name)
    return ds


def find_pixel_indices(ds, latitudes, longitudes):
    """
    Find nearest pixel indices in dataset for arrays of latitudes and longitudes.
    Supports both 1D and 2D lat/lon coordinates.
    Returns list of (y_idx, x_idx) tuples.
    """
    lat_ds, lon_ds = ds["latitude"], ds["longitude"]
    is_1d = (lat_ds.ndim == 1 and lon_ds.ndim == 1)

    if is_1d:
        lon2d, lat2d = np.meshgrid(lon_ds, lat_ds)
    else:
        # lat_ds and lon_ds are 2D with dims (y, x)
        lat2d, lon2d = lat_ds.values, lon_ds.values

    indices = []
    for lat, lon in zip(latitudes, longitudes):
        dist = (lat2d - lat) ** 2 + (lon2d - lon) ** 2
        y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)
        indices.append((y_idx, x_idx))
    return indices


def find_pixel(ds, latitudes, longitudes, var_name="t2m"):
    """
    Find closest grid points for given lat/lon arrays in dataset.
    Returns arrays of closest latitudes, longitudes, and variable values.
    Prints info for each point.
    """
    lat_ds, lon_ds = ds["latitude"], ds["longitude"]
    is_1d = (lat_ds.ndim == 1 and lon_ds.ndim == 1)

    if is_1d:
        lon2d, lat2d = np.meshgrid(lon_ds, lat_ds)
    else:
        lat2d, lon2d = lat_ds.values, lon_ds.values

    lat_t, lon_t, vals = [], [], []
    for lat, lon in zip(latitudes, longitudes):
        dist = (lat2d - lat) ** 2 + (lon2d - lon) ** 2
        y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)

        closest_lat = lat_ds[y_idx] if is_1d else lat2d[y_idx, x_idx]
        closest_lon = lon_ds[x_idx] if is_1d else lon2d[y_idx, x_idx]
        val = ds[var_name].values[y_idx, x_idx]

        lat_t.append(closest_lat.item() if hasattr(closest_lat, 'item') else closest_lat)
        lon_t.append(closest_lon.item() if hasattr(closest_lon, 'item') else closest_lon)
        vals.append(val)

        print(f"Closest grid point to ({lat:.4f},{lon:.4f}) is at "
              f"({closest_lat:.4f}, {closest_lon:.4f}) index=({y_idx}, {x_idx}) "
              f"with {var_name} value = {val}")

    return np.array(lat_t), np.array(lon_t), np.array(vals)


def plot_closest(ds, latitudes, longitudes, ax):
    """
    Plot picked points and closest grid points on given axes.
    """
    lat2, lon2, vals = find_pixel(ds, latitudes, longitudes)
    colors = ["orange", "green", "purple"]

    for idx, (lat1, lon1) in enumerate(zip(latitudes, longitudes)):
        ax.plot(lon1, lat1, "o", markersize=3, color=colors[idx],
                label=f"PICK:({lat1:.4f}, {lon1:.4f})")
        ax.plot(lon2[idx], lat2[idx], "x", markersize=5, color=colors[idx],
                label=f"REF:({lat2[idx]:.4f}, {lon2[idx]:.4f}), val={vals[idx]:.2f}")
        ax.plot([lon1, lon2[idx]], [lat1, lat2[idx]],
                color=colors[idx], linestyle="--", linewidth=1)


def reproj_tif(input_path, output_path, dst_crs="EPSG:4326"):
    """
    Reproject a raster to given CRS without changing resolution.
    """
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({"crs": dst_crs, "transform": transform,
                       "width": width, "height": height})

        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    return


def lon_diff(lon1, lon2):
    """
    Compute difference between two longitudes considering wrap-around.
    Result is in [-180, 180].
    """
    d = lon2 - lon1
    return (d + 180) % 360 - 180


def extract_time_series(ds, indices, varname="t2m"):
    """
    Extract time series for variable varname from dataset ds at given (y,x) indices.
    """
    ts_list = []
    for y_idx, x_idx in indices:
        # Use dims y and x if they exist, otherwise latitude/longitude dims
        if 'y' in ds.dims and 'x' in ds.dims:
            ts = ds[varname].isel(y=y_idx, x=x_idx)
        elif 'latitude' in ds.dims and 'longitude' in ds.dims:
            ts = ds[varname].isel(latitude=y_idx, longitude=x_idx)
        else:
            raise ValueError("Dataset does not have expected spatial dims.")
        ts_list.append(ts)
    return ts_list


def filter_time(ds, start_date, end_date):
    """
    Filter dataset along time coordinate between start_date and end_date inclusive.
    """
    time_coord = next((coord for coord in ds.coords if "time" in coord), None)
    if time_coord is None:
        raise ValueError("No time coordinate found in dataset")

    time_index = ds[time_coord].to_index()
    mask = (time_index >= pd.to_datetime(start_date)) & (
        time_index <= pd.to_datetime(end_date))
    return ds.sel({time_coord: mask})


def resample_to_3h(ts):
    """
    Resample time series to 3-hourly averages.
    """
    time_dim = next((dim for dim in ts.dims if np.issubdtype(
        ts[dim].dtype, np.datetime64)), None)
    if time_dim is None:
        raise ValueError("No datetime dimension found to resample over.")
    return ts.resample({time_dim: "3h"}).mean()


def get_time_dim(ts):
    """
    Return the name of the datetime dimension in a time series.
    """
    for dim in ts.dims:
        if np.issubdtype(ts[dim].dtype, np.datetime64):
            return dim
    raise ValueError("No datetime dimension found.")


# ─── Script Execution ────────────────────────────────────────────────────────────

basefol = os.getcwd()

input_tif_path = os.path.join(basefol, "pituffik_big.tif")
output_tif_path = os.path.join(basefol, "pituffik_big_reproj.tif")
carra_nc_path = os.path.join(basefol, "carra_2m_temperature_2023.nc")
era5_nc_path = os.path.join(basefol, "era5_2m_temperature_2023.nc")

# Open datasets with appropriate chunking for performance
ds_c_all = xr.open_dataset(carra_nc_path, chunks={"time": 100})
ds_c_all = convert_dataset_lon_360_to_180(ds_c_all, lon_name="longitude")

ds_e_all = xr.open_dataset(era5_nc_path, chunks={"valid_time": 100})
# ERA5 longitudes already in -180..180, no conversion needed

# Select first time step and load into memory
ds_c = ds_c_all.isel(time=0).compute()
ds_e = ds_e_all.isel(valid_time=0).compute()

print(f"CARRA dataset dims: {ds_c.dims}")
print(f"ERA5 dataset dims: {ds_e.dims}")

# Prepare target grid for regridding: CARRA grid has 2D lat/lon arrays (y, x)
carra_lat = ds_c["latitude"].values
carra_lon = ds_c["longitude"].values
target_grid = xr.Dataset(
    {"lat": (["y", "x"], carra_lat), "lon": (["y", "x"], carra_lon)})

# Prepare ERA5 grid for regridding (ERA5 lat/lon 1D)
era_lat = ds_e["latitude"].values
era_lon = ds_e["longitude"].values
ds_e_sel = ds_e[["t2m"]].rename(
    {"latitude": "lat", "longitude": "lon"}).chunk({"lat": 21, "lon": 21})
era5_grid = xr.Dataset({"lat": (["lat"], era_lat), "lon": (["lon"], era_lon)})

print("Creating regridder...")
regridder = xe.Regridder(
    ds_e_sel, target_grid, method="bilinear", periodic=False, reuse_weights=False)

print("Regridding ERA5 to CARRA grid...")
ds_e_on_carra = regridder(ds_e_sel)

era5_regridded_path = os.path.join(basefol, "era5_regridded_to_carra.nc")
ds_e_on_carra.to_netcdf(era5_regridded_path)
print(f"ERA5 regridded dataset saved: {era5_regridded_path}")

print("Reprojecting pituffik.tif to EPSG:4326 (keep native resolution)...")
if not os.path.exists(output_tif_path):
    reproj_tif(input_tif_path, output_tif_path, dst_crs="EPSG:4326")
    print(f"Raster reprojected and saved: {output_tif_path}")
else:
    print(f"Output file already exists: {output_tif_path}, skipping reprojection.")

# Points of interest
lat1 = np.array([76.5149, 76.52, 76.5])
lon1 = np.array([-68.7477, -68.74, -68.8])

print("Extracting pixel indices for time series ...")
carra_indices = find_pixel_indices(ds_c_all, lat1, lon1)
era5_indices = find_pixel_indices(ds_e_all, lat1, lon1)

with rasterio.open(output_tif_path) as src:
    fig, ax = plt.subplots(figsize=(10, 10))
    show(src, ax=ax)

    ax.set_xlim(-68.9, -68.6)
    ax.set_ylim(76.45, 76.6)
    ax.set_title(
        "Pituffik GeoTIFF (native res.) with ERA5 (blue) & CARRA (red) grids", fontsize=16)
    ax.axis("on")

    # Plot ERA5 grid in blue (1D lat/lon)
    lon_e, lat_e = ds_e["longitude"].values, ds_e["latitude"].values
    lon2d_e, lat2d_e = np.meshgrid(lon_e, lat_e)
    for i in range(lon2d_e.shape[0]):
        ax.plot(lon2d_e[i, :], lat2d_e[i, :], "b-", linewidth=0.5)
    for j in range(lon2d_e.shape[1]):
        ax.plot(lon2d_e[:, j], lat2d_e[:, j], "b-", linewidth=0.5)

    # Plot CARRA grid in red (2D lat/lon)
    carra_lat, carra_lon = ds_c["latitude"].values, ds_c["longitude"].values
    for i in range(carra_lat.shape[0]):
        ax.plot(carra_lon[i, :], carra_lat[i, :], "r-", linewidth=0.5)
    for j in range(carra_lat.shape[1]):
        ax.plot(carra_lon[:, j], carra_lat[:, j], "r-", linewidth=0.5)

    plot_closest(ds_c, lat1, lon1, ax)
    plot_closest(ds_e, lat1, lon1, ax)

    plt.legend()
    plt.show()

print("Extracting time series from datasets at selected points...")
ts_c = extract_time_series(ds_c_all, carra_indices, varname="t2m")
ts_e = extract_time_series(ds_e_all, era5_indices, varname="t2m")

print("Filtering time series for January 1-10, 2023 ...")
ds_c_jan = filter_time(ds_c_all, "2023-01-01", "2023-01-10")
ds_e_jan = filter_time(ds_e_all, "2023-01-01", "2023-01-10")

# Extract timeseries again for filtered time period
ts_c_jan = extract_time_series(ds_c_jan, carra_indices, varname="t2m")
ts_e_jan = extract_time_series(ds_e_jan, era5_indices, varname="t2m")

print("Plotting time series ...")
fig, ax = plt.subplots(figsize=(12, 6))
colors = ["orange", "green", "purple"]

for i in range(len(ts_c_jan)):
    time_dim_c = get_time_dim(ts_c_jan[i])
    ax.plot(ts_c_jan[i][time_dim_c], ts_c_jan[i].values, color=colors[i],
            label=f"CARRA pt{i+1} ({lat1[i]:.4f}, {lon1[i]:.4f})")

    time_dim_e = get_time_dim(ts_e_jan[i])
    ax.plot(ts_e_jan[i][time_dim_e], ts_e_jan[i].values, "b-", 
            label=f"ERA5 pt{i+1} ({lat1[i]:.4f}, {lon1[i]:.4f})")

ax.set_xlabel("Date")
ax.set_ylabel("2m Temperature (K)")
ax.set_title("2m Temperature Time Series January 1-10, 2023")
ax.legend()
plt.grid(True)
plt.show()