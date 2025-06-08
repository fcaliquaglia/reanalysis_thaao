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


def reproj_tif(input_path, output_path, dst_crs="EPSG:4326"):
    """Reproject a raster to given CRS without changing resolution."""
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


def convert_lon_360_to_180(lon):
    return (lon + 180) % 360 - 180


def adjust_lon(lon):
    lon = np.array(lon)
    lon[lon < 0] += 360
    return lon


def wrap_lon(lon):
    lon = np.array(lon)
    return ((lon + 180) % 360) - 180


def lon_diff(lon1, lon2):
    d = lon2 - lon1
    return (d + 180) % 360 - 180


def find_pixel(ds, lat1, lon1, var_name="t2m"):
    lat_ds, lon_ds = ds["latitude"], ds["longitude"]
    lon1_adj = adjust_lon(lon1)
    is_1d = (lat_ds.ndim == 1 and lon_ds.ndim == 1)
    if is_1d:
        lon2d, lat2d = np.meshgrid(lon_ds, lat_ds)
    else:
        lat2d, lon2d = lat_ds, lon_ds

    lat_t, lon_t, vals = [], [], []
    for lat, lon in zip(lat1, lon1_adj):
        dist = (lat2d - lat) ** 2 + lon_diff(lon2d, lon) ** 2
        arr = dist.values if hasattr(dist, "values") else dist
        y_idx, x_idx = np.unravel_index(np.argmin(arr), arr.shape)
        closest_lat = lat2d[y_idx, x_idx] if not is_1d else lat_ds[y_idx]
        closest_lon = lon2d[y_idx, x_idx] if not is_1d else lon_ds[x_idx]
        closest_lon = wrap_lon(closest_lon.item())

        if is_1d:
            val = ds[var_name].values[y_idx, x_idx]
        else:
            val = ds[var_name].values[y_idx, x_idx]

        lat_t.append(closest_lat.item())
        lon_t.append(closest_lon)
        vals.append(val)

        print(
            f"Closest grid point to ({lat:.4f},{wrap_lon(lon - 360):.4f}) "
            f"is at ({closest_lat:.4f}, {closest_lon:.4f}) index=({y_idx}, {x_idx}) "
            f"with {var_name} value = {val}")

    return np.array(lat_t), np.array(lon_t), np.array(vals)


def plot_closest(ds, lat1, lon1, ax):
    lat2, lon2, vals = find_pixel(ds, lat1, lon1)
    colors = ["orange", "green", "purple"]
    lon1_adj, lon2_adj = wrap_lon(lon1), wrap_lon(lon2)

    for idx in range(len(lat1)):
        ax.plot(lon1_adj[idx], lat1[idx], "o", markersize=3, color=colors[idx],
                label=f"PICK:({lat1[idx]:.4f}, {lon1_adj[idx]:.4f})")
        ax.plot(lon2_adj[idx], lat2[idx], "x", markersize=5, color=colors[idx],
                label=f"REF:({lat2[idx]:.4f}, {lon2_adj[idx]:.4f}), val={vals[idx]:.2f}")
        ax.plot([lon1_adj[idx], lon2_adj[idx]], [lat1[idx], lat2[idx]],
                color=colors[idx], linestyle="--", linewidth=1)


def find_pixel_indices(ds, lat1, lon1):
    lat_ds, lon_ds = ds["latitude"], ds["longitude"]
    lon1_adj = adjust_lon(lon1)
    is_1d = (lat_ds.ndim == 1 and lon_ds.ndim == 1)
    if is_1d:
        lon2d, lat2d = np.meshgrid(lon_ds, lat_ds)
    else:
        lat2d, lon2d = lat_ds, lon_ds

    indices = []
    for lat, lon in zip(lat1, lon1_adj):
        dist = (lat2d - lat) ** 2 + lon_diff(lon2d, lon) ** 2
        arr = dist.values if hasattr(dist, "values") else dist
        y_idx, x_idx = np.unravel_index(np.argmin(arr), arr.shape)
        indices.append((y_idx, x_idx))
    return indices


def extract_time_series(ds, indices, varname="t2m"):
    ts_list = []
    for y_idx, x_idx in indices:
        if 'latitude' in ds.dims and 'longitude' in ds.dims:
            ts = ds[varname].isel(latitude=y_idx, longitude=x_idx)
        else:
            ts = ds[varname].isel(y=y_idx, x=x_idx)
        ts_list.append(ts)
    return ts_list


def filter_time(ds):
    time_coord = None
    for coord in ds.coords:
        if 'time' in coord:
            time_coord = coord
            break
    if time_coord is None:
        raise ValueError("No time coordinate found in dataset")

    time_index = ds[time_coord].to_index()
    mask = (time_index >= pd.to_datetime(start_date)) & (
        time_index <= pd.to_datetime(end_date))
    return ds.sel({time_coord: mask})


def resample_to_3h(ts):
    time_dim = None
    for dim in ts.dims:
        if np.issubdtype(ts[dim].dtype, np.datetime64):
            time_dim = dim
            break
    if time_dim is None:
        raise ValueError("No datetime dimension found to resample over.")
    return ts.resample({time_dim: "3h"}).mean()


def get_time_dim(ts):
    for dim in ts.dims:
        if np.issubdtype(ts[dim].dtype, np.datetime64):
            return dim
    raise ValueError("No datetime dimension found.")

# ─── Script Execution ────────────────────────────────────────────────────────────


basefol = os.path.dirname(os.path.abspath(__file__))
input_tif_path = os.path.join(basefol, "pituffik_big.tif")
output_tif_path = os.path.join(basefol, "pituffik_big_reproj.tif")
carra_nc_path = os.path.join(basefol, "carra_2m_temperature_2023.nc")
era5_nc_path = os.path.join(basefol, "era5_2m_temperature_2023.nc")

ds_c_all = xr.open_dataset(carra_nc_path, chunks={})
ds_e_all = xr.open_dataset(era5_nc_path, chunks={})

ds_c = ds_c_all.isel(time=0).compute()
ds_c = ds_c.isel(y=slice(None, None, -1))
ds_e = ds_e_all.isel(valid_time=0).compute()

print(f"CARRA dataset dims: {ds_c.dims}")
print(f"ERA5 dataset dims: {ds_e.dims}")

carra_lat = ds_c["latitude"].values
carra_lon = ds_c["longitude"].values
target_grid = xr.Dataset(
    {"lat": (["y", "x"], carra_lat), "lon": (["y", "x"], carra_lon)})

era_lat = ds_e["latitude"].values
era_lon = ds_e["longitude"].values
ds_e_sel = ds_e[['t2m']].rename(
    {'latitude': 'lat', 'longitude': 'lon'}).chunk({'lat': 21, 'lon': 21})
era5_grid = xr.Dataset({'lat': (['lat'], era_lat), 'lon': (['lon'], era_lon)})

print("Creating regridder...")
regridder = xe.Regridder(ds_e_sel, target_grid, 'bilinear',
                         periodic=False, reuse_weights=False)
print("Regridding ERA5 to CARRA grid...")
ds_e_on_carra = regridder(ds_e_sel)

era5_regridded_path = os.path.join(basefol, "era5_regridded_to_carra.nc")
ds_e_on_carra.to_netcdf(era5_regridded_path)
print(f"ERA5 regridded dataset saved: {era5_regridded_path}")

print("Reprojecting pituffik.tif to EPSG:4326 (keep native resolution)...")
reproj_tif(input_tif_path, output_tif_path, dst_crs="EPSG:4326")
print(f"Raster reprojected and saved: {output_tif_path}")

lat1 = np.array([76.5149, 76.52, 76.5])
lon1 = np.array([-68.7477, -68.74, -68.8])

with rasterio.open(output_tif_path) as src:
    fig, ax = plt.subplots(figsize=(10, 10))
    show(src, ax=ax)

    lon_c_converted = convert_lon_360_to_180(ds_c["longitude"].values)

    ax.set_xlim(-68.9, -68.6)
    ax.set_ylim(76.45, 76.6)
    ax.set_title(
        "Pituffik GeoTIFF (native res.) with ERA5 (blue) & CARRA (red) grids", fontsize=16)
    ax.axis("on")

    lat_e = ds_e["latitude"].values
    lon_e = ds_e["longitude"].values
    lon2d_e, lat2d_e = np.meshgrid(lon_e, lat_e)
    for i in range(lon2d_e.shape[0]):
        ax.plot(lon2d_e[i, :], lat2d_e[i, :], color="blue", lw=1)
    for j in range(lon2d_e.shape[1]):
        ax.plot(lon2d_e[:, j], lat2d_e[:, j], color="blue", lw=1)

    lat_c = ds_c["latitude"].values
    lon_c = lon_c_converted
    for i in range(lon_c.shape[0]):
        ax.plot(lon_c[i, :], lat_c[i, :], color="red", lw=1)
    for j in range(lon_c.shape[1]):
        ax.plot(lon_c[:, j], lat_c[:, j], color="red", lw=1)

    print("Plotting closest points on ERA5 grid:")
    plot_closest(ds_e, lat1, lon1, ax)

    print("Plotting closest points on CARRA grid:")
    plot_closest(ds_c, lat1, lon1, ax)

    plt.savefig(os.path.join(basefol, "rean_regridded_to_carra_with_closest_points.png"),
                dpi=200, bbox_inches="tight")

print("Extracting pixel indices for time series ...")
carra_indices = find_pixel_indices(ds_c_all, lat1, lon1)
era5_indices = find_pixel_indices(ds_e_all, lat1, lon1)

print("Extracting time series for each point ...")
carra_ts = extract_time_series(ds_c_all, carra_indices, "t2m")
era5_ts = extract_time_series(ds_e_all, era5_indices, "t2m")

start_date = "2023-01-01"
end_date = "2023-06-30"
era5_ts_filtered = [filter_time(ts) for ts in era5_ts]
carra_ts_filtered = [filter_time(ts) for ts in carra_ts]

print("Plotting combined time series with stacked panels (Jan-Jun)...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

era5_ts_filtered[0].plot(ax=ax1, color='blue',
                         label=f"ERA5 ({lat1[0]:.4f},{lon1[0]:.4f})")
colors = ['orange', 'green', 'purple']
for i in range(len(lat1)):
    carra_ts_filtered[i].plot(ax=ax1, color=colors[i],
                              label=f"CARRA ({lat1[i]:.4f}, {lon1[i]:.4f})")

ax1.set_title("ERA5 + CARRA Time Series (Jan - Jun)")
ax1.set_ylabel("Temperature (K)")
ax1.grid(True)
ax1.legend(loc='upper left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax1.tick_params(axis='x', rotation=30)

for i in range(len(lat1)):
    carra_ts_filtered[i].plot(ax=ax2, color=colors[i],
                              label=f"CARRA ({lat1[i]:.4f}, {lon1[i]:.4f})")

ax2.set_title("CARRA Time Series Only (Jan - Jun)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Temperature (K)")
ax2.grid(True)
ax2.legend(loc='upper left')
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax2.tick_params(axis='x', rotation=30)

plt.tight_layout()
ts_figure_path = os.path.join(
    basefol, "temperature_time_series_stacked_panels_JanJun.png")
plt.savefig(ts_figure_path, dpi=200, bbox_inches="tight")
print(f"Stacked panels time series figure saved as: {ts_figure_path}")
plt.show()

era5_ts_3h = [resample_to_3h(ts) for ts in era5_ts_filtered]
time_dim_era5 = get_time_dim(era5_ts_3h[0])
time_dim_carra = get_time_dim(carra_ts_filtered[0])
common_times = era5_ts_3h[0][time_dim_era5].to_index().intersection(
    carra_ts_filtered[0][time_dim_carra].to_index())

era5_common = [ts.sel({time_dim_era5: common_times}) for ts in era5_ts_3h]
carra_common = [ts.sel({time_dim_carra: common_times})
                for ts in carra_ts_filtered]

era5_values = np.stack([ts.values for ts in era5_common], axis=1)
era5_mean_series = np.mean(era5_values, axis=1)

for i, carra_ts in enumerate(carra_common):
    carra_vals = carra_ts.values
    diff = carra_vals - era5_mean_series
    mean_diff = np.mean(diff)
    print(
        f"CARRA point {i+1} mean difference from ERA5 average (Jan-Jun): {mean_diff:.3f}")

plt.close("all")
gc.collect()
print("✅ All done!")
