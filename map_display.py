import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
import xesmf as xe

# -------------------
# 📁 Config
os.environ["GDAL_DATA"] = r"C:\Users\FCQ\anaconda3\envs\reanalysis_thaao\Lib\site-packages\rasterio"

basefol = r"H:\Shared drives\Reanalysis"
input_path = os.path.join(basefol, "pituffik.tif")
output_path = os.path.join(basefol, "pituffik_reproj.tif")


# -------------------
# 🔧 Utility functions

def wrap_lon(lon):
    return (lon + 180) % 360 - 180


def adjust_lon(lon):
    return lon + 360 if np.any(lon < 0) else lon


def find_pixel(ds, lat1, lon1):
    """Find closest pixel in dataset grid to given lat/lon points."""
    lat_ds, lon_ds = ds["latitude"], ds["longitude"]
    lon1_adj = adjust_lon(lon1)

    # Determine grid type (1D or 2D)
    is_1d = (lat_ds.ndim == 1 and lon_ds.ndim == 1)
    if is_1d:
        lat2d, lon2d = np.meshgrid(lat_ds, lon_ds, indexing="ij")
    else:
        lat2d, lon2d = lat_ds, lon_ds

    lat_t, lon_t = [], []
    for lat, lon in zip(lat1, lon1_adj):
        dist = (lat2d - lat) ** 2 + (lon2d - lon) ** 2
        y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)
        closest_lat = lat2d[y_idx, x_idx] if not is_1d else lat_ds[y_idx]
        closest_lon = lon2d[y_idx, x_idx] if not is_1d else lon_ds[x_idx]
        closest_lon = wrap_lon(closest_lon.item())

        lat_t.append(closest_lat.item())
        lon_t.append(closest_lon)

        print(
                f"Closest grid point to ({lat:.4f},{wrap_lon(lon - 360):.4f}) "
                f"is at ({closest_lat:.4f}, {closest_lon:.4f}) index=({y_idx}, {x_idx})")

    return np.array(lat_t), np.array(lon_t)


def plot_grid(ds, color, ax, xmin, xmax, ymin, ymax):
    """Plot grid lines for dataset lat/lon that intersect raster bounds."""
    lat, lon = ds["latitude"].values, ds["longitude"].values
    lon_adj = adjust_lon(lon)
    lon2d, lat2d = (np.meshgrid(lon_adj, lat, indexing="ij") if lat.ndim == 1 else (lon_adj, lat))

    # Rows & columns intersecting raster bounds
    rows = np.where(
            (np.max(lon2d, axis=1) >= xmin) & (np.min(lon2d, axis=1) <= xmax) & (np.max(lat2d, axis=1) >= ymin) & (
                    np.min(lat2d, axis=1) <= ymax))[0]
    cols = np.where(
            (np.max(lon2d, axis=0) >= xmin) & (np.min(lon2d, axis=0) <= xmax) & (np.max(lat2d, axis=0) >= ymin) & (
                    np.min(lat2d, axis=0) <= ymax))[0]

    # Plot
    for i in rows:
        ax.plot(lon2d[i, :], lat2d[i, :], color=color, lw=0.5)
    for j in cols:
        ax.plot(lon2d[:, j], lat2d[:, j], color=color, lw=0.5)

    # Reduced label density
    max_labels = 5000
    skip = max(1, int(np.ceil(np.sqrt(len(rows) * len(cols) / max_labels))))
    for i in rows[::skip]:
        for j in cols[::skip]:
            if xmin <= lon2d[i, j] <= xmax and ymin <= lat2d[i, j] <= ymax:
                ax.text(lon2d[i, j], lat2d[i, j], f"({i},{j})", fontsize=4, ha="center", va="center", color="black")


def plot_closest(ds, lat1, lon1, ax):
    """Plot closest dataset grid points to given lat/lon points."""
    lat2, lon2 = find_pixel(ds, lat1, lon1)
    colors = ["orange", "green", "purple"]
    lon1_adj, lon2_adj = wrap_lon(lon1), wrap_lon(lon2)

    for idx in range(len(lat1)):
        ax.plot(
                lon1_adj[idx], lat1[idx], "o", markersize=5, color=colors[idx],
                label=f"PICK:({lat1[idx]:.4f}, {lon1_adj[idx]:.4f})")
        ax.plot(
                lon2_adj[idx], lat2[idx], "x", markersize=7, color=colors[idx],
                label=f"REF:({lat2[idx]:.4f}, {lon2_adj[idx]:.4f})")
        ax.plot(
                [lon1_adj[idx], lon2_adj[idx]], [lat1[idx], lat2[idx]], color=colors[idx], linestyle="--", linewidth=1)


def reproj_tif(input_path, output_path, dst_crs="EPSG:4326"):
    """Reproject a raster to given CRS."""
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})

        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                        source=rasterio.band(src, i), destination=rasterio.band(dst, i), src_transform=src.transform,
                        src_crs=src.crs, dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.nearest)
    return output_path


def reproj_tif_if_needed(input_path, output_path, dst_crs="EPSG:4326"):
    """Reproject tif only if output does not exist."""
    if not os.path.exists(output_path):
        print(f"Reprojecting {input_path} to {output_path} ...")
        reproj_tif(input_path, output_path, dst_crs)
    else:
        print(f"Reprojected TIF already exists: {output_path}")
    return output_path


def reproj_carra_if_needed(ds_c_orig, output_path):
    """Regrid CARRA only if output does not exist."""
    if not os.path.exists(output_path):
        print(f"Regridding CARRA dataset and saving to {output_path} ...")
        lon_new = np.arange(-180, 180, 0.1)
        lat_new = np.arange(-90, 90, 0.1)
        lon2d_new, lat2d_new = np.meshgrid(lon_new, lat_new)

        ds_target = xr.Dataset(
                {"lat": (["lat", "lon"], lat2d_new), "lon": (["lat", "lon"], lon2d_new)},
                coords={"lat": lat_new, "lon": lon_new})

        regridder = xe.Regridder(ds_c_orig, ds_target, method="bilinear", periodic=False, reuse_weights=True)
        ds_c = regridder(ds_c_orig["t2m"])
        ds_c.to_netcdf(output_path)
    else:
        print(f"CARRA regridded dataset already exists: {output_path}")

    return xr.open_dataset(output_path)


# ------------------
# Usage:

carra_regrid_path = os.path.join(basefol, "carra_regridded.nc")
ds_c_orig = xr.open_dataset(os.path.join(basefol, "carra\\raw", "carra_2m_temperature_2023.nc"), decode_timedelta=True)
ds_c = reproj_carra_if_needed(ds_c_orig, carra_regrid_path)
ds_e = xr.open_dataset(os.path.join(basefol, "era5\\raw", "era5_2m_temperature_2023.nc"), decode_timedelta=True)

# Reproject TIF if needed
output_path = reproj_tif_if_needed(input_path, output_path)

# Input data
lat1 = np.array([76.5149, 76.52, 76.5])
lon1 = np.array([-68.7477, -68.74, -68.8])

# -------------------
# 🖼️ Plotting
with rasterio.open(output_path) as src:
    print("Raster CRS:", src.crs)
    fig, ax = plt.subplots(figsize=(10, 10))
    show(src, ax=ax)
    xmin, ymin, xmax, ymax = src.bounds

    plot_grid(ds_c, "red", ax, xmin, xmax, ymin, ymax)
    plot_grid(ds_e, "blue", ax, xmin, xmax, ymin, ymax)

    plot_closest(ds_c, lat1, lon1, ax)
    plot_closest(ds_e, lat1, lon1, ax)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("Reanalyses grid", fontsize=24, pad=0)
    ax.axis("off")

    plt.savefig(os.path.join(basefol, "rean.png"), dpi=200, bbox_inches="tight")
plt.close("all")
gc.collect()
