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
basefol = r"H:\Shared drives\Reanalysis"
input_path = os.path.join(basefol, "pituffik_big.tif")
output_path = os.path.join(basefol, "pituffik_big_reproj.tif")


# -------------------
# 🔧 Utility functions

def get_lat_lon(ds):
    """Return latitude and longitude DataArrays from dataset, handling variable naming differences."""
    lat_name = "lat" if "lat" in ds.variables else "latitude"
    lon_name = "lon" if "lon" in ds.variables else "longitude"
    return ds[lat_name], ds[lon_name]

def wrap_lon(lon):
    """Wrap longitude to [-180, 180] range."""
    lon = np.asarray(lon)
    return (lon + 180) % 360 - 180


def adjust_lon(lon):
    """Convert longitude to [0, 360) range element-wise."""
    lon = np.asarray(lon)
    return np.where(lon < 0, lon + 360, lon)


def find_pixel(ds, lat1, lon1):
    """Find closest pixel in dataset grid to given lat/lon points."""
    lat_ds, lon_ds = get_lat_lon(ds)

    lon1_adj = adjust_lon(lon1)
    if lon_ds.ndim == 1:
        lon_ds_adj = adjust_lon(lon_ds)
    else:
        lon_ds_adj = adjust_lon(lon_ds.values)

    is_1d = (lat_ds.ndim == 1 and lon_ds.ndim == 1)
    if is_1d:
        lat2d, lon2d = np.meshgrid(lat_ds, lon_ds_adj, indexing="ij")
    else:
        lat2d = lat_ds.values
        lon2d = lon_ds_adj

    lat_t, lon_t = [], []
    for lat, lon in zip(lat1, lon1_adj):
        dist = (lat2d - lat) ** 2 + (lon2d - lon) ** 2
        y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)
        closest_lat = lat2d[y_idx, x_idx] if not is_1d else lat_ds[y_idx]
        closest_lon = lon2d[y_idx, x_idx] if not is_1d else lon_ds_adj[x_idx]
        closest_lon = wrap_lon(closest_lon)

        lat_t.append(closest_lat.item())
        lon_t.append(closest_lon)

        print(
            f"Closest grid point to ({lat:.4f},{wrap_lon(lon - 360):.4f}) "
            f"is at ({closest_lat:.4f}, {closest_lon:.4f}) index=({y_idx}, {x_idx})")

    return np.array(lat_t), np.array(lon_t)


def plot_grid(ds, color, ax, xmin, xmax, ymin, ymax):
    """Plot grid lines for dataset lat/lon that intersect raster bounds."""
    lat_ds, lon_ds = get_lat_lon(ds)
    lat, lon = lat_ds.values, lon_ds.values

    # Wrap lon to [-180,180] for plotting
    lon_wrapped = wrap_lon(lon)

    if lat.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon_wrapped, lat, indexing="ij")
    else:
        lon2d = lon_wrapped
        lat2d = lat

    # Use raster bounds as is (likely in [-180,180])
    # Find rows and cols intersecting bounds
    rows = np.where(
        (np.max(lon2d, axis=1) >= xmin) & (np.min(lon2d, axis=1) <= xmax) &
        (np.max(lat2d, axis=1) >= ymin) & (np.min(lat2d, axis=1) <= ymax))[0]
    cols = np.where(
        (np.max(lon2d, axis=0) >= xmin) & (np.min(lon2d, axis=0) <= xmax) &
        (np.max(lat2d, axis=0) >= ymin) & (np.min(lat2d, axis=0) <= ymax))[0]

    # Plot
    for i in rows:
        ax.plot(lon2d[i, :], lat2d[i, :], color=color, lw=0.5)
    for j in cols:
        ax.plot(lon2d[:, j], lat2d[:, j], color=color, lw=0.5)

    # Reduced label density
    max_labels = 50
    skip = max(1, int(np.ceil(np.sqrt(len(rows) * len(cols) / max_labels))))
    for i in rows[::skip]:
        for j in cols[::skip]:
            if xmin <= lon2d[i, j] <= xmax and ymin <= lat2d[i, j] <= ymax:
                ax.text(lon2d[i, j], lat2d[i, j],
                        f"({i},{j})", fontsize=4, ha="center", va="center", color="black")


def plot_closest(ds, lat1, lon1, ax):
    """Plot closest dataset grid points to given lat/lon points."""
    lat2, lon2 = find_pixel(ds, lat1, lon1)
    colors = ["orange", "green", "purple"]
    lon1_adj, lon2_adj = wrap_lon(lon1), wrap_lon(lon2)

    for idx in range(len(lat1)):
        ax.plot(
            lon1_adj[idx], lat1[idx], ".", markersize=2, color=colors[idx],
            label=f"PICK:({lat1[idx]:.4f}, {lon1_adj[idx]:.4f})")
        ax.plot(
            lon2_adj[idx], lat2[idx], "x", markersize=4, color=colors[idx],
            label=f"REF:({lat2[idx]:.4f}, {lon2_adj[idx]:.4f})")
        ax.plot(
            [lon1_adj[idx], lon2_adj[idx]], [lat1[idx], lat2[idx]], color=colors[idx], linestyle="--", linewidth=0.5)


def reproj_tif(input_path, output_path, dst_crs="EPSG:4326"):
    """Reproject a raster to given CRS."""
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({"crs": dst_crs, "transform": transform,
                      "width": width, "height": height})

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
            {"lat2d": (["lat", "lon"], lat2d_new),
             "lon2d": (["lat", "lon"], lon2d_new)},
            coords={"lat": lat_new, "lon": lon_new})

        weights_file = os.path.join(basefol, "carra_to_target_weights.nc")
        
        if not os.path.exists(weights_file):
            # Generate and save weights
            regridder = xe.Regridder(
                ds_c_orig, ds_target, method="bilinear", periodic=False,
                reuse_weights=False, filename=weights_file)
        else:
            # Reuse weights from file
            regridder = xe.Regridder(
                ds_c_orig, ds_target, method="bilinear", periodic=False,
                reuse_weights=True, filename=weights_file)
        
        ds_c = regridder(ds_c_orig["t2m"])
        ds_c.to_netcdf(output_path)
    else:
        print(f"CARRA regridded dataset already exists: {output_path}")

    return xr.open_dataset(output_path)


# ------------------
# Usage:

carra1_regrid_path = os.path.join(basefol, "carra1_regridded.nc")
carra2_regrid_path = os.path.join(basefol, "carra2_regridded.nc")
ds_c1_orig = xr.open_dataset(os.path.join(
    basefol, "carra\\raw", "carra1_2m_temperature_2023.nc"), chunks={'time': 10}, decode_timedelta=True)
ds_c1 = reproj_carra_if_needed(ds_c1_orig, carra1_regrid_path)
ds_c2_orig = xr.open_dataset(os.path.join(
    basefol, "carra\\raw", "carra2_2m_temperature_2023.nc"), chunks={'time': 10}, decode_timedelta=True)
ds_c2 = reproj_carra_if_needed(ds_c2_orig, carra2_regrid_path)
ds_e = xr.open_dataset(os.path.join(
    basefol, "era5\\raw", "era5_2m_temperature_2023.nc"), decode_timedelta=True)

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

    plot_grid(ds_c1, "red", ax, xmin, xmax, ymin, ymax)
    plot_grid(ds_c2, "green", ax, xmin, xmax, ymin, ymax)
    plot_grid(ds_e, "blue", ax, xmin, xmax, ymin, ymax)

    plot_closest(ds_c1, lat1, lon1, ax)
    plot_closest(ds_c2, lat1, lon1, ax)
    plot_closest(ds_e, lat1, lon1, ax)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("Reanalyses grid", fontsize=24, pad=0)
    ax.axis("off")

    plt.savefig(os.path.join(basefol, "rean.png"),
                dpi=200, bbox_inches="tight")
plt.close("all")
gc.collect()
