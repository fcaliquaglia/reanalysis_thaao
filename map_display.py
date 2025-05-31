import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling


def wrap_lon(lon):
    """Wrap longitude to -180 to 180 range."""
    return (lon + 180) % 360 - 180


def adjust_lon(lon):
    """Adjust longitudes to 0-360 if any are negative."""
    return lon + 360 if np.any(lon < 0) else lon


def find_pixel(ds, lat1, lon1):
    """Find closest pixel in dataset grid to given lat/lon points."""
    lat_t, lon_t = [], []
    lat_ds, lon_ds = ds["latitude"], ds["longitude"]

    # Adjust input longitudes
    lon1_adj = adjust_lon(lon1)

    # Check if lat/lon in dataset are 1D or 2D
    is_1d = (lat_ds.ndim == 1 and lon_ds.ndim == 1)

    # Prepare 2D lat/lon if needed
    if is_1d:
        lat2d, lon2d = np.meshgrid(lat_ds, lon_ds, indexing="ij")

    for lat, lon in zip(lat1, lon1_adj):
        if is_1d:
            dist = (lat2d - lat) ** 2 + (lon2d - lon) ** 2
        else:
            dist = (lat_ds - lat) ** 2 + (lon_ds - lon) ** 2

        y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)
        closest_lat = lat_ds[y_idx] if is_1d else lat_ds[y_idx, x_idx]
        closest_lon = lon_ds[x_idx] if is_1d else lon_ds[y_idx, x_idx]

        # Wrap closest lon for display
        closest_lon = wrap_lon(closest_lon.item())
        lat_t.append(closest_lat.item())
        lon_t.append(closest_lon)

        print(
            f"Closest grid point to ({lat:.4f},{wrap_lon(lon - 360):.4f}) "
            f"is at ({closest_lat:.4f}, {closest_lon:.4f}) index=({y_idx}, {x_idx})"
        )

    return np.array(lat_t), np.array(lon_t)


def plot_grid(ds, color, ax, xmin, xmax, ymin, ymax):
    """Plot grid lines for dataset lat/lon that intersect raster bounds."""
    lat, lon = ds["latitude"].values, ds["longitude"].values
    lon_adj = adjust_lon(lon)

    lon2d, lat2d = (np.meshgrid(lon_adj, lat, indexing="ij")
                     if lat.ndim == 1 else (lon_adj, lat))

    x_grid, y_grid = lon2d, lat2d

    # Identify intersecting rows/columns
    rows = np.where(
        (np.max(x_grid, axis=1) >= xmin) & (np.min(x_grid, axis=1) <= xmax) &
        (np.max(y_grid, axis=1) >= ymin) & (np.min(y_grid, axis=1) <= ymax)
    )[0]

    cols = np.where(
        (np.max(x_grid, axis=0) >= xmin) & (np.min(x_grid, axis=0) <= xmax) &
        (np.max(y_grid, axis=0) >= ymin) & (np.min(y_grid, axis=0) <= ymax)
    )[0]

    # Plot grid lines
    for i in rows:
        ax.plot(x_grid[i, :], y_grid[i, :], color=color, lw=0.5)
    for j in cols:
        ax.plot(x_grid[:, j], y_grid[:, j], color=color, lw=0.5)

    # Label a reduced number of points
    max_labels = 5000
    skip = max(1, int(np.ceil(np.sqrt(len(rows) * len(cols) / max_labels))))
    for i in rows[::skip]:
        for j in cols[::skip]:
            if xmin <= x_grid[i, j] <= xmax and ymin <= y_grid[i, j] <= ymax:
                ax.text(x_grid[i, j], y_grid[i, j], f"({i},{j})",
                        fontsize=4, ha="center", va="center", color="black")


def plot_closest(ds, lat1, lon1, ax):
    """Plot closest points from dataset to input points."""
    lat2, lon2 = find_pixel(ds, lat1, lon1)
    colors = ["orange", "green", "purple"]

    lon1_adj = wrap_lon(lon1)
    lon2_adj = wrap_lon(lon2)

    for idx in range(len(lat1)):
        ax.plot(lon1_adj[idx], lat1[idx], marker="o", markersize=5,
                color=colors[idx], label=f"PICK:({lat1[idx]:.4f}, {lon1_adj[idx]:.4f})")
        ax.plot(lon2_adj[idx], lat2[idx], marker="x", markersize=7,
                color=colors[idx], label=f"REF:({lat2[idx]:.4f}, {lon2_adj[idx]:.4f})")
        ax.plot([lon1_adj[idx], lon2_adj[idx]], [lat1[idx], lat2[idx]],
                color=colors[idx], linestyle="--", linewidth=1)


def reproj_tif(input_path, output_path, dst_crs="EPSG:4326"):
    """Reproject a raster file to the given CRS."""
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)

        kwargs = src.meta.copy()
        kwargs.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})

        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
    return output_path


# Inputs
lat1 = np.array([76.5149, 76.52, 76.5])
lon1 = np.array([-68.7477, -68.74, -68.8])
basefol = r"H:\Shared drives\Reanalysis"
input_path = os.path.join(basefol, "pituffik.tif")
output_path = os.path.join(basefol, "pituffik_reproj.tif")

# Datasets
ds_c = xr.open_dataset(os.path.join(basefol, "carra\\raw", "carra_2m_temperature_2023.nc"), decode_timedelta=True)
ds_e = xr.open_dataset(os.path.join(basefol, "era5\\raw", "era5_2m_temperature_2023.nc"), decode_timedelta=True)

# Reproject if needed
# output_path = reproj_tif(input_path, output_path)

# Plotting
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
