import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from pyproj import Transformer
from rasterio.plot import show


def find_pixel(ds, lat1, lon1):
    lat_t = []
    lon_t = []

    # Adjust longitudes if needed
    if np.any(lon1 < 0):
        lon1 = lon1 + 360

    # Check if ds lat/lon are 1D or 2D
    lat_ds = ds["latitude"]
    lon_ds = ds["longitude"]
    is_1d = (lat_ds.ndim == 1 and lon_ds.ndim == 1)

    for lat, lon in zip(lat1, lon1):
        if is_1d:
            lat2d, lon2d = np.meshgrid(lat_ds, lon_ds, indexing="ij")
            dist = (lat2d - lat) ** 2 + (lon2d - lon) ** 2
            y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)

            closest_lat = lat_ds[y_idx].item()
            closest_lon = lon_ds[x_idx].item() - 360  # Subtract 360 for display

            print(
                    f"Closest grid point to ({lat:.4f},{lon - 360:.4f}) is at ({closest_lat:.4f}, {closest_lon:.4f}) index=({y_idx}, {x_idx})")

        else:
            dist = (lat_ds - lat) ** 2 + (lon_ds - lon) ** 2
            y_idx, x_idx = np.unravel_index(dist.argmin().item(), dist.shape)

            closest_lat = lat_ds[y_idx, x_idx].item()
            closest_lon = lon_ds[y_idx, x_idx].item()

            print(
                    f"Closest grid point to ({lat:.4f},{lon - 360:.4f}) is at ({closest_lat:.4f}, {closest_lon:.4f}) index=({y_idx}, {x_idx})")

        lat_t.append(closest_lat)
        lon_t.append(closest_lon)

    return np.array(lat_t), np.array(lon_t)


def plot_grid(ds, color, ax, transformer, xmin, xmax, ymin, ymax):
    """
    Plot grid lines for dataset lat/lon coordinates, only plotting rows/columns that intersect raster extent.
    """
    lat = ds['latitude'].values
    lon = ds['longitude'].values
    if np.any(lon < 0):
        lon += 360

    # Build 2D grid
    lon2d, lat2d = np.meshgrid(lon, lat) if lat.ndim == 1 else (lon, lat)

    # Transform coordinates
    lon_flat = lon2d.flatten()
    lat_flat = lat2d.flatten()
    x_flat, y_flat = transformer.transform(lon_flat, lat_flat)

    # Reshape back
    x_grid = x_flat.reshape(lat2d.shape)
    y_grid = y_flat.reshape(lat2d.shape)

    # Find rows/cols to plot
    rows = np.where(
            (np.max(x_grid, axis=1) >= xmin) & (np.min(x_grid, axis=1) <= xmax) & (np.max(y_grid, axis=1) >= ymin) & (
                    np.min(y_grid, axis=1) <= ymax))[0]
    cols = np.where(
            (np.max(x_grid, axis=0) >= xmin) & (np.min(x_grid, axis=0) <= xmax) & (np.max(y_grid, axis=0) >= ymin) & (
                    np.min(y_grid, axis=0) <= ymax))[0]

    # Plot lines
    for i in rows:
        ax.plot(x_grid[i, :], y_grid[i, :], color=color, lw=0.5)
    for j in cols:
        ax.plot(x_grid[:, j], y_grid[:, j], color=color, lw=0.5)

    # Label points (skip to limit clutter)
    max_labels = 500
    num_points = len(rows) * len(cols)
    skip = max(1, int(np.ceil(np.sqrt(num_points / max_labels))))

    for i in rows[::skip]:
        for j in cols[::skip]:
            if xmin <= x_grid[i, j] <= xmax and ymin <= y_grid[i, j] <= ymax:
                ax.text(x_grid[i, j], y_grid[i, j], f"({i},{j})", fontsize=4, ha='center', va='center', color='black')


def plot_closest(ds, lat1, lon1, transformer, ax):
    """
    Plot closest points from dataset to input lat/lon points.
    """
    lat2, lon2 = find_pixel(ds, lat1, lon1)
    colors = ['orange', 'green', 'purple']

    # Adjust longitudes if needed
    lon1_adj = lon1 + 360 if np.any(lon1 < 0) else lon1
    lon2_adj = lon2 + 360 if np.any(lon2 < 0) else lon2

    # Transform
    x1, y1 = transformer.transform(lon1_adj, lat1)
    x2, y2 = transformer.transform(lon2_adj, lat2)

    for idx in range(len(lat1)):
        ax.plot(
                x1[idx], y1[idx], marker='o', markersize=5, color=colors[idx],
                label=f'PICK:({lat1[idx]:.4f}, {lon1[idx]:.4f})')
        ax.plot(
                x2[idx], y2[idx], marker='x', markersize=7, color=colors[idx],
                label=f'REF:({lat2[idx]:.4f}, {lon2[idx]:.4f})')
        ax.plot([x1[idx], x2[idx]], [y1[idx], y2[idx]], color=colors[idx], linestyle='--', linewidth=1)

    ax.legend(loc='upper left', ncols=2, bbox_to_anchor=(0, 1), ncol=1, fancybox=True, shadow=True, fontsize=12)


# Input data
lat1 = np.array([76.5149, 76.52, 76.5])
lon1 = np.array([-68.7477, -68.74, -68.8])

basefol = r"H:\Shared drives\Reanalysis"
ds_c = xr.open_dataset(os.path.join(basefol, "carra\\raw", "carra_2m_temperature_2023.nc"), decode_timedelta=True)
ds_e = xr.open_dataset(os.path.join(basefol, "era5\\raw", "era5_2m_temperature_2023.nc"), decode_timedelta=True)

# Plot
with rasterio.open(os.path.join(basefol, "pituffik.tif")) as src:
    fig, ax = plt.subplots(figsize=(10, 10))
    show(src, ax=ax)

    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    xmin, ymin, xmax, ymax = src.bounds

    plot_grid(ds_c, 'red', ax, transformer, xmin, xmax, ymin, ymax)
    plot_grid(ds_e, 'blue', ax, transformer, xmin, xmax, ymin, ymax)

    plot_closest(ds_c, lat1, lon1, transformer, ax)
    plot_closest(ds_e, lat1, lon1, transformer, ax)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title('Reanalyses grid', fontsize=24, pad=0)
    ax.axis('off')

    plt.savefig(os.path.join(basefol, 'rean.png'), dpi=200, bbox_inches='tight')

# Cleanup
plt.close('all')
gc.collect()
