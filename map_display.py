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
            # Use broadcasting to create 2D distance array
            lat2d, lon2d = np.meshgrid(lat_ds.values, lon_ds.values, indexing="ij")
            dist = (lat2d - lat) ** 2 + (lon2d - lon) ** 2
            y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)

            closest_lat = lat_ds[y_idx].values
            closest_lon = lon_ds[x_idx].values - 360  # Convert back to [-180,180]
            print(
                f"Closest grid point to ({lat:.4f},{lon - 360:.4f}) is at ({closest_lat:.4f}, {closest_lon:.4f}) index=({y_idx}, {x_idx})")

        else:
            # 2D distance array directly
            dist = (lat_ds - lat) ** 2 + (lon_ds - lon) ** 2
            y_idx, x_idx = np.unravel_index(dist.argmin().values, dist.shape)

            closest_lat = lat_ds[y_idx, x_idx].values
            closest_lon = lon_ds[y_idx, x_idx].values - 360  # Convert back to [-180,180]
            print(
                f"Closest grid point to ({lat:.4f},{lon - 360:.4f}) is at ({closest_lat:.4f}, {closest_lon:.4f}) index=({y_idx}, {x_idx})")

        lat_t = np.append(lat_t, closest_lat)
        lon_t = np.append(lon_t, closest_lon)

    return [lat_t, lon_t]


def plot_grid(ds, col, ax, transformer, xmin, xmax, ymin, ymax):
    """
    Plots grid lines for the dataset's latitude and longitude coordinates.
    For rows/columns that intersect with the raster extent, it plots the entire row/column.

    Parameters:
        ds (xarray.Dataset): Dataset with latitude and longitude coordinates.
        col (str): Color for the grid lines.
        ax (matplotlib.axes.Axes): Axes to plot on.
        transformer (pyproj.Transformer): Transformer from EPSG:4326 to raster CRS.
        xmin, xmax, ymin, ymax (float): Bounds of the raster in the raster's CRS.
    """
    # Extract latitude and longitude arrays
    lat = ds['latitude'].values
    lon = ds['longitude'].values

    # Adjust longitude if needed
    if np.any(lon < 0):
        lon = lon + 360

    # Handle 1D or 2D lat/lon
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lat2d = lat
        lon2d = lon

    # Transform coordinates
    lon_flat = lon2d.flatten()
    lat_flat = lat2d.flatten()
    x_flat, y_flat = transformer.transform(lon_flat, lat_flat)

    # Reshape back
    x_grid = x_flat.reshape(lat2d.shape)
    y_grid = y_flat.reshape(lat2d.shape)

    # Check for rows and columns intersecting the extent
    rows_to_plot = np.where(
            (np.max(x_grid, axis=1) >= xmin) & (np.min(x_grid, axis=1) <= xmax) & (np.max(y_grid, axis=1) >= ymin) & (
                    np.min(y_grid, axis=1) <= ymax))[0]

    cols_to_plot = np.where(
            (np.max(x_grid, axis=0) >= xmin) & (np.min(x_grid, axis=0) <= xmax) & (np.max(y_grid, axis=0) >= ymin) & (
                    np.min(y_grid, axis=0) <= ymax))[0]

    # Plot full lines for these rows/columns
    for i in rows_to_plot:
        ax.plot(x_grid[i, :], y_grid[i, :], color=col, lw=0.5)
    for j in cols_to_plot:
        ax.plot(x_grid[:, j], y_grid[:, j], color=col, lw=0.5)

    # max_labels & skip calculation
    max_labels = 500
    num_points = len(rows_to_plot) * len(cols_to_plot)
    skip = max(1, int(np.ceil(np.sqrt(num_points / max_labels))))

    # Label points that are inside the extent only
    for i in rows_to_plot[::skip]:
        for j in cols_to_plot[::skip]:
            # Check if this point is inside the raster extent
            if (x_grid[i, j] >= xmin and x_grid[i, j] <= xmax and y_grid[i, j] >= ymin and y_grid[i, j] <= ymax):
                ax.text(
                        x_grid[i, j], y_grid[i, j], f"({i},{j})", fontsize=4, ha='center', va='center', color='black')


lat1 = np.array([76.5149, 76.52, 76.5])
lon1 = np.array([-68.7477, -68.74, -68.8])

# lat2 = [76.4918, 76.5226, 76.5110]
# lon2 = [-68.7533, -68.7207, -68.8030]

basefol = "H:\\Shared drives\\Reanalysis"
ds_c_path = os.path.join(basefol, "carra\\raw", "carra_2m_temperature_2023.nc")
ds_c = xr.open_dataset(ds_c_path, decode_timedelta=True, engine='netcdf4')
ds_e_path = os.path.join(basefol, "era5\\raw", "era5_2m_temperature_2023.nc")
ds_e = xr.open_dataset(ds_e_path, decode_timedelta=True, engine='netcdf4')

image_path = os.path.join(basefol, "pituffik.tif")

with rasterio.open(image_path) as src:
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot raster
    show(src, ax=ax)

# Raster CRS transformer
transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

# Get raster bounds in its CRS
bounds = src.bounds
xmin, ymin, xmax, ymax = bounds.left, bounds.bottom, bounds.right, bounds.top

plot_grid(ds_c, 'red', ax, transformer, xmin, xmax, ymin, ymax)
plot_grid(ds_e, 'blue', ax, transformer, xmin, xmax, ymin, ymax)


def plot_closest(ds):
    lat2, lon2 = find_pixel(ds, lat1, lon1)
    colors = ['red', 'green', 'purple']
    for idx, ((lat1_local, lon1_local), (lat2_local, lon2_local)) in enumerate(
            zip(zip(list(lat1), list(lon1)), zip(list(lat2), list(lon2)))):
        # Transform points from lat/lon to raster CRS
        x1, y1 = transformer.transform(lon1_local, lat1_local)
        x2, y2 = transformer.transform(lon2_local, lat2_local)

        # Plot first marker (from first list)
        ax.plot(x1, y1, marker='o', markersize=5, color=colors[idx], label=f'PICK:({lat1_local:.4f}, {lon1_local:.4f})')

        # Plot second marker (from second list)
        ax.plot(x2, y2, marker='x', markersize=7, color=colors[idx], label=f'REF:({lat2_local:.4f}, {lon2_local:.4f})')

        # Plot line connecting them
        ax.plot([x1, x2], [y1, y2], color=colors[idx], linestyle='--', linewidth=1)
    ax.legend(
            loc='upper left', ncols=2, bbox_to_anchor=(0.0, 1.0), ncol=1, fancybox=True, shadow=True, fontsize=12)


plot_closest(ds_c)
plot_closest(ds_e)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_title('Reanalyses grid', fontsize=24, pad=0)
ax.axis('off')
# plt.show()
plt.savefig(os.path.join(basefol, 'rean.png'), dpi=200, bbox_inches='tight')

# cleanup memory
import gc
import matplotlib.pyplot as plt

# Close all matplotlib figures
plt.close('all')

# Collect garbage (free memory)
gc.collect()
