import os

import matplotlib.pyplot as plt
import rasterio
import xarray as xr
from pyproj import Transformer
import numpy as np
from rasterio.plot import show

lat1 = [76.51493833333333, 76.52, 76.5]
lon1 = [-68.7476766666666, -68.74, -68.8]

basefol = "H:\\Shared drives\\Reanalysis"
ds_path = os.path.join(basefol, "carra\\raw", "carra_2m_temperature_2023.nc")
ds = xr.open_dataset(ds_path, decode_timedelta=True, engine='netcdf4')

image_path = os.path.join(basefol, "pituffik.tif")

with rasterio.open(image_path) as src:
    fig, ax = plt.subplots(figsize=(20, 20))

    # Plot raster
    show(src, ax=ax)

# Raster CRS transformer
transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

# Get raster bounds in its CRS
bounds = src.bounds
xmin, ymin, xmax, ymax = bounds.left, bounds.bottom, bounds.right, bounds.top

# Your lat/lon 2D arrays
lat = ds['latitude'].values  # ds["latitude"].values  # shape (y,x)
lon = ds["longitude"].values - 360

# Transform full arrays from lat/lon to raster CRS
lon_flat = lon.flatten()
lat_flat = lat.flatten()
x_flat, y_flat = transformer.transform(lon_flat, lat_flat)

# Reshape back to grid shape

x_grid = x_flat.reshape(lon.shape)[::-1, :]
y_grid = y_flat.reshape(lat.shape)[::-1, :]

for lat_local, lon_local in zip(lat1, lon1):
    x, y = transformer.transform(lon_local, lat_local)
    ax.plot(x, y, marker='o', markersize=2, label=(lat_local, lon_local))
ax.legend(
        loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=1, fancybox=True, shadow=True, fontsize=12)

# mask
# inside_mask = (x_grid > xmin) & (x_grid < xmax) & (y_grid > ymin) & (y_grid < ymax)
# no mask
inside_mask = (x_grid != np.nan) & (x_grid != np.nan) & (y_grid != np.nan) & (y_grid != np.nan)
rows_with_points = np.where(np.any(inside_mask, axis=1))[0]
cols_with_points = np.where(np.any(inside_mask, axis=0))[0]
# Plot grid lines for rows inside bounds
for i in rows_with_points[::10]:  # adjust step for clarity
    ax.plot(x_grid[i, :], y_grid[i, :], color='red', lw=0.5)

# Plot grid lines for columns inside bounds
for j in cols_with_points[::10]:  # adjust step for clarity
    ax.plot(x_grid[:, j], y_grid[:, j], color='red', lw=0.5)
ax.set_title('Reanalyses grid', fontsize=24, pad=0)
ax.axis('off')

# ax.set_xlim(xmin, xmax)
# ax.set_ylim(ymin, ymax)

# plt.show()
plt.savefig(os.path.join(basefol, 'rean.png'), dpi=200, bbox_inches='tight')
