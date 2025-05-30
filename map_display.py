import os

import matplotlib.pyplot as plt
import rasterio
import xarray as xr
from pyproj import Transformer
from rasterio.plot import show

lat1 = [76.5149, 76.52, 76.5]
lon1 = [-68.7477, -68.74, -68.8]

lat2 = [76.4918, 76.5226, 76.5110]
lon2 = [-68.7533, -68.7207, -68.8030]

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

# Plot grid lines for rows inside bounds
for i in range(x_grid.shape[0]):
    ax.plot(x_grid[i, :], y_grid[i, :], color='red', lw=0.5)

# Plot grid lines for columns inside bounds
for j in range(y_grid.shape[1]):  # adjust step for clarity
    ax.plot(x_grid[:, j], y_grid[:, j], color='red', lw=0.5)

for i in range(0, x_grid.shape[0], 5):
    for j in range(0, y_grid.shape[1], 5):
        x_val = x_grid[i, j]
        y_val = y_grid[i, j]
        ax.text(
            x_val, y_val,
            f"({i},{j})",
            fontsize=4, color='black', ha='center', va='center'
        )

colors1 = ['red', 'green', 'blue']
colors2 = ['red', 'green', 'blue']
for idx, (lat_local, lon_local) in enumerate(zip(lat1, lon1)):
    x, y = transformer.transform(lon_local, lat_local)
    ax.plot(
            x, y, marker='o', markersize=2, color=colors1[idx], label=f'PICK:({lat_local:.4f}, {lon_local:.4f})')
for idx, (lat_local, lon_local) in enumerate(zip(lat2, lon2)):
    x, y = transformer.transform(lon_local, lat_local)
    ax.plot(
            x, y, marker='x', markersize=5, color=colors2[idx], label=f'REF({lat_local:.4f}, {lon_local:.4f})')
ax.legend(
        loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=1, fancybox=True, shadow=True, fontsize=12)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_title('Reanalyses grid', fontsize=24, pad=0)
ax.axis('off')
# plt.show()
plt.savefig(os.path.join(basefol, 'rean.png'), dpi=200, bbox_inches='tight')
