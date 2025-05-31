import os
import xarray as xr
import rasterio
import matplotlib.pyplot as plt
from pyproj import Transformer
import numpy as np
from scipy.spatial import cKDTree

basefolder = r"H:\Shared drives\Reanalysis"
geotiff_path = os.path.join(basefolder, "pituffik_big.tif")
dataset_carra_path = os.path.join(basefolder, "carra\\raw", "carra_2m_temperature_2023.nc")
dataset_era5_path = os.path.join(basefolder, "era5\\raw", "era5_2m_temperature_2023.nc")

def find_lat_lon(ds):
    lat_names = ['latitude', 'lat']
    lon_names = ['longitude', 'lon']

    lat = None
    lon = None

    for name in lat_names:
        if name in ds:
            lat = ds[name].values
            break

    for name in lon_names:
        if name in ds:
            lon = ds[name].values
            break

    if lat is None or lon is None:
        raise ValueError("Latitude or longitude fields not found in dataset.")
    return lat, lon

def extract_and_transform_grid(dataset_path, src_crs_epsg, transformer_to_target):
    ds = xr.open_dataset(dataset_path, decode_timedelta = True)
    lat, lon = find_lat_lon(ds)
    ds.close()
    del ds
    lat_flat = lat.flatten()
    lon_flat = lon.flatten()
    x, y = transformer_to_target.transform(lon_flat, lat_flat)
    return x, y

# Load GeoTIFF for CRS and image
with rasterio.open(geotiff_path) as src:
    geotiff_img = src.read(1)
    geotiff_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    geotiff_crs = src.crs

# Prepare transformers for dataset CRS -> geotiff CRS
transformer_era5_to_geotiff = Transformer.from_crs("EPSG:4326", geotiff_crs, always_xy=True)
transformer_carra_to_geotiff = Transformer.from_crs("EPSG:3413", geotiff_crs, always_xy=True)
transformer_points_to_geotiff = Transformer.from_crs("EPSG:4326", geotiff_crs, always_xy=True)

# Check and extract only if variables don't exist (avoid reloading)
if not ('x_era5' in globals() and 'y_era5' in globals()):
    print("Loading and processing ERA5 dataset...")
    x_era5, y_era5 = extract_and_transform_grid(dataset_era5_path, 4326, transformer_era5_to_geotiff)

if not ('x_carra' in globals() and 'y_carra' in globals()):
    print("Loading and processing CARRA dataset...")
    x_carra, y_carra = extract_and_transform_grid(dataset_carra_path, 3413, transformer_carra_to_geotiff)

# Input points lat/lon in EPSG:4326
lat1 = np.array([76.5149, 76.52, 76.5])
lon1 = np.array([-68.7477, -68.74, -68.8])

# Transform input points
x_points, y_points = transformer_points_to_geotiff.transform(lon1, lat1)

# Build KDTree and query nearest neighbors
tree_era5 = cKDTree(np.vstack([x_era5, y_era5]).T)
tree_carra = cKDTree(np.vstack([x_carra, y_carra]).T)

_, idx_era5 = tree_era5.query(np.vstack([x_points, y_points]).T)
_, idx_carra = tree_carra.query(np.vstack([x_points, y_points]).T)

# Plotting
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(geotiff_img, extent=geotiff_extent, origin='upper', cmap='gray')

ax.scatter(x_era5, y_era5, s=1, color='blue', alpha=0.5, label='ERA5 Grid')
ax.scatter(x_carra, y_carra, s=1, color='red', alpha=0.5, label='CARRA Grid')

colors = ['green', 'orange', 'purple']
for i, color in enumerate(colors):
    ax.scatter(x_era5[idx_era5[i]], y_era5[idx_era5[i]], s=50, marker='x', color=color, label=f'ERA5 Closest Point {i+1}')
    ax.scatter(x_carra[idx_carra[i]], y_carra[idx_carra[i]], s=50, marker='x', color=color, label=f'CARRA Closest Point {i+1}')
    ax.scatter(x_points[i], y_points[i], s=60, marker='o', facecolors='none', edgecolors=color, linewidths=2, label=f'Input Point {i+1}')

ax.set_title('CARRA and ERA5 Grids with Highlighted Closest Points')
ax.legend(loc='upper right', fontsize='small', markerscale=0.7)

# Save figure
output_path = os.path.join(basefolder, "grid_points_comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.show()
