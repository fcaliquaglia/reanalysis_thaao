import xarray as xr
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

folder_path = r"H:\Shared drives\Dati_THAAO\thaao_arcsix\buoys\resource_map_doi_10_18739_A2T14TR46\data"
nc_files = sorted(glob.glob(os.path.join(folder_path, "2024*processed.nc")))

# Filter by filename character (e.g. 'J' to 'R')
nc_files_filtered = []
for f in nc_files:
    base = os.path.basename(f)
    if len(base) > 5:
        letter = base[4]
        if 'J' <= letter <= 'R':
            nc_files_filtered.append(f)

# --- Plot setup ---
fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([-75, -10, 65, 85], crs=ccrs.PlateCarree())

# Map features
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='azure')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False

# --- Plot each buoy ---
for nc_file in nc_files_filtered:
    ds = xr.open_dataset(nc_file)

    # Extract first trajectory
    lat = ds['latitude'].isel(trajectory=0).values
    lon = ds['longitude'].isel(trajectory=0).values
    temp = ds['air_temp'].isel(trajectory=0).values

    # Mask invalid values
    valid_mask = (
        ~np.isnan(lat) & ~np.isnan(lon) & ~np.isnan(temp) &
        (lat >= 59) & (lat <= 85) & (lon >= -75) & (lon <= -10)
    )
    if np.count_nonzero(valid_mask) == 0:
        continue

    lat_filtered = lat[valid_mask]
    lon_filtered = lon[valid_mask]
    temp_filtered = temp[valid_mask]

    # Plot color-coded trajectory
    sc = ax.scatter(lon_filtered, lat_filtered, c=temp_filtered,
                    cmap='coolwarm', s=10, transform=ccrs.PlateCarree(),
                    label=os.path.basename(nc_file).split('.')[0])

    # Mark final point with a label
    letter = ''.join(filter(str.isalpha, os.path.basename(nc_file)))[0]
    ax.text(lon_filtered[-1], lat_filtered[-1], letter, fontsize=10, fontweight='bold',
            transform=ccrs.PlateCarree(),
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

# Colorbar
cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, label='Surface Temperature (°C)')

# Title and show
plt.title('Buoy Trajectories (Color Coded by Surface Temperature) – 2024')
plt.savefig("buoy_trajectories_surface_temp_2024.png", dpi=300, bbox_inches='tight')
plt.show()
