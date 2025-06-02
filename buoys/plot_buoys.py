import xarray as xr
import matplotlib.pyplot as plt
import glob
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature

folder_path = r"H:\Shared drives\Dati_THAAO\thaao_arcsix\buoys\resource_map_doi_10_18739_A2T14TR46\data"
nc_files = sorted(glob.glob(os.path.join(folder_path, "2024*processed.nc")))


nc_files_filtered = []
for f in nc_files:
    base = os.path.basename(f)

    if len(base) > 5:  # just to be safe
        letter = base[4]
        if 'J' <= letter <= 'R':
            nc_files_filtered.append(f)

fig = plt.figure(figsize=(12, 10))

# Use North Polar Stereographic projection centered roughly on Greenland
ax = plt.axes(projection=ccrs.NorthPolarStereo())

# Set extent to Greenland region [min_lon, max_lon, min_lat, max_lat]
ax.set_extent([-75, -10, 65, 85], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='azure')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False

for nc_file in nc_files_filtered:
    ds = xr.open_dataset(nc_file)
    lat = ds['latitude'].isel(trajectory=0).values
    lon = ds['longitude'].isel(trajectory=0).values
    label = os.path.splitext(os.path.basename(nc_file))[0]

    # Filter points inside Greenland bounding box
    mask = (lat >= 59) & (lat <= 85) & (lon >= -75) & (lon <= -10)
    lat_filtered = lat[mask]
    lon_filtered = lon[mask]

    if len(lat_filtered) > 0:
        ax.plot(lon_filtered, lat_filtered, marker='.',
                label=label, transform=ccrs.PlateCarree())

        letter = ''.join(filter(str.isalpha, label))[0]
        ax.text(lon_filtered[-1], lat_filtered[-1], letter, fontsize=12, fontweight='bold',
                transform=ccrs.PlateCarree(),
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

plt.title('Buoy Trajectories 2024 - Greenland Region')
plt.show()
