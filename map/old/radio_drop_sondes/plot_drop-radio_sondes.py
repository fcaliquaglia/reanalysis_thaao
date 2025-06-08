import pandas as pd
import numpy as np 
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import xarray as xr
import cartopy.crs as ccrs
from matplotlib import gridspec
import cartopy.feature as cfeature

# Paths to folders
fol_path_dropsondes = r"H:\\Shared drives\\Dati_THAAO\\thaao_arcsix\\dropsondes"
folder_path_radiosondes = r"H:\Shared drives\Dati_THAAO\thaao_rs_sondes\txt\2024"

# Get sorted list of .nc files
radio_files = sorted(glob.glob(os.path.join(folder_path_radiosondes, '*.nc')))
drop_files = sorted(glob.glob(os.path.join(fol_path_dropsondes, 'ARCSIX-AVAPS-netCDF_G3*.nc')))

# Initialize lists
radio_times_all, radio_temp_all, radio_pres_all = [], [], []
drop_times_all, drop_temp_all, drop_pres_all = [], [], []
drop_lats_all, drop_lons_all, drop_surface_temp = [], [], []

# Process radiosondes
for rf in radio_files:
    radio_sonde = xr.open_dataset(rf)
    launch_time_str = radio_sonde.attrs.get('launch_time')
    launch_time = pd.to_datetime(launch_time_str, format='%Y%m%d_%H%M')
    temp_profile = radio_sonde['air_temperature'][0, :].values - 273.15
    pres_profile = radio_sonde['air_pressure'][0, :].values
    radio_times_all.append(launch_time)
    radio_temp_all.append(temp_profile)
    radio_pres_all.append(pres_profile)

# Process dropsondes
for ds_file in drop_files:
    ds = xr.open_dataset(ds_file)
    launch_time = pd.to_datetime(ds['launch_time'].values)
    pres = np.where(ds['pres'].values == -999.0, np.nan, ds['pres'].values)
    temp = np.where(ds['tdry'].values == -999.0, np.nan, ds['tdry'].values)
    lat = ds['lat'].values
    lon = ds['lon'].values
    drop_times_all.append(launch_time)
    drop_temp_all.append(temp)
    drop_pres_all.append(pres)
    drop_lats_all.append(lat)
    drop_lons_all.append(lon)
    # Surface temp: find highest pressure (i.e. lowest altitude)
    valid = ~np.isnan(temp) & ~np.isnan(pres)
    if valid.any():
        idx_surface = np.nanargmax(pres)
        drop_surface_temp.append(temp[idx_surface])
    else:
        drop_surface_temp.append(np.nan)

# --- Radiosonde and Dropsonde Profiles ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
all_times = pd.Series(radio_times_all + drop_times_all)
min_time = all_times.min()
max_time = all_times.max()
norm = mcolors.Normalize(mdates.date2num(min_time), mdates.date2num(max_time))
cmap_radio = plt.cm.jet
cmap_drop = plt.cm.jet

for i, (temp, pres) in enumerate(zip(radio_temp_all, radio_pres_all)):
    c = cmap_radio(norm(mdates.date2num(radio_times_all[i])))
    ax1.plot(temp, pres, marker='.', markersize=1, alpha=0.2, color=c, lw=0.2)
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Pressure (hPa)')
ax1.set_title('Radiosonde Profiles')

for i, (temp, pres) in enumerate(zip(drop_temp_all, drop_pres_all)):
    c = cmap_drop(norm(mdates.date2num(drop_times_all[i])))
    ax2.plot(temp, pres, marker='.', markersize=1, alpha=0.2, color=c, lw=0.2)
ax2.set_xlabel('Temperature (°C)')
ax2.set_title('Dropsonde Profiles')

cbar_ax1 = fig.add_axes([0.13, 0.1, 0.35, 0.03])
sm_radio = plt.cm.ScalarMappable(cmap=cmap_radio, norm=norm)
sm_radio.set_array([])
cbar1 = fig.colorbar(sm_radio, cax=cbar_ax1, orientation='horizontal')
cbar1.ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
cbar1.set_label('Radiosonde Launch Date')

cbar_ax2 = fig.add_axes([0.57, 0.1, 0.35, 0.03])
sm_drop = plt.cm.ScalarMappable(cmap=cmap_drop, norm=norm)
sm_drop.set_array([])
cbar2 = fig.colorbar(sm_drop, cax=cbar_ax2, orientation='horizontal')
cbar2.ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
cbar2.set_label('Dropsonde Launch Date')

all_pressures = np.concatenate([np.concatenate(radio_pres_all), np.concatenate(drop_pres_all)])
all_pressures = all_pressures[~np.isnan(all_pressures)]
ax1.set_ylim(all_pressures.max(), all_pressures.min()*0.20)
ax2.set_ylim(all_pressures.max(), all_pressures.min()*0.20)
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("sonde_profiles.png", dpi=300)
plt.show()
plt.close()


# --- Save Dropsonde Trajectories Map ---
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_title('Dropsonde Trajectories (Northern Greenland)')

# Add map features with ocean light blue and land light gray
ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='lightblue') 
ax.add_feature(cfeature.LAND, zorder=1, facecolor='lightgray') 
ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=2)

ax.gridlines(draw_labels=False, linestyle=':')

# Counter for valid trajectories
traj_count = 0

# --- Plot trajectories ---
for lat, lon in zip(drop_lats_all, drop_lons_all):
    lat = np.array(lat)
    lon = np.array(lon)

    # Basic validation
    if len(lat) < 2 or len(lon) < 2 or len(lat) != len(lon):
        continue

    # Mask invalid entries
    mask = ~np.isnan(lat) & ~np.isnan(lon)
    if np.count_nonzero(mask) < 2:
        continue

    traj_count += 1

    # Plot valid trajectory with a dark contrasting color
    ax.plot(lon[mask], lat[mask], color='darkred', alpha=0.8, transform=ccrs.PlateCarree(), lw=1.5)

    # Find lowermost (minimum latitude) valid point and plot a marker there
    lowermost_idx = np.nanargmin(lat[mask])
    low_lat = lat[mask][lowermost_idx]
    low_lon = lon[mask][lowermost_idx]
    ax.plot(low_lon, low_lat, 'o', color='black', markersize=6, markeredgewidth=1.2, markeredgecolor='yellow',
            transform=ccrs.PlateCarree())

# Add trajectory count text in upper left corner
ax.text(
    0.02, 0.98, f'N={traj_count}',
    transform=ax.transAxes,
    verticalalignment='top',
    horizontalalignment='left',
    fontsize=12,
    fontweight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
)

# --- Save figure ---
plt.savefig("dropsonde_trajectories.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()


surface_temps = []
surface_lats = []
surface_lons = []
marker_edges = []

traj_count=0
for lat, lon, temp in zip(drop_lats_all, drop_lons_all, drop_surface_temp):
    lat = np.array(lat)
    lon = np.array(lon)

    # Skip invalid temperature
    if np.isnan(temp):
        continue

    # Find indices where lat/lon are valid
    valid_indices = np.where(~np.isnan(lat) & ~np.isnan(lon))[0]
    if len(valid_indices) == 0:
        continue  # skip if all NaNs

    last_valid_idx = valid_indices[-1]
    used_fallback = (last_valid_idx != len(lat) - 1)

    surface_temps.append(temp)
    surface_lats.append(lat[last_valid_idx])
    surface_lons.append(lon[last_valid_idx])
    marker_edges.append('k' if used_fallback else 'none')
    traj_count += 1


# --- Plotting Surface Temperatures ---
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_title('Surface Temperature from Dropsondes (°C)')

# Add map features with ocean light blue and land light gray
ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='lightblue') 
ax.add_feature(cfeature.LAND, zorder=1, facecolor='lightgray') 
ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=2)

ax.gridlines(draw_labels=False, linestyle=':')

# Plot all points at once for proper color mapping
scatter = ax.scatter(
    surface_lons, surface_lats, c=surface_temps, cmap='coolwarm', s=40,
    edgecolor=marker_edges, linewidth=0.8, transform=ccrs.PlateCarree()
)

# After plotting, get the bounding box of the main map axes in figure coordinates
bbox = ax.get_position()

# Create colorbar axis with same height as map axes
cbar_ax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.02, bbox.height])

norm = plt.Normalize(vmin=np.nanmin(surface_temps), vmax=np.nanmax(surface_temps))
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array(surface_temps)

cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
cbar.set_label('Surface Temp (°C)')

# Example: Add trajectory count text in upper left corner
traj_count = len(surface_temps)  # or your actual trajectory count
ax.text(
    0.02, 0.98, f'N={traj_count}',
    transform=ax.transAxes,
    verticalalignment='top',
    horizontalalignment='left',
    fontsize=12,
    fontweight='bold',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
)

plt.savefig("dropsonde_surface_temp.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# 3D plot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# First create a 2D map as an image (PNG or array)
fig_map = plt.figure(figsize=(8, 8))
ax_map = plt.axes(projection=ccrs.NorthPolarStereo())
ax_map.set_extent([-90, 0, 60, 90], crs=ccrs.PlateCarree())
ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.7)
plt.axis('off')

# Save map as image
fig_map.canvas.draw()
map_img = np.array(fig_map.canvas.renderer.buffer_rgba())
plt.close(fig_map)

# Prepare 3D figure
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Dropsonde Profiles: Lat, Lon, Pressure (colored by Temperature)')

# Plot map as an image on the XY plane at max pressure (lowest altitude)
# Define bounding box matching your map extent
lon_min, lon_max = -90, 0
lat_min, lat_max = 60, 90
pressure_plane = np.nanmax(np.concatenate(drop_pres_all)) + 10  # a bit below max pressure for visibility

# Coordinates for image corners
x_img = np.linspace(lon_min, lon_max, map_img.shape[1])
y_img = np.linspace(lat_min, lat_max, map_img.shape[0])
X_img, Y_img = np.meshgrid(x_img, y_img)

# Plot the image on XY plane (Z = pressure_plane)
ax.plot_surface(
    X_img, Y_img, pressure_plane * np.ones_like(X_img),
    rstride=1, cstride=1, facecolors=map_img / 255,
    shade=False
)

# Normalize for temperature colormap
all_temps = np.concatenate([np.array(t) for t in drop_temp_all if len(t) > 0])
norm = plt.Normalize(vmin=np.nanmin(all_temps), vmax=np.nanmax(all_temps))
cmap = plt.cm.coolwarm

for lat, lon, pres, temp in zip(drop_lats_all, drop_lons_all, drop_pres_all, drop_temp_all):
    lat = np.array(lat)
    lon = np.array(lon)
    pres = np.array(pres)
    temp = np.array(temp)

    mask = ~np.isnan(lat) & ~np.isnan(lon) & ~np.isnan(pres) & ~np.isnan(temp)
    if np.count_nonzero(mask) < 2:
        continue

    # Plot trajectory lines
    ax.plot(lon[mask], lat[mask], pres[mask], color='gray', alpha=0.3, lw=0.8)

    # Scatter points colored by temperature
    sc = ax.scatter(lon[mask], lat[mask], pres[mask], c=temp[mask], cmap=cmap, norm=norm, s=20, edgecolor='none')

# Invert pressure axis
ax.set_zlim(np.nanmax(pres), np.nanmin(pres))

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Pressure (hPa)')

# Colorbar
cbar = fig.colorbar(sc, ax=ax, pad=0.1, fraction=0.03)
cbar.set_label('Temperature (°C)')

plt.show()
