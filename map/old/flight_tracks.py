import re
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from pathlib import Path

def read_ict_file(filepath):
    """Reads .ict file and returns dataframe from second 'Time_Start' header."""
    with open(filepath, 'r') as file:
        lines = file.readlines()

    count = 0
    header_line = None
    for i, line in enumerate(lines):
        if line.startswith("Time_Start"):
            count += 1
            if count == 2:
                header_line = i
                break
    if header_line is None:
        raise ValueError(f"Second 'Time_Start' header not found in file: {filepath}")

    df = pd.read_csv(filepath, skiprows=header_line, index_col='Time_Start')
    return df

def filter_files(folder_path):
    """Return sorted list of files with either no Lx or L2 in filename."""
    files = sorted(glob.glob(os.path.join(folder_path, '*R0*.ict')))
    filtered = []
    for f in files:
        base = os.path.basename(f)
        m = re.search(r'_L(\d)\.ict$', base)
        if not m or m.group(1) == '2':
            filtered.append(f)
    return filtered

# Folder paths
folder_path_g3 = r"H:\Shared drives\Dati_THAAO\thaao_arcsix\met_nav\G3"
folder_path_p3 = r"H:\Shared drives\Dati_THAAO\thaao_arcsix\met_nav\P3"

# Filter files
p3_files = filter_files(folder_path_p3)
g3_files = filter_files(folder_path_g3)

# Plot setup
proj = ccrs.NorthPolarStereo()
fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'projection': proj})

freq = 100
map_scale_res = '50m'

# Ocean (bottom)
ocean = NaturalEarthFeature('physical', 'ocean', scale=map_scale_res,
                           edgecolor='face', facecolor='#a6cee3')
ax.add_feature(ocean, zorder=0)

# Land (top)
land = NaturalEarthFeature('physical', 'land', scale=map_scale_res,
                          edgecolor='black', facecolor='#f0e6d2')
ax.add_feature(land, zorder=1)

ax.coastlines(resolution=map_scale_res, linewidth=0.8, color='black')
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_extent([-105, -10, 60, 84], crs=ccrs.PlateCarree())

def plot_trajectories(files, label, color, linestyle, ax, freq=100):
    first = True
    for file in files:
        try:
            df = read_ict_file(file)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        # Extract substring for filename date/time info — tweak as needed
        basename = os.path.basename(file)
        # Example extraction: characters 28-36 or fallback
        date_str = basename[28:36] if len(basename) > 36 else basename[18:25]

        print(f"Plotting {label} Flight track for {date_str}")
        ax.plot(
            df["Longitude"][::freq], df["Latitude"][::freq],
            label=label if first else None,
            linestyle=linestyle,
            transform=ccrs.PlateCarree(),
            linewidth=0.7,
            color=color, alpha=0.6
        )
        first = False

# Plot P-3 and G-3
plot_trajectories(p3_files, 'P-3', 'orange', '-', ax, freq)
plot_trajectories(g3_files, 'G-3', 'purple', '--', ax, freq)

# Mark THAAO site
ax.plot(-68.7477, 76.5149, marker='X', markersize=12, color='red', lw=0,
        transform=ccrs.PlateCarree(), label='THAAO')
ax.plot(-16.6667, 81.6, marker='X', markersize=12, color='cyan', lw=0,
        transform=ccrs.PlateCarree(), label='Villum')
ax.plot(-62.5072, 82.4508, marker='X', markersize=12, color='green', lw=0,
        transform=ccrs.PlateCarree(), label='Alert')

ax.set_title("Aircraft Trajectories in North Polar Projection", fontsize=14)
ax.legend(loc='lower left', fontsize=10)

plt.title('NASA ARCSIX Flight Tracks (2024)')
plt.savefig("arcsix_flight_tracks.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
