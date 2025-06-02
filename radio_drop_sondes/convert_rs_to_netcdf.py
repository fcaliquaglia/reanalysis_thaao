import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
import re
import io
import glob
import os

# Folder containing the radiosonde .txt files
folder_path = r"H:\Shared drives\Dati_THAAO\thaao_rs_sondes\txt\2024"

# List all .txt files in folder, sorted
txt_files = sorted(glob.glob(os.path.join(folder_path, '*.txt')))

for txt_file in txt_files:
    print(f"Processing {txt_file} ...")

    # --- 1. Load file ---
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # --- 2. Parse metadata ---
    metadata = {}
    data_start_index = None

    for idx, line in enumerate(lines):
        if line.startswith("# Station:"):
            metadata['station'] = 'THAAO - Thule High Arctic Atmospheric Observatory'
        elif line.startswith("# Date and Time"):
            match = re.search(r"(\d{8}_\d{4})", line)
            if match:
                metadata['launch_time'] = match.group(1)
        elif line.startswith("# Max elevation"):
            metadata['max_elevation'] = float(line.split(":")[1].strip())
        elif line.startswith("# Height,Pressure"):
            data_start_index = idx + 2
            break

    if data_start_index is None:
        print(f"Warning: Data start not found in {txt_file}, skipping...")
        continue

    # --- 3. Parse measurement data ---
    data_lines = lines[data_start_index:]
    data_str = ''.join(data_lines)
    df = pd.read_csv(io.StringIO(data_str),
                     sep='/s+',
                     names=['height', 'pressure', 'temperature', 'rh', 'wind_speed', 'wind_dir'])

    # --- 4. Calculate time ---
    launch_time = datetime.strptime(metadata['launch_time'], "%Y%m%d_%H%M")
    time_seconds = np.arange(len(df))
    time = np.array([launch_time + pd.Timedelta(seconds=int(t)) for t in time_seconds])

    # --- 5. Initial coordinates ---
    latitude0 = 76.5312
    longitude0 = -68.7031

    # --- 6. Calculate horizontal wind components (m/s) ---
    wd_rad = np.deg2rad(df['wind_dir'])
    u = -df['wind_speed'] * np.sin(wd_rad)  # eastward component
    v = -df['wind_speed'] * np.cos(wd_rad)  # northward component

    # --- 7. Calculate cumulative displacement (in meters) ---
    dx = np.cumsum(u)
    dy = np.cumsum(v)

    # --- 8. Convert displacements to lat/lon offsets ---
    dlat = dy / 111320
    dlon = dx / (111320 * np.cos(np.deg2rad(latitude0)))

    latitudes = latitude0 + dlat
    longitudes = longitude0 + dlon

    # --- 9. Create xarray Dataset ---
    ds = xr.Dataset(
        {
            'air_pressure': (['trajectory', 'obs'], [df['pressure'].values]),
            'air_temperature': (['trajectory', 'obs'], [df['temperature'].values + 273.15]),
            'relative_humidity': (['trajectory', 'obs'], [df['rh'].values]),
            'wind_speed': (['trajectory', 'obs'], [df['wind_speed'].values]),
            'wind_from_direction': (['trajectory', 'obs'], [df['wind_dir'].values]),
            'altitude': (['trajectory', 'obs'], [df['height'].values]),
        },
        coords={
            'time': (['trajectory', 'obs'], [np.array(time, dtype='datetime64[ns]')]),
            'latitude': (['trajectory', 'obs'], [latitudes]),
            'longitude': (['trajectory', 'obs'], [longitudes]),
            'trajectory': [0],
            'obs': np.arange(len(df)),
        },
        attrs={
            'title': f"Radiosonde Profile from {metadata['station']}",
            'institution': 'Processed at INGV, ENEA.',
            'source': 'Radiosonde observation with calculated position',
            'history': 'Converted to CF-compliant trajectory NetCDF by script',
            'references': 'giovanni.muscari@ingv.it',
            'launch_time': metadata['launch_time'],
            'max_elevation': metadata['max_elevation']
        }
    )

    # --- 10. CF Conventions ---
    ds['trajectory'].attrs['long_name'] = 'trajectory identifier'
    ds['obs'].attrs['long_name'] = 'observation index'
    ds['time'].attrs['standard_name'] = 'time'
    ds['latitude'].attrs['standard_name'] = 'latitude'
    ds['longitude'].attrs['standard_name'] = 'longitude'
    ds['altitude'].attrs.update({
        'standard_name': 'altitude',
        'units': 'm',
        'positive': 'up'
    })
    ds['air_pressure'].attrs.update({
        'standard_name': 'air_pressure',
        'units': 'hPa'
    })
    ds['air_temperature'].attrs.update({
        'standard_name': 'air_temperature',
        'units': 'K'
    })
    ds['relative_humidity'].attrs.update({
        'standard_name': 'relative_humidity',
        'units': '%'
    })
    ds['wind_speed'].attrs.update({
        'standard_name': 'wind_speed',
        'units': 'm s-1'
    })
    ds['wind_from_direction'].attrs.update({
        'standard_name': 'wind_from_direction',
        'units': 'degree'
    })

    # --- 11. Save NetCDF ---
    out_filename = os.path.splitext(os.path.basename(txt_file))[0] + '.nc'
    out_path = os.path.join(folder_path, out_filename)
    ds.to_netcdf(out_path, format='NETCDF4', engine='netcdf4')

    print(f"✅ Saved {out_path}")

print("✅ All files processed!")
