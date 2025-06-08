# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 09:30:17 2025

@author: FCQ
"""

import os
import glob

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


def radiosondes_netcdf(in_path, out_path):

    output_file = os.path.join(out_path, 'arcsix_radiosondes_combined.nc')

    if os.path.exists(output_file):
        print(f'NetCDF already exists at: {output_file}')
        return output_file

    # Find all txt files
    txt_files = sorted(glob.glob(os.path.join(in_path, '*.txt')))
    print(f"Found {len(txt_files)} txt files.")

    profiles_data = []
    launch_times = []

    for file in txt_files:
        with open(file, 'r') as f:
            lines = f.readlines()

        # Extract launch time
        launch_time_line = next(
            (l for l in lines if 'Date and Time' in l), None)
        if not launch_time_line:
            print(f'Warning: Could not find launch time in file: {file}')
            continue

        launch_time_str = launch_time_line.split(':')[-1].strip()
        try:
            launch_time = pd.to_datetime(launch_time_str, format='%Y%m%d_%H%M')
        except Exception as e:
            print(f'Error parsing launch time in file {file}: {e}')
            continue

        # Find data start
        data_start_idx = next(i for i, line in enumerate(lines) if line.strip(
        ) and not line.startswith('#') and any(char.isdigit() for char in line))

        # Read data
        df = pd.read_csv(
            file,
            sep='\s+',
            names=['height', 'pressure', 'temperature',
                   'relative_humidity', 'wind_speed', 'wind_direction'],
            skiprows=data_start_idx,
            comment='#',
            engine='python'
        ).drop_duplicates(subset='pressure').sort_values('pressure', ascending=False).reset_index(drop=True)

        profiles_data.append(df)
        launch_times.append(launch_time)

    if not profiles_data:
        raise ValueError("No valid radiosonde profiles found or parsed.")

    # Determine the maximum number of levels
    max_levels = max(len(df) for df in profiles_data)
    n_profiles = len(profiles_data)

    # Initialize 2D arrays (profile x level) with NaNs
    def init_array():
        return np.full((n_profiles, max_levels), np.nan, dtype=np.float32)

    data_arrays = {
        'temperature': init_array(),
        'relative_humidity': init_array(),
        'wind_speed': init_array(),
        'wind_direction': init_array(),
        'height': init_array(),
        'pressure': init_array(),
    }

    # Fill the arrays with profile data
    for i, df in enumerate(profiles_data):
        n_levels = len(df)
        data_arrays['temperature'][i, :n_levels] = df['temperature']
        data_arrays['relative_humidity'][i,
                                         :n_levels] = df['relative_humidity']
        data_arrays['wind_speed'][i, :n_levels] = df['wind_speed']
        data_arrays['wind_direction'][i, :n_levels] = df['wind_direction']
        data_arrays['height'][i, :n_levels] = df['height']
        data_arrays['pressure'][i, :n_levels] = df['pressure']

    # Create Dataset
    ds = xr.Dataset(
        data_vars={
            'temperature': (['profile', 'level'], data_arrays['temperature']),
            'relative_humidity': (['profile', 'level'], data_arrays['relative_humidity']),
            'wind_speed': (['profile', 'level'], data_arrays['wind_speed']),
            'wind_direction': (['profile', 'level'], data_arrays['wind_direction']),
            'height': (['profile', 'level'], data_arrays['height']),
            'pressure': (['profile', 'level'], data_arrays['pressure']),
        },
        coords={
            'profile': np.arange(n_profiles),
            'level': np.arange(max_levels),
            'time': ('profile', pd.to_datetime(launch_times)),
            'latitude': 76.5,
            'longitude': -68.8,
        },
        attrs={
            'title': 'ARCSIX Radiosonde Profiles: THAAO in Pituffik (BGTL)',
            'summary': 'Vertical profiles from ARCSIX radiosonde launches (temperature, RH, wind, etc.).',
            'institution': 'INGV',
            'creator_name': 'Filippo Calì Quaglia',
            'creator_email': 'filippo.caliquaglia@ingv.it',
            'history': f'Created on {pd.Timestamp.now()} using radiosondes_netcdf_profiles()',
            'geospatial_lat': 76.5,
            'geospatial_lon': -68.8,
            'geospatial_vertical_positive': 'up',
            'references': 'https://www.thuleatmos-it.it',
            'comment': 'Each profile stored as a vertical profile along "level" coordinate, identified by "profile" and "time".',
        }
    )

    # Save to NetCDF
    ds.to_netcdf(output_file)
    print(f'Done! Profiles dataset saved at: {output_file}')
    return output_file



def main():
    """ """

    fol_path_radiosondes = r"H:\\Shared drives\\Dati_THAAO\\thaao_rs_sondes\\txt\\2024"
    out_path = r"H:\\Shared drives\\Dati_THAAO\\thaao_arcsix"

    r_sonde_file = radiosondes_netcdf(fol_path_radiosondes, out_path)
    # r_sonde_file = radiosondes_netcdf_cf_compliant(fol_path_radiosondes, out_path)

    radio_sonde = xr.open_dataset(os.path.join(out_path, r_sonde_file))

    radio_sonde.info

    # radiosonde
    
    temperature = radio_sonde['temperature'].isel(trajectory=0)
    pressure = radio_sonde['pressure']
    
    plt.figure(figsize=(8,6))
    for i in range(temperature.sizes['profile']):
        plt.plot(temperature[i], pressure[i], lw=0, marker='.', markersize=1)
    
    plt.gca().invert_yaxis()
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Pressure (hPa)')
    plt.title('Temperature Profiles of All Radiosondes')
    plt.ylim(1013, 0)
    plt.xlim(-60, 20)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
