import os
import glob

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


def replace_missing_with_nan(ds):
    """
    Replace all _FillValue or missing_value == -999 or -999.0 with np.nan in the dataset.
    """
    for var in ds.data_vars:
        fill_value = None
        if '_FillValue' in ds[var].attrs:
            fill_value = ds[var].attrs['_FillValue']
        elif 'missing_value' in ds[var].attrs:
            fill_value = ds[var].attrs['missing_value']

        if fill_value is not None and (fill_value == -999 or fill_value == -999.0):
            ds[var] = ds[var].where(ds[var] != fill_value, np.nan)
    return ds


def find_nearest_time_index(time_array, target_time):
    """
    Find index in time_array closest to target_time.
    time_array and target_time are pandas.Timestamp or np.datetime64.
    """
    deltas = np.abs(time_array - target_time)
    return deltas.argmin()


def extract_model_profile_at_point(ds_model, time_idx, lat_pt, lon_pt, pres_pt):
    """
    Extract temperature and relative humidity from model at nearest lat/lon/time and closest vertical level to pres_pt.
    Assumes ds_model has dimensions: time, level, lat, lon and variables: pressure, temperature, relative_humidity.
    """
    lat_idx = np.abs(ds_model['latitude'].values - lat_pt).argmin()
    lon_idx = np.abs(ds_model['longitude'].values - lon_pt).argmin()

    # Extract vertical profiles
    pres_profile = ds_model['pressure'].isel(
        time=time_idx, lat=lat_idx, lon=lon_idx).values
    temp_profile = ds_model['temperature'].isel(
        time=time_idx, lat=lat_idx, lon=lon_idx).values
    rh_profile = ds_model['relative_humidity'].isel(
        time=time_idx, lat=lat_idx, lon=lon_idx).values

    # Handle missing data in model profiles (e.g. nan)
    valid_mask = ~np.isnan(pres_profile)
    if not np.any(valid_mask):
        return np.nan, np.nan

    pres_profile = pres_profile[valid_mask]
    temp_profile = temp_profile[valid_mask]
    rh_profile = rh_profile[valid_mask]

    # Find vertical index closest to sonde pressure
    vert_idx = np.abs(pres_profile - pres_pt).argmin()

    return temp_profile[vert_idx], rh_profile[vert_idx]


# def radiosondes_netcdf(in_path, out_path):
#     output_file = os.path.join(out_path, 'arcsix_radiosondes_combined.nc')

#     if os.path.exists(output_file):
#         print(f'Combined NetCDF already exists at: {output_file}')
#         return output_file

#     # Find all txt files in input path
#     txt_files = sorted(glob.glob(os.path.join(in_path, '*.txt')))
#     print(f"Found {len(txt_files)} txt files.")

#     profiles = []

#     for file in txt_files:
#         with open(file, 'r') as f:
#             lines = f.readlines()

#         # Extract launch time
#         launch_time_line = next((l for l in lines if 'Date and Time' in l), None)
#         if launch_time_line:
#             launch_time_str = launch_time_line.split(':')[-1].strip()
#             launch_time = pd.to_datetime(launch_time_str, format='%Y%m%d_%H%M')
#         else:
#             print(f'Warning: Could not find launch time in file: {file}')
#             continue  # skip file if no launch time

#         # Find data start
#         data_start_idx = 0
#         for i, line in enumerate(lines):
#             if line.strip() and not line.startswith('#') and any(char.isdigit() for char in line):
#                 data_start_idx = i
#                 break

#         # Read numeric data
#         df = pd.read_csv(file,
#                           sep='\s+',
#                           names=['height', 'pressure', 'temperature',
#                                  'relative_humidity', 'wind_speed', 'wind_direction'],
#                           skiprows=data_start_idx)

#         # Drop duplicate pressure levels
#         df = df.drop_duplicates(subset='pressure')

#         # Sort by pressure (descending for typical vertical profile convention)
#         df = df.sort_values('pressure', ascending=False).reset_index(drop=True)

#         # Create dataset for this profile
#         profile_ds = xr.Dataset(
#             {
#                 'temperature': ('pressure', df['temperature'].values),
#                 'height': ('pressure', df['height'].values),
#                 'wind_direction': ('pressure', df['wind_direction'].values),
#                 'wind_speed': ('pressure', df['wind_speed'].values),
#                 'relative_humidity': ('pressure', df['relative_humidity'].values),
#             },
#             coords={
#                 'pressure': df['pressure'].values,
#                 'launch_time': [launch_time]  # wrap in list to make it a dimension
#             },
#             attrs={
#                 'source_file': os.path.basename(file),
#                 'station': 'BGTL Pituffik Space Base'
#             }
#         )

#         profiles.append(profile_ds)

#     # Concatenate along launch_time dimension
#     combined_ds = xr.concat(profiles, dim='launch_time', combine_attrs='drop')

#     # Save to NetCDF
#     combined_ds.to_netcdf(output_file)

#     print(f'Done! Combined NetCDF file saved at: {output_file}')
#     return output_file


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


def dropsondes_netcdf(in_path, out_path, output_filename='arcsix_dropsondes_combined.nc'):
    attributes_arcsix = {
        'Conventions': 'CF-1.8',
        'featureType': 'trajectoryProfile',
        'title': 'NASA ARCSIX Dropsonde Profiles',
        'summary': (
            'This dataset contains vertical atmospheric profiles from NASA ARCSIX dropsonde launches.'
            'Each profile includes measurements of temperature, relative humidity, wind speed, '
            'wind direction, and geopotential height, sorted by pressure levels.'
        ),
        'institution': 'INGV',
        'creator_name': 'Filippo Calì Quaglia',
        'creator_email': 'filippo.caliquaglia@ingv.it',
        'history': f'Created on {pd.Timestamp.now()} using dropsondes_netcdf() function.',
        'trajectory_name': 'ARCSIX Dropsondes',
        'geospatial_lat_min': 76.5,
        'geospatial_lat_max': 76.5,
        'geospatial_lon_min': -68.8,
        'geospatial_lon_max': -68.8,
        'geospatial_vertical_positive': 'up',
        'project': 'NASA ARCSIX Dropsonde Campaign 2024',
        'references': 'NASA ARCSIX Repository',
        'license': 'CC-BY-4.0',
        'comment': 'This file is CF-1.8 compliant and suitable for trajectory profile analysis.',
    }

    output_file = os.path.join(out_path, 'arcsix_dropsondes_combined.nc')

    if os.path.exists(output_file):
        print(f'NetCDF already exists at: {output_file}')
        return output_file

    nc_files = sorted(glob.glob(os.path.join(in_path, '*.nc')))
    print(f"Found {len(nc_files)} dropsonde files")

    profiles = []
    launch_times = []

    for i, nc_file in enumerate(nc_files[20:]):
        ds = xr.open_dataset(nc_file)
        date_time = ds['launch_time'].values

        # Drop variables starting with "reference"
        vars_to_drop = [var for var in ds.data_vars if var.startswith('reference')]
        ds = ds.drop_vars(vars_to_drop)

        # Add trajectory dimension instead of profile
        ds = ds.expand_dims({'trajectory': [i]})

        # Assign trajectory_id coordinate
        ds = ds.assign_coords(trajectory_id=('trajectory', [os.path.basename(nc_file)]))

        # Handle 'time' variable
        if 'time' in ds.variables:
            ds['time'].attrs['axis'] = 'T'
            # Remove 'units' from attrs if it exists
            ds['time'].attrs.pop('units', None)
            # Add 'units' in encoding instead (CF compliant)
            ds['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00 UTC'
            ds = ds.set_coords('time')

        profiles.append(ds)
        launch_times.append(pd.to_datetime(date_time))

    # Concatenate all profiles along 'trajectory' dimension
    combined_ds = xr.concat(profiles, dim='trajectory')

    # Assign launch_times coordinate
    combined_ds = combined_ds.assign_coords(trajectory_time=('trajectory', launch_times))

    # Update global attributes with CF compliant metadata
    combined_ds.attrs.update(attributes_arcsix)

    # Save to NetCDF
    out_file = os.path.join(out_path, output_filename)
    combined_ds.to_netcdf(out_file)
    print(f"Combined dropsonde dataset saved to {out_file}")

    return output_filename



def main():
    """ """

    fol_path_dropsondes = r"H:\\Shared drives\\Dati_THAAO\\thaao_arcsix\\dropsondes"
    fol_path_radiosondes = r"H:\\Shared drives\\Dati_THAAO\\thaao_rs_sondes\\txt\\2024"
    out_path = r"H:\\Shared drives\\Dati_THAAO\\thaao_arcsix"

    d_sonde_file = dropsondes_netcdf(fol_path_dropsondes, out_path)
    r_sonde_file = radiosondes_netcdf(fol_path_radiosondes, out_path)
    # r_sonde_file = radiosondes_netcdf_cf_compliant(fol_path_radiosondes, out_path)

    drop_sonde = xr.open_dataset(os.path.join(out_path, d_sonde_file))
    radio_sonde = xr.open_dataset(os.path.join(out_path, r_sonde_file))

    radio_sonde.info
    drop_sonde.info

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

    # dropsondes
    plt.figure(figsize=(8, 6))

    # Loop over all dropsonde trajectories and plot
    for i in range(drop_sonde.sizes['trajectory']):
        # Plot temperature vs pressure for this dropsonde profile
        plt.plot(drop_sonde['tdry'][i], drop_sonde['pres']
                 [i], label=f"Trajectory {i}",  lw=0, marker='.', markersize=1)

    plt.gca().invert_yaxis()
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Pressure (hPa)')
    plt.title('Temperature Profiles of All Dropsondes')

    # Optional: show legend (comment out if too many profiles!)
    # plt.legend()
    plt.ylim(1013, 0)
    plt.xlim(-60, 20)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors

    # Extract times as pandas datetime for colormap normalization
    radio_times = pd.Series(pd.to_datetime(radio_sonde['time'].values[0]).ravel())
    drop_times = pd.Series(pd.to_datetime(drop_sonde['launch_time'].values).ravel())
    
    # Combine all times to get global min/max for normalization
    all_times = pd.concat([radio_times, drop_times], ignore_index=True)
    min_time = np.min(all_times.values)
    max_time = np.max(all_times.values)
    
    # Normalize times to [0,1] for colormap
    norm = mcolors.Normalize(mdates.date2num(min_time), mdates.date2num(max_time))
    
    # Create colormaps
    cmap_cold = plt.cm.Blues
    cmap_warm = plt.cm.Oranges
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot radiosonde with cold colors scaled by date
    for i in range(radio_sonde.sizes['profile']):
        dt_num = mdates.date2num(radio_times[i])
        color = cmap_cold(norm(dt_num))
        temp_i = radio_sonde['temperature'][0].isel(profile=i)
        pres_i = radio_sonde['pressure'].isel(profile=i)
        ax.plot(temp_i, pres_i, lw=0, marker='.', markersize=1, color=color)
    
    # Plot dropsonde with warm colors scaled by date, shifted +20°C
    for i in range(drop_sonde.sizes['trajectory']):
        dt_num = mdates.date2num(drop_times[i])
        color = cmap_warm(norm(dt_num))
        temp_i = drop_sonde['tdry'].isel(trajectory=i).values  # shape: (level,)
        pres_i = drop_sonde['pres'].isel(trajectory=i).values  # shape: (level,)
    
        shifted_temp = temp_i + 20
        ax.plot(shifted_temp, pres_i,
                lw=0, marker='.', markersize=1, color=color)
    
    ax.set_ylim(1013, 0)
    ax.set_xlim(-60, 40)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title('Temperature Profiles Colored by Launch Date\nRadiosondes (Blues), Dropsondes (Oranges, shifted +20°C)')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    
    plt.tight_layout()
    plt.show()

    print()
    # # --- Load 3D model dataset ---
    # model_file = "model_3d_pressure_3h.nc"  # Change to your actual model file path
    # ds_model = xr.open_dataset(model_file)
    # ds_model = replace_missing_with_nan(ds_model)

    # # Convert model times to pandas datetime
    # model_times = pd.to_datetime(ds_model['time'].values)

    # # --- Match dropsonde times to model times ---
    # nearest_time_indices = [find_nearest_time_index(
    #     model_times, t) for t in sonde_times]

    # # --- Extract model temperature and RH at dropsonde positions and pressures ---
    # model_temps = []
    # model_rhs = []

    # for i, time_idx in enumerate(nearest_time_indices):
    #     if np.isnan(sonde_pres[i]) or np.isnan(sonde_lat[i]) or np.isnan(sonde_lon[i]):
    #         model_temps.append(np.nan)
    #         model_rhs.append(np.nan)
    #         continue
    #     temp_val, rh_val = extract_model_profile_at_point(
    #         ds_model, time_idx, sonde_lat[i], sonde_lon[i], sonde_pres[i])
    #     model_temps.append(temp_val)
    #     model_rhs.append(rh_val)

    # model_temps = np.array(model_temps)
    # model_rhs = np.array(model_rhs)

    # # --- Plot dropsonde temperature vs model temperature ---
    # plt.figure(figsize=(6, 8))
    # plt.plot(sonde_tdry, sonde_pres, label='Dropsonde Tdry',
    #          marker='o', linestyle='-')
    # plt.plot(model_temps, sonde_pres, label='Model Temperature',
    #          marker='x', linestyle='--')
    # plt.gca().invert_yaxis()
    # plt.xlabel("Temperature (°C)")
    # plt.ylabel("Pressure (hPa)")
    # plt.title("Dropsonde vs Model Temperature Profile")
    # plt.legend()
    # plt.grid()
    # plt.show()


if __name__ == "__main__":
    main()
