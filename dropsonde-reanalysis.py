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
    pres_profile = ds_model['pressure'].isel(time=time_idx, lat=lat_idx, lon=lon_idx).values
    temp_profile = ds_model['temperature'].isel(time=time_idx, lat=lat_idx, lon=lon_idx).values
    rh_profile = ds_model['relative_humidity'].isel(time=time_idx, lat=lat_idx, lon=lon_idx).values

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

def main():

    # --- Load dropsonde dataset ---
    sonde_file = "ARCSIX-AVAPS-netCDF_G3_20240603143441_R0.nc"
    ds_sonde = xr.open_dataset(sonde_file)
    ds_sonde = replace_missing_with_nan(ds_sonde)

    # Convert dropsonde time to datetime (seconds since launch time)
    launch_time_str = ds_sonde['launch_time'].attrs['units'].split('since')[1].strip()
    launch_time = pd.to_datetime(launch_time_str)
    sonde_times = launch_time + pd.to_timedelta(ds_sonde['time'].values, unit='s')

    sonde_pres = ds_sonde['pres'].values
    sonde_tdry = ds_sonde['tdry'].values
    sonde_rh = ds_sonde['rh'].values
    sonde_lat = ds_sonde['lat'].values
    sonde_lon = ds_sonde['lon'].values

    # --- Load 3D model dataset ---
    model_file = "model_3d_pressure_3h.nc"  # Change to your actual model file path
    ds_model = xr.open_dataset(model_file)
    ds_model = replace_missing_with_nan(ds_model)

    # Convert model times to pandas datetime
    model_times = pd.to_datetime(ds_model['time'].values)

    # --- Match dropsonde times to model times ---
    nearest_time_indices = [find_nearest_time_index(model_times, t) for t in sonde_times]

    # --- Extract model temperature and RH at dropsonde positions and pressures ---
    model_temps = []
    model_rhs = []

    for i, time_idx in enumerate(nearest_time_indices):
        if np.isnan(sonde_pres[i]) or np.isnan(sonde_lat[i]) or np.isnan(sonde_lon[i]):
            model_temps.append(np.nan)
            model_rhs.append(np.nan)
            continue
        temp_val, rh_val = extract_model_profile_at_point(ds_model, time_idx, sonde_lat[i], sonde_lon[i], sonde_pres[i])
        model_temps.append(temp_val)
        model_rhs.append(rh_val)

    model_temps = np.array(model_temps)
    model_rhs = np.array(model_rhs)

    # --- Plot dropsonde temperature vs model temperature ---
    plt.figure(figsize=(6,8))
    plt.plot(sonde_tdry, sonde_pres, label='Dropsonde Tdry', marker='o', linestyle='-')
    plt.plot(model_temps, sonde_pres, label='Model Temperature', marker='x', linestyle='--')
    plt.gca().invert_yaxis()
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Pressure (hPa)")
    plt.title("Dropsonde vs Model Temperature Profile")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
