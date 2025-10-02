# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 08:50:17 2025

@author: FCQ
"""


import os
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import inputs as inpt
import re
import metpy.calc as mpcalc
from metpy.units import units
import yaml


def replace_none_with_nan(obj):
    if isinstance(obj, dict):
        return {k: replace_none_with_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_none_with_nan(v) for v in obj]
    return np.nan if obj is None else obj


def load_and_process_yaml(path: Path):
    if not path.exists():
        print(f'⚠️ Config file not found for variable: {path.stem}')
        return None

    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = replace_none_with_nan(cfg)

    # Replace placeholders in filenames for keys 'c' and 'e'
    for key in ('c', 'e'):
        if key in cfg and 'fn' in cfg[key]:
            cfg[key]['fn'] = (
                cfg[key]['fn']
                .replace('thaao_c', 'carra1')
                .replace('thaao_e', 'era5_NG')
            )
    return cfg

def parse_datetime_columns(df, file):
    """
    Detect datetime columns in ceilometer files and return a datetime index.
    Handles:
      - '# date[y-m-d]time[h:m:s]'   (combined)
      - 'date[y-m-d]time[h:m:s]'     (combined)
      - '# date[Y-M-D] time[h:m:s]'  (separate)
      - 'date[Y-M-D]' + 'time[h:m:s]' (separate)
    """
    # Clean up column names (strip whitespace and leading "#")
    df.columns = [c.strip().lstrip("#") for c in df.columns]

    cols = list(df.columns)

    # Case 2: Separate date and time columns
    date_col = "date[Y-M-D]"
    time_col = "time[h:m:s]"

    if date_col and time_col:
        datetime_str = df[date_col].astype(str) + " " + df[time_col].astype(str)
        return pd.to_datetime(datetime_str, errors="raise", format="mixed")

    raise ValueError(f"Unexpected datetime columns in {file.name}")

def get_common_paths(vr, y, prefix):

    base_out = Path(inpt.basefol['out']['parquets'])
    base_input = Path(inpt.basefol['t']['arcsix'])
    filename = f"{inpt.location}_{prefix}_{vr}_{y}.parquet"
    return base_out / filename, base_input


def check_empty_df(data, vr):
    try:
        if data is None:
            print("Empty DataFrame or Series")
            return pd.DataFrame(columns=[vr]), True

        # Check if it's an empty string (in case)
        if isinstance(data, str) and data.strip() == '':
            print("Empty DataFrame or Series")
            return pd.DataFrame(columns=[vr]), True

        # Check if it's a DataFrame and empty
        if isinstance(data, pd.DataFrame) and data.empty:
            print("Empty DataFrame")
            return pd.DataFrame(columns=[vr]), True

        # Check if it's a Series and empty
        if isinstance(data, pd.Series) and data.empty:
            print("Empty Series")
            return pd.DataFrame(columns=[vr]), True

        # If it's none of the above, return as is
        return data, False

    except TypeError as e:
        if "string indices must be integers" in str(e):
            print(
                "Caught TypeError - likely due to incorrect type (e.g. string used as dict/DataFrame)")
            return pd.DataFrame(columns=[vr]), False
        else:
            raise  # re-raise unexpected TypeErrors


def plot_vars_cleanup(p_vars, v_data):

    for vvrr in p_vars:
        data = v_data[vvrr]['data']

        # Remove if it's a string or not a DataFrame
        if isinstance(data, str) and data == '':
            p_vars.remove(vvrr)
            continue
        if not isinstance(data, pd.DataFrame):
            p_vars.remove(vvrr)
            continue
        # Optionally also remove if it's empty or all NaNs
        if data.empty or data.isna().all().all():
            p_vars.remove(vvrr)

    return p_vars


def calc_rh_from_tdp():
    """
    Calculate relative humidity from dew point temperature.

    This function processes input data by removing specific columns and adjusting
    the column structure for calculated data. It appears to use dew point
    temperature and environmental temperature to determine relative humidity, but
    the main functionality has been commented out and requires implementation.

    :return: None
    """
    dewpoint = inpt.extr["dewpt"]["e"]["data"]["dewpt"]
    temperature = inpt.extr["temp"]["e"]["data"]["temp"]

    relh = mpcalc.relative_humidity_from_dewpoint(
        temperature.values * units.K, dewpoint.values * units.K).to("percent")

    inpt.extr["rh"]["e"]["data"] = pd.DataFrame(
        {"rh": relh.magnitude}, index=dewpoint.index)

    return


def percentage_to_okta(percent):
    if pd.isna(percent):
        return 9  # optional: treat NaN as sky obscured
    elif percent == 0:
        return 0
    elif percent < 12.5:
        return 1
    elif percent < 25:
        return 2
    elif percent < 37.5:
        return 3
    elif percent < 50:
        return 4
    elif percent < 62.5:
        return 5
    elif percent < 75:
        return 6
    elif percent < 87.5:
        return 7
    elif percent <= 100:
        return 8
    else:
        return 9


def okta_to_percentage(okta_value):
    okta_percent_map = {
        0: 0.0,
        1: 12.5,
        2: 25.0,
        3: 37.5,
        4: 50.0,
        5: 62.5,
        6: 75.0,
        7: 87.5,
        8: 100.0,
        9: np.nan
    }
    return okta_percent_map.get(okta_value, None)


def convert_rs_to_iwv(df, tp):
    """
    Convertito concettualmente in python da codice di Giovanni: PWV_Gio.m
    :param tp: % of the max pressure value up to which calculate the iwv. it is necessary because interpolation fails.
    :param df:
    :return:
    """

    td = mpcalc.dewpoint_from_relative_humidity(
        df['temp'].to_xarray() * units("degC"), df['rh'].to_xarray() / 100)
    iwv = mpcalc.precipitable_water(
        df['pres'].to_xarray() * units("hPa"), td, bottom=None, top=np.nanmin(df['pres']) * tp * units('hPa'))

    return iwv


def decompose_wind(speed_series, direction_series):
    """
    Convert wind speed and direction (in degrees FROM) to u and v components using MetPy.
    Inputs must be pandas Series with units attached (knots, m/s, etc.).
    """
    wspd = speed_series.values * units.meter / units.second  # or your unit
    wdir = direction_series.values * units.degrees
    u, v = mpcalc.wind_components(wspd, wdir)
    return pd.Series(u.magnitude, index=speed_series.index), pd.Series(v.magnitude, index=speed_series.index)


def recompose_wind(u_series, v_series):
    """
    Convert u and v components back to wind speed and direction using MetPy.
    Returns pandas Series for speed and direction.
    """
    u = u_series.values * units.meter / units.second
    v = v_series.values * units.meter / units.second
    speed = mpcalc.wind_speed(u, v)
    direction = mpcalc.wind_direction(u, v)
    return (
        pd.Series(speed.magnitude, index=u_series.index),
        pd.Series(direction.magnitude, index=u_series.index)
    )


import pandas as pd
import warnings

def get_tres(data_typ, tres=None):
    """
    Return a valid frequency string and a tolerance string for resampling.
    
    Parameters:
    -----------
    data_typ : str
        Data component type ('c', 't', etc.)
    tres : str, optional
        Desired time resolution (default: inpt.tres)

    Returns:
    --------
    freq_str : str
        Resampling frequency (pandas offset alias)
    tolerance : str
        Maximum tolerance for nearest-neighbor matching (for pd.Timedelta)
    """
    if tres is None:
        tres = inpt.tres

    if tres != 'original':
        tres_up = tres.upper()

        # --- Monthly (month-end/start) ---
        if tres in ['1ME']:
            freq_str = tres_up   # '1ME' or '1MS' is fine for date_range
            tolerance = '15d'    # Timedelta-safe (~half a month)

        # --- Hourly / 3-hourly ---
        elif tres in ['1h', '3h']:
            freq_str = tres
            tolerance = '10min' if tres == '1h' else '30min'

        # --- Other frequencies ---
        else:
            freq_str = tres
            try:
                td = pd.Timedelta(freq_str)
                tolerance = pd.tseries.frequencies.to_offset(td / 6).freqstr
            except ValueError:
                warnings.warn(
                    f"[WARN] Unknown frequency '{tres}', falling back to 1h/10min."
                )
                freq_str = '1h'
                tolerance = '10min'

        # ✅ Final safety: make sure tolerance is valid
        try:
            pd.Timedelta(tolerance)
        except Exception:
            warnings.warn(
                f"[WARN] Invalid tolerance '{tolerance}', replaced with '1h'."
            )
            tolerance = '1h'

        return freq_str, tolerance

    # --- 'original' case ---
    freq_str = '1h' if inpt.var in inpt.cumulative_vars else ('3h' if data_typ == 'c' else '1h')
    td = pd.Timedelta(freq_str)
    tolerance = pd.tseries.frequencies.to_offset(td / 6).freqstr

    return freq_str, tolerance



def wait_for_complete_download(file_path, timeout=600, interval=5):
    """Wait until the file is fully downloaded by monitoring file size."""
    print(f"Waiting for file to be ready: {file_path}")
    start_time = time.time()
    prev_size = -1

    while True:
        try:
            current_size = os.path.getsize(file_path)
        except FileNotFoundError:
            current_size = -1

        # Check if file size has stabilized
        if current_size == prev_size and current_size > 0:
            print("File download appears complete.")
            break

        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"File did not complete downloading in {timeout} seconds: {file_path}.\n Increase timeout or download it manually.")

        prev_size = current_size
        time.sleep(interval)


def mask_low_count_intervals(df, data_typ, min_frac):

    originalres = df.index[1] - df.index[0]
    if isinstance(originalres, str) and not re.match(r'^\d+', originalres):
        originalres = '1' + originalres
    if isinstance(originalres, str):
        originalres = pd.to_timedelta(originalres)

    group_labels = df.index.floor(inpt.tres)
    counts = group_labels.value_counts()

    def fallback_thresh():
        return int(pd.Timedelta(inpt.tres) / originalres * min_frac)

    # Custom thresholds for 'iwv' var
    thresholds = {
        '1h': {'e': 1, 't': 1, 't2': 1},
        '3h': {'c': 1, 'e': 2, 't': 2, 't2': 2},
        '6h': {'c': 2, 't': 3},
        '12h': {'c': 4, 't': 5},
        '18h': {'c': 5, 't': 8},
        '24h': {'t': 10}
    }

    if inpt.var == 'iwv':
        tres_dict = thresholds.get(inpt.tres, {})
        threshold = tres_dict.get(data_typ, fallback_thresh())
    else:
        threshold = fallback_thresh()

    valid = counts[counts >= threshold].index
    df_masked = df.copy()
    df_masked[~group_labels.isin(valid)] = np.nan
    return df_masked


def process_rean(vr, data_typ, y):
    raw_dir = inpt.basefol[data_typ]['raw']
    filename = f"{inpt.extr[vr][data_typ]['fn']}{y}.nc"
    ds_path = os.path.join(raw_dir, filename)

    if os.path.exists(ds_path):
        wait_for_complete_download(ds_path)
        ds = xr.open_dataset(
            ds_path, decode_timedelta=True, engine="netcdf4")
        print(f'OK: {ds_path}')

    else:
        print(f'NOT FOUND: {ds_path}')
        return

    if data_typ == "c":
        ds["longitude"] = ds["longitude"] % 360

    filenam_grid = f"{data_typ}_grid_index_for_{inpt.location}_loc.txt"
    grid_path = os.path.join('txt_locations', filenam_grid)

    if not os.path.exists(grid_path):
        print(
            f"File with reference grid point for {data_typ} NOT found: {filenam_grid}.\n"
            f"You should cleanup folder txt_locations and then run sondes_buoys_flights.py again \n"
            f"Now exiting")
        sys.exit()

    coords = pd.read_csv(grid_path)

    active_key = next(k for k, v in inpt.datasets.items() if v.get('switch'))

    if active_key in ['THAAO', 'Villum', 'Summit', 'Alert', 'Sigma-A', 'Sigma-B']:
        y_idx = coords['y_idx'].to_numpy()
        x_idx = coords['x_idx'].to_numpy()
        time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        if data_typ == "c":
            lat_vals = np.array([ds['latitude'].values[y, x]
                                for y, x in zip(y_idx, x_idx)])
            lon_vals = np.array([ds['longitude'].values[y, x]
                                for y, x in zip(y_idx, x_idx)])
            lat_dim, lon_dim = 'y', 'x'
        elif data_typ == "e":
            lat_vals = ds["latitude"].isel(latitude=y_idx).values
            lon_vals = ds["longitude"].isel(longitude=x_idx).values
            lat_dim, lon_dim = 'latitude', 'longitude'
        else:
            raise ValueError(f"Unknown dataset_type: {data_typ}")

        print(f"Selected grid point at indices (y={y_idx[0]}, x={x_idx[0]}):")
        print(f"(First out of {len(lat_vals)}) Latitude = {lat_vals[0]:.4f}")
        print(
            f"(First out of {len(lon_vals)}) Longitude = {lon_vals[0]:.4f}", end="")
        if data_typ == "c":
            print(f" (also {lon_vals[0]-360:.4f})")
        else:
            print()

        var_name = inpt.extr[vr][data_typ]["var_name"]
        data_list = []

        if not (len(x_idx) == len(y_idx)):
            print("Something's wrong with indexes dimension!")
        for i in range(len(y_idx)):
            da = ds[var_name].isel({lat_dim: y_idx[i], lon_dim: x_idx[i]})
            da_small = da.drop_vars(
                ['step', 'surface', 'expver', 'number'], errors='ignore')

            df = da_small.to_dataframe().reset_index().set_index(time_dim)
            df.rename(columns={var_name: vr}, inplace=True)
            data_list.append(df)

    if active_key in ['buoys']:
        if (coords['t_idx'].size == 0 or np.all(np.isnan(coords['t_idx'].to_numpy())) or
            coords['x_idx'].size == 0 or np.all(np.isnan(coords['x_idx'].to_numpy())) or
                coords['y_idx'].size == 0 or np.all(np.isnan(coords['y_idx'].to_numpy()))):
            print("Something's wrong with indexes dimension!\nFor example, the dropsonde lat/lon could be outside the ROI.")
            return

        else:
            y_idx = coords['y_idx'].to_numpy().astype(int)
            x_idx = coords['x_idx'].to_numpy().astype(int)
            t_idx = coords['t_idx'].to_numpy().astype(int)

        time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        if data_typ == "c":
            lat_vals = np.array([ds['latitude'].values[y, x]
                                for y, x in zip(y_idx, x_idx)])
            lon_vals = np.array([ds['longitude'].values[y, x]
                                for y, x in zip(y_idx, x_idx)])
            time_vals = np.array(ds[time_dim].values[t_idx])
            lat_dim, lon_dim = 'y', 'x'
        elif data_typ == "e":
            lat_vals = ds["latitude"].isel(latitude=y_idx).values
            lon_vals = ds["longitude"].isel(longitude=x_idx).values
            time_vals = np.array(ds[time_dim].values[t_idx])
            lat_dim, lon_dim = 'latitude', 'longitude'
        else:
            raise ValueError(f"Unknown dataset_type: {data_typ}")

        print(
            f"Selected grid point at indices (t={t_idx[0]}, y={y_idx[0]}, x={x_idx[0]}):")
        print(
            f"(First out of {len(time_vals)}) Date = {pd.Timestamp(time_vals[0]).strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f"(First out of {len(lat_vals)}) Latitude = {lat_vals[0]:.4f}")
        print(
            f"(First out of {len(lon_vals)}) Longitude = {lon_vals[0]:.4f}", end="")

        if data_typ == "c":
            print(f" (also {lon_vals[0]-360:.4f})")
        else:
            print()

        var_name = inpt.extr[vr][data_typ]["var_name"]
        data_list = []

        if not (len(t_idx) == len(x_idx) == len(y_idx)):
            print("Something's wrong with indexes dimension!")
        for i in range(len(y_idx)):
            da = ds[var_name].isel(
                {lat_dim: y_idx[i], lon_dim: x_idx[i], time_dim: t_idx[i]})
            da_small = da.drop_vars(
                ['step', 'surface', 'expver', 'number'], errors='ignore')
            df = pd.DataFrame({
                time_dim: [pd.Timestamp(da_small.valid_time.values)],
                'latitude': [da_small.latitude.item()],
                'longitude': [da_small.longitude.item()],
                var_name: [da_small.item()]
            })
            df.set_index(time_dim, inplace=True)
            df.rename(columns={var_name: vr}, inplace=True)
            data_list.append(df)

    if active_key in ['dropsondes']:
        if (coords['t_idx'].size == 0 or np.isnan(coords['t_idx'].to_numpy()[0]) or
            coords['x_idx'].size == 0 or np.isnan(coords['x_idx'].to_numpy()[0]) or
                coords['y_idx'].size == 0 or np.isnan(coords['y_idx'].to_numpy()[0])):
            print("Something's wrong with indexes dimension!.\n For example, the dropsonde lat lon could be outside the ROI.")
            return

        else:
            y_idx = int(coords['y_idx'].to_numpy()[0])
            x_idx = int(coords['x_idx'].to_numpy()[0])
            t_idx = int(coords['t_idx'].to_numpy()[0])

        time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        if data_typ == "c":
            lat_vals = ds['latitude'].values[y_idx, x_idx]
            lon_vals = ds['longitude'].values[y_idx, x_idx]
            time_vals = ds[time_dim].values[t_idx]
            lat_dim, lon_dim = 'y', 'x'
        elif data_typ == "e":
            lat_vals = ds["latitude"].isel(latitude=y_idx).values
            lon_vals = ds["longitude"].isel(longitude=x_idx).values
            time_vals = ds[time_dim].values[t_idx]
            lat_dim, lon_dim = 'latitude', 'longitude'
        else:
            raise ValueError(f"Unknown dataset_type: {data_typ}")

        print(
            f"Selected grid point at indices (t={t_idx}, y={y_idx}, x={x_idx}):")
        print(
            f"Date = {pd.Timestamp(time_vals).strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f"Latitude = {lat_vals:.4f}")
        print(f"Longitude = {lon_vals:.4f}", end="")

        if data_typ == "c":
            print(f" (also {lon_vals-360:.4f})")
        else:
            print()

        var_name = inpt.extr[vr][data_typ]["var_name"]
        data_list = []

        da = ds[var_name].isel(
            {lat_dim: y_idx, lon_dim: x_idx, time_dim: t_idx})
        da_small = da.drop_vars(
            ['step', 'surface', 'expver', 'number'], errors='ignore')
        df = pd.DataFrame({
            time_dim: [pd.Timestamp(da_small.valid_time.values)],
            'latitude': [da_small.latitude.item()],
            'longitude': [da_small.longitude.item()],
            var_name: [da_small.item()]
        })
        df.set_index(time_dim, inplace=True)
        df.rename(columns={var_name: vr}, inplace=True)
        data_list.append(df)

    if not data_list:
        print(f"No data extracted for {filename}, skipping write.")
        return

    full_df = pd.concat(data_list)
    if full_df.empty:
        print(f"Warning: Empty DataFrame for {filename}, skipping write.")
        return
    out_path = os.path.join(
        inpt.basefol[data_typ]['parquets'],
        f"{inpt.extr[vr][data_typ]['fn']}{inpt.location}_{y}.parquet"
    )
    full_df.to_parquet(out_path)
    print(f"Saved processed data to {out_path}")
