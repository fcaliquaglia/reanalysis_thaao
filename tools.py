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
import metpy.calc as mpcalc
from metpy.units import units


def check_empty_df(data_e):
    import pandas as pd

    if data_e is None:
        print("Empty DataFrame")
        return pd.DataFrame(columns=[inpt.var])

    # Check if it's an empty string (in case)
    if isinstance(data_e, str) and data_e.strip() == '':
        print("Empty DataFrame")
        return pd.DataFrame(columns=[inpt.var])

    # Check if it's a DataFrame and empty
    if isinstance(data_e, pd.DataFrame) and data_e.empty:
        print("Empty DataFrame")
        return pd.DataFrame(columns=[inpt.var])

    # If it's none of the above, return as is
    return data_e


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


def get_common_paths(vr, y, prefix):
    location = next((v['fn']
                    for v in inpt.datasets.values() if v.get('switch')), None)
    base_out = Path(inpt.basefol['out']['processed'])
    base_input = Path(inpt.basefol['t']['arcsix'])
    filename = f"{location}_{prefix}_{vr}_{y}.parquet"
    return base_out / filename, base_input


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


def process_rean(vr, data_typ, y, loc):
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

    filenam_grid = f"{data_typ}_grid_index_for_{loc}_loc.txt"
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
        print(f"(First) Latitude = {lat_vals[0]:.4f}")
        print(f"(First) Longitude = {lon_vals[0]:.4f}", end="")
        if data_typ == "c":
            print(f" (also {lon_vals[0]-360:.4f})")
        else:
            print()

        var_name = inpt.extr[vr][data_typ]["var_name"]
        data_list = []

        if not (len(x_idx) == len(y_idx)):
            print("Something's worng with indexes dimension!")
        for i in range(len(y_idx)):
            da = ds[var_name].isel({lat_dim: y_idx[i], lon_dim: x_idx[i]})
            da_small = da.drop_vars(
                ['step', 'surface', 'expver', 'number'], errors='ignore')

            df = da_small.to_dataframe().reset_index().set_index(time_dim)
            df.rename(columns={var_name: vr}, inplace=True)
            data_list.append(df)

        if not data_list:
            print(f"No data extracted for {filename}, skipping write.")
            return

    if active_key in ['buoys']:
        y_idx = coords['y_idx'].to_numpy()
        x_idx = coords['x_idx'].to_numpy()
        t_idx = coords['t_idx'].to_numpy()

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
            f"(First) Date = {pd.Timestamp(time_vals[0]).strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f"(First) Latitude = {lat_vals[0]:.4f}")
        print(f"(First) Longitude = {lon_vals[0]:.4f}", end="")

        if data_typ == "c":
            print(f" (also {lon_vals[0]-360:.4f})")
        else:
            print()

        var_name = inpt.extr[vr][data_typ]["var_name"]
        data_list = []

        if not (len(t_idx) == len(x_idx) == len(y_idx)):
            print("Something's worng with indexes dimension!")
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

        if not data_list:
            print(f"No data extracted for {filename}, skipping write.")
            return

    full_df = pd.concat(data_list)
    if full_df.empty:
        print(f"Warning: Empty DataFrame for {filename}, skipping write.")
        return
    out_path = os.path.join(
        inpt.basefol[data_typ]['processed'],
        f"{inpt.extr[vr][data_typ]['fn']}{loc}_{y}.parquet"
    )
    full_df.to_parquet(out_path)
    print(f"Saved processed data to {out_path}")
