# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 08:50:17 2025

@author: FCQ
"""


import os
import time
import sys

import numpy as np
import pandas as pd
import xarray as xr
import inputs as inpt


def calc_rh_from_tdp():
    """
    Calculate relative humidity from dew point temperature.

    This function processes input data by removing specific columns and adjusting
    the column structure for calculated data. It appears to use dew point
    temperature and environmental temperature to determine relative humidity, but
    the main functionality has been commented out and requires implementation.

    :return: None
    """

    # e = pd.concat([inpt.extr[inpt.var]["t"]["data"], e_t], axis=1)

    # e["rh"] = relative_humidity_from_dewpoint(e["e_t"].values * units.K, e["e_td"].values * units.K).to("percent")
    inpt.extr[inpt.var]["e"]["data"].drop(
        columns=["e_t", "e_td"], inplace=True)
    inpt.extr[inpt.var]["e"]["data"].columns = [inpt.var]

    return

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
            raise TimeoutError(f"File did not complete downloading in {timeout} seconds: {file_path}.\n Increase timeout or download it manually.")

        prev_size = current_size
        time.sleep(interval)


def process_rean(vr, data_typ, y, loc):
    raw_dir = inpt.basefol[data_typ]['raw']
    filename = f"{inpt.extr[vr][data_typ]['fn']}{y}.nc"
    ds_path = os.path.join(raw_dir, filename)

    if os.path.exists(ds_path):
        try:
            wait_for_complete_download(ds_path)
            ds = xr.open_dataset(ds_path, decode_timedelta=True, engine="netcdf4")
            print(f'OK: {os.path.basename(ds_path)}')
        except FileNotFoundError:
            print(f'NOT FOUND: {os.path.basename(ds_path)}')
            return
    else:
        return

    if data_typ == "c":
        ds["longitude"] = ds["longitude"] % 360

    filenam_grid = f"{data_typ}_grid_index_for_{loc}_loc.txt"
    grid_path = os.path.join('txt_locations', filenam_grid)

    if not os.path.exists(grid_path):
        print(
            f"File with reference grid point for {data_typ} NOT found.\n"
            f"You should cleanup folder txt_locations and then run sondes_buoys_flights.py\n"
            f"Now exiting\n{filenam_grid}")
        sys.exit()

    coords = pd.read_csv(grid_path)
    y_idx = coords['y_idx'].to_numpy()
    x_idx = coords['x_idx'].to_numpy()

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

    for i in range(len(y_idx)):
        da = ds[var_name].isel({lat_dim: y_idx[i], lon_dim: x_idx[i]})
        da_small = da.drop_vars(
            ['step', 'surface', 'expver', 'number'], errors='ignore')
        time_dim = 'valid_time' if 'valid_time' in da_small.dims else 'time'
        df = da_small.to_dataframe().reset_index().set_index(time_dim)
        df.rename(columns={var_name: vr}, inplace=True)
        data_list.append(df)

    full_df = pd.concat(data_list)
    out_path = os.path.join(
        inpt.basefol[data_typ]['processed'],
        f"{inpt.extr[vr][data_typ]['fn']}{loc}_{y}.parquet"
    )
    full_df.to_parquet(out_path)
    print(f"Saved processed data to {out_path}")
