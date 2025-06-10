#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
#
"""
Brief description
"""

# =============================================================
# CREATED:
# AFFILIATION: INGV
# AUTHORS: Filippo Cali" Quaglia
# =============================================================
#
# -------------------------------------------------------------------------------
__author__ = "Filippo Cali' Quaglia"
__credits__ = ["??????"]
__license__ = "GPL"
__version__ = "0.1"
__email__ = "filippo.caliquaglia@ingv.it"
__status__ = "Research"
__lastupdate__ = ""

import datetime as dt
import os
import sys

import julian
import numpy as np
import pandas as pd
import xarray as xr
from metpy.calc import wind_direction, wind_speed
from metpy.units import units
import csv

from math import radians, cos, sin, sqrt, atan2
from geopy.distance import geodesic

import inputs as inpt


def to_180(lon):
    """Convert 0–360 longitude to -180–180."""
    return lon - 360 if lon > 180 else lon


# def find_index_in_grid(ds, ds_type, fn):
#     """
#     Find closest grid point in a dataset to a given lat/lon using geodesic distance.

#     Parameters:
#     - ds: xarray.Dataset (ERA5 or CARRA)
#     - ds_type: str, 'e' for ERA5 (1D lat/lon), 'c' for CARRA (2D lat/lon)
#     - fn: str, output file to write indices
#     - inpt: object with thaao_lat and thaao_lon attributes
#     """

#     target_point = (
#         inpt.thaao_lat,
#         inpt.thaao_lon if inpt.thaao_lon >= 0 else 360 + inpt.thaao_lon
#     )

#     if ds_type == 'c':
#         # CARRA: 2D lat/lon
#         lat_arr = ds['latitude'].values
#         lon_arr = ds['longitude'].values % 360
#     elif ds_type == 'e':
#         # ERA5: 1D lat/lon, need meshgrid
#         lat_1d = ds['latitude'].values
#         lon_1d = ds['longitude'].values % 360
#         lat_arr, lon_arr = np.meshgrid(lat_1d, lon_1d, indexing='ij')
#     else:
#         raise ValueError(f"Unknown dataset type: {ds_type}")

#     # Flatten for distance calculation
#     flat_lat = lat_arr.ravel()
#     flat_lon = lon_arr.ravel()

#     distances = np.array([
#         geodesic(target_point, (lat, lon)).meters
#         for lat, lon in zip(flat_lat, flat_lon)
#     ])

#     # Find closest index and convert back to 2D
#     min_idx = np.argmin(distances)
#     y_idx, x_idx = np.unravel_index(min_idx, lat_arr.shape)

#     # Retrieve matched lat/lon from dataset
#     lat_val = lat_arr[y_idx, x_idx]
#     lon_val = lon_arr[y_idx, x_idx]

#     print(
#         f"Matched closest point: lat={lat_val:.4f}, lon={lon_val:.4f}, idx=({y_idx}, {x_idx})")
#     print(
#         f"Closest to input point: lat={inpt.thaao_lat:.4f}, lon={inpt.thaao_lon:.4f}")

#     # Save index
#     with open(fn, "w") as f:
#         f.write(f"{y_idx}, {x_idx}\n")

#     return y_idx, x_idx


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Compute distance (in meters) between a point and arrays of lat/lon using the haversine formula.
    """
    R = 6371000  # Earth radius in meters
    lat1, lon1 = radians(lat1), radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def find_index_in_grid(ds, ds_type, in_file, out_file):
    """
    Find closest grid points for multiple coordinates in a file.

    Parameters:
    - ds: xarray.Dataset (ERA5 or CARRA)
    - ds_type: str, 'e' for ERA5 (1D lat/lon), 'c' for CARRA (2D lat/lon)
    - in_file: str, input CSV with: datetime, lat, lon, elev
    - out_file: str, output CSV with matched grid info
    """

    # Read input coordinates
    coords = pd.read_csv(os.path.join('txt_locations', in_file))

    # Prepare grid
    if ds_type == 'c':
        lat_arr = ds['latitude'].values
        lon_arr = ds['longitude'].values % 360
    elif ds_type == 'e':
        lat_1d = ds['latitude'].values
        lon_1d = ds['longitude'].values % 360
        lat_arr, lon_arr = np.meshgrid(lat_1d, lon_1d, indexing='ij')
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")

    flat_lat = lat_arr.ravel()
    flat_lon = lon_arr.ravel()

    ds_times = pd.to_datetime(ds['time'].values)

    # Output
    with open(os.path.join('txt_locations', out_file), "w") as out_f:
        header = "datetime,input_lat,input_lon,input_elev,y_idx,x_idx,z_idx,matched_lat,matched_lon,matched_elev,matched_time\n"
        out_f.write(header)

        for row in coords.itertuples(index=False):
            dt_str = row.datetime
            input_lat = row.lat
            input_lon = row.lon
            input_elev = row.elev
            distances = haversine_vectorized(
                input_lat, input_lon, flat_lat, flat_lon)
            min_idx = np.argmin(distances)
            y_idx, x_idx = np.unravel_index(min_idx, lat_arr.shape)
            z_idx = np.nan
            matched_lat = lat_arr[y_idx, x_idx]
            matched_lon = lon_arr[y_idx, x_idx]
            matched_elev = np.nan

            # Parse input datetime
            input_time = pd.to_datetime(dt_str)
            time_diffs = np.abs(ds_times - input_time)
            if np.all(np.isnat(time_diffs)):
                matched_time = np.nan
            else:
                time_idx = time_diffs.argmin()
                matched_time = ds_times[time_idx].strftime("%Y-%m-%dT%H:%M:%S")
            str_fmt = f"{dt_str},{input_lat:.6f},{input_lon:.6f},{input_elev:.2f},{y_idx},{x_idx},{z_idx},{matched_lat:.6f},{matched_lon:.6f},{matched_elev:.2f},{matched_time}\n"
            out_f.write(str_fmt)

    print(f"Index mapping written to {out_file}")


def read_rean(vr, dataset_type):
    """
    Generalized function to read data from different sources (CARRA, ERA5),
    process it, and store it in the inpt.extr dictionary.

    :param vr: Variable key
    :param dataset_type: One of ["c", "e"]
    """
    data_all = pd.DataFrame()

    if dataset_type == "c":
        basefol = inpt.basefol_c
        lon_name, lat_name = 'x', 'y'
        # Convert thaao_lon to 0–360 for CARRA
        thaao_lon = inpt.thaao_lon if inpt.thaao_lon >= 0 else 360 + inpt.thaao_lon
    elif dataset_type == "e":
        basefol = inpt.basefol_e
        lon_name, lat_name = 'longitude', 'latitude'
        # Keep -180 to 180 for ERA5
        thaao_lon = inpt.thaao_lon
    else:
        raise ValueError("dataset_type must be 'c' or 'e'")

    for year in inpt.years:
        ds_path = os.path.join(
            basefol, f'{inpt.extr[vr][dataset_type]["fn"]}{year}.nc')
        try:
            ds = xr.open_dataset(
                ds_path, decode_timedelta=True, engine="netcdf4")
            print(f'OK: {os.path.basename(ds_path)}')
        except FileNotFoundError:
            print(f'NOT FOUND: {os.path.basename(ds_path)}')
            continue

        if dataset_type == "c":
            # Normalize longitude coordinate
            ds["longitude"] = ds["longitude"] % 360  # Normalize to 0–360

        # find or read indexes
        # TODO: add distinzione tra field sites, buoys, track. serve uno switch, 
        # cartelle etc. per il momento fisso su THAAO 
        # oppure usare 2024N_processed per una boa
        tmp_test = 'THAAO'
        filenam_loc = f"{tmp_test}_loc.txt"
        filenam_grid = f"{dataset_type}_grid_index_for_{tmp_test}_loc.txt"
        if not os.path.exists(os.path.join('txt_locations', filenam_loc)):
            print("missing locations for grid comparison")
        else:
            if not os.path.exists(os.path.join('txt_locations', filenam_grid)):
                find_index_in_grid(ds, dataset_type, filenam_loc, filenam_grid)

            coords = pd.read_csv(os.path.join('txt_locations', filenam_grid))

        y_idx = coords['y_idx'].to_numpy()
        x_idx = coords['x_idx'].to_numpy()
        if dataset_type == "c":
            lat_vals = np.array([ds['latitude'].values[y, x]
                                for y, x in zip(y_idx, x_idx)])
            lon_vals = np.array([ds['longitude'].values[y, x]
                                for y, x in zip(y_idx, x_idx)])
            lat_dim, lon_dim = 'y', 'x'
        elif dataset_type == "e":
            lat_vals = ds["latitude"].isel(latitude=y_idx).values
            lon_vals = ds["longitude"].isel(longitude=x_idx).values
            lat_dim, lon_dim = 'latitude', 'longitude'
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        print(f"Selected grid point at indices (y={y_idx}, x={x_idx}):")
        print(f"Latitude = {lat_vals}")
        print(f"Longitude = {lon_vals}")

        # Extract timeseries at closest point
        var_name = inpt.extr[vr][dataset_type]["var_name"]
        data_list = []
        for i in range(len(y_idx)):
            da = ds[var_name].isel({lat_dim: y_idx[i], lon_dim: x_idx[i]})
            da_small = da.drop_vars(
                ['step', 'surface', 'valid_time'], errors='ignore')
            df = da_small.to_dataframe().reset_index()[['time']]
            df['latitude'] = lat_vals[i]
            df['longitude'] = lon_vals[i]
            df[var_name] = da.values
            data_list.append(df)

        # Combine all into one DataFrame
        # data_all = pd.concat(data_list, ignore_index=True)

    # Replace NaNs
    nan_val = inpt.var_dict[dataset_type]["nanval"]
    data_all[data_all == nan_val] = np.nan
    data_all = data_all[inpt.extr[vr][dataset_type]["var_name"]].to_frame()
    inpt.extr[vr][dataset_type]["data"] = data_all
    inpt.extr[vr][dataset_type]["data"].columns = pd.Index([vr])


def read_thaao_weather(vr):
    """
    Reads and processes weather data for the specified variable and updates the
    global input structure. The function attempts to load a NetCDF file associated
    with the given variable and converts it into a pandas DataFrame. It then filters
    and renames columns in the DataFrame based on predefined configurations in
    the global input structure.

    :param vr: The variable identifier used to fetch weather data.
    :type vr: str

    :raises FileNotFoundError: If the required NetCDF file does not exist.

    :return: None. The global input structure is updated directly.
    """
    try:
        inpt.extr[vr]["t"]["data"] = xr.open_dataset(
            os.path.join(inpt.basefol_t, "thaao_aws_vespa",
                         f'{inpt.extr[vr]["t"]["fn"]}.nc'),
            engine="netcdf4").to_dataframe()
        print(f'OK: {inpt.extr[vr]["t"]["fn"]}.nc')
    except FileNotFoundError:
        print(f'NOT FOUND: {inpt.extr[vr]["t"]["fn"]}.nc')
    inpt.extr[vr]["t"]["data"] = inpt.extr[vr]["t"]["data"][[
        inpt.extr[vr]["t"]["column"]]]
    inpt.extr[vr]["t"]["data"].columns = [vr]
    return


def read_thaao_rad(vr):
    """
    Reads and processes the Thaao radiation data for a specific variable over a specified
    date range and yearly subset defined within the configuration. This function iterates
    through years, attempts to load the corresponding data files for the input variable, and
    processes the data to generate a time-indexed DataFrame for further analysis. If a file is
    not found for a specific year, a message is logged to indicate the missing data.

    :param vr: The variable name used to index configuration details and process corresponding
               data.
    :type vr: str
    :return: None
    """
    t_tmp_all = pd.DataFrame()
    for i in inpt.rad_daterange[inpt.rad_daterange.year.isin(inpt.years)]:
        i_fmt = int(i.strftime("%Y"))
        try:
            t_tmp = pd.read_table(
                os.path.join(inpt.basefol_t, "thaao_rad",
                             f'{inpt.extr[vr]["t"]["fn"]}{i_fmt}_5MIN.dat'),
                engine="python", skiprows=None, header=0, decimal=".", sep=r"\s+")
            tmp = np.empty(t_tmp["JDAY_UT"].shape, dtype=dt.datetime)
            for ii, el in enumerate(t_tmp["JDAY_UT"]):
                new_jd_ass = el + \
                    julian.to_jd(dt.datetime(
                        i_fmt - 1, 12, 31, 0, 0), fmt="jd")
                tmp[ii] = julian.from_jd(new_jd_ass, fmt="jd")
                tmp[ii] = tmp[ii].replace(microsecond=0)
            t_tmp.index = pd.DatetimeIndex(tmp)
            t_tmp = t_tmp[[inpt.extr[vr]["t"]["column"]]]
            t_tmp_all = pd.concat([t_tmp_all, t_tmp], axis=0)
            print(f'OK: {inpt.extr[vr]["t"]["fn"]}{i_fmt}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {inpt.extr[vr]["t"]["fn"]}{i_fmt}.txt')
    inpt.extr[vr]["t"]["data"] = t_tmp_all
    inpt.extr[vr]["t"]["data"].columns = [vr]
    return


def read_thaao_hatpro(vr):
    """
    Reads and processes data from a specified text file into a structured dataframe. This function
    attempts to read and parse data for a given `vr` (variable identifier) from a specific file path.
    It processes datetime information, assigns appropriate column names, and stores the resultant
    data within a structured input object. If the specified file is not found, it logs the error
    without halting execution.

    :param vr: The variable identifier used to locate and process the associated data.
    :type vr: str
    :return: None
    """
    # t1_tmp_all = pd.DataFrame()
    try:
        #    t1_tmp = pd.read_table(
        #            os.path.join(
        #                    inpt.basefol_t, "thaao_hatpro",
        #                    f'{inpt.extr[vr]["t1"]["fn"]}', f'{inpt.extr[vr]["t1"]["fn"]}.DAT'),
        #            sep=r"\s+", engine="python", header=0, skiprows=9)
        # #   t1_tmp.columns = ["Date[y_m_d]", "Time[h:m]", "LWP[g/m2]", "STD_LWP[g/m2]", "Num"]
        #    # t1_tmp_all = t1_tmp

        #    t1_tmp.index = pd.to_datetime(
        #    (t1_tmp[["Date_y_m_d"]].values + " " + t1_tmp[["Time_h:m"]].values)[:,0],
        #    format="%Y-%m-%d %H:%M:%S")
        t1_tmp = pd.read_table(
            os.path.join(
                inpt.basefol_t, "thaao_hatpro", f'{inpt.extr[vr]["t1"]["fn"]}',
                f'{inpt.extr[vr]["t1"]["fn"]}.DAT'), sep=r"\s+", engine="python", header=0, skiprows=9,
            parse_dates={"datetime": [0, 1]}, date_format="%Y-%m-%d %H:%M:%S", index_col="datetime")

        inpt.extr[vr]["t1"]["data"] = t1_tmp[[inpt.extr[vr]["t1"]["column"]]]

        inpt.extr[vr]["t1"]["data"].columns = [vr]

        print(f'OK: {inpt.extr[vr]["t1"]["fn"]}.DAT')
    except FileNotFoundError:
        print(f'NOT FOUND: {inpt.extr[vr]["t1"]["fn"]}.DAT')


def read_thaao_ceilometer(vr):
    """
    Reads and processes ceilometer data for a specified variable from multiple files
    in a given directory structure. The data is collected from files corresponding
    to a specific date range and year(s). This function concatenates the data
    from multiple files, cleans it (replacing specific values with NaNs),
    and processes timestamps to create a time-indexed DataFrame.

    :param vr: The variable key indicating the specific type of data
        to be processed from the ceilometer files (e.g., temperature).
    :type vr: str

    :raises FileNotFoundError: Raised if a file for a given date is not found.
    :raises pd.errors.EmptyDataError: Raised if a file for a given date is empty.

    :return: None
    """
    t_tmp_all = pd.DataFrame()
    for i in inpt.ceilometer_daterange[inpt.ceilometer_daterange.year.isin(inpt.years)]:
        i_fmt = i.strftime("%Y%m%d")
        try:
            t_tmp = pd.read_table(
                os.path.join(
                    inpt.basefol_t, "thaao_ceilometer", "medie_tat_rianalisi",
                    f'{i_fmt}{inpt.extr[vr]["t"]["fn"]}.txt'), skipfooter=0, sep=r"\s+", header=0, skiprows=9,
                engine="python")
            t_tmp[t_tmp == inpt.var_dict["t"]["nanval"]] = np.nan
            t_tmp_all = pd.concat([t_tmp_all, t_tmp], axis=0)
            print(f'OK: {i_fmt}{inpt.extr[vr]["t"]["fn"]}.txt')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f'NOT FOUND: {i_fmt}{inpt.extr[vr]["t"]["fn"]}.txt')
    inpt.extr[vr]["t"]["data"] = t_tmp_all
    inpt.extr[vr]["t"]["data"].index = pd.to_datetime(
        inpt.extr[vr]["t"]["data"]["#"] + " " +
        inpt.extr[vr]["t"]["data"]["date[y-m-d]time[h:m:s]"],
        format="%Y-%m-%d %H:%M:%S")
    inpt.extr[vr]["t"]["data"].index.name = "datetime"
    inpt.extr[vr]["t"]["data"] = inpt.extr[vr]["t"]["data"].iloc[:, :].filter(
        [inpt.extr[vr]["t"]["column"]]).astype(float)
    inpt.extr[vr]["t"]["data"].columns = [vr]

    return


def read_aws_ecapac(vr):
    """
    Reads and processes AWS ECAPAC data for a specified variable across a specific date range
    defined in the input configuration. Concatenates data for all specified dates, handles missing
    or malformed files, and updates the data container with the formatted results.

    :param vr: Variable name (str) specifying the dataset key in the input extraction configuration.
    :return: None
    """
    t2_tmp_all = pd.DataFrame()
    for i in inpt.aws_ecapac_daterange[inpt.aws_ecapac_daterange.year.isin(inpt.years)]:
        i_fmt = i.strftime("%Y_%m_%d")
        try:
            file = os.path.join(
                inpt.basefol_t, "thaao_ecapac_aws_snow", "AWS_ECAPAC", i.strftime(
                    "%Y"),
                f'{inpt.extr[vr]["t2"]["fn"]}{i_fmt}_00_00.dat')
            t2_tmp = pd.read_csv(
                file, skiprows=[0, 3], header=0, decimal=".", delimiter=",", engine="python",
                index_col="TIMESTAMP").iloc[1:, :]
            t2_tmp_all = pd.concat([t2_tmp_all, t2_tmp], axis=0)

            print(f'OK: {inpt.extr[vr]["t2"]["fn"]}{i_fmt}_00_00.dat')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f'NOT_FOUND: {inpt.extr[vr]["t2"]["fn"]}{i_fmt}_00_00.dat')
    inpt.extr[vr]["t2"]["data"] = t2_tmp_all
    inpt.extr[vr]["t2"]["data"].index = pd.DatetimeIndex(
        inpt.extr[vr]["t2"]["data"].index)
    inpt.extr[vr]["t2"]["data"].index.name = "datetime"
    inpt.extr[vr]["t2"]["data"] = inpt.extr[vr]["t2"]["data"].iloc[:, :].filter(
        [inpt.extr[vr]["t2"]["column"]]).astype(float)
    inpt.extr[vr]["t2"]["data"].columns = [vr]
    return


def read_alb():
    """
    Reads and processes the input variable data from multiple sources including CARRA, ERA5,
    and THAAO. Adjusts the datasets by scaling and cleaning data points
    as per defined conditions.

    :raises KeyError: If required keys are missing in the input dictionary for any source.
    :raises TypeError: If data type mismatches occur when processing datasets within sources.
    :raises ValueError: If data contains invalid values not conforming to the expected range.
    """
    # CARRA
    read_rean(inpt.var, "c")
    inpt.extr[inpt.var]["c"]["data"] = inpt.extr[inpt.var]["c"]["data"] / 100.
    inpt.extr[inpt.var]["c"]["data"][inpt.extr[inpt.var]
                                     ["c"]["data"] <= 0.] = np.nan

    # ERA5
    read_rean(inpt.var, "e")
    inpt.extr[inpt.var]["c"]["data"][inpt.extr[inpt.var]
                                     ["c"]["data"] <= 0.] = np.nan

    # THAAO
    read_thaao_rad(inpt.var)

    return


def read_cbh():
    """
    Reads cloud base height (CBH) data from different sources and processes it
    based on the specified input variable. This function integrates data from
    multiple sources including CARRA, ERA5, and THAAO Ceilometer, providing a
    comprehensive dataset for further analysis.

    :raises ValueError: If the specified input variable is invalid or unsupported.
    :return: None
    """
    # CARRA
    read_rean(inpt.var, "c")

    # ERA5
    read_rean(inpt.var, "e")

    # THAAO
    read_thaao_ceilometer(inpt.var)

    return


def read_lwp():
    """
    Reads cloud liquid water path (LWP) data from multiple sources, processes it, and
    updates the respective datasets for further analysis. This function specifically
    handles data from three sources: CARRA, ERA5, and THAAO1. Upon reading the data,
    conditions are applied to clean and format it appropriately for use.

    :raises ValueError: If any required input data is missing or improperly formatted.

    :param None:

    :return: None
    """
    # CARRA
    read_rean(inpt.var, "c")
    inpt.extr[inpt.var]["c"]["data"] = inpt.extr[inpt.var]["c"]["data"]
    # inpt.extr[inpt.var]["c"]["data"][inpt.extr[inpt.var]["c"]["data"] < 0.01] = np.nan
    # c[c < 15] = 0

    # ERA5
    read_rean(inpt.var, "e")
    inpt.extr[inpt.var]["e"]["data"] = inpt.extr[inpt.var]["e"]["data"]
    inpt.extr[inpt.var]["e"]["data"][inpt.extr[inpt.var]
                                     ["e"]["data"] < 0.01] = np.nan
    # e[e < 15] = 0

    # THAAO1
    read_thaao_hatpro(inpt.var)
    inpt.extr[inpt.var]["t1"]["data"][inpt.extr[inpt.var]
                                      ["t1"]["data"] < 0.01] = np.nan
    # t1[t1 < 15] = 0

    return


def read_msl_pres():
    """
    Reads mean sea-level pressure (MSL) from the specified source.

    This function is responsible for reading mean sea-level pressure
    data using the method or procedure defined in the `read_carra`
    function. It uses the `inpt.var` as an input parameter for the
    read operation.

    :return: None
    """
    # CARRA
    read_rean(inpt.var, "c")

    return


def read_precip():
    """
    Reads and processes precipitation data from multiple sources.

    This function integrates data from CARRA, ERA5, and THAAO2 datasets, applying
    necessary transformations and storing resulting data within the provided input
    structure. The function is responsible for coordinating the reading of data
    via corresponding source-specific readers and ensuring compatibility of the
    processed data across sources.

    :return: None
    """
    # CARRA
    read_rean(inpt.var, "c")

    # ERA5
    read_rean(inpt.var, "e")
    inpt.extr[inpt.var]["e"]["data"] = inpt.extr[inpt.var]["e"]["data"].values * \
        1000.

    # THAAO2
    read_aws_ecapac(inpt.var)

    return


def read_lw_down():
    """
    Reads and processes longwave downward radiation ("lw_down") data from multiple sources such as
    CARRA, ERA5, and THAAO. The function handles data quality issues by filtering out
    non-physical values (values less than 0) and applies conversion factors to the raw data using
    values stored in the `var_dict` dictionary.

    The processed data is stored in the `inpt.extr` dictionary under specific keys corresponding to
    each data source ("c" for CARRA, "e" for ERA5, and "t" for THAAO).

    :raises KeyError: If keys expected in `inpt.extr` or `inpt.var_dict` for specific
        sources ("c", "e", "t") are missing.
    :raises ValueError: If data conversion or filtering operations encounter unexpected
        or malformed data values.

    :return: None
    """
    # CARRA
    read_rean("lw_down", "c")
    inpt.extr["lw_down"]["c"]["data"][inpt.extr["lw_down"]
                                      ["c"]["data"] < 0.] = np.nan
    inpt.extr["lw_down"]["c"]["data"] = inpt.extr["lw_down"]["c"]["data"] / \
        inpt.var_dict["c"]["rad_conv_factor"]

    # ERA5
    read_rean("lw_down", "e")
    inpt.extr["lw_down"]["e"]["data"][inpt.extr["lw_down"]
                                      ["e"]["data"] < 0.] = np.nan
    inpt.extr["lw_down"]["e"]["data"] = inpt.extr["lw_down"]["e"]["data"] / \
        inpt.var_dict["e"]["rad_conv_factor"]

    # THAAO
    read_thaao_rad("lw_down")
    inpt.extr["lw_down"]["t"]["data"][inpt.extr["lw_down"]
                                      ["t"]["data"] < 0.] = np.nan
    return


def read_sw_down():
    """
    Reads and processes shortwave downward radiation data from multiple data sources,
    including CARRA, ERA5, ERA5-LAND, and THAAO. Negative values of radiation data
    are replaced with NaN, and the data is scaled according to respective radiation
    conversion factors.

    :return: None
    """
    # CARRA
    read_rean("sw_down", "c")
    inpt.extr["sw_down"]["c"]["data"][inpt.extr["sw_down"]
                                      ["c"]["data"] < 0.] = np.nan
    inpt.extr["sw_down"]["c"]["data"] = inpt.extr["sw_down"]["c"]["data"] / \
        inpt.var_dict["c"]["rad_conv_factor"]

    # ERA5
    read_rean("sw_down", "e")
    inpt.extr["sw_down"]["e"]["data"][inpt.extr["sw_down"]
                                      ["e"]["data"] < 0.] = np.nan
    inpt.extr["sw_down"]["e"]["data"] = inpt.extr["sw_down"]["e"]["data"] / \
        inpt.var_dict["e"]["rad_conv_factor"]

    # THAAO
    read_thaao_rad("sw_down")
    inpt.extr["sw_down"]["t"]["data"][inpt.extr["sw_down"]
                                      ["t"]["data"] < 0.] = np.nan
    return


def read_lw_up():
    """
    Processes and calculates longwave upwelling radiation data for multiple datasets.

    This function reads and processes longwave radiation data, including
    net longwave radiation and upward longwave radiation, for various
    datasets including CARRA, ERA5, ERA5-LAND, and THAAO. The processing
    includes unit conversions using a radiation conversion factor,
    subtracting net radiation from downward radiation to compute upward
    radiation, and handling invalid or unusable data (e.g., setting negative
    values to NaN).

    The data is categorized into different keys for each dataset, and the
    relevant computations are conducted based on the respective radiation
    conversion factors and formulas for each dataset source.

    :raises KeyError: If required data keys are missing in input dictionaries.
    :raises TypeError: If the input data types are incompatible with the operations.
    :raises ValueError: If data contains invalid values for computations.
    :raises AttributeError: If required attributes are missing in the input object.
    :return: None
    """
    read_lw_down()

    # CARRA
    read_rean("lw_net", "c")
    inpt.extr["lw_net"]["c"]["data"] = inpt.extr["lw_net"]["c"]["data"] / \
        inpt.var_dict["c"]["rad_conv_factor"]
    inpt.extr["lw_up"]["c"]["data"] = pd.DataFrame(
        index=inpt.extr["lw_down"]["c"]["data"].index,
        data=inpt.extr["lw_down"]["c"]["data"].values - inpt.extr["lw_net"]["c"]["data"].values, columns=["lw_up"])
    inpt.extr["lw_up"]["c"]["data"][inpt.extr["lw_up"]
                                    ["c"]["data"] < 0.] = np.nan
    # del inpt.extr["lw_net"]["c"]["data"]

    # ERA5
    read_rean("lw_net", "e")
    inpt.extr["lw_net"]["e"]["data"] = inpt.extr["lw_net"]["e"]["data"] / \
        inpt.var_dict["e"]["rad_conv_factor"]
    inpt.extr["lw_up"]["e"]["data"] = pd.DataFrame(
        index=inpt.extr["lw_down"]["e"]["data"].index,
        data=inpt.extr["lw_down"]["e"]["data"].values - inpt.extr["lw_net"]["e"]["data"].values, columns=["lw_up"])
    inpt.extr["lw_up"]["e"]["data"][inpt.extr["lw_up"]
                                    ["e"]["data"] < 0.] = np.nan
    # del inpt.extr["lw_net"]["e"]["data"]

    # THAAO
    read_thaao_rad("lw_up")
    inpt.extr["lw_up"]["t"]["data"][inpt.extr["lw_up"]
                                    ["t"]["data"] < 0.] = np.nan

    return


def read_sw_up():
    """
    Reads and processes shortwave upward radiation data from various sources, including CARRA, ERA5,
    ERA5-LAND, and THAAO. The function adjusts and prepares this data by performing necessary unit
    conversions, calculations, and value filtering. Data from each source is extracted, recalculated
    using conversion factors, filtered for valid values, and any processed intermediate data is
    removed to optimize memory usage.

    :raise KeyError: if the expected keys are not found in the data structures.
    :raise ValueError: if the data contains invalid or unexpected values.

    :return: None
    """
    read_sw_down()

    # CARRA
    read_rean("sw_net", "c")
    inpt.extr["sw_net"]["c"]["data"] = inpt.extr["sw_net"]["c"]["data"] / \
        inpt.var_dict["c"]["rad_conv_factor"]
    inpt.extr["sw_up"]["c"]["data"] = pd.DataFrame(
        index=inpt.extr["sw_down"]["c"]["data"].index,
        data=inpt.extr["sw_down"]["c"]["data"].values - inpt.extr["sw_net"]["c"]["data"].values, columns=["sw_up"])
    inpt.extr["sw_up"]["c"]["data"][inpt.extr["sw_up"]
                                    ["c"]["data"] < 0.] = np.nan
    del inpt.extr["sw_net"]["c"]["data"]

    # ERA5
    read_rean("sw_net", "e")
    inpt.extr["sw_net"]["e"]["data"] = inpt.extr["sw_net"]["e"]["data"] / \
        inpt.var_dict["e"]["rad_conv_factor"]
    inpt.extr["sw_up"]["e"]["data"] = pd.DataFrame(
        index=inpt.extr["sw_down"]["e"]["data"].index,
        data=inpt.extr["sw_down"]["e"]["data"].values - inpt.extr["sw_net"]["e"]["data"].values, columns=["sw_up"])
    inpt.extr["sw_up"]["e"]["data"][inpt.extr["sw_up"]
                                    ["e"]["data"] < 0.] = np.nan
    del inpt.extr["sw_net"]["e"]["data"]

    # THAAO
    read_thaao_rad("sw_up")
    inpt.extr["sw_up"]["t"]["data"][inpt.extr["sw_up"]
                                    ["t"]["data"] < 0.] = np.nan

    return


def read_rh():
    """
    Reads and processes relative humidity data from various sources including
    CARRA, ERA5, ERA5-LAND, and THAAO2-based weather datasets. This function
    includes checks for missing data and calculates relative humidity from
    temperature and dew point temperature if necessary.

    :raises KeyError: Raised if there is a key-related issue during data access
        within the input variables.
    :raises ValueError: Raised if an unexpected data inconsistency is encountered.
    """
    # CARRA
    read_rean(inpt.var, "c")

    # ERA5
    read_rean(inpt.var, "e")
    if inpt.extr[inpt.var]["l"]["data"].empty:
        read_rean("temp", "e")
    calc_rh_from_tdp()

    # THAAO2
    read_thaao_weather(inpt.var)
    read_aws_ecapac(inpt.var)
    read_thaao_weather(inpt.var)

    return


def calc_rh_from_tdp():
    """
    Calculate relative humidity from dew point temperature.

    This function processes input data by removing specific columns and adjusting
    the column structure for calculated data. It appears to use dew point
    temperature and environmental temperature to determine relative humidity, but
    the main functionality has been commented out and requires implementation.

    :return: None
    """
    # TODO not working

    # e = pd.concat([inpt.extr[inpt.var]["t"]["data"], e_t], axis=1)

    # e["rh"] = relative_humidity_from_dewpoint(e["e_t"].values * units.K, e["e_td"].values * units.K).to("percent")
    inpt.extr[inpt.var]["e"]["data"].drop(
        columns=["e_t", "e_td"], inplace=True)
    inpt.extr[inpt.var]["e"]["data"].columns = [inpt.var]

    return


def read_surf_pres():
    """
    Reads surface pressure data from multiple sources, processes the data, and handles exceptional
    values by setting them to NaN. The function operates on multiple datasets - CARRA, ERA5, THAAO,
    and THAAO2, adjusting their scales or removing data in certain cases.

    :raises: KeyError if `inpt.var` is not found in the input data structure or if any expected
             fields are incomplete or missing.
    """
    # CARRA
    read_rean(inpt.var, "c")
    inpt.extr[inpt.var]["c"]["data"] = inpt.extr[inpt.var]["c"]["data"] / 100.
    inpt.extr[inpt.var]["c"]["data"][inpt.extr[inpt.var]
                                     ["c"]["data"] <= 900] = np.nan

    # ERA5
    read_rean(inpt.var, "e")
    inpt.extr[inpt.var]["e"]["data"] = inpt.extr[inpt.var]["e"]["data"] / 100.
    inpt.extr[inpt.var]["e"]["data"][inpt.extr[inpt.var]
                                     ["e"]["data"] <= 900] = np.nan

    # THAAO
    read_thaao_weather(inpt.var)
    inpt.extr[inpt.var]["t"]["data"][inpt.extr[inpt.var]
                                     ["t"]["data"] <= 900] = np.nan
    inpt.extr[inpt.var]["t"]["data"].loc["2021-10-11 00:00:00":"2021-10-19 00:00:00"] = np.nan
    inpt.extr[inpt.var]["t"]["data"].loc["2024-4-26 00:00:00":"2024-5-4 00:00:00"] = np.nan

    # THAAO2
    read_aws_ecapac(inpt.var)

    return


def read_tcc():
    """
    Reads data from multiple sources (CARRA, ERA5, and THAAO ceilometer) and processes them accordingly.

    This function performs the following:
    - Reads data from the CARRA dataset.
    - Reads data from the ERA5 dataset and processes this data by scaling certain values.
    - Reads data from the THAAO ceilometer dataset.
    Finally, the function does not return any data as it modifies `inpt` directly.

    :return: None
    """
    # CARRA
    read_rean(inpt.var, "c")

    # ERA5
    read_rean(inpt.var, "e")
    inpt.extr[inpt.var]["e"]["data"] = inpt.extr[inpt.var]["e"]["data"].values * 100.

    # THAAO
    read_thaao_ceilometer(inpt.var)

    return


def read_temp():
    """
    Transforms temperature data from various datasets to Celsius and retrieves
    reformatted sources into a structured container. The function processes data
    from the CARRA, ERA5, ERA5-LAND, THAAO, and THAAO2 datasets, converting values
    from Kelvin to Celsius for consistency.

    :return: None
    """
    # CARRA
    read_rean(inpt.var, "c")
    inpt.extr[inpt.var]["c"]["data"] = inpt.extr[inpt.var]["c"]["data"] - 273.15

    # ERA5
    read_rean(inpt.var, "e")
    inpt.extr[inpt.var]["e"]["data"] = inpt.extr[inpt.var]["e"]["data"] - 273.15

    # THAAO
    read_thaao_weather(inpt.var)
    inpt.extr[inpt.var]["t"]["data"] = inpt.extr[inpt.var]["t"]["data"] - 273.15

    # THAAO2
    read_aws_ecapac(inpt.var)
    return


def read_wind():
    """
    Calculates and populates wind speed and direction data from different
    data sources, including CARRA, ERA5, ERA5-LAND (currently deactivated),
    and AWS ECAPAC. The method reads data, computes wind speed and direction,
    and stores the processed results back into the `inpt` structure.

    The wind speed is calculated using components (u, v) provided by respective
    datasets. Similarly, the wind direction is derived from the same components
    where applicable. The method currently deactivates ERA5-LAND calculations
    due to unavailable data.

    :param inpt: Input structure that contains extracted data for various
        wind component keys such as "windu" and "windv" from different
        models. The method updates wind speed ("winds") and wind direction
        ("windd") within this structure based on computed results.
        Must already contain placeholders for updated wind speed and direction
        data for respective datasets.
    :type inpt: object

    :return: Updates the `inpt` structure in place with calculated wind speed
        ("winds") and wind direction ("windd") for available datasets.
    :rtype: None
    """
    # CARRA
    read_rean("winds", "c")
    read_rean("windd", "c")

    # ERA5
    read_rean("windu", "e")
    read_rean("windv", "e")
    e_ws = wind_speed(
        inpt.extr["windu"]["e"]["data"].values * units("m/s"),
        inpt.extr["windv"]["e"]["data"].values * units("m/s"))
    inpt.extr["winds"]["e"]["data"] = pd.DataFrame(
        index=inpt.extr["windu"]["e"]["data"].index, data=e_ws.magnitude, columns=["winds"])

    e_wd = wind_direction(
        inpt.extr["windu"]["e"]["data"].values * units("m/s"),
        inpt.extr["windv"]["e"]["data"].values * units("m/s"))
    inpt.extr["windd"]["e"]["data"] = pd.DataFrame(
        index=inpt.extr["windu"]["e"]["data"].index, data=e_wd.magnitude, columns=["windd"])

    # THAAO2
    read_aws_ecapac("winds")
    read_aws_ecapac("windd")

    return


def calc_rad_acc_era5_land(vr):
    """
    Calculates instantaneous radiation accumulation from daily accumulated ERA5-LAND data.

    This function processes the daily-accumulated radiation data for ERA5-LAND by
    calculating the difference between consecutive timesteps. It also handles specific
    timesteps (e.g., at 0100 hours), ensuring the data integrity for further analysis.

    :param vr: The key or identifier in the `inpt.extr` data dictionary used to locate
        the radiation dataset for processing.
    :type vr: str
    :return: None
    """
    # calculating instantaneous as difference with previous timestep
    inpt.extr[vr]["l"]["data_diff"] = inpt.extr[vr]["l"]["data"][vr].diff()
    # dropping value at 0100 which does not need any subtraction (it is the first of the day)
    inpt.extr[vr]["l"]["data_diff"] = inpt.extr[vr]["l"]["data_diff"][inpt.extr[vr]
                                                                      ["l"]["data_diff"].index.hour != 1]
    # selecting original value at 0100
    orig_filtered_data = inpt.extr[vr]["l"]["data"][inpt.extr[vr]
                                                    ["l"]["data"].index.hour == 1]
    # appending original value at 0100
    inpt.extr[vr]["l"]["data_diff"] = pd.concat(
        [inpt.extr[vr]["l"]["data_diff"], orig_filtered_data]).sort_index()
    inpt.extr[vr]["l"]["data"] = inpt.extr[vr]["l"]["data_diff"]

    print("ERA5-LAND data for radiation corrected because they are values daily-accumulated!")
    return


def read():
    """
    Reads data based on the variable type given in the `inpt` object and determines
    the appropriate function to be called. Different types of data such as
    "alb", "cbh", "msl_pres", "lwp", and others are supported, with each type
    corresponding to a specified reading function. The method selects and
    executes the specific reader function based on the value assigned to
    `inpt.var`.

    :return: The result of the specific data reading operation. The type of return
       value depends on the reader function executed.
    """
    if inpt.var == "alb":
        return read_alb()
    if inpt.var == "cbh":
        return read_cbh()
    if inpt.var == "msl_pres":
        return read_msl_pres()
    if inpt.var == "lwp":
        return read_lwp()
    if inpt.var == "lw_down":
        return read_lw_down()
    if inpt.var == "lw_up":
        return read_lw_up()
    if inpt.var == "precip":
        return read_precip()
    if inpt.var == "rh":
        return read_rh()
    if inpt.var == "surf_pres":
        return read_surf_pres()
    if inpt.var == "sw_down":
        return read_sw_down()
    if inpt.var == "sw_up":
        return read_sw_up()
    if inpt.var == "tcc":
        return read_tcc()
    if inpt.var == "temp":
        return read_temp()
    if inpt.var in ["winds", "windd"]:
        return read_wind()
