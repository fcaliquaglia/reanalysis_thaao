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
# AUTHORS: Filippo Cali' Quaglia, Monica Tosco
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
import read_func_thaao as rd_ft
import read_func_villum as rd_fv
from metpy.calc import wind_direction, wind_speed
from metpy.units import units
import inputs as inpt
import tools as tls


def read_rean(vr, dataset_type):
    """
    Generalized function to read and process data from CARRA or ERA5 datasets.

    This function checks for processed yearly parquet files for the given variable and dataset type.
    If any yearly file is missing, it triggers the processing function `tls.process_rean` to generate it.
    Finally, it reads all the yearly parquet files and concatenates them into a single DataFrame,
    which is stored in the global `inpt.extr` structure.

    :param vr: Variable name to read (e.g., "temp", "sw_down").
    :param dataset_type: Dataset type identifier ("c" for CARRA, "e" for ERA5).
    :return: None
    """
    # Determine location filename prefix based on active dataset switches
    location = next(
        (v['fn'] for k, v in inpt.datasets.items() if v.get('switch')), None)

    # Process missing yearly files if needed
    for year in inpt.years:
        output_file = f"{inpt.extr[vr][dataset_type]['fn']}{location}_{year}.parquet"
        output_path = os.path.join(
            inpt.basefol[dataset_type]['processed'], output_file)

        if not os.path.exists(output_path):
            tls.process_rean(vr, dataset_type, year, location)

    # Read all yearly parquet files and concatenate into a single DataFrame
    data_all = pd.DataFrame()
    for year in inpt.years:
        input_file = f"{inpt.extr[vr][dataset_type]['fn']}{location}_{year}.parquet"
        input_path = os.path.join(
            inpt.basefol[dataset_type]['processed'], input_file)
        try:
            data_tmp = pd.read_parquet(input_path)
            print(f"Loaded {input_path}")
        except FileNotFoundError as e:
            print(f"File not found: {input_path} ({e})")
            continue
        data_all = pd.concat([data_all, data_tmp], ignore_index=True)

    # Store the concatenated data in the input extraction dictionary
    inpt.extr[vr][dataset_type]["data"] = data_all


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
    inpt.extr[inpt.var]["c"]["data"][inpt.var] /= 100.
    inpt.extr[inpt.var]["c"]["data"][inpt.var].loc[
        inpt.extr[inpt.var]["c"]["data"][inpt.var] <= 0.] = np.nan

    # ERA5
    read_rean(inpt.var, "e")
    inpt.extr[inpt.var]["e"]["data"][inpt.var].loc[
        inpt.extr[inpt.var]["e"]["data"][inpt.var] <= 0.] = np.nan

    # THAAO
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_rad(inpt.var)

    return


def read_cbh():
    """
    Reads cloud base height (CBH) data from multiple sources and processes it
    based on the specified input variable. Supports data from CARRA, ERA5, and THAAO ceilometer.

    :raises ValueError: If the specified input variable is invalid or unsupported.
    :return: None
    """
    # CARRA
    read_rean(inpt.var, "c")

    # ERA5
    read_rean(inpt.var, "e")

    # THAAO ceilometer
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_ceilometer(inpt.var)

    return


def read_lwp():
    """
    Reads cloud liquid water path (LWP) data from CARRA, ERA5, and THAAO1,
    applies cleaning conditions (setting values < 0.01 to NaN), and updates datasets.
    """
    # --- CARRA ---
    read_rean(inpt.var, "c")
    lwp_c = inpt.extr[inpt.var]["c"]["data"][inpt.var]
    # Uncomment below line if you want to filter small values
    # lwp_c[lwp_c < 0.01] = np.nan
    inpt.extr[inpt.var]["c"]["data"][inpt.var] = lwp_c

    # --- ERA5 ---
    read_rean(inpt.var, "e")
    lwp_e = inpt.extr[inpt.var]["e"]["data"][inpt.var]
    lwp_e[lwp_e < 0.01] = np.nan
    inpt.extr[inpt.var]["e"]["data"][inpt.var] = lwp_e

    # --- THAAO1 ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_hatpro(inpt.var)
        lwp_t1 = inpt.extr[inpt.var]["t1"]["data"][inpt.var]
        lwp_t1[lwp_t1 < 0.01] = np.nan
        inpt.extr[inpt.var]["t1"]["data"][inpt.var] = lwp_t1

    return


def read_lw_down():
    """
    Reads and processes longwave downward radiation ("lw_down") data from CARRA, ERA5, and THAAO.
    Filters out negative values by setting them to NaN, then applies radiation conversion factors.
    """
    # --- CARRA ---
    read_rean("lw_down", "c")
    lw_down_c = inpt.extr["lw_down"]["c"]["data"][inpt.var]
    lw_down_c[lw_down_c < 0.] = np.nan
    lw_down_c /= inpt.var_dict["c"]["rad_conv_factor"]
    inpt.extr["lw_down"]["c"]["data"][inpt.var] = lw_down_c

    # --- ERA5 ---
    read_rean("lw_down", "e")
    lw_down_e = inpt.extr["lw_down"]["e"]["data"][inpt.var]
    lw_down_e[lw_down_e < 0.] = np.nan
    lw_down_e /= inpt.var_dict["e"]["rad_conv_factor"]
    inpt.extr["lw_down"]["e"]["data"][inpt.var] = lw_down_e

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_rad("lw_down")
        lw_down_t = inpt.extr["lw_down"]["t"]["data"]
        lw_down_t[lw_down_t < 0.] = np.nan

    return


def read_lw_up():
    """
    Reads and processes longwave upwelling radiation data from multiple sources (CARRA, ERA5, THAAO).
    Performs unit conversions, calculates upward LW radiation from net/downward components,
    and handles invalid values.
    """
    read_lw_down()

    # --- CARRA ---
    read_rean("lw_net", "c")
    lw_net_c = inpt.extr["lw_net"]["c"]["data"][inpt.var]
    lw_net_c /= inpt.var_dict["c"]["rad_conv_factor"]

    lw_down_c = inpt.extr["lw_down"]["c"]["data"][inpt.var]
    lw_up_c = lw_down_c - lw_net_c
    lw_up_c[lw_up_c < 0.] = np.nan
    inpt.extr["lw_up"]["c"]["data"][inpt.var] = lw_up_c.to_frame(name="lw_up")

    # --- ERA5 ---
    read_rean("lw_net", "e")
    lw_net_e = inpt.extr["lw_net"]["e"]["data"][inpt.var]
    lw_net_e /= inpt.var_dict["e"]["rad_conv_factor"]

    lw_down_e = inpt.extr["lw_down"]["e"]["data"][inpt.var]
    lw_up_e = lw_down_e - lw_net_e
    lw_up_e[lw_up_e < 0.] = np.nan
    inpt.extr["lw_up"]["e"]["data"][inpt.var] = lw_up_e.to_frame(name="lw_up")

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_rad("lw_up")
        lw_up_t = inpt.extr["lw_up"]["t"]["data"][inpt.var]
        lw_up_t[lw_up_t < 0.] = np.nan

    return


def read_precip():
    """
    Reads and processes precipitation data from CARRA, ERA5, and THAAO2 datasets.
    Scales ERA5 precipitation from meters to millimeters. Modifies `inpt` in place.
    """
    # --- CARRA ---
    read_rean(inpt.var, "c")

    # --- ERA5 ---
    read_rean(inpt.var, "e")
    precip_e = inpt.extr[inpt.var]["e"]["data"][inpt.var]
    precip_e *= 1000.  # Convert from meters to mm

    # --- THAAO2 ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_aws_ecapac(inpt.var)

    return


def read_rh():
    """
    Reads and processes relative humidity data from CARRA, ERA5, and THAAO2 datasets.
    If RH is not directly available, it is calculated from temperature and dew point.

    :raises KeyError: if data keys are missing.
    :raises ValueError: if unexpected data structure issues arise.
    """
    # --- CARRA ---
    read_rean(inpt.var, "c")

    # --- ERA5 ---
    read_rean("dewpt", "e")
    read_rean("temp", "e")
    tls.calc_rh_from_tdp()  # Compute RH from dew point and temp

    # --- THAAO2 ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_weather(inpt.var)
        rd_ft.read_thaao_aws_ecapac(inpt.var)
        # This call is repeated in your original code — was that intentional?
        # Removing the duplicate unless you confirm it's necessary.
        # rd_ft.read_thaao_weather(inpt.var)

    return


def read_surf_pres():
    """
    Reads surface pressure data from CARRA, ERA5, THAAO, and THAAO2.
    Converts units, filters out invalid data, and handles known corrupted periods.
    Modifies `inpt` in-place.
    """
    # --- CARRA ---
    read_rean(inpt.var, "c")
    pres_c = inpt.extr[inpt.var]["c"]["data"][inpt.var]
    pres_c /= 100.
    pres_c[pres_c <= 900.] = np.nan

    # --- ERA5 ---
    read_rean(inpt.var, "e")
    pres_e = inpt.extr[inpt.var]["e"]["data"][inpt.var]
    pres_e /= 100.
    pres_e[pres_e <= 900.] = np.nan

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_weather(inpt.var)
        pres_t = inpt.extr[inpt.var]["t"]["data"][inpt.var]
        pres_t[pres_t <= 900.] = np.nan
        pres_t.loc["2021-10-11 00:00:00":"2021-10-19 00:00:00"] = np.nan
        pres_t.loc["2024-04-26 00:00:00":"2024-05-04 00:00:00"] = np.nan

        # --- THAAO2 ---
        rd_ft.read_thaao_aws_ecapac(inpt.var)

    return


def read_sw_down():
    """
    Reads and processes shortwave downward radiation data from CARRA, ERA5, ERA5-LAND, and THAAO.
    Negative values are set to NaN, and values are scaled using radiation conversion factors.
    """
    # --- CARRA ---
    read_rean("sw_down", "c")
    sw_down_c = inpt.extr["sw_down"]["c"]["data"][inpt.var]
    sw_down_c[sw_down_c < 0.] = np.nan
    sw_down_c /= inpt.var_dict["c"]["rad_conv_factor"]

    # --- ERA5 ---
    read_rean("sw_down", "e")
    sw_down_e = inpt.extr["sw_down"]["e"]["data"][inpt.var]
    sw_down_e[sw_down_e < 0.] = np.nan
    sw_down_e /= inpt.var_dict["e"]["rad_conv_factor"]

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_rad("sw_down")
        sw_down_t = inpt.extr["sw_down"]["t"]["data"][inpt.var]
        sw_down_t[sw_down_t < 0.] = np.nan

    return


def read_sw_up():
    """
    Reads and processes shortwave upward radiation data from CARRA, ERA5, ERA5-LAND, and THAAO.
    Applies unit conversions, calculates upwelling radiation, and filters invalid values.
    Modifies `inpt` in-place.
    """
    read_sw_down()

    # --- CARRA ---
    read_rean("sw_net", "c")

    sw_net_c = inpt.extr["sw_net"]["c"]["data"][inpt.var]
    sw_net_c /= inpt.var_dict["c"]["rad_conv_factor"]

    sw_down_c = inpt.extr["sw_down"]["c"]["data"][inpt.var]
    sw_up_c = sw_down_c - sw_net_c
    sw_up_c[sw_up_c < 0.] = np.nan

    inpt.extr["sw_up"]["c"]["data"][inpt.var] = sw_up_c
    del inpt.extr["sw_net"]["c"]["data"][inpt.var]

    # --- ERA5 ---
    read_rean("sw_net", "e")

    sw_net_e = inpt.extr["sw_net"]["e"]["data"][inpt.var]
    sw_net_e /= inpt.var_dict["e"]["rad_conv_factor"][inpt.var]

    sw_down_e = inpt.extr["sw_down"]["e"]["data"][inpt.var]
    sw_up_e = sw_down_e - sw_net_e
    sw_up_e[sw_up_e < 0.] = np.nan

    inpt.extr["sw_up"]["e"]["data"][inpt.var] = sw_up_e
    del inpt.extr["sw_net"]["e"]["data"]

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_rad("sw_up")
        sw_up_t = inpt.extr["sw_up"]["t"]["data"][inpt.var]
        sw_up_t[sw_up_t < 0.] = np.nan

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
    inpt.extr[inpt.var]["e"]["data"][inpt.var] *= 100.0

    # THAAO
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_ceilometer(inpt.var)

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
    inpt.extr[inpt.var]["c"]["data"][inpt.var] -= 273.15

    # ERA5
    read_rean(inpt.var, "e")
    inpt.extr[inpt.var]["e"]["data"][inpt.var] -= 273.15

    # THAAO
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_weather(inpt.var)
        inpt.extr[inpt.var]["t"]["data"][inpt.var] -= 273.15

        # THAAO2
        rd_ft.read_thaao_aws_ecapac(inpt.var)

    if inpt.datasets['Villum']['switch']:
        rd_fv.read_villum_weather(inpt.var)

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
        inpt.extr["windu"]["e"]["data"][inpt.var].values * units("m/s"),
        inpt.extr["windv"]["e"]["data"][inpt.var].values * units("m/s"))
    inpt.extr["winds"]["e"]["data"] = pd.DataFrame(
        index=inpt.extr["windu"]["e"]["data"][inpt.var].index, data=e_ws.magnitude, columns=["winds"])

    e_wd = wind_direction(
        inpt.extr["windu"]["e"]["data"][inpt.var].values * units("m/s"),
        inpt.extr["windv"]["e"]["data"][inpt.var].values * units("m/s"))
    inpt.extr["windd"]["e"]["data"][inpt.var] = pd.DataFrame(
        index=inpt.extr["windu"]["e"]["data"][inpt.var].index, data=e_wd.magnitude, columns=["windd"])

    # THAAO2
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_aws_ecapac("winds")
        rd_ft.read_aws_ecapac("windd")

    return


# def calc_rad_acc_era5_land(vr):
#     """
#     Calculates instantaneous radiation accumulation from daily accumulated ERA5-LAND data.

#     This function processes the daily-accumulated radiation data for ERA5-LAND by
#     calculating the difference between consecutive timesteps. It also handles specific
#     timesteps (e.g., at 0100 hours), ensuring the data integrity for further analysis.

#     :param vr: The key or identifier in the `inpt.extr` data dictionary used to locate
#         the radiation dataset for processing.
#     :type vr: str
#     :return: None
#     """
#     # calculating instantaneous as difference with previous timestep
#     inpt.extr[vr]["l"]["data_diff"] = inpt.extr[vr]["l"]["data"][vr].diff()
#     # dropping value at 0100 which does not need any subtraction (it is the first of the day)
#     inpt.extr[vr]["l"]["data_diff"] = inpt.extr[vr]["l"]["data_diff"][inpt.extr[vr]
#                                                                       ["l"]["data_diff"].index.hour != 1]
#     # selecting original value at 0100
#     orig_filtered_data = inpt.extr[vr]["l"]["data"][inpt.extr[vr]
#                                                     ["l"]["data"].index.hour == 1]
#     # appending original value at 0100
#     inpt.extr[vr]["l"]["data_diff"] = pd.concat(
#         [inpt.extr[vr]["l"]["data_diff"], orig_filtered_data]).sort_index()
#     inpt.extr[vr]["l"]["data"] = inpt.extr[vr]["l"]["data_diff"]

#     print("ERA5-LAND data for radiation corrected because they are values daily-accumulated!")
#     return


def read():
    readers = {
        "alb": read_alb,
        "cbh": read_cbh,
        "lwp": read_lwp,
        "lw_down": read_lw_down,
        "lw_up": read_lw_up,
        "precip": read_precip,
        "rh": read_rh,
        "surf_pres": read_surf_pres,
        "sw_down": read_sw_down,
        "sw_up": read_sw_up,
        "tcc": read_tcc,
        "temp": read_temp,
        "winds": read_wind,
        "windd": read_wind,
    }
    reader_func = readers.get(inpt.var)
    if reader_func is None:
        raise ValueError(
            f"No reader function defined for variable '{inpt.var}'")
    return reader_func()
