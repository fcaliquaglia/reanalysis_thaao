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

    # Process missing yearly files if needed
    for year in inpt.years:
        output_file = f"{inpt.extr[vr][dataset_type]['fn']}{inpt.location}_{year}.parquet"
        output_path = os.path.join(
            inpt.basefol[dataset_type]['processed'], output_file)
    
        if not os.path.exists(output_path):
            tls.process_rean(vr, dataset_type, year)
    
    # Read all yearly parquet files and concatenate into a single DataFrame
    data_all = []
    
    for year in inpt.years:
        input_file = f"{inpt.extr[vr][dataset_type]['fn']}{inpt.location}_{year}.parquet"
        input_path = os.path.join(
            inpt.basefol[dataset_type]['processed'], input_file)
        try:
            data_tmp = pd.read_parquet(input_path)
            print(f"Loaded {input_path}")
            if not data_tmp.empty:
                data_all.append(data_tmp)
        except FileNotFoundError as e:
            print(f"File not found: {input_path} ({e})")
            continue
    
    # Combine all data if any was loaded
    if data_all:
        data_all = pd.concat(data_all)
    else:
        data_all = pd.DataFrame()

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
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA ---
    read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    var_dict["c"]["data"][vr] /= 100.
    # var_dict["c"]["data"].loc[var_dict["c"]["data"][vr] <= 0., vr] = np.nan

    # --- ERA5 ---
    read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    # var_dict["e"]["data"].loc[var_dict["e"]["data"][vr] <= 0., vr] = np.nan

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    return


def read_cbh():
    """
    Reads cloud base height (CBH) data from multiple sources and processes it
    based on the specified input variable. Supports data from CARRA, ERA5, and THAAO ceilometer.

    :raises ValueError: If the specified input variable is invalid or unsupported.
    :return: None
    """
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA ---
    read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

    # --- ERA5 ---
    read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_ceilometer(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    return


def read_lwp():
    """
    Reads cloud liquid water path (LWP) data from CARRA, ERA5, and THAAO1,
    applies cleaning conditions (setting values < 0.01 to NaN), and updates datasets.
    """
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA ---
    read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    lwp_c = var_dict["c"]["data"][vr]
    # Uncomment below line if you want to filter small values
    # lwp_c[lwp_c < 0.01] = np.nan
    var_dict["c"]["data"][vr] = lwp_c

    # --- ERA5 ---
    read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    lwp_e = var_dict["e"]["data"][vr]
    lwp_e[lwp_e < 0.01] = np.nan
    var_dict["e"]["data"][vr] = lwp_e

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_hatpro(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        lwp_t1 = var_dict["t1"]["data"][vr]
        lwp_t1[lwp_t1 < 0.01] = np.nan
        var_dict["t1"]["data"][vr] = lwp_t1

    return


def read_lw_down():
    """
    Reads and processes longwave downward radiation ("lw_down") data from CARRA, ERA5, and THAAO.
    Filters out negative values by setting them to NaN, then applies radiation conversion factors.
    """

    # --- CARRA ---
    vr = "lw_down"
    var_dict = inpt.extr[vr]
    read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    lw_down_c = var_dict["c"]["data"][vr].mask(
        var_dict["c"]["data"][vr] < 0., np.nan)
    lw_down_c /= inpt.var_dict["c"]["rad_conv_factor"]
    var_dict["c"]["data"][vr] = lw_down_c

    # --- ERA5 ---
    vr = "lw_down"
    var_dict = inpt.extr[vr]
    read_rean("lw_down", "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    lw_down_e = var_dict["e"]["data"][vr].mask(
        var_dict["e"]["data"][vr] < 0., np.nan)
    lw_down_e /= inpt.var_dict["e"]["rad_conv_factor"]
    var_dict["e"]["data"][vr] = lw_down_e

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_rad("lw_down")
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        lw_down_t = var_dict["t"]["data"][vr].mask(
            var_dict["t"]["data"][vr] < 0., np.nan)
        var_dict["t"]["data"][vr] = lw_down_t

    return


def read_lw_up():
    """
    Reads and processes longwave upwelling radiation data from multiple sources (CARRA, ERA5, THAAO).
    Performs unit conversions, calculates upward LW radiation from net/downward components,
    and handles invalid values.
    """

    # --- CARRA ---
    vr = "lw_net"
    var_dict = inpt.extr[vr]
    read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    lw_net_c = var_dict["c"]["data"][vr]
    lw_net_c /= inpt.var_dict["c"]["rad_conv_factor"]

    vr = 'lw_down'
    var_dict = inpt.extr[vr]
    lw_down_c = var_dict["c"]["data"][vr]
    read_lw_down()

    vr = "lw_up"
    var_dict = inpt.extr[vr]
    lw_up_c = lw_down_c - lw_net_c
    var_dict["data"] = lw_up_c.mask(lw_up_c < 0., np.nan)

    # --- ERA5 ---
    vr = "lw_net"
    var_dict = inpt.extr[vr]
    read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    lw_net_e = var_dict["e"]["data"][vr]
    lw_net_e /= inpt.var_dict["e"]["rad_conv_factor"]

    vr = 'lw_down'
    var_dict = inpt.extr[vr]
    lw_down_e = var_dict["e"]["data"][vr]

    vr = "lw_up"
    var_dict = inpt.extr[vr]
    lw_up_e = lw_down_e - lw_net_e
    var_dict["e"]["data"] = lw_up_e.mask(lw_up_e < 0., np.nan)

    # --- THAAO ---
    vr = "lw_up"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        lw_up_t = var_dict["t"]["data"]
        lw_up_t[vr] = lw_up_t[vr].mask(lw_up_t[vr] < 0., np.nan)
        var_dict["t"]["data"] = lw_up_t

    return


def read_precip():
    """
    Reads and processes precipitation data from CARRA, ERA5, and THAAO2 datasets.
    Scales ERA5 precipitation from meters to millimeters. Modifies `inpt` in place.
    """
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA ---
    read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

    # --- ERA5 ---
    read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    precip_e = var_dict["e"]["data"][vr]
    precip_e *= 1000.  # Convert from meters to mm

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_aws_ecapac(vr)
        var_dict["t2"]["data"], _ = tls.check_empty_df(
            var_dict["t2"]["data"], vr)

    return


def read_rh():
    """
    Reads and processes relative humidity data from CARRA, ERA5, and THAAO2 datasets.
    If RH is not directly available, it is calculated from temperature and dew point.

    :raises KeyError: if data keys are missing.
    :raises ValueError: if unexpected data structure issues arise.
    """
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA ---
    read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

    # --- ERA5 ---
    read_rean("dewpt", "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    read_rean("temp", "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    tls.calc_rh_from_tdp()  # Compute RH from dew point and temp

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        rd_ft.read_thaao_aws_ecapac(vr)
        var_dict["t2"]["data"], _ = tls.check_empty_df(
            var_dict["t2"]["data"], vr)

    # --- Villum ---
    if inpt.datasets['Villum']['switch']:
        rd_fv.read_villum_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-A ---
    if inpt.datasets['Sigma-A']['switch']:
        rd_fv.read_villum_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-B ---
    if inpt.datasets['Sigma-B']['switch']:
        rd_fv.read_villum_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Alert ---
    if inpt.datasets['Alert']['switch']:
        rd_fv.read_villum_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Summit ---
    if inpt.datasets['Summit']['switch']:
        rd_fv.read_villum_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    return


def read_surf_pres():
    """
    Reads surface pressure data from CARRA, ERA5, THAAO, and THAAO2.
    Converts units, filters out invalid data, and handles known corrupted periods.
    Modifies `inpt` in-place.
    """
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA ---
    read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    pres_c = var_dict["c"]["data"][vr]
    var_dict["c"]["data"][vr] /= 100.
    var_dict["c"]["data"].loc[var_dict["c"]["data"][vr] <= 900., vr] = np.nan

    # --- ERA5 ---
    read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    var_dict["e"]["data"][vr] /= 100.
    var_dict["e"]["data"].loc[var_dict["e"]["data"][vr] <= 900., vr] = np.nan

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        pres_t = var_dict["t"]["data"][vr]
        pres_t[pres_t <= 900.] = np.nan
        pres_t.loc["2021-10-11 00:00:00":"2021-10-19 00:00:00"] = np.nan
        pres_t.loc["2024-04-26 00:00:00":"2024-05-04 00:00:00"] = np.nan
        var_dict["t"]["data"][vr] = pres_t

        rd_ft.read_thaao_aws_ecapac(vr)
        var_dict["t2"]["data"], _ = tls.check_empty_df(
            var_dict["t2"]["data"], vr)

    # --- Villum ---
    if inpt.datasets['Villum']['switch']:
        rd_fv.read_villum_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-A ---
    if inpt.datasets['Sigma-A']['switch']:
        print('No data available for Sigma-A')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-B ---
    if inpt.datasets['Sigma-B']['switch']:
        print('No data available for Sigma-B')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Alert ---
    if inpt.datasets['Alert']['switch']:
        print('No data available for Alert')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Summit ---
    if inpt.datasets['Summit']['switch']:
        print('No data available for Summit')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Buoys ---
    if inpt.datasets['buoys']['switch']:
        data_all=pd.DataFrame()
        for y in inpt.years:
            path = os.path.join('txt_locations', f"{inpt.location}_loc.txt")
            data_tmp = pd.read_csv(path)
            data_all = pd.concat([data_all, data_tmp])
        data_all['time'] = pd.to_datetime(data_all['time'], errors='coerce')
        data_all = data_all.set_index('time')
        var_dict["t"]["data"] = data_all
        var_dict["t"]["data"], _ = tls.check_empty_df(var_dict["t"]["data"], vr)
    return


def read_sw_down():
    """
    Reads and processes shortwave downward radiation data from CARRA, ERA5, ERA5-LAND, and THAAO.
    Negative values are set to NaN, and values are scaled using radiation conversion factors.
    """

    # --- CARRA ---
    vr = "sw_down"
    var_dict = inpt.extr[vr]
    read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

    sw_down_c = var_dict["c"]["data"][vr].mask(
        var_dict["c"]["data"][vr] < 0., np.nan)
    sw_down_c /= inpt.var_dict["c"]["rad_conv_factor"]
    var_dict["c"]["data"][vr] = sw_down_c

    # --- ERA5 ---
    vr = "sw_down"
    var_dict = inpt.extr[vr]
    read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)

    sw_down_e = var_dict["e"]["data"][vr].mask(
        var_dict["e"]["data"][vr] < 0., np.nan)
    sw_down_e /= inpt.var_dict["e"]["rad_conv_factor"]
    var_dict["e"]["data"][vr] = sw_down_e

    # --- THAAO ---
    vr = "sw_down"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_rad(vr)
        sw_down_t = var_dict["t"]["data"][vr].mask(
            var_dict["t"]["data"][vr] < 0., np.nan)
        var_dict["t"]["data"][vr] = sw_down_c
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Villum ---
    if inpt.datasets['Villum']['switch']:
        rd_fv.read_villum_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-A ---
    if inpt.datasets['Sigma-A']['switch']:
        print('No data available for Sigma-A')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-B ---
    if inpt.datasets['Sigma-B']['switch']:
        print('No data available for Sigma-B')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Alert ---
    if inpt.datasets['Alert']['switch']:
        print('No data available for Alert')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Summit ---
    if inpt.datasets['Summit']['switch']:
        print('No data available for Summit')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Buoys ---
    if inpt.datasets['buoys']['switch']:
        data_all=pd.DataFrame()
        for y in inpt.years:
            path = os.path.join('txt_locations', f"{inpt.location}_loc.txt")
            data_tmp = pd.read_csv(path)
            data_all = pd.concat([data_all, data_tmp])
        data_all['time'] = pd.to_datetime(data_all['time'], errors='coerce')
        data_all = data_all.set_index('time')
        var_dict["t"]["data"] = data_all
        var_dict["t"]["data"], _ = tls.check_empty_df(var_dict["t"]["data"], vr)
    return


def read_sw_up():
    """
    Reads and processes shortwave upward radiation data from CARRA, ERA5, ERA5-LAND, and THAAO.
    Applies unit conversions, calculates upwelling radiation, and filters invalid values.
    Modifies `inpt` in-place.
    """

    # --- CARRA ---
    vr = "sw_net"
    var_dict = inpt.extr[vr]
    read_rean("sw_net", "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

    sw_net_c = var_dict["c"]["data"][vr]
    sw_net_c /= inpt.var_dict["c"]["rad_conv_factor"]

    vr = "sw_down"
    var_dict = inpt.extr[vr]
    read_sw_down()
    sw_down_c = var_dict["c"]["data"][vr]

    vr = "sw_up"
    var_dict = inpt.extr[vr]
    sw_up_c = sw_down_c - sw_net_c
    var_dict["c"]["data"] = sw_up_c.mask(sw_up_c < 0., np.nan)
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

    # --- ERA5 ---
    vr = "sw_net"
    var_dict = inpt.extr[vr]
    read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)

    sw_net_e = var_dict["e"]["data"][vr]
    sw_net_e /= inpt.var_dict["e"]["rad_conv_factor"]

    vr = "sw_down"
    var_dict = inpt.extr[vr]
    sw_down_e = var_dict["e"]["data"][vr]

    vr = "sw_up"
    var_dict = inpt.extr[vr]
    sw_up_e = sw_down_e - sw_net_e
    var_dict["e"]["data"] = sw_up_e.mask(sw_up_e < 0., np.nan)
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)

    # --- THAAO ---
    vr = "sw_up"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_rad(vr)
        sw_up_t = var_dict["t"]["data"][vr]
        var_dict["t"]["data"][vr] = sw_up_t.mask(sw_up_t < 0., np.nan)
        var_dict["t"]["data"] = sw_up_t
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        
    # --- Buoys ---
    if inpt.datasets['buoys']['switch']:
        data_all=pd.DataFrame()
        for y in inpt.years:
            path = os.path.join('txt_locations', f"{inpt.location}_loc.txt")
            data_tmp = pd.read_csv(path)
            data_all = pd.concat([data_all, data_tmp])
        data_all['time'] = pd.to_datetime(data_all['time'], errors='coerce')
        data_all = data_all.set_index('time')
        var_dict["t"]["data"] = data_all
        var_dict["t"]["data"], _ = tls.check_empty_df(var_dict["t"]["data"], vr)

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
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA ---
    read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"])

    # --- ERA5 ---
    read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"])
    var_dict["e"]["data"][vr] *= 100.0

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_ceilometer(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    return


def read_temp():
    """
    Transforms temperature data from various datasets to Celsius and retrieves
    reformatted sources into a structured container. The function processes data
    from the CARRA, ERA5, ERA5-LAND, THAAO, and THAAO2 datasets, converting values
    from Kelvin to Celsius for consistency.

    :return: None
    """
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA ---
    read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    var_dict["c"]["data"][vr] -= 273.15

    # --- ERA5 ---
    read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    var_dict["e"]["data"][vr] -= 273.15

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_weather(vr)
        var_dict["t"]["data"][vr] -= 273.15
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        rd_ft.read_thaao_aws_ecapac(vr)
        var_dict["t2"]["data"], _ = tls.check_empty_df(
            var_dict["t2"]["data"], vr)

    # --- Villum ---
    if inpt.datasets['Villum']['switch']:
        rd_fv.read_villum_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-A ---
    if inpt.datasets['Sigma-A']['switch']:
        print('No data available for Sigma-A')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-B ---
    if inpt.datasets['Sigma-B']['switch']:
        print('No data available for Sigma-B')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Alert ---
    if inpt.datasets['Alert']['switch']:
        print('No data available for Alert')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Summit ---
    if inpt.datasets['Summit']['switch']:
        print('No data available for Summit')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Buoys ---
    if inpt.datasets['buoys']['switch']:
        data_all=pd.DataFrame()
        for y in inpt.years:
            path = os.path.join('txt_locations', f"{inpt.location}_loc.txt")
            data_tmp = pd.read_csv(path)
            data_all = pd.concat([data_all, data_tmp])
        data_all['time'] = pd.to_datetime(data_all['time'], errors='coerce')
        data_all = data_all.set_index('time')
        var_dict["t"]["data"][vr] = data_all.mask(data_all ==0.0, np.nan)
        var_dict["t"]["data"] = data_all
        var_dict["t"]["data"], _ = tls.check_empty_df(var_dict["t"]["data"], vr)
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

    # --- CARRA ---
    vr = "winds"
    var_dict = inpt.extr[vr]
    read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    vr = "windd"
    var_dict = inpt.extr[vr]
    read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

    # --- ERA5 ---
    vr = "windu"
    var_dict = inpt.extr[vr]
    read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)

    vr = "windv"
    var_dict = inpt.extr[vr]
    read_rean("vr", "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    e_ws = wind_speed(
        inpt.extr["windu"]["e"]["data"][vr].values * units("m/s"),
        inpt.extr["windv"]["e"]["data"][vr].values * units("m/s"))
    inpt.extr["winds"]["e"]["data"] = pd.DataFrame(
        index=inpt.extr["windu"]["e"]["data"][vr].index, data=e_ws.magnitude, columns=["winds"])

    e_wd = wind_direction(
        inpt.extr["windu"]["e"]["data"][vr].values * units("m/s"),
        inpt.extr["windv"]["e"]["data"][vr].values * units("m/s"))
    inpt.extr["windd"]["e"]["data"][vr] = pd.DataFrame(
        index=inpt.extr["windu"]["e"]["data"][vr].index, data=e_wd.magnitude, columns=["windd"])

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_aws_ecapac("winds")
        var_dict["t2"]["data"], _ = tls.check_empty_df(
            var_dict["t2"]["data"], vr)
        rd_ft.read_aws_ecapac("windd")
        var_dict["t2"]["data"], _ = tls.check_empty_df(
            var_dict["t2"]["data"], vr)

    # --- Villum ---
    if inpt.datasets['Villum']['switch']:
        rd_fv.read_villum_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-A ---
    if inpt.datasets['Sigma-A']['switch']:
        print('No data available for Sigma-A')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-B ---
    if inpt.datasets['Sigma-B']['switch']:
        print('No data available for Sigma-B')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Alert ---
    if inpt.datasets['Alert']['switch']:
        print('No data available for Alert')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Summit ---
    if inpt.datasets['Summit']['switch']:
        print('No data available for Summit')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

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
