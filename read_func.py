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
    """

    location = next(
        (v['fn'] for k, v in inpt.datasets.items() if v.get('switch')), None)

    # First process missing files
    for year in inpt.years:
        output_file = f"{inpt.extr[vr][dataset_type]['fn']}{location}_{year}.parquet"
        output_path = os.path.join(
            inpt.basefol[dataset_type]['processed'], output_file)
        if not os.path.exists(output_path):
            tls.process_rean(vr, dataset_type, year, location)

    # Then read all processed files into a single DataFrame
    data_all = pd.DataFrame()
    for year in inpt.years:
        input_file = f"{inpt.extr[vr][dataset_type]['fn']}{location}_{year}.parquet"
        input_path = os.path.join(
            inpt.basefol[dataset_type]['processed'], input_file)
        try:
            data_tmp = pd.read_parquet(input_path)
        except FileNotFoundError as e:
            print(e)
            continue
        data_all = pd.concat([data_all, data_tmp])

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
    inpt.extr[inpt.var]["c"]["data"][inpt.var] = inpt.extr[inpt.var]["c"]["data"][inpt.var] / 100.
    inpt.extr[inpt.var]["c"]["data"][inpt.var][inpt.extr[inpt.var]
                                               ["c"]["data"][inpt.var] <= 0.] = np.nan

    # ERA5
    read_rean(inpt.var, "e")
    inpt.extr[inpt.var]["c"]["data"][inpt.var][inpt.extr[inpt.var]
                                               ["c"]["data"][inpt.var] <= 0.] = np.nan

    # THAAO
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_rad(inpt.var)

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
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_ceilometer(inpt.var)

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
    inpt.extr[inpt.var]["c"]["data"][inpt.var] = inpt.extr[inpt.var]["c"]["data"][inpt.var]
    # inpt.extr[inpt.var]["c"]["data"][inpt.var][inpt.extr[inpt.var]["c"]["data"][inpt.var] < 0.01] = np.nan
    # c[c < 15] = 0

    # ERA5
    read_rean(inpt.var, "e")
    inpt.extr[inpt.var]["e"]["data"] = inpt.extr[inpt.var]["e"]["data"]
    inpt.extr[inpt.var]["e"]["data"][inpt.var][inpt.extr[inpt.var]
                                               ["e"]["data"][inpt.var] < 0.01] = np.nan
    # e[e < 15] = 0

    # THAAO1
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_hatpro(inpt.var)
        inpt.extr[inpt.var]["t1"]["data"][inpt.var][inpt.extr[inpt.var]
                                                    ["t1"]["data"][inpt.var] < 0.01] = np.nan
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
    inpt.extr[inpt.var]["e"]["data"][inpt.var] = inpt.extr[inpt.var]["e"]["data"][inpt.var].values * \
        1000.

    # THAAO2
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_aws_ecapac(inpt.var)

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
    inpt.extr["lw_down"]["c"]["data"][inpt.var][inpt.extr["lw_down"]
                                                ["c"]["data"][inpt.var] < 0.] = np.nan
    inpt.extr["lw_down"]["c"]["data"][inpt.var] = inpt.extr["lw_down"]["c"]["data"][inpt.var] / \
        inpt.var_dict["c"]["rad_conv_factor"]

    # ERA5
    read_rean("lw_down", "e")
    inpt.extr["lw_down"]["e"]["data"][inpt.var][inpt.extr["lw_down"]
                                                ["e"]["data"][inpt.var] < 0.] = np.nan
    inpt.extr["lw_down"]["e"]["data"][inpt.var] = inpt.extr["lw_down"]["e"]["data"][inpt.var] / \
        inpt.var_dict["e"]["rad_conv_factor"]

    # THAAO
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_rad("lw_down")
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
    inpt.extr["sw_down"]["c"]["data"][inpt.var][inpt.extr["sw_down"]
                                                ["c"]["data"][inpt.var] < 0.] = np.nan
    inpt.extr["sw_down"]["c"]["data"][inpt.var] = inpt.extr["sw_down"]["c"]["data"][inpt.var] / \
        inpt.var_dict["c"]["rad_conv_factor"]

    # ERA5
    read_rean("sw_down", "e")
    inpt.extr["sw_down"]["e"]["data"][inpt.var][inpt.extr["sw_down"]
                                                ["e"]["data"][inpt.var] < 0.] = np.nan
    inpt.extr["sw_down"]["e"]["data"][inpt.var] = inpt.extr["sw_down"]["e"]["data"][inpt.var] / \
        inpt.var_dict["e"]["rad_conv_factor"]

    # THAAO
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_rad("sw_down")
        inpt.extr["sw_down"]["t"]["data"][inpt.var][inpt.extr["sw_down"]
                                                    ["t"]["data"][inpt.var] < 0.] = np.nan
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
    inpt.extr["lw_net"]["c"]["data"][inpt.var] = inpt.extr["lw_net"]["c"]["data"][inpt.var] / \
        inpt.var_dict["c"]["rad_conv_factor"]
    inpt.extr["lw_up"]["c"]["data"][inpt.var] = pd.DataFrame(
        index=inpt.extr["lw_down"]["c"]["data"][inpt.var].index,
        data=inpt.extr["lw_down"]["c"]["data"][inpt.var].values - inpt.extr["lw_net"]["c"]["data"][inpt.var].values, columns=["lw_up"])
    inpt.extr["lw_up"]["c"]["data"][inpt.var][inpt.extr["lw_up"]
                                              ["c"]["data"][inpt.var] < 0.] = np.nan
    # del inpt.extr["lw_net"]["c"]["data"]

    # ERA5
    read_rean("lw_net", "e")
    inpt.extr["lw_net"]["e"]["data"][inpt.var] = inpt.extr["lw_net"]["e"]["data"][inpt.var] / \
        inpt.var_dict["e"]["rad_conv_factor"]
    inpt.extr["lw_up"]["e"]["data"][inpt.var] = pd.DataFrame(
        index=inpt.extr["lw_down"]["e"]["data"][inpt.var].index,
        data=inpt.extr["lw_down"]["e"]["data"][inpt.var].values - inpt.extr["lw_net"]["e"]["data"][inpt.var].values, columns=["lw_up"])
    inpt.extr["lw_up"]["e"]["data"][inpt.var][inpt.var][inpt.extr["lw_up"]
                                                        ["e"]["data"][inpt.var] < 0.] = np.nan
    # del inpt.extr["lw_net"]["e"]["data"]

    # THAAO
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_rad("lw_up")
        inpt.extr["lw_up"]["t"]["data"][inpt.extr["lw_up"]
                                        ["t"]["data"][inpt.var] < 0.] = np.nan

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
    inpt.extr["sw_net"]["c"]["data"][inpt.var] = inpt.extr["sw_net"]["c"]["data"][inpt.var] / \
        inpt.var_dict["c"]["rad_conv_factor"]
    inpt.extr["sw_up"]["c"]["data"][inpt.var] = pd.DataFrame(
        index=inpt.extr["sw_down"]["c"]["data"][inpt.var].index,
        data=inpt.extr["sw_down"]["c"]["data"][inpt.var].values - inpt.extr["sw_net"]["c"]["data"][inpt.var].values, columns=["sw_up"])
    inpt.extr["sw_up"]["c"]["data"][inpt.var][inpt.extr["sw_up"]
                                              ["c"]["data"][inpt.var] < 0.] = np.nan
    del inpt.extr["sw_net"]["c"]["data"][inpt.var]

    # ERA5
    read_rean("sw_net", "e")
    inpt.extr["sw_net"]["e"]["data"][inpt.var] = inpt.extr["sw_net"]["e"]["data"][inpt.var] / \
        inpt.var_dict["e"]["rad_conv_factor"][inpt.var]
    inpt.extr["sw_up"]["e"]["data"][inpt.var] = pd.DataFrame(
        index=inpt.extr["sw_down"]["e"]["data"][inpt.var].index,
        data=inpt.extr["sw_down"]["e"]["data"][inpt.var].values - inpt.extr["sw_net"]["e"]["data"][inpt.var].values, columns=["sw_up"])
    inpt.extr["sw_up"]["e"]["data"][inpt.var][inpt.extr["sw_up"]
                                              ["e"]["data"][inpt.var] < 0.] = np.nan
    del inpt.extr["sw_net"]["e"]["data"]

    # THAAO
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_rad("sw_up")
        inpt.extr["sw_up"]["t"]["data"][inpt.var][inpt.extr["sw_up"]
                                                  ["t"]["data"][inpt.var] < 0.] = np.nan

    return


def read_rh():
    """
    Reads and processes relative humidity data from various sources including
    CARRA, ERA5, and THAAO2-based weather datasets. This function
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
    if inpt.extr[inpt.var]["e"]["data"].empty:
        read_rean("temp", "e")
    tls.calc_rh_from_tdp()

    # THAAO2
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_weather(inpt.var)
        rd_ft.read_thaao_aws_ecapac(inpt.var)
        rd_ft.read_thaao_weather(inpt.var)

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
    inpt.extr[inpt.var]["c"]["data"][inpt.var] = inpt.extr[inpt.var]["c"]["data"][inpt.var] / 100.
    inpt.extr[inpt.var]["c"]["data"][inpt.var][inpt.extr[inpt.var]
                                               ["c"]["data"][inpt.var] <= 900] = np.nan

    # ERA5
    read_rean(inpt.var, "e")
    inpt.extr[inpt.var]["e"]["data"][inpt.var] = inpt.extr[inpt.var]["e"]["data"][inpt.var] / 100.
    inpt.extr[inpt.var]["e"]["data"][inpt.var][inpt.extr[inpt.var]
                                               ["e"]["data"][inpt.var] <= 900] = np.nan

    # THAAO
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_weather(inpt.var)
        inpt.extr[inpt.var]["t"]["data"][inpt.var][inpt.extr[inpt.var]
                                                   ["t"]["data"][inpt.var] <= 900] = np.nan
        inpt.extr[inpt.var]["t"]["data"][inpt.var].loc["2021-10-11 00:00:00":"2021-10-19 00:00:00"] = np.nan
        inpt.extr[inpt.var]["t"]["data"][inpt.var].loc["2024-4-26 00:00:00":"2024-5-4 00:00:00"] = np.nan

        # THAAO2
        rd_ft.read_thaao_aws_ecapac(inpt.var)

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
    inpt.extr[inpt.var]["e"]["data"][inpt.var] = inpt.extr[inpt.var]["e"]["data"][inpt.var].values * 100.

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
    inpt.extr[inpt.var]["c"]["data"][inpt.var] = inpt.extr[inpt.var]["c"]["data"][inpt.var] - 273.15

    # ERA5
    read_rean(inpt.var, "e")
    inpt.extr[inpt.var]["e"]["data"][inpt.var] = inpt.extr[inpt.var]["e"]["data"][inpt.var] - 273.15

    # THAAO
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_thaao_weather(inpt.var)
        inpt.extr[inpt.var]["t"]["data"][inpt.var] = inpt.extr[inpt.var]["t"]["data"][inpt.var] - 273.15

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
        "msl_pres": read_msl_pres,
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
