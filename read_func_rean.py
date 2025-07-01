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

import os

import read_funcs as rd_funcs
import pandas as pd

import inputs as inpt
import tools as tls
import glob


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

    if inpt.datasets['dropsondes']['switch']:
        drop_files = sorted(glob.glob(os.path.join(
            'txt_locations', "ARCSIX-AVAPS-netCDF_G3*.txt")))
        # Process missing files if needed
        for file_path in drop_files:
            file_name = os.path.basename(file_path)
            year = 2024
            output_file = f"{inpt.extr[vr][dataset_type]['fn']}{file_name.replace('_loc.txt', '')}_{year}.parquet"
            print(output_file)
            output_path = os.path.join(
                inpt.basefol[dataset_type]['processed'], output_file)
            inpt.location = file_name.replace('_loc.txt', '')
            if not os.path.exists(output_path):
                tls.process_rean(vr, dataset_type, year)
    else:
        # Process missing yearly files if needed
        for year in inpt.years:
            output_file = f"{inpt.extr[vr][dataset_type]['fn']}{inpt.location}_{year}.parquet"
            output_path = os.path.join(
                inpt.basefol[dataset_type]['processed'], output_file)

            if not os.path.exists(output_path):
                tls.process_rean(vr, dataset_type, year)

    # Read all yearly parquet files and concatenate into a single DataFrame
    data_all = []

    if inpt.datasets['dropsondes']['switch']:
        for file_path in drop_files:
            file_name = os.path.basename(file_path)
            year = 2024
            input_file = f"{inpt.extr[vr][dataset_type]['fn']}{file_name.replace('_loc.txt', '')}_{year}.parquet"
            input_path = os.path.join(
                inpt.basefol[dataset_type]['processed'], input_file)
            try:
                data_tmp = pd.read_parquet(input_path)
                print(f"Loaded {input_path}")
                if not data_tmp.empty:
                    data_all.append(data_tmp)
            except FileNotFoundError as e:
                print(f"File not found: {input_path} ({e})")
    else:
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

    # Combine all data if any was loaded
    if data_all:
        data_all = pd.concat(data_all)
    else:
        data_all = pd.DataFrame()

    # Store the concatenated data in the input extraction dictionary
    inpt.extr[vr][dataset_type]["data"] = data_all


def read():
    readers = {
        "alb": rd_funcs.read_alb,
        "cbh": rd_funcs.read_cbh,
        "lwp": rd_funcs.read_lwp,
        "lw_down": rd_funcs.read_lw_down,
        "lw_up": rd_funcs.read_lw_up,
        "precip": rd_funcs.read_precip,
        "rh": rd_funcs.read_rh,
        "surf_pres": rd_funcs.read_surf_pres,
        "sw_down": rd_funcs.read_sw_down,
        "sw_up": rd_funcs.read_sw_up,
        "tcc": rd_funcs.read_tcc,
        "temp": rd_funcs.read_temp,
        "winds": rd_funcs.read_wind,
        "windd": rd_funcs.read_wind,
    }
    reader_func = readers.get(inpt.var)
    if reader_func is None:
        raise ValueError(
            f"No reader function defined for variable '{inpt.var}'")
    return reader_func()
