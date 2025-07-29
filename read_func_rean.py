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


def read_rean(vr, data_typ):
    """
    Read processed reanalysis data from parquet files or generate them if missing.

    :param vr: Variable name (e.g., "temp").
    :param data_typ: Dataset type (e.g., "c" for CARRA, "e" for ERA5).
    """

    parquet_paths = []

    if inpt.datasets['dropsondes']['switch']:
        drop_files = sorted(glob.glob(os.path.join('txt_locations', "ARCSIX-AVAPS-netCDF_G3*.txt")))
        for file_path in drop_files:
            file_name = os.path.basename(file_path)
            year = 2024  # fixed year for dropsondes
            output_file = f"{inpt.extr[vr][data_typ]['fn']}{file_name.replace('_loc.txt', '')}_{year}.parquet"
            output_path = os.path.join(inpt.basefol[data_typ]['parquets'], output_file)
            parquet_paths.append(output_path)
    else:
        for year in inpt.years:
            output_file = f"{inpt.extr[vr][data_typ]['fn']}{inpt.location}_{year}.parquet"
            output_path = os.path.join(inpt.basefol[data_typ]['parquets'], output_file)
            parquet_paths.append(output_path)

    # 🔍 Check if all parquet files exist
    missing_years = []
    for year in inpt.years:
        file_name = f"{inpt.extr[vr][data_typ]['fn']}{inpt.location}_{year}.parquet"
        parquet_path = os.path.join(inpt.basefol[data_typ]['parquets'], file_name)
        if not os.path.exists(parquet_path):
            missing_years.append(year)

    if not missing_years==[]:
        print("⚙️  Missing parquet files detected. Running processing...")
        if inpt.datasets['dropsondes']['switch']:
            for file_path in drop_files:
                file_name = os.path.basename(file_path)
                inpt.location = file_name.replace('_loc.txt', '')
                tls.process_rean(vr, data_typ, year=2024)
        else:
            for year in missing_years:
                tls.process_rean(vr, data_typ, year)

    # ✅ Now read all parquet files
    data_all = []
    for pq in parquet_paths:
        try:
            data_tmp = pd.read_parquet(pq)
            print(f"📥 Loaded {pq}")
            if not data_tmp.empty:
                data_all.append(data_tmp)
        except Exception as e:
            print(f"⚠️  Could not read {pq}: {e}")

    inpt.extr[vr][data_typ]["data"] = pd.concat(data_all) if data_all else pd.DataFrame()



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
