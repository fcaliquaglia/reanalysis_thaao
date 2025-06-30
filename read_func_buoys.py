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
from pathlib import Path
import pandas as pd
import inputs as inpt
import tools as tls


def read_villum_weather(vr):
    df_all = pd.DataFrame()
    cached_years = 0

    # Attempt to load cached data
    for year in inpt.years:
        path_out, _ = tls.get_common_paths(vr, year, "station")
        if os.path.exists(path_out):
            df_all = pd.concat([df_all, pd.read_parquet(path_out)])
            print(f"Loaded {path_out}")
            cached_years += 1

    # Read from raw CSV if any year is missing
    if cached_years < len(inpt.years):
        csv_file = Path(inpt.basefol['t']['arcsix']) / "Villum_2024.csv"
        column_map = {
            'DateTime': 'datetime',
            'VD(degrees 9m)': 'windd',
            'VS_Mean(m/s 9m)': 'winds',
            'VS_Max(m/s 9m)': None,
            'Temp(oC 9m)': 'temp',
            'RH(% 9m)': 'rh',
            'RAD(W/m2 3m)': 'sw_down',
            'Pressure(hPa 0m)': 'surf_pres',
            'Snow depth(m)': 'snow_depth'
        }

        try:
            df = pd.read_csv(csv_file, sep=';', parse_dates=['DateTime'], dayfirst=True)
            df.rename(columns=column_map, inplace=True)
            df.set_index('datetime', inplace=True)
            df.drop(columns=[col for col in [None] if col in df.columns], inplace=True)
            if vr not in df.columns:
                print(f"Variable '{vr}' not found in the dataset.")
                return

            df_all = df[[vr]]
            print(f"Processed raw CSV: {csv_file}")
        except FileNotFoundError:
            print(f"CSV file not found: {csv_file}")
            return
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            return

    # Save per-year parquet files
    for year in inpt.years:
        df_year = df_all[df_all.index.year == year]
        if not df_year.empty:
            path_out, _ = tls.get_common_paths(vr, year, "station")
            df_year.to_parquet(path_out)
            print(f"Saved {path_out}")

    # Update inpt
    inpt.extr[vr]["t"]["data"] = df_all
