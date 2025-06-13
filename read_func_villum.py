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
from pathlib import Path
import julian
import numpy as np
import pandas as pd
import inputs as inpt
import tools as tls


def read_villum_weather(vr):
    """
    Reads and processes weather data for the specified variable from VILLUM dataset.
    Updates the global input structure with the processed DataFrame.

    :param vr: Variable key to extract (e.g., 'temp', 'winds').
    :type vr: str
    """

    path_out, _ = tls.get_common_paths(vr, "2024")
    csv_file = Path(inpt.basefol['t']['arcsix']) / f"Villum_2024.csv"

    if os.path.exists(path_out):
        df = pd.read_parquet(path_out)
        inpt.extr[vr]["t"]["data"] = df
        print(f"Loaded {path_out}")
        return

    column_map = {
        'DateTime': 'datetime',
        'VD(degrees 9m)': 'windd',
        'VS_Mean(m/s 9m)': 'winds',
        'VS_Max(m/s 9m)': None,
        'Temp(oC 9m)': 'temp',
        'RH(% 9m)': 'rh',
        'RAD(W/m2 3m)': 'rad',
        'Pressure(hPa 0m)': 'surf_pres',
        'Snow depth(m)': 'snow_depth'
    }

    try:
        df = pd.read_csv(csv_file, sep=';', header=0,
                         parse_dates=['DateTime'], dayfirst=True)
        df.rename(columns=column_map, inplace=True)
        df.set_index('datetime', inplace=True)
        df.drop(columns=[col for col in [None]
                if col in df.columns], inplace=True)

        if vr not in df.columns:
            print(f"Variable '{vr}' not found in the dataset.")
            return

        df = df[[vr]]
        df.to_parquet(path_out)
        inpt.extr[vr]["t"]["data"] = df
        print(f'Processed and saved: {csv_file}')
    except FileNotFoundError:
        print(f'CSV file not found: {csv_file}')
        return
    except Exception as e:
        print(f'Error processing {csv_file}: {e}')
        return
