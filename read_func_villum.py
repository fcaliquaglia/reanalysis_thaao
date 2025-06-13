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

import julian
import numpy as np
import pandas as pd
import inputs as inpt


def read_villum_weather(vr):
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
        file_path = r"H:\Shared drives\Dati_THAAO\thaao_arcsix\Villum_2024.csv"

        column_names = [
            'DateTime', 'VD(degrees 9m)', 'VS_Mean(m/s 9m)', 'VS_Max(m/s 9m)',
            'Temp(oC 9m)', 'RH(% 9m)', 'RAD(W/m2 3m)', 'Pressure(hPa 0m)', 'Snow depth(m)'
        ]
        new_column_names = ['datetime', 'windd', 'winds', 'null',
                            'temp', 'rh', 'rad', 'surf_press', 'snow_depth']

        df = pd.read_csv(file_path, sep=';', names=new_column_names,
                         index_col='datetime', header=0, parse_dates=['datetime'], dayfirst=True)
        df.drop(columns=['null'], inplace=True)
        inpt.extr[vr]["t"]["data"] = df[vr].to_frame()
        inpt.extr[vr]["t1"]["data"] = df[vr].to_frame()*np.nan
        inpt.extr[vr]["t2"]["data"] = df[vr].to_frame()*np.nan
        print(f'OK: Villum')
    except FileNotFoundError:
        print(f'NOT FOUND: {file_path}')

    return
