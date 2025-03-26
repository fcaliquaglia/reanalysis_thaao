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
# AUTHORS: Filippo Cali' Quaglia
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
import xarray as xr

import inputs as inpt


def read_reanalysis(vr, source):
    """
    Generic function to read reanalysis data for the given variable from a specified source.

    :param vr: Variable for which the data is read
    :param source: Source of data (e.g., 'c' for Carra, 'e' for ERA5, 'l' for ERA5 Land)
    :return: None
    """

    data_tmp_all = []
    extr_source = inpt.extr[vr][source]
    base_path = inpt.directories[source]
    nanval = inpt.var_dict[source]['nanval']
    fn_template = extr_source['fn']
    years = inpt.years

    # Read data for each year
    for year in years:
        file_path = os.path.join(base_path, f'{fn_template}{year}.txt')
        try:
            # Read the file
            data_tmp = pd.read_table(
                    file_path, sep='\s+', header=None, skiprows=1, engine='python', skip_blank_lines=True)
            data_tmp[data_tmp == nanval] = np.nan
            data_tmp_all.append(data_tmp)
            print(f'OK: {fn_template}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn_template}{year}.txt')

    # Concatenate data once at the end
    if data_tmp_all:
        data_tmp_all = pd.concat(data_tmp_all, axis=0)
        # Set datetime index
        datetime_column = pd.to_datetime(data_tmp_all[0] + ' ' + data_tmp_all[1], format='%Y-%m-%d %H:%M:%S')
        data_tmp_all.index = datetime_column
        # Select relevant column based on source-specific column
        data_tmp_all = data_tmp_all[[extr_source['column']]]
        data_tmp_all.columns = [vr]  # Rename column to variable name

        extr_source['data'] = data_tmp_all  # Store the data
    else:
        print(f'No data found for {fn_template}')

    # Additional processing for radiation variables (only for ERA5 Land)
    if source == 'l' and inpt.var in ['sw_up', 'sw_down', 'lw_up', 'lw_down']:
        calc_rad_acc_era5_land(vr)

    return


def read_thaao_weather(vr):
    """
    Reads weather data for a specific variable (vr) from a NetCDF file.
    :param vr: The variable to be processed
    :return: None
    """
    file_path = os.path.join(inpt.directories['t'], 'thaao_meteo', f"{inpt.extr[vr]['t']['fn']}.nc")

    try:
        # Open NetCDF file and convert to DataFrame
        inpt.extr[vr]['t']['data'] = xr.open_dataset(file_path, engine='netcdf4').to_dataframe()
        print(f'OK: {inpt.extr[vr]["t"]["fn"]}.nc')

        # Filter the required column and rename it
        column = inpt.extr[vr]['t']['column']
        inpt.extr[vr]['t']['data'] = inpt.extr[vr]['t']['data'][[column]].astype(float)
        inpt.extr[vr]['t']['data'].columns = [vr]

    except FileNotFoundError:
        print(f'NOT FOUND: {inpt.extr[vr]["t"]["fn"]}.nc')


def read_thaao_rad(vr):
    """
    Reads radiation data for a specific variable (vr) from text files.
    :param vr: The variable to be processed
    :return: None
    """
    t_tmp_all = []

    for i in inpt.date_ranges['rad'][inpt.date_ranges['rad'].year.isin(inpt.years)]:
        i_fmt = i.strftime('%Y')
        file_path = os.path.join(inpt.directories['t'], 'thaao_rad', f"{inpt.extr[vr]['t']['fn']}{i_fmt}_5MIN.dat")

        try:
            # Read data from the file
            t_tmp = pd.read_table(file_path, engine='python', sep='\s+', header=0, skiprows=None, decimal='.')
            # Convert Julian dates to DatetimeIndex
            tmp = [julian.from_jd(julian.to_jd(dt.datetime(i_fmt - 1, 12, 31, 0, 0), fmt='jd') + el, fmt='jd').replace(
                    microsecond=0) for el in t_tmp['JDAY_UT']]

            t_tmp.index = pd.DatetimeIndex(tmp)
            t_tmp = t_tmp[[inpt.extr[vr]['t']['column']]]
            t_tmp_all.append(t_tmp)
            print(f'OK: {inpt.extr[vr]["t"]["fn"]}{i_fmt}.txt')

        except FileNotFoundError:
            print(f'NOT FOUND: {inpt.extr[vr]["t"]["fn"]}{i_fmt}.txt')

    # Concatenate all DataFrames after the loop
    if t_tmp_all:
        inpt.extr[vr]['t']['data'] = pd.concat(t_tmp_all, axis=0)
        inpt.extr[vr]['t']['data'].columns = [vr]


def read_thaao_hatpro(vr):
    """
    Reads hatpro data for a specific variable (vr) from text files.
    :param vr: The variable to be processed
    :return: None
    """
    t1_tmp_all = []

    for i in inpt.date_ranges['hatpro'][inpt.date_ranges['hatpro'].year.isin(inpt.years)]:
        i_fmt = i.strftime('%Y')
        file_path = os.path.join(
                inpt.directories['t'], 'thaao_hatpro', 'definitivi_da_giando',
                f"{inpt.extr[vr]['t1']['fn']}{i_fmt}.DAT")

        try:
            # Read the hatpro file
            t1_tmp = pd.read_table(file_path, sep='\s+', engine='python', header=None, skiprows=1)
            t1_tmp.columns = ['JD_rif', 'RF', 'N', 'LWP_gm-2', 'STD_LWP']

            # Convert Julian dates to DatetimeIndex
            tmp = [julian.from_jd(julian.to_jd(dt.datetime(i_fmt - 1, 12, 31, 0, 0), fmt='jd') + el, fmt='jd').replace(
                    microsecond=0) for el in t1_tmp['JD_rif']]

            t1_tmp.index = pd.DatetimeIndex(tmp)
            t1_tmp.drop(columns=['JD_rif', 'STD_LWP', 'RF', 'N'], axis=1, inplace=True)
            t1_tmp_all.append(t1_tmp)
            print(f'OK: {inpt.extr[vr]["t1"]["fn"]}{i_fmt}.DAT')

        except FileNotFoundError:
            print(f'NOT FOUND: {inpt.extr[vr]["t1"]["fn"]}{i_fmt}.DAT')

    # Concatenate all DataFrames after the loop
    if t1_tmp_all:
        inpt.extr[vr]['t1']['data'] = pd.concat(t1_tmp_all, axis=0)
        inpt.extr[vr]['t1']['data'].columns = [vr]


def read_thaao_ceilometer(vr):
    """
    Reads ceilometer data for a specific variable (vr) from text files.
    :param vr: The variable to be processed
    :return: None
    """
    t_tmp_all = []

    for i in inpt.date_ranges['ceilometer'][inpt.date_ranges['ceilometer'].year.isin(inpt.years)]:
        i_fmt = i.strftime('%Y%m%d')
        file_path = os.path.join(
                inpt.directories['t'], 'thaao_ceilometer', 'medie_tat_rianalisi',
                f"{i_fmt}{inpt.extr[vr]['t']['fn']}.txt")

        try:
            # Read data from the file
            t_tmp = pd.read_table(file_path, sep='\s+', engine='python', header=0, skiprows=9, skipfooter=0)
            t_tmp[t_tmp == inpt.var_dict['t']['nanval']] = np.nan
            t_tmp_all.append(t_tmp)
            print(f'OK: {i_fmt}{inpt.extr[vr]["t"]["fn"]}.txt')

        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f'NOT FOUND: {i_fmt}{inpt.extr[vr]["t"]["fn"]}.txt')

    # Concatenate all DataFrames after the loop
    if t_tmp_all:
        inpt.extr[vr]['t']['data'] = pd.concat(t_tmp_all, axis=0)
        inpt.extr[vr]['t']['data'].index = pd.to_datetime(
                inpt.extr[vr]['t']['data']['#'] + ' ' + inpt.extr[vr]['t']['data']['date[y-m-d]time[h:m:s]'],
                format='%Y-%m-%d %H:%M:%S')
        inpt.extr[vr]['t']['data'].index.name = 'datetime'
        column = inpt.extr[vr]['t']['column']
        inpt.extr[vr]['t']['data'] = inpt.extr[vr]['t']['data'][[column]].astype(float)
        inpt.extr[vr]['t']['data'].columns = [vr]


def read_thaao_aws_ecapac(vr):
    """
    Reads AWS ECAPAC data for a specific variable (vr) from .dat files.

    :param vr: The variable to be processed
    :return: None
    """
    t2_tmp_all = []  # List to collect dataframes

    for i in inpt.date_ranges['aws_ecapac'][inpt.date_ranges['aws_ecapac'].year.isin(inpt.years)]:
        i_fmt = i.strftime('%Y_%m_%d')
        file_path = os.path.join(
                inpt.directories['t'], 'thaao_ecapac_aws_snow', 'AWS_ECAPAC', i.strftime('%Y'),
                f"{inpt.extr[vr]['t2']['fn']}{i_fmt}_00_00.dat")

        try:
            # Read the file and process the data
            t2_tmp = pd.read_csv(
                    file_path, skiprows=[0, 3], header=0, decimal='.', delimiter=',', engine='python',
                    index_col='TIMESTAMP').iloc[1:, :]
            t2_tmp_all.append(t2_tmp)  # Append to the list

            print(f'OK: {inpt.extr[vr]["t2"]["fn"]}{i_fmt}_00_00.dat')

        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f'NOT_FOUND: {inpt.extr[vr]["t2"]["fn"]}{i_fmt}_00_00.dat')

    # Concatenate all DataFrames after the loop
    if t2_tmp_all:
        t2_tmp_all = pd.concat(t2_tmp_all, axis=0)
        t2_tmp_all.index = pd.to_datetime(t2_tmp_all.index)  # Convert index to DatetimeIndex
        t2_tmp_all.index.name = 'datetime'

        column = inpt.extr[vr]['t2']['column']
        inpt.extr[vr]['t2']['data'] = t2_tmp_all[[column]].astype(float)
        inpt.extr[vr]['t2']['data'].columns = [vr]


def calc_rad_acc_era5_land(vr):
    """

    :param vr:
    :return:
    """
    # caluculating instantaneous as difference with previous timestep
    inpt.extr[vr]['l']['data_diff'] = inpt.extr[vr]['l']['data'][vr].diff()
    # dropping value at 0100 which does not need any subtraction (it is the first of the day)
    inpt.extr[vr]['l']['data_diff'] = inpt.extr[vr]['l']['data_diff'][inpt.extr[vr]['l']['data_diff'].index.hour != 1]
    # selecting original value at 0100
    orig_filtered_data = inpt.extr[vr]['l']['data'][inpt.extr[vr]['l']['data'].index.hour == 1]
    # appending original value at 0100
    inpt.extr[vr]['l']['data_diff'] = pd.concat([inpt.extr[vr]['l']['data_diff'], orig_filtered_data]).sort_index()
    inpt.extr[vr]['l']['data'] = inpt.extr[vr]['l']['data_diff']

    print('ERA5-LAND data for radiation corrected because they are values daily-accumulated!')
    return
