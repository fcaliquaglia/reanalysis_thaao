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
from metpy.calc import wind_direction, wind_speed
from metpy.units import units

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

    :param vr:
    :return:
    """
    try:
        inpt.extr[vr]['t']['data'] = xr.open_dataset(
                os.path.join(inpt.directories['t'], 'thaao_meteo', f'{inpt.extr[vr]['t']['fn']}.nc'),
                engine='netcdf4').to_dataframe()
        print(f'OK: {inpt.extr[vr]['t']['fn']}.nc')
    except FileNotFoundError:
        print(f'NOT FOUND: {inpt.extr[vr]['t']['fn']}.nc')
    inpt.extr[vr]['t']['data'] = inpt.extr[vr]['t']['data'][[inpt.extr[vr]['t']['column']]]
    inpt.extr[vr]['t']['data'].columns = [vr]
    return


def read_thaao_rad(vr):
    """

    :param vr:
    :return:
    """
    t_tmp_all = pd.DataFrame()
    for i in inpt.rad_daterange[inpt.rad_daterange.year.isin(inpt.years)]:
        i_fmt = int(i.strftime('%Y'))
        try:
            t_tmp = pd.read_table(
                    os.path.join(inpt.directories['t'], 'thaao_rad', f'{inpt.extr[vr]['t']['fn']}{i_fmt}_5MIN.dat'),
                    engine='python', skiprows=None, header=0, decimal='.', sep='\s+')
            tmp = np.empty(t_tmp['JDAY_UT'].shape, dtype=dt.datetime)
            for ii, el in enumerate(t_tmp['JDAY_UT']):
                new_jd_ass = el + julian.to_jd(dt.datetime(i_fmt - 1, 12, 31, 0, 0), fmt='jd')
                tmp[ii] = julian.from_jd(new_jd_ass, fmt='jd')
                tmp[ii] = tmp[ii].replace(microsecond=0)
            t_tmp.index = pd.DatetimeIndex(tmp)
            t_tmp = t_tmp[[inpt.extr[vr]['t']['column']]]
            t_tmp_all = pd.concat([t_tmp_all, t_tmp], axis=0)
            print(f'OK: {inpt.extr[vr]['t']['fn']}{i_fmt}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {inpt.extr[vr]['t']['fn']}{i_fmt}.txt')
    inpt.extr[vr]['t']['data'] = t_tmp_all
    inpt.extr[vr]['t']['data'].columns = [vr]
    return


def read_thaao_hatpro(vr):
    """

    :param vr:
    :return:
    """
    t1_tmp_all = pd.DataFrame()
    for i in inpt.hatpro_daterange[inpt.hatpro_daterange.year.isin(inpt.years)]:
        i_fmt = int(i.strftime('%Y'))
        try:
            t1_tmp = pd.read_table(
                    os.path.join(
                            inpt.directories['t'], 'thaao_hatpro', 'definitivi_da_giando',
                            f'{inpt.extr[vr]['t1']['fn']}{i_fmt}', f'{inpt.extr[vr]['t1']['fn']}{i_fmt}.DAT'),
                    sep='\s+', engine='python', header=None, skiprows=1)
            t1_tmp.columns = ['JD_rif', 'RF', 'N', 'LWP_gm-2', 'STD_LWP']
            tmp = np.empty(t1_tmp['JD_rif'].shape, dtype=dt.datetime)
            for ii, el in enumerate(t1_tmp['JD_rif']):
                new_jd_ass = el + julian.to_jd(dt.datetime(i_fmt - 1, 12, 31, 0, 0), fmt='jd')
                tmp[ii] = julian.from_jd(new_jd_ass, fmt='jd')
                tmp[ii] = tmp[ii].replace(microsecond=0)
            t1_tmp.index = pd.DatetimeIndex(tmp)
            t1_tmp.drop(columns=['JD_rif', 'STD_LWP', 'RF', 'N'], axis=1, inplace=True)
            t1_tmp_all = pd.concat([t1_tmp_all, t1_tmp], axis=0)
            print(f'OK: {inpt.extr[vr]['t1']['fn']}{i_fmt}.DAT')
        except FileNotFoundError:
            print(f'NOT FOUND: {inpt.extr[vr]['t1']['fn']}{i_fmt}.DAT')
    inpt.extr[vr]['t1']['data'] = t1_tmp_all
    inpt.extr[vr]['t1']['data'].columns = [vr]
    return


def read_thaao_ceilometer(vr):
    """

    :param vr:
    :return:
    """
    t_tmp_all = pd.DataFrame()
    for i in inpt.ceilometer_daterange[inpt.ceilometer_daterange.year.isin(inpt.years)]:
        i_fmt = i.strftime('%Y%m%d')
        try:
            t_tmp = pd.read_table(
                    os.path.join(
                            inpt.directories['t'], 'thaao_ceilometer', 'medie_tat_rianalisi',
                            f'{i_fmt}{inpt.extr[vr]['t']['fn']}.txt'), skipfooter=0, sep='\s+', header=0, skiprows=9,
                    engine='python')
            t_tmp[t_tmp == inpt.var_dict['t']['nanval']] = np.nan
            t_tmp_all = pd.concat([t_tmp_all, t_tmp], axis=0)
            print(f'OK: {i_fmt}{inpt.extr[vr]['t']['fn']}.txt')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f'NOT FOUND: {i_fmt}{inpt.extr[vr]['t']['fn']}.txt')
    inpt.extr[vr]['t']['data'] = t_tmp_all
    inpt.extr[vr]['t']['data'].index = pd.to_datetime(
            inpt.extr[vr]['t']['data']['#'] + ' ' + inpt.extr[vr]['t']['data']['date[y-m-d]time[h:m:s]'],
            format='%Y-%m-%d %H:%M:%S')
    inpt.extr[vr]['t']['data'].index.name = 'datetime'
    inpt.extr[vr]['t']['data'] = inpt.extr[vr]['t']['data'].iloc[:, :].filter(
            [inpt.extr[vr]['t']['column']]).astype(float)
    inpt.extr[vr]['t']['data'].columns = [vr]

    return


def read_aws_ecapac(vr):
    """

    :param vr:
    :return:
    """
    t2_tmp_all = pd.DataFrame()
    for i in inpt.aws_ecapac_daterange[inpt.aws_ecapac_daterange.year.isin(inpt.years)]:
        i_fmt = i.strftime('%Y_%m_%d')
        try:
            file = os.path.join(
                    inpt.directories['t'], 'thaao_ecapac_aws_snow', 'AWS_ECAPAC', i.strftime('%Y'),
                    f'{inpt.extr[vr]['t2']['fn']}{i_fmt}_00_00.dat')
            t2_tmp = pd.read_csv(
                    file, skiprows=[0, 3], header=0, decimal='.', delimiter=',', engine='python',
                    index_col='TIMESTAMP').iloc[1:, :]
            t2_tmp_all = pd.concat([t2_tmp_all, t2_tmp], axis=0)

            print(f'OK: {inpt.extr[vr]['t2']['fn']}{i_fmt}_00_00.dat')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f'NOT_FOUND: {inpt.extr[vr]['t2']['fn']}{i_fmt}_00_00.dat')
    inpt.extr[vr]['t2']['data'] = t2_tmp_all
    inpt.extr[vr]['t2']['data'].index = pd.DatetimeIndex(inpt.extr[vr]['t2']['data'].index)
    inpt.extr[vr]['t2']['data'].index.name = 'datetime'
    inpt.extr[vr]['t2']['data'] = inpt.extr[vr]['t2']['data'].iloc[:, :].filter(
            [inpt.extr[vr]['t2']['column']]).astype(float)
    inpt.extr[vr]['t2']['data'].columns = [vr]
    return


def read_alb():
    # CARRA
    read_reanalysis(inpt.var, 'c')
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data'] / 100.
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] <= 0.] = np.nan

    # ERA5
    read_reanalysis(inpt.var, 'e')
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] <= 0.] = np.nan

    # ERA5-LAND
    read_reanalysis(inpt.var, 'l')
    inpt.extr[inpt.var]['l']['data'][inpt.extr[inpt.var]['l']['data'] <= 0.] = np.nan

    # THAAO
    read_thaao_rad(inpt.var)

    return


def read_cbh():
    # CARRA
    read_reanalysis(inpt.var, 'c')
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] <= 20.] = np.nan

    # ERA5
    read_reanalysis(inpt.var, 'e')
    inpt.extr[inpt.var]['e']['data'][inpt.extr[inpt.var]['e']['data'] <= 20.] = np.nan
    inpt.extr[inpt.var]['e']['data'] += inpt.thaao_elev

    # THAAO
    read_thaao_ceilometer(inpt.var)
    inpt.extr[inpt.var]['t']['data'][inpt.extr[inpt.var]['t']['data'] <= 20.] = np.nan

    return


def read_lwp():
    # CARRA
    read_reanalysis(inpt.var, 'c')
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data']
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] < 0.01] = np.nan
    # c[c < 15] = 0

    # ERA5
    read_reanalysis(inpt.var, 'e')
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data']
    inpt.extr[inpt.var]['e']['data'][inpt.extr[inpt.var]['e']['data'] < 0.01] = np.nan
    # e[e < 15] = 0

    # THAAO1
    read_thaao_hatpro(inpt.var)
    inpt.extr[inpt.var]['t1']['data'][inpt.extr[inpt.var]['t1']['data'] < 0.01] = np.nan
    # t1[t1 < 15] = 0

    return


def read_msl_pres():
    # CARRA
    read_reanalysis(inpt.var, 'c')

    return


def read_precip():
    # CARRA
    read_reanalysis(inpt.var, 'c')

    # ERA5
    read_reanalysis(inpt.var, 'e')
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'].values * 1000.

    # THAAO2
    read_aws_ecapac(inpt.var)

    return


def read_lw_down():
    # CARRA
    read_reanalysis('lw_down', 'c')
    inpt.extr['lw_down']['c']['data'][inpt.extr['lw_down']['c']['data'] < 0.] = np.nan
    inpt.extr['lw_down']['c']['data'] = inpt.extr['lw_down']['c']['data'] / inpt.var_dict['c']['rad_conv_factor']

    # ERA5
    read_reanalysis('lw_down', 'e')
    inpt.extr['lw_down']['e']['data'][inpt.extr['lw_down']['e']['data'] < 0.] = np.nan
    inpt.extr['lw_down']['e']['data'] = inpt.extr['lw_down']['e']['data'] / inpt.var_dict['e']['rad_conv_factor']

    # ERA5-LAND
    read_reanalysis('lw_down', 'l')
    inpt.extr['lw_down']['l']['data'][inpt.extr['lw_down']['l']['data'] < 0.] = np.nan
    inpt.extr['lw_down']['l']['data'] = inpt.extr['lw_down']['l']['data'] / inpt.var_dict['l']['rad_conv_factor']

    # THAAO
    read_thaao_rad('lw_down')
    inpt.extr['lw_down']['t']['data'][inpt.extr['lw_down']['t']['data'] < 0.] = np.nan
    return


def read_sw_down():
    # CARRA
    read_reanalysis('sw_down', 'c')
    inpt.extr['sw_down']['c']['data'][inpt.extr['sw_down']['c']['data'] < 0.] = np.nan
    inpt.extr['sw_down']['c']['data'] = inpt.extr['sw_down']['c']['data'] / inpt.var_dict['c']['rad_conv_factor']

    # ERA5
    read_reanalysis('sw_down', 'e')
    inpt.extr['sw_down']['e']['data'][inpt.extr['sw_down']['e']['data'] < 0.] = np.nan
    inpt.extr['sw_down']['e']['data'] = inpt.extr['sw_down']['e']['data'] / inpt.var_dict['e']['rad_conv_factor']

    # ERA5-LAND
    read_reanalysis('sw_down', 'l')
    inpt.extr['sw_down']['l']['data'][inpt.extr['sw_down']['l']['data'] < 0.] = np.nan
    inpt.extr['sw_down']['l']['data'] = inpt.extr['sw_down']['l']['data'] / inpt.var_dict['l']['rad_conv_factor']

    # THAAO
    read_thaao_rad('sw_down')
    inpt.extr['sw_down']['t']['data'][inpt.extr['sw_down']['t']['data'] < 0.] = np.nan
    return


def read_lw_up():
    read_lw_down()

    # CARRA
    read_reanalysis('lw_net', 'c')
    inpt.extr['lw_net']['c']['data'] = inpt.extr['lw_net']['c']['data'] / inpt.var_dict['c']['rad_conv_factor']
    inpt.extr['lw_up']['c']['data'] = pd.DataFrame(
            index=inpt.extr['lw_down']['c']['data'].index,
            data=inpt.extr['lw_down']['c']['data'].values - inpt.extr['lw_net']['c']['data'].values, columns=['lw_up'])
    inpt.extr['lw_up']['c']['data'][inpt.extr['lw_up']['c']['data'] < 0.] = np.nan
    # del inpt.extr['lw_net']['c']['data']

    # ERA5
    read_reanalysis('lw_net', 'e')
    inpt.extr['lw_net']['e']['data'] = inpt.extr['lw_net']['e']['data'] / inpt.var_dict['e']['rad_conv_factor']
    inpt.extr['lw_up']['e']['data'] = pd.DataFrame(
            index=inpt.extr['lw_down']['e']['data'].index,
            data=inpt.extr['lw_down']['e']['data'].values - inpt.extr['lw_net']['e']['data'].values, columns=['lw_up'])
    inpt.extr['lw_up']['e']['data'][inpt.extr['lw_up']['e']['data'] < 0.] = np.nan
    # del inpt.extr['lw_net']['e']['data']

    # ERA5-LAND
    read_reanalysis('lw_net', 'l')
    inpt.extr['lw_net']['l']['data'] = inpt.extr['lw_net']['l']['data'] / inpt.var_dict['l']['rad_conv_factor']
    inpt.extr['lw_up']['l']['data'] = pd.DataFrame(
            index=inpt.extr['lw_down']['l']['data'].index,
            data=inpt.extr['lw_down']['l']['data'].values - inpt.extr['lw_net']['l']['data'].values, columns=['lw_up'])
    inpt.extr['lw_up']['l']['data'][inpt.extr['lw_up']['l']['data'] < 0.] = np.nan
    # del inpt.extr['lw_net']['l']['data']

    # THAAO
    read_thaao_rad('lw_up')
    inpt.extr['lw_up']['t']['data'][inpt.extr['lw_up']['t']['data'] < 0.] = np.nan

    return


def read_sw_up():
    read_sw_down()

    # CARRA
    read_reanalysis('sw_net', 'c')
    inpt.extr['sw_net']['c']['data'] = inpt.extr['sw_net']['c']['data'] / inpt.var_dict['c']['rad_conv_factor']
    inpt.extr['sw_up']['c']['data'] = pd.DataFrame(
            index=inpt.extr['sw_down']['c']['data'].index,
            data=inpt.extr['sw_down']['c']['data'].values - inpt.extr['sw_net']['c']['data'].values, columns=['sw_up'])
    inpt.extr['sw_up']['c']['data'][inpt.extr['sw_up']['c']['data'] < 0.] = np.nan
    del inpt.extr['sw_net']['c']['data']

    # ERA5
    read_reanalysis('sw_net', 'e')
    inpt.extr['sw_net']['e']['data'] = inpt.extr['sw_net']['e']['data'] / inpt.var_dict['e']['rad_conv_factor']
    inpt.extr['sw_up']['e']['data'] = pd.DataFrame(
            index=inpt.extr['sw_down']['e']['data'].index,
            data=inpt.extr['sw_down']['e']['data'].values - inpt.extr['sw_net']['e']['data'].values, columns=['sw_up'])
    inpt.extr['sw_up']['e']['data'][inpt.extr['sw_up']['e']['data'] < 0.] = np.nan
    del inpt.extr['sw_net']['e']['data']

    # ERA5-LAND
    read_reanalysis('sw_net', 'l')
    inpt.extr['sw_net']['l']['data'] = inpt.extr['sw_net']['l']['data'] / inpt.var_dict['l']['rad_conv_factor']
    inpt.extr['sw_up']['l']['data'] = pd.DataFrame(
            index=inpt.extr['sw_down']['l']['data'].index,
            data=inpt.extr['sw_down']['l']['data'].values - inpt.extr['sw_net']['l']['data'].values, columns=['sw_up'])
    inpt.extr['sw_up']['l']['data'][inpt.extr['sw_up']['l']['data'] < 0.] = np.nan
    del inpt.extr['sw_net']['l']['data']

    # THAAO
    read_thaao_rad('sw_up')
    inpt.extr['sw_up']['t']['data'][inpt.extr['sw_up']['t']['data'] < 0.] = np.nan

    return


def read_rh():
    # CARRA
    read_reanalysis(inpt.var, 'c')

    # ERA5
    read_reanalysis(inpt.var, 'e')
    if inpt.extr[inpt.var]['l']['data'].empty:
        read_reanalysis('temp', 'e')
    calc_rh_from_tdp()

    # ERA5-LAND
    read_reanalysis(inpt.var, 'l')
    if inpt.extr[inpt.var]['l']['data'].empty:
        read_reanalysis('temp', 'e')
    calc_rh_from_tdp()

    # e.g. l_td[l_td_tmp == -32767.0] = np.nan

    # THAAO2
    read_thaao_weather(inpt.var)
    read_aws_ecapac(inpt.var)
    read_thaao_weather(inpt.var)

    return


def calc_rh_from_tdp():
    # TODO not working

    # e = pd.concat([inpt.extr[inpt.var]['t']['data'], e_t], axis=1)

    # e['rh'] = relative_humidity_from_dewpoint(e['e_t'].values * units.K, e['e_td'].values * units.K).to('percent')
    inpt.extr[inpt.var]['e']['data'].drop(columns=['e_t', 'e_td'], inplace=True)
    inpt.extr[inpt.var]['e']['data'].columns = [inpt.var]

    return


def read_surf_pres():
    # CARRA
    read_reanalysis(inpt.var, 'c')
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data'] / 100.
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] <= 900] = np.nan

    # ERA5
    read_reanalysis(inpt.var, 'e')
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'] / 100.
    inpt.extr[inpt.var]['e']['data'][inpt.extr[inpt.var]['e']['data'] <= 900] = np.nan

    # THAAO
    read_thaao_weather(inpt.var)
    inpt.extr[inpt.var]['t']['data'][inpt.extr[inpt.var]['t']['data'] <= 900] = np.nan
    inpt.extr[inpt.var]['t']['data'].loc['2021-10-11 00:00:00':'2021-10-19 00:00:00'] = np.nan
    inpt.extr[inpt.var]['t']['data'].loc['2024-4-26 00:00:00':'2024-5-4 00:00:00'] = np.nan

    # THAAO2
    read_aws_ecapac(inpt.var)

    return


def read_tcc():
    # CARRA
    read_reanalysis(inpt.var, 'c')

    # ERA5
    read_reanalysis(inpt.var, 'e')
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'].values * 100.

    # THAAO
    read_thaao_ceilometer(inpt.var)

    return


def read_temp():
    # CARRA
    read_reanalysis(inpt.var, 'c')
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data'] - 273.15

    # ERA5
    read_reanalysis(inpt.var, 'e')
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'] - 273.15

    # ERA5-LAND
    read_reanalysis(inpt.var, 'l')
    inpt.extr[inpt.var]['l']['data'] = inpt.extr[inpt.var]['l']['data'] - 273.15

    # THAAO
    read_thaao_weather(inpt.var)
    inpt.extr[inpt.var]['t']['data'] = inpt.extr[inpt.var]['t']['data'] - 273.15

    # THAAO2
    read_aws_ecapac(inpt.var)
    return


def read_wind():
    # CARRA
    read_reanalysis('winds', 'c')
    read_reanalysis('windd', 'c')

    # ERA5
    read_reanalysis('windu', 'e')
    read_reanalysis('windv', 'e')
    e_ws = wind_speed(
            inpt.extr['windu']['e']['data'].values * units('m/s'),
            inpt.extr['windv']['e']['data'].values * units('m/s'))
    inpt.extr['winds']['e']['data'] = pd.DataFrame(
            index=inpt.extr['windu']['e']['data'].index, data=e_ws.magnitude, columns=['winds'])

    e_wd = wind_direction(
            inpt.extr['windu']['e']['data'].values * units('m/s'),
            inpt.extr['windv']['e']['data'].values * units('m/s'))
    inpt.extr['windd']['e']['data'] = pd.DataFrame(
            index=inpt.extr['windu']['e']['data'].index, data=e_wd.magnitude, columns=['windd'])

    # ERA5-LAND
    # TODO activate when files are available
    # read_reanalysis('windu', 'l')
    # read_reanalysis('windv', 'l')
    # l_ws = wind_speed(
    #         inpt.extr['windu']['l']['data'].values * units('m/s'),
    #         inpt.extr['windv']['l']['data'].values * units('m/s'))
    # inpt.extr['winds']['l']['data'] = pd.DataFrame(
    #     index=inpt.extr['windu']['l']['data'].index, data=l_ws.magnitude, columns=['winds'])
    #
    # l_wd = wind_direction(
    #         inpt.extr['windu']['l']['data'].values * units('m/s'),
    #         inpt.extr['windv']['l']['data'].values * units('m/s'))
    # inpt.extr['windd']['l']['data'] = pd.DataFrame(
    #     index=inpt.extr['windu']['l']['data'].index, data=l_wd.magnitude, columns=['windd'])

    # THAAO2
    read_aws_ecapac('winds')
    read_aws_ecapac('windd')

    return


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


def read():
    """
    Read the data based on the value of inpt.var.
    """

    # Create a mapping of variable names to functions
    read_functions = {
        'alb': read_alb,
        'cbh': read_cbh,
        'msl_pres': read_msl_pres,
        'lwp': read_lwp,
        'lw_down': read_lw_down,
        'lw_up': read_lw_up,
        'precip': read_precip,
        'rh': read_rh,
        'surf_pres': read_surf_pres,
        'sw_down': read_sw_down,
        'sw_up': read_sw_up,
        'tcc': read_tcc,
        'temp': read_temp,
        'winds': read_wind,
        'windd': read_wind,
    }

    # Use the mapping to call the correct function
    var = inpt.var
    if var in read_functions:
        return read_functions[var]()
    else:
        raise ValueError(f"Unknown variable: {var}")
