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


def read_carra(vr):
    """

    :param vr:
    :return:
    """
    c_tmp_all = pd.DataFrame()
    for year in inpt.years:
        try:
            c_tmp = pd.read_table(
                    os.path.join(inpt.basefol_c, f'{inpt.extr[vr]['c']['fn']}{year}.txt'), sep='\s+', header=None,
                    skiprows=1, engine='python', skip_blank_lines=True)
            c_tmp[c_tmp == inpt.var_dict['c']['nanval']] = np.nan
            c_tmp_all = pd.concat([c_tmp_all, c_tmp], axis=0)
            print(f'OK: {inpt.extr[vr]['c']['fn']}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {inpt.extr[vr]['c']['fn']}{year}.txt')
    inpt.extr[vr]['c']['data'] = c_tmp_all
    inpt.extr[vr]['c']['data'].index = pd.to_datetime(
            inpt.extr[vr]['c']['data'][0] + ' ' + inpt.extr[vr]['c']['data'][1], format='%Y-%m-%d %H:%M:%S')
    inpt.extr[vr]['c']['data'] = inpt.extr[vr]['c']['data'][[inpt.extr[vr]['c']['column']]]
    inpt.extr[vr]['c']['data'].columns = [vr]
    return


def read_era5(vr):
    """

    :param vr:
    :return:
    """
    e_tmp_all = pd.DataFrame()
    for year in inpt.years:
        try:
            e_tmp = pd.read_table(
                    os.path.join(inpt.basefol_e, f'{inpt.extr[vr]['e']['fn']}{year}.txt'), sep='\s+', header=None,
                    skiprows=1, engine='python')
            e_tmp[e_tmp == inpt.var_dict['e']['nanval']] = np.nan
            e_tmp_all = pd.concat([e_tmp_all, e_tmp], axis=0)
            print(f'OK: {inpt.extr[vr]['e']['fn']}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {inpt.extr[vr]['e']['fn']}{year}.txt')
    inpt.extr[vr]['e']['data'] = e_tmp_all
    inpt.extr[vr]['e']['data'].index = pd.to_datetime(
            inpt.extr[vr]['e']['data'][0] + ' ' + inpt.extr[vr]['e']['data'][1], format='%Y-%m-%d %H:%M:%S')
    inpt.extr[vr]['e']['data'] = inpt.extr[vr]['e']['data'][[inpt.extr[vr]['e']['column']]]
    inpt.extr[vr]['e']['data'].columns = [vr]
    return


def read_era5_land(vr):
    """

    :param vr:
    :return:
    """
    l_tmp_all = pd.DataFrame()
    for year in inpt.years:
        try:
            l_tmp = pd.read_table(
                    os.path.join(inpt.basefol_l, f'{inpt.extr[vr]['l']['fn']}{year}.txt'), sep='\s+', header=None,
                    skiprows=1, engine='python')
            l_tmp[l_tmp == inpt.var_dict['l']['nanval']] = np.nan
            l_tmp_all = pd.concat([l_tmp_all, l_tmp], axis=0)
            print(f'OK: {inpt.extr[vr]['l']['fn']}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {inpt.extr[vr]['l']['fn']}{year}.txt')
    inpt.extr[vr]['l']['data'] = l_tmp_all
    inpt.extr[vr]['l']['data'].index = pd.to_datetime(
            inpt.extr[vr]['l']['data'][0] + ' ' + inpt.extr[vr]['l']['data'][1], format='%Y-%m-%d %H:%M:%S')
    inpt.extr[vr]['l']['data'] = inpt.extr[vr]['l']['data'][[inpt.extr[vr]['l']['column']]]
    inpt.extr[vr]['l']['data'].columns = [vr]

    # only for radiation variables
    if inpt.var in ['sw_up', 'sw_down', 'lw_up', 'lw_down']:
        calc_rad_acc_era5_land(vr)

    return


def read_thaao_weather(vr):
    """

    :param vr:
    :return:
    """
    try:
        inpt.extr[vr]['t']['data'] = xr.open_dataset(
                os.path.join(inpt.basefol_t, 'thaao_meteo', f'{inpt.extr[vr]['t']['fn']}.nc'),
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
                    os.path.join(inpt.basefol_t, 'thaao_rad', f'{inpt.extr[vr]['t']['fn']}{i_fmt}_5MIN.dat'),
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
                            inpt.basefol_t, 'thaao_hatpro', 'definitivi_da_giando',
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
                            inpt.basefol_t_elab, 'thaao_ceilometer_elab', 'medie_tat_rianalisi',
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
                    inpt.basefol_t, 'thaao_ecapac_aws_snow', 'AWS_ECAPAC', i.strftime('%Y'),
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
    read_carra(inpt.var)
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data'] / 100.
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] <= 0.] = np.nan

    # ERA5
    read_era5(inpt.var)
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] <= 0.] = np.nan

    # ERA5-LAND
    read_era5_land(inpt.var)
    inpt.extr[inpt.var]['l']['data'][inpt.extr[inpt.var]['l']['data'] <= 0.] = np.nan

    # THAAO
    read_thaao_rad(inpt.var)

    return


def read_cbh():
    # CARRA
    read_carra(inpt.var)

    # ERA5
    read_era5(inpt.var)

    # THAAO
    read_thaao_ceilometer(inpt.var)

    return


def read_lwp():
    # CARRA
    read_carra(inpt.var)
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data']
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] < 0.01] = np.nan
    # c[c < 15] = 0

    # ERA5
    read_era5(inpt.var)
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
    read_carra(inpt.var)

    return


def read_precip():
    # CARRA
    read_carra(inpt.var)

    # ERA5
    read_era5(inpt.var)
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'].values * 1000.

    # THAAO2
    read_aws_ecapac(inpt.var)

    return


def read_lw_down():
    # CARRA
    read_carra('lw_down')
    inpt.extr['lw_down']['c']['data'][inpt.extr['lw_down']['c']['data'] < 0.] = np.nan
    inpt.extr['lw_down']['c']['data'] = inpt.extr['lw_down']['c']['data'] / inpt.var_dict['c']['rad_conv_factor']

    # ERA5
    read_era5('lw_down')
    inpt.extr['lw_down']['e']['data'][inpt.extr['lw_down']['e']['data'] < 0.] = np.nan
    inpt.extr['lw_down']['e']['data'] = inpt.extr['lw_down']['e']['data'] / inpt.var_dict['e']['rad_conv_factor']

    # ERA5-LAND
    read_era5_land('lw_down')
    inpt.extr['lw_down']['l']['data'][inpt.extr['lw_down']['l']['data'] < 0.] = np.nan
    inpt.extr['lw_down']['l']['data'] = inpt.extr['lw_down']['l']['data'] / inpt.var_dict['l']['rad_conv_factor']

    # THAAO
    read_thaao_rad('lw_down')
    inpt.extr['lw_down']['t']['data'][inpt.extr['lw_down']['t']['data'] < 0.] = np.nan
    return


def read_sw_down():
    # CARRA
    read_carra('sw_down')
    inpt.extr['sw_down']['c']['data'][inpt.extr['sw_down']['c']['data'] < 0.] = np.nan
    inpt.extr['sw_down']['c']['data'] = inpt.extr['sw_down']['c']['data'] / inpt.var_dict['c']['rad_conv_factor']

    # ERA5
    read_era5('sw_down')
    inpt.extr['sw_down']['e']['data'][inpt.extr['sw_down']['e']['data'] < 0.] = np.nan
    inpt.extr['sw_down']['e']['data'] = inpt.extr['sw_down']['e']['data'] / inpt.var_dict['e']['rad_conv_factor']

    # ERA5-LAND
    read_era5_land('sw_down')
    inpt.extr['sw_down']['l']['data'][inpt.extr['sw_down']['l']['data'] < 0.] = np.nan
    inpt.extr['sw_down']['l']['data'] = inpt.extr['sw_down']['l']['data'] / inpt.var_dict['l']['rad_conv_factor']

    # THAAO
    read_thaao_rad('sw_down')
    inpt.extr['sw_down']['t']['data'][inpt.extr['sw_down']['t']['data'] < 0.] = np.nan
    return


def read_lw_up():
    read_lw_down()

    # CARRA
    read_carra('lw_net')
    inpt.extr['lw_net']['c']['data'] = inpt.extr['lw_net']['c']['data'] / inpt.var_dict['c']['rad_conv_factor']
    inpt.extr['lw_up']['c']['data'] = pd.DataFrame(
            index=inpt.extr['lw_down']['c']['data'].index,
            data=inpt.extr['lw_down']['c']['data'].values - inpt.extr['lw_net']['c']['data'].values, columns=['lw_up'])
    inpt.extr['lw_up']['c']['data'][inpt.extr['lw_up']['c']['data'] < 0.] = np.nan
    # del inpt.extr['lw_net']['c']['data']

    # ERA5
    read_era5('lw_net')
    inpt.extr['lw_net']['e']['data'] = inpt.extr['lw_net']['e']['data'] / inpt.var_dict['e']['rad_conv_factor']
    inpt.extr['lw_up']['e']['data'] = pd.DataFrame(
            index=inpt.extr['lw_down']['e']['data'].index,
            data=inpt.extr['lw_down']['e']['data'].values - inpt.extr['lw_net']['e']['data'].values, columns=['lw_up'])
    inpt.extr['lw_up']['e']['data'][inpt.extr['lw_up']['e']['data'] < 0.] = np.nan
    # del inpt.extr['lw_net']['e']['data']

    # ERA5-LAND
    read_era5_land('lw_net')
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
    read_carra('sw_net')
    inpt.extr['sw_net']['c']['data'] = inpt.extr['sw_net']['c']['data'] / inpt.var_dict['c']['rad_conv_factor']
    inpt.extr['sw_up']['c']['data'] = pd.DataFrame(
            index=inpt.extr['sw_down']['c']['data'].index,
            data=inpt.extr['sw_down']['c']['data'].values - inpt.extr['sw_net']['c']['data'].values, columns=['sw_up'])
    inpt.extr['sw_up']['c']['data'][inpt.extr['sw_up']['c']['data'] < 0.] = np.nan
    del inpt.extr['sw_net']['c']['data']

    # ERA5
    read_era5('sw_net')
    inpt.extr['sw_net']['e']['data'] = inpt.extr['sw_net']['e']['data'] / inpt.var_dict['e']['rad_conv_factor']
    inpt.extr['sw_up']['e']['data'] = pd.DataFrame(
            index=inpt.extr['sw_down']['e']['data'].index,
            data=inpt.extr['sw_down']['e']['data'].values - inpt.extr['sw_net']['e']['data'].values, columns=['sw_up'])
    inpt.extr['sw_up']['e']['data'][inpt.extr['sw_up']['e']['data'] < 0.] = np.nan
    del inpt.extr['sw_net']['e']['data']

    # ERA5-LAND
    read_era5_land('sw_net')
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
    read_carra(inpt.var)

    # ERA5
    read_era5(inpt.var)
    if inpt.extr[inpt.var]['l']['data'].empty:
        read_era5('temp')
    calc_rh_from_tdp()

    # ERA5-LAND
    read_era5_land(inpt.var)
    if inpt.extr[inpt.var]['l']['data'].empty:
        read_era5('temp')
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
    read_carra(inpt.var)
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data'] / 100.
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] <= 900] = np.nan

    # ERA5
    read_era5(inpt.var)
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
    read_carra(inpt.var)

    # ERA5
    read_era5(inpt.var)
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'].values * 100.

    # THAAO
    read_thaao_ceilometer(inpt.var)

    return


def read_temp():
    # CARRA
    read_carra(inpt.var)
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data'] - 273.15

    # ERA5
    read_era5(inpt.var)
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'] - 273.15

    # ERA5-LAND
    read_era5_land(inpt.var)
    inpt.extr[inpt.var]['l']['data'] = inpt.extr[inpt.var]['l']['data'] - 273.15

    # THAAO
    read_thaao_weather(inpt.var)
    inpt.extr[inpt.var]['t']['data'] = inpt.extr[inpt.var]['t']['data'] - 273.15

    # THAAO2
    read_aws_ecapac(inpt.var)
    return


def read_wind():
    # CARRA
    read_carra('winds')
    read_carra('windd')

    # ERA5
    read_era5('windu')
    read_era5('windv')
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
    # read_era5_land('windu')
    # read_era5_land('windv')
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
    return


# def extract_values(fn, year):
#     if not os.path.exists(os.path.join(inpt.basefol_c, fn + str(year) + '.nc')):
#         try:
#             filen = os.path.join(inpt.basefol_t, 'reanalysis', 'carra', '_'.join(fn.split('_')[1:]) + str(year) + '.nc')
#             NC = xr.open_dataset(str(filen), decode_cf=True, decode_times=True)
#
#             # tmp = NC.sel(x=y, y=x, method='nearest')
#         except FileNotFoundError:
#             print(f'cannot find {filen}')
#
#     return f'thaao_{fn}'


def read():
    """

    :return:
    """
    if inpt.var == 'alb':
        return read_alb()
    if inpt.var == 'cbh':
        return read_cbh()
    if inpt.var == 'msl_pres':
        return read_msl_pres()
    if inpt.var == 'lwp':
        return read_lwp()
    if inpt.var == 'lw_down':
        return read_lw_down()
    if inpt.var == 'lw_up':
        return read_lw_up()
    if inpt.var == 'precip':
        return read_precip()
    if inpt.var == 'rh':
        return read_rh()
    if inpt.var == 'surf_pres':
        return read_surf_pres()
    if inpt.var == 'sw_down':
        return read_sw_down()
    if inpt.var == 'sw_up':
        return read_sw_up()
    if inpt.var == 'tcc':
        return read_tcc()
    if inpt.var == 'temp':
        return read_temp()
    if inpt.var in ['winds', 'windd']:
        return read_wind()
