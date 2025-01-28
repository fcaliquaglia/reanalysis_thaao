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
from metpy.calc import relative_humidity_from_dewpoint, wind_direction, wind_speed
from metpy.units import units

import inputs as inpt


def read_carra(vr=inpt.var):
    for year in inpt.years:
        try:
            c_tmp = pd.read_table(
                    os.path.join(inpt.basefol_c, f'{inpt.extr[vr]['c']['fn']}{year}.txt'),
                    sep='\s+', header=None, skiprows=1, engine='python', skip_blank_lines=True)
            c_tmp[c_tmp == inpt.var_dict['c']['nanval']] = np.nan
            inpt.extr[vr]['c']['data'] = pd.concat([inpt.extr[vr]['c']['data'], c_tmp], axis=0)
            print(f'OK: {inpt.extr[vr]['c']['fn']}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {inpt.extr[vr]['c']['fn']}{year}.txt')
    inpt.extr[vr]['c']['data'].index = pd.to_datetime(
            inpt.extr[vr]['c']['data'][0] + ' ' + inpt.extr[vr]['c']['data'][1], format='%Y-%m-%d %H:%M:%S')
    inpt.extr[vr]['c']['data'] = inpt.extr[vr]['c']['data'][[inpt.extr[vr]['c']['column']]]
    inpt.extr[vr]['c']['data'].columns = [vr]
    return


def read_era5(vr=inpt.var):
    for year in inpt.years:
        try:
            e_tmp = pd.read_table(
                    os.path.join(inpt.basefol_e, f'{inpt.extr[vr]['e']['fn']}{year}.txt'),
                    sep='\s+', header=None, skiprows=1, engine='python')
            e_tmp[e_tmp == inpt.var_dict['e']['nanval']] = np.nan
            inpt.extr[vr]['e']['data'] = pd.concat([inpt.extr[vr]['e']['data'], e_tmp], axis=0)
            print(f'OK: {inpt.extr[vr]['e']['fn']}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {inpt.extr[vr]['e']['fn']}{year}.txt')
    inpt.extr[vr]['e']['data'].index = pd.to_datetime(
            inpt.extr[vr]['e']['data'][0] + ' ' + inpt.extr[vr]['e']['data'][1], format='%Y-%m-%d %H:%M:%S')
    inpt.extr[vr]['e']['data'] = inpt.extr[vr]['e']['data'][[inpt.extr[vr]['e']['column']]]
    inpt.extr[vr]['e']['data'].columns = [vr]
    return


def read_era5_land(vr=inpt.var):
    for year in inpt.years:
        try:
            l_tmp = pd.read_table(
                    os.path.join(inpt.basefol_l, f'{inpt.extr[vr]['l']['fn']}{year}.txt'),
                    sep='\s+', header=None, skiprows=1, engine='python')
            l_tmp[l_tmp == inpt.var_dict['l']['nanval']] = np.nan
            inpt.extr[vr]['l']['data'] = pd.concat([inpt.extr[vr]['l']['data'], l_tmp], axis=0)
            print(f'OK: {inpt.extr[vr]['l']['fn']}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {inpt.extr[vr]['l']['fn']}{year}.txt')
    inpt.extr[vr]['l']['data'].index = pd.to_datetime(
            inpt.extr[vr]['l']['data'][0] + ' ' + inpt.extr[vr]['l']['data'][1], format='%Y-%m-%d %H:%M:%S')
    inpt.extr[vr]['l']['data'] = inpt.extr[vr]['l']['data'][[inpt.extr[vr]['l']['column']]]
    inpt.extr[vr]['l']['data'].columns = [vr]
    return


def read_thaao_weather(vr=inpt.var):
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


def read_thaao_rad(vr=inpt.var):
    for i in inpt.aws_rad_daterange[inpt.aws_rad_daterange.year.isin(inpt.years)]:
        i_fmt = int(i.strftime('%Y'))
        try:
            t_tmp = pd.read_table(
                    os.path.join(inpt.basefol_t, 'thaao_rad', f'{inpt.extr[vr]['t']['fn']}{i_fmt}_5MIN.dat'),
                    engine='python',
                    skiprows=None, header=0, decimal='.', sep='\s+')
            tmp = np.empty(t_tmp['JDAY_UT'].shape, dtype=dt.datetime)
            for ii, el in enumerate(t_tmp['JDAY_UT']):
                new_jd_ass = el + julian.to_jd(dt.datetime(i_fmt - 1, 12, 31, 0, 0), fmt='jd')
                tmp[ii] = julian.from_jd(new_jd_ass, fmt='jd')
                tmp[ii] = tmp[ii].replace(microsecond=0)
            t_tmp.index = pd.DatetimeIndex(tmp)
            t_tmp = t_tmp[[inpt.extr[vr]['t']['column']]]
            inpt.extr[vr]['t']['data'] = pd.concat([inpt.extr[vr]['t']['data'], t_tmp], axis=0)
            print(f'OK: {inpt.extr[vr]['t']['fn']}{i_fmt}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {inpt.extr[vr]['t']['fn']}{i_fmt}.txt')
    inpt.extr[vr]['t']['data'].columns = [vr]
    return


def read_thaao_hatpro(vr=inpt.var):
    for i in inpt.aws_hatpro_daterange[inpt.aws_hatpro_daterange.year.isin(inpt.years)]:
        i_fmt = int(i.strftime('%Y'))
        try:
            t1_tmp = pd.read_table(
                    os.path.join(
                            inpt.basefol_t, 'thaao_hatpro', 'definitivi_da_giando',
                            f'{inpt.extr[vr]['t1']['fn']}{i_fmt}',
                            f'{inpt.extr[vr]['t1']['fn']}{i_fmt}.DAT'), sep='\s+', engine='python', header=None,
                    skiprows=1)
            t1_tmp.columns = ['JD_rif', f'{vr.upper()}', f'STD_{vr.upper()}', 'RF', 'N']
            tmp = np.empty(t1_tmp['JD_rif'].shape, dtype=dt.datetime)
            for ii, el in enumerate(t1_tmp['JD_rif']):
                new_jd_ass = el + julian.to_jd(dt.datetime(i_fmt - 1, 12, 31, 0, 0), fmt='jd')
                tmp[ii] = julian.from_jd(new_jd_ass, fmt='jd')
                tmp[ii] = tmp[ii].replace(microsecond=0)
            t1_tmp.index = pd.DatetimeIndex(tmp)
            t1_tmp.drop(columns=['JD_rif', f'STD_{vr.upper()}', 'RF', 'N'], axis=1, inplace=True)
            inpt.extr[vr]['t1']['data'] = pd.concat([inpt.extr[vr]['t1']['data'], t1_tmp], axis=0)
            print(f'OK: {inpt.extr[vr]['t1']['fn']}{i_fmt}.DAT')
        except FileNotFoundError:
            print(f'NOT FOUND: {inpt.extr[vr]['t1']['fn']}{i_fmt}.DAT')
    inpt.extr[vr]['t1']['data'].columns = [vr]
    return


def read_thaao_ceilometer(vr=inpt.var):
    for i in inpt.aws_ceilometer_daterange[inpt.aws_ceilometer_daterange.year.isin(inpt.years)]:
        i_fmt = i.strftime('%Y%m%d')
        try:
            t_tmp = pd.read_table(
                    os.path.join(
                            inpt.basefol_t_elab, 'thaao_ceilometer_elab', 'medie_tat_rianalisi',
                            f'{i_fmt}{inpt.extr[vr]['t']['fn']}.txt'), skipfooter=0, sep='\s+', header=0,
                    skiprows=9,
                    engine='python')
            t_tmp[t_tmp == inpt.var_dict['t']['nanval']] = np.nan
            inpt.extr[vr]['t']['data'] = pd.concat([inpt.extr[vr]['t']['data'], t_tmp], axis=0)
            print(f'OK: {i_fmt}{inpt.extr[vr]['t']['fn']}.txt')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f'NOT FOUND: {i_fmt}{inpt.extr[vr]['t']['fn']}.txt')
    inpt.extr[vr]['t']['data']
    inpt.extr[vr]['t']['data'].index = pd.to_datetime(
            inpt.extr[vr]['t']['data']['#'] + ' ' + inpt.extr[vr]['t']['data']['date[y-m-d]time[h:m:s]'],
            format='%Y-%m-%d %H:%M:%S')
    inpt.extr[vr]['t']['data'].index.name = 'datetime'
    inpt.extr[vr]['t']['data'] = inpt.extr[vr]['t']['data'].iloc[:, :].filter(
            [inpt.extr[vr]['t']['column']]).astype(float)
    inpt.extr[vr]['t']['data'].columns = [vr]

    return


def read_aws_ecapac(vr=inpt.var):
    for i in inpt.aws_ecapac_daterange[inpt.aws_ecapac_daterange.year.isin(inpt.years)]:
        i_fmt = i.strftime('%Y_%m_%d')
        try:
            file = os.path.join(
                    inpt.basefol_t, 'thaao_ecapac_aws_snow', 'AWS_ECAPAC', i.strftime('%Y'),
                    f'{inpt.extr[vr]['t2']['fn']}{i_fmt}_00_00.dat')
            t2_tmp = pd.read_csv(
                    file, skiprows=[0, 3], header=0, decimal='.', delimiter=',', engine='python',
                    index_col='TIMESTAMP').iloc[1:, :]
            inpt.extr[vr]['t2']['data'] = pd.concat([inpt.extr[vr]['t2']['data'], t2_tmp], axis=0)
            print(f'OK: {inpt.extr[vr]['t2']['fn']}{i_fmt}_00_00.dat')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f'NOT_FOUND: {inpt.extr[vr]['t2']['fn']}{i_fmt}_00_00.dat')
    inpt.extr[vr]['t2']['data'].index = pd.DatetimeIndex(inpt.extr[vr]['t2']['data'].index)
    inpt.extr[vr]['t2']['data'].index.name = 'datetime'
    inpt.extr[vr]['t2']['data'] = inpt.extr[vr]['t2']['data'].iloc[:, :].filter(
            [inpt.extr[vr]['t2']['column']]).astype(float)
    inpt.extr[vr]['t2']['data'].columns = [vr]
    return


def read_alb():
    # CARRA
    read_carra()
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data'] / 100.
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] <= 0.] = np.nan

    # ERA5
    read_era5()
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] <= 0.] = np.nan

    # ERA5-LAND
    read_era5_land()
    inpt.extr[inpt.var]['l']['data'][inpt.extr[inpt.var]['l']['data'] <= 0.] = np.nan

    # THAAO
    read_thaao_rad()

    return


def read_cbh():
    # CARRA
    read_carra()

    # ERA5
    read_era5()

    # THAAO
    read_thaao_ceilometer()

    return


def read_lwp():
    # CARRA
    read_carra()
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data'] * 10000000
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] < 0.01] = np.nan
    # c[c < 15] = 0

    # ERA5
    read_era5()
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'] * 1000
    inpt.extr[inpt.var]['e']['data'][inpt.extr[inpt.var]['e']['data'] < 0.01] = np.nan
    # e[e < 15] = 0

    # THAAO1
    read_thaao_hatpro()
    inpt.extr[inpt.var]['t1']['data'][inpt.extr[inpt.var]['t1']['data'] < 0.01] = np.nan
    # t1[t1 < 15] = 0

    return


def read_rad():
    # CARRA
    read_carra()
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] < 0.] = np.nan
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data'] / inpt.var_dict['c']['rad_conv_factor']

    # ERA5
    read_era5()
    inpt.extr[inpt.var]['e']['data'][inpt.extr[inpt.var]['e']['data'] < 0.] = np.nan
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'] / inpt.var_dict['e']['rad_conv_factor']

    # ERA5_LAND
    read_era5_land()
    inpt.extr[inpt.var]['l']['data'][inpt.extr[inpt.var]['l']['data'] < 0.] = np.nan
    inpt.extr[inpt.var]['l']['data'] = inpt.extr[inpt.var]['l']['data'] / inpt.var_dict['l']['rad_conv_factor']

    # THAAO
    if inpt.var not in ['sw_net', 'lw_net']:
        read_thaao_rad()
        inpt.extr[inpt.var]['t']['data'][inpt.extr[inpt.var]['t']['data'] < 0.] = np.nan


def read_lw_up():
    # CARRA
    fn1 = 'thaao_carra_surface_net_thermal_radiation_'
    fn2 = 'thaao_carra_thermal_surface_radiation_downwards_'

    for yy, year in enumerate(inpt.years):

        try:
            c_tmp = pd.read_table(
                    os.path.join(inpt.basefol_c, f'{fn1}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=2,
                    engine='python')[[0, 1, 4]]
            c_n = pd.concat([c_n, c_tmp], axis=0)
            print(f'OK: {fn1}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn1}{year}.txt')
    c_n.index = pd.to_datetime(c_n[0] + ' ' + c_n[1], format='%Y-%m-%d %H:%M:%S')
    c_n.drop(columns=[0, 1], inplace=True)
    c_n[2] = c_n.values / 3600.
    c_n.drop(columns=[4], inplace=True)
    c_n.columns = ['surface_net_thermal_radiation']

    for yy, year in enumerate(inpt.years):

        try:
            c_tmp = pd.read_table(
                    os.path.join(inpt.basefol_c, f'{fn2}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=2,
                    engine='python')[[0, 1, 4]]
            c_d = pd.concat([c_d, c_tmp], axis=0)
            print(f'OK: {fn2}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn2}{year}.txt')
    c_d.index = pd.to_datetime(c_d[0] + ' ' + c_d[1], format='%Y-%m-%d %H:%M:%S')
    c_d.drop(columns=[0, 1], inplace=True)
    c_d[2] = c_d.values / 3600.
    c_d.drop(columns=[4], inplace=True)
    c_d.columns = ['surface_thermal_radiation_downwards']

    c = pd.concat([c_n, c_d], axis=1)

    c['surface_thermal_radiation_upwards'] = c['surface_thermal_radiation_downwards'] - c[
        'surface_net_thermal_radiation']
    c.drop(columns=['surface_net_thermal_radiation', 'surface_thermal_radiation_downwards'], inplace=True)
    c.columns = [inpt.var]
    # cleaning data
    c[c < 0.] = np.nan

    # ERA5
    fn1 = 'thaao_era5_surface_net_thermal_radiation_'
    fn2 = 'thaao_era5_surface_thermal_radiation_downwards_'
    for yy, year in enumerate(inpt.years):
        try:
            e_tmp = pd.read_table(
                    os.path.join(inpt.basefol_e, f'{fn1}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
                    engine='python')
            e_tmp[e_tmp == inpt.var_dict['e']['nanval']] = np.nan
            e_n = pd.concat([e_n, e_tmp], axis=0)
            print(f'OK: {fn1}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn1}{year}.txt')
    e_n.index = pd.to_datetime(e_n[0] + ' ' + e_n[1], format='%Y-%m-%d %H:%M:%S')
    e_n.drop(columns=[0, 1], inplace=True)
    e_n[2] = e_n.values / 3600.  # originele in J*m-2
    e_n.columns = ['surface_net_thermal_radiation']

    for yy, year in enumerate(inpt.years):
        try:
            e_tmp = pd.read_table(
                    os.path.join(inpt.basefol_e, f'{fn2}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
                    engine='python')
            e_tmp[e_tmp == inpt.var_dict['e']['nanval']] = np.nan
            e_d = pd.concat([e_d, e_tmp], axis=0)
            print(f'OK: {fn2}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn2}{year}.txt')
    e_d.index = pd.to_datetime(e_d[0] + ' ' + e_d[1], format='%Y-%m-%d %H:%M:%S')
    e_d.drop(columns=[0, 1], inplace=True)
    e_d[2] = e_d.values / 3600.  # originele in J*m-2
    e_d.columns = ['surface_thermal_radiation_downwards']

    e = pd.concat([e_n, e_d], axis=1)

    e['surface_thermal_radiation_upwards'] = e['surface_thermal_radiation_downwards'] - e[
        'surface_net_thermal_radiation']
    e.drop(columns=['surface_net_thermal_radiation', 'surface_thermal_radiation_downwards'], inplace=True)
    e.columns = [inpt.var]
    # cleaning data
    e[e < 0.] = np.nan

    # THAAO
    read_thaao_rad()
    inpt.extr[inpt.var]['t']['data'][inpt.extr[inpt.var]['t']['data'] < 0.] = np.nan

    return


def read_msl_pres():
    # CARRA
    read_carra()

    return


def read_precip():
    # CARRA
    read_carra()

    # ERA5
    read_era5()
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'].values * 1000.

    # THAAO2
    read_aws_ecapac()

    return


def read_rh():
    # CARRA
    read_carra()

    # ERA5
    read_era5()
    if inpt.extr[inpt.var]['l']['data'].empty:
        read_era5(vr='temp')
    calc_rh_from_tdp()

    # ERA5-LAND
    read_era5_land()
    if inpt.extr[inpt.var]['l']['data'].empty:
        read_era5(vr='temp')
    calc_rh_from_tdp()

    # e.g. l_td[l_td_tmp == -32767.0] = np.nan

    # THAAO2
    read_thaao_weather()
    read_aws_ecapac()
    read_thaao_weather()

    return


def calc_rh_from_tdp():
    # TODO not working

    e = pd.concat([inpt.extr[inpt.var]['t']['data'], e_t], axis=1)

    e['rh'] = relative_humidity_from_dewpoint(e['e_t'].values * units.K, e['e_td'].values * units.K).to('percent')
    inpt.extr[inpt.var]['e']['data'].drop(columns=['e_t', 'e_td'], inplace=True)
    inpt.extr[inpt.var]['e']['data'].columns = [inpt.var]

    return


def read_surf_pres():
    # CARRA
    read_carra()
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data'] / 100.
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] <= 900] = np.nan

    # ERA5
    read_era5()
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'] / 100.
    inpt.extr[inpt.var]['e']['data'][inpt.extr[inpt.var]['e']['data'] <= 900] = np.nan

    # THAAO
    read_thaao_weather()
    inpt.extr[inpt.var]['t']['data'][inpt.extr[inpt.var]['t']['data'] <= 900] = np.nan
    inpt.extr[inpt.var]['t']['data'].loc['2021-10-11 00:00:00':'2021-10-19 00:00:00'] = np.nan
    inpt.extr[inpt.var]['t']['data'].loc['2024-4-26 00:00:00':'2024-5-4 00:00:00'] = np.nan

    # THAAO2
    read_aws_ecapac()

    return


def read_sw_up():
    # CARRA
    fn1 = 'thaao_carra_surface_net_solar_radiation_'
    fn2 = 'thaao_carra_surface_solar_radiation_downwards_'

    for yy, year in enumerate(inpt.years):
        # fn = extract_values(fn, year)
        try:
            c_tmp = pd.read_table(
                    os.path.join(inpt.basefol_c, f'{fn1}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=2,
                    engine='python')[[0, 1, 4]]
            c_n = pd.concat([c_n, c_tmp], axis=0)
            print(f'OK: {fn1}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn1}{year}.txt')
    c_n.index = pd.to_datetime(c_n[0] + ' ' + c_n[1], format='%Y-%m-%d %H:%M:%S')
    c_n.drop(columns=[0, 1], inplace=True)
    c_n[2] = c_n.values / 3600.
    c_n.drop(columns=[4], inplace=True)
    c_n.columns = ['surface_net_solar_radiation']

    for yy, year in enumerate(inpt.years):
        # fn = extract_values(fn, year)
        try:
            c_tmp = pd.read_table(
                    os.path.join(inpt.basefol_c, f'{fn2}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=2,
                    engine='python')[[0, 1, 4]]
            c_d = pd.concat([c_d, c_tmp], axis=0)
            print(f'OK: {fn2}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn2}{year}.txt')
    c_d.index = pd.to_datetime(c_d[0] + ' ' + c_d[1], format='%Y-%m-%d %H:%M:%S')
    c_d.drop(columns=[0, 1], inplace=True)
    c_d[2] = c_d.values / 3600.
    c_d.drop(columns=[4], inplace=True)
    c_d.columns = ['surface_solar_radiation_downwards']

    c = pd.concat([c_n, c_d], axis=1)

    c['surface_solar_radiation_upwards'] = c['surface_solar_radiation_downwards'] - c['surface_net_solar_radiation']
    c.drop(columns=['surface_net_solar_radiation', 'surface_solar_radiation_downwards'], inplace=True)
    c.columns = [inpt.var]
    # cleaning data
    c[c < 0.] = np.nan

    # ERA5
    fn1 = 'thaao_era5_surface_net_solar_radiation_'
    fn2 = 'thaao_era5_surface_solar_radiation_downwards_'
    for yy, year in enumerate(inpt.years):
        try:
            e_tmp = pd.read_table(
                    os.path.join(inpt.basefol_e, f'{fn1}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
                    engine='python')
            e_tmp[e_tmp == inpt.var_dict['e']['nanval']] = np.nan
            e_n = pd.concat([e_n, e_tmp], axis=0)
            print(f'OK: {fn1}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn1}{year}.txt')
    e_n.index = pd.to_datetime(e_n[0] + ' ' + e_n[1], format='%Y-%m-%d %H:%M:%S')
    e_n.drop(columns=[0, 1], inplace=True)
    e_n[2] = e_n.values / 3600.  # originele in J*m-2
    e_n.columns = ['surface_net_solar_radiation']

    for yy, year in enumerate(inpt.years):
        try:
            e_tmp = pd.read_table(
                    os.path.join(inpt.basefol_e, f'{fn2}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
                    engine='python')
            e_tmp[e_tmp == inpt.var_dict['e']['nanval']] = np.nan
            e_d = pd.concat([e_d, e_tmp], axis=0)
            print(f'OK: {fn2}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn2}{year}.txt')
    e_d.index = pd.to_datetime(e_d[0] + ' ' + e_d[1], format='%Y-%m-%d %H:%M:%S')
    e_d.drop(columns=[0, 1], inplace=True)
    e_d[2] = e_d.values / 3600.  # originele in J*m-2
    e_d.columns = ['surface_solar_radiation_downwards']

    e = pd.concat([e_n, e_d], axis=1)

    e['surface_solar_radiation_upwards'] = e['surface_solar_radiation_downwards'] - e['surface_net_solar_radiation']
    e.drop(columns=['surface_net_solar_radiation', 'surface_solar_radiation_downwards'], inplace=True)
    e.columns = [inpt.var]
    # cleaning data
    e[e < 0.] = np.nan

    read_thaao_rad()
    inpt.extr[inpt.var]['t']['data'][inpt.extr[inpt.var]['t']['data'] < 0.] = np.nan

    return


def read_tcc():
    # CARRA
    read_carra()

    # ERA5
    read_era5()
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'].values * 100.

    # THAAO
    read_thaao_ceilometer()

    return


def read_temp():
    # CARRA
    read_carra()
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data'] - 273.15

    # ERA5
    read_era5()
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'] - 273.15

    # ERA5-LAND
    read_era5_land()
    inpt.extr[inpt.var]['l']['data'] = inpt.extr[inpt.var]['l']['data'] - 273.15

    # THAAO
    read_thaao_weather()
    inpt.extr[inpt.var]['t']['data'] = inpt.extr[inpt.var]['t']['data'] - 273.15

    # THAAO2
    read_aws_ecapac()
    return


# def read_windd():
#     # CARRA
#     read_carra()
#
#     # ERA5
#     fn_u = 'thaao_era5_10m_u_component_of_wind_'
#     fn_v = 'thaao_era5_10m_v_component_of_wind_'
#     e_u = pd.DataFrame()
#     e_v = pd.DataFrame()
#     for yy, year in enumerate(inpt.years):
#         try:
#             e_u_tmp = pd.read_table(
#                     os.path.join(inpt.basefol_e, f'{fn_u}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
#                     engine='python')
#             e_u = pd.concat([e_u, e_u_tmp], axis=0)
#             print(f'OK: {fn_u}{year}.txt')
#         except FileNotFoundError:
#             print(f'NOT FOUND: {fn_u}{year}.txt')
#     e_u.drop(columns=[0, 1], inplace=True)
#
#     for yy, year in enumerate(inpt.years):
#         try:
#             e_v_tmp = pd.read_table(
#                     os.path.join(inpt.basefol_e, f'{fn_v}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
#                     engine='python')
#             e_v = pd.concat([e_v, e_v_tmp], axis=0)
#             print(f'OK: {fn_v}{year}.txt')
#         except FileNotFoundError:
#             print(f'NOT FOUND: {fn_v}{year}.txt')
#     e_v.index = pd.to_datetime(e_v[0] + ' ' + e_v[1], format='%Y-%m-%d %H:%M:%S')
#     e_v.drop(columns=[0, 1], inplace=True)
#
#     e_wd = wind_direction(e_u.values * units('m/s'), e_v.values * units('m/s'))
#     e.index = e_v.index
#     e[inpt.var] = e_wd.magnitude
#     e.columns = [inpt.var]
#
#     return


def read_wind():
    # CARRA
    read_carra(vr='winds')
    read_carra(vr='windd')

    # ERA5
    read_era5(vr='windu')
    read_era5(vr='windv')
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
    # read_era5_land(vr='windu')
    # read_era5_land(vr='windv')
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
    read_aws_ecapac(vr='winds')
    read_aws_ecapac(vr='windd')

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
        return read_rad()
    if inpt.var == 'lw_net':
        return read_rad()
    if inpt.var == 'lw_up':
        return read_rad()
    if inpt.var == 'precip':
        return read_precip()
    if inpt.var == 'rh':
        return read_rh()
    if inpt.var == 'surf_pres':
        return read_surf_pres()
    if inpt.var == 'sw_down':
        return read_rad()
    if inpt.var == 'sw_net':
        return read_rad()
    if inpt.var == 'sw_up':
        return read_sw_up()
    if inpt.var == 'tcc':
        return read_tcc()
    if inpt.var == 'temp':
        return read_temp()
    if (inpt.var == 'winds') | (inpt.var == 'windd'):
        return read_wind()
