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

import julian
import xarray as xr
from metpy.calc import dewpoint_from_relative_humidity, precipitable_water, relative_humidity_from_dewpoint, \
    wind_direction, wind_speed
from metpy.units import units

import inputs as inpt
from inputs import *


def read_carra(vr):
    for year in years:
        try:
            c_tmp = pd.read_table(
                    os.path.join(basefol_c, f'{extr[vr]['c']['fn']}{year}.txt'), skipfooter=1, sep='\s+', header=None,
                    skiprows=1, engine='python')
            extr[vr]['c']['data'] = pd.concat([extr[vr]['c']['data'], c_tmp], axis=0)
            print(f'OK: {extr[vr]['c']['fn']}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {extr[vr]['c']['fn']}{year}.txt')
    extr[vr]['c']['data'].index = pd.to_datetime(
            extr[vr]['c']['data'][0] + ' ' + extr[vr]['c']['data'][1], format='%Y-%m-%d %H:%M:%S')
    extr[vr]['c']['data'].drop(columns=[0, 1], inplace=True)
    extr[vr]['c']['data'].columns = [vr]
    return


def read_era5(vr):
    for year in years:
        try:
            e_tmp = pd.read_table(
                    os.path.join(basefol_e, f'{extr[vr]['e']['fn']}{year}.txt'), skipfooter=1, sep='\s+', header=None,
                    skiprows=1, engine='python')
            e_tmp[e_tmp == -32767.0] = np.nan
            extr[vr]['e']['data'] = pd.concat([extr[vr]['e']['data'], e_tmp], axis=0)

            print(f'OK: {extr[vr]['e']['fn']}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {extr[vr]['e']['fn']}{year}.txt')
    extr[vr]['e']['data'].index = pd.to_datetime(
            extr[vr]['e']['data'][0] + ' ' + extr[vr]['e']['data'][1], format='%Y-%m-%d %H:%M:%S')
    extr[vr]['e']['data'].drop(columns=[0, 1], inplace=True)
    extr[vr]['e']['data'].columns = [vr]
    return


def read_era5_land(vr):
    for year in years:
        try:
            l_tmp = pd.read_table(
                    os.path.join(basefol_l, f'{extr[vr]['l']['fn']}{year}.txt'), skipfooter=1, sep='\s+', header=None,
                    skiprows=1, engine='python')
            l_tmp[l_tmp == -32767.0] = np.nan
            extr[vr]['l']['data'] = pd.concat([extr[vr]['l']['data'], l_tmp], axis=0)

            print(f'OK: {extr[vr]['l']['fn']}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {extr[vr]['l']['fn']}{year}.txt')
    extr[vr]['l']['data'].index = pd.to_datetime(
            extr[vr]['l']['data'][0] + ' ' + extr[vr]['l']['data'][1], format='%Y-%m-%d %H:%M:%S')
    extr[vr]['l']['data'].drop(columns=[0, 1], inplace=True)
    extr[vr]['l']['data'].columns = [vr]
    return


def read_thaao_weather(vr, drop_param):
    try:
        extr[vr]['t']['data'] = xr.open_dataset(
                os.path.join(basefol_t, 'thaao_meteo', f'{extr[vr]['t']['fn']}.nc'), engine='netcdf4').to_dataframe()
        print(f'OK: {extr[vr]['t']['fn']}.nc')
    except FileNotFoundError:
        print(f'NOT FOUND: {extr[vr]['t']['fn']}.nc')
    extr[vr]['t']['data'].drop(columns=drop_param, inplace=True)
    extr[vr]['t']['data'].columns = [vr]
    return


def read_thaao_vespa(vr):
    try:
        extr[vr]['t']['data'] = pd.read_table(
                os.path.join(basefol_t, 'thaao_vespa', f'{extr[vr]['t']['fn']}.txt'), skipfooter=1, sep='\s+',
                header=None, skiprows=1, engine='python')
        print(f'OK: {extr[vr]['t']['fn']}.txt')
    except FileNotFoundError:
        print(f'NOT FOUND: {extr[vr]['t']['fn']}.txt')
    extr[vr]['t']['data'].index = pd.to_datetime(
            extr[vr]['t']['data'][0] + ' ' + extr[vr]['t']['data'][1], format='%Y-%m-%d %H:%M:%S')
    extr[vr]['t']['data'].drop(columns=[0, 1, 3, 4, 5], inplace=True)
    extr[vr]['t']['data'].columns = [inpt.var_in_use]
    return


def read_thaao_hatpro(vr):
    for yy, year in enumerate(years):
        try:
            t1_tmp = pd.read_table(
                    os.path.join(
                            basefol_t, 'thaao_hatpro', 'definitivi_da_giando', f'{extr[vr]['t']['fn']}{year}',
                            f'{extr[vr]['t']['fn']}{year}.DAT'), sep='\s+', engine='python', header=None, skiprows=1)
            t1_tmp.columns = ['JD_rif', 'IWV', 'STD_IWV', 'RF', 'N']
            tmp = np.empty(t1_tmp['JD_rif'].shape, dtype=dt.datetime)
            for ii, el in enumerate(t1_tmp['JD_rif']):
                new_jd_ass = el + julian.to_jd(dt.datetime(year - 1, 12, 31, 0, 0), fmt='jd')
                tmp[ii] = julian.from_jd(new_jd_ass, fmt='jd')
                tmp[ii] = tmp[ii].replace(microsecond=0)
            t1_tmp.index = pd.DatetimeIndex(tmp)
            t1_tmp.drop(columns=['JD_rif', 'STD_IWV', 'RF', 'N'], axis=1, inplace=True)
            extr[vr]['t']['data'] = pd.concat([extr[vr]['t']['data'], t1_tmp], axis=0)
            print(f'OK: {extr[vr]['t']['fn']}{year}.DAT')
        except FileNotFoundError:
            print(f'NOT FOUND: {extr[vr]['t']['fn']}{year}.DAT')
    extr[vr]['t']['data']['IWV'] = extr[vr]['t']['data']['IWV'].values
    extr[vr]['t']['data'].columns = [inpt.var_in_use]
    return


def read_thaao_ceilometer( vr, param):
    for i in ceilometer_daterange:
        i_fmt = i.strftime('%Y%m%d')
        try:
            t_tmp = pd.read_table(
                    os.path.join(
                            basefol_t_elab, 'thaao_ceilometer_elab', 'medie_tat_rianalisi',
                            f'{i_fmt}{extr[vr]['t']['fn']}.txt'), skipfooter=0, sep='\s+', header=0, skiprows=9,
                    engine='python')
            t_tmp[t_tmp == -9999.9] = np.nan
            t_tmp = pd.concat([extr[vr]['t']['data'], t_tmp], axis=0)
            print(f'OK: {i_fmt}{extr[vr]['t']['fn']}.txt')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f'NOT FOUND: {i_fmt}{extr[vr]['t']['fn']}.txt')
    extr[vr]['t']['data']
    extr[vr]['t']['data'].index = pd.to_datetime(
            extr[vr]['t']['data']['#'] + ' ' + extr[vr]['t']['data']['date[y-m-d]time[h:m:s]'],
            format='%Y-%m-%d %H:%M:%S')
    extr[vr]['t']['data'].index.name = 'datetime'
    extr[vr]['t']['data'] = extr[vr]['t']['data'].iloc[:, :].filter([param]).astype(float)
    extr[vr]['t']['data'].columns = [inpt.var_in_use]

    return


def read_aws_ecapac(vr, param):
    fn = 'AWS_THAAO_'
    for i in aws_ecapac_daterange[aws_ecapac_daterange.year.isin(years)]:
        i_fmt = i.strftime('%Y_%m_%d')
        try:
            file = os.path.join(
                    basefol_t, 'thaao_ecapac_aws_snow', 'AWS_ECAPAC', i.strftime('%Y'), f'{fn}{i_fmt}_00_00.dat')
            t2_tmp = pd.read_csv(
                    file, skiprows=[0, 3], header=0, decimal='.', delimiter=',', engine='python',
                    index_col='TIMESTAMP').iloc[1:, :]
            extr[vr]['t2']['data'] = pd.concat([extr[vr]['t2']['data'], t2_tmp], axis=0)
            print(f'OK: {fn}{i_fmt}_00_00.dat')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f'NOT_FOUND: {fn}{i_fmt}_00_00.dat')
    extr[vr]['t2']['data'].index = pd.DatetimeIndex(extr[vr]['t2']['data'].index)
    extr[vr]['t2']['data'].index.name = 'datetime'
    extr[vr]['t2']['data'] = extr[vr]['t2']['data'].iloc[:, :].filter([param]).astype(float)
    extr[vr]['t2']['data'].columns = [inpt.var_in_use]
    return


def read_temp():
    read_thaao_weather(inpt.var_in_use, drop_param=['BP_hPa', 'RH_%'])
    extr[inpt.var_in_use]['t']['data']['Air_K'] = extr[inpt.var_in_use]['t']['data'].values - 273.15
    read_aws_ecapac(inpt.var_in_use, param='AirTC')
    read_carra(inpt.var_in_use)
    extr[inpt.var_in_use]['c']['data'][2] = extr[inpt.var_in_use]['c']['data'].values - 273.15
    read_era5(inpt.var_in_use)
    extr[inpt.var_in_use]['e']['data'][2] = extr[inpt.var_in_use]['e']['data'].values - 273.15
    read_era5_land(inpt.var_in_use)
    extr[inpt.var_in_use]['l']['data'][2] = extr[inpt.var_in_use]['l']['data'].values - 273.15

    return


def read_msl_pres():
    read_carra(inpt.var_in_use)

    return


def read_surf_pres():
    # cleanup
    read_carra(inpt.var_in_use)
    extr[inpt.var_in_use]['c']['data'][2] = extr[inpt.var_in_use]['c']['data'][2].values / 100.
    extr[inpt.var_in_use]['c']['data'][extr[inpt.var_in_use]['c']['data'] <= 900] = np.nan

    read_era5(inpt.var_in_use)
    extr[inpt.var_in_use]['e']['data'][2] = extr[inpt.var_in_use]['e']['data'].values / 100.
    extr[inpt.var_in_use]['e']['data'][extr[inpt.var_in_use]['e']['data'] <= 900] = np.nan

    read_thaao_weather(inpt.var_in_use, drop_param=['Air_K', 'RH_%'])
    extr[inpt.var_in_use]['t']['data'][extr[inpt.var_in_use]['e']['data'] <= 900] = np.nan
    extr[inpt.var_in_use]['t']['data'].loc['2021-10-11 00:00:00':'2021-10-19 00:00:00'] = np.nan
    extr[inpt.var_in_use]['t']['data'].loc['2024-4-26 00:00:00':'2024-5-4 00:00:00'] = np.nan

    read_aws_ecapac(inpt.var_in_use, param='BP_mbar')


def read_rh():
    # TODO: variable cleanup
    # e.g. l_td[l_td_tmp == -32767.0] = np.nan

    read_thaao_weather(drop_param=['BP_hPa', 'Air_K'])
    read_aws_ecapac(param='RH')

    read_thaao_weather(drop_param=['BP_hPa', 'Air_K'])
    read_carra()

    read_era5()
    if extr[vr]['l']['data'].empty:
        read_era5(vr='temp')
    calc_rh_from_tdp()

    read_era5_land()
    if extr[vr]['l']['data'].empty:
        read_era5(vr='temp')
    calc_rh_from_tdp()
    return


def calc_rh_from_tdp():
    # TODO not working
    e = pd.concat([extr[vr]['t']['data'], e_t], axis=1)

    e['rh'] = relative_humidity_from_dewpoint(e['e_t'].values * units.K, e['e_td'].values * units.K).to('percent')
    extr[vr]['e']['data'].drop(columns=['e_t', 'e_td'], inplace=True)
    extr[vr]['e']['data'].columns = [inpt.var_in_use]

    return


def read_alb():
    read_carra()

    extr[inpt.var_in_use]['c']['data'][2] = extr[inpt.var_in_use]['c']['data'].values / 100.
    extr[inpt.var_in_use]['c']['data'][extr[inpt.var_in_use]['c']['data'] <= 0.1] = np.nan

    read_era5()
    extr[inpt.var_in_use]['c']['data'][extr[inpt.var_in_use]['c']['data'] <= 0.1] = np.nan

    # # ERA5
    # fn = 'thaao_era5_snow_albedo_'
    # for yy, year in enumerate(years):
    #     try:
    #         t2_tmp = pd.read_table(
    #                 os.path.join(basefol_e, f'{fn}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
    #                 engine='python')
    #         t2_tmp[t2_tmp == -32767.0] = np.nan
    #         t2 = pd.concat([t2, t2_tmp], axis=0)
    #         print(f'OK: {fn}{year}.txt')
    #     except FileNotFoundError:
    #         print(f'NOT FOUND: {fn}{year}.txt')
    # t2.index = pd.to_datetime(t2[0] + ' ' + t2[1], format='%Y-%m-%d %H:%M:%S')
    # t2.drop(columns=[0, 1], inplace=True)
    # t2.columns = [inpt.var_in_use]
    # t2[t2 <= 0.1] = np.nan

    # THAAO
    # TODO: sostituire con questo blocco che prende direttamente dal file MERGED_SW_LW_UP_DW_METEO_YYYY.dat

    # fn = 'MERGED_SW_LW_UP_DW_METEO_'
    # for yy, year in enumerate(years):
    #     try:
    #         t_tmp = pd.read_table(
    #                 os.path.join(basefol_t, 'thaao_rad', f'{fn}{year}_5MIN.dat'), engine='python',
    #                 skiprows=None, header=0, decimal='.', sep='\s+')
    #         tmp = np.empty(t_tmp['JDAY_UT'].shape, dtype=dt.datetime)
    #         for ii, el in enumerate(t_tmp['JDAY_UT']):
    #             new_jd_ass = el + julian.to_jd(dt.datetime(year - 1, 12, 31, 0, 0), fmt='jd')
    #             tmp[ii] = julian.from_jd(new_jd_ass, fmt='jd')
    #             tmp[ii] = tmp[ii].replace(microsecond=0)
    #         t_tmp.index = pd.DatetimeIndex(tmp)
    #         t_tmp.drop(
    #                 ['JDAY_UT', 'JDAY_LOC', 'SZA', 'SW_DOWN', 'SW_UP','PAR_DOWN', 'PAR_UP', 'LW_DOWN', 'LW_UP', 'TBP',
    #                  'ALBEDO_LW', 'ALBEDO_PAR', 'P', 'T', 'RH', 'PE', 'RR2'], axis=1, inplace=True)
    #         t = pd.concat([t, t_tmp], axis=0)
    #         print(f'OK: {fn}{year}.txt')
    #     except FileNotFoundError:
    #         print(f'NOT FOUND: {fn}{year}.txt')
    # t.columns = [inpt.var_in_use]

    fn = 'ALBEDO_SW_'
    for yy, year in enumerate(years):
        try:
            t_tmp = pd.read_table(
                    os.path.join(basefol_t, 'thaao_rad', f'{fn}{year}_5MIN.DAT'), engine='python', skiprows=None,
                    header=0, decimal='.', sep='\s+')
            tmp = np.empty(t_tmp['JDAY_UT'].shape, dtype=dt.datetime)
            for ii, el in enumerate(t_tmp['JDAY_UT']):
                new_jd_ass = el + julian.to_jd(dt.datetime(year - 1, 12, 31, 0, 0), fmt='jd')
                tmp[ii] = julian.from_jd(new_jd_ass, fmt='jd')
                tmp[ii] = tmp[ii].replace(microsecond=0)
            t_tmp.index = pd.DatetimeIndex(tmp)
            t_tmp.drop(['JDAY_UT', 'JDAY_LOC', 'SZA', 'SW_DOWN', 'SW_UP'], axis=1, inplace=True)
            t = pd.concat([t, t_tmp], axis=0)
            print(f'OK: {fn}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn}{year}.txt')
    t.columns = [inpt.var_in_use]
    t[t <= 0.1] = np.nan

    return


def read_iwv():
    read_carra()
    extr[inpt.var_in_use]['c']['data'][extr[inpt.var_in_use]['c']['data'] <= 0] = np.nan

    read_era5()

    read_thaao_vespa()

    read_thaao_hatpro()
    extr[inpt.var_in_use]['t1']['data'][extr[inpt.var_in_use]['t1']['data'] < 0] = np.nan
    extr[inpt.var_in_use]['t1']['data'][extr[inpt.var_in_use]['t1']['data'] > 30] = np.nan

    # RS (sondes)
    for yy, year in enumerate(years):
        try:
            fol_input = os.path.join(basefol_t, 'thaao_rs_sondes', 'txt', f'{year}')
            file_l = os.listdir(fol_input)
            file_l.sort()
            for i in file_l:
                print(i)
                file_date = dt.datetime.strptime(i[9:22], '%Y%m%d_%H%M')
                kw = dict(
                        skiprows=17, skipfooter=1, header=None, delimiter=" ", na_values="nan", na_filter=True,
                        skipinitialspace=False, decimal=".", names=['height', 'pres', 'temp', 'rh'], engine='python',
                        usecols=[0, 1, 2, 3])
                dfs = pd.read_table(os.path.join(fol_input, i), **kw)
                # unphysical values checks
                dfs.loc[(dfs['pres'] > 1013) | (dfs['pres'] < 0), 'pres'] = np.nan
                dfs.loc[(dfs['height'] < 0), 'height'] = np.nan
                dfs.loc[(dfs['temp'] < -100) | (dfs['temp'] > 30), 'temp'] = np.nan
                dfs.loc[(dfs['rh'] < 1.) | (dfs['rh'] > 120), 'rh'] = np.nan
                dfs.dropna(subset=['temp', 'pres', 'rh'], inplace=True)
                dfs.drop_duplicates(subset=['height'], inplace=True)
                # min_pres_ind exclude values recorded during descent
                min_pres = np.nanmin(dfs['pres'])
                min_pres_ind = np.nanmin(np.where(dfs['pres'] == min_pres)[0])
                dfs1 = dfs.iloc[:min_pres_ind]
                dfs2 = dfs1.set_index(['height'])
                rs_iwv = convert_rs_to_iwv(dfs2, 1.01)
                t2_tmp = pd.DataFrame(index=pd.DatetimeIndex([file_date]), data=[rs_iwv.magnitude])
                t2 = pd.concat([t2, t2_tmp], axis=0)
            print(f'OK: year {year}')
        except FileNotFoundError:
            print(f'NOT FOUND: year {year}')
    t2.columns = [inpt.var_in_use]
    # t2.to_csv(os.path.join(basefol_t, 'rs_pwv.txt'), index=True)

    return


def read_winds():
    read_carra()

    # ERA5
    fn_u = 'thaao_era5_10m_u_component_of_wind_'
    fn_v = 'thaao_era5_10m_v_component_of_wind_'
    e_u = pd.DataFrame()
    e_v = pd.DataFrame()
    for yy, year in enumerate(years):
        try:
            e_u_tmp = pd.read_table(
                    os.path.join(basefol_e, f'{fn_u}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
                    engine='python')
            e_u = pd.concat([e_u, e_u_tmp], axis=0)
            print(f'OK: {fn_u}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn_u}{year}.txt')
    e_u.drop(columns=[0, 1], inplace=True)

    for yy, year in enumerate(years):
        try:
            e_v_tmp = pd.read_table(
                    os.path.join(basefol_e, f'{fn_v}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
                    engine='python')
            e_v = pd.concat([e_v, e_v_tmp], axis=0)
            print(f'OK: {fn_v}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn_v}{year}.txt')
    e_v.index = pd.to_datetime(e_v[0] + ' ' + e_v[1], format='%Y-%m-%d %H:%M:%S')
    e_v.drop(columns=[0, 1], inplace=True)

    e_ws = wind_speed(e_u.values * units('m/s'), e_v.values * units('m/s'))

    e.index = e_v.index
    e[inpt.var_in_use] = e_ws.magnitude
    e.columns = [inpt.var_in_use]

    # AWS ECAPAC
    read_aws_ecapac(param='WS_aws')

    return


def read_windd():
    read_carra()

    # ERA5
    fn_u = 'thaao_era5_10m_u_component_of_wind_'
    fn_v = 'thaao_era5_10m_v_component_of_wind_'
    e_u = pd.DataFrame()
    e_v = pd.DataFrame()
    for yy, year in enumerate(years):
        try:
            e_u_tmp = pd.read_table(
                    os.path.join(basefol_e, f'{fn_u}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
                    engine='python')
            e_u = pd.concat([e_u, e_u_tmp], axis=0)
            print(f'OK: {fn_u}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn_u}{year}.txt')
    e_u.drop(columns=[0, 1], inplace=True)

    for yy, year in enumerate(years):
        try:
            e_v_tmp = pd.read_table(
                    os.path.join(basefol_e, f'{fn_v}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
                    engine='python')
            e_v = pd.concat([e_v, e_v_tmp], axis=0)
            print(f'OK: {fn_v}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn_v}{year}.txt')
    e_v.index = pd.to_datetime(e_v[0] + ' ' + e_v[1], format='%Y-%m-%d %H:%M:%S')
    e_v.drop(columns=[0, 1], inplace=True)

    e_wd = wind_direction(e_u.values * units('m/s'), e_v.values * units('m/s'))
    e.index = e_v.index
    e[inpt.var_in_use] = e_wd.magnitude
    e.columns = [inpt.var_in_use]

    # AWS ECAPAC
    read_aws_ecapac(param='WD_aws')

    return


def read_tcc():
    read_carra()
    read_era5()
    extr[inpt.var_in_use]['e']['data'][2] = extr[inpt.var_in_use]['e']['data'].values * 100.

    read_thaao_ceilometer(param='TCC[okt]')


def read_cbh():
    read_carra()

    read_era5()

    read_thaao_ceilometer(param='CBH_L1[m]')
    return


def read_precip():
    read_carra()
    read_era5()
    extr[inpt.var_in_use]['e']['data'][2] = extr[inpt.var_in_use]['e']['data'].values * 1000.

    read_aws_ecapac(param='PR')

    return


def read_lwp():
    read_carra()
    extr[inpt.var_in_use]['c']['data'][2] = extr[inpt.var_in_use]['c']['data'].values * 10000000
    extr[inpt.var_in_use]['c']['data'][extr[inpt.var_in_use]['c']['data'] < 0.01] = np.nan
    # c[c < 15] = 0

    read_era5()
    extr[inpt.var_in_use]['e']['data'][2] = extr[inpt.var_in_use]['e']['data'].values * 1000
    extr[inpt.var_in_use]['e']['data'][extr[inpt.var_in_use]['e']['data'] < 0.01] = np.nan
    # e[e < 15] = 0

    # THAAO (hatpro)
    fn = 'LWP_15_min_'
    for yy, year in enumerate(years):
        try:
            t1_tmp = pd.read_table(
                    os.path.join(
                            basefol_t, 'thaao_hatpro', 'definitivi_da_giando', f'{fn}{year}_SITO',
                            f'{fn}{year}_SITO.dat'), sep='\s+', engine='python')
            tmp = np.empty(t1_tmp['JD_rif'].shape, dtype=dt.datetime)
            for ii, el in enumerate(t1_tmp['JD_rif']):
                new_jd_ass = el + julian.to_jd(dt.datetime(year - 1, 12, 31, 0, 0), fmt='jd')
                tmp[ii] = julian.from_jd(new_jd_ass, fmt='jd')
                tmp[ii] = tmp[ii].replace(microsecond=0)
            t1_tmp.index = pd.DatetimeIndex(tmp)
            t1_tmp['LWP_gm-2'] = t1_tmp['LWP_gm-2'].values
            t1_tmp.drop(columns=['JD_rif', 'RF', 'N', 'STD_LWP'], axis=1, inplace=True)
            t1 = pd.concat([t1, t1_tmp], axis=0)

            print(f'OK: {fn}{year}.dat')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn}{year}.dat')
    t1.columns = [inpt.var_in_use]
    # cleaning HATPRO DATA
    t1[t1 < 0.01] = np.nan
    # t1[t1 < 15] = 0

    return


def read_lw_down():
    # CARRA
    fn = 'thaao_carra_thermal_surface_radiation_downwards_'

    for yy, year in enumerate(years):
        # fn = extract_values(fn, year)
        try:
            c_tmp = pd.read_table(
                    os.path.join(basefol_c, f'{fn}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=2,
                    engine='python')[[0, 1, 4]]
            c = pd.concat([c, c_tmp], axis=0)
            print(f'OK: {fn}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn}{year}.txt')
    c.index = pd.to_datetime(c[0] + ' ' + c[1], format='%Y-%m-%d %H:%M:%S')
    c.drop(columns=[0, 1], inplace=True)
    c[2] = c.values / 3600.
    c.drop(columns=[4], inplace=True)
    c.columns = [inpt.var_in_use]
    c[c < 0.] = np.nan

    # ERA5
    fn = 'thaao_era5_surface_thermal_radiation_downwards_'
    for yy, year in enumerate(years):
        try:
            e_tmp = pd.read_table(
                    os.path.join(basefol_e, f'{fn}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
                    engine='python')
            e_tmp[e_tmp == -32767.0] = np.nan
            e = pd.concat([e, e_tmp], axis=0)
            print(f'OK: {fn}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn}{year}.txt')
    e.index = pd.to_datetime(e[0] + ' ' + e[1], format='%Y-%m-%d %H:%M:%S')
    e.drop(columns=[0, 1], inplace=True)
    e[2] = e.values / 3600.  # originale in J*m-2
    e.columns = [inpt.var_in_use]
    # cleaning data
    e[e < 0.] = np.nan

    # THAAO
    fn = 'MERGED_SW_LW_UP_DW_METEO_'
    for yy, year in enumerate(years):
        try:
            t_tmp = pd.read_table(
                    os.path.join(basefol_t, 'thaao_rad', f'{fn}{year}_5MIN.dat'), engine='python', skiprows=None,
                    header=0, decimal='.', sep='\s+')
            tmp = np.empty(t_tmp['JDAY_UT'].shape, dtype=dt.datetime)
            for ii, el in enumerate(t_tmp['JDAY_UT']):
                new_jd_ass = el + julian.to_jd(dt.datetime(year - 1, 12, 31, 0, 0), fmt='jd')
                tmp[ii] = julian.from_jd(new_jd_ass, fmt='jd')
                tmp[ii] = tmp[ii].replace(microsecond=0)
            t_tmp.index = pd.DatetimeIndex(tmp)
            t_tmp.drop(
                    ['JDAY_UT', 'JDAY_LOC', 'SZA', 'SW_DOWN', 'PAR_DOWN', 'PAR_UP', 'SW_UP', 'LW_UP', 'TBP',
                     'ALBEDO_SW', 'ALBEDO_LW', 'ALBEDO_PAR', 'P', 'T', 'RH', 'PE', 'RR2'], axis=1, inplace=True)
            t = pd.concat([t, t_tmp], axis=0)
            print(f'OK: {fn}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn}{year}.txt')
    t.columns = [inpt.var_in_use]
    # cleaning data
    t[t < 0.] = np.nan

    return


def read_lw_up():
    # CARRA
    fn1 = 'thaao_carra_surface_net_thermal_radiation_'
    fn2 = 'thaao_carra_thermal_surface_radiation_downwards_'

    for yy, year in enumerate(years):

        try:
            c_tmp = pd.read_table(
                    os.path.join(basefol_c, f'{fn1}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=2,
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

    for yy, year in enumerate(years):

        try:
            c_tmp = pd.read_table(
                    os.path.join(basefol_c, f'{fn2}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=2,
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
    c.columns = [inpt.var_in_use]
    # cleaning data
    c[c < 0.] = np.nan

    # ERA5
    fn1 = 'thaao_era5_surface_net_thermal_radiation_'
    fn2 = 'thaao_era5_surface_thermal_radiation_downwards_'
    for yy, year in enumerate(years):
        try:
            e_tmp = pd.read_table(
                    os.path.join(basefol_e, f'{fn1}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
                    engine='python')
            e_tmp[e_tmp == -32767.0] = np.nan
            e_n = pd.concat([e_n, e_tmp], axis=0)
            print(f'OK: {fn1}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn1}{year}.txt')
    e_n.index = pd.to_datetime(e_n[0] + ' ' + e_n[1], format='%Y-%m-%d %H:%M:%S')
    e_n.drop(columns=[0, 1], inplace=True)
    e_n[2] = e_n.values / 3600.  # originele in J*m-2
    e_n.columns = ['surface_net_thermal_radiation']

    for yy, year in enumerate(years):
        try:
            e_tmp = pd.read_table(
                    os.path.join(basefol_e, f'{fn2}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
                    engine='python')
            e_tmp[e_tmp == -32767.0] = np.nan
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
    e.columns = [inpt.var_in_use]
    # cleaning data
    e[e < 0.] = np.nan

    # THAAO
    fn = 'MERGED_SW_LW_UP_DW_METEO_'
    for yy, year in enumerate(years):
        try:
            t_tmp = pd.read_table(
                    os.path.join(basefol_t, 'thaao_rad', f'{fn}{year}_5MIN.dat'), engine='python', skiprows=None,
                    header=0, decimal='.', sep='\s+')
            tmp = np.empty(t_tmp['JDAY_UT'].shape, dtype=dt.datetime)
            for ii, el in enumerate(t_tmp['JDAY_UT']):
                new_jd_ass = el + julian.to_jd(dt.datetime(year - 1, 12, 31, 0, 0), fmt='jd')
                tmp[ii] = julian.from_jd(new_jd_ass, fmt='jd')
                tmp[ii] = tmp[ii].replace(microsecond=0)
            t_tmp.index = pd.DatetimeIndex(tmp)
            t_tmp.drop(
                    ['JDAY_UT', 'JDAY_LOC', 'SZA', 'SW_DOWN', 'PAR_DOWN', 'PAR_UP', 'LW_DOWN', 'SW_UP', 'TBP',
                     'ALBEDO_SW', 'ALBEDO_LW', 'ALBEDO_PAR', 'P', 'T', 'RH', 'PE', 'RR2'], axis=1, inplace=True)
            t = pd.concat([t, t_tmp], axis=0)
            print(f'OK: {fn}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn}{year}.txt')
    t.columns = [inpt.var_in_use]
    # cleaning data
    t[t < 0.] = np.nan

    return


def read_sw_down():
    # CARRA
    fn = 'thaao_carra_surface_solar_radiation_downwards_'
    for yy, year in enumerate(years):
        # fn = extract_values(fn, year)
        try:
            c_tmp = pd.read_table(
                    os.path.join(basefol_c, f'{fn}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=2,
                    engine='python')[[0, 1, 4]]
            c = pd.concat([c, c_tmp], axis=0)
            print(f'OK: {fn}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn}{year}.txt')
    c.index = pd.to_datetime(c[0] + ' ' + c[1], format='%Y-%m-%d %H:%M:%S')
    c.drop(columns=[0, 1], inplace=True)
    c[2] = c.values / 3600.
    c.drop(columns=[4], inplace=True)
    c.columns = [inpt.var_in_use]
    # cleaning data
    c[c < 0.] = np.nan

    # ERA5
    fn = 'thaao_era5_surface_solar_radiation_downwards_'
    for yy, year in enumerate(years):
        try:
            e_tmp = pd.read_table(
                    os.path.join(basefol_e, f'{fn}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
                    engine='python')
            e_tmp[e_tmp == -32767.0] = np.nan
            e = pd.concat([e, e_tmp], axis=0)
            print(f'OK: {fn}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn}{year}.txt')
    e.index = pd.to_datetime(e[0] + ' ' + e[1], format='%Y-%m-%d %H:%M:%S')
    e.drop(columns=[0, 1], inplace=True)
    e[2] = e.values / 3600.  # originale in J*m-2
    e.columns = [inpt.var_in_use]
    # cleaning data
    e[e < 0.] = np.nan

    # THAAO
    fn = 'MERGED_SW_LW_UP_DW_METEO_'
    for yy, year in enumerate(years):
        try:
            t_tmp = pd.read_table(
                    os.path.join(basefol_t, 'thaao_rad', f'{fn}{year}_5MIN.dat'), engine='python', skiprows=None,
                    header=0, decimal='.', sep='\s+')
            file_ok = True
            print(f'OK: {fn}{year}.txt')
        except FileNotFoundError:
            file_ok = False
            print(f'NOT FOUND: {fn}{year}.txt')

        if file_ok:
            tmp = np.empty(t_tmp['JDAY_UT'].shape, dtype=dt.datetime)
            for ii, el in enumerate(t_tmp['JDAY_UT']):
                new_jd_ass = el + julian.to_jd(dt.datetime(year - 1, 12, 31, 0, 0), fmt='jd')
                tmp[ii] = julian.from_jd(new_jd_ass, fmt='jd')
                tmp[ii] = tmp[ii].replace(microsecond=0)
            t_tmp.index = pd.DatetimeIndex(tmp)
            t_tmp.drop(
                    ['JDAY_UT', 'JDAY_LOC', 'SZA', 'SW_UP', 'PAR_DOWN', 'PAR_UP', 'LW_DOWN', 'LW_UP', 'TBP',
                     'ALBEDO_SW', 'ALBEDO_LW', 'ALBEDO_PAR', 'P', 'T', 'RH', 'PE', 'RR2'], axis=1, inplace=True)
            t = pd.concat([t, t_tmp], axis=0)
    t.columns = [inpt.var_in_use]
    # cleaning data
    t[t < 0.] = np.nan

    return


def read_sw_up():
    # CARRA
    fn1 = 'thaao_carra_surface_net_solar_radiation_'
    fn2 = 'thaao_carra_surface_solar_radiation_downwards_'

    for yy, year in enumerate(years):
        # fn = extract_values(fn, year)
        try:
            c_tmp = pd.read_table(
                    os.path.join(basefol_c, f'{fn1}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=2,
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

    for yy, year in enumerate(years):
        # fn = extract_values(fn, year)
        try:
            c_tmp = pd.read_table(
                    os.path.join(basefol_c, f'{fn2}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=2,
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
    c.columns = [inpt.var_in_use]
    # cleaning data
    c[c < 0.] = np.nan

    # ERA5
    fn1 = 'thaao_era5_surface_net_solar_radiation_'
    fn2 = 'thaao_era5_surface_solar_radiation_downwards_'
    for yy, year in enumerate(years):
        try:
            e_tmp = pd.read_table(
                    os.path.join(basefol_e, f'{fn1}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
                    engine='python')
            e_tmp[e_tmp == -32767.0] = np.nan
            e_n = pd.concat([e_n, e_tmp], axis=0)
            print(f'OK: {fn1}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn1}{year}.txt')
    e_n.index = pd.to_datetime(e_n[0] + ' ' + e_n[1], format='%Y-%m-%d %H:%M:%S')
    e_n.drop(columns=[0, 1], inplace=True)
    e_n[2] = e_n.values / 3600.  # originele in J*m-2
    e_n.columns = ['surface_net_solar_radiation']

    for yy, year in enumerate(years):
        try:
            e_tmp = pd.read_table(
                    os.path.join(basefol_e, f'{fn2}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
                    engine='python')
            e_tmp[e_tmp == -32767.0] = np.nan
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
    e.columns = [inpt.var_in_use]
    # cleaning data
    e[e < 0.] = np.nan

    # THAAO
    fn = 'MERGED_SW_LW_UP_DW_METEO_'
    for yy, year in enumerate(years):
        try:
            t_tmp = pd.read_table(
                    os.path.join(basefol_t, 'thaao_rad', f'{fn}{year}_5MIN.dat'), engine='python', skiprows=None,
                    header=0, decimal='.', sep='\s+')
            tmp = np.empty(t_tmp['JDAY_UT'].shape, dtype=dt.datetime)
            for ii, el in enumerate(t_tmp['JDAY_UT']):
                new_jd_ass = el + julian.to_jd(dt.datetime(year - 1, 12, 31, 0, 0), fmt='jd')
                tmp[ii] = julian.from_jd(new_jd_ass, fmt='jd')
                tmp[ii] = tmp[ii].replace(microsecond=0)
            t_tmp.index = pd.DatetimeIndex(tmp)
            t_tmp.drop(
                    ['JDAY_UT', 'JDAY_LOC', 'SZA', 'SW_DOWN', 'PAR_DOWN', 'PAR_UP', 'LW_DOWN', 'LW_UP', 'TBP',
                     'ALBEDO_SW', 'ALBEDO_LW', 'ALBEDO_PAR', 'P', 'T', 'RH', 'PE', 'RR2'], axis=1, inplace=True)
            # 'JDAY_UT', 'JDAY_LOC', 'SZA', 'SW_DOWN', 'SW_UP', 'PAR_DOWN', 'PAR_UP', 'LW_DOWN', 'LW_UP', 'TBP', 'ALBEDO_SW', 'ALBEDO_LW', 'ALBEDO_PAR', 'P', 'T', 'RH', 'PE', 'RR2'
            t = pd.concat([t, t_tmp], axis=0)
            print(f'OK: {fn}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn}{year}.txt')
    t.columns = [inpt.var_in_use]
    # cleaning data
    t[t < 0.] = np.nan

    return


def convert_rs_to_iwv(df, tp):
    """
    Convertito concettualmente in python da codice di Giovanni: PWV_Gio.m
    :param tp: % of the max pressure value up to which calculate the iwv. it is necessary because interpolation fails.
    :param df:
    :return:
    """

    td = dewpoint_from_relative_humidity(
            df['temp'].to_xarray() * units("degC"), df['rh'].to_xarray() / 100)
    iwv = precipitable_water(
            df['pres'].to_xarray() * units("hPa"), td, bottom=None, top=np.nanmin(df['pres']) * tp * units('hPa'))
    # avo_num = 6.022 * 1e23  # [  # /mol]
    # gas_c = 8.314  # [J / (K * mol)]
    # M_water = 18.015e-3  # kg / mol
    # eps = 0.622
    # pressure = df['pres'] * 100  # Pa moltiplico per 100 dato che è in hPa
    # height = df.index.values / 1000  # km divido per 1000 perchè è in metri
    # tempK = df['temp'] + 273.15  # T in K
    # tempC = df['temp']  # T in C
    #
    # RH = df['rh'] / 100  # frazione di umidità relativa
    #
    # ## calcolo la pressione di vapore saturo sull'acqua liquida in Pa (Buck, 1996)
    # a = 18.678 - tempC / 234.5
    # b = 257.14 + tempC
    # c = a * tempC / b
    # Ps = 611.21 * np.exp(c)  # pressione di vapore saturo [Pa]
    #
    # ## pressione parziale di vapore acqueo
    # Ph2o = RH * Ps
    #
    # ## vmr vapore acqueo
    # # vmr = Ph2o / pressure
    # vmr = mixing_ratio_from_relative_humidity(
    #         df['pres'].to_xarray() * units("Pa"), df['temp'].to_xarray() * units("degC"),
    #         df['rh'].to_xarray() * units("percent"))
    #
    # ## mixmass
    # mix_mass = vmr * eps
    #
    # ## Concentrazione di vapore acqueo
    # conc_h2o = (Ph2o * avo_num / (gas_c * tempK))  ##/m^3
    #
    # ## calcolo il numero totale di molecole
    # conc_tot = np.sum(conc_h2o * 100)
    #
    # ## calcolo Tatm
    # # T_sup = T_int(1)
    # # Tatm = sum(T_int * conc_h2o_int) * 100 / conc_tot + T_int(1) * conc_h2o_int(1) * (bot - 50 - zGrd) / conc_tot;
    #
    # ## calcolo PWV
    # mol_tot = conc_tot / avo_num  # numero totale di moli
    # m_tot = mol_tot * M_water  # massa totale in kg
    # # carico la funzione di densità dell'acqua liquida
    # rho_lw_T = load('rho_liquid_water vs T')  # in kg al m^3
    # rho_lw = rho_lw_T.rho_lw
    # T_lw = rho_lw_T.T_lw
    # # [mini, ind] = min(abs(T_sup - T_lw))  # trovo a quale T ci troviamo
    #
    # PWV = m_tot / rho_lw(ind) * 1000  # [mm] area unitaria e moltiplico per 1000 per averlo im mm

    return iwv


def extract_values(fn, year):
    if not os.path.exists(os.path.join(basefol_c, fn + str(year) + '.nc')):
        try:
            filen = os.path.join(basefol_t, 'reanalysis', 'carra', '_'.join(fn.split('_')[1:]) + str(year) + '.nc')
            NC = xr.open_dataset(str(filen), decode_cf=True, decode_times=True)

            # tmp = NC.sel(x=y, y=x, method='nearest')
        except FileNotFoundError:
            print(f'cannot find {filen}')

    return f'thaao_{fn}'


def read():
    """

    :return:
    """
    if inpt.var_in_use == 'temp':
        return read_temp()
    if inpt.var_in_use == 'rh':
        return read_rh()
    if inpt.var_in_use == 'surf_pres':
        return read_surf_pres()
    if inpt.var_in_use == 'msl_pres':
        return read_msl_pres()
    if inpt.var_in_use == 'iwv':
        return read_iwv()
    if inpt.var_in_use == 'lwp':
        return read_lwp()
    if inpt.var_in_use == 'winds':
        return read_winds()
    if inpt.var_in_use == 'windd':
        return read_windd()
    if inpt.var_in_use == 'alb':
        return read_alb()
    if inpt.var_in_use == 'precip':
        return read_precip()
    if inpt.var_in_use == 'cbh':
        return read_cbh()
    if inpt.var_in_use == 'tcc':
        return read_tcc()
    if inpt.var_in_use == 'lw_down':
        return read_lw_down()
    if inpt.var_in_use == 'lw_up':
        return read_lw_up()
    if inpt.var_in_use == 'sw_down':
        return read_sw_down()
    if inpt.var_in_use == 'sw_up':
        return read_sw_up()
