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

import xarray as xr
from metpy.calc import dewpoint_from_relative_humidity, precipitable_water
from metpy.units import units

from inputs import *


def read_iwv():
    c = pd.DataFrame()
    e = pd.DataFrame()
    l = pd.DataFrame()
    t = pd.DataFrame()
    t1 = pd.DataFrame()
    t2 = pd.DataFrame()
    vr = 'iwv'

    # CARRA
    fn = 'thaao_carra_total_column_integrated_water_vapour_'
    for yy, year in enumerate(years):
        try:
            # ciccio = extract_values(fn, year)
            c_tmp = pd.read_table(
                    os.path.join(basefol_c, f'{fn}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
                    engine='python')
            # workaround for new files extracted which are different
            if len(c_tmp.columns) > 3:
                c1_tmp = pd.read_table(
                        os.path.join(basefol_c, f'{fn}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=2,
                        engine='python', usecols=[0, 1, 4])
                c1_tmp.columns = [0, 1, 2]
            else:
                c1_tmp = c_tmp
            c = pd.concat([c, c1_tmp], axis=0)
            print(f'OK: {fn}{year}.txt')
        except FileNotFoundError:
            print(f'NOT FOUND: {fn}{year}.txt')
    c.index = pd.to_datetime(c[0] + ' ' + c[1], format='%Y-%m-%d %H:%M:%S')
    c.drop(columns=[0, 1], inplace=True)
    c[c <= 0] = np.nan
    c.columns = [vr]

    # # ERA5
    # fn = 'thaao_era5_total_column_water_vapour_'
    # for yy, year in enumerate(years):
    #     try:
    #         e_tmp = pd.read_table(
    #                 os.path.join(basefol_e, f'{fn}{year}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
    #                 engine='python')
    #         e_tmp[e_tmp == -32767.0] = np.nan
    #         e = pd.concat([e, e_tmp], axis=0)
    #         print(f'OK: {fn}{year}.txt')
    #     except FileNotFoundError:
    #         print(f'NOT FOUND: {fn}{year}.txt')
    # e.index = pd.to_datetime(e[0] + ' ' + e[1], format='%Y-%m-%d %H:%M:%S')
    # e.drop(columns=[0, 1], inplace=True)
    # e.columns = [vr]

    # # THAAO (vespa)
    # fn = 'vespaPWVClearSky'
    # try:
    #     t = pd.read_table(
    #             os.path.join(basefol_t, 'thaao_vespa', f'{fn}.txt'), skipfooter=1, sep='\s+', header=None, skiprows=1,
    #             engine='python')
    #     print(f'OK: {fn}.txt')
    # except FileNotFoundError:
    #     print(f'NOT FOUND: {fn}{year}.txt')
    # t.index = pd.to_datetime(t[0] + ' ' + t[1], format='%Y-%m-%d %H:%M:%S')
    # t.drop(columns=[0, 1, 3, 4, 5], inplace=True)
    # t.columns = [vr]

    # # THAAO (hatpro)
    # fn = 'QC_IWV_15_min_'
    # for yy, year in enumerate(years):
    #     try:
    #         t1_tmp = pd.read_table(
    #                 os.path.join(
    #                         basefol_t, 'thaao_hatpro', 'definitivi_da_giando', f'{fn}{year}', f'{fn}{year}.DAT'),
    #                 sep='\s+', engine='python', header=None, skiprows=1)
    #         t1_tmp.columns = ['JD_rif', 'IWV', 'STD_IWV', 'RF', 'N']
    #         tmp = np.empty(t1_tmp['JD_rif'].shape, dtype=dt.datetime)
    #         for ii, el in enumerate(t1_tmp['JD_rif']):
    #             new_jd_ass = el + julian.to_jd(dt.datetime(year - 1, 12, 31, 0, 0), fmt='jd')
    #             tmp[ii] = julian.from_jd(new_jd_ass, fmt='jd')
    #             tmp[ii] = tmp[ii].replace(microsecond=0)
    #         t1_tmp.index = pd.DatetimeIndex(tmp)
    #         t1_tmp.drop(columns=['JD_rif', 'STD_IWV', 'RF', 'N'], axis=1, inplace=True)
    #         t1 = pd.concat([t1, t1_tmp], axis=0)
    #         print(f'OK: {fn}{year}.DAT')
    #     except FileNotFoundError:
    #         print(f'NOT FOUND: {fn}{year}.DAT')
    # t1['IWV'] = t1['IWV'].values
    # t1.columns = [vr]
    # # cleaning HATPRO DATA
    # t1[t1 < 0] = np.nan
    # t1[t1 > 30] = np.nan

    # RS (sondes)
    for yy, year in enumerate(years):
        try:
            fol_input = os.path.join(basefol_t, 'thaao_rs_sondes', 'txt', f'{year}')
            file_l = os.listdir(fol_input)
            file_l.sort()
            for i in file_l:
                print(i)
                try:
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
                except ValueError:
                    print('issue with ' + i)
            print(f'OK: year {year}')
        except FileNotFoundError:
            print(f'NOT FOUND: year {year}')
    t2.columns = [vr]
    # np.savetxt(os.path.join(basefol_t, 'rs_pwv.txt'), t2, fmt='%s')
    t2.to_csv(os.path.join(basefol_t, 'rs_pwv.txt'), index=True)

    return [c, e, l, t, t1, t2]


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

    return iwv


# def extract_values(fn, year):
#     if not os.path.exists(os.path.join(basefol_c, fn + str(year) + '.nc')):
#         try:
#             filen = os.path.join(basefol_r, 'carra', '_'.join(fn.split('_')[1:]) + str(year) + '.nc')
#             NC = xr.open_dataset(str(filen), decode_cf=True, decode_times=True)
#
#             # import cfgrib
#             # ds = cfgrib.open_dataset('era5-levels-members.grib')
#             # ds
#
#             # tmp = NC.sel(x=y, y=x, method='nearest')
#         except FileNotFoundError:
#             print(f'cannot find {filen}')
#
#     return f'thaao_{fn}'


def read(var):
    """

    :param var:
    :return:
    """
    if var == 'iwv':
        return read_iwv()
