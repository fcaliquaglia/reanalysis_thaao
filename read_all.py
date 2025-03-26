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

from metpy.calc import wind_direction, wind_speed
from metpy.units import units

from read_reanalysis import *


def read_all():
    """
    Read the data based on the value of inpt.var.
    """

    # Create a mapping of variable names to functions
    read_functions = {'alb'    : read_all_alb, 'cbh': read_all_cbh, 'msl_pres': read_all_msl_pres, 'lwp': read_all_lwp,
                      'lw_down': read_all_lw_down, 'lw_up': read_all_lw_up, 'precip': read_all_precip,
                      'rh'     : read_all_rh, 'surf_pres': read_all_surf_pres, 'sw_down': read_all_sw_down,
                      'sw_up'  : read_all_sw_up, 'tcc': read_all_tcc, 'temp': read_all_temp, 'winds': read_all_wind,
                      'windd'  : read_all_wind, }

    # Use the mapping to call the correct function
    var = inpt.var
    if var in read_functions:
        return read_functions[var]()
    else:
        raise ValueError(f"Unknown variable: {var}")


def read_all_alb():
    # CARRA
    read_reanalysis(inpt.var, 'c')
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data'] / inpt.extr[inpt.var]['rescale_fact']
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] < inpt.extr[inpt.var]['excl_thresh']] = np.nan

    # ERA5
    read_reanalysis(inpt.var, 'e')
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] < inpt.extr[inpt.var]['excl_thresh']] = np.nan

    # ERA5-LAND
    read_reanalysis(inpt.var, 'l')
    inpt.extr[inpt.var]['l']['data'][inpt.extr[inpt.var]['l']['data'] < inpt.extr[inpt.var]['excl_thresh']] = np.nan

    # THAAO
    read_thaao_rad(inpt.var)

    return


def read_all_cbh():
    # CARRA
    read_reanalysis(inpt.var, 'c')
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] <= inpt.extr[inpt.var]['excl_thresh']] = np.nan

    # ERA5
    read_reanalysis(inpt.var, 'e')
    inpt.extr[inpt.var]['e']['data'][inpt.extr[inpt.var]['e']['data'] <= inpt.extr[inpt.var]['excl_thresh']] = np.nan
    inpt.extr[inpt.var]['e']['data'] += inpt.thaao_elev

    # THAAO
    read_thaao_ceilometer(inpt.var)
    inpt.extr[inpt.var]['t']['data'][inpt.extr[inpt.var]['t']['data'] <= inpt.extr[inpt.var]['excl_thresh']] = np.nan

    return


def read_all_lwp():
    # CARRA
    read_reanalysis(inpt.var, 'c')
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data']
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] < inpt.extr[inpt.var]['excl_thresh']] = np.nan

    # ERA5
    read_reanalysis(inpt.var, 'e')
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data']
    inpt.extr[inpt.var]['e']['data'][inpt.extr[inpt.var]['e']['data'] < inpt.extr[inpt.var]['excl_thresh']] = np.nan

    # THAAO1
    read_thaao_hatpro(inpt.var)
    inpt.extr[inpt.var]['t1']['data'][inpt.extr[inpt.var]['t1']['data'] < inpt.extr[inpt.var]['excl_thresh']] = np.nan

    return


def read_all_msl_pres():
    # CARRA
    read_reanalysis(inpt.var, 'c')

    return


def read_all_precip():
    # CARRA
    read_reanalysis(inpt.var, 'c')

    # ERA5
    read_reanalysis(inpt.var, 'e')
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'].values * 1000.

    # THAAO2
    read_thaao_aws_ecapac(inpt.var)

    return


def read_all_lw_down():
    vr = 'lw_down'
    # CARRA
    read_reanalysis(vr, 'c')
    inpt.extr[vr]['c']['data'][inpt.extr[vr]['c']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan
    inpt.extr[vr]['c']['data'] = inpt.extr[vr]['c']['data'] / inpt.extr[vr]['rescale_factor']

    # ERA5
    read_reanalysis(vr, 'e')
    inpt.extr[vr]['e']['data'][inpt.extr[vr]['e']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan
    inpt.extr[vr]['e']['data'] = inpt.extr[vr]['e']['data'] / inpt.extr[vr]['rescale_factor']

    # ERA5-LAND
    read_reanalysis(vr, 'l')
    inpt.extr[vr]['l']['data'][inpt.extr[vr]['l']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan
    inpt.extr[vr]['l']['data'] = inpt.extr[vr]['l']['data'] / inpt.extr[vr]['rescale_factor']

    # THAAO
    read_thaao_rad(vr)
    inpt.extr[vr]['t']['data'][inpt.extr[vr]['t']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan
    return


def read_all_sw_down():
    vr = 'sw_down'
    # CARRA
    read_reanalysis(vr, 'c')
    inpt.extr[vr]['c']['data'][inpt.extr[vr]['c']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan
    inpt.extr[vr]['c']['data'] = inpt.extr[vr]['c']['data'] / inpt.extr[vr]['rescale_factor']

    # ERA5
    read_reanalysis(vr, 'e')
    inpt.extr[vr]['e']['data'][inpt.extr[vr]['e']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan
    inpt.extr[vr]['e']['data'] = inpt.extr[vr]['e']['data'] / inpt.extr[vr]['rescale_factor']

    # ERA5-LAND
    read_reanalysis(vr, 'l')
    inpt.extr[vr]['l']['data'][inpt.extr[vr]['l']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan
    inpt.extr[vr]['l']['data'] = inpt.extr[vr]['l']['data'] / inpt.extr[vr]['rescale_factor']

    # THAAO
    read_thaao_rad(vr)
    inpt.extr[vr]['t']['data'][inpt.extr[vr]['t']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan
    return


def read_all_lw_up():
    vr = 'lw_up'
    read_all_lw_down()

    # CARRA
    read_reanalysis('lw_net', 'c')
    inpt.extr['lw_net']['c']['data'] = inpt.extr['lw_net']['c']['data'] / inpt.extr[vr]['rescale_factor']
    inpt.extr[vr]['c']['data'] = pd.DataFrame(
            index=inpt.extr['lw_down']['c']['data'].index,
            data=inpt.extr['lw_down']['c']['data'].values - inpt.extr['lw_net']['c']['data'].values, columns=[vr])
    inpt.extr[vr]['c']['data'][inpt.extr[vr]['c']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan
    # del inpt.extr['lw_net']['c']['data']

    # ERA5
    read_reanalysis('lw_net', 'e')
    inpt.extr['lw_net']['e']['data'] = inpt.extr['lw_net']['e']['data'] / inpt.extr[vr]['rescale_factor']
    inpt.extr[vr]['e']['data'] = pd.DataFrame(
            index=inpt.extr['lw_down']['e']['data'].index,
            data=inpt.extr['lw_down']['e']['data'].values - inpt.extr['lw_net']['e']['data'].values, columns=[vr])
    inpt.extr[vr]['e']['data'][inpt.extr[vr]['e']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan
    # del inpt.extr['lw_net']['e']['data']

    # ERA5-LAND
    read_reanalysis('lw_net', 'l')
    inpt.extr['lw_net']['l']['data'] = inpt.extr['lw_net']['l']['data'] / inpt.extr[vr]['rescale_factor']
    inpt.extr[vr]['l']['data'] = pd.DataFrame(
            index=inpt.extr['lw_down']['l']['data'].index,
            data=inpt.extr['lw_down']['l']['data'].values - inpt.extr['lw_net']['l']['data'].values, columns=[vr])
    inpt.extr[vr]['l']['data'][inpt.extr[vr]['l']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan
    # del inpt.extr['lw_net']['l']['data']

    # THAAO
    read_thaao_rad(vr)
    inpt.extr[vr]['t']['data'][inpt.extr[vr]['t']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan

    return


def read_all_sw_up():
    vr = 'sw_up'
    read_all_sw_down()

    # CARRA
    read_reanalysis('sw_net', 'c')
    inpt.extr['sw_net']['c']['data'] = inpt.extr['sw_net']['c']['data'] / inpt.extr[vr]['rescale_factor']
    inpt.extr[vr]['c']['data'] = pd.DataFrame(
            index=inpt.extr['sw_down']['c']['data'].index,
            data=inpt.extr['sw_down']['c']['data'].values - inpt.extr['sw_net']['c']['data'].values, columns=[vr])
    inpt.extr[vr]['c']['data'][inpt.extr[vr]['c']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan
    del inpt.extr['sw_net']['c']['data']

    # ERA5
    read_reanalysis('sw_net', 'e')
    inpt.extr['sw_net']['e']['data'] = inpt.extr['sw_net']['e']['data'] / inpt.extr[vr]['rescale_factor']
    inpt.extr[vr]['e']['data'] = pd.DataFrame(
            index=inpt.extr['sw_down']['e']['data'].index,
            data=inpt.extr['sw_down']['e']['data'].values - inpt.extr['sw_net']['e']['data'].values, columns=[vr])
    inpt.extr[vr]['e']['data'][inpt.extr[vr]['e']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan
    del inpt.extr['sw_net']['e']['data']

    # ERA5-LAND
    read_reanalysis('sw_net', 'l')
    inpt.extr['sw_net']['l']['data'] = inpt.extr['sw_net']['l']['data'] / inpt.extr[vr]['rescale_factor']
    inpt.extr[vr]['l']['data'] = pd.DataFrame(
            index=inpt.extr['sw_down']['l']['data'].index,
            data=inpt.extr['sw_down']['l']['data'].values - inpt.extr['sw_net']['l']['data'].values, columns=[vr])
    inpt.extr[vr]['l']['data'][inpt.extr[vr]['l']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan
    del inpt.extr['sw_net']['l']['data']

    # THAAO
    read_thaao_rad(vr)
    inpt.extr[vr]['t']['data'][inpt.extr[vr]['t']['data'] < inpt.extr[vr]['excl_thresh']] = np.nan

    return


def read_all_rh():
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
    read_thaao_aws_ecapac(inpt.var)
    read_thaao_weather(inpt.var)

    return


def read_all_surf_pres():
    # CARRA
    read_reanalysis(inpt.var, 'c')
    inpt.extr[inpt.var]['c']['data'] = inpt.extr[inpt.var]['c']['data'] / inpt.extr[inpt.var]['rescale_fact']
    inpt.extr[inpt.var]['c']['data'][inpt.extr[inpt.var]['c']['data'] < inpt.extr[inpt.var]['excl_thresh']] = np.nan

    # ERA5
    read_reanalysis(inpt.var, 'e')
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'] / inpt.extr[inpt.var]['rescale_fact']
    inpt.extr[inpt.var]['e']['data'][inpt.extr[inpt.var]['e']['data'] < inpt.extr[inpt.var]['excl_thresh']] = np.nan

    # THAAO
    read_thaao_weather(inpt.var)
    inpt.extr[inpt.var]['t']['data'][inpt.extr[inpt.var]['t']['data'] < inpt.extr[inpt.var]['excl_thresh']] = np.nan
    inpt.extr[inpt.var]['t']['data'].loc['2021-10-11 00:00:00':'2021-10-19 00:00:00'] = np.nan
    inpt.extr[inpt.var]['t']['data'].loc['2024-4-26 00:00:00':'2024-5-4 00:00:00'] = np.nan

    # THAAO2
    read_thaao_aws_ecapac(inpt.var)

    return


def read_all_tcc():
    # CARRA
    read_reanalysis(inpt.var, 'c')

    # ERA5
    read_reanalysis(inpt.var, 'e')
    inpt.extr[inpt.var]['e']['data'] = inpt.extr[inpt.var]['e']['data'].values * 100.

    # THAAO
    read_thaao_ceilometer(inpt.var)

    return


def read_all_temp():
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
    read_thaao_aws_ecapac(inpt.var)
    return


def read_all_wind():
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
    read_thaao_aws_ecapac('winds')
    read_thaao_aws_ecapac('windd')

    return


def calc_rh_from_tdp():
    # TODO not working

    # e = pd.concat([inpt.extr[inpt.var]['t']['data'], e_t], axis=1)

    # e['rh'] = relative_humidity_from_dewpoint(e['e_t'].values * units.K, e['e_td'].values * units.K).to('percent')
    inpt.extr[inpt.var]['e']['data'].drop(columns=['e_t', 'e_td'], inplace=True)
    inpt.extr[inpt.var]['e']['data'].columns = [inpt.var]

    return
