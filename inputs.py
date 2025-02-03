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
import string

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

## FOLDERS
basefol_c = os.path.join('H:\\Shared drives', 'Reanalysis', 'carra', 'thaao', 'v1')
basefol_e = os.path.join('H:\\Shared drives', 'Reanalysis', 'era5', 'thaao', 'v1')
basefol_l = os.path.join('H:\\Shared drives', 'Reanalysis', 'era5-land', 'thaao', 'v1')
basefol_t = os.path.join('H:\\Shared drives', 'Dati_THAAO')
basefol_out = os.path.join('H:\\Shared drives', 'Dati_elab_docs', 'thaao_reanalysis')

thaao_c = 'thaao_carra'
thaao_e = 'thaao_era5'
thaao_l = 'thaao_era5-land'
thaao_t = 'thaao'

##
tres_list = ['3h']
list_var = ['cbh']
# OK ['lwp', 'surf_pres', 'winds', 'windd', 'cbh', 'alb', 'temp', 'sw_down', 'lw_down', 'sw_up', 'lw_up']
# NOT OK 'tcc' 'precip' 'rh', 'lwp'

tres = ''
var = ''

years = np.arange(2016, 2024, 1)

aws_ecapac_daterange = pd.date_range(start=dt.datetime(2023, 4, 1), end=dt.datetime(2024, 12, 31), freq='1D')
ceilometer_daterange = pd.date_range(start=dt.datetime(2019, 9, 1), end=dt.datetime(2024, 12, 31), freq='1D')
rad_daterange = pd.date_range(start=dt.datetime(2009, 9, 1), end=dt.datetime(2024, 12, 31), freq='YE')
hatpro_daterange = pd.date_range(start=dt.datetime(2016, 9, 1), end=dt.datetime(2024, 10, 30), freq='YE')

SMALL_SIZE = 12

myFmt = mdates.DateFormatter('%d-%b')
letters = list(string.ascii_lowercase)

seass = {'all': {'name': 'all', 'months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'col': 'pink'},
         'DJF': {'name': 'DJF', 'months': [12, 1, 2], 'col': 'blue'},
         'MAM': {'name': 'MAM', 'months': [3, 4, 5], 'col': 'green'},
         'JJA': {'name': 'JJA', 'months': [6, 7, 8], 'col': 'orange'},
         'SON': {'name': 'SON', 'months': [9, 10, 11], 'col': 'brown'},
         # 'MA' : {'name': 'MA', 'months': [3, 4], 'col': 'yellow'},
         # 'MJ' : {'name': 'MJ', 'months': [5, 6], 'col': 'cyan'},
         # 'JA': {'name': 'JA', 'months': [7, 8], 'col': 'grey'},
         # 'SO' : {'name': 'SO', 'months': [9, 10], 'col': 'purple'}
         }

var_dict = {'c' : {'name'     : 'c', 'nanval': '', 'col': 'red', 'col_ori': 'orange', 'label': 'CARRA',
                   'label_uom': '', 'rad_conv_factor': 3600},
            'e' : {'name'     : 'e', 'nanval': -32767.0, 'col': 'blue', 'col_ori': 'cyan', 'label': 'ERA5',
                   'label_uom': '', 'rad_conv_factor': 3600},
            'l' : {'name'     : 'l', 'nanval': -32767.0, 'col': 'darkgreen', 'col_ori': 'lightgreen', 'label': 'ERA5-L',
                   'label_uom': '', 'rad_conv_factor': 3600},
            't' : {'name'     : 't', 'nanval': -9999.9, 'col': 'black', 'col_ori': 'grey', 'label': 'THAAO',
                   'label_uom': ''},
            't1': {'name'     : 't1', 'nanval': '', 'col': 'green', 'col_ori': 'lightgreen', 'label': 'HATPRO',
                   'label_uom': ''},
            't2': {'name'     : 't2', 'nanval': '', 'col': 'purple', 'col_ori': 'violet', 'label': 'AWS_ECAPAC',
                   'label_uom': ''}}

extr = {'alb'      : {'name': 'alb', 'ref_x': 't', 'min': 0, 'max': 1, 'res_min': -0.5, 'res_max': 0.5,
                      'uom' : '[none]', 'comps': ['c', 'e', 'l'], 'bin_nr': 200,
                      'c'   : {'fn'      : f'{thaao_c}_albedo_', 'column': 2, 'data': '',
                               'data_res': ''},
                      'e'   : {'fn'      : f'{thaao_e}_forecast_albedo_', 'column': 2, 'data': '',
                               'data_res': ''},
                      'l'   : {'fn'      : f'{thaao_l}_forecast_albedo_', 'column': 4, 'data': '',
                               'data_res': ''},
                      't'   : {'fn'      : 'MERGED_SW_LW_UP_DW_METEO_', 'column': 'ALBEDO_SW', 'data': '',
                               'data_res': ''},
                      't1'  : {'fn': '', 'column': 2, 'data': '', 'data_res': ''},
                      't2'  : {'fn': '', 'column': 2, 'data': '', 'data_res': ''}},
        'cbh'      : {'name': 'cbh', 'ref_x': 't', 'min': 0, 'max': 10000, 'res_min': -1500, 'res_max': 1500,
                      'uom' : '[m]', 'comps': ['c', 'e'], 'bin_nr': 200,
                      'c'   : {'fn'      : f'{thaao_c}_cloud_base_', 'column': 2, 'data': '',
                               'data_res': ''},
                      'e'   : {'fn'      : f'{thaao_e}_cloud_base_height_', 'column': 2, 'data': '',
                               'data_res': ''},
                      'l'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't'   : {'fn'      : '_Thule_CHM190147_000_0060cloud', 'column': 'CBH_L1[m]',
                               'data'    : pd.DataFrame(),
                               'data_res': ''},
                      't1'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't2'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''}},
        'dewpt'    : {'name' : 'dewpt', 'ref_x': '', 'min': '', 'max': '', 'res_min': '', 'res_max': '', 'uom': '[deg]',
                      'comps': [], 'bin_nr': np.nan,
                      'c'    : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      'e'    : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      'l'    : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't'    : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't1'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't2'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''}},
        'lwp'      : {'name': 'lwp', 'ref_x': 't1', 'min': 0, 'max': 50, 'res_min': -20, 'res_max': 20,
                      'uom' : '[kg/m2]', 'comps': ['c', 'e'], 'bin_nr': 200,
                      'c'   : {'fn'  : f'{thaao_c}_total_column_cloud_liquid_water_', 'column': 2,
                               'data': '', 'data_res': ''},
                      'e'   : {'fn'  : f'{thaao_e}_total_column_cloud_liquid_water_', 'column': 2,
                               'data': '', 'data_res': ''},
                      'l'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't1'  : {'fn'      : 'LWP_15_min_', 'column': np.nan, 'data': '',
                               'data_res': ''},
                      't2'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''}},
        'lw_down'  : {'name': 'lw_down', 'ref_x': 't', 'min': 100, 'max': 400, 'res_min': -20, 'res_max': 20,
                      'uom' : '[W/m2]', 'comps': ['c', 'e', 'l'], 'bin_nr': 200,
                      'c'   : {'fn'  : f'{thaao_c}_thermal_surface_radiation_downwards_', 'column': 4,
                               'data': '', 'data_res': ''},
                      'e'   : {'fn'  : f'{thaao_e}_surface_thermal_radiation_downwards_', 'column': 2,
                               'data': '', 'data_res': ''},
                      'l'   : {'fn'  : f'{thaao_l}_surface_thermal_radiation_downwards_', 'column': 4,
                               'data': '', 'data_res': ''},
                      't'   : {'fn'      : 'MERGED_SW_LW_UP_DW_METEO_', 'column': 'LW_DOWN', 'data': '',
                               'data_res': ''},
                      't1'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't2'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''}},
        'lw_net'   : {'name': 'lw_down', 'ref_x': 't', 'min': 0, 'max': 600, 'res_min': -20, 'res_max': 20,
                      'uom' : '[W/m2]', 'comps': ['c', 'e', 'l'], 'bin_nr': 200,
                      'c'   : {'fn'  : f'{thaao_c}_surface_net_thermal_radiation_', 'column': 4,
                               'data': '', 'data_res': ''},
                      'e'   : {'fn'  : f'{thaao_e}_surface_net_thermal_radiation_', 'column': 2,
                               'data': '', 'data_res': ''},
                      'l'   : {'fn'  : f'{thaao_l}_surface_net_thermal_radiation_', 'column': 4,
                               'data': '', 'data_res': ''},
                      't'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't1'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't2'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''}},
        'lw_up'    : {'name': 'lw_up', 'ref_x': 't', 'min': 100, 'max': 500, 'res_min': -20, 'res_max': 20,
                      'uom' : '[W/m2]', 'comps': ['c', 'e', 'l'], 'bin_nr': 200,
                      'c'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      'e'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      'l'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't'   : {'fn'      : 'MERGED_SW_LW_UP_DW_METEO_', 'column': 'LW_UP', 'data': '',
                               'data_res': ''},
                      't1'  : {'fn': '', 'column': 2, 'data': '', 'data_res': ''},
                      't2'  : {'fn': '', 'column': 2, 'data': '', 'data_res': ''}},
        'msl'      : {'name': 'winds', 'ref_x': 't', 'min': 0, 'max': 30, 'res_min': -10, 'res_max': 10,
                      'uom' : '[m/s]', 'comps': ['c', 'e', 't1', 't2'], 'bin_nr': 200,
                      'c'   : {'fn'      : f'{thaao_c}_mean_sea_level_pressure_', 'column': 2, 'data': '',
                               'data_res': ''},
                      'e'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      'l'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't1'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't2'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''}},
        'precip'   : {'name': 'precip', 'ref_x': 't2', 'min': 0, 'max': 5, 'res_min': -10, 'res_max': 10,
                      'uom' : '[mm?]', 'comps': ['c', 'e'], 'bin_nr': 200,
                      'c'   : {'fn'      : f'{thaao_c}_total_precipitation_', 'column': 2, 'data': '',
                               'data_res': ''},
                      'e'   : {'fn'      : f'{thaao_e}_total_precipitation_', 'column': 2, 'data': '',
                               'data_res': ''},
                      'l'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't1'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't2'  : {'fn'      : 'AWS_THAAO_', 'column': 'RH', 'data': '',
                               'data_res': ''}},
        'rh'       : {'name' : 'rh', 'ref_x': 't', 'min': 0, 'max': 100, 'res_min': -10, 'res_max': 10, 'uom': '[%]',
                      'comps': ['c', 'e', 'l', 't2'], 'bin_nr': 200,
                      'c'    : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      'e'    : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      'l'    : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't'    : {'fn'      : 'Meteo_weekly_all', 'column': 'RH_%', 'data': '',
                                'data_res': ''},
                      't1'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't2'   : {'fn'      : 'AWS_THAAO_', 'column': 'RH', 'data': '',
                                'data_res': ''}},
        'surf_pres': {'name': 'surf_pres', 'ref_x': 't', 'min': 935, 'max': 1013, 'res_min': -10, 'res_max': 10,
                      'uom' : '[hPa]', 'comps': ['c', 'e', 't2'], 'bin_nr': 200,
                      'c'   : {'fn'      : f'{thaao_c}_surface_pressure_', 'column': 2, 'data': '',
                               'data_res': ''},
                      'e'   : {'fn'      : f'{thaao_e}_surface_pressure_', 'column': 2, 'data': '',
                               'data_res': ''},
                      'l'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't'   : {'fn'      : 'Meteo_weekly_all', 'column': 'BP_hPa', 'data': '',
                               'data_res': ''},
                      't1'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't2'  : {'fn'      : 'AWS_THAAO_', 'column': 'BP_mbar', 'data': '',
                               'data_res': ''}},
        'sw_down'  : {'name': 'sw_down', 'ref_x': 't', 'min': 0, 'max': 700, 'res_min': -20, 'res_max': 20,
                      'uom' : '[W/m2]', 'comps': ['c', 'e', 'l'], 'bin_nr': 200,
                      'c'   : {'fn'  : f'{thaao_c}_surface_solar_radiation_downwards_', 'column': 4,
                               'data': '', 'data_res': ''},
                      'e'   : {'fn'  : f'{thaao_e}_surface_solar_radiation_downwards_', 'column': 2,
                               'data': '', 'data_res': ''},
                      'l'   : {'fn'  : f'{thaao_l}_surface_solar_radiation_downwards_', 'column': 4,
                               'data': '', 'data_res': ''},
                      't'   : {'fn'      : 'MERGED_SW_LW_UP_DW_METEO_', 'column': 'SW_DOWN', 'data': '',
                               'data_res': ''},
                      't1'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't2'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''}},
        'sw_net'   : {'name': 'sw_down', 'ref_x': 't', 'min': 0, 'max': 600, 'res_min': -20, 'res_max': 20,
                      'uom' : '[W/m2]', 'comps': ['c', 'e', 'l'], 'bin_nr': 200,
                      'c'   : {'fn'  : f'{thaao_c}_surface_net_solar_radiation_', 'column': 4,
                               'data': '', 'data_res': ''},
                      'e'   : {'fn'  : f'{thaao_e}_surface_net_solar_radiation_', 'column': 2,
                               'data': '', 'data_res': ''},
                      'l'   : {'fn'  : f'{thaao_l}_surface_net_solar_radiation_', 'column': 4,
                               'data': '', 'data_res': ''},
                      't'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't1'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't2'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''}},
        'sw_up'    : {'name': 'sw_up', 'ref_x': 't', 'min': 0, 'max': 600, 'res_min': -20, 'res_max': 20,
                      'uom' : '[W/m2]', 'comps': ['c', 'e', 'l'], 'bin_nr': 200,
                      'c'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      'e'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      'l'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't'   : {'fn'      : 'MERGED_SW_LW_UP_DW_METEO_', 'column': 'SW_UP', 'data': '',
                               'data_res': ''},
                      't1'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't2'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''}},
        'temp'     : {'name'   : 'temp', 'ref_x': 't', 'min': -40, 'max': 20, 'res_min': -15,
                      'res_max': 15, 'uom': '[deg]', 'comps': ['c', 'e', 'l', 't2'], 'bin_nr': 200,
                      'c'      : {'fn'      : f'{thaao_c}_2m_temperature_', 'column': 2, 'data': '',
                                  'data_res': ''},
                      'e'      : {'fn'      : f'{thaao_e}_2m_temperature_', 'column': 2, 'data': '',
                                  'data_res': ''},
                      'l'      : {'fn'      : f'{thaao_l}_2m_temperature_', 'column': 2, 'data': '',
                                  'data_res': ''},
                      't'      : {'fn'      : 'Meteo_weekly_all', 'column': 'Air_K', 'data': '',
                                  'data_res': ''},
                      't1'     : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't2'     : {'fn'      : 'AWS_THAAO_', 'column': 'AirTC', 'data': '',
                                  'data_res': ''}},
        'tcc'      : {'name': 'tcc', 'ref_x': 't', 'min': 0, 'max': 1, 'res_min': -50, 'res_max': 50,
                      'uom' : '[octave]', 'comps': ['c', 'e'], 'bin_nr': 200,
                      'c'   : {'fn'      : f'{thaao_c}_total_cloud_cover_', 'column': 2, 'data': '',
                               'data_res': ''},
                      'e'   : {'fn'      : f'{thaao_e}_total_cloud_cover_', 'column': 2, 'data': '',
                               'data_res': ''},
                      'l'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't'   : {'fn'      : '_Thule_CHM190147_000_0060cloud', 'column': 'TCC[okt]',
                               'data'    : pd.DataFrame(),
                               'data_res': ''},
                      't1'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't2'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''}},
        'windd'    : {'name': 'windd', 'ref_x': 't', 'min': 0, 'max': 360, 'res_min': -90, 'res_max': 90,
                      'uom' : '[deg]', 'comps': ['c', 'e', 'l', 't2'], 'bin_nr': 200,
                      'c'   : {'fn'      : f'{thaao_c}_10m_wind_direction_', 'column': 2, 'data': '',
                               'data_res': ''},
                      'e'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      'l'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't'   : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't1'  : {'fn': '', 'column': np.nan, 'data': '', 'data_res': ''},
                      't2'  : {'fn'      : 'AWS_THAAO_', 'column': 'WD_aws', 'data': '',
                               'data_res': ''}},
        'winds'    : {'name': 'winds', 'ref_x': 't', 'min': 0, 'max': 30, 'res_min': -10, 'res_max': 10,
                      'uom' : '[m/s]', 'comps': ['c', 'e', 'l', 't2'], 'bin_nr': 200,
                      'c'   : {'fn'      : f'{thaao_c}_10m_wind_speed_', 'column': 2, 'data': pd.DataFrame(),
                               'data_res': ''},
                      'e'   : {'fn': '', 'column': np.nan, 'data': pd.DataFrame(), 'data_res': ''},
                      'l'   : {'fn': '', 'column': np.nan, 'data': pd.DataFrame(), 'data_res': ''},
                      't'   : {'fn': '', 'column': np.nan, 'data': pd.DataFrame(), 'data_res': ''},
                      't1'  : {'fn': '', 'column': np.nan, 'data': pd.DataFrame(), 'data_res': ''},
                      't2'  : {'fn'      : 'AWS_THAAO_', 'column': 'WS_aws', 'data': pd.DataFrame(),
                               'data_res': ''}},
        'windu'    : {'name': 'windd', 'ref_x': '', 'min': np.nan, 'max': np.nan, 'res_min': np.nan, 'res_max': np.nan,
                      'uom' : '[m/s]', 'comps': [], 'bin_nr': np.nan,
                      'c'   : {'fn'      : f'', 'column': np.nan, 'data': pd.DataFrame(),
                               'data_res': ''},
                      'e'   : {'fn'  : f'{thaao_e}_10m_u_component_of_wind_', 'column': 4,
                               'data': pd.DataFrame(), 'data_res': ''},
                      'l'   : {'fn'  : f'{thaao_l}_10m_u_component_of_wind_', 'column': 4,
                               'data': pd.DataFrame(), 'data_res': ''},
                      't'   : {'fn': '', 'column': np.nan, 'data': pd.DataFrame(), 'data_res': ''},
                      't1'  : {'fn': '', 'column': np.nan, 'data': pd.DataFrame(), 'data_res': ''},
                      't2'  : {'fn'      : 'AWS_THAAO_', 'column': np.nan, 'data': pd.DataFrame(),
                               'data_res': ''}},
        'windv'    : {'name': 'winds', 'ref_x': 't', 'min': 0, 'max': 30, 'res_min': -10, 'res_max': 10,
                      'uom' : '[m/s]', 'comps': [], 'bin_nr': 200,
                      'c'   : {'fn'      : f'', 'column': np.nan, 'data': pd.DataFrame(),
                               'data_res': ''},
                      'e'   : {'fn'  : f'{thaao_e}_10m_v_component_of_wind_', 'column': 2,
                               'data': pd.DataFrame(), 'data_res': ''},
                      'l'   : {'fn'  : f'{thaao_l}_10m_v_component_of_wind_', 'column': 2,
                               'data': pd.DataFrame(), 'data_res': ''},
                      't'   : {'fn': '', 'column': np.nan, 'data': pd.DataFrame(), 'data_res': ''},
                      't1'  : {'fn': '', 'column': np.nan, 'data': pd.DataFrame(), 'data_res': ''},
                      't2'  : {'fn'      : 'AWS_THAAO_', 'column': np.nan, 'data': pd.DataFrame(),
                               'data_res': ''}}
        }
