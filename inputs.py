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
basefol_t_elab = os.path.join('H:\\Shared drives', 'Dati_elab_docs')
basefol_out = os.path.join('H:\\Shared drives', 'Dati_elab_docs', 'thaao_reanalysis')

thaao_c = 'thaao_carra'
thaao_e = 'thaao_era5'
thaao_l = 'thaao_era5-land'
thaao_t = 'thaao'

##
tres = '3h'
list_var = ['lw_down', 'sw_down']  # , 'lw_down', 'sw_down']  # 'sw_down', 'sw_up', 'lw_down', 'lw_up']
# 'lw_down',]  # ['temp', 'rh', 'alb', 'cbh', 'precip', 'windd', 'winds', 'surf_pres', 'sw_down', 'sw_up', 'lw_up',
# 'lw_down', 'lwp', 'tcc']
var = ''

years = np.arange(2022, 2024, 1)

seass = {'all': {'name'      : 'all', 'months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'col': 'pink',
                 'col_CARRA' : 'red', 'col_ERA5': 'blue', 'col_ERA5-L': 'purple', 'col_THAAO': 'grey',
                 'col_HATPRO': 'grey', 'col_VESPA': 'grey', 'col_AWS_ECAPAC': 'purple'},
         'DJF': {'name': 'DJF', 'months': [12, 1, 2], 'col': 'blue'},
         'MAM': {'name': 'MAM', 'months': [3, 4, 5], 'col': 'green'},
         'JJA': {'name': 'JJA', 'months': [6, 7, 8], 'col': 'orange'},
         'SON': {'name': 'SON', 'months': [9, 10, 11], 'col': 'brown'},
         # 'MA' : {'name': 'MA', 'months': [3, 4], 'col': 'yellow'},
         # 'MJ' : {'name': 'MJ', 'months': [5, 6], 'col': 'cyan'},
         # 'JA': {'name': 'JA', 'months': [7, 8], 'col': 'grey'},
         # 'SO' : {'name': 'SO', 'months': [9, 10], 'col': 'purple'}
         }

SMALL_SIZE = 12

myFmt = mdates.DateFormatter('%d-%b')
letters = list(string.ascii_lowercase)

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
                      'c'   : {'fn'      : f'{thaao_c}_albedo_', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      'e'   : {'fn'      : f'{thaao_e}_forecast_albedo_', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      'l'   : {'fn'      : f'{thaao_l}_forecast_albedo_', 'col_nr': 4, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      't'   : {'fn'      : 'MERGED_SW_LW_UP_DW_METEO_', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      't1'  : {'fn': '', 'col_nr': 2, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'  : {'fn': '', 'col_nr': 2, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()}},
        'cbh'      : {'name': 'cbh', 'ref_x': 't', 'min': 0, 'max': 10000, 'res_min': -500, 'res_max': 500,
                      'uom' : '[m]', 'comps': ['c', 'e'], 'bin_nr': 200,
                      'c'   : {'fn'      : f'{thaao_c}_cloud_base_', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      'e'   : {'fn'      : f'{thaao_e}_cloud_base_height_', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      'l'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'   : {'fn'      : '_Thule_CHM190147_000_0060cloud', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      't1'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()}},
        'temp'     : {'name'   : 'temp', 'ref_x': 't', 'min': -40, 'max': 20, 'res_min': -10,
                      'res_max': 10, 'uom': '[deg]', 'comps': ['c', 'e', 'l', 't2'], 'bin_nr': 200,
                      'c'      : {'fn'      : f'{thaao_c}_2m_temperature_', 'col_nr': 2, 'data': pd.DataFrame(),
                                  'data_res': pd.DataFrame()},
                      'e'      : {'fn'      : f'{thaao_e}_2m_temperature_', 'col_nr': 2, 'data': pd.DataFrame(),
                                  'data_res': pd.DataFrame()},
                      'l'      : {'fn'      : f'{thaao_l}_2m_temperature_', 'col_nr': 2, 'data': pd.DataFrame(),
                                  'data_res': pd.DataFrame()},
                      't'      : {'fn'      : 'Meteo_weekly_all', 'col_nr': np.nan, 'data': pd.DataFrame(),
                                  'data_res': pd.DataFrame()},
                      't1'     : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'     : {'fn'      : 'AWS_THAAO_', 'col_nr': np.nan, 'data': pd.DataFrame(),
                                  'data_res': pd.DataFrame()}},
        'dewpt'    : {'name' : 'dewpt', 'ref_x': '', 'min': '', 'max': '', 'res_min': '', 'res_max': '', 'uom': '[deg]',
                      'comps': ['c', 'e', 't1', 't2'], 'bin_nr': 200,
                      'c'    : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'e'    : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'l'    : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'    : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't1'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()}},

        'lwp'      : {'name': 'lwp', 'ref_x': 't1', 'min': 0, 'max': 50, 'res_min': -20, 'res_max': 20,
                      'uom' : '[kg/m2]', 'comps': ['c', 'e', 't1'], 'bin_nr': 200,
                      'c'   : {'fn'  : f'{thaao_c}_total_column_cloud_liquid_water_', 'col_nr': 2,
                               'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'e'   : {'fn'  : f'{thaao_e}_total_column_cloud_liquid_water_', 'col_nr': 2,
                               'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'l'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't1'  : {'fn'      : 'LWP_15_min_', 'col_nr': np.nan, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      't2'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()}},
        'lw_down'  : {'name': 'lw_down', 'ref_x': 't', 'min': 0, 'max': 500, 'res_min': -20, 'res_max': 20,
                      'uom' : '[W/m2]', 'comps': ['c', 'e', 'l'], 'bin_nr': 200,
                      'c'   : {'fn'  : f'{thaao_c}_thermal_surface_radiation_downwards_', 'col_nr': 4,
                               'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'e'   : {'fn'  : f'{thaao_e}_surface_thermal_radiation_downwards_', 'col_nr': 2,
                               'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'l'   : {'fn'  : f'{thaao_l}_surface_thermal_radiation_downwards_', 'col_nr': 4,
                               'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'   : {'fn'      : 'MERGED_SW_LW_UP_DW_METEO_', 'col_nr': np.nan, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      't1'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()}},
        'lw_net'   : {'name': 'lw_down', 'ref_x': 't', 'min': 0, 'max': 500, 'res_min': -20, 'res_max': 20,
                      'uom' : '[W/m2]', 'comps': ['c', 'e', 'l'], 'bin_nr': 200,
                      'c'   : {'fn'  : f'{thaao_c}_surface_net_thermal_radiation_', 'col_nr': 2,
                               'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'e'   : {'fn'  : f'{thaao_e}_surface_net_thermal_radiation_', 'col_nr': 2,
                               'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'l'   : {'fn'  : f'{thaao_l}_surface_net_thermal_radiation_', 'col_nr': 4,
                               'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't1'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()}},
        'lw_up'    : {'name': 'lw_up', 'ref_x': 't', 'min': 0, 'max': 500, 'res_min': -20, 'res_max': 20,
                      'uom' : '[W/m2]', 'comps': ['c', 'e', 'l'], 'bin_nr': 200,
                      'c'   : {'fn': '', 'col_nr': 2, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'e'   : {'fn': '', 'col_nr': 2, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'l'   : {'fn': '', 'col_nr': 2, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'   : {'fn'      : 'MERGED_SW_LW_UP_DW_METEO_', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      't1'  : {'fn': '', 'col_nr': 2, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'  : {'fn': '', 'col_nr': 2, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()}},
        'msl'      : {'name': 'winds', 'ref_x': 't', 'min': 0, 'max': 30, 'res_min': -10, 'res_max': 10,
                      'uom' : '[m/s]', 'comps': ['c', 'e', 't1', 't2'], 'bin_nr': 200,
                      'c'   : {'fn'      : f'{thaao_c}_mean_sea_level_pressure_', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      'e'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'l'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't1'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()}},
        'precip'   : {'name': 'precip', 'ref_x': 't2', 'min': 0, 'max': 5, 'res_min': -10, 'res_max': 10,
                      'uom' : '[mm?]', 'comps': ['c', 'e'], 'bin_nr': 200,
                      'c'   : {'fn'      : f'{thaao_c}_total_precipitation_', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      'e'   : {'fn'      : f'{thaao_e}_total_precipitation_', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      'l'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't1'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'  : {'fn'      : 'AWS_THAAO_', 'col_nr': np.nan, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()}},
        'rh'       : {'name' : 'rh', 'ref_x': '', 'min': 0, 'max': 100, 'res_min': -10, 'res_max': 10, 'uom': '[%]',
                      'comps': ['c', 'e', 'l', 't2'], 'bin_nr': 200,
                      'c'    : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'e'    : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'l'    : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'    : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't1'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'   : {'fn'      : 'AWS_THAAO_', 'col_nr': np.nan, 'data': pd.DataFrame(),
                                'data_res': pd.DataFrame()}},
        'surf_pres': {'name': 'surf_pres', 'ref_x': 't', 'min': 925, 'max': 1013, 'res_min': -10, 'res_max': 10,
                      'uom' : '[hPa]', 'comps': ['c', 'e', 't2'], 'bin_nr': 200,
                      'c'   : {'fn'      : f'{thaao_c}_surface_pressure_', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      'e'   : {'fn'      : f'{thaao_e}_surface_pressure_', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      'l'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'   : {'fn'      : 'Meteo_weekly_all', 'col_nr': np.nan, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      't1'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'  : {'fn'      : 'AWS_THAAO_', 'col_nr': np.nan, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()}},
        'sw_down'  : {'name': 'sw_down', 'ref_x': 't', 'min': 0, 'max': 500, 'res_min': -20, 'res_max': 20,
                      'uom' : '[W/m2]', 'comps': ['c', 'e', 'l'], 'bin_nr': 200,
                      'c'   : {'fn'  : f'{thaao_c}_surface_solar_radiation_downwards_', 'col_nr': 2,
                               'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'e'   : {'fn'  : f'{thaao_e}_surface_solar_radiation_downwards_', 'col_nr': 2,
                               'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'l'   : {'fn'  : f'{thaao_l}_surface_solar_radiation_downwards_', 'col_nr': 4,
                               'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'   : {'fn'      : 'MERGED_SW_LW_UP_DW_METEO_', 'col_nr': np.nan, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      't1'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()}},
        'sw_net'   : {'name': 'sw_down', 'ref_x': 't', 'min': 0, 'max': 500, 'res_min': -20, 'res_max': 20,
                      'uom' : '[W/m2]', 'comps': ['c', 'e', 'l'], 'bin_nr': 200,
                      'c'   : {'fn'  : f'{thaao_c}_surface_net_solar_radiation_', 'col_nr': 2,
                               'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'e'   : {'fn'  : f'{thaao_e}_surface_net_solar_radiation_', 'col_nr': 2,
                               'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'l'   : {'fn'  : f'{thaao_l}_surface_net_solar_radiation_', 'col_nr': 4,
                               'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't1'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()}},
        'sw_up'    : {'name': 'sw_up', 'ref_x': 't', 'min': 0, 'max': 500, 'res_min': -20, 'res_max': 20,
                      'uom' : '[W/m2]', 'comps': ['c', 'e', 'l'], 'bin_nr': 200,
                      'c'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'e'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'l'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'   : {'fn'      : 'MERGED_SW_LW_UP_DW_METEO_', 'col_nr': np.nan, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      't1'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()}},
        'tcc'      : {'name': 'tcc', 'ref_x': 't', 'min': 0, 'max': 1, 'res_min': -50, 'res_max': 50,
                      'uom' : '[octave]', 'comps': ['c', 'e'], 'bin_nr': 200,
                      'c'   : {'fn'      : f'{thaao_c}_total_cloud_cover_', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      'e'   : {'fn'      : f'{thaao_e}_total_cloud_cover_', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      'l'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'   : {'fn'      : '_Thule_CHM190147_000_0060cloud', 'col_nr': np.nan, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      't1'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()}},
        'windd'    : {'name': 'windd', 'ref_x': 't', 'min': 0, 'max': 360, 'res_min': -90, 'res_max': 90,
                      'uom' : '[deg]', 'comps': ['c', 'e', 't', 't1'], 'bin_nr': 200,
                      'c'   : {'fn'      : f'{thaao_c}_10m_wind_direction_', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      'e'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'l'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't1'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'  : {'fn'      : 'AWS_THAAO_', 'col_nr': np.nan, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()}},
        'winds'    : {'name': 'winds', 'ref_x': 't', 'min': 0, 'max': 30, 'res_min': -10, 'res_max': 10,
                      'uom' : '[m/s]', 'comps': ['c', 'e', 't', 't1'], 'bin_nr': 200,
                      'c'   : {'fn'      : f'{thaao_c}_10m_wind_speed_', 'col_nr': 2, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()},
                      'e'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      'l'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't'   : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't1'  : {'fn': '', 'col_nr': np.nan, 'data': pd.DataFrame(), 'data_res': pd.DataFrame()},
                      't2'  : {'fn'      : 'AWS_THAAO_', 'col_nr': np.nan, 'data': pd.DataFrame(),
                               'data_res': pd.DataFrame()}}}

aws_ecapac_daterange = pd.date_range(start=dt.datetime(2023, 4, 1), end=dt.datetime(2024, 12, 31), freq='1D')
ceilometer_daterange = pd.date_range(start=dt.datetime(2019, 9, 1), end=dt.datetime(2024, 12, 31), freq='1D')
