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

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

## FOLDERS
basefol_c = os.path.join('H:\\Shared drives', 'Dati_elab_docs', 'reanalysis', 'carra', 'thaao')
basefol_e = os.path.join('H:\\Shared drives', 'Dati_elab_docs', 'reanalysis', 'era5', 'thaao')
basefol_l = os.path.join('H:\\Shared drives', 'Dati_elab_docs', 'reanalysis', 'era5-land', 'thaao')
basefol_t = os.path.join('H:\\Shared drives', 'Dati_THAAO')
basefol_t_elab = os.path.join('H:\\Shared drives', 'Dati_elab_docs')
basefol_r = os.path.join('H:\\Shared drives', 'Reanalysis')
basefol_out = os.path.join('H:\\Shared drives', 'Dati_elab_docs', 'thaao_reanalysis')

##
tres = '3h'
tres_rs = tres  # only for radiosoundings
var_list = ['temp']
# 'tcc'
years = np.arange(2000, 2025, 1)

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
c_col = 'red'
e_col = 'blue'
l_col = 'darkgreen'
t_col = 'black'
t1_col = 'green'
t2_col = 'purple'

c_col_ori = 'orange'
e_col_ori = 'cyan'
l_col_ori = 'lightgreen'
t_col_ori = 'grey'
t1_col_ori = 'lightgreen'
t2_col_ori = 'violet'

myFmt = mdates.DateFormatter('%d-%b')

extr = {'iwv': {'min': 0, 'max': 20, 'res_min': -5, 'res_max': 5}}