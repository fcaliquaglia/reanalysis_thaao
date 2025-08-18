#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Setup configuration for data paths, datasets, variables, and seasonal settings.
"""

__author__ = "Filippo Cali' Quaglia"
__credits__ = []
__license__ = "GPL"
__version__ = "0.1"
__email__ = "filippo.caliquaglia@ingv.it"
__status__ = "Research"

import string
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import tools as tls


# ========== PLOTTING CONFIGURATION ==========
dpi = 300
SMALL_SIZE = 12
myFmt = mdates.DateFormatter('%d-%b')
letters = list(string.ascii_lowercase)

# ========== FILE SYSTEM PATHS ==========
shared_drive = Path(r'H:\Shared drives')

basefol = {
    'c': {
        'base': shared_drive / 'Reanalysis' / 'carra1',
        'parquets': shared_drive / 'Reanalysis' / 'carra1' / 'parquets',
        'raw': shared_drive / 'Reanalysis' / 'carra1' / 'raw',
    },
    'e': {
        'base': shared_drive / 'Reanalysis' / 'era5',
        'parquets': shared_drive / 'Reanalysis' / 'era5' / 'parquets',
        'raw': shared_drive / 'Reanalysis' / 'era5' / 'raw',
    },
    't': {
        'base': shared_drive / 'Dati_THAAO',
        'arcsix': shared_drive / 'Dati_THAAO' / 'thaao_arcsix',
    },
    'out': {
        'base': shared_drive / 'Dati_elab_docs' / 'thaao_reanalysis',
        'parquets': shared_drive / 'Dati_elab_docs' / 'thaao_reanalysis' / 'parquets',
    }
}

# ========== DATASET CONFIGURATION ==========
datasets = {
    'THAAO': {'switch': False, 'fn': 'THAAO'},
    'Alert': {'switch': False, 'fn': 'Alert'},
    'Villum': {'switch': False, 'fn': 'Villum'},
    'Sigma-A': {'switch': False, 'fn': 'Sigma-A'},
    'Sigma-B': {'switch': False, 'fn': 'Sigma-B'},
    'Summit': {'switch': False, 'fn': 'Summit'},
    'buoys': {'switch': True, 'fn': '2024Rprocessed'},
    'dropsondes': {'switch': False, 'fn': ''},
    'p3_tracks': {'switch': False, 'fn': ''},
    'g3_tracks': {'switch': False, 'fn': ''},
    'radiosondes': {'switch': False, 'fn': ''}
}

# Active dataset location
location = next((info['fn']
                for info in datasets.values() if info.get('switch')), None)

# ========== VARIABLE GROUPS ==========
met_vars = ['temp', 'surf_pres', 'rh', 'iwv', 'windd', 'winds', 'precip']
rad_comps_vars = ['sw_up', 'lw_up', 'lw_down', 'sw_down']
rad_flux_vars = ['alb', 'sw_lw_net', 'sw_net', 'lw_net']
cloud_vars = ['tcc', 'cbh']  # lwp
technical_vars = ['windu', 'windv', 'dewpt', 'sw_net', 'lw_net']
extra_vars = ['orog']

# Variables requiring custom handling during resampling
cumulative_vars = {'alb', 'lw_net', 'sw_net', 'sw_up',
                   'sw_down', 'lw_up', 'lw_down', 'sw_lw_net', 'precip'}

# Primary list of variables to analyze
if datasets['buoys']['switch'] == True:
    list_var = ['sw_up', 'sw_down', 'temp',
                'surf_pres', 'rh', 'windd', 'winds', 'orog']
else:
    list_var = met_vars + cloud_vars + rad_comps_vars + rad_flux_vars

# keep time resolution in order for taylor diagrams!
tres_list = ['original', '6h', '12h',  '24h']
tres = var = ''

years = np.arange(2016, 2025)

# ========== RESAMPLING THRESHOLDS ==========
min_frac = 0.75             # Required fraction of native data per resample window
rad_low_thresh = 1e0        # Minimum valid radiation threshold
precip_low_thresh = 1e-3    # Minimum valid precipitation threshold
cbh_low_thresh = 250        # Minimum valid cloud base height threshold
carra1_ground_elev = 150
era5_ground_elev = 181
thaao_ground_elev = 220

# ========== DATE RANGES ==========
dateranges = {
    'aws_ecapac': pd.date_range(start='2023-04-01', end='2024-12-31'),
    'ceilometer': pd.date_range(start='2019-09-01', end='2024-12-31'),
    'rad': pd.date_range(start='2009-09-01', end='2024-12-31', freq='YE'),
    'hatpro': pd.date_range(start='2016-09-01', end='2024-10-30', freq='YE'),
}

# ========== SEASONAL DEFINITIONS ==========
all_seasons = {'all': {'months': list(range(1, 13)), 'col': 'pink'}}
seasons = {
    'DJF': {'months': [12, 1, 2], 'col': 'blue'},
    'MAM': {'months': [3, 4, 5], 'col': 'green'},
    'JJA': {'months': [6, 7, 8], 'col': 'orange'},
    'SON': {'months': [9, 10, 11], 'col': 'brown'}
}
seasons_subset = {k: v for k, v in seasons.items() if k != 'all'}

# ========== VARIABLE METADATA ==========
var_dict = {
    'c': {
        'nanval': np.nan,
        'col': 'red',
        'col_ori': 'indianred',
        'col_distr': 'salmon',
        'label': 'CARRA1',
        'cmap': 'jet',
        'cmap_pos': (0.2, 0.85, 0.6, 0.1),
        'label_uom': ''
    },
    'e': {
        'nanval': -32767.0,
        'col': 'blue',
        'col_ori': 'dodgerblue',
        'col_distr': 'skyblue',
        'label': 'ERA5',
        'cmap': 'viridis',
        'cmap_pos': (0.2, 0.65, 0.6, 0.1),
        'label_uom': ''
    },
    't': {
        'nanval': -9999.9,
        'col': 'black',
        'col_ori': 'dimgray',
        'col_distr': 'slategray',
        'label': location,
        'cmap': 'Greys',
        'cmap_pos': (0.2, 0.05, 0.6, 0.1),
        'label_uom': ''
    },
    't1': {
        'nanval': np.nan,
        'col': 'green',
        'col_ori': 'seagreen',
        'col_distr': 'mediumseagreen',
        'label': 'HATPRO',
        'cmap': 'plasma',
        'cmap_pos': (0.2, 0.45, 0.6, 0.1),
        'label_uom': ''
    },
    't2': {
        'nanval': np.nan,
        'col': 'purple',
        'col_ori': 'darkorchid',
        'col_distr': 'orchid',
        'label': 'AWS_ECAPAC',
        'cmap': 'cividis',
        'cmap_pos': (0.2, 0.25, 0.6, 0.1),
        'label_uom': ''
    }
}

# ========== LOAD VARIABLE CONFIGURATION FILES ==========
config_dir = Path('config')
extr = {}

for var_name in met_vars + rad_flux_vars + rad_comps_vars + cloud_vars + technical_vars + extra_vars:
    cfg_path = config_dir / f'{var_name}.yaml'
    cfg = tls.load_and_process_yaml(cfg_path)
    if cfg is not None:
        extr[var_name] = cfg
