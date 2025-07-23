#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Setup configuration for data paths, datasets, variables, and seasonal settings.
"""

__author__ = "Filippo Cali' Quaglia"
__credits__ = ["??????"]
__license__ = "GPL"
__version__ = "0.1"
__email__ = "filippo.caliquaglia@ingv.it"
__status__ = "Research"

import datetime as dt
import string
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml


def replace_none_with_nan(obj):
    if isinstance(obj, dict):
        return {k: replace_none_with_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_none_with_nan(v) for v in obj]
    return np.nan if obj is None else obj


def load_and_process_yaml(path: Path):
    if not path.exists():
        print(f'⚠️ Config file not found for variable: {path.stem}')
        return None

    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = replace_none_with_nan(cfg)

    # Replace placeholders in filenames for keys 'c' and 'e'
    for key in ('c', 'e'):
        if key in cfg and 'fn' in cfg[key]:
            cfg[key]['fn'] = (
                cfg[key]['fn']
                .replace('thaao_c', 'carra1')
                .replace('thaao_e', 'era5_NG')
            )
    return cfg


# ========== CONFIGURATION ==========
dpi = 300
SMALL_SIZE = 12
myFmt = mdates.DateFormatter('%d-%b')
letters = list(string.ascii_lowercase)

# ========== FILE SYSTEM ==========
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

# ========== DATASET SELECTION ==========
datasets = {
    'THAAO': {'switch': True, 'fn': 'THAAO'},
    'Alert': {'switch': False, 'fn': 'Alert'},
    'Villum': {'switch': False, 'fn': 'Villum'},
    'Sigma-A': {'switch': False, 'fn': 'Sigma-A'},
    'Sigma-B': {'switch': False, 'fn': 'Sigma-B'},
    'Summit': {'switch': False, 'fn': 'Summit'},
    'buoys': {'switch': False, 'fn': '2024Rprocessed'},
    'dropsondes': {'switch': False, 'fn': ''},
    'p3_tracks': {'switch': False, 'fn': ''},
    'g3_tracks': {'switch': False, 'fn': ''},
    'radiosondes': {'switch': False, 'fn': ''}
}

location = next((info['fn']
                for info in datasets.values() if info.get('switch')), None)

# ========== VARIABLES ==========
thaao_c, thaao_e, thaao_t = 'carra1', 'era5_NG', 'thaao'

met_vars = ['temp', 'surf_pres', 'rh', 'iwv', 'windd', 'winds', 'precip']
rad_vars = ['lw_net', 'sw_up', 'lw_up', 'lw_down', 'sw_down']
cloud_vars = ['cbh', 'lwp', 'tcc']
technical_vars = ['windu', 'windv', 'dewpt', 'sw_net', 'lw_net']
extra_vars = ['orog']

# met_vars + rad_vars  # you can add + cloud_vars if needed
list_var = rad_vars  + met_vars #  ['orog'] # 
tres_list = ['original', '24h'] # '3h', '6h', '12h', 
tres = var = ''

years = np.arange(2016, 2025)

# ========== RESAMPLING THRESHOLD ==========
# we require at least 75% of the values at the native resolution 
# (1h ERA5, 3h CARRA, 1min ECAPAC, 15min HATPRO, 20 min fro meteo VESPA)
# in the resampling interval to be valid for avaraging.

# For IWV VESPA, custom thesholds. CHeck the fucntion in tools.py
min_frac = 0.75

# ========== DATE RANGES ==========
dateranges = {
    'aws_ecapac': pd.date_range(start=dt.datetime(2023, 4, 1), end=dt.datetime(2024, 12, 31)),
    'ceilometer': pd.date_range(start=dt.datetime(2019, 9, 1), end=dt.datetime(2024, 12, 31)),
    'rad': pd.date_range(start=dt.datetime(2009, 9, 1), end=dt.datetime(2024, 12, 31), freq='YE'),
    'hatpro': pd.date_range(start=dt.datetime(2016, 9, 1), end=dt.datetime(2024, 10, 30), freq='YE')
}

# ========== SEASONAL SETTINGS ==========
all_seasons={'all': {'months': list(range(1, 13)), 'col': 'pink'}}
seasons = {
    'DJF': {'months': [12, 1, 2], 'col': 'blue'},
    'MAM': {'months': [3, 4, 5], 'col': 'green'},
    'JJA': {'months': [6, 7, 8], 'col': 'orange'},
    'SON': {'months': [9, 10, 11], 'col': 'brown'}
}

seasons_subset = {k: v for k, v in seasons.items() if k != 'all'}

# ========== VARIABLE METADATA ==========
var_dict = {
    'c': {'nanval': np.nan, 'col': 'red', 'col_ori': 'orange', 'label': 'CARRA1',
          'cmap': 'jet', 'cmap_pos': (0.2, 0.85, 0.6, 0.1), 'label_uom': ''},
    'e': {'nanval': -32767.0, 'col': 'blue', 'col_ori': 'cyan', 'label': 'ERA5',
          'cmap': 'viridis', 'cmap_pos': (0.2, 0.65, 0.6, 0.1), 'label_uom': ''},
    't': {'nanval': -9999.9, 'col': 'black', 'col_ori': 'grey', 'label': location, 'label_uom': ''},
    't1': {'nanval': np.nan, 'col': 'green', 'col_ori': 'lightgreen', 'label': 'HATPRO',
           'cmap': 'plasma', 'cmap_pos': (0.2, 0.45, 0.6, 0.1), 'label_uom': ''},
    't2': {'nanval': np.nan, 'col': 'purple', 'col_ori': 'violet', 'label': 'AWS_ECAPAC',
           'cmap': 'cividis', 'cmap_pos': (0.2, 0.25, 0.6, 0.1), 'label_uom': ''}
}

# ========== LOAD YAML CONFIGS ==========
config_dir = Path('config')
extr = {}

for var_name in met_vars + rad_vars + cloud_vars + technical_vars + extra_vars:
    cfg = load_and_process_yaml(config_dir / f'{var_name}.yaml')
    if cfg is not None:
        extr[var_name] = cfg
