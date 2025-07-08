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

import sys
import inputs as inpt
import tools as tls
import pandas as pd
import numpy as np


def get_closest_subset_with_tolerance(df, freq, tol_minutes):
    """
    Returns a DataFrame indexed by regular timestamps (e.g., every 3 hours from 00:00),
    where each row is filled with the closest actual data within a given tolerance.
    If no data is found within tolerance, the row is NaN.
    """
    if df.empty:
        return pd.DataFrame(columns=df.columns)

    # Generate regular time grid (target)
    target_times = pd.date_range(df.index.min().normalize(), df.index.max(), freq=freq)
    # Get closest real indices for each target time
    indexer = df.index.get_indexer(target_times, method='nearest')

    # Identify invalid indices and calculate time differences
    valid = indexer != -1
    valid_targets = target_times[valid]
    matched_times = df.index[indexer[valid]]
    diffs = np.abs((matched_times - valid_targets).total_seconds()) / 60

    # Apply tolerance mask
    within_tol = diffs <= tol_minutes
    final_indexer = np.where(valid, np.where(within_tol, indexer, -1), -1)

    # Build output using masked assignment
    result = pd.DataFrame(index=target_times, columns=df.columns)
    result = result.astype(df.dtypes.to_dict())

    valid_rows = final_indexer != -1
    if valid_rows.any():
        selected_rows = df.iloc[final_indexer[valid_rows]]
        result.iloc[valid_rows] = selected_rows

    return result


def data_resampling(vr):
    """
    Resample or copy data within the input extraction structure based on the variable and requested time resolution.

    Parameters:
    -----------
    vr : str
        Variable region/key for which data resampling should be performed. This is used to access
        the specific data components and reference data in `inpt.extr[vr]`.

    Behavior:
    ---------
    - Immediately exits with a message if the current variable (`inpt.var`) is one of ['winds', 'windd', 'precip'],
      as these variables should not be resampled.
    - Iterates over all components listed in `inpt.extr[vr]['comps']` plus the reference `ref_x`.
    - For each component (`vvrr`):
        - Checks if the data DataFrame is empty via `tls.check_empty_df`.
        - If data is not empty:
            - If `inpt.tres` (time resolution) is 'original':
                - For components 'c' or 'e', copies the data as is without resampling.
                - For components 't', 't1', 't2', extracts data closest to every 3-hour and 1-hour timestamps,
                  within tolerances of ±30 minutes for 3-hour intervals and ±10 minutes for 1-hour intervals.
                  Also stores the native data without modification.
                - For any other components, copies data without resampling.
            - If `inpt.tres` is not 'original':
                - If dropsondes resampling switch is ON, copies data without resampling.
                - Otherwise, resamples data using `.resample(inpt.tres).mean()`.
        - If data is empty, copies data without resampling and logs a message.

    The resampled or copied data is saved back to `inpt.extr[vr][vvrr]['data_res']`.

    Notes:
    ------
    - Assumes that `data` DataFrames are indexed by pandas `DatetimeIndex`.
    - The helper function `get_closest_subset_with_tolerance` finds nearest existing timestamps
      to specified intervals but only includes those within a defined tolerance.
    - The function prints messages to indicate processing steps and decisions.
    - Exits the program early if resampling is not allowed for specific variables.

    Returns:
    --------
    None
    """

    res_strategy = {
        'temp': 'closest',
        'cbh': 'closest',
        'iwv': 'closest',
        'lwp': 'closest',
        'lw_down': 'closest',
        'lw_up': 'closest',
        'precip': 'cumul',
        'rh': 'closest',
        'surf_pres': 'closest',
        'sw_down': 'closest',
        'sw_up': 'closest',
        'tcc': 'closest',
        'temp': 'closest',
        'windd': 'closest',
        'winds': 'closest'
    }

    for data_typ in inpt.extr[vr]['comps'] + [inpt.extr[vr]['ref_x']]:           
        data = inpt.extr[vr][data_typ]['data']
        data, chk = tls.check_empty_df(data, vr)
        
        if inpt.datasets['dropsondes']['switch']:
            print('NO TIME RESAMPLING FOR DROPSONDES')
            inpt.extr[vr][data_typ]['data_res'] = {inpt.tres: data}
            return
        if vr in ['winds', 'windd'] and data_typ=='c':
            resampled_data = {'original': data}
            wspd = inpt.extr['winds'][data_typ]['data']['winds']
            wdir = inpt.extr['windd'][data_typ]['data']['windd']
            wspd, chk1 = tls.check_empty_df(wspd, 'winds')
            wdir, chk2 = tls.check_empty_df(wdir, 'windd')
            
            if not chk1 and not chk2:
                common_index = wspd.index.intersection(wdir.index)
                wspd = wspd.loc[common_index]
                wdir = wdir.loc[common_index]

                u, v = tls.decompose_wind(wspd, wdir)
                uv_df = pd.DataFrame({'u': u, 'v': v}, index=wspd.index)
        
                resampled_uv = {
                    'original': uv_df,
                    '1h': uv_df.resample('1h').mean(),
                    '3h': uv_df.resample('3h').mean(),
                    inpt.tres: uv_df.resample(inpt.tres).mean()
                }
        
                winds_resampled = {}
                windd_resampled = {}
                for key, df in resampled_uv.items():
                    spd, dire = tls.recompose_wind(df['u'], df['v'])
                    winds_resampled[key] = pd.DataFrame(spd, columns=['winds'])
                    windd_resampled[key] = pd.DataFrame(dire, columns=['windd'])
        
                inpt.extr['winds'][data_typ]['data_res'] = winds_resampled
                inpt.extr['windd'][data_typ]['data_res'] = windd_resampled
        
                print(f"Wind resampled via MetPy and recomposed for {data_typ}")
                continue
            
        if not chk:
            resampled_data = {'original': data}

            if data_typ == 'c':
                if res_strategy[vr] == 'closest':
                    resampled_data.update({
                        '1h': get_closest_subset_with_tolerance(data, '1h', tol_minutes=10),
                        '3h': get_closest_subset_with_tolerance(data, '3h', tol_minutes=30),
                        inpt.tres: data.resample(inpt.tres).mean()
                    })
                if res_strategy[vr] =='mean':
                    resampled_data.update({
                        '3h': data,
                        inpt.tres: data.resample(inpt.tres).mean()
                    })
                if res_strategy[vr] == 'cumul':
                    resampled_data.update({
                        '1h': data,
                        '3h': data.resample('3h').sum(),
                        inpt.tres: data.resample(inpt.tres).sum()
                    })
                print(
                    f'Resampled (mean or cumul) for {data_typ}, {vr}.')
                
            elif data_typ == 'e':
                if res_strategy[vr] == 'closest':
                    resampled_data.update({
                        '1h': get_closest_subset_with_tolerance(data, '1h', tol_minutes=10),
                        '3h': get_closest_subset_with_tolerance(data, '3h', tol_minutes=30),
                        inpt.tres: data.resample(inpt.tres).mean()
                    })
                if res_strategy[vr] == 'mean':
                    resampled_data.update({
                        '1h': data,
                        '3h': data.resample('3h').mean(),
                        inpt.tres: data.resample(inpt.tres).mean()
                    })
                if res_strategy[vr] == 'cumul':
                    resampled_data.update({
                        '1h': data,
                        '3h': data.resample('3h').sum(),
                        inpt.tres: data.resample(inpt.tres).sum()
                    })
                print(
                    f'Resampled (mean or cumul) for {data_typ}, {vr}.')

            elif data_typ in ['t', 't1', 't2']:
                if res_strategy[vr] == 'closest':
                    resampled_data.update({
                        '1h': get_closest_subset_with_tolerance(data, '1h', tol_minutes=10),
                        '3h': get_closest_subset_with_tolerance(data, '3h', tol_minutes=30),
                        '24h': data.resample('24h').mean(),
                    })
                if res_strategy[vr] == 'mean':
                    resampled_data.update({
                        '1h': data.resample('1h').mean(),
                        '3h': data.resample('3h').mean(),
                        '24h': data.resample('24h').mean(),
                    })
                if res_strategy[vr] == 'cumul':
                    resampled_data.update({
                        '1h': data.resample('1h').sum(),
                        '3h': data.resample('3h').sum(),
                        '24h': data.resample('24h').sum(),
                    })
                print(
                    f'Extracted closest timestamps within tolerance for {data_typ}, {vr}.')
            inpt.extr[vr][data_typ]['data_res'] = resampled_data

        else:
            # Empty DataFrame case
            inpt.extr[vr][data_typ]['data_res'] = {inpt.tres: data}
            print(
                f'NOT Resampled for {data_typ}, {vr} at {inpt.tres} resolution. Probably empty DataFrame.')

    return

