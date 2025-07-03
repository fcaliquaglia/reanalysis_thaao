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
    Selects rows closest to regular timestamps (e.g., every 3 hours from 00:00), 
    only if within a time tolerance.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.
    freq : str
        Target frequency like '1H', '3H'.
    tol_minutes : float
        Maximum allowed time difference in minutes from the ideal target times.

    Returns:
    --------
    pd.DataFrame
        Subset of df with rows closest to ideal timestamps, within tolerance.
    """
    if df.empty:
        return df.copy()

    # Generate aligned time grid (e.g., 00:00, 03:00...) across data span
    start_time = df.index.min().normalize()  # midnight of first day
    end_time = df.index.max()
    target_times = pd.date_range(start=start_time, end=end_time, freq=freq)

    # Get closest real data points to each target
    pos_array = df.index.get_indexer(target_times, method='nearest')
    valid_mask = pos_array != -1

    pos_array = pos_array[valid_mask]
    target_times = target_times[valid_mask]
    closest_times = df.index[pos_array]

    # Check distances and apply tolerance
    time_diffs = np.abs((closest_times - target_times).total_seconds()) / 60
    within_tol_mask = time_diffs <= tol_minutes

    final_positions = pos_array[within_tol_mask]

    # Deduplicate (preserve order)
    seen = set()
    unique_positions = [
        p for p in final_positions if not (p in seen or seen.add(p))]

    return df.iloc[unique_positions]


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
        'alb': 'mean',
        'temp': 'closest',
        'alb': 'mean',
        'cbh': 'closest',
        'iwv': 'closest',
        'lwp': 'mean',
        'lw_down': 'mean',
        'lw_up': 'mean',
        'precip': 'cumul',
        'rh': 'closest',
        'surf_pres': 'closest',
        'sw_down': 'mean',
        'sw_up': 'mean',
        'tcc': 'closest',
        'temp': 'closest',
        'windd': 'closest',
        'winds': 'closest'
    }

    # Early exit if variable is wind or precipitation-related, no resampling allowed
    if (inpt.var in ['winds', 'windd']) and (inpt.tres != 'original'):
        print('NO TIME RESAMPLING FOR WIND!')
        sys.exit()

    for vvrr in inpt.extr[vr]['comps'] + [inpt.extr[vr]['ref_x']]:
        data = inpt.extr[vr][vvrr]['data']
        data, chk = tls.check_empty_df(data, vr)
        if inpt.datasets['dropsondes']['switch']:
            print('NO TIME RESAMPLING FOR DROPSONDES')
            inpt.extr[vr][vvrr]['data_res'] = {inpt.tres: data}
            return

        if not chk:
            resampled_data = {'original': data}

            if vvrr == 'c':
                if res_strategy[vr] in ['closest', 'mean']:
                    resampled_data.update({
                        '3h': data,
                        '24h': data.resample('24h').mean(),
                    })
                if res_strategy[vr] == 'cumul':
                    print('CUM RESAMPLING STRATEGY not implemented yet.')
                print(
                    f'Copied data for {vvrr}, {vr} at different time resolutions')

            elif vvrr == 'e':
                if res_strategy[vr] in ['closest', 'mean']:
                    resampled_data.update({
                        '1h': data,
                        '3h': data.resample('3h').mean(),
                        '24h': data.resample('24h').mean(),
                    })
                if res_strategy[vr] == 'cumul':
                    print('CUM RESAMPLING STRATEGY not implemented yet.')
                print(
                    f'Copied data for {vvrr}, {vr} at different time resolutions')

            elif vvrr in ['t', 't1', 't2']:
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
                    f'Extracted closest timestamps within tolerance for {vvrr}, {vr} at 3h, 1h, and native resolution + 24h averages.')

            inpt.extr[vr][vvrr]['data_res'] = resampled_data

        else:
            # Empty DataFrame case
            inpt.extr[vr][vvrr]['data_res'] = {inpt.tres: data}
            print(
                f'NOT Resampled for {vvrr}, {vr} at {inpt.tres} resolution. Probably empty DataFrame.')

    return


# def data_resampling(vr):
#     """
#     Resamples the data for a specified variable to the defined temporal resolution. The function checks
#     the specified variable for compatibility with the resampling process. If the variable is related
#     to wind or precipitation data, resampling is not allowed, and the function exits. For compatible
#     variables, it performs resampling over the associated components and reference data.

#     :param vr: The variable for which the data resampling is performed.
#     :type vr: str
#     :return: None
#     """
#     if inpt.var in ['winds', 'windd', 'precip']:
#         print('NO WIND/PRECIP RESAMPLING!')
#         sys.exit()


#     for vvrr in inpt.extr[vr]['comps'] + [inpt.extr[vr]['ref_x']]:
#         data = inpt.extr[vr][vvrr]['data']
#         data, chk = tls.check_empty_df(data, vr)
#         if not chk:
#             if inpt.datasets['dropsondes']['switch']:
#                 print('NO TIME RESAMPLING FOR DROPSONDES')
#                 inpt.extr[vr][vvrr]['data_res'] = data
#             else:
#                 data_res = data.resample(inpt.tres).mean()
#                 inpt.extr[vr][vvrr]['data_res'] = data_res
#             print(f'Resampled for {vvrr}, {vr} at {inpt.tres} resolution')
#         else:
#             inpt.extr[vr][vvrr]['data_res'] = data
#             print(
#                 f'NOT Resampled for {vvrr}, {vr} at {inpt.tres} resolution. Probably empty DataFrame.')

#     return
