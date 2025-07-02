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
    Select rows from df whose timestamps are closest to regular intervals
    at frequency 'freq', but only include if the closest is within tol_minutes.

    Parameters:
    -----------
    df : pandas.DataFrame
        Time-indexed DataFrame to subset.
    freq : str
        Frequency string compatible with pandas date_range (e.g. '3H', '1H').
    tol_minutes : float
        Maximum allowed difference in minutes between target timestamp and closest timestamp.

    Returns:
    --------
    pandas.DataFrame
        Subset of df containing rows closest to target times within tolerance.
    """
    target_times = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq=freq)
    selected_indices = []

    for tt in target_times:
        pos = df.index.get_indexer(
            [tt], method='nearest')[0]
        closest_time = df.index[pos]
        diff_minutes = abs(
            (closest_time - tt).total_seconds()) / 60

        if diff_minutes <= tol_minutes:
            selected_indices.append(pos)

    unique_indices = sorted(set(selected_indices))
    return df.iloc[unique_indices]


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
            - If `inpt.tres` (time resolution) is 'native':
                - For components 'c' or 'e', copies the data as is without resampling.
                - For components 't', 't1', 't2', extracts data closest to every 3-hour and 1-hour timestamps,
                  within tolerances of ±30 minutes for 3-hour intervals and ±10 minutes for 1-hour intervals.
                  Also stores the native data without modification.
                - For any other components, copies data without resampling.
            - If `inpt.tres` is not 'native':
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

    # Early exit if variable is wind or precipitation-related, no resampling allowed
    if inpt.var in ['winds', 'windd', 'precip']:
        print('NO WIND/PRECIP RESAMPLING!')
        sys.exit()

    for vvrr in inpt.extr[vr]['comps'] + [inpt.extr[vr]['ref_x']]:
        data = inpt.extr[vr][vvrr]['data']
        data, chk = tls.check_empty_df(data, vr)

        if not chk:
            if inpt.tres == 'native':
                if vvrr in ['c', 'e']:
                    # Direct copy for these components at native resolution
                    resampled_data = {}
                    resampled_data[inpt.tres] = data
                    inpt.extr[vr][vvrr]['data_res'] = resampled_data
                    print(
                        f'Copied data for {vvrr}, {vr} at native resolution (no resampling)')
                elif vvrr in ['t', 't1', 't2']:
                    # Extract closest timestamps within tolerance windows for specific time intervals
                    resampled_data = {}
                    resampled_data['3h'] = get_closest_subset_with_tolerance(
                        data, '3h', tol_minutes=30)
                    resampled_data['1h'] = get_closest_subset_with_tolerance(
                        data, '1h', tol_minutes=10)
                    resampled_data['native'] = data

                    inpt.extr[vr][vvrr]['data_res'] = resampled_data
                    print(
                        f'Extracted closest timestamps within tolerance for {vvrr}, {vr} at 3h, 1h, and native resolution')
                # else:
                #     # For other components, just copy data as-is at native resolution
                #     inpt.extr[vr][vvrr]['data_res'] = data
                #     print(
                #         f'Copied data for {vvrr}, {vr} at native resolution (no resampling)')
            else:
                # Non-native time resolutions
                if inpt.datasets['dropsondes']['switch']:
                    print('NO TIME RESAMPLING FOR DROPSONDES')
                    resampled_data = {}
                    resampled_data[inpt.tres] = data
                    inpt.extr[vr][vvrr]['data_res'] = resampled_data
                else:
                    resampled_data = {}
                    resampled_data[inpt.tres] = data.resample(inpt.tres).mean()
                    inpt.extr[vr][vvrr]['data_res'] = resampled_data
                print(f'Resampled for {vvrr}, {vr} at {inpt.tres} resolution')
        else:
            # Empty DataFrame case: copy as-is and print message
            resampled_data = {}
            resampled_data[inpt.tres] = data
            inpt.extr[vr][vvrr]['data_res'] = resampled_data
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
