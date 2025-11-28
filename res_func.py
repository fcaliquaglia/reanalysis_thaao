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
    target_times = pd.date_range(
        df.index.min().normalize(), df.index.max(), freq=freq)
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
    dtype_dict = {}
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_integer_dtype(dtype):
            dtype_dict[col] = 'Int64'
        else:
            dtype_dict[col] = dtype
    
    result = result.astype(dtype_dict)

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

    for data_typ in inpt.extr[vr]['comps'] + [inpt.extr[vr]['ref_x']]:
        data = inpt.extr[vr][data_typ]['data']
        data, chk = tls.check_empty_df(data, vr)
        data = data[~data.index.duplicated(keep="first")]

        if inpt.datasets['dropsondes']['switch']:
            print('NO TIME RESAMPLING FOR DROPSONDES')
            inpt.extr[vr][data_typ]['data_res'] = {inpt.tres: data}
            return

        if not chk:

            resampled_data = {'original': data}

            # Always get closest for 1h and 3h
            resampled_data.update({
                '1h': get_closest_subset_with_tolerance(data, '1h', tol_minutes=10),
                '3h': get_closest_subset_with_tolerance(data, '3h', tol_minutes=30),
            })

                
            if inpt.tres != 'original':
                if vr != 'precip':
                    if inpt.tres == '1ME':
                        resampled_data[inpt.tres] = data.resample(
                        inpt.tres).mean()
                        resampled_data[inpt.tres].index = resampled_data[inpt.tres].index.to_period('M').to_timestamp(how='start') + pd.Timedelta(days=14)
                    else:
                        masked = tls.mask_low_count_intervals(
                            data, data_typ, min_frac=inpt.min_frac)
                        resampled_data[inpt.tres] = masked.resample(
                        inpt.tres).mean()
                else:
                    if inpt.tres == '1ME':
                        resampled_data[inpt.tres] = data.resample(
                        inpt.tres).sum()
                        resampled_data[inpt.tres].index = resampled_data[inpt.tres].index.to_period('M').to_timestamp(how='start') + pd.Timedelta(days=14)
                    else:
                        masked = tls.mask_low_count_intervals(
                            data, data_typ, min_frac=inpt.min_frac)
                        resampled_data[inpt.tres] = masked.resample(inpt.tres).apply(
                        lambda x: x.sum() if x.notna().any() else np.nan)
            else:
                resampled_data[inpt.tres] = data
            print(f"Resampled (closest or mean) for {data_typ}, {vr}.")

            inpt.extr[vr][data_typ]['data_res'] = resampled_data

        else:
            inpt.extr[vr][data_typ]['data_res'] = {inpt.tres: data}
            print(
                f"NOT Resampled for {data_typ}, {vr} at {inpt.tres} resolution. Probably empty DataFrame.")

    return
