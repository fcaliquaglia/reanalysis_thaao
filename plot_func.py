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

import copy as cp
import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd

import inputs as inpt
import tools as tls


# import matplotlib
# matplotlib.use('WebAgg')


def plot_ts(period_label):
    """
    Plots time series data for each year in the provided dataset.

    This function generates a multi-panel plot with each panel corresponding to one year
    of data. It plots the original and resampled resolutions for various variables and
    optionally overlays vertical lines to indicate specific periods. The resulting plot
    is saved as an image in the output directory specified in `inpt`.

    :param period_label: A string used in the filename to describe the period of the data.
    :type period_label: str
    :return: None
    """
    print('TIMESERIES')
    plt.ioff()
    n_years = len(inpt.years)
    fig, ax = plt.subplots(n_years, 1, figsize=(12, 17), dpi=inpt.dpi)
    ax = np.atleast_1d(ax)
    str_name = f"{inpt.tres} {period_label} ts {inpt.var} {inpt.var_dict['t']['label']} {inpt.years[0]}-{inpt.years[-1]}"
    fig.suptitle(str_name, fontweight='bold')
    fig.subplots_adjust(top=0.93)

    # Plotting style kwargs
    kwargs_ori = {'alpha': 0.02, 'lw': 0, 'marker': '.', 'ms': 1}
    kwargs_res = {'lw': 0, 'marker': '.', 'ms': 2}

    # Cache frequently used variables
    var_data = inpt.extr[inpt.var]
    comps = var_data['comps']
    ref_x = var_data['ref_x']
    plot_vars_all = comps + [ref_x]
    plot_vars = tls.plot_vars_cleanup(plot_vars_all, var_data)

    for i, year in enumerate(inpt.years):
        print(f"plotting {year}")

        # Boolean mask for original and resampled data for this year
        for data_typ in plot_vars:
            # Original data for the year
            null, chck = tls.check_empty_df(var_data[data_typ]['data'][inpt.var], inpt.var)
            if chck:
                continue
            data_ori = var_data[data_typ]['data'][inpt.var]
            mask_ori = data_ori.index.year == year
            if mask_ori.any():
                ax[i].plot(data_ori.loc[mask_ori],
                           color=inpt.var_dict[data_typ]['col_ori'], **kwargs_ori)

            # Resampled data for the year
            data_res = var_data[data_typ]['data_res'][inpt.var]
            mask_res = data_res.index.year == year
            if mask_res.any():
                ax[i].plot(data_res.loc[mask_res], color=inpt.var_dict[data_typ]['col'],
                           label=inpt.var_dict[data_typ]['label'], **kwargs_res)

        # Plot vertical lines for 'alb' variable during specific date ranges
        if inpt.var == 'alb':
            freq = inpt.tres
            # Define date ranges
            range1 = pd.date_range(start=pd.Timestamp(
                year, 1, 1), end=pd.Timestamp(year, 2, 15), freq=freq)
            range2 = pd.date_range(start=pd.Timestamp(
                year, 11, 1), end=pd.Timestamp(year, 12, 31), freq=freq)
            ax[i].vlines(range1.values, 0, 1, color='grey', alpha=0.3)
            ax[i].vlines(range2.values, 0, 1, color='grey', alpha=0.3)

        # Format the subplot axes (assuming this function is defined elsewhere)
        format_ts(ax, year, i)

    plt.xlabel('Time')
    plt.legend(ncol=2)
    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_residuals(period_label):
    """
    Plots residuals for each year in the dataset, comparing components against the reference data.
    Adds visual guides such as zero lines and seasonal vertical ranges for specific variables.

    :param period_label: Label identifying the specific period for the residual plots
    :type period_label: str
    :return: None
    """
    print('RESIDUALS')
    plt.ioff()
    n_years = len(inpt.years)
    fig, ax = plt.subplots(n_years, 1, figsize=(12, 17), dpi=inpt.dpi)
    ax = np.atleast_1d(ax)
    str_name = f"{inpt.tres} {period_label} residuals_{inpt.var} {inpt.var_dict['t']['label']} {inpt.years[0]}-{inpt.years[-1]}"
    fig.suptitle(str_name, fontweight='bold')
    fig.subplots_adjust(top=0.93)

    plot_kwargs = {'lw': 1, 'marker': '.', 'ms': 0}

    var_data = inpt.extr[inpt.var]
    comps_all = var_data['comps']
    ref_x = var_data['ref_x']
    ref_data_res = var_data[ref_x]['data_res'][inpt.var]

    comps = tls.plot_vars_cleanup(comps_all, var_data)

    for i, year in enumerate(inpt.years):
        print(f"plotting {year}")

        # Plot horizontal zero line for residual reference
        daterange = pd.date_range(start=pd.Timestamp(
            year, 1, 1), end=pd.Timestamp(year, 12, 31))
        ax[i].plot(daterange, np.zeros(len(daterange)),
                   color='black', lw=2, ls='--')

        # Plot residuals (component - reference) for the year
        for comp_var in comps:
            comp_data_res = var_data[comp_var]['data_res'][inpt.var]
            mask_comp = comp_data_res.index.year == year
            mask_ref = ref_data_res.index.year == year
            if mask_comp.any() and mask_ref.any():
                residuals = comp_data_res.loc[mask_comp] - \
                    ref_data_res.loc[mask_ref]
                ax[i].plot(residuals, color=inpt.var_dict[comp_var]['col'],
                           label=inpt.var_dict[comp_var]['label'], **plot_kwargs)

        # Add seasonal vertical lines for 'alb' variable
        if inpt.var == 'alb':
            freq = inpt.tres
            range1 = pd.date_range(start=pd.Timestamp(
                year, 1, 1), end=pd.Timestamp(year, 2, 15), freq=freq)
            range2 = pd.date_range(start=pd.Timestamp(
                year, 11, 1), end=pd.Timestamp(year, 12, 31), freq=freq)
            ax[i].vlines(range1.values, -0.5, 0.5, color='grey', alpha=0.3)
            ax[i].vlines(range2.values, -0.5, 0.5, color='grey', alpha=0.3)

        # Format axis (assuming format_ts accepts residuals flag)
        format_ts(ax, year, i, residuals=True)

    plt.xlabel('Time')
    plt.legend()
    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_scatter(period_label):
    """
    Generates a 2x2 grid of scatter plots or 2D histograms for the specified period label.

    Each subplot corresponds to a component of the dataset, filtered by season or 'all'.
    Polynomial fits are calculated and plots formatted before saving.

    :param period_label: Identifier for the time period (e.g., season) used to filter and label plots.
    :type period_label: str
    :return: None
    """
    print(f"SCATTERPLOTS {period_label}")
    plt.ioff()
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=inpt.dpi)
    axs = ax.ravel()

    str_name = f"{inpt.tres} {inpt.seass[period_label]['name']} scatter {inpt.var} {inpt.var_dict['t']['label']} {inpt.years[0]}-{inpt.years[-1]}"
    fig.suptitle(str_name, fontweight='bold')
    fig.subplots_adjust(top=0.93)

    var_data = inpt.extr[inpt.var]
    comps_all = var_data['comps']
    ref_x = var_data['ref_x']
    x = var_data[ref_x]['data_res'][inpt.var]
    comps = tls.plot_vars_cleanup(comps_all, var_data)

    # Remove unused frames based on number of components
    frame_and_axis_removal(axs, len(comps))

    # Create complete time index across all years
    time_range = pd.date_range(
        start=pd.Timestamp(inpt.years[0], 1, 1, 0, 0),
        end=pd.Timestamp(inpt.years[-1], 12, 31, 23, 59),
        freq=inpt.tres
    )

    # Reindex reference data once outside the loop
    x_all = x.reindex(time_range).astype(float)  # ensures NaNs for missing
    season_months = inpt.seass[period_label]['months']
    x_season = x_all.loc[x_all.index.month.isin(season_months)]

    for i, comp in enumerate(comps):
        y = var_data[comp]['data_res'][inpt.var]
        y_all = y.reindex(time_range).astype(float)
        y_season = y_all.loc[y_all.index.month.isin(season_months)]

        # Boolean index where neither x nor y are NaN for the variable
        valid_idx = ~(x_season.isna() | y_season.isna())

        print(
            f"plotting scatter {inpt.var_dict['t']['label']}-{inpt.var_dict[comp]['label']}")

        if inpt.seass[period_label]['name'] != 'all':
            axs[i].scatter(
                x_season[valid_idx], y_season[valid_idx],
                s=5, facecolors='none', edgecolors=inpt.seass[period_label]['col'],
                alpha=0.5, label=period_label
            )
        else:
            bin_edges = np.linspace(
                var_data['min'], var_data['max'], var_data['bin_nr'])
            bin_size = (var_data['max'] - var_data['min']) / var_data['bin_nr']

            h = axs[i].hist2d(
                x_season[valid_idx], y_season[valid_idx],
                bins=[bin_edges, bin_edges], cmap=plt.cm.jet, cmin=1, vmin=1
            )
            axs[i].text(
                0.10, 0.90, f"bin_size={bin_size:.3f}", transform=axs[i].transAxes)

        # Check for enough data points to fit
        if valid_idx.sum() < 2:
            print('ERROR: Not enough data points for proper fit (need at least 2).')
        else:
            calc_draw_fit(
                axs, i, x_season[valid_idx], y_season[valid_idx], period_label)

        format_scatterplot(axs, comp, i)

    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_scatter_cum():
    """
    Plots cumulative scatter plots for each season (excluding 'all'), comparing components
    against reference data with fits, customized appearance, and saves the resulting figure.

    :raises ValueError: If insufficient data points are available for fitting.

    :return: None
    """
    print(f"SCATTERPLOTS cumulative")
    plt.ioff()
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=inpt.dpi)
    axs = ax.ravel()
    str_name = f"{inpt.tres} all CumSeas scatter {inpt.var} {inpt.var_dict['t']['label']} {inpt.years[0]}-{inpt.years[-1]}"
    fig.suptitle(str_name, fontweight='bold')
    fig.subplots_adjust(top=0.93)

    seass_new = {k: v for k, v in inpt.seass.items() if k != 'all'}

    var_data = inpt.extr[inpt.var]
    comps_all = var_data['comps']
    ref_x = var_data['ref_x']
    x = var_data[ref_x]['data_res'][inpt.var]
    comps = tls.plot_vars_cleanup(comps_all, var_data)

    # Prepare full time range for reindexing once
    time_range = pd.date_range(
        start=pd.Timestamp(inpt.years[0], 1, 1, 0, 0),
        end=pd.Timestamp(inpt.years[-1], 12, 31, 23, 59),
        freq=inpt.tres
    )
    x_all = x.reindex(time_range).astype(float)

    frame_and_axis_removal(axs, len(comps))

    for period_label, season in seass_new.items():
        print(f"SCATTERPLOTS CUMULATIVE {period_label}")

        season_months = season['months']
        x_season = x_all.loc[x_all.index.month.isin(season_months)]

        for i, comp in enumerate(comps):
            y = var_data[comp]['data_res'][inpt.var]
            y_all = y.reindex(time_range).astype(float)
            y_season = y_all.loc[y_all.index.month.isin(season_months)]

            valid_idx = ~(x_season.isna() |
                          y_season.isna())

            axs[i].scatter(
                x_season[valid_idx], y_season[valid_idx],
                s=5, color=season['col'], edgecolors='none', alpha=0.5, label=period_label
            )

            if valid_idx.sum() < 2:
                print('ERROR: Not enough data points for proper fit (need at least 2).')
                # Optionally raise ValueError here if needed
                # raise ValueError("Insufficient data for fitting.")
            else:
                calc_draw_fit(axs, i, x_season[valid_idx], y_season[valid_idx],
                              period_label, print_stats=False)

            format_scatterplot(axs, comp, i)
            axs[i].legend()

    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def calc_draw_fit(axs, i, xxx, yyy, per_lab, print_stats=True):
    """
    Performs linear regression on data (xxx, yyy), plots the fit line and 1:1 line,
    and optionally annotates stats (R, N, MBE, RMSE) on subplot axs[i].

    :param axs: Matplotlib axes array.
    :param i: Index of subplot to plot on.
    :param xxx: x-data (pd.Series or np.array).
    :param yyy: y-data (pd.Series or np.array).
    :param per_lab: Key for color in inpt.seass.
    :param print_stats: Whether to display stats text.
    """
    xx = xxx.values.flatten()
    yy = yyy.values.flatten()
    b, a = np.polyfit(xx, yy, 1)  # slope, intercept
    var_min = inpt.extr[inpt.var]['min']
    var_max = inpt.extr[inpt.var]['max']
    xseq = np.linspace(var_min, var_max, 1000)

    axs[i].plot(xseq, a + b * xseq,
                color=inpt.seass[per_lab]['col'],
                lw=2.5, ls='--', alpha=0.5)
    axs[i].plot([var_min, var_max], [var_min, var_max],
                color='black', lw=1.5, ls='-')

    if print_stats:
        corcoef = np.corrcoef(xx, yy)[0, 1]
        N = len(yy)
        rmse = np.sqrt(np.nanmean((yy - xx) ** 2))
        mbe = np.nanmean(yy - xx)
        stats_text = (f"R={corcoef:.2f} N={N}\n"
                      f"y={b:+.2f}x{a:+.2f}\n"
                      f"MBE={mbe:.2f} RMSE={rmse:.2f}")
        axs[i].text(0.50, 0.30, stats_text,
                    transform=axs[i].transAxes,
                    fontsize=14, color='black',
                    ha='left', va='center',
                    bbox=dict(facecolor='white', edgecolor='white'))


def format_scatterplot(axs, comp, i):
    """
    Sets title, labels, limits, and annotations for a scatterplot axs[i] of component `comp`.

    :param axs: Array-like of matplotlib Axes.
    :param comp: Component key for labeling.
    :param i: Index of subplot.
    """
    var = inpt.var
    var_dict = inpt.var_dict
    var_min = inpt.extr[var]['min']
    var_max = inpt.extr[var]['max']
    ref_x = inpt.extr[var]['ref_x']

    axs[i].set_title(var_dict[comp]['label'])
    axs[i].set_xlabel(var_dict[ref_x]['label'])
    axs[i].set_ylabel(var_dict[comp]['label'])
    axs[i].set_xlim(var_min, var_max)
    axs[i].set_ylim(var_min, var_max)
    axs[i].text(0.01, 0.95, inpt.letters[i] + ')',
                transform=axs[i].transAxes)
    axs[i].plot([var_min, var_max], [var_min, var_max],
                color='black', lw=1.5, ls='-')


def format_ts(ax, year, yy, residuals=False):
    """
    Formats a time series subplot ax[yy] for a given year, setting x-axis limits, labels,
    and optionally y-axis limits for residuals.

    :param ax: Axes array/dict.
    :param year: Year to plot.
    :param yy: Index/key of subplot.
    :param residuals: If True, use residuals y-limits.
    """
    ax[yy].xaxis.set_major_formatter(inpt.myFmt)
    ax[yy].set_xlim(dt.datetime(year, 1, 1), dt.datetime(year, 12, 31))
    ax[yy].text(0.5, 0.90, str(year),
                transform=ax[yy].transAxes,
                horizontalalignment='center')
    ax[yy].text(0.01, 0.95, inpt.letters[yy] + ')',
                transform=ax[yy].transAxes)

    if residuals:
        ax[yy].set_ylim(inpt.extr[inpt.var]['res_min'],
                        inpt.extr[inpt.var]['res_max'])
    else:
        ax[yy].set_ylim(inpt.extr[inpt.var]['min'],
                        inpt.extr[inpt.var]['max'])


def frame_and_axis_removal(ax, len_comps):
    """
    Disables frame and axes for unused subplots based on the number of components.

    :param ax: Array-like of matplotlib Axes.
    :param len_comps: Number of active components.
    """
    # Define which axes to disable based on len_comps
    disable_indices = {
        1: [1, 2, 3],
        2: [2, 3],
        3: [3],
        4: []
    }.get(len_comps, [])

    for idx in disable_indices:
        ax[idx].axis('off')
        ax[idx].get_xaxis().set_visible(False)
        ax[idx].get_yaxis().set_visible(False)


# def plot_ba(period_label):
#     """
#
#     :param vr:
#     :param period_label:
#     :return:
#     """
#     print('BLAND-ALTMAN')
#     [vr_c, vr_e, vr_l, vr_t, vr_t1, vr_t2, vr_c_res, vr_e_res, vr_l_res, vr_t_res, vr_t1_res, vr_t2_res] = avar
#     seas_name = seass[period_label]['name']
#     fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=inpt.dpi)
#     ax = np.atleast_1d(ax)
#     axs = ax.ravel()
#
#     # define which is the reference measurement for each variable
#     if vr == 'lwp':
#         comps = ['c', 'e', 't', 't1']
#         x = vr_t1_res[inpt.var]
#         xlabel = 'HATPRO'
#     elif vr in ['windd', 'winds', 'precip']:
#         comps = ['c', 'e', 't', 't1']
#         x = vr_t2_res[inpt.var]
#         xlabel = 'AWS_ECAPAC'
#     elif vr == 'temp':
#         comps = ['c', 'e', 't2']
#         x = vr_t_res[inpt.var]
#         xlabel = 'THAAO'
#     else:
#         comps = ['c', 'e', 't1', 't2']
#         x = vr_t_res[inpt.var]
#         xlabel = 'THAAO'
#
#     for i, comp in enumerate(comps):
#         axs[i].set_xlabel(xlabel)
#         if comp == 'c':
#             label = 'CARRA'
#             axs[i].set_ylabel(label)
#             try:
#                 y = vr_c_res[inpt.var]
#             except KeyError:
#                 print(f"error with {label}")
#                 continue
#         if comp == 'e':
#             label = 'ERA5'
#             axs[i].set_ylabel(label)
#             try:
#                 y = vr_e_res[inpt.var]
#             except KeyError:
#                 print(f"error with {label}")
#                 continue
#         if comp == 't':
#             label = 'THAAO'
#             axs[i].set_ylabel(label)
#             try:
#                 y = vr_t_res[inpt.var]
#             except KeyError:
#                 print(f"error with {label}")
#                 continue
#         if comp == 't1':
#             label = 'HATPRO'
#             axs[i].set_ylabel(label)
#             try:
#                 y = vr_t1_res[inpt.var]
#             except KeyError:
#                 print(f"error with {label}")
#                 continue
#         if comp == 't2':
#             if vr == 'alb':
#                 label = 'ERA5 snow alb'
#             else:
#                 label = 'AWS ECAPAC'
#             axs[i].set_ylabel(label)
#             try:
#                 y = vr_t2_res[inpt.var]
#             except KeyError:
#                 print(f"error with {label}")
#                 continue
#         try:
#             print(f"plotting ba THAAO-{label}")
#
#             fig.suptitle(f"{vr.upper()} {seas_name} {tres}", fontweight='bold')
#             axs[i].set_title(label)
#             axs[i].text(0.01, 0.90, inpt.letters[i] + ')', transform=axs[i].transAxes)
#
#             time_list = pd.date_range(start=dt.datetime(2016, 1, 1), end=dt.datetime(2024, 12, 31), freq=tres)
#             if x.empty | y.empty:
#                 continue
#             x_all = x.reindex(time_list)
#             x_s = x_all.loc[(x_all.index.month.isin(seass[period_label]['months']))]
#             y_all = y.reindex(time_list)
#             y_s = y_all.loc[(y_all.index.month.isin(seass[period_label]['months']))]
#
#             idx = np.isfinite(x_s) & np.isfinite(y_s)
#
#             blandAltman(
#                     x_s[idx], y_s[idx], ax=axs[i], limitOfAgreement=1.96, confidenceInterval=95,
#                     confidenceIntervalMethod='approximate', detrend=None,
#                     percentage=False)  # confidenceIntervalMethod='exact paired' or 'approximate'  # detrend='Linear' or 'None'
#
#             # b, a = np.polyfit(x_s[idx], y_s[idx], deg=1)  # xseq = np.linspace(inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max'], num=1000)  # axs[i].plot(xseq, a + b * xseq, color='red', lw=2.5, ls='--')  # axs[i].plot(  #         [inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max']], [inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max']], color='black', lw=1.5,  #         ls='-')  # corcoef = ma.corrcoef(x_s[idx], y_s[idx])  #  # N = x_s[idx].shape[0]  # rmse = np.sqrt(np.nanmean((x_s[idx] - y_s[idx]) ** 2))  # mae = np.nanmean(np.abs(x_s[idx] - y_s[idx]))  # axs[i].text(  #         0.60, 0.15, f"R={corcoef[0, 1]:1.3}\nrmse={rmse:1.3}\nN={N}\nmae={mae:1.3}', fontsize=14,  #         transform=axs[i].transAxes)  # axs[i].set_xlim(extr[inpt.var]['min'], inpt.extr[inpt.var]['max'])  # axs[i].set_ylim(extr[inpt.var]['min'], inpt.extr[inpt.var]['max'])
#         except:
#             print(f"error with {label}")
#
#     plt.savefig(os.path.join(inpt.basefol['out']['base'], tres, f"{tres}_ba_{seas_name}_{vr}.png"), bbox_inches='tight')
#     plt.close('all')
