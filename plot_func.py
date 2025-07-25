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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import inputs as inpt
import tools as tls
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyCompare import blandAltman
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib import colors as mcolors
import csv


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

    var_data = inpt.extr[inpt.var]
    comps = var_data['comps']
    ref_x = var_data['ref_x']
    comps_all = comps + [ref_x]
    plot_vars = tls.plot_vars_cleanup(comps_all, var_data)

    str_name = f"{inpt.tres} {period_label} ts {inpt.var} {inpt.var_dict[ref_x]['label']} {inpt.years[0]}-{inpt.years[-1]}"
    fig.suptitle(str_name, fontweight='bold')
    fig.subplots_adjust(top=0.93)

    # Plotting style kwargs
    kwargs_ori = {'alpha': 0.02, 'lw': 0, 'marker': '.', 'ms': 1}
    kwargs_res = {'lw': 0, 'marker': 'o', 'ms': 2, 'markerfacecolor': 'none'}

    # Cache frequently used variables

    for i, year in enumerate(inpt.years):
        print(f"plotting {year}")

        # Boolean mask for original and resampled data for this year
        for data_typ in plot_vars:
            # Original data for the year
            if not isinstance(var_data[data_typ]['data'], str):
                null, chck = tls.check_empty_df(
                    var_data[data_typ]['data'][inpt.var], inpt.var)
            else:
                continue
            if chck:
                continue
            y_ori = var_data[data_typ]['data_res']['original'][inpt.var]
            y_ori_mask = y_ori.index.year == year
            if y_ori_mask.any():
                y_ori_ = y_ori.loc[y_ori_mask].dropna()
                ax[i].plot(y_ori_,
                           color=inpt.var_dict[data_typ]['col_ori'], **kwargs_ori)

            # Resampled data for the year
            y_res = var_data[data_typ]['data_res'][inpt.tres][inpt.var]
            y_res_mask = y_res.index.year == year
            if y_res_mask.any():
                y_res_ = y_res.loc[y_res_mask].dropna()
                ax[i].plot(y_res_, color=inpt.var_dict[data_typ]['col'],
                           label=inpt.var_dict[data_typ]['label'], **kwargs_res)

        # Format the subplot axes (assuming this function is defined elsewhere)
        format_ts(ax, year, i)

    plt.xlabel('Time')
    plt.legend(ncol=2)
    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


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

    var_data = inpt.extr[inpt.var]
    comps = var_data['comps']
    ref_x = var_data['ref_x']
    plot_vars = tls.plot_vars_cleanup(comps, var_data)

    str_name = f"{inpt.tres} {period_label} residuals_{inpt.var} {inpt.var_dict[ref_x]['label']} {inpt.years[0]}-{inpt.years[-1]}"
    fig.suptitle(str_name, fontweight='bold')
    fig.subplots_adjust(top=0.93)

    for i, year in enumerate(inpt.years):
        print(f"plotting {year}")

        # Plot horizontal zero line for residual reference
        daterange = pd.date_range(start=pd.Timestamp(
            year, 1, 1), end=pd.Timestamp(year, 12, 31))
        ax[i].plot(daterange, np.zeros(len(daterange)),
                   color='black', lw=2, ls='--')

        # Plot residuals (component - reference) for the year
        for data_typ in plot_vars:
            tres, tres_tol = tls.get_tres(data_typ)
            x = var_data[ref_x]['data_res'][tres][inpt.var]
            y = var_data[data_typ]['data_res'][tres][inpt.var]
            null, chck = tls.check_empty_df(x, inpt.var)
            if chck:
                return
            null, chck = tls.check_empty_df(y, inpt.var)
            if chck:
                continue
            x_mask = x.index.year == year
            y_mask = y.index.year == year
            if y_mask.any() and x_mask.any():
                residuals = y.loc[y_mask] - x.loc[x_mask]
                residuals = residuals.dropna()
                null, chck = tls.check_empty_df(residuals, inpt.var)
                if chck:
                    continue
                # ax[i].stem(residuals.index,
                #               residuals.values, color=inpt.var_dict[data_typ]['col'],
                #               label=inpt.var_dict[data_typ]['label'], marker='.')
                markerline, stemlines, baseline = ax[i].stem(residuals.index,
                                                             residuals.values, label=inpt.var_dict[data_typ]['label'])
                markerline.set_color(inpt.var_dict[data_typ]['col'])
                stemlines.set_color(inpt.var_dict[data_typ]['col'])
                baseline.set_visible(False)
                stemlines.set_linewidth(0.1)
                markerline.set_markersize(1)

        # Format axis
        format_ts(ax, year, i, residuals=True)

    plt.xlabel('Time')
    plt.legend(ncols=2)
    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


def plot_ba(period_label):
    """

    :param vr:
    :param period_label:
    :return:
    """
    print('BLAND-ALTMAN')
    plt.ioff()
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=inpt.dpi)
    axs = ax.ravel()
    # Cache frequently used variables
    var_data = inpt.extr[inpt.var]
    comps = var_data['comps']
    ref_x = var_data['ref_x']
    plot_vars = tls.plot_vars_cleanup(comps, var_data)

    str_name = f"{inpt.tres} {period_label} bland-altman {inpt.var} {inpt.var_dict[ref_x]['label']} {inpt.years[0]}-{inpt.years[-1]}"
    fig.suptitle(str_name, fontweight='bold')
    fig.subplots_adjust(top=0.93)

    frame_and_axis_removal(axs, len(comps))

    for i, data_typ in enumerate(plot_vars):
        tres, tres_tol = tls.get_tres(data_typ)
        x = var_data[ref_x]['data_res'][tres][inpt.var]
        time_range = pd.date_range(
            start=pd.Timestamp(inpt.years[0], 1, 1),
            end=pd.Timestamp(inpt.years[-1], 12, 31, 23, 59),
            freq=tres
        )
        x_all = x.reindex(time_range, method='nearest',
                          tolerance=pd.Timedelta(tres_tol)).astype(float)
        y_all = var_data[data_typ]['data_res'][tres][inpt.var].reindex(
            time_range).astype(float)
        valid_idx = ~(x_all.isna() | y_all.isna())
        x_valid, y_valid = x_all[valid_idx], y_all[valid_idx]

        perc = False
        if inpt.var == 'windd':
            return

        if inpt.var == 'precip':
            perc = True
            threshold = 1e-1
            mean = (x_valid + y_valid) / 2
            valid_mask = mean > threshold
            x_valid, y_valid = x_valid[valid_mask], y_valid[valid_mask]
            x_valid, y_valid = np.log1p(x_valid), np.log1p(y_valid)

        blandAltman(
            y_valid, x_valid, ax=axs[i], limitOfAgreement=1.96, confidenceInterval=95,
            confidenceIntervalMethod='approximate', detrend=None,
            percentage=perc, pointColour='blue')

        format_ba(axs, data_typ, i)

    if perc:
        save_path = os.path.join(
            inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}_perc.png")
    else:
        save_path = os.path.join(
            inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


def plot_scatter_all(period_label):
    """
    Plots a 2x2 grid of scatter plots or 2D histograms by season or full period.
    Applies polynomial fitting, formatting, and saves the figure.
    """

    print(f"SCATTERPLOTS {period_label}")
    plt.ioff()
    fig = plt.figure(figsize=(12, 12), dpi=inpt.dpi)
    var_data = inpt.extr[inpt.var]
    comps = var_data['comps']
    ref_x = var_data['ref_x']
    plot_vars = tls.plot_vars_cleanup(comps, var_data)

    str_name = f"{inpt.tres} {period_label} scatter {inpt.var} {inpt.var_dict[ref_x]['label']} {inpt.years[0]}-{inpt.years[-1]}"
    fig.suptitle(str_name, fontweight='bold')
    fig.subplots_adjust(top=0.93)

    # Layout constants for 2x2 main plots each with marginal histograms
    ncols = 2
    width_ratios = [4, 1]
    height_ratios = [1, 4]

    joint_axes = []

    for i, data_typ in enumerate(plot_vars):

        # Determine position in grid (row, col)
        row, col = divmod(i, ncols)

        # Compute left/right/bottom/top margins per subplot
        left = 0.05 + col * 0.475
        right = left + 0.4
        bottom = 0.05 + (1 - row) * 0.485
        top = bottom + 0.4

        gs = GridSpec(
            2, 2,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            left=left, right=right, bottom=bottom, top=top,
            wspace=0.05, hspace=0.05,
            figure=fig
        )

        ax_marg_x = fig.add_subplot(gs[0, 0])
        ax_joint = fig.add_subplot(gs[1, 0], sharex=ax_marg_x)
        joint_axes.append(ax_joint)
        ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)

        # Hide inner tick labels for marginal plots
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)

        tres, tres_tol = tls.get_tres(data_typ)

        var_data[ref_x]['data_marg_distr'] = {}
        var_data[ref_x]['data_marg_distr'][tres] = {}
        var_data[ref_x]['data_marg_distr'][tres][inpt.var] = {}
        var_data[data_typ]['data_marg_distr'] = {}
        var_data[data_typ]['data_marg_distr'][tres] = {}
        var_data[data_typ]['data_marg_distr'][tres][inpt.var] = {}

        var_data[ref_x]['data_marg_distr']['tres'] = tres
        var_data[ref_x]['data_marg_distr']['tres_tol'] = tres_tol
        var_data[data_typ]['data_marg_distr']['tres'] = tres
        var_data[data_typ]['data_marg_distr']['tres_tol'] = tres_tol

        x = var_data[ref_x]['data_res'][tres][inpt.var]
        time_range = pd.date_range(
            start=pd.Timestamp(inpt.years[0], 1, 1),
            end=pd.Timestamp(inpt.years[-1], 12, 31, 23, 59),
            freq=tres
        )
        x_all = x.reindex(time_range, method='nearest',
                          tolerance=pd.Timedelta(tres_tol)).astype(float)
        season_months = inpt.all_seasons['all']['months']
        x_season = x_all.loc[x_all.index.month.isin(season_months)]
        y = var_data[data_typ]['data_res'][tres][inpt.var].reindex(
            time_range).astype(float)
        y_season = y.loc[y.index.month.isin(season_months)]

        valid_idx = ~(x_season.isna() | y_season.isna())
        x_valid, y_valid = x_season[valid_idx], y_season[valid_idx]

        print(
            f"Plotting scatter {inpt.var_dict[ref_x]['label']} - {inpt.var_dict[data_typ]['label']}")

        vmin, vmax = var_data['min'], var_data['max']
        if vmin >= vmax or not (np.isfinite(vmin) and np.isfinite(vmax)):
            ax_joint.text(0.1, 0.5, "Invalid histogram range",
                          transform=ax_joint.transAxes)
        else:
            # Set up bin edges
            bin_edges = np.arange(
                vmin, vmax+var_data['bin_size'], var_data['bin_size'])
            # First draw to compute counts
            counts, _, _, _ = ax_joint.hist2d(
                x_valid, y_valid,
                bins=[bin_edges, bin_edges],
                cmin=1
            )
            pctl = 99
            filtered_counts = counts[counts > 0]
            if filtered_counts.size == 0:
                vmax_hist = 1  # or 0, or some safe default value
            else:
                vmax_hist = np.percentile(filtered_counts, pctl)
            has_overflow = np.any(counts > vmax_hist)
            extend_opt = 'max' if has_overflow else 'neither'

            # Clear and re-plot for actual display
            ax_joint.cla()
            h = ax_joint.hist2d(
                x_valid, y_valid,
                bins=[bin_edges, bin_edges],
                cmap=inpt.var_dict[data_typ]['cmap'],
                cmin=1,
                vmin=1,
                vmax=vmax_hist
            )
            quadmesh = h[-1]  # Correct QuadMesh for colorbar

            # Marginal histograms
            var_data[ref_x]['data_marg_distr'][tres][inpt.var], _, _ = ax_marg_x.hist(x_valid, bins=bin_edges,
                                                                                      color='orange', alpha=0.5, density=True)
            var_data[data_typ]['data_marg_distr'][tres][inpt.var], _, _ = ax_marg_y.hist(y_valid, bins=bin_edges,
                                                                                         orientation='horizontal', color='blue', alpha=0.5, density=True)

            # Sync the density axes (max of both histograms)
            max_density = max(np.max(var_data[ref_x]['data_marg_distr'][tres][inpt.var]), np.max(
                var_data[data_typ]['data_marg_distr'][tres][inpt.var]))

            ax_marg_x.set_ylim(0, max_density)
            ax_marg_x.set_xlim(ax_joint.get_xlim())
            ax_marg_x.yaxis.set_major_locator(
                MaxNLocator(nbins=3, prune='both'))
            ax_marg_x.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            ax_marg_y.set_xlim(0, max_density)
            ax_marg_y.set_ylim(ax_joint.get_ylim())
            ax_marg_y.xaxis.set_major_locator(
                MaxNLocator(nbins=3, prune='both'))
            ax_marg_y.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            # Colorbar (linked to QuadMesh, not x-axis)
            cax = inset_axes(ax_joint,
                             width="80%", height="25%", loc='lower center',
                             bbox_to_anchor=(0, -0.15, 1, 0.1),
                             bbox_transform=ax_joint.transAxes,
                             borderpad=0)

            cbar = fig.colorbar(
                quadmesh, cax=cax, orientation='horizontal', extend=extend_opt)
            cbar.set_label(
                f'Counts  max: {pctl}pctl', fontsize='small')
            cbar.ax.xaxis.set_major_formatter(
                FormatStrFormatter('%d'))  # Format counts as integers
            cbar.ax.xaxis.set_ticks_position('bottom')
            cbar.ax.xaxis.set_label_position('bottom')

            format_hist2d(
                ax_joint,
                xlabel=inpt.var_dict[ref_x]['label'],
                ylabel=inpt.var_dict[data_typ]['label'],
                letter=inpt.letters[i] + ')',
                xlim=(vmin, vmax),
                ylim=(vmin, vmax),
                binsize=var_data['bin_size']
            )

            # Fit
            if valid_idx.sum() >= 2:
                calc_draw_fit(joint_axes, i, x_valid, y_valid, inpt.tres,
                              inpt.all_seasons['all']['col'], data_typ, print_stats=True)
            else:
                calc_draw_fit(joint_axes, i, x_valid, y_valid, inpt.tres,
                              inpt.all_seasons['all']['col'], data_typ, print_stats=True)
                print("ERROR: Not enough data points for fit.")

    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def kl_divergence(P, Q, eps=1e-12):
    """
    Compute the KL divergence D(P || Q) for discrete probability distributions.

    Parameters:
        P (list or np.array): True distribution
        Q (list or np.array): Approximate distribution
        eps (float): Small constant to avoid log(0)

    Returns:
        float: KL divergence value
    """
    P = np.asarray(P, dtype=np.float64) + eps
    Q = np.asarray(Q, dtype=np.float64) + eps
    return np.sum(P * np.log(P / Q))


def plot_scatter_seasonal(period_label):
    print(f"SCATTERPLOTS {period_label}")
    plt.ioff()

    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=inpt.dpi)
    axs = ax.ravel()
    var_data = inpt.extr[inpt.var]
    comps = var_data['comps']
    ref_x = var_data['ref_x']
    plot_vars = tls.plot_vars_cleanup(comps, var_data)

    str_name = f"{inpt.tres} {period_label} scatter {inpt.var} {inpt.var_dict[ref_x]['label']} {inpt.years[0]}-{inpt.years[-1]}"
    fig.suptitle(str_name, fontweight='bold')
    fig.subplots_adjust(top=0.93)

    frame_and_axis_removal(axs, len(comps))

    for i, data_typ in enumerate(plot_vars):
        tres, tres_tol = tls.get_tres(data_typ)

        x = var_data[ref_x]['data_res'][tres][inpt.var]
        y = var_data[data_typ]['data_res'][tres][inpt.var]

        time_range = pd.date_range(
            start=pd.Timestamp(inpt.years[0], 1, 1),
            end=pd.Timestamp(inpt.years[-1], 12, 31, 23, 59),
            freq=tres
        )

        x_all = x.reindex(time_range, method='nearest',
                          tolerance=pd.Timedelta(tres_tol)).astype(float)
        y_all = y.reindex(time_range).astype(float)

        season_months = inpt.seasons[period_label]['months']
        x_season = x_all.loc[x_all.index.month.isin(season_months)]
        y_season = y_all.loc[y_all.index.month.isin(season_months)]

        valid_idx = ~(x_season.isna() | y_season.isna())
        x_valid, y_valid = x_season[valid_idx], y_season[valid_idx]

        print(
            f"Plotting scatter {inpt.var_dict[ref_x]['label']} - {inpt.var_dict[data_typ]['label']}")

        axs[i].scatter(
            x_valid, y_valid,
            s=5, facecolors='none',
            edgecolors=inpt.seasons[period_label]['col'],
            alpha=0.5
        )

        if valid_idx.sum() >= 2:
            calc_draw_fit(axs, i, x_valid, y_valid, inpt.tres,
                          inpt.seasons[period_label]['col'], data_typ, print_stats=True)
        else:
            calc_draw_fit(axs, i, x_valid, y_valid, inpt.tres,
                          inpt.seasons[period_label]['col'], data_typ, print_stats=False)
            print("ERROR: Not enough data points for fit.")

        format_scatterplot(axs, data_typ, i)

    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres,
        f"{str_name.replace(' ', '_')}.png"
    )
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_scatter_cum():
    """
    Plots cumulative scatter plots for each season (excluding 'all'), comparing components
    against reference data with fits, customized appearance, and saves the resulting figure.

    :raises ValueError: If insufficient data points are available for fitting.

    :return: None
    """
    print("SCATTERPLOTS cumulative")
    plt.ioff()
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=inpt.dpi)
    axs = ax.ravel()
    var_data = inpt.extr[inpt.var]
    comps = var_data['comps']
    ref_x = var_data['ref_x']
    plot_vars = tls.plot_vars_cleanup(comps, var_data)

    str_name = f"{inpt.tres} all CumSeas scatter {inpt.var} {inpt.var_dict[ref_x]['label']} {inpt.years[0]}-{inpt.years[-1]}"
    fig.suptitle(str_name, fontweight='bold')
    fig.subplots_adjust(top=0.93)

    frame_and_axis_removal(axs, len(comps))

    if inpt.datasets['dropsondes']['switch']:
        period_label = 'all'
        print(f"SCATTERPLOTS CUMULATIVE {period_label}")

        for i, data_typ in enumerate(plot_vars):
            tres, tres_tol = tls.get_tres(data_typ)
            x = var_data[ref_x]['data_res'][tres][inpt.var]
            # Prepare full time range for reindexing once
            time_range = pd.date_range(
                start=pd.Timestamp(inpt.years[0], 1, 1),
                end=pd.Timestamp(inpt.years[-1], 12, 31, 23, 59),
                freq=tres
            )
            x_all = x.reindex(time_range, method='nearest',
                              tolerance=pd.Timedelta(tres_tol)).astype(float)
            y = var_data[data_typ]['data_res'][inpt.tres][inpt.var]

            x_clean = x.dropna()
            y_clean = y.dropna()

            x_clean.index = pd.to_datetime(x_clean.index, errors='coerce')
            y_clean.index = pd.to_datetime(y_clean.index, errors='coerce')

            x_clean = x_clean[~x_clean.index.isna()]
            y_clean = y_clean[~y_clean.index.isna()]

            x_df = x_clean.sort_index().to_frame(name='x')
            y_df = y_clean.groupby(y_clean.index).mean(
            ).sort_index().to_frame(name='y')

            merged = pd.merge_asof(
                x_df, y_df,
                left_index=True,
                right_index=True,
                tolerance=tres_tol,
                direction='nearest'
            ).dropna()

            axs[i].scatter(
                merged['x'], merged['y'],
                s=5, color='blue', edgecolors='none', alpha=0.5, label=period_label
            )

            calc_draw_fit(axs, i,  merged['x'],  merged['y'], inpt.tres,
                          inpt.seasons[period_label]['col'], data_typ, print_stats=True)

            axs[i].legend()
            format_scatterplot(axs, data_typ, i)
    else:
        for period_label, season in inpt.seasons_subset.items():
            print(f"SCATTERPLOTS CUMULATIVE {period_label}")
            for i, data_typ in enumerate(plot_vars):
                tres, tres_tol = tls.get_tres(data_typ)
                x = var_data[ref_x]['data_res'][tres][inpt.var]
                time_range = pd.date_range(
                    start=pd.Timestamp(inpt.years[0], 1, 1),
                    end=pd.Timestamp(inpt.years[-1], 12, 31, 23, 59),
                    freq=tres
                )
                x_all = x.reindex(time_range, method='nearest',
                                  tolerance=pd.Timedelta(tres_tol)).astype(float)
                season_months = inpt.seasons[period_label]['months']
                x_season = x_all.loc[x_all.index.month.isin(season_months)]
                y = var_data[data_typ]['data_res'][tres][inpt.var].reindex(
                    time_range).astype(float)
                y_season = y.loc[y.index.month.isin(season_months)]

                valid_idx = ~(x_season.isna() | y_season.isna())
                x_valid, y_valid = x_season[valid_idx], y_season[valid_idx]

                axs[i].scatter(
                    x_valid, y_valid,
                    s=5, color=season['col'], edgecolors='none', alpha=0.5, label=period_label
                )

                if valid_idx.sum() < 2:
                    print(
                        'ERROR: Not enough data points for proper fit (need at least 2).')
                    # Optionally raise ValueError here if needed
                    # raise ValueError("Insufficient data for fitting.")
                else:
                    calc_draw_fit(axs, i, x_valid, y_valid, inpt.tres,
                                  inpt.seasons[period_label]['col'], data_typ, print_stats=False)

                    axs[i].legend()
                format_scatterplot(axs, data_typ, i)

    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


# def get_shaded_color(base_color, resolution):
#     """
#     Returns a shaded version of the base color depending on resolution.
#     Higher temporal resolution (smaller steps like 1h) → darker.
#     Coarser resolution (like 6h, 12h) → lighter.
#     'original' is darkest to highlight it.
#     """
#     cmap = cm.get_cmap('Blues') if base_color == 'blue' else cm.get_cmap('Reds')

#     def res_to_norm(res):
#         if res == 'original':
#             return 0.0  # darkest
#         try:
#             res_hour = int(res.replace('h', ''))
#             # 1h → 0.2 (dark), 12h → 0.95 (light)
#             norm_val = max(0.2, min(0.95, 0.2 + (res_hour - 1) / 15))
#             return norm_val
#         except:
#             return 0.5  # fallback gray

#     norm = res_to_norm(resolution)
#     shaded = cmap(norm)
#     return shaded[:3]


def get_color_by_resolution(base_color, resolution):
    blues = ['navy', 'mediumblue', 'royalblue', 'dodgerblue', 'lightblue']
    reds = ['darkred', 'firebrick', 'crimson', 'red', 'salmon']
    res_order = ['original', '1h', '2h', '3h', '6h']

    try:
        idx = res_order.index(resolution)
    except ValueError:
        idx = -1

    base_color = base_color.lower() if isinstance(base_color, str) else 'gray'

    if base_color == 'blue':
        return blues[idx] if idx >= 0 else 'lightblue'
    elif base_color == 'red':
        return reds[idx] if idx >= 0 else 'salmon'
    else:
        return 'gray'


def plot_taylor(var_list):
    if var_list[0] in inpt.met_vars:
        plot_name = 'Weather variables'
        var_list = inpt.met_vars
        available_markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    elif var_list[0] in inpt.rad_vars:
        plot_name = 'Radiation variables'
        available_markers = ['X', 'H', '>', '<', '8', 'd']
    else:
        raise ValueError("Unknown variable category.")

    print(f"Taylor Diagram {plot_name}")
    str_name = f"Taylor Diagram {plot_name} {inpt.years[0]}-{inpt.years[-1]}"

    combined_stdrefs = []
    combined_stds = []
    combined_cors = []
    combined_labels = []
    combined_colors = []
    combined_markers = []
    var_marker_map = {}

    for var_idx, var in enumerate(var_list):
        print(var)
        marker = available_markers[var_idx % len(available_markers)]
        var_marker_map[var] = marker
        inpt.var = var
        var_data = inpt.extr[var]
        comps = ['c', 'e']
        ref_x = var_data['ref_x']
        plot_vars = tls.plot_vars_cleanup(comps, var_data)

        for tres in inpt.tres_list:
            for data_typ in plot_vars:
                std_y = var_data[data_typ]['data_stats'][tres]['std_y']
                std_x = var_data[ref_x]['data_stats'][tres]['std_x']
                r2 = var_data[data_typ]['data_stats'][tres]['r2']
                combined_stdrefs.append(1)
                combined_stds.append(std_y / std_x)
                combined_cors.append(np.sqrt(r2))
                combined_labels.append(f"{data_typ} ({var}, {tres})")

                if data_typ == 'c':
                    color = 'red'
                elif data_typ == 'e':
                    color = 'blue'
                else:
                    color = inpt.var_dict.get(
                        data_typ, {}).get('col', 'purple')

                combined_colors.append(color)
                combined_markers.append(var_marker_map[var])

    fig = plt.figure(figsize=(12, 10), dpi=inpt.dpi)
    ax = fig.add_subplot(111, polar=True)
    fig.suptitle(str_name, fontweight='bold')
    fig.subplots_adjust(top=0.93, bottom=0.15)

    std_ref = 1.0
    plot_taylor_dia(ax, std_ref, combined_stds, combined_cors, combined_labels,
                    colors=combined_colors, markers=combined_markers,
                    var_marker_map=var_marker_map, inpt=inpt)

    save_path = os.path.join(
        inpt.basefol['out']['base'], f"{str_name.replace(' ', '_')}.png")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_taylor_dia(ax, std_ref, std_models, corr_coeffs, model_labels,
                    ref_label='REF', colors=None, markers=None,
                    var_marker_map=None, inpt=None):

    std_models = np.array(std_models)
    corr_coeffs = np.array(corr_coeffs)
    rmax = 2

    ax.set_ylim(0, rmax)
    ax.set_theta_direction(1)
    ax.set_theta_zero_location('E')
    ax.set_thetamin(0)
    ax.set_thetamax(90)

    corr_values = [1.0, 0.99, 0.95, 0.9, 0.8, 0.7,
                   0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    theta_degrees = np.degrees(np.arccos(corr_values))
    ticks, label_texts = ax.set_thetagrids(
        theta_degrees, labels=[f"{c:.2f}" for c in corr_values])
    for label in label_texts:
        label.set_color('darkgoldenrod')

    ax.set_rlabel_position(135)
    radial_ticks = np.arange(0, rmax + 0.2, 0.2)
    radial_labels = [ref_label if r ==
                     1.0 else f"{r:.2f}" for r in radial_ticks]
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels(radial_labels, fontsize=10, color='black')

    ax.yaxis.grid(True, color='darksalmon',
                  linestyle='-', linewidth=1., alpha=0.3)

    ax.text(-0.10, 1.0, "Normalized Standard Deviations",
            ha='center', va='top', fontsize='medium')
    ax.text(np.radians(45), rmax + 0.15, "Correlation (R)",
            rotation=-45, ha='center', va='center', fontsize='medium')

    for theta in np.radians(theta_degrees):
        ax.plot([theta, theta], [0, rmax], color='darkgoldenrod',
                linestyle='--', linewidth=0.8, alpha=0.5)

    for rtick in np.arange(0.2, rmax + 0.2, 0.2):
        angles = np.linspace(-np.pi, np.pi, 300)
        x_arc = std_ref + rtick * np.cos(angles)
        y_arc = rtick * np.sin(angles)
        ax.plot(np.arctan2(y_arc, x_arc), np.sqrt(x_arc**2 + y_arc**2),
                color='darkgreen', linestyle='--', linewidth=0.7, alpha=0.6)

    ax.add_artist(plt.Circle((0, 0), std_ref, transform=ax.transData._b,
                             color='black', fill=False, linestyle='--', linewidth=3))

    point_map = {}

    def parse_res(res):
        return 0 if res == 'original' else int(res.strip('h'))

    for i, (std, corr, label) in enumerate(zip(std_models, corr_coeffs, model_labels)):
        theta = np.arccos(corr)
        try:
            data_typ, meta = label.split('(')
            var_name, resolution = [x.strip(' )') for x in meta.split(',')]
            data_typ = data_typ.strip()
        except Exception:
            var_name = label
            data_typ = 'unknown'
            resolution = 'original'

        key = (var_name, data_typ)
        color_base = colors[i]
        marker = markers[i]
        color = get_color_by_resolution(color_base, resolution)

        if key not in point_map:
            point_map[key] = {'original': None, 'others': []}

        res_hour = parse_res(resolution)
        if resolution == 'original':
            ax.plot(theta, std, marker='o', color='black', markersize=10,
                    linestyle='None', markerfacecolor='none')
            ax.plot(theta, std, marker=marker, markerfacecolor=color,
                    linestyle='None', markersize=6, markeredgecolor='none')
            point_map[key]['original'] = (theta, std, res_hour)
        else:
            ax.plot(theta, std, marker=marker, markerfacecolor=color,
                    linestyle='None', markeredgecolor='none')
            point_map[key]['others'].append((theta, std, res_hour))

    for key, pts in point_map.items():
        all_pts = []
        if pts['original']:
            all_pts.append(pts['original'])
        all_pts.extend(pts['others'])
        # sorted_pts = sorted(all_pts, key=lambda x: x[2])
        # for i in range(len(sorted_pts) - 1):
        #     ax.annotate("",
        #                 xy=(sorted_pts[i+1][0], sorted_pts[i+1][1]),
        #                 xytext=(sorted_pts[i][0], sorted_pts[i][1]),
        #                 arrowprops=dict(arrowstyle='->', color='gray', lw=1),
        #                 annotation_clip=False)

    # Create first legend handles (variables)
    legend_elements = [
        Line2D([], [], color='black', marker=mark,
               linestyle='None', label=var)
        for var, mark in var_marker_map.items()
    ]

    # Create second legend handles (models)
    model_keys = ['c', 'e']
    model_legend = [
        Line2D([], [], color=inpt.var_dict[k]['col'], marker='o',
               linestyle='None', label=inpt.var_dict[k]['label'])
        for k in model_keys if k in inpt.var_dict
    ]
    model_legend.append(Line2D([], [], color='black', marker='o',
                               linestyle='None', markerfacecolor='none',
                               markersize=10, label='Original resolution'))

    # Combine handles and labels
    all_handles = legend_elements + model_legend
    all_labels = [h.get_label() for h in all_handles]

    ax.legend(all_handles, all_labels, loc='upper right',
              fontsize='small', title_fontsize='medium', title='Legend')


def calc_draw_fit(axs, i, xxx, yyy, tr, col, data_typ, print_stats=True):
    """
    Performs linear regression on data (xxx, yyy), plots the fit line and 1:1 line,
    and optionally annotates stats (R, N, MBE, RMSE) on subplot axs[i].

    :param axs: Matplotlib axes array.
    :param i: Index of subplot to plot on.
    :param xxx: x-data (pd.Series or np.array).
    :param yyy: y-data (pd.Series or np.array).
    :param per_lab: Key for color in inpt.seasons.
    :param print_stats: Whether to display stats text.
    """
    var = inpt.var
    var_dict = inpt.var_dict
    ref_x = inpt.extr[var]['ref_x']

    xx = xxx.values.flatten()
    yy = yyy.values.flatten()
    # Make sure xx and yy are numpy arrays
    xx_all = np.asarray(xx)
    yy_all = np.asarray(yy)

    # Mask out non-finite values
    mask = np.isfinite(xx_all) & np.isfinite(yy_all)
    xx = xx_all[mask]
    yy = yy_all[mask]
    b, a = np.polyfit(xx, yy, 1)  # slope, intercept
    var_min = inpt.extr[inpt.var]['min']
    var_max = inpt.extr[inpt.var]['max']
    xseq = np.linspace(var_min, var_max, 1000)

    axs[i].plot(xseq, a + b * xseq,
                color=col,
                lw=2.5, ls='--', alpha=0.5)
    axs[i].plot([var_min, var_max], [var_min, var_max],
                color='black', lw=1.5, ls='-')

    fn = f"{data_typ}_stats_{inpt.var}.csv"
    if print_stats:
        if os.exists(fn):
            pass
            # TODO read_csv
        else:
            r2, N, rmse, mbe, std_x, std_y, KL_bits = calc_stats(
                xx, yy, data_typ, tr, fn)

        def escape_label(label):
            return label.replace('_', r'\_')

        stats_text = (
            f"R² = {r2:.2f}\n"
            f"N = {N}\n"
            f"y = {b:+.2f}x {a:+.2f}\n"
            f"MBE = {mbe:.2f}\n"
            f"RMSE = {rmse:.2f}\n"
            f"$\\sigma_{{{escape_label(var_dict[ref_x]['label'])}}} = {std_x:.2f}$\n"
            f"$\\sigma_{{{escape_label(var_dict[data_typ]['label'])}}} = {std_y:.2f}$\n"
            f"$KL = {KL_bits:.3f}$ bits"
        )

        axs[i].text(0.57, 0.20, stats_text,
                    transform=axs[i].transAxes,
                    fontsize=10, color='black',
                    ha='left', va='center',
                    bbox=dict(facecolor='white', edgecolor='white'))


def calc_stats(x, y, data_typ, tr, fn):
    ref_x = inpt.extr[inpt.var]['ref_x']
    corcoeff = np.corrcoef(x, y)[0, 1]
    diff = y-x
    if 'data_stats' not in inpt.extr[inpt.var][data_typ]:
        inpt.extr[inpt.var][data_typ]['data_stats'] = {}
    if 'data_stats' not in inpt.extr[inpt.var][ref_x]:
        inpt.extr[inpt.var][ref_x]['data_stats'] = {}

    var_data = inpt.extr[inpt.var]
    var_data[data_typ]['data_stats'][tr] = {}
    var_data[ref_x]['data_stats'][tr] = {}
    var_data[data_typ]['data_stats'][tr]['r2'] = corcoeff*corcoeff

    var_data[data_typ]['data_stats'][tr]['N'] = len(y)

    var_data[data_typ]['data_stats'][tr]['rmse'] = np.sqrt(
        np.nanmean(diff ** 2))
    var_data[data_typ]['data_stats'][tr]['mbe'] = np.nanmean(diff)
    var_data[ref_x]['data_stats'][tr]['std_x'] = np.std(x)
    var_data[data_typ]['data_stats'][tr]['std_y'] = np.std(y)

    tres_ref_x = var_data[ref_x]['data_marg_distr']['tres']
    tres = var_data[data_typ]['data_marg_distr']['tres']
    P = np.array(var_data[ref_x]['data_marg_distr']
                 [tres_ref_x][inpt.var]*var_data['bin_size'])
    Q = np.array(var_data[data_typ]['data_marg_distr']
                 [tres][inpt.var]*var_data['bin_size'])

    # Compute KL divergences
    var_data[data_typ]['data_stats'][tr]['kl_bits'] = kl_divergence(
        P, Q)/np.log(2)
    save_stats(fn, data_typ, ref_x)

    return (inpt.extr[inpt.var][data_typ]['data_stats'][tr]['r2'], inpt.extr[inpt.var][data_typ]['data_stats'][tr]['N'], inpt.extr[inpt.var][data_typ]['data_stats'][tr]['rmse'], inpt.extr[inpt.var][data_typ]['data_stats'][tr]['mbe'], inpt.extr[inpt.var][ref_x]['data_stats'][tr]['std_x'],  inpt.extr[inpt.var][data_typ]['data_stats'][tr]['std_y'], inpt.extr[inpt.var][data_typ]['data_stats'][tr]['kl_bits'])


def save_stats(fn, data_typ, ref_x):
    out_dir = os.path.join(inpt.basefol['out']['base'], 'scatter_stats')
    os.makedirs(out_dir, exist_ok=True)
    stats_file = os.path.join(out_dir, f"{fn}")
    write_header = not os.path.exists(stats_file)

    with open(stats_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if write_header:
            writer.writerow(['Variable', 'Data_Type', 'Resolution',
                            'R2', 'N', 'RMSE', 'MBE', 'STD_X', 'STD_Y', 'KL_BITS'])

        for tr, stats in inpt.extr[inpt.var][data_typ]['data_stats'].items():
            try:
                ref_stats = inpt.extr[inpt.var][ref_x]['data_stats'][tr]

                writer.writerow([
                    inpt.var,
                    data_typ,
                    tr,
                    stats['r2'],
                    stats['N'],
                    stats['rmse'],
                    stats['mbe'],
                    ref_stats['std_x'],
                    stats['std_y'],
                    stats['kl_bits']
                ])
            except KeyError as e:
                print(f"⚠️ Missing data for {data_typ}, {tr}: {e}")


def format_ax(ax, xlabel='', ylabel='', title=None, letter=None,
              xlim=None, ylim=None, identity_line=False,
              fontweight='bold', fontsize='medium', binsize=None):
    """
    Generic axis formatting helper.

    :param ax: Matplotlib Axes object.
    :param xlabel: Label for x-axis.
    :param ylabel: Label for y-axis.
    :param title: Optional title.
    :param letter: Optional subplot label (e.g. 'a)').
    :param xlim: Tuple for x-axis limits.
    :param ylim: Tuple for y-axis limits.
    :param identity_line: If True, draw a y=x identity line.
    :param fontweight: Font weight for annotation letter.
    :param fontsize: Font size for annotation letter.
    """
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if letter:
        ax.text(0.01, 0.95, letter, transform=ax.transAxes,
                fontsize=fontsize, fontweight=fontweight,
                verticalalignment='top')

    if identity_line and xlim and ylim:
        min_val = min(xlim[0], ylim[0])
        max_val = max(xlim[1], ylim[1])
        ax.plot([min_val, max_val], [min_val, max_val],
                color='black', lw=1.5, ls='-')


def format_ts(ax, year, yy, residuals=False):
    ax[yy].xaxis.set_major_formatter(inpt.myFmt)
    ax[yy].set_xlim(dt.datetime(year, 1, 1), dt.datetime(year, 12, 31))
    ylim = (inpt.extr[inpt.var]['res_min'], inpt.extr[inpt.var]['res_max']) if residuals \
        else (inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max'])

    format_ax(ax[yy],
              ylim=ylim,
              letter=f"{inpt.letters[yy]})    {year}")


def format_ba(axs, data_typ, i):
    var = inpt.var
    var_dict = inpt.var_dict
    ref_x = inpt.extr[var]['ref_x']

    format_ax(axs[i],
              xlabel=f"mean({var_dict[ref_x]['label']}, {var_dict[data_typ]['label']})",
              ylabel=f"{var_dict[data_typ]['label']} - {var_dict[ref_x]['label']}",
              title=var_dict[data_typ]['label'],
              letter=inpt.letters[i] + ')')


def format_hist2d(ax, xlabel, ylabel, letter, xlim=None, ylim=None, binsize=None):
    format_ax(ax,
              xlabel=xlabel,
              ylabel=ylabel,
              xlim=xlim,
              ylim=ylim,
              letter=f"{letter}    bin size={binsize:.2f}")


def format_scatterplot(axs, data_typ, i):
    var = inpt.var
    var_dict = inpt.var_dict
    vmin = inpt.extr[var]['min']
    vmax = inpt.extr[var]['max']
    ref_x = inpt.extr[var]['ref_x']

    format_ax(axs[i],
              xlabel=var_dict[ref_x]['label'],
              ylabel=var_dict[data_typ]['label'],
              title=var_dict[data_typ]['label'],
              xlim=(vmin, vmax), ylim=(vmin, vmax),
              letter=inpt.letters[i] + ')',
              identity_line=True)


def frame_and_axis_removal(axs, n_plots):
    """
    Hide extra axes beyond the number of plots needed.

    :param axs: Flattened array of matplotlib axes.
    :param n_plots: Number of actual plots to show.
    """
    for i in range(n_plots, len(axs)):
        axs[i].set_visible(False)
