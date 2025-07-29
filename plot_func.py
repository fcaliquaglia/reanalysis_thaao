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

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import inputs as inpt
import plot_tools as plt_tls
import tools as tls
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyCompare import blandAltman
from matplotlib.lines import Line2D


def plot_ts(period_label):
    """
    Generate a time series plot for each year, overlaying original and resampled datasets.

    :param period_label: Descriptive label for the period being analyzed.
    :type period_label: str
    """
    print(f"[INFO] Plotting time series for period: {period_label}")
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
        print(f"===={year}")

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
            y_ori_ = var_data[data_typ]['data_res']['original'][inpt.var]
            y_ori_mask = y_ori_.index.year == year
            if y_ori_mask.any():
                y_ori = y_ori_.loc[y_ori_mask].dropna()
                ax[i].plot(y_ori,
                           color=inpt.var_dict[data_typ]['col_ori'], **kwargs_ori)

            # Resampled data for the year
            y_res = var_data[data_typ]['data_res'][inpt.tres][inpt.var]
            y_res_mask = y_res.index.year == year
            if y_res_mask.any():
                y_res_ = y_res.loc[y_res_mask].dropna()
                ax[i].plot(y_res_, color=inpt.var_dict[data_typ]['col'],
                           label=inpt.var_dict[data_typ]['label'], **kwargs_res)

        # Format the subplot axes (assuming this function is defined elsewhere)
        plt_tls.format_ts(ax, year, i)

    plt.xlabel('Time')
    plt.legend(ncol=2)
    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


def plot_residuals(period_label):
    """
    Plot the residuals (differences) between model/component outputs and reference data
    over the specified time period.

    :param period_label: Label describing the period for which residuals are plotted.
    :type period_label: str
    """
    print(f"[INFO] Plotting residuals for period: {period_label}")
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
        print(f"===={year}")

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
        plt_tls.format_ts(ax, year, i, residuals=True)

    plt.xlabel('Time')
    plt.legend(ncols=2)
    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


def plot_ba(period_label):
    """
    Generate Bland-Altman plots to assess agreement between component datasets
    and reference data over the specified time period.

    :param period_label: Label describing the period being evaluated.
    :type period_label: str
    """
    print(f"[INFO] Generating Bland-Altman plots for period: {period_label}")
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

    plt_tls.frame_and_axis_removal(axs, len(comps))

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

        plt_tls.format_ba(axs, data_typ, i)

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
    Plot a 2x2 grid of scatter plots or 2D histograms by season or for the full period.
    Applies polynomial fitting to the data, formats the plots, and saves the figure.

    :param period_label: Label indicating the time period or season for the plots.
    :type period_label: str
    """
    print(f"[INFO] Generating scatter plots for period: {period_label}")
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
    marg_x_axes = []
    marg_y_axes = []

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
        ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)

        joint_axes.append(ax_joint)
        marg_x_axes.append(ax_marg_x)
        marg_y_axes.append(ax_marg_y)

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
            f"===={inpt.var_dict[ref_x]['label']} - {inpt.var_dict[data_typ]['label']}")

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
            var_data[ref_x]['data_marg_distr'][tres][inpt.var], _, _ = \
                ax_marg_x.hist(x_valid, bins=bin_edges,
                               color=inpt.var_dict[data_typ]['col_distr'], 
                               alpha=0.5, density=True)
            var_data[data_typ]['data_marg_distr'][tres][inpt.var], _, _ = \
                ax_marg_y.hist(y_valid, bins=bin_edges, orientation='horizontal', 
                               color=inpt.var_dict[data_typ]['col_distr'], 
                               alpha=0.5, density=True)
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

            plt_tls.format_hist2d(
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
                plt_tls.calc_draw_fit(joint_axes, i, x_valid, y_valid, inpt.tres,
                                      inpt.all_seasons['all']['col'], data_typ, print_stats=True)

    # same axis for marginal distroibutions
    all_data_typs = var_data['comps'] + [var_data['ref_x']]

    global_max_density = 0

    for data_typ in all_data_typs:
        tres, tres_tol = tls.get_tres(data_typ)
        try:
            data_distr = var_data[data_typ]['data_marg_distr'][tres][inpt.var]
            max_val = np.max(data_distr)
            if max_val > global_max_density:
                global_max_density = max_val
        except KeyError:
            continue

    for ax_marg_x, ax_joint in zip(marg_x_axes, joint_axes):
        ax_marg_x.set_ylim(0, global_max_density)
        ax_marg_x.set_xlim(ax_joint.get_xlim())
        ax_marg_x.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
        ax_marg_x.yaxis.set_major_formatter(
            FuncFormatter(plt_tls.smart_formatter))

    for ax_marg_y, ax_joint in zip(marg_y_axes, joint_axes):
        ax_marg_y.set_xlim(0, global_max_density)
        ax_marg_y.set_ylim(ax_joint.get_ylim())
        ax_marg_y.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
        ax_marg_y.xaxis.set_major_formatter(
            FuncFormatter(plt_tls.smart_formatter))

    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_scatter_seasonal(period_label):
    """
    Plot scatter plots separated by season.
    Applies polynomial fitting to seasonal data, formats the plots, and saves the figure.

    :param period_label: Label indicating the time period or season for the plots.
    :type period_label: str
    """
    print(
        f"[INFO] Generating seasonal scatter plots for period: {period_label}")
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

    plt_tls.frame_and_axis_removal(axs, len(comps))

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
            f"===={inpt.var_dict[ref_x]['label']} - {inpt.var_dict[data_typ]['label']}")

        axs[i].scatter(
            x_valid, y_valid,
            s=5, facecolors='none',
            edgecolors=inpt.seasons[period_label]['col'],
            alpha=0.5
        )

        if valid_idx.sum() >= 2:
            plt_tls.calc_draw_fit(axs, i, x_valid, y_valid, inpt.tres,
                                  inpt.seasons[period_label]['col'], data_typ, print_stats=True)
        else:
            plt_tls.calc_draw_fit(axs, i, x_valid, y_valid, inpt.tres,
                                  inpt.seasons[period_label]['col'], data_typ, print_stats=False)
            print("ERROR: Not enough data points for fit.")

        plt_tls.format_scatterplot(axs, data_typ, i)

    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres,
        f"{str_name.replace(' ', '_')}.png"
    )
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_scatter_cum():
    """
    Plot cumulative scatter plots for each season (excluding 'all'), comparing components
    against reference data with polynomial fits, customized appearance, and saving the figure.

    :raises ValueError: If insufficient data points are available for fitting.

    :return: None
    """
    print("[INFO] Generating cumulative scatter plots")
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

    plt_tls.frame_and_axis_removal(axs, len(comps))

    if inpt.datasets['dropsondes']['switch']:
        period_label = 'all'
        print(f"===={period_label}")

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

            plt_tls.calc_draw_fit(axs, i,  merged['x'],  merged['y'], inpt.tres,
                                  inpt.seasons[period_label]['col'], data_typ, print_stats=True)

            axs[i].legend()
            plt_tls.format_scatterplot(axs, data_typ, i)
    else:
        for period_label, season in inpt.seasons_subset.items():
            print(f"===={period_label}")
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
                    plt_tls.calc_draw_fit(axs, i, x_valid, y_valid, inpt.tres,
                                          inpt.seasons[period_label]['col'], data_typ, print_stats=False)

                    axs[i].legend()
                plt_tls.format_scatterplot(axs, data_typ, i)

    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


def plot_taylor(var_list):
    """
    Generate a Taylor diagram for a list of variables to assess statistical agreement
    (correlation, standard deviation) between models/components and reference data.

    :param var_list: List of variable names to include in the Taylor diagram.
    :type var_list: list
    """

    if var_list[0] in inpt.met_vars:
        plot_name = 'Weather variables'
        inpt.met_vars.remove('surf_pres')
        available_markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    if var_list[0] in inpt.rad_vars:
        plot_name = 'Radiation variables'
        available_markers = ['X', 'H', '>', '<', '8', 'd', 's']
    if var_list[0] in inpt.cloud_vars:
        plot_name = 'Cloud variables'
        available_markers = ['X', 'H', '>', '<', '8', 'd', 's']
    print(f"[INFO] Taylor Diagram for {plot_name}")
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

    ref_std = 1.0
    plot_taylor_dia(ax, ref_std, combined_stds, combined_cors, combined_labels,
                    colors=combined_colors, markers=combined_markers,
                    var_marker_map=var_marker_map)

    save_path = os.path.join(
        inpt.basefol['out']['base'], f"{str_name.replace(' ', '_')}.png")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_taylor_dia(ax, std_ref, std_models, corrs, labels,
                    colors, markers, var_marker_map):
    """
    Draw a Taylor diagram on a polar axis.

    :param ax: Matplotlib axis object to plot on.
    :param std_ref: Reference standard deviation.
    :param std_models: List of standard deviations of the models.
    :param corr_coeffs: List of correlation coefficients.
    :param model_labels: List of labels for the model points.
    :param ref_label: Label for the reference circle.
    :param colors: List of colors for each model point.
    :param markers: List of marker styles for each model point.
    :param var_marker_map: Dictionary mapping variable names to markers.
    :param inpt: Input configuration module (used for labels and colors).
    """

    std_models = np.array(std_models)
    corrs = np.array(corrs)
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
    radial_labels = ['REF' if r ==
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

    for i, (std, corr, label) in enumerate(zip(std_models, corrs, labels)):
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
        color = plt_tls.get_color_by_resolution(color_base, resolution)

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
