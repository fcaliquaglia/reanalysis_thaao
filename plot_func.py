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
    kwargs_res = {'lw': 0, 'marker': 'o', 'ms': 2, 'markerfacecolor': 'none'}

    # Cache frequently used variables
    var_data = inpt.extr[inpt.var]
    comps = var_data['comps']
    ref_x = var_data['ref_x']
    comps_all = comps + [ref_x]
    plot_vars = tls.plot_vars_cleanup(comps_all, var_data)

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
    str_name = f"{inpt.tres} {period_label} residuals_{inpt.var} {inpt.var_dict['t']['label']} {inpt.years[0]}-{inpt.years[-1]}"
    fig.suptitle(str_name, fontweight='bold')
    fig.subplots_adjust(top=0.93)

    var_data = inpt.extr[inpt.var]
    comps = var_data['comps']
    ref_x = var_data['ref_x']
    plot_vars = tls.plot_vars_cleanup(comps, var_data)

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
    str_name = f"{inpt.tres} {period_label} bland-altman {inpt.var} {inpt.var_dict['t']['label']} {inpt.years[0]}-{inpt.years[-1]}"
    fig.suptitle(str_name, fontweight='bold')
    fig.subplots_adjust(top=0.93)

    # Cache frequently used variables
    var_data = inpt.extr[inpt.var]
    comps = var_data['comps']
    ref_x = var_data['ref_x']
    plot_vars = tls.plot_vars_cleanup(comps, var_data)

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
    str_name = f"{inpt.tres} {period_label} scatter {inpt.var} {inpt.var_dict['t']['label']} {inpt.years[0]}-{inpt.years[-1]}"
    fig.suptitle(str_name, fontweight='bold')

    var_data = inpt.extr[inpt.var]
    comps = var_data['comps']
    ref_x = var_data['ref_x']
    plot_vars = tls.plot_vars_cleanup(comps, var_data)

    # Layout constants for 2x2 main plots each with marginal histograms
    nrows, ncols = 2, 2
    width_ratios = [4, 1]
    height_ratios = [1, 4]

    joint_axes = []
    for i, data_typ in enumerate(plot_vars):
        # Determine position in grid (row, col)
        row, col = divmod(i, ncols)

        # Compute left/right/bottom/top margins per subplot
        left = 0.05 + col * 0.475
        right = left + 0.4
        bottom = 0.05 + (1 - row) * 0.475
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
            f"Plotting scatter {inpt.var_dict['t']['label']} - {inpt.var_dict[data_typ]['label']}")

        vmin, vmax = var_data['min'], var_data['max']
        if vmin >= vmax or not (np.isfinite(vmin) and np.isfinite(vmax)):
            ax_joint.text(0.1, 0.5, "Invalid histogram range",
                          transform=ax_joint.transAxes)
        else:
            # Set up bin edges
            bin_edges = np.linspace(vmin, vmax, var_data['bin_nr'])
            bin_size = (vmax - vmin) / var_data['bin_nr']

            # First draw to compute counts
            counts, _, _, _ = ax_joint.hist2d(
                x_valid, y_valid,
                bins=[bin_edges, bin_edges],
                cmin=1
            )
            pctl = 99
            vmax_hist = np.percentile(counts[counts > 0], pctl)
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

            # Fit
            if valid_idx.sum() >= 2:
                calc_draw_fit(joint_axes, i, x_valid, y_valid, tres,
                              inpt.all_seasons['all']['col'], data_typ, print_stats=True)
            else:
                calc_draw_fit(joint_axes, i, x_valid, y_valid, tres,
                              inpt.all_seasons['all']['col'], data_typ, print_stats=False)
                print("ERROR: Not enough data points for fit.")

            # Marginal histograms
            ax_marg_x.hist(x_valid, bins=bin_edges,
                           color='orange', alpha=0.5, density=True)
            ax_marg_x.set_xlim(ax_joint.get_xlim())
            ax_marg_x.yaxis.set_major_locator(
                MaxNLocator(nbins=3, prune='both'))
            ax_marg_x.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            ax_marg_y.hist(y_valid, bins=bin_edges,
                           orientation='horizontal', color='blue', alpha=0.5, density=True)
            ax_marg_y.set_ylim(ax_joint.get_ylim())
            ax_marg_y.xaxis.set_major_locator(
                MaxNLocator(nbins=3, prune='both'))
            ax_marg_y.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            # Colorbar (linked to QuadMesh, not x-axis)
            cax = inset_axes(ax_joint,
                             width="80%", height="25%", loc='lower center',
                             bbox_to_anchor=(0, -0.10, 1, 0.1),
                             bbox_transform=ax_joint.transAxes,
                             borderpad=0)

            cbar = fig.colorbar(
                quadmesh, cax=cax, orientation='horizontal', extend=extend_opt)
            cbar.set_label(
                f'Counts {inpt.var_dict[data_typ]["label"]} max: {pctl}pctl')
            cbar.ax.xaxis.set_major_formatter(
                FormatStrFormatter('%d'))  # Format counts as integers
            cbar.ax.xaxis.set_ticks_position('bottom')
            cbar.ax.xaxis.set_label_position('bottom')

            # Debug print (optional)
            print("Colorbar limits:", quadmesh.get_clim())

    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_scatter_seasonal(period_label):
    print(f"SCATTERPLOTS {period_label}")
    plt.ioff()

    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=inpt.dpi)
    axs = ax.ravel()
    # Keep str_name for title and saving
    str_name = f"{inpt.tres} {period_label} scatter {inpt.var} {inpt.var_dict['t']['label']} {inpt.years[0]}-{inpt.years[-1]}"
    fig.suptitle(str_name, fontweight='bold')

    var_data = inpt.extr[inpt.var]
    comps = var_data['comps']
    ref_x = var_data['ref_x']
    plot_vars = tls.plot_vars_cleanup(comps, var_data)

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
            f"Plotting scatter {inpt.var_dict['t']['label']} - {inpt.var_dict[data_typ]['label']}")

        axs[i].scatter(
            x_valid, y_valid,
            s=5, facecolors='none',
            edgecolors=inpt.seasons[period_label]['col'],
            alpha=0.5
        )

        if valid_idx.sum() >= 2:
            calc_draw_fit(axs, i, x_valid, tres,
                          y_valid, inpt.seasons[period_label]['col'], data_typ, print_stats=True)
        else:
            calc_draw_fit(axs, i, x_valid, tres,
                          y_valid, inpt.seasons[period_label]['col'], data_typ, print_stats=False)
            print("ERROR: Not enough data points for fit.")

        format_scatterplot(axs, data_typ, i)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

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
    str_name = f"{inpt.tres} all CumSeas scatter {inpt.var} {inpt.var_dict['t']['label']} {inpt.years[0]}-{inpt.years[-1]}"
    fig.suptitle(str_name, fontweight='bold')
    fig.subplots_adjust(top=0.93)

    var_data = inpt.extr[inpt.var]
    comps = var_data['comps']
    ref_x = var_data['ref_x']
    plot_vars = tls.plot_vars_cleanup(comps, var_data)

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

            calc_draw_fit(axs, i,  merged['x'],  merged['y'], tres,
                          inpt.seasons[period_label]['col'], data_typ, print_stats=True)

            format_scatterplot(axs, data_typ, i)
            axs[i].legend()

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
                    calc_draw_fit(axs, i, x_valid, y_valid, tres,
                                  inpt.seasons[period_label]['col'], data_typ, print_stats=False)

                format_scatterplot(axs, data_typ, i)
                axs[i].legend()

    save_path = os.path.join(
        inpt.basefol['out']['base'], inpt.tres, f"{str_name.replace(' ', '_')}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


def plot_taylor_dia(ax, std_ref, std_models, corr_coeffs, model_labels,
                    ref_label='REF', colors=None, markers=None,
                    srange=(0, 1.5),
                    figsize=(8, 6), legend_loc='upper right',
                    var_marker_map=None):

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    std_models = np.array(std_models)
    corr_coeffs = np.clip(np.array(corr_coeffs), -1, 1)

    # Range per asse radiale (std dev)
    rmax = std_ref * srange[1]
    ax.set_ylim(0, rmax)

    # Setup polare
    ax.set_theta_direction(-1)            # senso orario
    ax.set_theta_zero_location('N')       # 0° a destra, REF sull'asse x

    ax.set_thetamin(0)
    ax.set_thetamax(90)

    # Etichette dei coefficienti di correlazione
    tgrid_degrees = np.arccos([1.0, 0.95, 0.9, 0.8, 0.6, 0.0]) * 180 / np.pi
    labels = [f"{np.cos(np.deg2rad(t)):.2f}" for t in tgrid_degrees]
    ax.set_thetagrids(tgrid_degrees, labels=labels)
    ax.set_rlabel_position(135)

    # ✅ Cerchio riferimento deviazione standard (ref)
    ref_circle = plt.Circle((0, 0), std_ref, transform=ax.transData._b,
                            color='black', fill=False, linestyle='--',
                            linewidth=1, label='Std Ref')
    ax.add_artist(ref_circle)

    # Punto di riferimento (REF)
    ax.plot(0, std_ref, 'ko', label=ref_label, markersize=6)

    if colors is None:
        colors = plt.cm.tab10.colors
    if markers is None:
        markers = ['o'] * len(std_models)

    # Punti modello
    for i, (std, corr) in enumerate(zip(std_models, corr_coeffs)):
        theta = np.arccos(corr)

        label = model_labels[i]
        tres = None
        try:
            tres = label.split('(')[1].split(')')[0].split(',')[-1].strip()
        except Exception:
            pass

        if tres == 'original':
            # Contorno nero (vuoto)
            ax.plot(theta, std, marker='o', color='black',
                    markersize=10, linestyle='None', markerfacecolor='none')
            # Marker interno
            ax.plot(theta, std,
                    marker=markers[i],
                    markerfacecolor=colors[i % len(colors)],
                    markeredgecolor=colors[i % len(colors)],
                    markersize=6,
                    linestyle='None')
        else:
            ax.plot(theta, std,
                    marker=markers[i],
                    color=colors[i % len(colors)],
                    linestyle='None')

    # Legend for variables (markers)
    if var_marker_map:
        legend_elements = [
            Line2D([], [], color='black', marker=mark,
                   linestyle='None', label=var)
            for var, mark in var_marker_map.items()
        ]
        leg1 = ax.legend(handles=legend_elements, title='Variables',
                         loc=legend_loc, fontsize='small', title_fontsize='medium')
        ax.add_artist(leg1)  # Add first legend manually

        model_keys = ['e', 'c', 't1', 't2']
        model_color_legend = [
            Line2D([], [], color=inpt.var_dict[k]['col'], marker='o',
                   linestyle='None', label=inpt.var_dict[k]['label'])
            for k in model_keys
        ]
        ax.legend(handles=model_color_legend, title='Models',
                  loc='lower right', fontsize='small', title_fontsize='medium')


def plot_taylor():
    print("Taylor Diagram")
    str_name = f"Taylor Diagram {inpt.var_dict['t']['label']} {inpt.years[0]}-{inpt.years[-1]}"

    combined_stdrefs = []
    combined_stds = []
    combined_cors = []
    combined_labels = []
    combined_colors = []
    combined_markers = []
    var_marker_map = {}
    available_markers = ['o', 's', '^', 'D',
                         'v', 'P', '*', 'X', 'H', '>', '<', '8']

    for var_idx, var in enumerate(inpt.list_var):
        marker = available_markers[var_idx % len(available_markers)]
        var_marker_map[var] = marker
        inpt.var = var
        var_data = inpt.extr[var]
        comps = var_data['comps']
        ref_x = var_data['ref_x']
        plot_vars = tls.plot_vars_cleanup(comps, var_data)

        for tres in inpt.tres_list:
            for model in plot_vars:
                tres, tres_tol = tls.get_tres(model, tres)
                try:
                    x = var_data[ref_x]['data_res'][tres][var]
                    y = var_data[model]['data_res'][tres][var]
                except KeyError:
                    continue  # skip if missing data

                time_range = pd.date_range(
                    start=pd.Timestamp(inpt.years[0], 1, 1),
                    end=pd.Timestamp(inpt.years[-1], 12, 31, 23, 59),
                    freq=tres
                )

                x_all = x.reindex(time_range, method='nearest',
                                  tolerance=tres_tol).astype(float)
                y_all = y.reindex(time_range).astype(float)

                valid_idx = ~(x_all.isna() | y_all.isna())
                x_valid = x_all[valid_idx]
                y_valid = y_all[valid_idx]

                if len(x_valid) == 0 or len(y_valid) == 0:
                    continue

                combined_stdrefs.append(np.std(x_valid))
                combined_stds.append(np.std(y_valid))
                combined_cors.append(np.corrcoef(x_valid, y_valid)[0, 1])
                combined_labels.append(f"{model} ({var}, {tres})")
                # Debug print summary
                if model == 'c':
                    color = 'red'
                elif model == 'e':
                    color = 'blue'
                elif model == 't2':
                    color = 'purple'
                elif model == 't1':
                    color = 'cyan'
                else:
                    color = inpt.var_dict[model]['col'] if model in inpt.var_dict else 'gray'

                combined_colors.append(color)
                combined_markers.append(var_marker_map[var])
                print(f"DEBUG: Model={model}, Var={var}, Tres={tres} | "
                      f"StdRef={combined_stdrefs[-1]:.3f}, StdModel={combined_stds[-1]:.3f}, "
                      f"Corr={combined_cors[-1]:.3f}, Color={combined_colors[-1]}, Marker={combined_markers[-1]}")

    # Plot
    fig = plt.figure(figsize=(12, 10), dpi=inpt.dpi)
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    fig.suptitle(f"{str_name}", fontweight='bold')
    fig.subplots_adjust(top=0.93)

    std_ref = combined_stdrefs[0] if combined_stdrefs else 1.0

    plot_taylor_dia(ax, std_ref,
                    combined_stds,
                    combined_cors,
                    combined_labels,
                    ref_label='Obs',
                    colors=combined_colors,
                    markers=combined_markers,
                    var_marker_map=var_marker_map)

    save_path = os.path.join(
        inpt.basefol['out']['base'], f"{str_name.replace(' ', '_')}.png")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


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

    if print_stats:
        calc_stats(xx, yy, data_typ, tr)
        r2 = inpt.extr[inpt.var][data_typ]['data_stats'][tr]['r2']
        N = inpt.extr[inpt.var][data_typ]['data_stats'][tr]['N']
        rmse = inpt.extr[inpt.var][data_typ]['data_stats'][tr]['rmse']
        mbe = inpt.extr[inpt.var][data_typ]['data_stats'][tr]['mbe']
        std_x = inpt.extr[inpt.var][ref_x]['data_stats'][tr]['std_x']
        std_y = inpt.extr[inpt.var][data_typ]['data_stats'][tr]['std_y']

        def escape_label(label):
            return label.replace('_', r'\_')

        stats_text = (
            f"R² = {r2:.2f}\n"
            f"N = {N}\n"
            f"y = {b:+.2f}x {a:+.2f}\n"
            f"MBE = {mbe:.2f}\n"
            f"RMSE = {rmse:.2f}\n"
            f"$\\sigma_{{{escape_label(var_dict[ref_x]['label'])}}} = {std_x:.2f}$\n"
            f"$\\sigma_{{{escape_label(var_dict[data_typ]['label'])}}} = {std_y:.2f}$"
        )

        axs[i].text(0.57, 0.20, stats_text,
                    transform=axs[i].transAxes,
                    fontsize=12, color='black',
                    ha='left', va='center',
                    bbox=dict(facecolor='white', edgecolor='white'))


def calc_stats(x, y, data_typ, tr):
    ref_x = inpt.extr[inpt.var]['ref_x']
    corcoeff = np.corrcoef(x, y)[0, 1]
    diff = y-x
    if 'data_stats' not in inpt.extr[inpt.var][data_typ]:
        inpt.extr[inpt.var][data_typ]['data_stats'] = {}
    if 'data_stats' not in inpt.extr[inpt.var][ref_x]:
        inpt.extr[inpt.var][ref_x]['data_stats'] = {}
    
    inpt.extr[inpt.var][data_typ]['data_stats'][tr] = {}
    inpt.extr[inpt.var][ref_x]['data_stats'][tr] = {}
    inpt.extr[inpt.var][data_typ]['data_stats'][tr]['r2'] = corcoeff*corcoeff

    inpt.extr[inpt.var][data_typ]['data_stats'][tr]['N'] = len(y)

    inpt.extr[inpt.var][data_typ]['data_stats'][tr]['rmse'] = np.sqrt(
        np.nanmean(diff ** 2))
    inpt.extr[inpt.var][data_typ]['data_stats'][tr]['mbe'] = np.nanmean(diff)
    inpt.extr[inpt.var][ref_x]['data_stats'][tr]['std_x'] = np.std(x)
    inpt.extr[inpt.var][data_typ]['data_stats'][tr]['std_y'] = np.std(y)
    return


def format_ba(axs, data_typ, i):
    """
    Sets title, labels, limits, and annotations for a scatterplot axs[i] of component `comp`.

    :param axs: Array-like of matplotlib Axes.
    :param data_typ: Component key for labeling.
    :param i: Index of subplot.
    """
    var = inpt.var
    var_dict = inpt.var_dict
    # var_min = -2
    # var_max = 2
    ref_x = inpt.extr[var]['ref_x']

    axs[i].set_title(var_dict[data_typ]['label'])
    axs[i].set_xlabel(
        f"mean({var_dict[ref_x]['label']},{var_dict[data_typ]['label']})")
    axs[i].set_ylabel(
        f"{var_dict[data_typ]['label']}-{var_dict[ref_x]['label']}")
    # axs[i].set_xlim(var_min, var_max)
    # axs[i].set_ylim(var_min, var_max)
    axs[i].text(0.01, 0.95, inpt.letters[i] + ')',
                transform=axs[i].transAxes)
    # pos = axs[i].get_position()  # get current position: Bbox(x0, y0, x1, y1)

    # new_pos = [pos.x0, pos.y0 + 0.03, pos.x1, pos.y1 + 0.03]  # shift up by 0.03
    # axs[i].set_position(new_pos)


def format_scatterplot(axs, data_typ, i):
    """
    Sets title, labels, limits, and annotations for a scatterplot axs[i] of component `comp`.

    :param axs: Array-like of matplotlib Axes.
    :param data_typ: Component key for labeling.
    :param i: Index of subplot.
    """
    var = inpt.var
    var_dict = inpt.var_dict
    var_min = inpt.extr[var]['min']
    var_max = inpt.extr[var]['max']
    ref_x = inpt.extr[var]['ref_x']

    axs[i].set_title(var_dict[data_typ]['label'])
    axs[i].set_xlabel(var_dict[ref_x]['label'])
    axs[i].set_ylabel(var_dict[data_typ]['label'])
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
