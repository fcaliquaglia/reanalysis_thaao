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
import numpy.ma as ma
import pandas as pd

import inputs as inpt


def plot_ts(period_label):
    """

    :param period_label:
    :return:
    """
    # with plt.xkcd():
    print('TIMESERIES')
    fig, ax = plt.subplots(len(inpt.years), 1, figsize=(12, 17), dpi=300)
    fig.suptitle(f"{inpt.var.upper()} all {inpt.tres}", fontweight='bold')
    kwargs_ori = {'alpha': 0.02, 'lw': 0, 'marker': '.', 'ms': 1}
    kwargs = {'lw': 0, 'marker': '.', 'ms': 2}

    for (yy, year) in enumerate(inpt.years):
        print(f"plotting {year}")

        # original resolution
        for varvar in inpt.extr[inpt.var]['comps'] + [inpt.extr[inpt.var]['ref_x']]:
            try:
                data = inpt.extr[inpt.var][varvar]['data'][inpt.extr[inpt.var][varvar]['data'].index.year == year]
                ax[yy].plot(data, color=inpt.var_dict[varvar]['col_ori'], **kwargs_ori)
            except AttributeError:
                pass

        # resampled resolution
        for varvar in inpt.extr[inpt.var]['comps'] + [inpt.extr[inpt.var]['ref_x']]:
            try:
                data = inpt.extr[inpt.var][varvar]['data_res'][
                    inpt.extr[inpt.var][varvar]['data_res'].index.year == year]
                ax[yy].plot(data, color=inpt.var_dict[varvar]['col'], label=inpt.var_dict[varvar]['label'], **kwargs)
            except AttributeError:
                pass

        if inpt.var == 'alb':
            range1 = pd.date_range(dt.datetime(year, 1, 1), dt.datetime(year, 2, 15), freq=inpt.tres)
            range2 = pd.date_range(dt.datetime(year, 11, 1), dt.datetime(year, 12, 31), freq=inpt.tres)
            ax[yy].vlines(range1.values, 0, 1, color='grey', alpha=0.3)
            ax[yy].vlines(range2.values, 0, 1, color='grey', alpha=0.3)
            ax[yy].set_ylim(inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max'])
        else:
            pass
        ax[yy].set_ylim(inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max'])
        ax[yy].text(0.5, 0.90, year, transform=ax[yy].transAxes, horizontalalignment='center')
        ax[yy].xaxis.set_major_formatter(inpt.myFmt)
        ax[yy].set_xlim(dt.datetime(year, 1, 1), dt.datetime(year, 12, 31))
        ax[yy].text(0.01, 0.95, inpt.letters[yy] + ')', transform=ax[yy].transAxes)
    plt.xlabel('Time')
    plt.legend(ncol=2)
    plt.savefig(os.path.join(inpt.basefol_out, inpt.tres, f"{inpt.tres}_{period_label}_{inpt.var}.png"))
    plt.close('all')


def plot_residuals(period_label):
    """

    :param period_label:
    :return:
    """
    print('RESIDUALS')

    fig, ax = plt.subplots(len(inpt.years), 1, figsize=(12, 17), dpi=300)
    fig.suptitle(f"residuals {inpt.var.upper()} all {inpt.tres}", fontweight='bold')
    kwargs = {'lw': 1, 'marker': '.', 'ms': 0}

    for [yy, year] in enumerate(inpt.years):
        print(f"plotting {year}")

        daterange = pd.date_range(dt.datetime(year, 1, 1), dt.datetime(year, 12, 31))
        ax[yy].plot(daterange, np.repeat(0, len(daterange)), color='black', lw=2, ls='--')

        vr_ref = inpt.extr[inpt.var][inpt.extr[inpt.var]['ref_x']]['data_res']
        # resampled resolution
        for varvar in inpt.extr[inpt.var]['comps']:
            try:
                data = inpt.extr[inpt.var][varvar]['data_res'][
                    inpt.extr[inpt.var][varvar]['data_res'].index.year == year]
                ax[yy].plot(
                        data - vr_ref[vr_ref.index.year == year], color=inpt.var_dict[varvar]['col'],
                        label=inpt.var_dict[varvar]['label'], **kwargs)
            except AttributeError:
                pass

        if inpt.var == 'alb':
            range1 = pd.date_range(dt.datetime(year, 1, 1), dt.datetime(year, 2, 15), freq=inpt.tres)
            range2 = pd.date_range(dt.datetime(year, 11, 1), dt.datetime(year, 12, 31), freq=inpt.tres)
            ax[yy].vlines(range1.values, -0.5, 0.5, color='grey', alpha=0.3)
            ax[yy].vlines(range2.values, -0.5, 0.5, color='grey', alpha=0.3)
        else:
            pass

        ax[yy].set_ylim(inpt.extr[inpt.var]['res_min'], inpt.extr[inpt.var]['res_max'])
        ax[yy].text(0.5, 0.90, year, transform=ax[yy].transAxes, horizontalalignment='center')
        ax[yy].xaxis.set_major_formatter(inpt.myFmt)
        ax[yy].set_xlim(dt.datetime(year, 1, 1), dt.datetime(year, 12, 31))
        # panel letters
        ax[yy].text(0.01, 0.95, inpt.letters[yy] + ')', transform=ax[yy].transAxes)
    plt.xlabel('Time')
    plt.legend()
    plt.savefig(os.path.join(inpt.basefol_out, inpt.tres, f"{inpt.tres}_{period_label}_residuals_{inpt.var}.png"))
    plt.close('all')


def plot_scatter(period_label):
    """

    :param period_label:
    :return:
    """
    print(f"SCATTERPLOTS {period_label}")

    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
    axs = ax.ravel()

    x = inpt.extr[inpt.var][inpt.extr[inpt.var]['ref_x']]['data_res']
    for i, comp in enumerate(inpt.extr[inpt.var]['comps']):

        y = inpt.extr[inpt.var][comp]['data_res']
        axs[i].set_ylabel(inpt.var_dict[comp]['label'])
        axs[i].set_xlabel(inpt.var_dict[inpt.extr[inpt.var]['ref_x']]['label'])

        try:
            print(f"plotting scatter {inpt.var_dict['t']['label']}-{inpt.var_dict[comp]['label']}")

            fig.suptitle(f"{inpt.var.upper()} {inpt.seass[period_label]['name']} {inpt.tres}", fontweight='bold')
            axs[i].set_title(inpt.var_dict[comp]['label'])

            time_list = pd.date_range(
                    start=dt.datetime(inpt.years[0], 1, 1, 0, 0), end=dt.datetime(inpt.years[-1], 12, 31, 23, 59),
                    freq=inpt.tres)

            x_all = x.reindex(time_list).fillna(np.nan)
            x_s = x_all.loc[(x_all.index.month.isin(inpt.seass[period_label]['months']))]
            y_all = y.reindex(time_list).fillna(np.nan)
            y_s = y_all.loc[(y_all.index.month.isin(inpt.seass[period_label]['months']))]
            idx = ~(np.isnan(x_s[inpt.var]) | np.isnan(y_s[inpt.var]))

            if inpt.seass[period_label]['name'] != 'all':
                axs[i].scatter(
                        x_s[inpt.var][idx], y_s[inpt.var][idx], s=5, color=inpt.seass[period_label]['col'],
                        facecolor='none', alpha=0.5,
                        label=period_label)
            else:
                bin_size = (inpt.extr[inpt.var]['max'] - inpt.extr[inpt.var]['min']) / inpt.extr[inpt.var]['bin_nr']
                h = axs[i].hist2d(
                        x_s[inpt.var][idx], y_s[inpt.var][idx], bins=[np.linspace(
                                inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max'], inpt.extr[inpt.var]['bin_nr']),
                            np.linspace(
                                    inpt.extr[inpt.var]['min'],
                                    inpt.extr[inpt.var]['max'],
                                    inpt.extr[inpt.var]['bin_nr'])],
                        cmap=plt.cm.jet,
                        cmin=1, vmin=1)
                axs[i].text(
                        0.10, 0.90, f"bin_size={bin_size}",
                        transform=axs[i].transAxes)  # fig.colorbar(h[3], ax=axs[i], extend='both')

                # if len(x_s[idx]) < 2 | len(y_s[idx]) < 2:
                #     print('ERROR, ERROR, NO DATA ENOUGH FOR PROPER FIT (i.e. only 1 point available)')
                # else:
                #     calc_draw_fit(axs, i, idx, x_s, y_s)

            axs[i].set_xlim(inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max'])
            axs[i].set_ylim(inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max'])
            axs[i].text(0.01, 0.95, inpt.letters[i] + ')', transform=axs[i].transAxes)
            axs[i].plot(
                    [inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max']],
                    [inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max']],
                    color='black', lw=1.5, ls='-')
        except:
            print(f"error with {inpt.var_dict[comp]['label']}")

    plt.savefig(
            os.path.join(
                    inpt.basefol_out, inpt.tres,
                    f"{inpt.tres}_scatter_{inpt.seass[period_label]['name']}_{inpt.var}.png"))
    plt.close('all')


def plot_scatter_cum():
    """

    :return:
    """
    import copy as cp
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
    seass_new = cp.copy(inpt.seass)
    seass_new.pop('all')

    for period_label in seass_new:
        print(f"SCATTERPLOTS CUMULATIVE {period_label}")

        axs = ax.ravel()

        x = inpt.extr[inpt.var][inpt.extr[inpt.var]['ref_x']]['data_res']
        for i, comp in enumerate(inpt.extr[inpt.var]['comps']):
            y = inpt.extr[inpt.var][comp]['data_res']
            axs[i].set_ylabel(inpt.var_dict[comp]['label'])
            axs[i].set_xlabel(inpt.var_dict[inpt.extr[inpt.var]['ref_x']]['label'])
            try:
                print(
                        f"plotting scatter {inpt.var_dict[inpt.extr[inpt.var]['ref_x']]['label']}-{inpt.var_dict[comp]['label']}")

                fig.suptitle(f"{inpt.var.upper()} cumulative plot", fontweight='bold')
                axs[i].set_title(inpt.var_dict[comp]['label'])

                time_list = pd.date_range(
                        start=dt.datetime(inpt.years[0], 1, 1, 0, 0), end=dt.datetime(inpt.years[-1], 12, 31, 23, 59),
                        freq=inpt.tres)

                x_all = x.reindex(time_list).fillna(np.nan)
                x_s = x_all.loc[(x_all.index.month.isin(seass_new[period_label]['months']))]
                y_all = y.reindex(time_list).fillna(np.nan)
                y_s = y_all.loc[(y_all.index.month.isin(seass_new[period_label]['months']))]
                idx = ~(np.isnan(x_s[inpt.var]) | np.isnan(y_s[inpt.var]))

                axs[i].scatter(
                        x_s[inpt.var][idx], y_s[inpt.var][idx], s=5, color=seass_new[period_label]['col'],
                        edgecolors='none', alpha=0.5,
                        label=period_label)

                # if len(x_s[idx]) < 2 | len(y_s[idx]) < 2:
                #     print('ERROR, ERROR, NO DATA ENOUGH FOR PROPER FIT (i.e. only 1 point available)')
                # else:
                #     calc_draw_fit(axs, i, idx, x_s[inpt.var], y_s[inpt.var], print_stats=False)

                axs[i].set_xlim(inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max'])
                axs[i].set_ylim(inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max'])
                axs[i].text(0.01, 0.95, inpt.letters[i] + ')', transform=axs[i].transAxes)
                axs[i].plot(
                        [inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max']],
                        [inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max']],
                        color='black', lw=1.5, ls='-')
                axs[i].legend()
            except:
                print(f"error with {inpt.var_dict[comp]['label']}")

    plt.savefig(os.path.join(inpt.basefol_out, inpt.tres, f"{inpt.tres}_scatter_cum_{inpt.var}.png"))
    plt.close('all')


def calc_draw_fit(axs, i, xx, yy, per_lab, print_stats=True):
    """

    :param per_lab:
    :param axs:
    :param i:
    :param xx:
    :param yy:
    :param print_stats:
    :return:
    """

    b, a = np.polyfit(xx.values, yy.values, deg=1)
    xseq = np.linspace(inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max'], num=1000)
    axs[i].plot(xseq, a + b * xseq, color=inpt.seass[per_lab]['col'], lw=2.5, ls='--', alpha=0.5)
    axs[i].plot(
            [inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max']],
            [inpt.extr[inpt.var]['min'], inpt.extr[inpt.var]['max']], color='black', lw=1.5, ls='-')
    if print_stats:
        corcoef = ma.corrcoef(xx, yy)
        N = len(yy)
        rmse = np.sqrt(np.nanmean((yy - xx) ** 2))
        mbe = np.nanmean(yy - xx)
        axs[i].text(
                0.50, 0.30, f"R={corcoef[0, 1]:.2f} N={N} \n y={b:+.2f}x{a:+.2f} \n MBE={mbe:.2f} RMSE={rmse:.2f}",
                transform=axs[i].transAxes, fontsize=14, color='black', ha='left', va='center',
                bbox=dict(facecolor='white', edgecolor='white'))

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
#     fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
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
#         comps = ['c', 'e', 'l', 't2']
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
#         if comp == 'l':
#             label = 'ERA5-L'
#             axs[i].set_ylabel(label)
#             try:
#                 y = vr_l_res[inpt.var]
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
#     plt.savefig(os.path.join(basefol_out, tres, f"{tres}_ba_{seas_name}_{vr}.png"))
#     plt.close('all')
#
