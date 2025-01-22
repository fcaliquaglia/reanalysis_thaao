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

import string

import matplotlib.pyplot as plt
import numpy.ma as ma

from inputs import *

letters = list(string.ascii_lowercase)


def plot_ts(vr, avar, period_label):
    """

    :param vr:
    :param avar:
    :param period_label:
    :return:
    """
    print('TIMESERIES')
    [vr_c, vr_e, vr_l, vr_t, vr_t1, vr_t2, vr_c_res, vr_e_res, vr_l_res, vr_t_res, vr_t1_res, vr_t2_res] = avar
    fig, ax = plt.subplots(len(years), 1, figsize=(12, 17), dpi=300)
    fig.suptitle(f'{vr.upper()} all {tres}', fontweight='bold')
    kwargs_ori = {'alpha': 0.02, 'lw': 0, 'marker': '.', 'ms': 1}
    kwargs = {'lw': 0, 'marker': '.', 'ms': 2}

    if vr != 'iwv':
        var_dict['t2']['label'] = 'AWS ECAPAC'
        var_dict['t2']['label_uom'] = 'AWS ECAPAC'
    else:
        var_dict['t2']['label'] = 'RS'
        var_dict['t2']['label_uom'] = 'RS'

    for [yy, year] in enumerate(years):
        print(f'plotting {year}')

        # original resolution
        for (varvar, vr_n) in zip([vr_c, vr_e, vr_l, vr_t, vr_t1, vr_t2], var_names):
            try:
                data = varvar[varvar.index.year == year]
                ax[yy].plot(data, color=var_dict[vr_n]['col_ori'], **kwargs_ori)
            except AttributeError:
                pass

        # resampled resolution
        for (varvar, vr_n) in zip([vr_c_res, vr_e_res, vr_l_res, vr_t_res, vr_t1_res, vr_t2_res], var_names):
            try:
                data = varvar[varvar.index.year == year]
                ax[yy].plot(data, color=var_dict[vr_n]['col'], label=var_dict[vr_n]['label'], **kwargs)
            except AttributeError:
                pass

        if vr == 'alb':
            range1 = pd.date_range(dt.datetime(year, 1, 1), dt.datetime(year, 2, 15), freq=tres)
            range2 = pd.date_range(dt.datetime(year, 11, 1), dt.datetime(year, 12, 31), freq=tres)
            ax[yy].vlines(range1.values, 0, 1, color='grey', alpha=0.3)
            ax[yy].vlines(range2.values, 0, 1, color='grey', alpha=0.3)
            ax[yy].set_ylim(extr[vr]['min'], extr[vr]['max'])
        else:
            pass
        ax[yy].set_ylim(extr[vr]['min'], extr[vr]['max'])
        ax[yy].text(0.45, 0.85, year, transform=ax[yy].transAxes)
        ax[yy].xaxis.set_major_formatter(myFmt)
        ax[yy].set_xlim(dt.datetime(year, 1, 1), dt.datetime(year, 12, 31))
        ax[yy].text(0.01, 0.90, letters[yy] + ')', transform=ax[yy].transAxes)
    plt.xlabel('Time')
    plt.legend(ncol=2)
    plt.savefig(os.path.join(basefol_out, tres, f'{tres}_{period_label}_{vr}.png'))
    plt.close('all')


def plot_residuals(vr, avar, period_label):
    """

    :param vr:
    :param avar:
    :param period_label:
    :return:
    """
    print('RESIDUALS')
    [vr_c, vr_e, vr_l, vr_t, vr_t1, vr_t2, vr_c_res, vr_e_res, vr_l_res, vr_t_res, vr_t1_res, vr_t2_res] = avar
    fig, ax = plt.subplots(len(years), 1, figsize=(12, 17), dpi=300)
    fig.suptitle(f'residuals {vr.upper()} all {tres}', fontweight='bold')
    kwargs = {'lw': 1, 'marker': '.', 'ms': 0}

    for [yy, year] in enumerate(years):
        print(f'plotting {year}')

        daterange = pd.date_range(dt.datetime(year, 1, 1), dt.datetime(year, 12, 31))
        ax[yy].plot(daterange, np.repeat(0, len(daterange)), color='black', lw=2, ls='--')

        # resampled resolution
        vr_ref = vr_t_res.resample(tres).mean()
        for (varvar, vr_n) in zip([vr_c_res, vr_e_res, vr_l_res, vr_t_res, vr_t1_res, vr_t2_res], var_names):
            try:
                data = varvar[varvar.index.year == year]
                ax[yy].plot(
                        (data - vr_ref[vr_ref.index.year == year]), color=var_dict[vr_n]['col'],
                        label=var_dict[vr_n]['label'], **kwargs)
            except AttributeError:
                pass

        if vr == 'alb':
            range1 = pd.date_range(dt.datetime(year, 1, 1), dt.datetime(year, 2, 15), freq=tres)
            range2 = pd.date_range(dt.datetime(year, 11, 1), dt.datetime(year, 12, 31), freq=tres)
            ax[yy].vlines(range1.values, -0.5, 0.5, color='grey', alpha=0.3)
            ax[yy].vlines(range2.values, -0.5, 0.5, color='grey', alpha=0.3)
        else:
            pass

        ax[yy].set_ylim(extr[vr]['res_min'], extr[vr]['res_max'])
        ax[yy].text(0.45, 0.85, year, transform=ax[yy].transAxes)
        ax[yy].xaxis.set_major_formatter(myFmt)
        ax[yy].set_xlim(dt.datetime(year, 1, 1), dt.datetime(year, 12, 31))
        # panel letters
        ax[yy].text(0.01, 0.90, letters[yy] + ')', transform=ax[yy].transAxes)
    plt.xlabel('Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(basefol_out, tres, f'{tres}_{period_label}_residuals_{vr}.png'))
    plt.close('all')


def plot_scatter(vr, avar, period_label):
    """

    :param vr:
    :param avar:
    :param period_label:
    :return:
    """
    print('SCATTERPLOTS')
    [vr_c, vr_e, vr_l, vr_t, vr_t1, vr_t2, vr_c_res, vr_e_res, vr_l_res, vr_t_res, vr_t1_res, vr_t2_res] = avar
    seas_name = seass[period_label]['name']
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
    axs = ax.ravel()

    comps, x, refx = var_comp_x_selection(vr, avar)

    for i, comp in enumerate(comps):
        # define which is the reference measurement for each variable
        y, var_t_res = var_selection(vr, avar, comp)
        axs[i].set_ylabel(var_dict[comp]['label'])
        axs[i].set_xlabel(var_dict[refx]['label'])

        try:
            print(f'plotting scatter THAAO-{var_dict[comp]['label']}')

            fig.suptitle(f'{vr.upper()} {seas_name} {tres}', fontweight='bold')
            axs[i].set_title(var_dict[comp]['label'])

            time_list = pd.date_range(start=dt.datetime(years[0], 1, 1), end=dt.datetime(years[-1], 12, 31), freq=tres)
            if x.empty | y.empty:
                continue
            x_all = x.reindex(time_list).fillna(np.nan)
            x_s = x_all.loc[(x_all.index.month.isin(seass[period_label]['months']))]
            y_all = y.reindex(time_list).fillna(np.nan)
            y_s = y_all.loc[(y_all.index.month.isin(seass[period_label]['months']))]
            idx = np.isfinite(x_s) & np.isfinite(y_s)

            if seas_name != 'all':
                axs[i].scatter(x_s[idx], y_s[idx], color=seass[period_label]['col'])
            else:
                if var_dict[comp]['label'] == 'RS':
                    axs[i].scatter(x_s[idx], y_s[idx], color=seass[period_label]['col'])
                else:
                    if tres == '1ME':
                        axs[i].scatter(
                                x_s[idx], y_s[idx], color=seass[period_label]['col' + '_' + var_dict[comp]['label']])
                    else:
                        bin_size = extr[vr]['max'] / bin_nr
                        h = axs[i].hist2d(x_s[idx], y_s[idx], bins=bin_nr, cmap=plt.cm.jet, cmin=1, vmin=1)
                        axs[i].text(
                                0.10, 0.80, f'bin_size={bin_size} {extr[vr]['uom']}', transform=axs[i].transAxes)

            if len(x_s[idx]) < 2 | len(y_s[idx]) < 2:
                print('ERROR, ERROR, NO DATA ENOUGH FOR PROPER FIT (i.e. only 1 point available)')
            else:
                calc_draw_fit(axs, i, idx, vr, x_s, y_s)

            axs[i].set_xlim(extr[vr]['min'], extr[vr]['max'])
            axs[i].set_ylim(extr[vr]['min'], extr[vr]['max'])
            axs[i].text(0.05, 0.95, letters[i] + ')', transform=axs[i].transAxes)
        except:
            print(f'error with {var_dict[comp]['label']}')

    plt.savefig(os.path.join(basefol_out, tres, f'{tres}_scatter_{seas_name}_{vr}.png'))
    plt.close('all')


def calc_draw_fit(axs, i, idx, vr, x_s, y_s, print_stats=True):
    """

    :param axs:
    :param i:
    :param idx:
    :param vr:
    :param x_s:
    :param y_s:
    :param print_stats:
    :return:
    """
    b, a = np.polyfit(x_s[idx], y_s[idx], deg=1)
    xseq = np.linspace(extr[vr]['min'], extr[vr]['max'], num=1000)
    axs[i].plot(xseq, a + b * xseq, color=var_dict[vr]['col'], lw=2.5, ls='--', alpha=0.5)
    axs[i].plot(
            [extr[vr]['min'], extr[vr]['max']], [extr[vr]['min'], extr[vr]['max']], color='black', lw=1.5, ls='-')
    if print_stats:
        corcoef = ma.corrcoef(x_s[idx], y_s[idx])
        N = len(y_s[idx])
        rmse = np.sqrt(np.nanmean((y_s[idx] - x_s[idx]) ** 2))
        mbe = np.nanmean(y_s[idx] - x_s[idx])
        axs[i].text(
                0.50, 0.30, f'R={corcoef[0, 1]:.2f} N={N} \n y={b:+.2f}x{a:+.2f} \n MBE={mbe:.2f} RMSE={rmse:.2f}',
                transform=axs[i].transAxes, fontsize=14, color='black', ha='left', va='center',
                bbox=dict(facecolor='white', edgecolor='white'))


# def plot_ba(vr, avar, period_label):
#     """
#
#     :param vr:
#     :param avar:
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
#         x = vr_t1_res[vr]
#         xlabel = 'HATPRO'
#     elif vr in ['windd', 'winds', 'precip']:
#         comps = ['c', 'e', 't', 't1']
#         x = vr_t2_res[vr]
#         xlabel = 'AWS_ECAPAC'
#     elif vr == 'iwv':
#         comps = ['c', 'e', 't1', 't2']
#         x = vr_t_res[vr]
#         xlabel = 'VESPA'
#     elif vr == 'temp':
#         comps = ['c', 'e', 'l', 't2']
#         x = vr_t_res[vr]
#         xlabel = 'THAAO'
#     else:
#         comps = ['c', 'e', 't1', 't2']
#         x = vr_t_res[vr]
#         xlabel = 'THAAO'
#
#     for i, comp in enumerate(comps):
#         axs[i].set_xlabel(xlabel)
#         if comp == 'c':
#             label = 'CARRA'
#             axs[i].set_ylabel(label)
#             try:
#                 y = vr_c_res[vr]
#             except KeyError:
#                 print(f'error with {label}')
#                 continue
#         if comp == 'e':
#             label = 'ERA5'
#             axs[i].set_ylabel(label)
#             try:
#                 y = vr_e_res[vr]
#             except KeyError:
#                 print(f'error with {label}')
#                 continue
#         if comp == 'l':
#             label = 'ERA5-L'
#             axs[i].set_ylabel(label)
#             try:
#                 y = vr_l_res[vr]
#             except KeyError:
#                 print(f'error with {label}')
#                 continue
#         if comp == 't':
#             label = 'THAAO'
#             axs[i].set_ylabel(label)
#             try:
#                 y = vr_t_res[vr]
#             except KeyError:
#                 print(f'error with {label}')
#                 continue
#         if comp == 't1':
#             label = 'HATPRO'
#             axs[i].set_ylabel(label)
#             try:
#                 y = vr_t1_res[vr]
#             except KeyError:
#                 print(f'error with {label}')
#                 continue
#         if comp == 't2':
#             if vr == 'alb':
#                 label = 'ERA5 snow alb'
#             else:
#                 label = 'AWS ECAPAC'
#             axs[i].set_ylabel(label)
#             try:
#                 y = vr_t2_res[vr]
#             except KeyError:
#                 print(f'error with {label}')
#                 continue
#         try:
#             print(f'plotting ba THAAO-{label}')
#
#             fig.suptitle(f'{vr.upper()} {seas_name} {tres}', fontweight='bold')
#             axs[i].set_title(label)
#             axs[i].text(0.01, 0.90, letters[i] + ')', transform=axs[i].transAxes)
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
#             # b, a = np.polyfit(x_s[idx], y_s[idx], deg=1)  # xseq = np.linspace(extr[vr]['min'], extr[vr]['max'], num=1000)  # axs[i].plot(xseq, a + b * xseq, color='red', lw=2.5, ls='--')  # axs[i].plot(  #         [extr[vr]['min'], extr[vr]['max']], [extr[vr]['min'], extr[vr]['max']], color='black', lw=1.5,  #         ls='-')  # corcoef = ma.corrcoef(x_s[idx], y_s[idx])  #  # N = x_s[idx].shape[0]  # rmse = np.sqrt(np.nanmean((x_s[idx] - y_s[idx]) ** 2))  # mae = np.nanmean(np.abs(x_s[idx] - y_s[idx]))  # axs[i].text(  #         0.60, 0.15, f'R={corcoef[0, 1]:1.3}\nrmse={rmse:1.3}\nN={N}\nmae={mae:1.3}', fontsize=14,  #         transform=axs[i].transAxes)  # axs[i].set_xlim(extr[vr]['min'], extr[vr]['max'])  # axs[i].set_ylim(extr[vr]['min'], extr[vr]['max'])
#         except:
#             print(f'error with {label}')
#
#     plt.savefig(os.path.join(basefol_out, tres, f'{tres}_ba_{seas_name}_{vr}.png'))
#     plt.close('all')
#


def plot_scatter_cum(vr, avar):
    """

    :param vr:
    :param avar:
    :return:
    """
    import copy as cp
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
    seass_new = cp.copy(seass)
    seass_new.pop('all')

    for period_label in seass_new:
        print('SCATTERPLOTS')
        seas_name = seass[period_label]['name']
        axs = ax.ravel()
        comps, x, refx = var_comp_x_selection(vr, avar)
        for i, comp in enumerate(comps):

            axs[i].set_ylabel(var_dict[comp]['label_uom'])
            x, y, vr_t_res = var_selection(vr, avar, comp)

            try:
                print(f'plotting scatter VESPA-{var_dict[comp]['label']}')

                fig.suptitle(f'{vr.upper()} cumulative plot', fontweight='bold')
                axs[i].set_title(var_dict[comp]['label'])

                time_list = pd.date_range(
                        start=dt.datetime(years[0], 1, 1), end=dt.datetime(years[-1], 12, 31), freq=tres)

                x_all = x.reindex(time_list).fillna(np.nan)
                x_s = x_all.loc[(x_all.index.month.isin(seass[period_label]['months']))]
                y_all = y.reindex(time_list).fillna(np.nan)
                y_s = y_all.loc[(y_all.index.month.isin(seass[period_label]['months']))]
                idx = ~(np.isnan(x_s) | np.isnan(y_s))

                if seas_name != 'all':
                    if var_dict[comp]['label'] == 'RS':
                        y_s = y.loc[(y.index.month.isin(seass[period_label]['months']))]
                        x_s = pd.Series(vr_t_res.reindex(y_s.index)[vr])

                        idx = ~(np.isnan(x_s) | np.isnan(y_s))
                        axs[i].scatter(
                                x_s[idx].values, y_s[idx].values, s=50, facecolor='none',
                                color=seass[period_label]['col'], label=period_label)
                        axs[i].set_xlabel(var_dict[comp]['label_uom'])

                    else:
                        axs[i].scatter(
                                x_s[idx], y_s[idx], s=5, color=seass[period_label]['col'], edgecolors='none', alpha=0.5,
                                label=period_label)
                        axs[i].set_xlabel(var_dict['vr_t']['label_uom'])

                if len(x_s[idx]) < 2 | len(y_s[idx]) < 2:
                    print('ERROR, ERROR, NO DATA ENOUGH FOR PROPER FIT (i.e. only 1 point available)')
                else:
                    calc_draw_fit(axs, i, idx, x_s, y_s, print_stats=False)

                    axs[i].set_xlim(extr[vr]['min'], extr[vr]['max'])
                    axs[i].set_ylim(extr[vr]['min'], extr[vr]['max'])
                    axs[i].text(0.05, 0.95, letters[i] + ')', transform=axs[i].transAxes)
                    axs[i].legend()
            except:
                print(f'error with {var_dict[comp]['label']}')

    plt.savefig(os.path.join(basefol_out, tres, f'{tres}_scatter_cum_{vr}_only.png'))
    plt.close('all')


def var_comp_x_selection(vr, avar):
    [vr_c, vr_e, vr_l, vr_t, vr_t1, vr_t2, vr_c_res, vr_e_res, vr_l_res, vr_t_res, vr_t1_res, vr_t2_res] = avar
    if vr == 'lwp':
        cmps = ['c', 'e', 't', 't1']
        x = vr_t1_res[vr]
        ref_x = 't1'
    elif vr in ['windd', 'winds', 'precip']:
        cmps = ['c', 'e', 't', 't1']
        x = vr_t2_res[vr]
        ref_x = 't2'
    elif vr == 'iwv':
        cmps = ['c', 'e', 't1', 't2']
        x = vr_t_res[vr]
        ref_x = 't'
    elif vr in ['temp']:
        cmps = ['c', 'e', 'l', 't2']
        x = vr_t_res[vr]
        ref_x = 't'
    elif vr in ['precip']:
        cmps = ['c', 'e']
        x = vr_t1_res[vr]
        ref_x = 't2'
    else:
        cmps = ['c', 'e', 't1', 't2']
        x = vr_t_res[vr]
        ref_x = 't'

    return cmps, x, ref_x


def var_selection(vr, avar, comp):
    [vr_c, vr_e, vr_l, vr_t, vr_t1, vr_t2, vr_c_res, vr_e_res, vr_l_res, vr_t_res, vr_t1_res, vr_t2_res] = avar

    if comp == 'c':
        try:
            y = vr_c_res[vr]
        except KeyError:
            print(f'error with {var_dict[comp]['label']}')
    if comp == 'e':
        try:
            y = vr_e_res[vr]
        except KeyError:
            print(f'error with {var_dict[comp]['label']}')
    if comp == 'l':
        try:
            y = vr_l_res[vr]
        except KeyError:
            print(f'error with {var_dict[comp]['label']}')
    if comp == 't':
        try:
            y = vr_t_res[vr]
        except KeyError:
            print(f'error with {var_dict[comp]['label']}')
    if comp == 't1':
        try:
            y = vr_t1_res[vr]
        except KeyError:
            print(f'error with {var_dict[comp]['label']}')
    if comp == 't2':
        if vr == 'alb':
            var_dict['t2']['label'] = 'ERA5 snow alb'
        elif vr == 'iwv':
            var_dict['t2']['label'] = 'RS'
        else:
            var_dict['t2']['label'] = 'AWS ECAPAC'
        try:
            y = vr_t2_res[vr]
        except KeyError:
            print(f'error with {var_dict[comp]['label']}')

    return y, vr_t_res
