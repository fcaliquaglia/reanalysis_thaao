# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 08:50:17 2025

@author: FCQ
"""


import os
import numpy as np
import inputs as inpt
import datetime as dt
import csv


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


def smart_formatter(x, _):
    if x == 0:
        return "0"
    elif abs(x) >= 1:
        return f"{x:.0f}"
    else:
        return f"{x:.{abs(int(np.floor(np.log10(abs(x)))))}f}"


def frame_and_axis_removal(axs, n_plots):
    """
    Hide extra axes beyond the number of plots needed.

    :param axs: Flattened array of matplotlib axes.
    :param n_plots: Number of actual plots to show.
    """
    for i in range(n_plots, len(axs)):
        axs[i].set_visible(False)


def get_color_by_resolution(base_color, resolution):
    blues = ['#0000FF',  # Blue
             '#1E90FF',  # Dodger Blue
             '#00BFFF',  # Deep Sky Blue
             '#87CEFA',  # Light Sky Blue
             '#ADD8E6']  # Light Blue
    reds = ['#FF0000',   # Red
            '#FF4500',   # Orange Red
            '#FF6347',   # Tomato
            '#F08080',   # Light Coral
            '#FFE4E1']   # Misty Rose

    try:
        idx = inpt.tres_list.index(resolution)
    except ValueError:
        idx = -1

    base_color = base_color.lower() if isinstance(base_color, str) else 'gray'

    if base_color == 'blue':
        return blues[idx] if idx > 0 else 'blue'
    elif base_color == 'red':
        return reds[idx] if idx > 0 else 'red'


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

    fn = f"{tr}_{data_typ}_stats_{inpt.var}.csv"
    if print_stats:
        stats_path = os.path.join(inpt.basefol['out']['base'], 'stats', tr, fn)
        if os.path.exists(stats_path):
            read_stats_from_csv(fn, data_typ, tr, ref_x)
        else:
            calc_stats(xx, yy, data_typ, tr, fn)

        stats = inpt.extr[inpt.var][data_typ]['data_stats'][tr]
        ref_stats = inpt.extr[inpt.var][ref_x]['data_stats'][tr]
        r2 = stats['r2']
        N = stats['N']
        rmse = stats['rmse']
        mbe = stats['mbe']
        std_x = ref_stats['std_x']
        std_y = stats['std_y']
        KL_bits = stats['kl_bits']

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


def read_stats_from_csv(fn, data_typ, tr, ref_x):
    stats_path = os.path.join(inpt.basefol['out']['base'], 'stats', tr, fn)

    with open(stats_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            var = row['Variable']
            tr = row['T_Res']

            # Initialize nested structure if it doesn't exist
            for key in [data_typ, ref_x]:
                if 'data_stats' not in inpt.extr[var][key]:
                    inpt.extr[var][key]['data_stats'] = {}
                if tr not in inpt.extr[var][key]['data_stats']:
                    inpt.extr[var][key]['data_stats'][tr] = {}

            # Assign stats
            inpt.extr[var][data_typ]['data_stats'][tr]['r2'] = float(row['r2'])
            inpt.extr[var][data_typ]['data_stats'][tr]['N'] = int(row['N'])
            inpt.extr[var][data_typ]['data_stats'][tr]['rmse'] = float(
                row['rmse'])
            inpt.extr[var][data_typ]['data_stats'][tr]['mbe'] = float(
                row['mbe'])
            inpt.extr[var][data_typ]['data_stats'][tr]['std_y'] = float(
                row['std_y'])
            inpt.extr[var][data_typ]['data_stats'][tr]['kl_bits'] = float(
                row['KL_bits'])

            inpt.extr[var][ref_x]['data_stats'][tr]['std_x'] = float(
                row['std_x'])

    print(f"✅ Stats loaded from {stats_path}")


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
    # Compute KL divergences
    P = np.array(var_data[ref_x]['data_marg_distr']
                 [tres_ref_x][inpt.var]*var_data['bin_size'])
    Q = np.array(var_data[data_typ]['data_marg_distr']
                 [tres][inpt.var]*var_data['bin_size'])

    var_data[data_typ]['data_stats'][tr]['kl_bits'] = kl_divergence(
        P, Q)/np.log(2)
    save_stats(fn, data_typ, tr, ref_x)

    return (inpt.extr[inpt.var][data_typ]['data_stats'][tr]['r2'], inpt.extr[inpt.var][data_typ]['data_stats'][tr]['N'], inpt.extr[inpt.var][data_typ]['data_stats'][tr]['rmse'], inpt.extr[inpt.var][data_typ]['data_stats'][tr]['mbe'], inpt.extr[inpt.var][ref_x]['data_stats'][tr]['std_x'],  inpt.extr[inpt.var][data_typ]['data_stats'][tr]['std_y'], inpt.extr[inpt.var][data_typ]['data_stats'][tr]['kl_bits'])


def save_stats(fn, data_typ, tr, ref_x):
    out_dir = os.path.join(inpt.basefol['out']['base'], 'stats', tr)
    os.makedirs(out_dir, exist_ok=True)
    stats_file = os.path.join(out_dir, f"{fn}")

    with open(stats_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['Variable', 'Model_Obs', 'T_Res',
                         'r2', 'N', 'rmse', 'mbe', 'std_x', 'std_y', 'KL_bits'])

        ref_stats = inpt.extr[inpt.var][ref_x]['data_stats'][tr]
        var_stats = inpt.extr[inpt.var][data_typ]['data_stats'][tr]
        writer.writerow([
            inpt.var,
            inpt.var_dict[data_typ]['label'],
            tr,
            f"{var_stats['r2']:.4f}",
            f"{var_stats['N']}",
            f"{var_stats['rmse']:.4f}",
            f"{var_stats['mbe']:.4f}",
            f"{ref_stats['std_x']:.4f}",
            f"{var_stats['std_y']:.4f}",
            f"{var_stats['kl_bits']:.4f}"
        ])

        print(f"✅ Stats saved to {stats_file}")


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
