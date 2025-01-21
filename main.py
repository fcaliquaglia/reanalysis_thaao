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

from plot_func import plot_residuals, plot_scatter, plot_ts
from read_func import *
from inputs import *
from res_func import data_resampling

if __name__ == "__main__":

    for var in var_list:
        print(var)
        var_c = pd.DataFrame()
        var_e = pd.DataFrame()
        var_l = pd.DataFrame()
        var_t = pd.DataFrame()
        var_t1 = pd.DataFrame()
        var_t2 = pd.DataFrame()
        if var in ['lwp']:
            [var_c, var_e, var_l, var_t, var_t1] = read(var)
        elif var in ['temp', 'msl_pres', 'surf_pres', 'rh', 'winds', 'windd', 'precip', 'alb', 'iwv']:
            [var_c, var_e, var_l, var_t, var_t1, var_t2] = read(var)
        else:
            [var_c, var_e, var_l, var_t] = read(var)

        # time RESAMPLING (specific for windd --> using wind components, and precip--> cumulative)
        var_c_res, var_e_res, var_l_res, var_t_res, var_t1_res, var_t2_res = data_resampling(
                var, tres, var_c, var_e, var_l, var_t, var_t1, var_t2)

        all_var = [var_c, var_e, var_l, var_t, var_t1, var_t2, var_c_res, var_e_res, var_l_res, var_t_res, var_t1_res,
                   var_t2_res]
        # plot_ts(var, all_var, 'all')
        # plot_residuals(var, all_var, 'all')
        for seas in seass:
            plot_scatter(var, all_var, seas)
            # plot_ba(var, all_var, seas)
