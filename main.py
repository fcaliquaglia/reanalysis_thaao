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

from plot_func import plot_residuals, plot_scatter, plot_scatter_cum, plot_ts
from read_func import *
from inputs import *
import inputs as inpt
from res_func import data_resampling

if __name__ == "__main__":

    for var in var_list:
        inpt.var_in_use = var
        print(var)

        # data reading
        read()

        # time RESAMPLING (specific for windd --> using wind components, and precip --> cumulative)
        data_resampling()

        plot_ts('all')
        plot_residuals('all')
        plot_scatter_cum()
        for seas in seass:
            plot_scatter(seas)
            # plot_ba(var, all_var, seas)
