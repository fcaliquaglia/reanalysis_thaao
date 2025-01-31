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
import plot_func as plt_f
import read_func as rd_f
import res_func as rs_f

if __name__ == "__main__":
    for tres in inpt.tres_list:
        inpt.tres = tres
        print(tres)

        for var in inpt.list_var:
            inpt.var = var
            print(var)

            # data reading
            rd_f.read()

            # time RESAMPLING (specific for windd --> using wind components, and precip --> cumulative)
            rs_f.data_resampling(inpt.var)

            plt_f.plot_ts('all')
            plt_f.plot_residuals('all')
            plt_f.plot_scatter_cum()
            for seas in inpt.seass:
                plt_f.plot_scatter(seas)  # plot_ba(var, all_var, seas)
