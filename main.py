#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
Script for data reading, resampling, and plotting.
"""

# =============================================================
# CREATED:
# AFFILIATION: INGV
# AUTHOR: Filippo Cali' Quaglia
# =============================================================

__author__ = "Filippo Cali' Quaglia"
__credits__ = ["??????"]
__license__ = "GPL"
__version__ = "0.1"
__email__ = "filippo.caliquaglia@ingv.it"
__status__ = "Research"
__lastupdate__ = ""

import inputs as inpt
import plot_func as plt_f
import read_all
import res_all

if __name__ == "__main__":
    for inpt.tres in inpt.tres_list:
        print(f"Processing time resolution: {inpt.tres}")

        for inpt.var in inpt.list_var:
            print(f"  Processing variable: {inpt.var}")

            # Read data
            read_all.read()

            # Resample data
            res_all.data_resampling(inpt.var)

            # Generate plots
            plt_f.plot_ts('all')
            plt_f.plot_residuals('all')
            plt_f.plot_scatter_cum()

            # Seasonal scatter plots
            for seas in inpt.seass:
                plt_f.plot_scatter(seas)
