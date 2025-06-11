#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for reading, processing, and plotting climate reanalysis data.

Authors:
    Filippo Cali' Quaglia (filippo.caliquaglia@ingv.it)

Affiliation:
    INGV

License:
    GPL

Version:
    0.1
"""

__author__ = "Filippo Cali' Quaglia"
__credits__ = []
__license__ = "GPL"
__version__ = "0.1"
__email__ = "filippo.caliquaglia@ingv.it"
__status__ = "Research"
__lastupdate__ = ""

import inputs as inpt
import plot_func as plt_f
import read_func as rd_f
import res_func as rs_f
import grid_selection as grid_sel


def main():
    for tres in inpt.tres_list:
        inpt.tres = tres
        print(f"Processing time resolution: {tres}")

        for var in inpt.list_var:
            inpt.var = var
            print(f"Processing variable: {var}")

            # Selec relevant pixel for each dataset/reanalyses
            grid_sel.pixel_sel()
            
            # Read data
            rd_f.read()

            # Resample data (special handling for wind and precipitation)
            rs_f.data_resampling(inpt.var)

            # Plotting
            plt_f.plot_ts('all')
            plt_f.plot_residuals('all')
            plt_f.plot_scatter_cum()
            for seas in inpt.seass:
                plt_f.plot_scatter(seas)


if __name__ == "__main__":
    main()
