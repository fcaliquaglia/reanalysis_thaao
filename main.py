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
import read_funcs as rd_funcs
import res_func as rs_f


def main():
    print(inpt.location, end="\n\n")
    for tres in inpt.tres_list:
        inpt.tres = tres
        print(f"Processing time resolution: {tres}")

        for var in inpt.list_var:
            inpt.var = var
            print(f"**************Processing variable: {var}**************\n")

            # Read data
            rd_funcs.read()

            # Resample data (special handling for wind and precipitation)
            rs_f.data_resampling(inpt.var)

            # Plotting
            if not inpt.datasets['dropsondes']['switch']:
                plt_f.plot_ts('all')
                plt_f.plot_residuals('all')
                for seas in inpt.seass:
                    plt_f.plot_scatter(seas)

            plt_f.plot_scatter_cum()


if __name__ == "__main__":
    main()
