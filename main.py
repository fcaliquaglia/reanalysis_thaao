#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for reading, processing, and plotting climate reanalysis data.

Author:
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


def process_variable(var, tres):
    """Reads, processes, and plots data for a single variable at a given time resolution."""
    inpt.var = var

    print("\n----------------------------")
    print(f"Variable:        {var}")
    print(f"Time Resolution: {tres}")
    print("----------------------------")

    rd_funcs.read()
    rs_f.data_resampling(var)

    plt_f.plot_ts('all')
    plt_f.plot_residuals('all')
    plt_f.plot_scatter_cum()

    if not inpt.datasets.get('dropsondes', {}).get('switch', False):
        plt_f.plot_ts('all')
        plt_f.plot_residuals('all')
        plt_f.plot_ba('all')
        plt_f.plot_scatter_all('all')
        for season in inpt.seasons:
            plt_f.plot_scatter_seasonal(season)
            plt_f.plot_ts(season)
            plt_f.plot_residuals(season)


def main():
    print(f"\n\n=== Location: {inpt.location} ===\n")

    for tres in inpt.tres_list:
        inpt.tres = tres
        print(f"\n=== Processing Time Resolution: {tres} ===\n\n")

        for var in inpt.list_var:
            process_variable(var, tres)

    # Taylor Diagrams
    if inpt.datasets.get('THAAO', {}).get('switch', True):
       plt_f.plot_taylor('met')
       plt_f.plot_taylor('rad_comps')
       plt_f.plot_taylor('rad_flux')


if __name__ == "__main__":
    main()
