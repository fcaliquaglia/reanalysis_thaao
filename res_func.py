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

import sys

import inputs as inpt


def data_resampling(vr):
    if inpt.var in['winds', 'windd', 'precip']:
        print('NO WIND/PRECIP RESAMPLING!')
        sys.exit()

    if vr in ['lw_up','sw_up']:
        vr_list = ['lw_up', 'sw_up']
    else:
        vr_list = [inpt.var]

    for (vvrr, rrvv) in zip(list(inpt.var_dict.keys()), vr_list):
        try:
            inpt.extr[rrvv][vvrr]['data_res'] = inpt.extr[rrvv][vvrr]['data'].resample(inpt.tres).mean()
        except (TypeError, NameError):
            pass

    return
