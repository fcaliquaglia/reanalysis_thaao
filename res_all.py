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
    if inpt.var in ['winds', 'windd', 'precip']:
        print('NO WIND/PRECIP RESAMPLING!')
        sys.exit()

    for vvrr in inpt.extr[vr]['comps'] + [inpt.extr[vr]['ref_x']]:
        try:
            data_res = inpt.extr[vr][vvrr]['data'].resample(inpt.tres).mean()
            inpt.extr[vr][vvrr]['data_res'] = data_res
        except (TypeError, NameError):
            pass

    return
