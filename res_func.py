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
from inputs import *


def data_resampling():
    vr = inpt.var_in_use
    vr_data = inpt.var_in_use[vr]

    for vvrr in list(var_dict.keys()):
        if inpt.var_in_use == 'iwv' & vvrr == 't2':
            try:
                vr_data[vvrr]['data_res'] = vr_data[vvrr]['data'].resample(tres).mean()
            except (TypeError, NameError):
                pass
        else:
            try:
                vr_data[vvrr]['data_res'] = vr_data[vvrr]['data'].resample(tres).mean()
            except (TypeError, NameError):
                pass

    return
