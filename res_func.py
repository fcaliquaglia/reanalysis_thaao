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

    for vvrr in list(var_dict.keys()):
        try:
            extr[vr][vvrr]['data_res'] = extr[vr][vvrr]['data'].resample(tres).mean()
        except (TypeError, NameError):
            pass

    return
