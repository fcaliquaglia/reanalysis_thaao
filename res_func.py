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


def data_resampling():
    for vvrr in list(inpt.var_dict.keys()):
        try:
            inpt.extr[inpt.var][vvrr]['data_res'] = inpt.extr[inpt.var][vvrr]['data'].resample(inpt.tres).mean()
        except (TypeError, NameError):
            pass

    return
