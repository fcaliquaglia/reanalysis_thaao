#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Brief description
"""

__author__ = "Filippo Cali' Quaglia"
__credits__ = ["??????"]
__license__ = "GPL"
__version__ = "0.1"
__email__ = "filippo.caliquaglia@ingv.it"
__status__ = "Research"
__lastupdate__ = ""

import os
import numpy as np
import pandas as pd

import read_func_rean as rd_frea
import read_func_thaao as rd_ft
import read_func_villum as rd_fv
import read_func_sigmaa as rd_fsa
import read_func_sigmab as rd_fsb

from metpy.calc import wind_components
from metpy.units import units
from metpy.constants import g

import inputs as inpt
import tools as tls


def read_alb():
    """
    Reads and processes shortwave upward radiation data from CARRA1, ERA5, ERA5-LAND, and THAAO.
    Applies unit conversions, calculates upwelling radiation, and filters invalid values.
    Modifies `inpt` in-place.
    """

    # both CARRA1 AND ERA5
    vr = 'sw_down'
    var_dict = inpt.extr[vr]
    read_sw_down()
    sw_down_c = var_dict["c"]["data"][vr]

    sw_down_e = var_dict["e"]["data"][vr]

    # --- CARRA1 ALB ---
    vr = "sw_net"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    sw_net_c = var_dict["c"]["data"][vr]
    sw_net_c /= pd.Timedelta('1h').total_seconds()

    vr = "sw_up"
    var_dict = inpt.extr[vr]
    # CARRA/ERA5: `net` already includes sign (net = up + down), so
    # up = net - down
    sw_up_c = sw_down_c - sw_net_c
    sw_up_c = sw_up_c.mask(sw_up_c < inpt.rad_low_thresh, np.nan)
    sw_up_c.name = vr
    var_dict["c"]["data"] = pd.DataFrame({vr: sw_up_c})
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

    # --- ERA5 ALB ---
    vr = "sw_net"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    sw_net_e = var_dict["e"]["data"][vr]
    sw_net_e /= pd.Timedelta('1h').total_seconds()

    vr = "sw_up"
    var_dict = inpt.extr[vr]
    # CARRA/ERA5: `net` already includes sign (net = up + down), so
    # up = net - down
    sw_up_e = sw_down_e - sw_net_e
    sw_up_e = sw_up_e.mask(sw_up_e < inpt.rad_low_thresh, np.nan)
    sw_up_e.name = vr
    var_dict["e"]["data"] = pd.DataFrame({vr: sw_up_e})
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)

    # --- THAAO ALB ---
    vr = "sw_up"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        sw_up_t = var_dict["t"]["data"][vr].mask(
            var_dict["t"]["data"][vr] < inpt.rad_low_thresh, np.nan)
        var_dict["t"]["data"][vr] = sw_up_t

    vr = "sw_down"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        sw_down_t = var_dict["t"]["data"][vr].mask(
            var_dict["t"]["data"][vr] < inpt.rad_low_thresh, np.nan)
        var_dict["t"]["data"][vr] = sw_down_t

    # ALB
    vr = "alb"
    var_dict = inpt.extr[vr]

    alb_c_tmp = sw_up_c/sw_down_c
    alb_c = alb_c_tmp.where((alb_c_tmp >= 0.01) & (alb_c_tmp <= .99))
    alb_c.name = vr
    alb_e_tmp = sw_up_e/sw_down_e
    alb_e = alb_e_tmp.where((alb_e_tmp >= 0.01) & (alb_e_tmp <= 0.99))
    alb_e.name = vr
    alb_t_tmp = sw_up_t/sw_down_t
    alb_t = alb_t_tmp.where((alb_t_tmp >= 0.01) & (alb_t_tmp <= 0.99))
    alb_t.name = vr

    var_dict["c"]["data"] = pd.DataFrame({vr: alb_c})
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

    var_dict["e"]["data"] = pd.DataFrame({vr: alb_e})
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)

    var_dict["t"]["data"] = pd.DataFrame({vr: alb_t})
    var_dict["t"]["data"], _ = tls.check_empty_df(var_dict["t"]["data"], vr)

    return


def read_cbh():
    """
    Reads cloud base height (CBH) data from multiple sources and processes it
    based on the specified input variable. Supports data from CARRA1, ERA5, and THAAO ceilometer.

    :raises ValueError: If the specified input variable is invalid or unsupported.
    :return: None
    """
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA1---
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    var_dict["c"]["data"][vr] += inpt.carra1_ground_elev
    var_dict["c"]["data"][vr] = var_dict["c"]["data"][vr].mask(
        var_dict["c"]["data"][vr] < inpt.cbh_low_thresh, np.nan)

    # --- ERA5 ---
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    var_dict["e"]["data"][vr] += inpt.era5_ground_elev
    var_dict["e"]["data"][vr] = var_dict["e"]["data"][vr].mask(
        var_dict["e"]["data"][vr] < inpt.cbh_low_thresh, np.nan)

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_ceilometer(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        var_dict["t"]["data"][vr] += inpt.thaao_ground_elev
        var_dict["t"]["data"][vr] = var_dict["t"]["data"][vr].mask(
            var_dict["t"]["data"][vr] < inpt.cbh_low_thresh, np.nan)

    return


def read_iwv():
    """
    Reads integrated water vapor (IWV) data from multiple sources including:
    CARRA1, ERA5, THAAO (VESPA and HATPRO), and radiosonde measurements.

    Each dataset is loaded, cleaned, and converted into a standardized
    pandas DataFrame. Returns data from each source as a list.

    :return: [CARRA1, ERA5, VESPA (t), HATPRO (t1), Radiosondes (t2)]
    """
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA1---
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

    # --- ERA5 ---
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)

    # --- THAAO (VESPA) ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_iwv_vespa(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- THAAO (HATPRO) ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_hatpro(vr)
        var_dict["t1"]["data"], _ = tls.check_empty_df(
            var_dict["t1"]["data"], vr)


def read_lwp():
    """
    Reads cloud liquid water path (LWP) data from CARRA1, ERA5, and THAAO1,
    applies cleaning conditions (setting values < 0.01 to NaN), and updates datasets.
    """
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA1---
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    var_dict["c"]["data"][vr] *= 1000.
    lwp_c = var_dict["c"]["data"][vr]
    # lwp_c[lwp_c < 5] = np.nan
    # lwp_c[lwp_c > 1000] = np.nan
    var_dict["c"]["data"][vr] = lwp_c

    # --- ERA5 ---
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    var_dict["e"]["data"][vr] *= 1000.
    lwp_e = var_dict["e"]["data"][vr]
    # lwp_e[lwp_e < 5] = np.nan
    # lwp_e[lwp_e > 1000] = np.nan
    var_dict["e"]["data"][vr] = lwp_e

    # --- THAAO (HATPRO) ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_hatpro(vr)
        var_dict["t1"]["data"], _ = tls.check_empty_df(
            var_dict["t1"]["data"], vr)
        lwp_t1 = var_dict["t1"]["data"][vr]
        # lwp_t1[lwp_t1 < 5] = np.nan
        # lwp_t1[lwp_t1 > 1000] = np.nan
        var_dict["t1"]["data"][vr] = lwp_t1

    return


def read_lw_down():
    """
    Reads and processes longwave downward radiation ("lw_down") data from CARRA1, ERA5, and THAAO.
    Filters out negative values by setting them to NaN, then applies radiation conversion factors.
    """

    # --- CARRA1---
    vr = "lw_down"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    var_dict["c"]["data"][vr] /= pd.Timedelta('1h').total_seconds()

    # --- ERA5 ---
    vr = "lw_down"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    var_dict["e"]["data"][vr] /= pd.Timedelta('1h').total_seconds()

    # --- THAAO ---
    vr = "lw_down"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-A ---
    if inpt.datasets['Sigma-A']['switch']:
        rd_fsa.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-B ---
    if inpt.datasets['Sigma-B']['switch']:
        rd_fsb.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    return


def read_lw_net():
    """
    Reads and processes longwave upwelling radiation data from multiple sources (CARRA1, ERA5, THAAO).
    Performs unit conversions, calculates upward LW radiation from net/downward components,
    and handles invalid values.
    """

    # both CARRA1 AND ERA5
    # --- CARRA1 ---
    vr = "lw_net"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    lw_net_c = var_dict["c"]["data"][vr]
    lw_net_c /= pd.Timedelta('1h').total_seconds()

    # --- ERA5 ---
    vr = "lw_net"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    lw_net_e = var_dict["e"]["data"][vr]
    lw_net_e /= pd.Timedelta('1h').total_seconds()

    # --- THAAO ---
    vr = "lw_up"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        lw_up = var_dict["t"]["data"][vr]
        var_dict["t"]["data"][vr] = lw_up

    vr = "lw_down"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        lw_down = var_dict["t"]["data"][vr]

    vr = "lw_net"
    var_dict = inpt.extr[vr]
    var_dict["t"]["data"], _ = tls.check_empty_df(
        var_dict["t"]["data"], vr)
    var_dict["t"]["data"][vr] = lw_down - lw_up

    return


def read_lw_up():
    """
    Reads and processes longwave upwelling radiation data from multiple sources (CARRA1, ERA5, THAAO).
    Performs unit conversions, calculates upward LW radiation from net/downward components,
    and handles invalid values.
    """

    # both CARRA1 AND ERA5
    vr = 'lw_down'
    var_dict = inpt.extr[vr]
    read_lw_down()
    lw_down_c = var_dict["c"]["data"][vr]
    lw_down_e = var_dict["e"]["data"][vr]

    # --- CARRA1 ---
    vr = "lw_net"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    lw_net_c = var_dict["c"]["data"][vr]
    lw_net_c /= pd.Timedelta('1h').total_seconds()

    vr = "lw_up"
    var_dict = inpt.extr[vr]
    # CARRA/ERA5: `lw_net` includes sign as net = up + down, therefore
    # up = net - down
    lw_up_c = lw_down_c - lw_net_c
    lw_up_c = lw_up_c.mask(lw_up_c < inpt.rad_low_thresh, np.nan)
    lw_up_c.name = vr
    var_dict["c"]["data"] = pd.DataFrame({vr: lw_up_c})
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

    # --- ERA5 ---
    vr = "lw_net"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    lw_net_e = var_dict["e"]["data"][vr]
    lw_net_e /= pd.Timedelta('1h').total_seconds()

    vr = "lw_up"
    var_dict = inpt.extr[vr]
    # CARRA/ERA5: `lw_net` includes sign as net = up + down, therefore
    # up = net - down
    lw_up_e = lw_down_e - lw_net_e
    lw_up_e = lw_up_e.mask(lw_up_e < inpt.rad_low_thresh, np.nan)
    lw_up_e.name = vr
    var_dict["e"]["data"] = pd.DataFrame({vr: lw_up_e})
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)

    # --- THAAO ---
    vr = "lw_up"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        lw_up_t = var_dict["t"]["data"][vr].mask(
            var_dict["t"]["data"][vr] < inpt.rad_low_thresh, np.nan)
        var_dict["t"]["data"][vr] = lw_up_t

    # --- Sigma-A ---
    if inpt.datasets['Sigma-A']['switch']:
        rd_fsa.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-B ---
    if inpt.datasets['Sigma-B']['switch']:
        rd_fsb.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    return


def read_orog():
    """
    """
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA1 ---
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

    # --- ERA5 ---
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    var_dict["e"]["data"][vr] /= g

    return


def read_precip():
    """
    Reads and processes precipitation data from CARRA1, ERA5, and THAAO2 datasets.
    Scales ERA5 precipitation from meters to millimeters. Modifies `inpt` in place.
    """
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA1 ---
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    var_dict["c"]["data"][vr] = var_dict["c"]["data"][vr].mask(
        var_dict["c"]["data"][vr] <= inpt.precip_low_thresh, np.nan)

    # --- ERA5 ---
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    var_dict["e"]["data"][vr] *= 1000
    var_dict["e"]["data"][vr] = var_dict["e"]["data"][vr].mask(
        var_dict["e"]["data"][vr] <= inpt.precip_low_thresh, np.nan)

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_aws_ecapac(vr)
        var_dict["t2"]["data"], _ = tls.check_empty_df(
            var_dict["t2"]["data"], vr)
    var_dict["t2"]["data"][vr] = var_dict["t2"]["data"][vr].mask(
        var_dict["t2"]["data"][vr] <= inpt.precip_low_thresh, np.nan)
    return


def read_rh():
    """
    Reads and processes relative humidity data from CARRA1, ERA5, and THAAO2 datasets.
    If RH is not directly available, it is calculated from temperature and dew point.

    :raises KeyError: if data keys are missing.
    :raises ValueError: if unexpected data structure issues arise.
    """
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA1 ---
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

    # --- ERA5 ---
    rd_frea.read_rean("dewpt", "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    rd_frea.read_rean("temp", "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    tls.calc_rh_from_tdp()  # Compute RH from dew point and temp

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        rd_ft.read_aws_ecapac(vr)
        var_dict["t2"]["data"], _ = tls.check_empty_df(
            var_dict["t2"]["data"], vr)

    # --- Villum ---
    if inpt.datasets['Villum']['switch']:
        rd_fv.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-A ---
    if inpt.datasets['Sigma-A']['switch']:
        rd_fsa.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-B ---
    if inpt.datasets['Sigma-B']['switch']:
        rd_fsb.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Alert ---
    if inpt.datasets['Alert']['switch']:
        rd_fv.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Summit ---
    if inpt.datasets['Summit']['switch']:
        rd_fv.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    return


def read_surf_pres():
    """
    Reads surface pressure data from CARRA1, ERA5, THAAO, and THAAO2.
    Converts units, filters out invalid data, and handles known corrupted periods.
    Modifies `inpt` in-place.
    """
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA1 ---
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    var_dict["c"]["data"][vr] /= 100.
    var_dict["c"]["data"].loc[var_dict["c"]["data"][vr] <= 900., vr] = np.nan
    var_dict["c"]["data"].loc[var_dict["c"]["data"][vr] >= 1050., vr] = np.nan

    # --- ERA5 ---
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    var_dict["e"]["data"][vr] /= 100.
    var_dict["e"]["data"].loc[var_dict["e"]["data"][vr] <= 900., vr] = np.nan
    var_dict["e"]["data"].loc[var_dict["e"]["data"][vr] >= 1050., vr] = np.nan

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        pres_t = var_dict["t"]["data"][vr]
        pres_t[pres_t <= 900.] = np.nan
        pres_t[pres_t >= 1050.] = np.nan
        pres_t.loc["2021-10-11 00:00:00":"2021-10-19 00:00:00"] = np.nan
        pres_t.loc["2024-04-26 00:00:00":"2024-05-04 00:00:00"] = np.nan
        var_dict["t"]["data"][vr] = pres_t

        rd_ft.read_aws_ecapac(vr)
        var_dict["t2"]["data"], _ = tls.check_empty_df(
            var_dict["t2"]["data"], vr)

    # --- Villum ---
    if inpt.datasets['Villum']['switch']:
        rd_fv.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-A ---
    if inpt.datasets['Sigma-A']['switch']:
        rd_fsa.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-B ---
    if inpt.datasets['Sigma-B']['switch']:
        rd_fsb.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
    # --- Alert ---
    if inpt.datasets['Alert']['switch']:
        print('No data available for Alert')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Summit ---
    if inpt.datasets['Summit']['switch']:
        print('No data available for Summit')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Buoys ---
    if inpt.datasets['buoys']['switch']:
        data_all = pd.DataFrame()
        for y in inpt.years:
            path = os.path.join('txt_locations', f"{inpt.location}_loc.txt")
            data_tmp = pd.read_csv(path)
            data_all = pd.concat([data_all, data_tmp])
        data_all['time'] = pd.to_datetime(data_all['time'], errors='coerce')
        data_all = data_all.set_index('time')
        var_dict["t"]["data"] = data_all
        var_dict["t"]["data"][inpt.var] = var_dict["t"]["data"][inpt.var].mask(
            var_dict["t"]["data"][inpt.var] == 0.0, np.nan)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    return


def read_sw_down():
    """
    Reads and processes shortwave downward radiation data from CARRA1, ERA5, ERA5-LAND, and THAAO.
    Negative values are set to NaN, and values are scaled using radiation conversion factors.
    """

    # --- CARRA1 ---
    vr = "sw_down"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    var_dict["c"]["data"][vr] /= pd.Timedelta('1h').total_seconds()
    var_dict["c"]["data"][vr] = var_dict["c"]["data"][vr].mask(
        var_dict["c"]["data"][vr] < inpt.rad_low_thresh, np.nan)

    # --- ERA5 ---
    vr = "sw_down"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    var_dict["e"]["data"][vr] /= pd.Timedelta('1h').total_seconds()
    var_dict["e"]["data"][vr] = var_dict["e"]["data"][vr].mask(
        var_dict["e"]["data"][vr] < inpt.rad_low_thresh, np.nan)

    # --- THAAO ---
    vr = "sw_down"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        sw_down_t = var_dict["t"]["data"][vr].mask(
            var_dict["t"]["data"][vr] < inpt.rad_low_thresh, np.nan)
        var_dict["t"]["data"][vr] = sw_down_t

    # --- Villum ---
    if inpt.datasets['Villum']['switch']:
        rd_fv.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-A ---
    if inpt.datasets['Sigma-A']['switch']:
        rd_fsa.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-B ---
    if inpt.datasets['Sigma-B']['switch']:
        rd_fsb.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Alert ---
    if inpt.datasets['Alert']['switch']:
        print('No data available for Alert')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Summit ---
    if inpt.datasets['Summit']['switch']:
        print('No data available for Summit')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Buoys ---
    if inpt.datasets['buoys']['switch']:
        data_all = pd.DataFrame()
        for y in inpt.years:
            path = os.path.join('txt_locations', f"{inpt.location}_loc.txt")
            data_tmp = pd.read_csv(path)
            data_all = pd.concat([data_all, data_tmp])
        data_all['time'] = pd.to_datetime(data_all['time'], errors='coerce')
        data_all = data_all.set_index('time')
        var_dict["t"]["data"] = data_all
        var_dict["t"]["data"][inpt.var] = var_dict["t"]["data"][inpt.var].mask(
            var_dict["t"]["data"][inpt.var] == 0.0, np.nan)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    return


def read_sw_lw_net():
    """
    Reads and processes shortwave upward radiation data from CARRA1, ERA5, ERA5-LAND, and THAAO.
    Applies unit conversions, calculates upwelling radiation, and filters invalid values.
    Modifies `inpt` in-place.
    """

    # --- CARRA1 SW_NET ---
    vr = "sw_net"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    sw_net_c = var_dict["c"]["data"][vr] / pd.Timedelta('1h').total_seconds()
    sw_net_c = sw_net_c.mask(sw_net_c < inpt.rad_low_thresh, np.nan)
    var_dict["c"]["data"][vr] = sw_net_c

    # --- CARRA1 LW_NET ---
    vr = "lw_net"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    lw_net_c = var_dict["c"]["data"][vr] / pd.Timedelta('1h').total_seconds()
    var_dict["c"]["data"][vr] = lw_net_c

    # --- ERA5 SW_NET ---
    vr = "sw_net"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    sw_net_e = var_dict["e"]["data"][vr] / pd.Timedelta('1h').total_seconds()
    sw_net_e = sw_net_e.mask(sw_net_e < inpt.rad_low_thresh, np.nan)
    var_dict["e"]["data"][vr] = sw_net_e

    # --- ERA5 LW_NET ---
    vr = "lw_net"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    lw_net_e = var_dict["e"]["data"][vr] / pd.Timedelta('1h').total_seconds()
    var_dict["e"]["data"][vr] = lw_net_e

    # --- THAAO SW_NET ---
    vr = "sw_up"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        sw_up_t = var_dict["t"]["data"][vr].mask(
            var_dict["t"]["data"][vr] < inpt.rad_low_thresh, np.nan)
        var_dict["t"]["data"][vr] = sw_up_t

    vr = "sw_down"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        sw_down_t = var_dict["t"]["data"][vr].mask(
            var_dict["t"]["data"][vr] < inpt.rad_low_thresh, np.nan)
        var_dict["t"]["data"][vr] = sw_down_t

    vr = "sw_net"
    var_dict = inpt.extr[vr]
    var_dict["t"]["data"], _ = tls.check_empty_df(
        var_dict["t"]["data"], vr)
    sw_net_t = sw_down_t - sw_up_t
    var_dict["t"]["data"][vr] = sw_net_t

    # --- THAAO LW_NET ---
    vr = "lw_up"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        lw_up_t = var_dict["t"]["data"][vr]
        var_dict["t"]["data"][vr] = lw_up_t

    vr = "lw_down"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        lw_down_t = var_dict["t"]["data"][vr]

    vr = "lw_net"
    var_dict = inpt.extr[vr]
    var_dict["t"]["data"], _ = tls.check_empty_df(
        var_dict["t"]["data"], vr)
    lw_net_t = lw_down_t - lw_up_t
    var_dict["t"]["data"][vr] = lw_net_t

    # SW_LW_NET
    vr = "sw_lw_net"
    var_dict = inpt.extr[vr]
    sw_lw_net_c = sw_net_c.combine(
        lw_net_c, lambda a, b: a if pd.isna(b) else b if pd.isna(a) else a + b)
    sw_lw_net_e = sw_net_e.combine(
        lw_net_e, lambda a, b: a if pd.isna(b) else b if pd.isna(a) else a + b)
    sw_lw_net_t = sw_net_t.combine(
        lw_net_t, lambda a, b: a if pd.isna(b) else b if pd.isna(a) else a + b)

    var_dict["c"]["data"] = pd.DataFrame(sw_lw_net_c, columns=[vr])
    var_dict["e"]["data"] = pd.DataFrame(sw_lw_net_e, columns=[vr])
    var_dict["t"]["data"] = pd.DataFrame(sw_lw_net_t, columns=[vr])

    return


def read_sw_net():
    """
    Reads and processes shortwave upward radiation data from CARRA1, ERA5, ERA5-LAND, and THAAO.
    Applies unit conversions, calculates upwelling radiation, and filters invalid values.
    Modifies `inpt` in-place.
    """

    # --- CARRA1 ---
    vr = "sw_net"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    sw_net_c = var_dict["c"]["data"][vr]
    sw_net_c /= pd.Timedelta('1h').total_seconds()

    # --- ERA5 ---
    vr = "sw_net"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    sw_net_e = var_dict["e"]["data"][vr]
    sw_net_e /= pd.Timedelta('1h').total_seconds()

    # --- THAAO ---
    vr = "sw_up"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        sw_up_t = var_dict["t"]["data"][vr].mask(
            var_dict["t"]["data"][vr] < inpt.rad_low_thresh, np.nan)
        var_dict["t"]["data"][vr] = sw_up_t

    vr = "sw_down"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        sw_down_t = var_dict["t"]["data"][vr].mask(
            var_dict["t"]["data"][vr] < inpt.rad_low_thresh, np.nan)
        var_dict["t"]["data"][vr] = sw_down_t

    vr = "sw_net"
    var_dict = inpt.extr[vr]
    var_dict["t"]["data"], _ = tls.check_empty_df(
        var_dict["t"]["data"], vr)
    sw_net_t = sw_down_t - sw_up_t
    var_dict["t"]["data"][vr] = sw_net_t

    # # --- Sigma-A ---
    # if inpt.datasets['Sigma-A']['switch']:
    #     rd_fsa.read_sigmaa_weather(vr)
    #     var_dict["t"]["data"], _ = tls.check_empty_df(
    #         var_dict["t"]["data"], vr)

    # # --- Sigma-B ---
    # if inpt.datasets['Sigma-B']['switch']:
    #     rd_fsb.read_weather(vr)
    #     var_dict["t"]["data"], _ = tls.check_empty_df(
    #         var_dict["t"]["data"], vr)

    # # --- Buoys ---
    # if inpt.datasets['buoys']['switch']:
    #     data_all = pd.DataFrame()
    #     for y in inpt.years:
    #         path = os.path.join('txt_locations', f"{inpt.location}_loc.txt")
    #         data_tmp = pd.read_csv(path)
    #         data_all = pd.concat([data_all, data_tmp])
    #     data_all['time'] = pd.to_datetime(data_all['time'], errors='coerce')
    #     data_all = data_all.set_index('time')
    #     var_dict["t"]["data"] = data_all
    #     var_dict["t"]["data"][inpt.var] = var_dict["t"]["data"][inpt.var].mask(
    #         var_dict["t"]["data"][inpt.var] == 0.0, np.nan)
    #     var_dict["t"]["data"], _ = tls.check_empty_df(
    #         var_dict["t"]["data"], vr)

    return


def read_sw_up():
    """
    Reads and processes shortwave upward radiation data from CARRA1, ERA5, ERA5-LAND, and THAAO.
    Applies unit conversions, calculates upwelling radiation, and filters invalid values.
    Modifies `inpt` in-place.
    """

    # both CARRA1 AND ERA5
    vr = 'sw_down'
    var_dict = inpt.extr[vr]
    read_sw_down()
    sw_down_c = var_dict["c"]["data"][vr]
    sw_down_e = var_dict["e"]["data"][vr]

    # --- CARRA1 ---
    vr = "sw_net"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    sw_net_c = var_dict["c"]["data"][vr]
    sw_net_c /= pd.Timedelta('1h').total_seconds()

    vr = "sw_up"
    var_dict = inpt.extr[vr]
    sw_up_c = sw_down_c - sw_net_c
    sw_up_c = sw_up_c.mask(sw_up_c < inpt.rad_low_thresh, np.nan)
    sw_up_c.name = vr
    var_dict["c"]["data"] = pd.DataFrame({vr: sw_up_c})
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

    # --- ERA5 ---
    vr = "sw_net"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    sw_net_e = var_dict["e"]["data"][vr]
    sw_net_e /= pd.Timedelta('1h').total_seconds()

    vr = "sw_up"
    var_dict = inpt.extr[vr]
    sw_up_e = sw_down_e - sw_net_e
    sw_up_e = sw_up_e.mask(sw_up_e < inpt.rad_low_thresh, np.nan)
    sw_up_e.name = vr
    var_dict["e"]["data"] = pd.DataFrame({vr: sw_up_e})
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)

    # --- THAAO ---
    vr = "sw_up"
    var_dict = inpt.extr[vr]
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_rad(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        sw_up_t = var_dict["t"]["data"][vr].mask(
            var_dict["t"]["data"][vr] < inpt.rad_low_thresh, np.nan)
        var_dict["t"]["data"][vr] = sw_up_t

    # --- Sigma-A ---
    if inpt.datasets['Sigma-A']['switch']:
        rd_fsa.read_sigmaa_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-B ---
    if inpt.datasets['Sigma-B']['switch']:
        rd_fsb.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Buoys ---
    if inpt.datasets['buoys']['switch']:
        data_all = pd.DataFrame()
        for y in inpt.years:
            path = os.path.join('txt_locations', f"{inpt.location}_loc.txt")
            data_tmp = pd.read_csv(path)
            data_all = pd.concat([data_all, data_tmp])
        data_all['time'] = pd.to_datetime(data_all['time'], errors='coerce')
        data_all = data_all.set_index('time')
        var_dict["t"]["data"] = data_all
        var_dict["t"]["data"][inpt.var] = var_dict["t"]["data"][inpt.var].mask(
            var_dict["t"]["data"][inpt.var] == 0.0, np.nan)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    return


def read_tcc():
    """
    Reads data from multiple sources (CARRA1, ERA5, and THAAO ceilometer) and processes them accordingly.

    This function performs the following:
    - Reads data from the CARRA1 dataset.
    - Reads data from the ERA5 dataset and processes this data by scaling certain values.
    - Reads data from the THAAO ceilometer dataset.
    Finally, the function does not return any data as it modifies `inpt` directly.

    :return: None
    """
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA1 ---
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    var_dict["c"]["data"]['tcc'] = var_dict["c"]["data"]['tcc'].apply(
        tls.percentage_to_okta)

    # --- ERA5 ---
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    var_dict["e"]["data"][vr] *= 100.0
    var_dict["e"]["data"]['tcc'] = var_dict["e"]["data"]['tcc'].apply(
        tls.percentage_to_okta)

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_ceilometer(vr)
        # var_dict["t"]["data"]['tcc'] = var_dict["t"]["data"]['tcc'].apply(tls.okta_to_percentage)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    return


def read_temp():
    """
    Transforms temperature data from various datasets to Celsius and retrieves
    reformatted sources into a structured container. The function processes data
    from the CARRA1, ERA5, ERA5-LAND, THAAO, and THAAO2 datasets, converting values
    from Kelvin to Celsius for consistency.

    :return: None
    """
    vr = inpt.var
    var_dict = inpt.extr[vr]

    # --- CARRA1 ---
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    var_dict["c"]["data"][vr] -= 273.15

    # --- ERA5 ---
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)
    var_dict["e"]["data"][vr] -= 273.15

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_weather(vr)
        var_dict["t"]["data"][vr] -= 273.15
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        rd_ft.read_aws_ecapac(vr)
        var_dict["t2"]["data"], _ = tls.check_empty_df(
            var_dict["t2"]["data"], vr)

    # --- Villum ---
    if inpt.datasets['Villum']['switch']:
        rd_fv.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-A ---
    if inpt.datasets['Sigma-A']['switch']:
        rd_fsa.read_sigmaa_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-B ---
    if inpt.datasets['Sigma-B']['switch']:
        rd_fsb.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Alert ---
    if inpt.datasets['Alert']['switch']:
        print('No data available for Alert')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Summit ---
    if inpt.datasets['Summit']['switch']:
        print('No data available for Summit')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Buoys ---
    if inpt.datasets['buoys']['switch']:
        data_all = pd.DataFrame()
        for y in inpt.years:
            path = os.path.join('txt_locations', f"{inpt.location}_loc.txt")
            data_tmp = pd.read_csv(path)
            data_all = pd.concat([data_all, data_tmp])
        data_all['time'] = pd.to_datetime(data_all['time'], errors='coerce')
        data_all = data_all.set_index('time')
        var_dict["t"]["data"] = data_all
        var_dict["t"]["data"][inpt.var] = var_dict["t"]["data"][inpt.var].mask(
            var_dict["t"]["data"][inpt.var] == 0.0, np.nan)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        # needed for THAAO AWS ECAPAC. It creates an empty dataframe
        var_dict["t2"]["data"], _ = tls.check_empty_df(
            var_dict["t2"]["data"], vr)

    # --- Dropsondes ---
    if inpt.datasets['dropsondes']['switch']:
        data_all = pd.read_parquet(os.path.join(
            'txt_locations', 'dropsondes_surface_level_temp.parquet'))
        data_all.set_index('time', inplace=True)
        data_all.index.name = 'valid_time'
        var_dict["t"]["data"] = data_all
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)
        # needed for THAAO AWS ECAPAC. It creates an empty dataframe
        var_dict["t2"]["data"], _ = tls.check_empty_df(
            var_dict["t2"]["data"], vr)

    return


def read_wind():
    """
    Calculates and populates wind speed and direction data from different
    data sources, including CARRA1, ERA5, ERA5-LAND (currently deactivated),
    and AWS ECAPAC. The method reads data, computes wind speed and direction,
    and stores the processed results back into the `inpt` structure.

    The wind speed is calculated using components (u, v) provided by respective
    datasets. Similarly, the wind direction is derived from the same components
    where applicable. The method currently deactivates ERA5-LAND calculations
    due to unavailable data.

    :param inpt: Input structure that contains extracted data for various
        wind component keys such as "windu" and "windv" from different
        models. The method updates wind speed ("winds") and wind direction
        ("windd") within this structure based on computed results.
        Must already contain placeholders for updated wind speed and direction
        data for respective datasets.
    :type inpt: object

    :return: Updates the `inpt` structure in place with calculated wind speed
        ("winds") and wind direction ("windd") for available datasets.
    :rtype: None
    """
    start_date = '2023-01-01'
    # --- CARRA1 ---
    vr = "winds"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"] = var_dict["c"]["data"][var_dict["c"]["data"].index >= start_date]
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)
    vr = "windd"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "c")
    var_dict["c"]["data"] = var_dict["c"]["data"][var_dict["c"]["data"].index >= start_date]
    var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

    windu, windv=wind_components(inpt.extr["winds"]["c"]["data"]["winds"].values * units("m/s"), inpt.extr["windd"]["c"]["data"]["windd"].values * units("degrees"))
    inpt.extr["windu"]["c"]["data"] = pd.DataFrame(index=inpt.extr["winds"]["c"]["data"]["winds"].index, data=windu.magnitude, columns=["windu"])
    inpt.extr["windv"]["c"]["data"] = pd.DataFrame(index=inpt.extr["winds"]["c"]["data"]["winds"].index, data=windv.magnitude, columns=["windv"])

    # --- ERA5 ---
    vr = "windu"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"] = var_dict["e"]["data"][var_dict["e"]["data"].index >= start_date]
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)

    vr = "windv"
    var_dict = inpt.extr[vr]
    rd_frea.read_rean(vr, "e")
    var_dict["e"]["data"] = var_dict["e"]["data"][var_dict["e"]["data"].index >= start_date]
    var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)

    # --- THAAO ---
    if inpt.datasets['THAAO']['switch']:
        rd_ft.read_aws_ecapac("winds")
        var_dict["t2"]["data"], _ = tls.check_empty_df(var_dict["t2"]["data"], vr)
        rd_ft.read_aws_ecapac("windd")
        var_dict["t2"]["data"], _ = tls.check_empty_df(var_dict["t2"]["data"], vr)

        windu, windv=wind_components(inpt.extr["winds"]["t2"]["data"]["winds"].values * units("m/s"), inpt.extr["windd"]["t2"]["data"]["windd"].values * units("degrees"))
        inpt.extr["windu"]["t2"]["data"] = pd.DataFrame(index=inpt.extr["winds"]["t2"]["data"]["winds"].index, data=windu.magnitude, columns=["windu"])
        inpt.extr["windv"]["t2"]["data"] = pd.DataFrame(index=inpt.extr["winds"]["t2"]["data"]["winds"].index, data=windv.magnitude, columns=["windv"])

    # --- Villum ---
    if inpt.datasets['Villum']['switch']:
        rd_fv.read_weather(vr)
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-A ---
    if inpt.datasets['Sigma-A']['switch']:
        print('No data available for Sigma-A')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Sigma-B ---
    if inpt.datasets['Sigma-B']['switch']:
        print('No data available for Sigma-B')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Alert ---
    if inpt.datasets['Alert']['switch']:
        print('No data available for Alert')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    # --- Summit ---
    if inpt.datasets['Summit']['switch']:
        print('No data available for Summit')
        var_dict["t"]["data"], _ = tls.check_empty_df(
            var_dict["t"]["data"], vr)

    return


def read():
    # Map variable names to reader functions
    readers = {
        "alb": read_alb,
        "cbh": read_cbh,
        "iwv": read_iwv,
        "lwp": read_lwp,
        "lw_down": read_lw_down,
        "lw_net": read_lw_net,
        "lw_up": read_lw_up,
        "orog": read_orog,
        "precip": read_precip,
        "rh": read_rh,
        "surf_pres": read_surf_pres,
        "sw_down": read_sw_down,
        "sw_lw_net": read_sw_lw_net,
        "sw_net": read_sw_net,
        "sw_up": read_sw_up,
        "tcc": read_tcc,
        "temp": read_temp,
        "windu": read_wind,
        "windv": read_wind,
    }

    reader_func = readers.get(inpt.var)
    if reader_func is None:
        raise ValueError(
            f"No reader function defined for variable '{inpt.var}'")
    return reader_func()
    def read_windd():
        """
        Reads and processes wind direction data from CARRA1, ERA5, and THAAO.
        """
        vr = inpt.var
        var_dict = inpt.extr[vr]

        # --- CARRA1 ---
        rd_frea.read_rean(vr, "c")
        var_dict["c"]["data"], _ = tls.check_empty_df(var_dict["c"]["data"], vr)

        # --- ERA5 ---
        rd_frea.read_rean(vr, "e")
        var_dict["e"]["data"], _ = tls.check_empty_df(var_dict["e"]["data"], vr)

        # --- THAAO ---
        if inpt.datasets['THAAO']['switch']:
            rd_ft.read_aws_ecapac(vr)
            var_dict["t2"]["data"], _ = tls.check_empty_df(
                var_dict["t2"]["data"], vr)

        return