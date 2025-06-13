from pathlib import Path
import datetime as dt
import julian
import os
import numpy as np
import pandas as pd
import xarray as xr
import inputs as inpt
import tools as tls


def read_thaao_weather(vr):
    path_out, _ = tls.get_common_paths(vr, "VESPA")
    nc_file = Path(inpt.basefol["t"]['base']) / \
        "thaao_aws_vespa" / f"{inpt.extr[vr]['t']['fn']}.nc"

    if os.path.exists(path_out):
        df = pd.read_parquet(path_out)
        inpt.extr[vr]["t"]["data"] = df
        print(f"Loaded {path_out}")
        return

    try:
        df = xr.open_dataset(nc_file, engine="netcdf4").to_dataframe()
        print(f"OK: {nc_file.name}")
        df = df[[inpt.extr[vr]["t"]["column"]]]
        df.columns = [vr]
        df.to_parquet(path_out)
        inpt.extr[vr]["t"]["data"] = df
    except FileNotFoundError:
        print(f"NOT FOUND: {nc_file.name}")


def read_thaao_rad(vr):
    path_out, _ = tls.get_common_paths(vr, "rad")
    t_all = []

    if os.path.exists(path_out):
        df = pd.read_parquet(path_out)
        inpt.extr[vr]["t"]["data"] = df
        print(f"Loaded {path_out}")
        return

    for i in inpt.rad_daterange[inpt.rad_daterange.year.isin(inpt.years)]:
        year = i.year
        file = Path(inpt.basefol["t"]['base']) / "thaao_rad" / \
            f"{inpt.extr[vr]['t']['fn']}{year}_5MIN.dat"
        try:
            df = pd.read_table(file, sep=r"\s+", engine="python")
            jd_base = julian.to_jd(dt.datetime(year - 1, 12, 31))
            times = [julian.from_jd(
                jd_base + jd, fmt="jd").replace(microsecond=0) for jd in df["JDAY_UT"]]
            df.index = pd.DatetimeIndex(times)
            t_all.append(df[[inpt.extr[vr]["t"]["column"]]])
            print(f"OK: {file.name}")
        except FileNotFoundError:
            print(f"NOT FOUND: {file.name}")

    if t_all:
        final_df = pd.concat(t_all)
        final_df.columns = [vr]
        final_df.to_parquet(path_out)
        inpt.extr[vr]["t"]["data"] = final_df


def read_thaao_hatpro(vr):
    path_out, _ = tls.get_common_paths(vr, "HATPRO")
    file = Path(inpt.basefol["t"]['base']) / "thaao_hatpro" / \
        f"{inpt.extr[vr]['t1']['fn']}" / f"{inpt.extr[vr]['t1']['fn']}.DAT"

    if os.path.exists(path_out):
        df = pd.read_parquet(path_out)
        inpt.extr[vr]["t1"]["data"] = df
        print(f"Loaded {path_out}")
        return

    try:
        df = pd.read_table(file, sep=r"\s+", engine="python", header=0, skiprows=9,
                           parse_dates={"datetime": [0, 1]}, index_col="datetime",
                           date_format="%Y-%m-%d %H:%M:%S")
        print(f"OK: {file.name}")
        df = df[[inpt.extr[vr]["t1"]["column"]]]
        df.columns = [vr]
        df.to_parquet(path_out)
        inpt.extr[vr]["t1"]["data"] = df
    except FileNotFoundError:
        print(f"NOT FOUND: {file.name}")


def read_thaao_ceilometer(vr):
    path_out, _ = tls.get_common_paths(vr, "ceil")
    t_all = []

    if os.path.exists(path_out):
        df = pd.read_parquet(path_out)
        inpt.extr[vr]["t"]["data"] = df
        print(f"Loaded {path_out}")
        return

    for i in inpt.ceilometer_daterange[inpt.ceilometer_daterange.year.isin(inpt.years)]:
        date_str = i.strftime("%Y%m%d")
        file = Path(inpt.basefol["t"]['base']) / "thaao_ceilometer" / \
            "medie_tat_rianalisi" / f"{date_str}{inpt.extr[vr]['t']['fn']}.txt"
        try:
            df = pd.read_table(
                file, sep=r"\s+", engine="python", header=0, skiprows=9)
            df[df == inpt.var_dict["t"]["nanval"]] = np.nan
            df.index = pd.to_datetime(
                df["#"] + " " + df["date[y-m-d]time[h:m:s]"], format="%Y-%m-%d %H:%M:%S")
            df = df[[inpt.extr[vr]["t"]["column"]]].astype(float)
            t_all.append(df)
            print(f"OK: {file.name}")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"NOT FOUND: {file.name}")

    if t_all:
        final_df = pd.concat(t_all)
        final_df.columns = [vr]
        final_df.to_parquet(path_out)
        inpt.extr[vr]["t"]["data"] = final_df


def read_thaao_aws_ecapac(vr):
    path_out, _ = tls.get_common_paths(vr, "ECAPAC")
    t_all = []

    if os.path.exists(path_out):
        df = pd.read_parquet(path_out)
        inpt.extr[vr]["t2"]["data"] = df
        print(f"Loaded {path_out}")
        return

    for i in inpt.aws_ecapac_daterange[inpt.aws_ecapac_daterange.year.isin(inpt.years)]:
        date_str = i.strftime("%Y_%m_%d")
        file = Path(inpt.basefol["t"]['base']) / "thaao_ecapac_aws_snow" / "AWS_ECAPAC" / \
            i.strftime("%Y") / \
            f"{inpt.extr[vr]['t2']['fn']}{date_str}_00_00.dat"
        try:
            df = pd.read_csv(file, skiprows=[0, 3], header=0, decimal=".", delimiter=",",
                             engine="python", index_col="TIMESTAMP").iloc[1:]
            df.index = pd.to_datetime(df.index)
            df = df[[inpt.extr[vr]["t2"]["column"]]].astype(float)
            t_all.append(df)
            print(f"OK: {file.name}")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"NOT FOUND: {file.name}")

    if t_all:
        final_df = pd.concat(t_all)
        final_df.columns = [vr]
        final_df.to_parquet(path_out)
        inpt.extr[vr]["t2"]["data"] = final_df

