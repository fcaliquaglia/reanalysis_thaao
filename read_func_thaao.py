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

    df_all = pd.DataFrame()
    count = 0
    for year in inpt.years:
        path_out, _ = tls.get_common_paths(vr, year, "VESPA")
        if os.path.exists(path_out):
            df_tmp = pd.read_parquet(path_out)
            df_all = pd.concat([df_all, df_tmp])
            print(f"Loaded {path_out}")
            count += 1

    if count < len(inpt.years):
        nc_file = Path(inpt.basefol["t"]['base']) / \
            "thaao_aws_vespa" / f"{inpt.extr[vr]['t']['fn']}.nc"
        try:
            df = xr.open_dataset(nc_file, engine="netcdf4").to_dataframe()
            print(f"OK: {nc_file.name}")
            df = df[[inpt.extr[vr]["t"]["column"]]].rename(
                columns={inpt.extr[vr]["t"]["column"]: vr})
            df_all = df
        except FileNotFoundError:
            print(f"NOT FOUND: {nc_file.name}")
            df_all = pd.DataFrame(columns=[vr])

    # Save filtered data per year if not empty
    for year in inpt.years:
        df_year = df_all[df_all.index.year == year]
        path_out, _ = tls.get_common_paths(vr, year, "VESPA")
        df_year.to_parquet(path_out)
        print(f"Saved {path_out}")

    # Store in global container
    inpt.extr[vr]["t"]["data"] = df_all


def read_thaao_rad(vr):
    df_all = pd.DataFrame()
    count = 0

    # Step 1: Load all existing yearly parquet files
    for year in inpt.years:
        path_out, _ = tls.get_common_paths(vr, year, "RAD")
        if os.path.exists(path_out):
            df_tmp = pd.read_parquet(path_out)
            df_all = pd.concat([df_all, df_tmp])
            print(f"Loaded {path_out}")
            count += 1

    # Step 2: If some files are missing, process .dat files
    if count < len(inpt.years):
        t_all = []
        for year in inpt.years:
            file = Path(inpt.basefol["t"]['base']) / "thaao_rad" / \
                f"{inpt.extr[vr]['t']['fn']}{year}_5MIN.dat"
            try:
                df = pd.read_table(file, sep=r"\s+", engine="python")
                jd_base = julian.to_jd(dt.datetime(year - 1, 12, 31))
                times = [
                    julian.from_jd(
                        jd_base + jd, fmt="jd").replace(microsecond=0)
                    for jd in df["JDAY_UT"]
                ]
                df.index = pd.DatetimeIndex(times)
                df = df[[inpt.extr[vr]["t"]["column"]]].rename(
                    columns={inpt.extr[vr]["t"]["column"]: vr})
                t_all.append(df)
                print(f"OK: {file.name}")
            except FileNotFoundError:
                print(f"NOT FOUND: {file.name}")

        # Step 3: Combine and save each year's data to .parquet
        if t_all:
            df_all = pd.concat(t_all)
            for year in inpt.years:
                df_year = df_all[df_all.index.year == year]
                if not df_year.empty:
                    path_out, _ = tls.get_common_paths(vr, year, "RAD")
                    df_year.to_parquet(path_out)
                    print(f"Saved {path_out}")

    # Step 4: Store final result
    inpt.extr[vr]["t"]["data"] = df_all


def read_thaao_hatpro(vr):
    df_all = pd.DataFrame()
    count = 0

    # Attempt to load available parquet files per year
    for year in inpt.years:
        path_out, _ = tls.get_common_paths(vr, year, "HATPRO")
        if os.path.exists(path_out):
            df_tmp = pd.read_parquet(path_out)
            df_all = pd.concat([df_all, df_tmp])
            print(f"Loaded {path_out}")
            count += 1

    # If not all years loaded, parse .DAT file
    if count < len(inpt.years):
        file = Path(inpt.basefol["t"]['base']) / "thaao_hatpro" / \
            f"{inpt.extr[vr]['t1']['fn']}" / f"{inpt.extr[vr]['t1']['fn']}.DAT"
        try:
            df = pd.read_table(
                file,
                sep=r"\s+",
                engine="python",
                header=0,
                skiprows=9,
                parse_dates={"datetime": [0, 1]},
                index_col="datetime",
                date_format="%Y-%m-%d %H:%M:%S"
            )
            print(f"OK: {file.name}")
            df = df[[inpt.extr[vr]["t1"]["column"]]].rename(
                columns={inpt.extr[vr]["t1"]["column"]: vr})
            df_all = df
        except FileNotFoundError:
            print(f"NOT FOUND: {file.name}")
            df_all = pd.DataFrame(columns=[vr])

    # Save per-year parquet files
    for year in inpt.years:
        df_year = df_all[df_all.index.year == year]
        if not df_year.empty:
            path_out, _ = tls.get_common_paths(vr, year, "HATPRO")
            df_year.to_parquet(path_out)
            print(f"Saved {path_out}")

    # Store in container
    inpt.extr[vr]["t1"]["data"] = df_all


def read_thaao_ceilometer(vr):
    df_all = pd.DataFrame()
    count = 0

    # Load existing per-year parquet files
    for year in inpt.years:
        path_out, _ = tls.get_common_paths(vr, year, "CEIL")
        if os.path.exists(path_out):
            df_tmp = pd.read_parquet(path_out)
            df_all = pd.concat([df_all, df_tmp])
            print(f"Loaded {path_out}")
            count += 1

    # If not all years were loaded, parse original .txt files
    if count < len(inpt.years):
        t_all = []
        for i in inpt.ceilometer_daterange[inpt.ceilometer_daterange.year.isin(inpt.years)]:
            date_str = i.strftime("%Y%m%d")
            file = Path(inpt.basefol["t"]['base']) / "thaao_ceilometer" / \
                "medie_tat_rianalisi" / \
                f"{date_str}{inpt.extr[vr]['t']['fn']}.txt"
            try:
                df = pd.read_table(
                    file, sep=r"\s+", engine="python", header=0, skiprows=9)
                df[df == inpt.var_dict["t"]["nanval"]] = np.nan
                df.index = pd.to_datetime(
                    df["#"] + " " + df["date[y-m-d]time[h:m:s]"],
                    format="%Y-%m-%d %H:%M:%S"
                )
                df = df[[inpt.extr[vr]["t"]["column"]]].astype(float).rename(
                    columns={inpt.extr[vr]["t"]["column"]: vr}
                )
                t_all.append(df)
                print(f"OK: {file.name}")
            except (FileNotFoundError, pd.errors.EmptyDataError):
                print(f"NOT FOUND or EMPTY: {file.name}")

        if t_all:
            df_all = pd.concat(t_all)

    # Save per-year filtered data
    for year in inpt.years:
        df_year = df_all[df_all.index.year == year]
        if not df_year.empty:
            path_out, _ = tls.get_common_paths(vr, year, "CEIL")
            df_year.to_parquet(path_out)
            print(f"Saved {path_out}")

    # Final assignment
    inpt.extr[vr]["t"]["data"] = df_all


def read_thaao_aws_ecapac(vr):
    df_all = pd.DataFrame()
    count = 0

    # Try loading from per-year parquet files
    for year in inpt.years:
        path_out, _ = tls.get_common_paths(vr, year, "ECAPAC")
        if os.path.exists(path_out):
            df_tmp = pd.read_parquet(path_out)
            df_all = pd.concat([df_all, df_tmp])
            print(f"Loaded {path_out}")
            count += 1

    # If not all years found, read raw .dat files
    if count < len(inpt.years):
        t_all = []
        for i in inpt.aws_ecapac_daterange[inpt.aws_ecapac_daterange.year.isin(inpt.years)]:
            date_str = i.strftime("%Y_%m_%d")
            file = (
                Path(inpt.basefol["t"]['base']) / "thaao_ecapac_aws_snow" /
                "AWS_ECAPAC" / i.strftime("%Y") /
                f"{inpt.extr[vr]['t2']['fn']}{date_str}_00_00.dat"
            )
            try:
                df = pd.read_csv(
                    file,
                    skiprows=[0, 3],
                    header=0,
                    decimal=".",
                    delimiter=",",
                    engine="python",
                    index_col="TIMESTAMP"
                ).iloc[1:]
                df.index = pd.to_datetime(df.index)
                df = df[[inpt.extr[vr]["t2"]["column"]]].astype(float).rename(
                    columns={inpt.extr[vr]["t2"]["column"]: vr}
                )
                t_all.append(df)
                print(f"OK: {file.name}")
            except (FileNotFoundError, pd.errors.EmptyDataError):
                print(f"NOT FOUND: {file.name}")

        if t_all:
            df_all = pd.concat(t_all)

    # Save per-year data
    for year in inpt.years:
        df_year = df_all[df_all.index.year == year]
        if not df_year.empty:
            path_out, _ = tls.get_common_paths(vr, year, "ECAPAC")
            df_year.to_parquet(path_out)
            print(f"Saved {path_out}")

    # Store final result
    inpt.extr[vr]["t2"]["data"] = df_all
