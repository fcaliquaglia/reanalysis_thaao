from pathlib import Path
import datetime as dt
import julian
import os
import numpy as np
import pandas as pd
import xarray as xr
import inputs as inpt
import tools as tls


def read_weather(vr):
    # Try loading from per-year parquet files
    df_all, count = load_per_year_parquets(vr, "VESPA")

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

    # Save per-year data
    save_per_year_parquets(vr, df_all, 'VESPA')

    # Store in global container
    inpt.extr[vr]["t"]["data"] = df_all


def read_rad(vr):
    # Try loading from per-year parquet files
    df_all, count = load_per_year_parquets(vr, "RAD")

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

        if t_all:
            df_all = pd.concat(t_all)
        else:
            df_all = pd.DataFrame()

    # Save per-year data
    save_per_year_parquets(vr, df_all, 'RAD')

    # Step 4: Store final result
    inpt.extr[vr]["t"]["data"] = df_all


def read_hatpro(vr):
    # Try loading from per-year parquet files
    df_all, count = load_per_year_parquets(vr, "HATPRO")

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
                index_col="datetime"
            )

            df = df[[inpt.extr[vr]["t1"]["column"]]].rename(
                columns={inpt.extr[vr]["t1"]["column"]: vr})
            df.index.name = 'datetime'
            df_all = df
            print(f"OK: {file.name}")
        except FileNotFoundError:
            print(f"NOT FOUND: {file.name}")
            df_all = pd.DataFrame(columns=[vr])

    # Save per-year data
    save_per_year_parquets(vr, df_all, 'HATPRO')

    # Store in container
    inpt.extr[vr]["t1"]["data"] = df_all


def read_ceilometer(vr):
    # Try loading from per-year parquet files
    df_all, count = load_per_year_parquets(vr, "CEIL")

    # If not all years were loaded, parse original .txt files

    if count < len(inpt.years):
        t_all = [] 
        for i in inpt.dateranges["ceilometer"][inpt.dateranges["ceilometer"].year.isin(inpt.years)]:
            date_str = i.strftime("%Y%m%d")
            file = Path(inpt.basefol["t"]["base"]) / "thaao_ceilometer" / \
                "medie_tat_rianalisi" / \
                f"{date_str}{inpt.extr[vr]['t']['fn']}.txt"
            try:
                df = pd.read_table(
                    file, sep=r"\s+", engine="python", header=0, skiprows=9)

                # Replace NaN values coded as specific "nanval"
                df.replace(inpt.var_dict["t"]["nanval"], np.nan, inplace=True)

                # Combine the two datetime columns into a single datetime index
                # The file may use column names like '#', or 'date[y-m-d]time[h:m:s]' or 'date[Y-M-D] time[h:m:s]'
                # So we try a flexible approach:

                if "#" in df.columns and "date[y-m-d]time[h:m:s]" in df.columns:
                    datetime_str = df["#"] + " " + df["date[y-m-d]time[h:m:s]"]
                elif "date[y-m-d]time[h:m:s]" in df.columns:
                    datetime_str = df["date[y-m-d]time[h:m:s]"]
                elif "date[Y-M-D]" in df.columns and "time[h:m:s]" in df.columns:
                    datetime_str = df["date[Y-M-D]"].astype(
                        str) + " " + df["time[h:m:s]"].astype(str)
                else:
                    raise ValueError(
                        f"Unexpected datetime columns in {file.name}")

                df.index = pd.to_datetime(
                    datetime_str, errors='raise', format='mixed')
                df.index.name = 'datetime'

                # Select relevant variable column and rename to vr
                df = df[[inpt.extr[vr]["t"]["column"]]].astype(
                    float).rename(columns={inpt.extr[vr]["t"]["column"]: vr})

                t_all.append(df)
                print(f"OK: {file.name}")
            except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
                print(f"NOT FOUND or EMPTY or FORMAT ERROR: {file.name} - {e}")

        # concatenate all loaded dataframes
        if t_all:
            df_all = pd.concat(t_all)
        else:
            df_all = pd.DataFrame()

        # Save per-year data
        save_per_year_parquets(vr, df_all, 'CEIL')

    # Final assignment
    inpt.extr[vr]["t"]["data"] = df_all


def read_aws_ecapac(vr):
    # Try loading from per-year parquet files
    df_all, count = load_per_year_parquets(vr, "ECAPAC")

    # If not all years found, read raw .dat files
    if count < len(inpt.years):
        t_all = []
        for i in inpt.dateranges["aws_ecapac"][inpt.dateranges["aws_ecapac"].year.isin(inpt.years)]:
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
                df.index.name = 'datetime'
                df.loc[df[vr] > 0.5, vr] = np.nan
                t_all.append(df)
                print(f"OK: {file.name}")
            except (FileNotFoundError, pd.errors.EmptyDataError):
                print(f"NOT FOUND: {file.name}")

        if t_all:
            df_all = pd.concat(t_all)
            df_all = df_all.resample('1h').apply(
                lambda x: x.sum() if x.notna().any() else np.nan)
            df_all.loc[df_all[vr] > 20, vr] = np.nan
        else:
            df_all = pd.DataFrame()

    # Save per-year data
    save_per_year_parquets(vr, df_all, 'ECAPAC')

    # Store final result
    inpt.extr[vr]["t2"]["data"] = df_all


def read_iwv_rs(vr):

    # Try loading from per-year parquet files
    df_all, count = load_per_year_parquets(vr, "RS")

    # If not all years found, read raw .dat files

    if count < len(inpt.years):
        t_all = pd.DataFrame()
        for year in inpt.years:
            files = os.listdir(
                Path(inpt.basefol["t"]['base']) / "thaao_rs_sondes" / "txt" / f'{year}' /
                "*.txt")
            for f in files:
                try:
                    file_date = dt.datetime.strptime(
                        f[9:22], '%Y%m%d_%H%M')
                    df = pd.read_table(
                        f,
                        skiprows=17, skipfooter=1, header=None, delimiter=" ", engine='python',
                        names=['height', 'pres', 'temp', 'rh'], usecols=[0, 1, 2, 3],
                        na_values="nan"
                    )
                    df.loc[(df['pres'] > 1013) | (
                        df['pres'] < 0), 'pres'] = np.nan
                    df.loc[(df['height'] < 0), 'height'] = np.nan
                    df.loc[(df['temp'] < -100) |
                           (df['temp'] > 30), 'temp'] = np.nan
                    df.loc[(df['rh'] < 1.) | (
                        df['rh'] > 120), 'rh'] = np.nan
                    df.dropna(subset=['temp', 'pres', 'rh'], inplace=True)
                    df.drop_duplicates(subset=['height'], inplace=True)

                    min_pres = df['pres'].min()
                    min_index = df[df['pres'] == min_pres].index.min()
                    df = df.iloc[:min_index]
                    df = df.set_index('height')
                    df.columns = [vr]

                    iwv = tls.convert_rs_to_iwv(df, 1.01)
                    t_all.append(pd.DataFrame(
                        index=[file_date], data=[iwv.magnitude]))
                    print(f'OK: year {year}')
                except FileNotFoundError:
                    print(f'NOT FOUND: year {year}')

        if t_all:
            df_all = pd.concat(t_all)
        else:
            df_all = pd.DataFrame()

    # Save per-year data
    save_per_year_parquets(vr, df_all, 'IWV_RS')

    # Store final result
    inpt.extr[vr]["t2"]["data"] = df_all


def read_iwv_vespa(vr):

    # Try loading from per-year parquet files
    df_all, count = load_per_year_parquets(vr, "ECAPAC")

    # If not all years found, read raw .dat files
    if count < len(inpt.years):
        file = (
            Path(inpt.basefol["t"]['base']) / "thaao_vespa" /
            "vespaPWVClearSky.txt"
        )
        try:
            df = pd.read_table(file, sep='\s+', skipfooter=1,
                               skiprows=1, header=None, engine='python'
                               )
            df.index = pd.to_datetime(
                df[0] + ' ' + df[1], format='%Y-%m-%d %H:%M:%S')
            df.drop(columns=[0, 1, 3, 4, 5], inplace=True)
            df.index.name = 'datetime'
            df.columns = [vr]
            df_all = df
            df_all.index = pd.to_datetime(df_all.index)
            print(f'OK: {file}')
        except FileNotFoundError:
            print(f'NOT FOUND: {file}.txt')

    # Save per-year data
    save_per_year_parquets(vr, df_all, 'IWV_VESPA')

    # Store final result
    inpt.extr[vr]["t"]["data"] = df_all


def load_per_year_parquets(vr, source):
    """
    Loads and concatenates yearly Parquet files using global vr, inpt, and tls,
    with the given source string.

    Parameters:
    -----------
    source : str
        Source identifier passed to get_common_paths, e.g. "RS".

    Returns:
    --------
    pd.DataFrame
        Combined DataFrame from all available years.
    list
        List of years successfully loaded.
    """
    df_all = pd.DataFrame()
    loaded_year = []

    for year in inpt.years:
        path_out, _ = tls.get_common_paths(vr, year, source)
        if os.path.exists(path_out):
            try:
                df_tmp = pd.read_parquet(path_out)
                loaded_year.append(year)
                if df_tmp.empty:
                    print(f"⚠️ Loaded EMPTY {path_out}!")
                else:
                    df_all = pd.concat([df_all, df_tmp])
                    print(f"✅ Loaded {path_out}")
            except Exception as e:
                print(f"⚠️ Failed to load {path_out}: {e}")
        else:
            print(f"⚠️ File not found: {path_out}")

    return df_all, len(loaded_year)


def save_per_year_parquets(vr, df_all, source):
    """
    Save data for each year from df_all into separate Parquet files.

    Parameters:
    -----------
    df_all : pd.DataFrame
        DataFrame with DatetimeIndex including multiple years.
    source : str, optional
        Source identifier passed to get_common_paths (default: "RS").

    Returns:
    --------
    None
    """
    for year in inpt.years:
        df_year = df_all[df_all.index.year == year]
        path_out, _ = tls.get_common_paths(vr, year, source)
        df_year.to_parquet(path_out)
        if df_year.empty:
            print(
                f"⚠️ Saved EMPTY data for year {year} to {path_out}!")
        else:
            print(f"✅ Saved data for year {year} to {path_out}")
