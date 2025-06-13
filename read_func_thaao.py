from pathlib import Path
import datetime as dt
import julian
import os
import numpy as np
import pandas as pd
import xarray as xr
import inputs as inpt


def get_common_paths(vr, prefix):
    location = next((v['fn']
                    for v in inpt.datasets.values() if v.get('switch')), None)
    base_out = Path(inpt.basefol['out']['processed'])
    base_input = Path(inpt.basefol['t']['arcsix'])
    filename = f"{location}_{prefix}_{vr}.parquet"
    return base_out / filename, base_input


def read_thaao_weather(vr):
    path_out, _ = get_common_paths(vr, "VESPA")
    nc_file = Path(inpt.basefol["t"]['base']) / \
        "thaao_aws_vespa" / f"{inpt.extr[vr]['t']['fn']}.nc"
    
    if os.path.exists(path_out):
        df = pd.read_parquet(path_out)
        inpt.extr[vr]["t"]["data"] = df
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
    path_out, _ = get_common_paths(vr, "rad")
    t_all = []

    if os.path.exists(path_out):
        df = pd.read_parquet(path_out)
        inpt.extr[vr]["t"]["data"] = df
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
    path_out, _ = get_common_paths(vr, "HATPRO")
    file = Path(inpt.basefol["t"]['base']) / "thaao_hatpro" / \
        f"{inpt.extr[vr]['t1']['fn']}" / f"{inpt.extr[vr]['t1']['fn']}.DAT"

    if os.path.exists(path_out):
        df = pd.read_parquet(path_out)
        inpt.extr[vr]["t1"]["data"] = df
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
    path_out, _ = get_common_paths(vr, "ceil")
    t_all = []

    if os.path.exists(path_out):
        df = pd.read_parquet(path_out)
        inpt.extr[vr]["t"]["data"] = df
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
    path_out, _ = get_common_paths(vr, "ECAPAC")
    t_all = []

    if os.path.exists(path_out):
        df = pd.read_parquet(path_out)
        inpt.extr[vr]["t2"]["data"] = df
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


# #!/usr/local/bin/python3
# # -*- coding: utf-8 -*-
# # -------------------------------------------------------------------------------
# #
# """
# Brief description
# """

# # =============================================================
# # CREATED:
# # AFFILIATION: INGV
# # AUTHORS: Filippo Cali' Quaglia, Monica Tosco
# # =============================================================
# #
# # -------------------------------------------------------------------------------
# __author__ = "Filippo Cali' Quaglia"
# __credits__ = ["??????"]
# __license__ = "GPL"
# __version__ = "0.1"
# __email__ = "filippo.caliquaglia@ingv.it"
# __status__ = "Research"
# __lastupdate__ = ""

# import datetime as dt
# import os

# import julian
# import numpy as np
# import pandas as pd
# import xarray as xr
# import inputs as inpt


# def read_thaao_weather(vr):
#     """
#     Reads and processes weather data for the specified variable and updates the
#     global input structure. The function attempts to load a NetCDF file associated
#     with the given variable and converts it into a pandas DataFrame. It then filters
#     and renames columns in the DataFrame based on predefined configurations in
#     the global input structure.

#     :param vr: The variable identifier used to fetch weather data.
#     :type vr: str

#     :raises FileNotFoundError: If the required NetCDF file does not exist.

#     :return: None. The global input structure is updated directly.
#     """
#     location = next(
#         (v['fn'] for k, v in inpt.datasets.items() if v.get('switch')), None)
#     inout_path = os.path.join(
#         inpt.basefol['out']['processed'], f"{location}_VESPA_{vr}.parquet")
#     input_path = os.path.join(inpt.basefol['t']['arcsix'])

#     try:
#         t_all = xr.open_dataset(
#             os.path.join(inpt.basefol["t"]['base'], "thaao_aws_vespa",
#                          f'{inpt.extr[vr]["t"]["fn"]}.nc'),
#             engine="netcdf4").to_dataframe()
#         print(f'OK: {inpt.extr[vr]["t"]["fn"]}.nc')
#     except FileNotFoundError:
#         print(f'NOT FOUND: {inpt.extr[vr]["t"]["fn"]}.nc')
#     t_all = t_all[[inpt.extr[vr]["t"]["column"]]]
#     t_all.columns = [vr]
#     t_all.to_parquet(inout_path)

#     try:
#         t_all = pd.read_parquet(inout_path)
#         print(f"Loaded {input_path}")
#     except FileNotFoundError as e:
#         print(e)

#     inpt.extr[vr]["t"]["data"] = t_all

#     return


# def read_thaao_rad(vr):
#     """
#     Reads and processes the Thaao radiation data for a specific variable over a specified
#     date range and yearly subset defined within the configuration. This function iterates
#     through years, attempts to load the corresponding data files for the input variable, and
#     processes the data to generate a time-indexed DataFrame for further analysis. If a file is
#     not found for a specific year, a message is logged to indicate the missing data.

#     :param vr: The variable name used to index configuration details and process corresponding
#                data.
#     :type vr: str
#     :return: None
#     """
#     location = next(
#         (v['fn'] for k, v in inpt.datasets.items() if v.get('switch')), None)
#     inout_path = os.path.join(
#         inpt.basefol['out']['processed'], f"{location}_rad_{vr}.parquet")
#     input_path = os.path.join(inpt.basefol['t']['arcsix'])

#     t_all = pd.DataFrame()
#     for i in inpt.rad_daterange[inpt.rad_daterange.year.isin(inpt.years)]:
#         i_fmt = int(i.strftime("%Y"))
#         try:
#             t_tmp = pd.read_table(
#                 os.path.join(inpt.basefol["t"]['base'], "thaao_rad",
#                              f'{inpt.extr[vr]["t"]["fn"]}{i_fmt}_5MIN.dat'),
#                 engine="python", skiprows=None, header=0, decimal=".", sep=r"\s+")
#             tmp = np.empty(t_tmp["JDAY_UT"].shape, dtype=dt.datetime)
#             for ii, el in enumerate(t_tmp["JDAY_UT"]):
#                 new_jd_ass = el + \
#                     julian.to_jd(dt.datetime(
#                         i_fmt - 1, 12, 31, 0, 0), fmt="jd")
#                 tmp[ii] = julian.from_jd(new_jd_ass, fmt="jd")
#                 tmp[ii] = tmp[ii].replace(microsecond=0)
#             t_tmp.index = pd.DatetimeIndex(tmp)
#             t_tmp = t_tmp[[inpt.extr[vr]["t"]["column"]]]
#             t_all = pd.concat([t_all, t_tmp], axis=0)
#             print(f'OK: {inpt.extr[vr]["t"]["fn"]}{i_fmt}.txt')
#         except FileNotFoundError:
#             print(f'NOT FOUND: {inpt.extr[vr]["t"]["fn"]}{i_fmt}.txt')
#     t_all.columns = [vr]
#     t_all.to_parquet(inout_path)

#     try:
#         t_all = pd.read_parquet(inout_path)
#         print(f"Loaded {input_path}")
#     except FileNotFoundError as e:
#         print(e)

#     inpt.extr[vr]["t"]["data"] = t_all

#     return


# def read_thaao_hatpro(vr):
#     """
#     Reads and processes data from a specified text file into a structured dataframe. This function
#     attempts to read and parse data for a given `vr` (variable identifier) from a specific file path.
#     It processes datetime information, assigns appropriate column names, and stores the resultant
#     data within a structured input object. If the specified file is not found, it logs the error
#     without halting execution.

#     :param vr: The variable identifier used to locate and process the associated data.
#     :type vr: str
#     :return: None
#     """
#     location = next(
#         (v['fn'] for k, v in inpt.datasets.items() if v.get('switch')), None)
#     inout_path = os.path.join(
#         inpt.basefol['out']['processed'], f"{location}_HATPRO_{vr}.parquet")
#     input_path = os.path.join(inpt.basefol['t']['arcsix'])

#     # t1_tmp_all = pd.DataFrame()
#     try:
#         #    t1_tmp = pd.read_table(
#         #            os.path.join(
#         #                    inpt.basefol["t"]['base'], "thaao_hatpro",
#         #                    f'{inpt.extr[vr]["t1"]["fn"]}', f'{inpt.extr[vr]["t1"]["fn"]}.DAT'),
#         #            sep=r"\s+", engine="python", header=0, skiprows=9)
#         # #   t1_tmp.columns = ["Date[y_m_d]", "Time[h:m]", "LWP[g/m2]", "STD_LWP[g/m2]", "Num"]
#         #    # t1_tmp_all = t1_tmp

#         #    t1_tmp.index = pd.to_datetime(
#         #    (t1_tmp[["Date_y_m_d"]].values + " " + t1_tmp[["Time_h:m"]].values)[:,0],
#         #    format="%Y-%m-%d %H:%M:%S")
#         t1_tmp = pd.read_table(
#             os.path.join(
#                 inpt.basefol["t"]['base'], "thaao_hatpro", f'{inpt.extr[vr]["t1"]["fn"]}',
#                 f'{inpt.extr[vr]["t1"]["fn"]}.DAT'), sep=r"\s+", engine="python", header=0, skiprows=9,
#             parse_dates={"datetime": [0, 1]}, date_format="%Y-%m-%d %H:%M:%S", index_col="datetime")

#         print(f'OK: {inpt.extr[vr]["t1"]["fn"]}.DAT')
#     except FileNotFoundError:
#         print(f'NOT FOUND: {inpt.extr[vr]["t1"]["fn"]}.DAT')

#     t1_tmp_all = t1_tmp[[inpt.extr[vr]["t1"]["column"]]]

#     t1_tmp_all.columns = [vr]
#     t1_tmp_all.to_parquet(inout_path)

#     try:
#         t1_tmp_all = pd.read_parquet(inout_path)
#         print(f"Loaded {input_path}")
#     except FileNotFoundError as e:
#         print(e)

#     inpt.extr[vr]["t1"]["data"] = t1_tmp_all

#     return


# def read_thaao_ceilometer(vr):
#     """
#     Reads and processes ceilometer data for a specified variable from multiple files
#     in a given directory structure. The data is collected from files corresponding
#     to a specific date range and year(s). This function concatenates the data
#     from multiple files, cleans it (replacing specific values with NaNs),
#     and processes timestamps to create a time-indexed DataFrame.

#     :param vr: The variable key indicating the specific type of data
#         to be processed from the ceilometer files (e.g., temperature).
#     :type vr: str

#     :raises FileNotFoundError: Raised if a file for a given date is not found.
#     :raises pd.errors.EmptyDataError: Raised if a file for a given date is empty.

#     :return: None
#     """
#     location = next(
#         (v['fn'] for k, v in inpt.datasets.items() if v.get('switch')), None)
#     inout_path = os.path.join(
#         inpt.basefol['out']['processed'], f"{location}_ceil_{vr}.parquet")
#     input_path = os.path.join(inpt.basefol['t']['arcsix'])

#     t_all = pd.DataFrame()
#     for i in inpt.ceilometer_daterange[inpt.ceilometer_daterange.year.isin(inpt.years)]:
#         i_fmt = i.strftime("%Y%m%d")
#         try:
#             t_tmp = pd.read_table(
#                 os.path.join(
#                     inpt.basefol["t"]['base'], "thaao_ceilometer", "medie_tat_rianalisi",
#                     f'{i_fmt}{inpt.extr[vr]["t"]["fn"]}.txt'), skipfooter=0, sep=r"\s+", header=0, skiprows=9,
#                 engine="python")
#             t_tmp[t_tmp == inpt.var_dict["t"]["nanval"]] = np.nan
#             t_all = pd.concat([t_all, t_tmp], axis=0)
#             print(f'OK: {i_fmt}{inpt.extr[vr]["t"]["fn"]}.txt')
#         except (FileNotFoundError, pd.errors.EmptyDataError):
#             print(f'NOT FOUND: {i_fmt}{inpt.extr[vr]["t"]["fn"]}.txt')

#     t_all.index = pd.to_datetime(
#         t_all["#"] + " " + t_all["date[y-m-d]time[h:m:s]"], format="%Y-%m-%d %H:%M:%S")
#     t_all.index.name = "datetime"
#     t_all = t_all.iloc[:, :].filter(
#         [inpt.extr[vr]["t"]["column"]]).astype(float)
#     t_all.columns = [vr]
#     t_all.to_parquet(inout_path)

#     try:
#         t_all = pd.read_parquet(inout_path)
#         print(f"Loaded {input_path}")
#     except FileNotFoundError as e:
#         print(e)

#     inpt.extr[vr]["t"]["data"] = t_all
#     return


# def read_thaao_aws_ecapac(vr):
#     """
#     Reads and processes AWS ECAPAC data for a specified variable across a specific date range
#     defined in the input configuration. Concatenates data for all specified dates, handles missing
#     or malformed files, and updates the data container with the formatted results.

#     :param vr: Variable name (str) specifying the dataset key in the input extraction configuration.
#     :return: None
#     """
#     location = next(
#         (v['fn'] for k, v in inpt.datasets.items() if v.get('switch')), None)
#     inout_path = os.path.join(
#         inpt.basefol['out']['processed'], f"{location}_ECAPAC_{vr}.parquet")
#     input_path = os.path.join(inpt.basefol['t']['arcsix'])

#     t2_all = pd.DataFrame()
#     for i in inpt.aws_ecapac_daterange[inpt.aws_ecapac_daterange.year.isin(inpt.years)]:
#         i_fmt = i.strftime("%Y_%m_%d")
#         try:
#             file = os.path.join(
#                 inpt.basefol["t"]['base'], "thaao_ecapac_aws_snow", "AWS_ECAPAC", i.strftime(
#                     "%Y"),
#                 f'{inpt.extr[vr]["t2"]["fn"]}{i_fmt}_00_00.dat')
#             t2_tmp = pd.read_csv(
#                 file, skiprows=[0, 3], header=0, decimal=".", delimiter=",", engine="python",
#                 index_col="TIMESTAMP").iloc[1:, :]
#             t2_all = pd.concat([t2_all, t2_tmp], axis=0)

#             print(f'OK: {inpt.extr[vr]["t2"]["fn"]}{i_fmt}_00_00.dat')
#         except (FileNotFoundError, pd.errors.EmptyDataError):
#             print(f'NOT_FOUND: {inpt.extr[vr]["t2"]["fn"]}{i_fmt}_00_00.dat')

#     t2_all.index = pd.DatetimeIndex(
#         t2_all.index)
#     t2_all.index.name = "datetime"
#     t2_all = t2_all.iloc[:, :].filter(
#         [inpt.extr[vr]["t2"]["column"]]).astype(float)
#     t2_all.columns = [vr]
#     t2_all.to_parquet(inout_path)

#     try:
#         t2_all = pd.read_parquet(inout_path)
#         print(f"Loaded {input_path}")
#     except FileNotFoundError as e:
#         print(e)

#     inpt.extr[vr]["t2"]["data"] = t2_all
#     return
