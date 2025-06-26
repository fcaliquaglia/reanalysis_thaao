from pathlib import Path
import os
import pandas as pd
import inputs as inpt
import tools as tls


def read_sigmaa_weather(vr):

    df_all = pd.DataFrame()
    cached_years = 0

    # Attempt to load cached data
    for year in inpt.years:
        path_out, _ = tls.get_common_paths(vr, year, "station")
        if os.path.exists(path_out):
            df_all = pd.concat([df_all, pd.read_parquet(path_out)])
            print(f"Loaded {path_out}")
            cached_years += 1

    # Read from raw CSV if any year is missing
    if cached_years < len(inpt.years):
        csv_file = Path(inpt.basefol['t']['arcsix']) / \
            "Sigma-A" / "SIGMA-A_2024summer_Lv1.3.csv"
        column_map = {
            'date': 'datetime',
            'WD1': 'windd',
            'U1': 'winds',
            'T1': 'temp',
            'RH1': 'rh',
            'SWd': 'sw_down',
            'SWu': 'sw_up',
            'LWd': 'lw_down',
            'LWu': 'lw_up',
            'sh': 'snow_height',
            'Pa': 'surf_pres',
            'albedo': 'alb'
        }

        try:
            df = pd.read_csv(csv_file, parse_dates=['date'])
            df.rename(columns=column_map, inplace=True)
            df.set_index('datetime', inplace=True)
            df = df.mask(df.isin([-9999, -9998, -9997, -8888]))
            df.drop(columns=[col for col in [None]
                    if col in df.columns], inplace=True)

            if vr not in df.columns:
                print(f"Variable '{vr}' not found in the dataset.")
                return

            df_all = df[[vr]]
            print(f"Processed raw CSV: {csv_file}")
        except FileNotFoundError:
            print(f"CSV file not found: {csv_file}")
            return
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            return

    # Save per-year parquet files
    for year in inpt.years:
        df_year = df_all[df_all.index.year == year]
        if not df_year.empty:
            path_out, _ = tls.get_common_paths(vr, year, "station")
            df_year.to_parquet(path_out)
            print(f"Saved {path_out}")

    # Update inpt
    inpt.extr[vr]["t"]["data"] = df_all
