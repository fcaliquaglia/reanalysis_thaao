import os
import glob
import re
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import mpl_toolkits.axes_grid1.inset_locator as inset_locator


# ---------------------------- SETTINGS ---------------------------- #
plot_flags = dict(
    ground_sites=True,
    buoys=True,
    dropsondes=True,
    p3_tracks=True,
    g3_tracks=True,
    radiosondes=True
)

basefol = r"H:\Shared drives\Dati_THAAO"
folders = {
    "dropsondes": os.path.join(basefol, r"thaao_arcsix\dropsondes"),
    "radiosondes": os.path.join(basefol, r"thaao_rs_sondes\txt\2024"),
    "buoys": os.path.join(basefol, r"thaao_arcsix\buoys\resource_map_doi_10_18739_A2T14TR46\data"),
    "g3": os.path.join(basefol, r"thaao_arcsix\met_nav\G3"),
    "p3": os.path.join(basefol, r"thaao_arcsix\met_nav\P3")
}

sites = {
    "THAAO": (-68.7477, 76.5149, "red"),
    "Villum": (-16.6667, 81.6, "cyan"),
    "Alert": (-62.5072, 82.4508, "green")
}


lon_min = -105
lon_max = -10
lat_min = 60
lat_max = 85
bounds = (lat_min, lat_max, lon_min, lon_max)
full_extent = [lon_min, lon_max, lat_min, lat_max]

zoom_extent = [-80, -5, 75, 87]

start_arcsix = np.datetime64("2024-05-15")
end_arcsix = np.datetime64("2024-08-15")

subsample_step_buoys = 10
subsample_step_tracks = 100

proj = ccrs.NorthPolarStereo(central_longitude=-40)
transform_pc = ccrs.PlateCarree()

dpi = 100

# ---------------------------- END SETTINGS ---------------------------- #

# ---------------------------- FUNCTIONS ---------------------------- #


def read_ict_file(filepath):
    """Reads second "Time_Start" header in an ICT file into a DataFrame."""
    with open(filepath, "r") as f:
        header_line = [i for i, line in enumerate(
            f) if line.startswith("Time_Start")]
    if len(header_line) < 2:
        raise ValueError(f"Second Time_Start header not found: {filepath}")
    return pd.read_csv(filepath, skiprows=header_line[1], index_col="Time_Start")


def filter_coords(lat, lon, bounds=None):
    """Filter coordinates and optional temp array within given bounds."""
    mask = (~np.isnan(lat)) & (~np.isnan(lon))
    if bounds:
        lat_min, lat_max, lon_min, lon_max = bounds
        mask &= (lat >= lat_min) & (lat <= lat_max) & (
            lon >= lon_min) & (lon <= lon_max)
    return mask, lat[mask], lon[mask]


def process_nc_folder(path, pattern):
    return sorted(glob.glob(os.path.join(path, pattern)))


def plot_ground_sites(ax):
    """
    Plot ground site markers and labels on the given axis.

    Parameters:
    - ax: matplotlib axis with cartopy projection
    - sites: dict mapping label -> (lon, lat, color)
    """

    for label, (lon, lat, color) in sites.items():
        # Plot marker
        ax.plot(
            lon, lat,
            marker="X",
            markersize=12,
            color=color,
            linestyle="None",
            transform=transform_pc,
            label=label
        )
        # Plot text label next to marker
        ax.text(
            lon, lat, label,
            fontsize=10,
            fontweight="bold",
            transform=transform_pc,
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"),
            verticalalignment="center",
            horizontalalignment="left",
            zorder=15
        )


def colorbar_extend(d_all, vmn, vmx):
    if d_all.size > 0:
        extend_min = np.nanmin(d_all) < vmn
        extend_max = np.nanmax(d_all) > vmx

        if extend_min and extend_max:
            ext = "both"
        elif extend_min:
            ext = "min"
        elif extend_max:
            ext = "max"
        else:
            ext = "neither"
    else:
        ext = "neither"

    return ext


def plot_background(ax, extent, title, add_grid=True):
    ax.set_extent(extent, crs=transform_pc)
    ax.add_feature(NaturalEarthFeature(
        "physical", "ocean", "10m", facecolor="#a6cee3"))
    ax.add_feature(NaturalEarthFeature("physical", "land",
                   "10m", edgecolor="black", facecolor="#f0e6d2"))
    ax.coastlines("10m")
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="azure")

    if add_grid:
        ax.gridlines(draw_labels=True, dms=True,
                     x_inline=False, y_inline=False)
    ax.set_title(title)


def generate_status_string(flags):
    lines = []
    if flags.get("ground_sites"):
        lines.append("Ground sites  N={:<4}".format(len(sites.keys())))

    if flags.get("dropsondes"):
        lines.append("Dropsondes    N={:<4}".format(len(drop_data)))

    if flags.get("buoys"):
        lines.append("Buoys         N={:<4}".format(len(buoy_data)))

    if flags.get("p3_tracks"):
        lines.append("P-3 tracks    N={:<4}".format(len(p3_data)))

    if flags.get("g3_tracks"):
        lines.append("G-III tracks  N={:<4}".format(len(g3_data)))

    return "\n".join(lines)

# ---------------------------- END FUNCTIONS ---------------------------- #


# ------------------------- PLOTTING FUNCTIONS ----------------------------- #


def plot_trajectories(seq, plot_flags=plot_flags):
    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={"projection": proj})
    plot_background(ax, full_extent, "Trajectory Plot")

    transform_pc = ccrs.PlateCarree()

    # --- Buoys ---
    if plot_flags["buoys"]:
        for i, d in enumerate(buoy_data):
            ax.scatter(
                d["lon"], d["lat"],
                color="blue", marker=".", s=10, alpha=0.6,
                transform=transform_pc, zorder=8,
                label="Buoys" if i == 0 else None
            )
            letter = "".join(
                filter(str.isalpha, os.path.basename(d['filename'])))[0]
            ax.text(
                d["lon"][0], d["lat"][0], letter,
                fontsize=10, fontweight="bold",
                transform=transform_pc,
                bbox=dict(facecolor="white", alpha=0.7,
                          boxstyle="round,pad=0.2")
            )

    # --- Dropsondes ---
    if plot_flags["dropsondes"]:
        for i, d in enumerate(drop_data):
            is_first = (i == 0)
            ax.plot(
                d["lon"], d["lat"],
                color="darkred", lw=1.5, transform=transform_pc,
                label="Dropsondes traj" if is_first else None
            )
            valid_idx = np.where(~np.isnan(d["lon"]) & ~np.isnan(d["lat"]))[0]

            if len(valid_idx) > 0:
                last_valid_index = valid_idx[-1]
                lon = d["lon"][last_valid_index]
                lat = d["lat"][last_valid_index]

                ax.plot(
                    lon, lat,
                    "o", color="black", markeredgecolor="yellow", markersize=6,
                    transform=transform_pc,
                    label="Dropsondes@surf" if is_first else None
                )

    # --- G3 Aircraft Tracks ---
    if plot_flags["g3_tracks"]:
        for i, d in enumerate(g3_data):
            is_first = (i == 0)
            ax.plot(
                d["lon"][::subsample_step_tracks],
                d["lat"][::subsample_step_tracks],
                lw=0.7, linestyle="--", alpha=0.6,
                color="purple", transform=transform_pc,
                label="G-3" if is_first else None
            )

    # --- P3 Aircraft Tracks ---
    if plot_flags["p3_tracks"]:
        for i, d in enumerate(p3_data):
            is_first = (i == 0)
            ax.plot(
                d["lon"][::subsample_step_tracks],
                d["lat"][::subsample_step_tracks],
                lw=0.7, linestyle="--", alpha=0.6,
                color="orange", transform=transform_pc,
                label="P-3" if is_first else None
            )

    # --- Ground Sites ---
    if plot_flags["ground_sites"]:
        plot_ground_sites(ax)

    # --- Status Text ---
    status_text = generate_status_string(plot_flags)
    ax.text(
        0.02, 0.98, status_text,
        transform=ax.transAxes,
        verticalalignment="top", horizontalalignment="left",
        fontsize=12, fontweight="bold", fontfamily="monospace",
        bbox=dict(facecolor="white", alpha=0.7,
                  edgecolor="none", boxstyle="round,pad=0.3")
    )

    # --- Legend ---
    legend = ax.legend(loc="lower right")
    legend.set_zorder(10)

    # --- Save Plots ---
    plt.savefig(
        f"all_trajectories_{seq}.png", dpi=dpi, bbox_inches="tight")
    ax.set_extent(zoom_extent, crs=transform_pc)
    plt.savefig(
        f"all_trajectories_{seq}_zoom.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_surf_temp(seq, plot_flags=plot_flags):
    # --- Setup figure and main axis ---
    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={"projection": proj})
    plot_background(ax, full_extent,
                    "Buoy and Dropsonde Surface Temperatures (2024)")

    norm = plt.Normalize(vmin=-10, vmax=10, clip=False)
    cmap = plt.get_cmap("coolwarm")

    # --- Gather all buoy data ---
    if plot_flags["buoys"]:
        all_buoy_lons = []
        all_buoy_lats = []
        all_buoy_temps = []
        for d in buoy_data:
            valid_mask = ((~np.isnan(d["temp"])) &
                          (d["time"] >= start_arcsix) &
                          (d["time"] <= end_arcsix)
                          )

            all_buoy_lats.extend(d["lat"][valid_mask])
            all_buoy_lons.extend(d["lon"][valid_mask])
            all_buoy_temps.extend(d["temp"][valid_mask])

        all_buoy_lats = all_buoy_lats[::subsample_step_buoys]
        all_buoy_lons = all_buoy_lons[::subsample_step_buoys]
        all_buoy_temps = all_buoy_temps[::subsample_step_buoys]

        # --- Plot buoy temps scatter ---
        ax.scatter(all_buoy_lons, all_buoy_lats, c=all_buoy_temps,
                   cmap=cmap, norm=norm, s=20, alpha=0.9,
                   edgecolor="none", linewidth=0.5, marker="s",
                   transform=transform_pc, zorder=10,
                   label=f"Buoys (each {subsample_step_buoys}th pnt)")

    # --- Prepare dropsonde surface temps ---
    if plot_flags["dropsondes"]:
        all_drop_surf_lons = []
        all_drop_surf_lats = []
        all_drop_surf_temps = []
        all_drop_surf_pres = []
        for d in drop_data:
            valid_idx = np.nanargmax(d["pres"]) if np.any(
                ~np.isnan(d["pres"])) else None

            all_drop_surf_temps.append(
                d["temp"][valid_idx] if valid_idx is not None else np.nan)
            all_drop_surf_pres.append(
                d["pres"][valid_idx] if valid_idx is not None else np.nan)
            all_drop_surf_lons.append(
                d["lon"][valid_idx] if valid_idx is not None else np.nan)
            all_drop_surf_lats.append(
                d["lat"][valid_idx] if valid_idx is not None else np.nan)

        # --- Plot dropsonde temps scatter ---
        ax.scatter(all_drop_surf_lons, all_drop_surf_lats, c=all_drop_surf_temps,
                   cmap=cmap, norm=norm, s=30,
                   edgecolor="none", linewidth=1.2,
                   marker="o", alpha=0.9,
                   transform=transform_pc, label="Dropsondes", zorder=11)

    # Ground sites
    if plot_flags["ground_sites"]:
        plot_ground_sites(ax)

    # --- Colorbar with inset axis for better sizing ---
    # Determine whether to extend colorbar
        vmin = norm.vmin
        vmax = norm.vmax
    try:
        data_all = np.array(all_buoy_temps + all_drop_surf_temps)
    except Exception:
        data_all = np.array([])

    extend = colorbar_extend(data_all, vmin, vmax)

    # Create the colorbar
    cax = inset_locator.inset_axes(
        ax,
        width="3%",
        height="40%",
        loc="lower right",
        bbox_to_anchor=(-0.05, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=2
    )
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="vertical",
        label="Surface Temperature (°C)",
        extend=extend
    )
    plt.setp(cbar.ax.get_yticklabels(), ha="right")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.set_ylabel("Surface Temperature (°C)", labelpad=10)
    cbar.ax.yaxis.tick_right()
    cbar.ax.tick_params(axis="y", pad=30)

    # --- Legend ---
    legend = ax.legend(loc="lower left")
    legend.set_zorder(10)
    status_text = generate_status_string(plot_flags)

    ax.text(0.02, 0.98, status_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=12,
            fontweight="bold",
            fontfamily="monospace",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none",
                      boxstyle="round,pad=0.3"))

    plt.savefig(f"all_surface_temperatures_{seq}.png",
                dpi=dpi, bbox_inches="tight")

    ax.set_extent(zoom_extent, crs=transform_pc)
    plt.savefig(f"all_surface_temperatures_{seq}_zoom.png",
                dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_surf_date(seq, plot_flags=plot_flags):
    # --- Setup figure and main axis ---
    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={"projection": proj})
    plot_background(ax, full_extent, "Buoy and Dropsonde Surface Dates (2024)")
    # --- Convert all dates to matplotlib numeric format ---
    vmin = mdates.date2num(start_arcsix)
    vmax = mdates.date2num(end_arcsix)
    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=False)
    cmap = plt.get_cmap("viridis")

    # --- Gather all buoy data ---
    if plot_flags["buoys"]:
        all_buoy_lons = []
        all_buoy_lats = []
        all_buoy_times = []
        for d in buoy_data:
            valid_mask = ((d["time"] >= start_arcsix) &
                          (d["time"] <= end_arcsix)
                          )

            all_buoy_lats.append(d["lat"][valid_mask])
            all_buoy_lons.append(d["lon"][valid_mask])
            all_buoy_times.append(d["time"][valid_mask])
        all_buoy_lats = np.concatenate(all_buoy_lats)[::subsample_step_buoys]
        all_buoy_lons = np.concatenate(all_buoy_lons)[::subsample_step_buoys]
        all_buoy_times = np.concatenate(all_buoy_times)[::subsample_step_buoys]
        all_buoy_nums = mdates.date2num(all_buoy_times)
        # Compute RGBA colors for each point via the colormap
        all_buoy_rgba = cmap(norm(all_buoy_nums))
        mask_buoy_out = (all_buoy_nums < vmin) | (all_buoy_nums > vmax)
        all_buoy_rgba[mask_buoy_out,    3] = 0.2

        ax.scatter(
            all_buoy_lons, all_buoy_lats,
            c=all_buoy_rgba,
            s=30,
            edgecolor="none",
            linewidth=0.5,
            marker="s",
            transform=transform_pc,
            label="Buoys",
            zorder=10
        )

    # --- Prepare dropsonde surface temps ---
    if plot_flags["dropsondes"]:
        all_drop_surf_lons = []
        all_drop_surf_lats = []
        all_drop_surf_times = []
        for d in drop_data:
            print(d['filename'])
            valid_idx = -1            
            all_drop_surf_lats.append(d["lat"][valid_idx])
            all_drop_surf_lons.append(d["lon"][valid_idx])
            all_drop_surf_times.append(d["time"][valid_idx])
        all_drop_surf_lats = np.array(all_drop_surf_lats)
        all_drop_surf_lons = np.array(all_drop_surf_lons)
        all_drop_surf_times = np.array(all_drop_surf_times)
        all_drop_surf_nums = mdates.date2num(all_drop_surf_times)
        all_drop_surf_rgba = cmap(norm(all_drop_surf_nums))
        all_mask_drop_surf_out = (all_drop_surf_nums < vmin) | (
            all_drop_surf_nums > vmax)
        all_drop_surf_rgba[all_mask_drop_surf_out, 3] = 0.2
        ax.scatter(
            all_drop_surf_lons, all_drop_surf_lats,
            c=all_drop_surf_rgba,
            s=30,
            edgecolor="none",
            linewidth=1.2,
            marker="o",
            transform=transform_pc,
            label="Dropsondes",
            zorder=11
        )

    if plot_flags["ground_sites"]:
        plot_ground_sites(ax)

    # Determine whether to extend colorbar
    try:
        data_all = np.concatenate((all_buoy_nums, all_drop_surf_nums))
    except Exception:
        data_all = np.array([])
        vmin = mdates.date2num(start_arcsix)
        vmax = mdates.date2num(end_arcsix)
    extend = colorbar_extend(data_all, vmin, vmax)

    # --- Colorbar with inset axis for better sizing ---
    cax = inset_locator.inset_axes(
        ax,
        width="3%",
        height="40%",
        loc="lower right",
        bbox_to_anchor=(-0.05, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=2
    )

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="vertical",
        label="Date",
        extend=extend  # Automatically determined
    )

    # --- Format colorbar ticks as dates ---
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    plt.setp(cbar.ax.get_yticklabels(), ha="right")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.set_ylabel("Date", labelpad=10)
    cbar.ax.yaxis.tick_right()
    cbar.ax.tick_params(axis="y", pad=30)

    # --- Legend ---
    legend = ax.legend(loc="lower left")
    legend.set_zorder(10)
    status_text = generate_status_string(plot_flags)

    ax.text(0.02, 0.98, status_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=12,
            fontweight="bold",
            fontfamily="monospace",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none",
                      boxstyle="round,pad=0.3"))

    plt.savefig(
        f"all_surface_dates_{seq}.png", dpi=dpi, bbox_inches="tight")

    ax.set_extent(zoom_extent, crs=transform_pc)
    plt.savefig(f"all_surface_dates_{seq}_zoom.png",
                dpi=dpi, bbox_inches="tight")
    plt.close()

# ------------------------ END PLOTTING FUNCTIONS ------------------------ #


if __name__ == "__main__":

    # Radiosondes
    if plot_flags['radiosondes']:
        radio_files = process_nc_folder(folders["radiosondes"], "*.nc")
        radio_data = []
        for rf in radio_files:
            print(rf)
            ds = xr.open_dataset(rf)
            time = pd.to_datetime(ds.attrs["launch_time"],
                                  format="%Y%m%d_%H%M")
            temp = ds["air_temperature"][0].values - 273.15
            pres = ds["air_pressure"][0].values
            radio_data.append(
                {"filename": os.path.basename(rf),
                    "time": time,
                    "temp": temp,
                    "pres": pres
                 })

    # Dropsondes
    if plot_flags['dropsondes']:
        drop_files = process_nc_folder(
            folders["dropsondes"], "ARCSIX-AVAPS-netCDF_G3*.nc")
        drop_data = []
        for df in drop_files:
            print(df)
            ds = xr.open_dataset(df)
            lat = ds["lat"].values
            lon = ds["lon"].values
            msk, lat, lon = filter_coords(lat, lon, bounds=bounds)
            if not msk.any():
                print("Skipped – no valid coordinates after filtering.")
                continue
            else:
                print("OK")
            temp = ds["tdry"][msk].values
            pres = ds["pres"][msk].values
            time = ds["time"][msk].values
            temp = np.where(temp == -999.0, np.nan, temp)
            pres = np.where(pres == -999.0, np.nan, pres)
            drop_data.append({"filename": os.path.basename(df),
                              "lat": lat, "lon": lon, "temp": temp,
                              "time": time, "pres": pres
                              })

    # Buoys
    if plot_flags['buoys']:
        buoy_files = [f for f in process_nc_folder(
            folders["buoys"], "2024*processed.nc")]
        buoy_data = []
        for bf in buoy_files:
            print(bf)
            ds = xr.open_dataset(bf)
            lat = ds["latitude"].isel(trajectory=0).values
            lon = ds["longitude"].isel(trajectory=0).values
            msk, lat, lon = filter_coords(lat, lon, bounds=bounds)
            if not msk.any():
                print("Skipped – no valid coordinates after filtering.")
                continue
            else:
                print("OK")
            temp = ds["air_temp"].isel(trajectory=0).values[msk]
            time = ds["time"].isel(trajectory=0).values[msk]
            buoy_data.append({"filename": os.path.basename(bf),
                              "lat": lat, "lon": lon, "temp": temp, "time": time
                              })

    # G3 tracks
    if plot_flags['g3_tracks']:
        g3_files = glob.glob(os.path.join(folders["g3"], "*R0*.ict"))
        g3_data = []
        for gf in g3_files:
            print(gf)
            base = os.path.basename(gf)
            m = re.search(r"_L(\d)\.ict$", base)
            if not m or m.group(1) == "2":
                ds = read_ict_file(gf)
                lat = ds["Latitude"].values
                lon = ds["Longitude"].values
                msk, lat, lon = filter_coords(lat, lon, bounds=bounds)
                if not msk.any():
                    print("Skipped – no valid coordinates after filtering.")
                    continue
                else:
                    print("OK")
                g3_data.append({"filename": os.path.basename(gf),
                                "lat": lat, "lon": lon, "temp": np.nan, "time": np.nan
                                })

    # P3 tracks
    if plot_flags['p3_tracks']:
        p3_files = glob.glob(os.path.join(folders["p3"], "*R0*.ict"))
        p3_data = []
        for pf in p3_files:
            print(pf)
            base = os.path.basename(pf)
            m = re.search(r"_L(\d)\.ict$", base)
            if not m or m.group(1) == "2":
                ds = read_ict_file(pf)
                lat = ds["Latitude"].values
                lon = ds["Longitude"].values
                msk, lat, lon = filter_coords(lat, lon, bounds=bounds)
                if not msk.any():
                    print("Skipped – no valid coordinates after filtering.")
                    continue
                else:
                    print("OK")
                p3_data.append({"filename": os.path.basename(pf),
                                "lat": lat, "lon": lon, "temp": np.nan, "time": np.nan
                                })

    del_list = ["ds", "temp", "time", "pres", "lat", "lon", "msk"]
    for var_name in del_list:
        try:
            del globals()[var_name]
        except KeyError:
            print(f"Variable '{var_name}' not found.")

    # ---------------------------- EXECUTION ---------------------------- #
    # plot_flags = {k: True for k in plot_flags}
    # plot_trajectories("all", plot_flags)
    # plot_flags = {k: False for k in plot_flags}
    # keys = list(plot_flags.keys())
    # for i, key in enumerate(keys, start=1):
    #     current_flags = {k: (j < i) for j, k in enumerate(keys)}
    #     plot_trajectories(i, current_flags)

    # current_flags = {k: False for k in plot_flags}
    # current_flags["ground_sites"] = True
    # plot_surf_temp("1", current_flags)
    # current_flags["buoys"] = True
    # plot_surf_temp("2", current_flags)
    # current_flags["dropsondes"] = True
    # plot_surf_temp("3", current_flags)

    current_flags = {k: False for k in plot_flags}
    current_flags["ground_sites"] = True
    plot_surf_date("1", current_flags)
    current_flags["buoys"] = True
    plot_surf_date("2", current_flags)
    current_flags["dropsondes"] = True
    plot_surf_date("3", current_flags)
