import pandas as pd
import numpy as np 
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import xarray as xr

# Paths to folders
fol_path_dropsondes = r"H:\\Shared drives\\Dati_THAAO\\thaao_arcsix\\dropsondes"
folder_path_radiosondes = r"H:\Shared drives\Dati_THAAO\thaao_rs_sondes\txt\2024"

# Get sorted list of .nc files
radio_files = sorted(glob.glob(os.path.join(folder_path_radiosondes, '*.nc')))
drop_files = sorted(glob.glob(os.path.join(fol_path_dropsondes, '*.nc')))

# Initialize lists to store all times and profiles for normalization and plotting
radio_times_all = []
radio_temp_all = []
radio_pres_all = []

drop_times_all = []
drop_temp_all = []
drop_pres_all = []

# Loop over radiosonde files
for rf in radio_files:
    radio_sonde = xr.open_dataset(rf)
    
    # Extract launch time from file or from dataset attribute if exists
    # For example, you can parse 'launch_time' global attribute
    launch_time_str = radio_sonde.attrs.get('launch_time')
    # Convert launch_time_str like '20240814_2354' to datetime
    launch_time = pd.to_datetime(launch_time_str, format='%Y%m%d_%H%M')
    
    # Extract profile data
    temp_profile = radio_sonde['air_temperature'][0, :].values - 273.15  # convert K to °C
    pres_profile = radio_sonde['air_pressure'][0, :].values
    
    # Store launch time and profiles in lists
    radio_times_all.append(launch_time)
    radio_temp_all.append(temp_profile)
    radio_pres_all.append(pres_profile)

# Loop over dropsonde files
for ds_file in drop_files:
    ds = xr.open_dataset(ds_file)
    
    launch_time = pd.to_datetime(ds['launch_time'].values)
    
    # Extract profiles for pressure and temperature (tdry in °C)
    pres_profile = ds['pres'].values
    temp_profile = ds['tdry'].values
    
    # Mask missing values (-999.0)
    pres_profile = np.where(pres_profile == -999.0, np.nan, pres_profile)
    temp_profile = np.where(temp_profile == -999.0, np.nan, temp_profile)
    
    drop_times_all.append(launch_time)
    drop_pres_all.append(pres_profile)
    drop_temp_all.append(temp_profile)



# Create figure and 2 panels

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
# Combine all times for normalization
all_times = pd.Series(radio_times_all + drop_times_all)
min_time = all_times.min()
max_time = all_times.max()

norm = mcolors.Normalize(mdates.date2num(min_time), mdates.date2num(max_time))
# Choose distinct colormaps (not fixed red/blue)
cmap_radio = plt.cm.jet
cmap_drop = plt.cm.jet
# Convert times to numeric
all_times = radio_times_all + drop_times_all
all_times_num = mdates.date2num(all_times)

# Plot radiosonde on ax1
for i, (temp, pres) in enumerate(zip(radio_temp_all, radio_pres_all)):
    c = cmap_radio(norm(mdates.date2num(radio_times_all[i])))
    ax1.plot(temp, pres, marker='.', markersize=1, alpha=0.2, color=c, lw=0.2)
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Pressure (hPa)')
ax1.set_title('Radiosonde Profiles')

# Plot dropsonde on ax2
for i, (temp, pres) in enumerate(zip(drop_temp_all, drop_pres_all)):
    c = cmap_drop(norm(mdates.date2num(drop_times_all[i])))
    ax2.plot(temp, pres, marker='.', markersize=1, alpha=0.2, color=c, lw=0.2)
ax2.set_xlabel('Temperature (°C)')
ax2.set_title('Dropsonde Profiles')


# Create colorbars below each panel
# Position [left, bottom, width, height]
cbar_ax1 = fig.add_axes([0.13, 0.1, 0.35, 0.03])  # below left plot
sm_radio = plt.cm.ScalarMappable(cmap=cmap_radio, norm=norm)
sm_radio.set_array([])
cbar1 = fig.colorbar(sm_radio, cax=cbar_ax1, orientation='horizontal')
cbar1.ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
cbar1.set_label('Radiosonde Launch Date')

cbar_ax2 = fig.add_axes([0.57, 0.1, 0.35, 0.03])  # below right plot
sm_drop = plt.cm.ScalarMappable(cmap=cmap_drop, norm=norm)
sm_drop.set_array([])
cbar2 = fig.colorbar(sm_drop, cax=cbar_ax2, orientation='horizontal')
cbar2.ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
cbar2.set_label('Dropsonde Launch Date')

# Combine all pressure values (flatten and remove nans)
all_pressures = np.concatenate([np.concatenate(radio_pres_all), np.concatenate(drop_pres_all)])
all_pressures = all_pressures[~np.isnan(all_pressures)]

# Set y-limits explicitly to force increasing pressure upward
ax1.set_ylim(all_pressures.max(), all_pressures.min()*0.20)
ax2.set_ylim(all_pressures.max(), all_pressures.min()*0.20)

plt.tight_layout(rect=[0, 0.15, 1, 1])  # leave space at bottom for colorbars
plt.show()
