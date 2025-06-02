import os
import glob

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


def dropsondes_netcdf(in_path, out_path, output_filename='arcsix_dropsondes_combined.nc'):
    attributes_arcsix = {
        'Conventions': 'CF-1.8',
        'featureType': 'trajectoryProfile',
        'title': 'NASA ARCSIX Dropsonde Profiles',
        'summary': (
            'This dataset contains vertical atmospheric profiles from NASA ARCSIX dropsonde launches.'
            'Each profile includes measurements of temperature, relative humidity, wind speed, '
            'wind direction, and geopotential height, sorted by pressure levels.'
        ),
        'institution': 'INGV',
        'creator_name': 'Filippo Calì Quaglia',
        'creator_email': 'filippo.caliquaglia@ingv.it',
        'history': f'Created on {pd.Timestamp.now()} using dropsondes_netcdf() function.',
        'trajectory_name': 'ARCSIX Dropsondes',
        'geospatial_lat_min': 76.5,
        'geospatial_lat_max': 76.5,
        'geospatial_lon_min': -68.8,
        'geospatial_lon_max': -68.8,
        'geospatial_vertical_positive': 'up',
        'project': 'NASA ARCSIX Dropsonde Campaign 2024',
        'references': 'NASA ARCSIX Repository',
        'license': 'CC-BY-4.0',
        'comment': 'This file is CF-1.8 compliant and suitable for trajectory profile analysis.',
    }

    output_file = os.path.join(out_path, 'arcsix_dropsondes_combined.nc')

    if os.path.exists(output_file):
        print(f'NetCDF already exists at: {output_file}')
        return output_file

    nc_files = sorted(glob.glob(os.path.join(in_path, '*.nc')))
    print(f"Found {len(nc_files)} dropsonde files")

    profiles = []
    launch_times = []

    for i, nc_file in enumerate(nc_files[20:]):
        ds = xr.open_dataset(nc_file)
        date_time = ds['launch_time'].values

        # Drop variables starting with "reference"
        vars_to_drop = [var for var in ds.data_vars if var.startswith('reference')]
        ds = ds.drop_vars(vars_to_drop)

        # Add trajectory dimension instead of profile
        ds = ds.expand_dims({'trajectory': [i]})

        # Assign trajectory_id coordinate
        ds = ds.assign_coords(trajectory_id=('trajectory', [os.path.basename(nc_file)]))

        # Handle 'time' variable
        if 'time' in ds.variables:
            ds['time'].attrs['axis'] = 'T'
            # Remove 'units' from attrs if it exists
            ds['time'].attrs.pop('units', None)
            # Add 'units' in encoding instead (CF compliant)
            ds['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00 UTC'
            ds = ds.set_coords('time')

        profiles.append(ds)
        launch_times.append(pd.to_datetime(date_time))

    # Concatenate all profiles along 'trajectory' dimension
    combined_ds = xr.concat(profiles, dim='trajectory')

    # Assign launch_times coordinate
    combined_ds = combined_ds.assign_coords(trajectory_time=('trajectory', launch_times))

    # Update global attributes with CF compliant metadata
    combined_ds.attrs.update(attributes_arcsix)

    # Save to NetCDF
    out_file = os.path.join(out_path, output_filename)
    combined_ds.to_netcdf(out_file)
    print(f"Combined dropsonde dataset saved to {out_file}")

    return output_filename



def main():
    """ """

    fol_path_dropsondes = r"H:\\Shared drives\\Dati_THAAO\\thaao_arcsix\\dropsondes"
    out_path = r"H:\\Shared drives\\Dati_THAAO\\thaao_arcsix"

    d_sonde_file = dropsondes_netcdf(fol_path_dropsondes, out_path)

    drop_sonde = xr.open_dataset(os.path.join(out_path, d_sonde_file))

    drop_sonde.info


    # dropsondes
    plt.figure(figsize=(8, 6))

    # Loop over all dropsonde trajectories and plot
    for i in range(drop_sonde.sizes['trajectory']):
        # Plot temperature vs pressure for this dropsonde profile
        plt.plot(drop_sonde['tdry'][i], drop_sonde['pres']
                 [i], label=f"Trajectory {i}",  lw=0, marker='.', markersize=1)

    plt.gca().invert_yaxis()
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Pressure (hPa)')
    plt.title('Temperature Profiles of All Dropsondes')

    # Optional: show legend (comment out if too many profiles!)
    # plt.legend()
    plt.ylim(1013, 0)
    plt.xlim(-60, 20)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



    
    
    plt.tight_layout()
    plt.show()

    print()

if __name__ == "__main__":
    main()
