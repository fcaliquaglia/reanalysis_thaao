import os
import gc

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.warp import reproject, calculate_default_transform, Resampling
import xarray as xr
import xesmf as xe


def reproj_tif(input_path, output_path, dst_crs="EPSG:4326"):
    """Reproject a raster to given CRS without changing resolution."""
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({"crs": dst_crs, "transform": transform,
                      "width": width, "height": height})

        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)  # Keep nearest to avoid smoothing
    return


# Paths
basefol = os.path.dirname(os.path.abspath(__file__))  # folder of this script

input_tif_path = os.path.join(basefol, "pituffik_big.tif")
output_tif_path = os.path.join(basefol, "pituffik_big_reproj.tif")

carra_nc_path = os.path.join(basefol, "carra_2m_temperature_2023.nc")
era5_nc_path = os.path.join(basefol, "era5_2m_temperature_2023.nc")

# Open datasets with chunking
ds_c_all = xr.open_dataset(carra_nc_path, chunks={})
ds_e_all = xr.open_dataset(era5_nc_path, chunks={})

# Select one timestep from each to speed up processing
# Adjust dimension names if needed
ds_c = ds_c_all.isel(time=0).compute()  # CARRA usually has 'step' as time dim
ds_e = ds_e_all.isel(valid_time=0).compute()  # ERA5 usually has 'valid_time' as time dim

print(f"CARRA dataset dims: {ds_c.dims}")
print(f"ERA5 dataset dims: {ds_e.dims}")

# Extract CARRA lat/lon (2D)
carra_lat = ds_c["latitude"].values
carra_lon = ds_c["longitude"].values

# Build target grid dataset for regridding (CARRA grid)
target_grid = xr.Dataset({
    "lat": (["y", "x"], carra_lat),
    "lon": (["y", "x"], carra_lon),
})

print(f"Target grid shape: lat {carra_lat.shape}, lon {carra_lon.shape}")

# ERA5 coords (usually 1D)
era_lat = ds_e["latitude"].values
era_lon = ds_e["longitude"].values

# Prepare ERA5 data for regridding
# Create xarray Dataset with lat/lon dims
ds_e_sel = ds_e[['t2m']].rename({
    'latitude': 'lat',
    'longitude': 'lon'
})

# Rechunk if needed for xesmf
ds_e_sel = ds_e_sel.chunk({'lat': 21, 'lon': 21})

# Create ERA5 grid Dataset
era5_grid = xr.Dataset({
    'lat': (['lat'], era_lat),
    'lon': (['lon'], era_lon),
})

print("Creating regridder...")
regridder = xe.Regridder(ds_e_sel, target_grid, 'bilinear', periodic=False, reuse_weights=False)

print("Regridding ERA5 to CARRA grid...")
ds_e_on_carra = regridder(ds_e_sel)

# Save regridded ERA5
era5_regridded_path = os.path.join(basefol, "era5_regridded_to_carra.nc")
ds_e_on_carra.to_netcdf(era5_regridded_path)
print(f"ERA5 regridded dataset saved: {era5_regridded_path}")

# Reproject Pituffik raster to EPSG:4326 (no resampling to CARRA grid)
print("Reprojecting pituffik.tif to EPSG:4326 (keep native resolution)...")
reproj_tif(input_tif_path, output_tif_path, dst_crs="EPSG:4326")
print(f"Raster reprojected and saved: {output_tif_path}")

# Plot everything
print("Plotting results ...")
def convert_lon_360_to_180(lon):
    lon_new = (lon + 180) % 360 - 180
    return lon_new

# Example input points
lat1 = np.array([76.5149, 76.52, 76.5])
lon1 = np.array([-68.7477, -68.74, -68.8])

def adjust_lon(lon):
    """Adjust longitude array to 0-360 if needed."""
    lon = np.array(lon)
    lon[lon < 0] += 360
    return lon

def wrap_lon(lon):
    """Wrap longitude to -180 to 180."""
    lon = np.array(lon)
    lon = ((lon + 180) % 360) - 180
    return lon
def lon_diff(lon1, lon2):
    d = lon2 - lon1
    d = (d + 180) % 360 - 180
    return d
def find_pixel(ds, lat1, lon1):
    lat_ds, lon_ds = ds["latitude"], ds["longitude"]
    lon1_adj = adjust_lon(lon1)

    is_1d = (lat_ds.ndim == 1 and lon_ds.ndim == 1)
    if is_1d:
        lon2d, lat2d = np.meshgrid(lon_ds, lat_ds)  # lon on x-axis, lat on y-axis
    else:
        lat2d, lon2d = lat_ds, lon_ds

    lat_t, lon_t = [], []
    for lat, lon in zip(lat1, lon1_adj):

        dist = (lat2d - lat) ** 2 + lon_diff(lon2d, lon) ** 2
        if hasattr(dist, "values"):
            arr = dist.values
        else:
            arr = dist
        
        y_idx, x_idx = np.unravel_index(np.argmin(arr), arr.shape)
        closest_lat = lat2d[y_idx, x_idx] if not is_1d else lat_ds[y_idx]
        closest_lon = lon2d[y_idx, x_idx] if not is_1d else lon_ds[x_idx]
        closest_lon = wrap_lon(closest_lon.item())

        lat_t.append(closest_lat.item())
        lon_t.append(closest_lon)

        print(
            f"Closest grid point to ({lat:.4f},{wrap_lon(lon - 360):.4f}) "
            f"is at ({closest_lat:.4f}, {closest_lon:.4f}) index=({y_idx}, {x_idx})")

    return np.array(lat_t), np.array(lon_t)

def plot_closest(ds, lat1, lon1, ax):
    lat2, lon2 = find_pixel(ds, lat1, lon1)
    colors = ["orange", "green", "purple"]
    lon1_adj, lon2_adj = wrap_lon(lon1), wrap_lon(lon2)

    for idx in range(len(lat1)):
        ax.plot(
            lon1_adj[idx], lat1[idx], "o", markersize=3, color=colors[idx],
            label=f"PICK:({lat1[idx]:.4f}, {lon1_adj[idx]:.4f})")
        ax.plot(
            lon2_adj[idx], lat2[idx], "x", markersize=5, color=colors[idx],
            label=f"REF:({lat2[idx]:.4f}, {lon2_adj[idx]:.4f})")
        ax.plot(
            [lon1_adj[idx], lon2_adj[idx]], [lat1[idx], lat2[idx]], color=colors[idx], linestyle="--", linewidth=1)


with rasterio.open(output_tif_path) as src:
    fig, ax = plt.subplots(figsize=(10, 10))

    # Show raster background
    show(src, ax=ax)

    # Prepare grids
    lon_c_converted = convert_lon_360_to_180(ds_c["longitude"].values)

    xmin, ymin, xmax, ymax = src.bounds
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    x_half = (xmax - xmin) / 2
    y_half = (ymax - ymin) / 2
    new_xmin = xmin # x_center - 1.5 * x_half
    new_xmax = xmax # x_center + 1.5 * x_half
    new_ymin = ymin # y_center - 3 * y_half
    new_ymax = ymax # y_center + 3 * y_half

    # ERA5 grid (blue)
    lat_e = ds_e["latitude"].values
    lon_e = ds_e["longitude"].values
    lon2d_e, lat2d_e = np.meshgrid(lon_e, lat_e)
    for i in range(0, lon2d_e.shape[0], 1):
        ax.plot(lon2d_e[i, :], lat2d_e[i, :], color="blue", lw=0.3)
    for j in range(0, lon2d_e.shape[1], 1):
        ax.plot(lon2d_e[:, j], lat2d_e[:, j], color="blue", lw=0.3)

    # CARRA grid (red)
    lat_c = ds_c["latitude"].values
    lon_c = lon_c_converted
    for i in range(0, lon_c.shape[0], 1):
        ax.plot(lon_c[i, :], lat_c[i, :], color="red", lw=0.3)
    for j in range(0, lon_c.shape[1], 1):
        ax.plot(lon_c[:, j], lat_c[:, j], color="red", lw=0.3)

    # Expand plot limits
    ax.set_xlim(new_xmin, new_xmax)
    ax.set_ylim(new_ymin, new_ymax)
    ax.set_title("Pituffik Raster (native res.) with ERA5 & CARRA grids", fontsize=16)
    ax.axis("on")

    # Plot closest grid points on ERA5 and CARRA grids
    print("Plotting closest points on ERA5 grid:")
    plot_closest(ds_e, lat1, lon1, ax)

    print("Plotting closest points on CARRA grid:")
    plot_closest(ds_c, lat1, lon1, ax)

    #ax.legend()
    plt.savefig(os.path.join(basefol, "rean_regridded_to_carra_with_closest_points.png"), dpi=200, bbox_inches="tight")


plt.close("all")
gc.collect()
print("✅ All done!")
