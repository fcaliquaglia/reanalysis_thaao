import os
import gc

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.warp import reproject, Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling
import xarray as xr
import xesmf as xe

def reproj_tif(input_path, output_path, dst_crs="EPSG:4326"):
    """Reproject a raster to given CRS."""
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({"crs": dst_crs, "transform": transform,
                      "width": width, "height": height})

        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i), destination=rasterio.band(dst, i), src_transform=src.transform,
                    src_crs=src.crs, dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.nearest)
    return 

# 🔧 Paths
basefol = r"H:\Shared drives\Reanalysis"
input_tif_path = os.path.join(basefol, "pituffik.tif")
output_tif_path = os.path.join(basefol, "pituffik_reproj.tif")

# 1️⃣ Open datasets
ds_c = xr.open_dataset(os.path.join(
    basefol, "carra\\raw", "carra_2m_temperature_2023.nc"), chunks={'time': 10}, decode_timedelta=True)
ds_e = xr.open_dataset(os.path.join(
    basefol, "era5\\raw", "era5_2m_temperature_2023.nc"), decode_timedelta=True)

reproj_tif(input_tif_path, output_tif_path, dst_crs="EPSG:4326")

with rasterio.open(output_tif_path) as src:
    transform = src.transform
    width = src.width
    height = src.height

    # Create grid of row and col indices
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # rasterio.transform.xy returns flat arrays (not 2D!), so:
    lon1d, lat1d = rasterio.transform.xy(transform, rows, cols, offset='center')

    # Convert 1D arrays to 2D
    lon2d = np.array(lon1d).reshape(height, width)
    lat2d = np.array(lat1d).reshape(height, width)



# 2️⃣ Prepare CARRA’s grid as target
target_grid = xr.Dataset({
    "lat": (["y", "x"], lat2d),
    "lon": (["y", "x"], lon2d),
})

# 3️⃣ Regrid ds_e to CARRA grid
print("Regridding ERA5 to CARRA grid ...")
regridder_e = xe.Regridder(ds_e, target_grid, method="bilinear", periodic=False, reuse_weights=False)
ds_e_on_carra = regridder_e(ds_e)
era5_regridded_path = os.path.join(basefol, "era5_regridded_to_carra.nc")
ds_e_on_carra.to_netcdf(era5_regridded_path)
print("ERA5 regridded dataset saved:", era5_regridded_path)

# 4️⃣ Regrid raster (pituffik.tif) to CARRA grid
print("Regridding pituffik.tif to CARRA grid ...")
with rasterio.open(input_tif_path) as src:
    # Create new transform matching CARRA’s grid
    carra_transform = rasterio.transform.from_bounds(
        west=ds_c["longitude"].min().item(),
        south=ds_c["latitude"].min().item(),
        east=ds_c["longitude"].max().item(),
        north=ds_c["latitude"].max().item(),
        width=ds_c.dims["lon"],
        height=ds_c.dims["lat"],
    )

    # Update metadata
    kwargs = src.meta.copy()
    kwargs.update({
        "crs": "EPSG:4326",
        "transform": carra_transform,
        "width": ds_c.dims["lon"],
        "height": ds_c.dims["lat"],
    })

    regridded_raster_path = os.path.join(basefol, "pituffik_regridded_to_carra.tif")
    with rasterio.open(regridded_raster_path, "w", **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=carra_transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
            )
print("Raster regridded and saved:", regridded_raster_path)

# 5️⃣ Plot everything
print("Plotting results ...")
with rasterio.open(regridded_raster_path) as src:
    fig, ax = plt.subplots(figsize=(10, 10))
    show(src, ax=ax)
    xmin, ymin, xmax, ymax = src.bounds

    # Plot ERA5 grid (blue)
    lat, lon = ds_e_on_carra["latitude"].values, ds_e_on_carra["longitude"].values
    lon2d, lat2d = np.meshgrid(lon, lat, indexing="ij")
    for i in range(0, lon2d.shape[0], 10):
        ax.plot(lon2d[i, :], lat2d[i, :], color="blue", lw=0.3)
    for j in range(0, lat2d.shape[1], 10):
        ax.plot(lon2d[:, j], lat2d[:, j], color="blue", lw=0.3)

    # Plot CARRA grid (red)
    lat_c, lon_c = ds_c["latitude"].values, ds_c["longitude"].values
    lon2d_c, lat2d_c = np.meshgrid(lon_c, lat_c, indexing="ij")
    for i in range(0, lon2d_c.shape[0], 10):
        ax.plot(lon2d_c[i, :], lat2d_c[i, :], color="red", lw=0.3)
    for j in range(0, lat2d_c.shape[1], 10):
        ax.plot(lon2d_c[:, j], lat2d_c[:, j], color="red", lw=0.3)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("Raster & Reanalyses Regridded to CARRA Grid", fontsize=16)
    ax.axis("off")

    plt.savefig(os.path.join(basefol, "rean_regridded_to_carra.png"), dpi=200, bbox_inches="tight")
    plt.show()

# Cleanup
plt.close("all")
gc.collect()
print("✅ All done!")
