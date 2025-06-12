import pandas as pd
import matplotlib.pyplot as plt


# Load ERA5 data
buoy_era5 = pd.read_parquet(
    r"H:\Shared drives\Reanalysis\era5\processed\era5_NG_2m_temperature_2024Kprocessed_2024.parquet"
)

# Load buoy data
buoy_original = pd.read_csv(
    r"H:\Shared drives\Dati_THAAO\thaao_arcsix\buoys\resource_map_doi_10_18739_A2T14TR46\data\2024Kprocessed.csv",
    index_col='time_stamp'
)

# Ensure datetime index
buoy_original.index = pd.to_datetime(buoy_original.index, dayfirst=True, errors='coerce')
buoy_era5.index = pd.to_datetime(buoy_era5.index, errors='coerce')

# Drop any rows with invalid timestamps (NaT)
buoy_original = buoy_original[buoy_original.index.notna()]
buoy_era5 = buoy_era5[buoy_era5.index.notna()]

# Forward fill buoy data (optional)
buoy_original.sort_index(inplace=True)
buoy_original.ffill(inplace=True)

# Align both DataFrames on common timestamps (inner join on index)
common_index = buoy_original.index.intersection(buoy_era5.index)
aligned = pd.DataFrame({
    'buoy_temp': buoy_original.loc[common_index, 'air_temp'],
    'era5_temp': buoy_era5.loc[common_index, 'temp']
})

# Drop rows with missing values
aligned.dropna(inplace=True)

# Plot scatter

n=350

plt.figure(figsize=(8, 6))
plt.scatter(aligned['buoy_temp'][0:n], aligned['era5_temp'][0:n]- 273.15, alpha=0.7, edgecolors='k')
plt.xlabel("Buoy Air Temperature (°C)")
plt.ylabel("ERA5 Temperature (°C)")
plt.title("Scatter Plot: Buoy vs ERA5 Air Temperature")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(aligned['era5_temp'][0:n]- 273.15, label="ERA5 Temperature (°C)")
plt.plot(aligned['buoy_temp'][0:n], label="Buoy Air Temperature (°C)")
plt.legend()
plt.show()

plt.plot(buoy_original.loc['incident'])
plt.show()
