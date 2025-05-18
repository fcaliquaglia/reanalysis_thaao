# THAAO comparison with reanalysis

> [!IMPORTANT]  
> This is a preliminary analysis.

### TODO

- [ ] extract better pixel
- [ ] revise strategy for pixel extraction using read_func.extract_values()
- [ ] riportare valori di pressione al livello del mare?
- [x] download more ERA-L products
- [ ] focus on weather variables
- [ ] recuperare da Giovanni gli rs che io non ho e devo aggiungere in archivio per l'analisi
- [X] extract CARRA radiation
- [x] download ``mean_sea_level_pressure``
- [ ] controllare i nan value per ogni serie e la pulizia (dati negativi, outliers?, etc)

## Environment packages needed

- python3
- scipy
- netcdf4
- mtplotlib
- julian
- pandas
- xarray
- metpy

## Reanalysis considered

### CARRA (Copernicus Arctic Regional Reanalysis)

- 3h resolution at 2.5 km
- 10.24381/cds.713858f6
- Ridal, Martin, et al. "CERRA, the Copernicus European Regional Reanalysis system." Quarterly Journal of the Royal
  Meteorological Society (2024).
- [cds.climate.copernicus.eu/datasets/reanalysis-carra-single-levels](https://cds.climate.copernicus.eu/datasets/reanalysis-carra-single-levels?tab=overview)

> [!NOTE]
> (from the official website) The C3S Arctic Regional Reanalysis (CARRA) dataset contains 3-hourly analyses and hourly
> short term forecasts of
> atmospheric and surface meteorological variables (surface and near-surface temperature, surface and top of atmosphere
> fluxes, precipitation, cloud, humidity, wind, pressure, snow and sea variables) at 2.5 km resolution. Additionally,
> forecasts up to 30 hours initialised from the analyses at 00 and 12 UTC are available.
> The dataset includes two domains. The West domain covers Greenland, the Labrador Sea, Davis Strait, Baffin Bay,
> Denmark Strait, Iceland, Jan Mayen, the Greenland Sea and Svalbard. The East domain covers Svalbard, Jan Mayen, Franz
> Josef Land, Novaya Zemlya, Barents Sea, and the Northern parts of the Norwegian Sea and Scandinavia.
> The dataset has been produced with the use of the HARMONIE-AROME state-of-the-art non-hydrostatic regional numerical
> weather prediction model. High resolution reanalysis for the Arctic region is particularly important because the
> climate change is more pronounced in the Arctic region than elsewhere in the Earth. This fact calls for a better
> description of this region providing additional details with respect to the global reanalyses (ERA5 for instance). The
> additional information is provided by the higher horizontal resolution, more local observations (from the Nordic
> countries and Greenland), better description of surface characteristics (high resolution satellite and physiographic
> data), high resolution non-hydrostatic dynamics and improved physical parameterisation of clouds and precipitation in
> particular.
> The inputs to CARRA reanalysis are the observations, the ERA5 global reanalysis as lateral boundary conditions and the
> physiographic datasets describing the surface characteristics of the model. The observation values and information
> about their quality are used together to constrain the reanalysis where observations are available and provide
> information for the data assimilation system in areas in where less observations are available.

### ERA5

- 1 h resolution at 0.25° x 0.25°
- 10.24381/cds.143582cf
- [cds.climate.copernicus.eu/datasets/reanalysis-era5-complete](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-complete?tab=overview)

> [!NOTE]
> (from the official website) ERA5 is the fifth generation ECMWF atmospheric reanalysis of the global climate covering
> the
> period from January 1940
> to present1. It is produced by the Copernicus Climate Change Service (C3S) at ECMWF and provides hourly estimates of a
> large number of atmospheric, land and oceanic climate variables. The data cover the Earth on a 31km grid and resolve
> the atmosphere using 137 levels from the surface up to a height of 80km. ERA5 includes an ensemble component at half
> the resolution to provide information on synoptic uncertainty of its products.
> ERA5 uses a state-of-the-art numerical weather prediction model to assimilate a variety of observations,
> including satellite and ground-based measurements, and produces a comprehensive and consistent view of the Earth's
> atmosphere. These products are widely used by researchers and practitioners in various fields, including climate
> science, weather forecasting, energy production and machine learning among others, to understand and analyse past and
> current weather and climate conditions.

### ERA5-LAND

- 1 h resolution at 0.1° x 0.1°
- 10.24381/cds.e2161bac
- [cds.climate.copernicus.eu/datasets/reanalysis-era5-land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview)

> [!NOTE]
> (from the official website) ERA5-Land is a reanalysis dataset providing a consistent view of the evolution of land
> variables over several decades at an enhanced resolution compared to ERA5. ERA5-Land has been produced by replaying
> the land component of the ECMWF ERA5 climate reanalysis. Reanalysis combines model data with observations from across
> the world into a globally complete and consistent dataset using the laws of physics. Reanalysis produces data that
> goes several decades back in time, providing an accurate description of the climate of the past.
> ERA5-Land uses as input to control the simulated land fields ERA5 atmospheric variables, such as air temperature and
> air humidity. This is called the atmospheric forcing. Without the constraint of the atmospheric forcing, the
> model-based estimates can rapidly deviate from reality. Therefore, while observations are not directly used in the
> production of ERA5-Land, they have an indirect influence through the atmospheric forcing used to run the simulation.
> In addition, the input air temperature, air humidity and pressure used to run ERA5-Land are corrected to account for
> the altitude difference between the grid of the forcing and the higher resolution grid of ERA5-Land. This correction
> is called 'lapse rate correction'.
> The ERA5-Land dataset, as any other simulation, provides estimates which have some degree of uncertainty. Numerical
> models can only provide a more or less accurate representation of the real physical processes governing different
> components of the Earth System. In general, the uncertainty of model estimates grows as we go back in time, because
> the number of observations available to create a good quality atmospheric forcing is lower. ERA5-land parameter fields
> can currently be used in combination with the uncertainty of the equivalent ERA5 fields. The temporal and spatial
> resolutions of ERA5-Land makes this dataset very useful for all kind of land surface applications such as flood or
> drought forecasting. The temporal and spatial resolution of this dataset, the period covered in time, as well as the
> fixed grid used for the data distribution at any period enables decisions makers, businesses and individuals to access
> and use more accurate information on land states.

# Useful bibliography

- [] Batrak, Yurii, Bin Cheng, and Viivi Kallio-Myers. "Sea ice cover in the Copernicus Arctic Regional Reanalysis." The
  Cryosphere 18.3 (2024): 1157-1183.
- [] Køltzow, Morten, et al. "Value of the Copernicus Arctic Regional Reanalysis (CARRA) in representing near-surface
  temperature and wind speed in the north-east European Arctic." Polar Research 41 (2022).
- [thesis, polar mesoscale cyclones, polar lows, ERA5, CARRA ] Cheng, Zhaohui. Polar Mesoscale Cyclones in ERA5 and
  CARRA. 2023. Department of Earth Sciences, Uppsala
  University. https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1765122&dswid=-2975
- [OCEANO, no CARRA] Xie, Jiping, et al. "Quality assessment of the TOPAZ4 reanalysis in the Arctic over the period
  1991–2013." Ocean Science 13.1 (2017): 123-144.
- https://climate.copernicus.eu/sites/default/files/2023-10/CARRA_user_workshop_Koeltzow.pdf
- [CARRA wind] Lundesgaard, Øyvind, et al. ‘Import of Atlantic Water and Sea Ice Controls the Ocean Environment in the
  Northern Barents Sea’. Ocean Science, vol. 18, no. 5, Sept. 2022, pp.
  1389–418, https://doi.org/10.5194/os-18-1389-2022.
- [CARRA, atmospheric rivers, dropsondes] Dorff, Henning, et al. ‘Observability of Moisture Transport Divergence in
  Arctic Atmospheric Rivers by Dropsondes’. Atmospheric Chemistry and Physics, vol. 24, no. 15, Aug. 2024, pp.
  8771–95, https://doi.org/10.5194/acp-24-8771-2024.
- [CARRA precipitation Greenland] Box, Jason E., et al. ‘Greenland Ice Sheet Rainfall Climatology, Extremes and
  Atmospheric River Rapids’. Meteorological Applications, vol. 30, no. 4, July 2023, p.
  e2134, https://doi.org/10.1002/met.2134.
- [CARRA, ERA5, ERA5-L Greenland, zenith tropospheric delay] Jiang, Chunhua, et al. ‘Comparison of ZTD Derived from
  CARRA, ERA5 and ERA5-Land over the Greenland Based on GNSS’. Advances in Space Research, vol. 72, no. 11, Dec. 2023,
  pp. 4692–706, https://doi.org/10.1016/j.asr.2023.09.002.
- [] Kirbus, Benjamin, et al. ‘Thermodynamic and Cloud Evolution in a Cold-Air Outbreak during HALO-(AC)3:
  Quasi-Lagrangian Observations Compared to the ERA5 and CARRA Reanalyses’. Atmospheric Chemistry and Physics, vol. 24,
  no. 6, Apr. 2024, pp. 3883–904. , https://doi.org/10.5194/acp-24-3883-2024.
- [] Isaksen, Ketil, et al. ‘Exceptional Warming over the Barents Area’. Scientific Reports, vol. 12, no. 1, June 2022,
  p. 9371, https://doi.org/10.1038/s41598-022-13568-5.
- [preprint] 'Fram Strait Marine Cold Air Outbreaks in CARRA and ERA5: Effects on Surface Turbulent Heat Fluxes and the
  Vertical Structure of the Troposphere', https://doi.org/10.22541/essoar.167898508.82732727/v1.
- [ERA5-LAND] https://doi.org/10.5194/essd-13-4349-2021
- [ERA5] https://doi.org/10.1002/qj.4803
- [ERA5 precipitation] https://doi.org/10.1002/joc.8877
- [5 reanalyses] 'Improved Performance of ERA5 in Arctic Gateway Relative to Four Global Atmospheric
  Reanalyses', https://doi.org/10.1029/2019GL082781
- [other] Unlike CARRA, ERA5 lacks a parameterization of a snow layer on top of the sea-ice (Batrak & Muller, 2019).

# THAAO reference instruments

Instrument involved are HATPRO (LWP), aws_ECAPAC (temp, press, rh, windd, winds), aws_vespa (temp, press, rh,
windd, winds), sw radiometers (up and down),
The reference values are always from THAAO measurements, except for IWV (ref: VESPA) and LWP (ref:HATPRO)

> [!IMPORTANT]
> The code can be run at whichever time resolution.

> [!IMPORTANT]
> The pixel extraction is done before running this code using ``cdo remapnn`` for pixel 76.5, -68.8 which has a few
> hundreds meters offset from the exact location.

> [!TIP]
> wind component combination, conversion from dewpoint temperature to rh and similar, have been performed using the
``metpy`` package.

# Statistics

## Time series

## Scatterplots

> [!WARNING]
> The scatterplots containing all season data is a 2D density histogram, meaning that not all data are
> plotted (there are too many, and they would result overlaid), therefore the colorbar towards the warm color indicates
> higher density of values in that bin. For scatterplots with a limited number of data the result is a few number of
> points compared to the variable N (printed). Seasonal scatterplots are standard.

### Bland-Altman plot

A reference for this type of plots used for atmospheric analysis can be found
here: [Validation of the Cloud_CCI (Cloud Climate Change Initiative) cloud products in the Arctic](https://amt.copernicus.org/articles/16/2903/2023/).

### N

Total number of points analysed.

### MAE

> mae = np.nanmean(np.abs(y - x))

Excluding nan values. x(t): reference value, usually THAAO, see above; y(t): reanalysis or other

### R

Pearson correlation coefficient

### RMSE

> rmse = np.sqrt(np.nanmean((y - x) ** 2)

Excluding nan values. x(t): reference value; y(t): reanalysis or other

# Variables

## WEATHER

### Surface Pressure (``surf_pres``)

> [!WARNING]
> Values masked to nan for surf_pres<900, since they are unrealistic.

- CARRA: ``surface_pressure``
- ERA-5: ``surface_pressure``
- ERA5-L: /
- THAAO (vespa): values in these periods have been excluded: 2021-10-11 --> 2021-10-19 and 2024-4-28 --> 2024-5-4
- THAAO (aws_ECAPAC):

### Surface temperature (``temp``)

- CARRA: ``2m_temperature``
- ERA-5: ``2m_temperature``
- ERA5-L: ``2m_temperature``
- THAAO (vespa):
- THAAO (aws_ECAPAC):

### Relative Humidity (``rh``)

- CARRA: ``2m_relative_humidity``
- ERA-5: ``2m_dewpoint_temperature`` + ``2m_temperature`` (descrivere processo per ottenere rh)
- ERA5-L: /
- THAAO (vespa):
- THAAO (aws_ECAPAC):

### Wind Direction (``windd``)

- CARRA:``10m_wind_direction``
- ERA-5: ``10m_u_component_of_wind`` + ``10m_v_component_of_wind`` (descrivere processo per ottenere velocità e
  direzione)
- ERA5-L: /
- THAAO (vespa):
- THAAO (aws_ECAPAC):

### Wind Speed (``winds``)

- CARRA: ``10m_wind_speed``
- ERA-5: ``10m_u_component_of_wind`` + ``10m_v_component_of_wind`` (descrivere processo per ottenere velocità e
  direzione)
- ERA5-L: /
- THAAO (aws_ECAPAC):

## RADIATION

> [!WARNING]
> For CARRA radiation values. These forecast variables are released t different leadtimes, with 1-hour frequency.
> Therefore, we consider only leadtime 1, obtaining every three hours, hourly forecast valued for the following hour
> w.r.t the chose timeframe. For example, we choose April 1, 2023 at 6:00 UTC, we analyze forecast values on April 1,
> 2023 at 7:00 UTC. All the radiation dataset have been cleaned for values <0.

> [!WARNING]
> Accumulations rates for ERA5-Land >> from https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation#ERA5Land:datadocumentation-Temporalfrequency
> Please, note that the convention for accumulations used in ERA5-Land differs with that for ERA5. The accumulations in
> the short forecasts of ERA5-Land (with hourly steps from 01 to 24) are treated the same as those in ERA-Interim or
> ERA-Interim/Land, i.e., they are accumulated from the beginning of the forecast to the end of the forecast step. For
> example, runoff at day=D, step=12 will provide runoff accumulated from day=D, time=0 to day=D, time=12. The maximum
> accumulation is over 24 hours, i.e., from day=D, time=0 to day=D+1,time=0 (step=24).
> - HRES: accumulations are from 00 UTC to the hour ending at the forecast step
> - For the CDS time, or validity time, of 00 UTC, the accumulations are over the 24 hours ending at 00 UTC i.e. the
    > accumulation is during the previous day
> - Synoptic monthly means (stream=mnth): accumulations have units of "variable_units per forecast_step hours"
    Monthly means of daily means (stream=moda): accumulations have units that include "per day", see section Monthly means

### Downward shortwave irradiance - DSI (``sw_down``)

- CARRA: ``surface_solar_radiation_downwards`` (forecast) + ``surface_net_solar_radiation`` (forecast).
- ERA-5: ``surface_net_solar_radiation`` + ``surface_solar_radiation_downwards``
- ERA5-L: /
- THAAO (pyrgeometers): ``DSI``

### Upward shortwave irradiance - USI (``sw_up``)

- CARRA: ``surface_solar_radiation_downwards`` (forecast) + ``surface_net_solar_radiation`` (forecast).
- ERA-5: ``surface_net_solar_radiation`` + ``surface_solar_radiation_downwards``
- ERA5-L: /
- THAAO (pyrgeometers): ``USI``

### Downward longwave irradiance - DLI (``lw_down``)

- CARRA: ``thermal_surface_radiation_downwards`` (forecast) + ``surface_net_thermal_radiation`` (forecast).
- ERA-5: ``surface_net_thermal_radiation`` + ``surface_thermal_radiation_downwards``
- ERA5-L: /
- THAAO (pyranometers): ``DLI``

### Upward longwave irradiance - ULI (``lw_up``)

- CARRA: ``thermal_surface_radiation_downwards`` (forecast) + ``surface_net_thermal_radiation`` (forecast).
- ERA-5: ``surface_net_thermal_radiation`` + ``surface_thermal_radiation_downwards``
- ERA5-L: /
- THAAO (pyranometers): ``ULI``

### Surface albedo (``alb``)

- CARRA: ``albedo`` (forecast). Values masked to nan for alb<0.1, since they are unrealistic.
- ERA-5: ``forecast_albedo`` (also ``snow_albedo``)
- ERA5-L: /
- THAAO (pyrgeometers): ``DSI``+``USI``

## CLOUD & ATMOSPHERE

## Precipitation (``precip``)

- CARRA: ``total_precipitation``
- ERA-5: ``total_precipitation``
- ERA5-L: /
- THAAO (rain gauge): It is calculated as cumulative value over the resampling time.

## Cloud Base Height (``cbh``)

- CARRA: ``cloud_base``
- ERA-5: ``cloud_base_height``
- ERA5-L: /
- THAAO (ceilometer): ``tcc`` CBH is calculated as the median value over 1 h form the original 15 s time resolution,
  then averaged for the comparison.

## Total Cloud Cover (``tcc``)

- CARRA: ``total_cloud_cover``
- ERA-5: ``total_cloud_cover``
- ERA5-L: /
- THAAO (ceilometer): ``cbh`` (lowermost level)

## Liquid Water Path - LWP (``lwp``)

> [!CAUTION]
> LWP values have issues, at least for CARRA which has been divided by 10E-06 instead of 10e-03 as expected from the
> declared uom. All LWP values have been masked to nan for LWP<0.0. Perhaps HATPRO

- CARRA: ``total_column_cloud_liquid_water``
- ERA-5: ``total_column_cloud_liquid_water``
- ERA5-L: /
- THAAO (hatpro): ???