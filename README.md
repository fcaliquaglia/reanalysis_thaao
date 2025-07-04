# THAAO comparison with reanalysis

> [!IMPORTANT]  
> This is a preliminary analysis.

### TODO

- [ ] recuperare da Giovanni gli rs che io non ho e devo aggiungere in archivio per l'analisi
- [ ] controllare i nan value per ogni serie e la pulizia (dati negativi, outliers?, etc)

## Environment packages needed

- python3
- scipy
- netcdf4
- matplotlib
- julian
- pandas
- xarray
- metpy
- pyarrow
- (spyder-kernels=3.0)

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


# Useful bibliography

- CARRA Documentation: https://confluence.ecmwf.int/pages/viewpage.action?pageId=272321315 
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
- [ERA5, MAR, RACMO, temp] Covi, Federico, Regine Hock, Åsa Rennermalm, Xavier Fettweis, and Brice Noël. 2025.
  ‘Spatiotemporal Variability of Air Temperature Biases in Regional Climate Models over the Greenland Ice Sheet’.
  Journal of Glaciology 71:e64. https://doi.org/10.1017/jog.2025.38.
- [CARRA PWV] https://doi.org/10.1016/j.jastp.2025.106431

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
- THAAO (vespa): values in these periods have been excluded: 2021-10-11 --> 2021-10-19 and 2024-4-28 --> 2024-5-4
- THAAO (aws_ECAPAC):

### Surface temperature (``temp``)

- CARRA: ``2m_temperature``
- ERA-5: ``2m_temperature``
- THAAO (vespa):
- THAAO (aws_ECAPAC):

### Relative Humidity (``rh``)

- CARRA: ``2m_relative_humidity``
- ERA-5: ``2m_dewpoint_temperature`` + ``2m_temperature`` (descrivere processo per ottenere rh)
- THAAO (vespa):
- THAAO (aws_ECAPAC):

### Wind Direction (``windd``)

- CARRA:``10m_wind_direction``
- ERA-5: ``10m_u_component_of_wind`` + ``10m_v_component_of_wind`` (descrivere processo per ottenere velocità e
  direzione)
- THAAO (aws_ECAPAC):

### Wind Speed (``winds``)

- CARRA: ``10m_wind_speed``
- ERA-5: ``10m_u_component_of_wind`` + ``10m_v_component_of_wind`` (descrivere processo per ottenere velocità e
  direzione)
- THAAO (aws_ECAPAC):

## RADIATION

> [!WARNING]
> For CARRA radiation values. These forecast variables are released t different leadtimes, with 1-hour frequency.
> Therefore, we consider only leadtime 1, obtaining every three hours, hourly forecast valued for the following hour
> w.r.t the chosen timeframe. For example, we choose April 1, 2023 at 6:00 UTC, we analyze forecast values on April 1,
> 2023 at 7:00 UTC. All the radiation dataset have been cleaned for values <0.
> Therefore, radiation values are provided at 1hour resolution BUT every 3-hour interval!

### Downward shortwave irradiance - DSI (``sw_down``)

- CARRA: ``surface_solar_radiation_downwards`` (forecast) + ``surface_net_solar_radiation`` (forecast).
- ERA-5: ``surface_net_solar_radiation`` + ``surface_solar_radiation_downwards``
- THAAO (pyrgeometers): ``DSI``

### Upward shortwave irradiance - USI (``sw_up``)

- CARRA: ``surface_solar_radiation_downwards`` (forecast) + ``surface_net_solar_radiation`` (forecast).
- ERA-5: ``surface_net_solar_radiation`` + ``surface_solar_radiation_downwards``
- THAAO (pyrgeometers): ``USI``

### Downward longwave irradiance - DLI (``lw_down``)

- CARRA: ``thermal_surface_radiation_downwards`` (forecast) + ``surface_net_thermal_radiation`` (forecast).
- ERA-5: ``surface_net_thermal_radiation`` + ``surface_thermal_radiation_downwards``
- THAAO (pyranometers): ``DLI``

### Upward longwave irradiance - ULI (``lw_up``)

- CARRA: ``thermal_surface_radiation_downwards`` (forecast) + ``surface_net_thermal_radiation`` (forecast).
- ERA-5: ``surface_net_thermal_radiation`` + ``surface_thermal_radiation_downwards``
- THAAO (pyranometers): ``ULI``

## CLOUDS & ATMOSPHERE
CARRA:
All cloud and water vapour variables are instantaneous, 
i.e. they are given for the time step at which they are output. 
Vertically integrated water vapour is given in units of kg/m2. 
It is vertically integrated from the surface to the top of the atmosphere. 
In practice it is computed from the specific water vapour on the 65 model levels 
(see section 4.4). Likewise, integrated cloud liquid water, integrated cloud ice water, 
and integrated graupel are computed from the specific cloud liquid water, 
cloud ice and graupel on the 65 model levels.

Total cloud cover, as given in the output, is computed from model level cloud cover 
with the nearly maximum-random cloud overlap assumption with a scaling 
coefficient of 0.8. If this had been 0.0 random cloud overlap would be assumed, 
which means that the cloud covers at the model levels are assumed to be independent. 
If the scaling coefficient had been 1.0 maximum-random cloud overlap is assumed, 
which means that all vertically connected cloud layers are assumed to overlap 
perfectly. Note that this definition of cloud cover is not consistent 
with maximum-random cloud cover used within the model for computing radiative 
fluxes! The same nearly maximum-random cloud overlap assumption is used to 
compute high, medium and low cloud covers. Following the WMO definitions 
high cloud cover is above 5 km height, while low cloud cover is below or at 
2 km height. Medium cloud cover is in between. Note that height here is 
considered relative to the surface! Fog is the cloud cover at the lowest 
model level that has a thickness of approximately 20 m. The unit for all cloud 
cover and fog output is % in the range from 0% to 100%.

### Precipitation (``precip``)

- CARRA: ``total_precipitation``
- ERA-5: ``total_precipitation``
- THAAO (rain gauge): It is calculated as cumulative value over the resampling time.

### Cloud Base Height (``cbh``)

- CARRA: ``cloud_base``
- ERA-5: ``cloud_base_height``
- THAAO (ceilometer): ``tcc`` CBH is calculated as the median value over 1 h form the original 15 s time resolution,
  then averaged for the comparison.

### Total Cloud Cover (``tcc``)

- CARRA: ``total_cloud_cover``
- ERA-5: ``total_cloud_cover``
- THAAO (ceilometer): ``cbh`` (lowermost level)

### Liquid Water Path - LWP (``lwp``)

> [!CAUTION]
> CARRA LWP values have issues or ERA5?

- CARRA: ``total_column_cloud_liquid_water``
- ERA-5: ``total_column_cloud_liquid_water``
- THAAO (hatpro): LWP

## Integrated water vapour - IWV (``iwv``)

> [!WARNING]
- CARRA: ``total_column_integrated_water_vapour``
> C'è un problema per CARRA iwv nel dicembre 2023. i valori sono
> tutti nulli. Ho provato a riscaricare a fine 2024 ma non cambia. 

- ERA-5: ``total_column_water_vapour``
- THAAO (rs): The vertical integration for rs is **missing**.
- THAAO (vespa):

> [!WARNING]
- THAAO (hatpro):