# THAAO Comparison with Reanalysis

> \[!IMPORTANT\]\
> This is a preliminary analysis.

------------------------------------------------------------------------

## TODO

-   [ ] (to be filled)

------------------------------------------------------------------------

## Environment Packages Needed

-   python3\
-   scipy\
-   netcdf4\
-   matplotlib\
-   julian\
-   pandas\
-   xarray\
-   metpy\
-   pyarrow\
-   pyCompare\
-   skillmetrics\
-   (spyder-kernels=3.0)

------------------------------------------------------------------------

## Reanalysis Considered

### CARRA (Copernicus Arctic Regional Reanalysis)

-   Resolution: 3h at 2.5 km\
-   DOI: 10.24381/cds.713858f6\
-   Model: HARMONIE-AROME (non-hydrostatic)\
-   Features:
    -   3-hourly analyses & 1-hour forecasts (up to 30h, initialized at
        00 & 12 UTC)\
    -   High-resolution physiography & satellite data\
    -   Assimilation of local Nordic & Greenland observations\
    -   Focused on Arctic climate variability
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

> [!NOTE]
> All cloud and water vapour variables are instantaneous, 
> i.e. they are given for the time step at which they are output. 
> Vertically integrated water vapour is given in units of kg/m2. 
> It is vertically integrated from the surface to the top of the atmosphere. 
> In practice it is computed from the specific water vapour on the 65 model levels 
> (see section 4.4). Likewise, integrated cloud liquid water, integrated cloud ice water, 
> and integrated graupel are computed from the specific cloud liquid water, 
> cloud ice and graupel on the 65 model levels.
> 
> Total cloud cover, as given in the output, is computed from model level cloud cover 
> with the nearly maximum-random cloud overlap assumption with a scaling 
> coefficient of 0.8. If this had been 0.0 random cloud overlap would be assumed, 
> which means that the cloud covers at the model levels are assumed to be independent. 
> If the scaling coefficient had been 1.0 maximum-random cloud overlap is assumed, 
> which means that all vertically connected cloud layers are assumed to overlap 
> perfectly. Note that this definition of cloud cover is not consistent 
> with maximum-random cloud cover used within the model for computing radiative 
> fluxes! The same nearly maximum-random cloud overlap assumption is used to 
> compute high, medium and low cloud covers. Following the WMO definitions 
> high cloud cover is above 5 km height, while low cloud cover is below or at 
> 2 km height. Medium cloud cover is in between. Note that height here is 
> considered relative to the surface! Fog is the cloud cover at the lowest 
> model level that has a thickness of approximately 20 m. The unit for all cloud 
> cover and fog output is % in the range from 0% to 100%.

------------------------------------------------------------------------

### ERA5

-   Resolution: 1h at 0.25° × 0.25° (\~31 km)\
-   DOI: 10.24381/cds.143582cf\
-   Model: ECMWF IFS (137 vertical levels, up to 80 km)\
-   Features:
    -   Global reanalysis from 1940--present\
    -   Hourly atmospheric, land, and ocean variables\
    -   Includes ensemble at half resolution\
    -   Assimilates satellite + ground-based observations
    
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
------------------------------------------------------------------------

## THAAO Reference Instruments

-   **HATPRO** → LWP\
-   **AWS_ECAPAC** → T, P, RH, wind direction, wind speed\
-   **AWS_vespa** → T, P, RH, wind direction, wind speed\
-   **Radiometers** → SW/LW irradiance (up & down)

> \[!IMPORTANT\]\
> Reference values are from THAAO, except:\
> - IWV → VESPA\
> - LWP → HATPRO

> \[!TIP\]\
> MetPy used for conversions (e.g., dewpoint → RH, wind components).

------------------------------------------------------------------------

## Statistics

### Time Series

Standard analysis.

### Scatterplots

-   Seasonal scatterplots → plotted as points.\
-   Full dataset scatterplots → 2D density histograms (colorbar =
    density).

### Bland-Altman Plots

Reference: *Validation of the Cloud_CCI cloud products in the Arctic
(2023)*.

### Metrics

-   **N** → number of points used.\
-   **MAE** → `np.nanmean(np.abs(y - x))`\
-   **R** → Pearson correlation coefficient.\
-   **RMSE** → `np.sqrt(np.nanmean((y - x) ** 2))`

------------------------------------------------------------------------

## Variables

### WEATHER

-   **Surface Pressure (`surf_pres`)**
    -   Masked for \<900 hPa (unrealistic).\
    -   CARRA: `surface_pressure`\
    -   ERA5: `surface_pressure`\
    -   THAAO (vespa): excluded 2021-10-11 → 2021-10-19, 2024-04-28 →
        2024-05-04\
    -   THAAO (aws_ECAPAC): available
-   **Surface Temperature (`temp`)**
    -   CARRA: `2m_temperature`\
    -   ERA5: `2m_temperature`\
    -   THAAO (vespa, aws_ECAPAC)
-   **Relative Humidity (`rh`)**
    -   CARRA: `2m_relative_humidity`\
    -   ERA5: from `2m_temperature` + `2m_dewpoint_temperature`\
    -   THAAO (vespa, aws_ECAPAC)
-   **Wind Direction (`windd`)**
    -   CARRA: `10m_wind_direction`\
    -   ERA5: computed from `u10` + `v10`\
    -   THAAO (aws_ECAPAC)
-   **Wind Speed (`winds`)**
    -   CARRA: `10m_wind_speed`\
    -   ERA5: computed from `u10` + `v10`\
    -   THAAO (aws_ECAPAC)

------------------------------------------------------------------------

### RADIATION

> \[!WARNING\]\
> CARRA radiation → forecast variables, only leadtime=1 considered.\
> Effective resolution: hourly values every 3 hours.\
> All values \<0 removed.

-   **Downward SW (`sw_down`)**
    -   CARRA: `ssrd` + `ssr`\
    -   ERA5: `ssrd` + `ssr`\
    -   THAAO: pyrgeometers (`DSI`)
-   **Upward SW (`sw_up`)**
    -   CARRA: `ssrd` + `ssr`\
    -   ERA5: `ssrd` + `ssr`\
    -   THAAO: pyrgeometers (`USI`)
-   **Downward LW (`lw_down`)**
    -   CARRA: `strd` + `str`\
    -   ERA5: `strd` + `str`\
    -   THAAO: pyranometers (`DLI`)
-   **Upward LW (`lw_up`)**
    -   CARRA: `strd` + `str`\
    -   ERA5: `strd` + `str`\
    -   THAAO: pyranometers (`ULI`)

------------------------------------------------------------------------

### CLOUDS & ATMOSPHERE

-   Cloud variables (CARRA):
    -   Instantaneous, vertically integrated from 65 model levels.\
    -   Cloud cover computed with **nearly max-random overlap
        (scaling=0.8)**.
-   **Precipitation (`precip`)**
    -   CARRA: `total_precipitation`\
    -   ERA5: `total_precipitation`\
    -   THAAO: rain gauge (cumulative over resampling time)
-   **Cloud Base Height (`cbh`)**
    -   CARRA: `cloud_base`\
    -   ERA5: `cloud_base_height`\
    -   THAAO: ceilometer (`median 1h from 15s data`)
-   **Total Cloud Cover (`tcc`)**
    -   CARRA: `total_cloud_cover`\
    -   ERA5: `total_cloud_cover`\
    -   THAAO: ceilometer (lowermost layer)
-   **Liquid Water Path (`lwp`)**\
    \> \[!CAUTION\] Possible issues in either CARRA or ERA5 values.
    -   CARRA: `tclw`\
    -   ERA5: `tclw`\
    -   THAAO: HATPRO
-   **Integrated Water Vapour (`iwv`)**\
    \> \[!WARNING\]\
    \> CARRA IWV missing values in Dec 2023 (all NaN). Re-download
    didn't fix.
    -   CARRA: `tcwv`\
    -   ERA5: `tcwv`\
    -   THAAO:
        -   VESPA: available\
        -   HATPRO: available (⚠️ flagged)


## Useful bibliography

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
