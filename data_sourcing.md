# Data Sourcing

## Overview

The Gen-HRES project sources its meteorological data from the **NASA POWER (Prediction Of Worldwide Energy Resources)** API. This API provides satellite-derived and model-assimilated datasets specifically tailored for renewable energy, building energy efficiency, and agricultural applications. The data fetching logic is implemented in [`fetch_nasa_power.py`](fetch_nasa_power.py).

---

## Data Source: NASA POWER API

| Property | Detail |
|---|---|
| **Provider** | NASA Langley Research Center (LaRC) |
| **API Endpoint** | `https://power.larc.nasa.gov/api/temporal/hourly/point` |
| **Temporal Resolution** | Hourly |
| **Temporal Range** | January 1, 2015 – December 31, 2025 |
| **Community** | RE (Renewable Energy) |
| **Output Format** | CSV |

## Parameters Collected

Three meteorological parameters are fetched for each location:

| Parameter Code | Description | Unit |
|---|---|---|
| `ALLSKY_SFC_SW_DWN` | All-Sky Surface Shortwave Downward Irradiance (Solar Irradiance) | W/m² |
| `WS50M` | Wind Speed at 50 Meters Above Ground | m/s |
| `T2M` | Temperature at 2 Meters Above Ground | °C |

These three variables were chosen because they are the primary drivers for hybrid renewable energy system (HRES) modeling — capturing solar generation potential, wind generation potential, and thermal operating conditions simultaneously.

---

## Geographic Locations

Data is sourced for two Nigerian cities representing distinct climatic zones:

| Location | Latitude | Longitude | Climate Characteristic |
|---|---|---|---|
| **Kano** | 12.0022°N | 8.5920°E | Northern semi-arid (Sahel), higher solar irradiance, lower humidity |
| **Lagos** | 6.5244°N | 3.3792°E | Southern coastal tropical, more cloud cover, higher humidity |

By selecting both Kano and Lagos, the dataset captures a meaningful range of Nigerian weather conditions — from the dry, high-irradiance Sahel environment to the humid, cloud-prone coastal tropics.

---

## Fetching Process

The data fetching pipeline in `fetch_nasa_power.py` follows these steps:

### 1. Pre-Flight Check
Before making any API calls, the script checks whether data files already exist in the `nasa_power_data/` directory. If all expected files are present, the script exits early to avoid redundant downloads.

```python
def check_files_exist():
    for name in locations.keys():
        filename = f"nasa_power_data/{name.lower()}_meteorological_data.csv"
        if not os.path.exists(filename):
            return False
    return True
```

### 2. API Request Construction
For each location, an HTTP GET request is constructed with the following query parameters:
- **parameters**: `ALLSKY_SFC_SW_DWN,WS50M,T2M`
- **community**: `RE`
- **longitude / latitude**: Coordinates of the target location
- **start / end**: Date range in `YYYYMMDD` format
- **format**: `CSV`

### 3. Response Handling & Storage
Successful responses (HTTP 200) are written directly to CSV files in the `nasa_power_data/` directory:
- `nasa_power_data/kano_meteorological_data.csv`
- `nasa_power_data/lagos_meteorological_data.csv`

Each file contains hourly records with columns for Year, Month, Day, Hour, and the three meteorological parameters.

---

## Output Files

| File | Description | Approximate Size |
|---|---|---|
| `kano_meteorological_data.csv` | Hourly weather data for Kano (2015–2025) | ~2.8 MB |
| `lagos_meteorological_data.csv` | Hourly weather data for Lagos (2015–2025) | ~2.8 MB |

---

## Data Volume Estimate

With ~10 years of hourly data across 2 locations:
- **Hours per year**: 8,760
- **Total records per location**: ~87,600
- **Total records combined**: ~175,200 hourly observations

---

## Dependencies

The data sourcing step requires only the `requests` library for HTTP communication.

```
pip install requests
```

---

<!-- GRAPHIC PLACEHOLDER: Map of Nigeria showing Kano and Lagos locations with their coordinates -->
> 📌 **Suggested Graphic**: A map of Nigeria highlighting the two data collection sites (Kano and Lagos) with their geographic coordinates and climatic zone labels.

<!-- GRAPHIC PLACEHOLDER: Data availability timeline -->
> 📌 **Suggested Graphic**: A timeline bar chart showing the temporal coverage of the NASA POWER data (2015–2025) with any notable gaps or missing-data periods highlighted.

---

## Notes & Considerations

- The NASA POWER API returns `-999.0` as a sentinel value for missing observations. These are handled downstream in the [preprocessing](preprocessing.md) stage.
- If the specified end date extends beyond the latest available data, the API returns data up to the most recent available date.
- The script uses a simple early-exit mechanism to prevent re-downloading existing data, making the pipeline idempotent.
