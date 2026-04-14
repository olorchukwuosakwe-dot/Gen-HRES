# Preprocessing

## Overview

The preprocessing pipeline transforms raw NASA POWER CSV data into clean, normalized, and structured arrays ready for GAN training. All logic resides in [`preprocess_nasa_data.py`](preprocess_nasa_data.py). The pipeline produces three output artifacts: a long-format CSV, a NumPy array of daily profiles, and a labels CSV.

---

## Pipeline Stages

```
Raw CSVs (NASA POWER)
    │
    ▼
┌──────────────────────┐
│  1. Load & Parse      │  ← Robust header detection, column cleaning
├──────────────────────┤
│  2. Missing Values    │  ← Replace -999.0 sentinels, interpolate
├──────────────────────┤
│  3. Datetime Creation │  ← Combine YEAR/MO/DY/HR into timestamps
├──────────────────────┤
│  4. Label Engineering │  ← Season tags, extreme-day classification
├──────────────────────┤
│  5. Normalization     │  ← Min-Max to [-1, 1] range
├──────────────────────┤
│  6. Daily Profiling   │  ← Reshape into (N, 24, 3) arrays
└──────────────────────┘
    │
    ▼
Output: .npy profiles + labels CSV
```

---

## Stage 1: Loading & Parsing

NASA POWER CSV files contain metadata headers before the actual data begins. The script robustly locates the header row by scanning for the line containing `Year`, `Month`, `Day`, and `Hour`:

```python
for i, line in enumerate(lines):
    if "Year" in line and "Month" in line and "Day" in line and "Hour" in line:
        header_row = i
        break
```

After loading, column names are stripped of leading/trailing whitespace to prevent key-matching errors.

Data is loaded for both locations (**Kano** and **Lagos**) and a `Location` column is appended to each DataFrame before concatenation.

---

## Stage 2: Missing Value Treatment

NASA POWER uses `-999.0` as a sentinel value for missing or unavailable observations. The cleaning strategy applies only to the meteorological feature columns, leaving date columns intact:

| Step | Method | Purpose |
|---|---|---|
| Replace `-999.0` with `NaN` | `df.replace(-999.0, np.nan)` | Mark missing values |
| Linear interpolation | `df.interpolate(method='linear')` | Fill interior gaps smoothly |
| Backward + Forward fill | `df.bfill().ffill()` | Handle edge cases at start/end of series |

This three-step approach ensures no `NaN` values remain while preserving temporal continuity.

<!-- GRAPHIC PLACEHOLDER: Before/after missing value treatment -->
> 📌 **Suggested Graphic**: A time-series plot showing a sample feature (e.g., Solar Irradiance) before and after missing value interpolation, highlighting the `-999.0` sentinel values and their filled replacements.

---

## Stage 3: Datetime Construction

A proper `Date` column is constructed by combining the individual temporal columns:

```python
df["Date"] = pd.to_datetime(
    df["YEAR"].astype(str) + "-" +
    df["MO"].astype(str) + "-" +
    df["DY"].astype(str) + " " +
    df["HR"].astype(str) + ":00:00"
)
```

The combined DataFrame is then sorted by `[Location, Date]` to ensure chronological ordering within each site.

---

## Stage 4: Label Engineering

Two categorical labels are created for conditional GAN training:

### Season Classification

Each record is tagged as **Dry** or **Wet** season based on the month:

| Months | Season |
|---|---|
| January, February, March, November, December | Dry |
| April – October | Wet |

This reflects the West African monsoon cycle.

### Extreme Day Detection

An "extreme" day is defined as a day whose **mean daily solar irradiance** falls below **60% of its monthly average**. This identifies days with anomalously low solar resource — the critical scenarios for HRES resilience planning.

The detection process:

1. **Aggregate** hourly data to daily mean irradiance per location.
2. **Compute** the monthly mean irradiance for each location-month.
3. **Flag** days where: `daily_mean < 0.6 × monthly_mean`

```python
daily["is_extreme"] = daily["ALLSKY_SFC_SW_DWN"] < 0.6 * daily["Monthly_Irr_Mean"]
```

<!-- GRAPHIC PLACEHOLDER: Extreme day distribution -->
> 📌 **Suggested Graphic**: A histogram or calendar heatmap showing the distribution of extreme vs. normal days across the full dataset, broken down by location and season.

<!-- VALIDATION PLACEHOLDER: Extreme day statistics -->
> 📊 **Suggested Validation Table**: Summary statistics showing the count and percentage of extreme days per location and per season, confirming the threshold captures meaningful outliers (~10–20% of days).

---

## Stage 5: Normalization

All three meteorological features are normalized to the **[-1, 1]** range using global Min-Max scaling:

$$
x_{\text{norm}} = 2 \times \frac{x - x_{\min}}{x_{\max} - x_{\min}} - 1
$$

| Feature | Normalization Range |
|---|---|
| `ALLSKY_SFC_SW_DWN` → `ALLSKY_SFC_SW_DWN_norm` | [-1, 1] |
| `WS50M` → `WS50M_norm` | [-1, 1] |
| `T2M` → `T2M_norm` | [-1, 1] |

The [-1, 1] range is chosen (rather than [0, 1]) because it aligns with the `tanh` activation commonly used in GAN generator outputs, improving training stability.

<!-- GRAPHIC PLACEHOLDER: Feature distributions before and after normalization -->
> 📌 **Suggested Graphic**: Side-by-side violin plots or histograms of each feature before and after normalization, showing the transformation from physical units to the [-1, 1] range.

---

## Stage 6: Daily Profile Construction

The normalized data is reshaped into fixed-length daily profiles for GAN ingestion:

1. **Group** records by `(Location, Date)`.
2. **Filter** to complete days only (exactly 24 hourly records).
3. **Stack** the 3 normalized features into a `(24, 3)` matrix per day.

The result is a 3D NumPy array of shape **(N, 24, 3)**, where:
- **N** = number of complete days
- **24** = hours in a day
- **3** = features (solar irradiance, wind speed, temperature)

Each profile is paired with a label record containing `location`, `date`, `season`, and `is_extreme`.

---

## Output Files

| File | Format | Shape / Description |
|---|---|---|
| `preprocessed_long.csv` | CSV | Full hourly dataset with normalized features, labels, and datetime (~24 MB) |
| `preprocessed_daily_profiles.npy` | NumPy | `(N, 24, 3)` array of normalized daily weather profiles (~4.6 MB) |
| `daily_labels.csv` | CSV | Per-day metadata: location, date, season, is_extreme (~210 KB) |

---

<!-- VALIDATION PLACEHOLDER: Preprocessing summary statistics -->
> 📊 **Suggested Validation**: A summary table reporting:
> - Total complete days per location
> - Extreme day counts and percentages
> - Feature min/max before and after normalization
> - Number of missing values filled per feature

---

## Dependencies

```
pandas
numpy<2.0.0
```
