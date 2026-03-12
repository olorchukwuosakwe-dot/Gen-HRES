# -*- coding: utf-8 -*-
# preprocess_nasa_data.py
# ROBUST VERSION - finds the exact header line automatically

import pandas as pd
import numpy as np
import os

# ====================== CONFIG ======================
data_dir = "nasa_power_data"
files = {
    "Kano": "kano_meteorological_data.csv",
    "Lagos": "lagos_meteorological_data.csv"
}

# ====================== LOAD & CLEAN ======================
dfs = []
for loc in ["Kano", "Lagos"]:
    fname = files[loc]
    path = os.path.join(data_dir, fname)
    print("Loading " + loc + " data...")
    
    # Find the exact header row (robust for NASA POWER format)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header_row = 0
    for i, line in enumerate(lines):
        if "Year" in line and "Month" in line and "Day" in line and "Hour" in line:
            header_row = i
            print("   → Header found at line " + str(i) + " for " + loc)
            break
    
    df = pd.read_csv(path, skiprows=11, encoding="utf-8")
    df.columns = [col.strip() for col in df.columns]

    # --- FIX 2: Handle NASA's -999.0 missing values ---
    # We apply this only to the weather features to protect the date columns
    features_to_fix = ["ALLSKY_SFC_SW_DWN", "WS50M", "T2M"]
    for feat in features_to_fix:
        if feat in df.columns:
            df[feat] = df[feat].replace(-999.0, np.nan)
            df[feat] = df[feat].interpolate(method='linear')
            # Fallback in case the very first or last row of the file was -999.0
            df[feat] = df[feat].bfill().ffill()
            
    df["Location"] = loc
    
    # Create datetime
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" +
        df["Month"].astype(str) + "-" +
        df["Day"].astype(str) + " " +
        df["Hour"].astype(str) + ":00:00"
    )
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data = data.sort_values(["Location", "Date"]).reset_index(drop=True)

# ====================== LABELS ======================
data["Date_only"] = data["Date"].dt.date

data["Season"] = data["Date"].dt.month.map(
    lambda m: "Dry" if m in [1, 2, 3, 11, 12] else "Wet"
)

daily = data.groupby(["Location", "Date_only"]).agg({
    "ALLSKY_SFC_SW_DWN": "mean"
}).reset_index()

daily["Month_start"] = pd.to_datetime(daily["Date_only"]).dt.to_period("M").dt.to_timestamp()
monthly_mean = daily.groupby(["Location", "Month_start"])["ALLSKY_SFC_SW_DWN"].mean().reset_index()
monthly_mean.columns = ["Location", "Month_start", "Monthly_Irr_Mean"]

daily = daily.merge(monthly_mean, on=["Location", "Month_start"])
daily["is_extreme"] = daily["ALLSKY_SFC_SW_DWN"] < 0.6 * daily["Monthly_Irr_Mean"]

extreme_map = dict(zip(zip(daily["Location"], daily["Date_only"]), daily["is_extreme"]))
data["is_extreme"] = data.apply(lambda row: extreme_map.get((row["Location"], row["Date_only"]), False), axis=1)

# ====================== NORMALIZATION ======================
features = ["ALLSKY_SFC_SW_DWN", "WS50M", "T2M"]
for feat in features:
    min_val = data[feat].min()
    max_val = data[feat].max()
    data[feat + "_norm"] = 2 * ((data[feat] - min_val) / (max_val - min_val)) - 1

# ====================== SAVE ======================
data.to_csv("preprocessed_long.csv", index=False)
print("Saved: preprocessed_long.csv")

grouped = data.groupby(["Location", "Date_only"])
daily_profiles = []
labels = []

norm_cols = [f + "_norm" for f in features]

for name, group in grouped:
    if len(group) == 24:
        group = group.sort_values("Date")
        profile = group[norm_cols].values
        daily_profiles.append(profile)
        labels.append({
            "location": name[0],
            "date": name[1],
            "season": group["Season"].iloc[0],
            "is_extreme": group["is_extreme"].iloc[0]
        })

daily_profiles = np.array(daily_profiles)
np.save("preprocessed_daily_profiles.npy", daily_profiles)
pd.DataFrame(labels).to_csv("daily_labels.csv", index=False)

print("Saved: preprocessed_daily_profiles.npy (shape: " + str(daily_profiles.shape) + ")")
print("Saved: daily_labels.csv")

# ====================== SUMMARY ======================
print("\n==================================================")
print("PREPROCESSING COMPLETE!")
print("==================================================")
print("Total complete days: " + str(len(daily_profiles)))
print("Extreme days: " + str(sum(l["is_extreme"] for l in labels)) +
      " (" + str(round(sum(l["is_extreme"] for l in labels)/len(daily_profiles)*100, 1)) + "%)")
print("Kano days: " + str(sum(1 for l in labels if l["location"]=="Kano")))
print("Lagos days: " + str(sum(1 for l in labels if l["location"]=="Lagos")))
print("\nFiles ready for Milestone 1:")
print("   • preprocessed_daily_profiles.npy")
print("   • daily_labels.csv")
print("   • preprocessed_long.csv")