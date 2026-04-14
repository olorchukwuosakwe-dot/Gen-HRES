# validate_gan.py
# Validates GAN output using K-S Test, J-S Divergence, and Autocorrelation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

print("Loading datasets for validation...")

# 1. Load the Data
# We only want to compare the GAN's "extreme" output against the REAL "extreme" days
real_df = pd.read_csv('preprocessed_long.csv')
real_extreme = real_df[real_df['is_extreme'] == True]

fake_df = pd.read_csv('simulink_extreme_forecast.csv')

# Feature mapping (Real Column Name -> Fake Column Name -> Plot Title)
features = [
    ('ALLSKY_SFC_SW_DWN', 'Solar_Irradiance_W_m2', 'Solar Irradiance (W/m²)'),
    ('WS50M', 'Wind_Speed_m_s', 'Wind Speed (m/s)'),
    ('T2M', 'Temperature_C', 'Temperature (°C)')
]

# Set up the plotting grid for the IEEE paper
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

print("\n=== Validation Metrics ===")

for i, (real_col, fake_col, title) in enumerate(features):
    real_data = real_extreme[real_col].dropna().values
    fake_data = fake_df[fake_col].dropna().values
    
    # ---------------------------------------------------------
    # Metric 1 & 2: K-S Test and J-S Divergence
    # ---------------------------------------------------------
    # K-S Test
    ks_stat, p_val = ks_2samp(real_data, fake_data)
    
    # J-S Divergence (requires aligned histograms)
    bins = np.histogram_bin_edges(np.concatenate([real_data, fake_data]), bins=50)
    real_hist, _ = np.histogram(real_data, bins=bins, density=True)
    fake_hist, _ = np.histogram(fake_data, bins=bins, density=True)
    
    # Add a tiny epsilon to prevent divide-by-zero
    real_hist = real_hist + 1e-10
    fake_hist = fake_hist + 1e-10
    js_div = jensenshannon(real_hist, fake_hist)
    
    print(f"\n{title}:")
    print(f"  -> K-S Statistic: {ks_stat:.4f} (Closer to 0 is better)")
    print(f"  -> J-S Divergence: {js_div:.4f} (Closer to 0 is better)")
    
    # Plot 1: Marginal Distributions (Histograms)
    ax_dist = axes[i, 0]
    ax_dist.hist(real_data, bins=50, alpha=0.5, density=True, label='Real Extreme', color='blue')
    ax_dist.hist(fake_data, bins=50, alpha=0.5, density=True, label='GAN Synthetic', color='orange')
    ax_dist.set_title(f"Distribution: {title}")
    ax_dist.set_ylabel("Density")
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)
    
    # ---------------------------------------------------------
    # Metric 3: Autocorrelation (Temporal Memory)
    # ---------------------------------------------------------
    # Calculate Autocorrelation for 24 hours (Lag 1 to 24)
    lags = 24
    
    def calc_acf(data, max_lag):
        s = pd.Series(data)
        return [s.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
    
    real_acf = calc_acf(real_data, lags)
    fake_acf = calc_acf(fake_data, lags)
    
    # Plot 2: Autocorrelation
    ax_acf = axes[i, 1]
    ax_acf.plot(range(1, lags + 1), real_acf, marker='o', label='Real Data', color='blue')
    ax_acf.plot(range(1, lags + 1), fake_acf, marker='x', label='GAN Synthetic', color='orange', linestyle='--')
    ax_acf.set_title(f"Autocorrelation (24h): {title}")
    ax_acf.set_xlabel("Lag (Hours)")
    ax_acf.set_ylabel("ACF")
    ax_acf.legend()
    ax_acf.grid(True, alpha=0.3)

# Save the plot for the IEEE Paper
plt.savefig("validation_results.png", dpi=300, bbox_inches='tight')
print("\n✅ Validation complete! Saved plots to 'validation_results.png'")