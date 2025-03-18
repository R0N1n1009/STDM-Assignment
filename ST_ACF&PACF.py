import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
import warnings

warnings.filterwarnings("ignore")  # Ignore possible warnings

# 🔹 **1. Load the data**
file_path = "D:/ucl/Spatial-Temporal Data Analysis and Data Mining/STDMAssignment/STDM-Assignment/Data5.csv"
try:
    df = pd.read_csv(file_path, parse_dates=['transit_timestamp'])
    df.set_index('transit_timestamp', inplace=True)
except FileNotFoundError:
    print("File not found, please check the path!")
    exit()

# 🔹 **2. Calculate total NYC ridership (aggregated by hour)**
hourly_ridership = df.resample("H")["ridership"].sum().fillna(0)  # Fill possible missing values with 0

# 🔹 **3. Compute and plot ST-ACF**
def compute_st_acf(series, max_lag=72):
    """Compute and plot the Spatio-Temporal Autocorrelation Function (ST-ACF)"""
    lags = np.arange(1, max_lag + 1)
    acfs = [series.autocorr(lag=lag) for lag in lags]

    # Use default colors
    plt.figure(figsize=(12, 6))
    plt.stem(lags, acfs, basefmt=" ")  # Use default colors
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)  # Add zero line
    plt.xlabel("Lag (hours)", fontsize=12)
    plt.ylabel("ST-ACF", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("st-acf.png", dpi=666, bbox_inches='tight')
    plt.show()

    return acfs

# 🔹 **4. Compute and plot ST-PACF**
def compute_st_pacf(series, max_lag=72):
    """Compute and plot the Spatio-Temporal Partial Autocorrelation Function (ST-PACF)"""
    plt.figure(figsize=(12, 6))
    plot_pacf(series, lags=max_lag, method='ywm',  # Use Yule-Walker method
              title=None,
              ax=plt.gca())
    plt.xlabel("Lag (hours)", fontsize=12)
    plt.ylabel("ST-PACF", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("st-pacf.png", dpi=666, bbox_inches='tight')
    plt.show()

# 🔹 **5. Execute computation and plotting**
st_acf_values = compute_st_acf(hourly_ridership)
compute_st_pacf(hourly_ridership)