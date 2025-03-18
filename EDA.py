import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import geopandas as gpd
import esda
import libpysal
import folium
from DataPreprocessing import DataPreprocessor
from splot.esda import moran_scatterplot
from folium.plugins import HeatMap
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

# 🔹 **Set global DPI**
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 666  # 设置全局 DPI

# 🔹 **Load and preprocess the dataset**
file_path = "D:/ucl/Spatial-Temporal Data Analysis and Data Mining/STDMAssignment/STDM-Assignment/Data5.csv"
processor = DataPreprocessor(file_path)
processor.preprocess_data()
processor.calculate_distance_matrix()

df = processor.get_processed_data()
stations = processor.get_stations()

# 🔹 **Aggregate ridership data over time**
df_grouped = df.groupby('transit_timestamp')['ridership'].sum()
df_grouped.index = pd.to_datetime(df_grouped.index)  # Ensure correct timestamp format

# Compute day values for the x-axis
x_values = (df_grouped.index - df_grouped.index.min()).days + (df_grouped.index.hour / 24)

# 🔹 **Plot total ridership trend over time**
plt.figure(figsize=(10, 4), dpi=666)  # Ensure high DPI
plt.plot(x_values, df_grouped, color="black", linewidth=0.8, linestyle="-")
plt.xlabel("Days", fontsize=12)
plt.ylabel("Total Ridership", fontsize=12)
plt.xticks(np.arange(0, x_values.max() + 1, step=5), fontsize=10)
plt.yticks(fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig("ridership_trend.png", dpi=666, bbox_inches='tight')  # Save with high DPI
plt.show()

# 🔹 **Augmented Dickey-Fuller (ADF) test for stationarity**
adf_result = adfuller(df_grouped)
print(f"ADF Statistic: {adf_result[0]:.3f}")
print(f"p-value: {adf_result[1]:.3f}")

# 🔹 **Plot ridership trend by hour of the day**
plt.figure(figsize=(12, 5), dpi=666)
sns.lineplot(x=df_grouped.index.hour, y=df_grouped.values, marker="o", color="black")
plt.xlabel("Hour of Day")
plt.ylabel("Average Ridership")
plt.xticks(range(0, 24, 2))  # Tick every 2 hours
plt.grid()
plt.savefig("hourly_trend.png", dpi=666, bbox_inches='tight')
plt.show()

# 🔹 **Plot ridership trend by day of the week**
plt.figure(figsize=(12, 5), dpi=666)
sns.barplot(x=df_grouped.index.dayofweek, y=df_grouped.values, color="black")
plt.xlabel("Day of the Week (0=Monday, 6=Sunday)")
plt.ylabel("Average Ridership")
plt.xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
plt.grid()
plt.savefig("weekly_trend.png", dpi=666, bbox_inches='tight')
plt.show()

# Perform STL decomposition with a period of 24 (daily cycle)
stl = STL(df_grouped, period=24)
res = stl.fit()

# Create subplots with increased figure size and shared X-axis
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, dpi=666)

# Plot observed ridership data
res.observed.plot(ax=axes[0], color="black", linewidth=1)
axes[0].set_ylabel("Ridership", fontsize=12)

# Plot trend component
res.trend.plot(ax=axes[1], color="blue", linewidth=1)
axes[1].set_ylabel("Trend", fontsize=12)

# Plot seasonal component
res.seasonal.plot(ax=axes[2], color="green", linewidth=1)
axes[2].set_ylabel("Seasonal", fontsize=12)

# Plot residual component with markers for better visualization
res.resid.plot(ax=axes[3], marker="o", linestyle="None", markersize=3, color="red")
axes[3].set_ylabel("Residual", fontsize=12)
axes[3].set_xlabel("Time", fontsize=12)

# Rotate X-axis labels to prevent overlapping
plt.xticks(rotation=30, fontsize=10)

# Adjust layout to prevent overlapping elements
plt.tight_layout()
plt.savefig("stl_decomposition.png", dpi=666, bbox_inches='tight')
plt.show()

# 🔹 **Global Moran's I test for spatial autocorrelation**
plt.figure(figsize=(8, 6), dpi=666)
df_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']))
df_gdf = df_gdf.groupby('station_complex_id', as_index=False).agg({
    'ridership': 'sum',
    'geometry': 'first'
})
w = libpysal.weights.KNN.from_dataframe(df_gdf, k=10)
w.transform = 'r'
moran = esda.Moran(df_gdf['ridership'], w)
moran_scatterplot(moran)
plt.title(f"Global Moran’s I Analysis (I={moran.I:.3f}, p={moran.p_sim:.3f})")
plt.savefig("moran_global.png", dpi=666, bbox_inches='tight')
plt.show()

# 🔹 **Generate subway ridership heatmap**
m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
heat_data = [[row['latitude'], row['longitude'], row['ridership']] for _, row in df.iterrows()]
HeatMap(heat_data).add_to(m)
m.save("nyc_subway_heatmap.html")
