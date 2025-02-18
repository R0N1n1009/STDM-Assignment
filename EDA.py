import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import geopandas as gpd
import esda
import libpysal
import folium
from DataPreprocessing import DataPreprocessor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from splot.esda import moran_scatterplot
from folium.plugins import HeatMap
from esda.moran import Moran_Local

file_path = "D:/ucl/Spatial-Temporal Data Analysis and Data Mining/STDMAssignment/STDM-Assignment/MTA_Subway_Hourly_Ridership__2020-2024_20250130.csv"
processor = DataPreprocessor(file_path)

processor.preprocess_data()

processor.calculate_distance_matrix()

df = processor.get_processed_data()
stations = processor.get_stations()


# Plot the ridership trend
# Calculate the total ridership
df_grouped = df.groupby('transit_timestamp')['ridership'].sum()

# Calculate the days
x_values = (df_grouped.index - df_grouped.index.min()).days + (df_grouped.index.hour / 24)

plt.figure(figsize=(10, 4))
plt.plot(x_values, df_grouped, color="black", linewidth=0.8, linestyle="-")

plt.xlabel("Days", fontsize=12)
plt.ylabel("Total Ridership", fontsize=12)
plt.title("NYC Subway Ridership Trend", fontsize=14)

plt.xticks(np.arange(0, x_values.max() + 1, step=5), fontsize=10)
plt.yticks(fontsize=10)

# Remove the upper and right sidebar
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()


# Make sure the format of index is timestamp
df_grouped.index = pd.to_datetime(df_grouped.index)
# ACF
plt.figure(figsize=(12, 5))
plot_acf(df_grouped, lags=48)  # 48 hours
plt.title("ACF - NYC Subway Ridership (Hourly)")
plt.show()

# PACF
plt.figure(figsize=(12, 5))
plot_pacf(df_grouped, lags=48)
plt.title("PACF - NYC Subway Ridership (Hourly)")
plt.show()

# Calculate the ridership per hour
df_grouped_hourly = df_grouped.groupby(df_grouped.index.hour).mean()

plt.figure(figsize=(12, 5))
sns.lineplot(x=df_grouped_hourly.index, y=df_grouped_hourly.values, marker="o", color="black")

plt.title("Hourly Ridership Trend")
plt.xlabel("Hour of Day")
plt.ylabel("Average Ridership")
plt.xticks(range(0, 24, 2))  # per 2 hour
plt.grid()
plt.show()

# Calculate ridership per week (0=Monday, 6=Sunday)
df_grouped_weekly = df_grouped.groupby(df_grouped.index.dayofweek).mean()

plt.figure(figsize=(12, 5))
sns.barplot(x=df_grouped_weekly.index, y=df_grouped_weekly.values, color="black")

plt.title("Ridership Trend by Day of Week")
plt.xlabel("Day of Week (0=Monday, 6=Sunday)")
plt.ylabel("Average Ridership")
plt.xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
plt.grid()
plt.show()

# Global Moran's I
df_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']))
df_gdf = df_gdf.groupby('station_complex_id', as_index=False).agg({
    'ridership': 'sum',
    'geometry': 'first'
})

w = libpysal.weights.KNN.from_dataframe(df_gdf, k=10)
w.transform = 'r'

moran = esda.Moran(df_gdf['ridership'], w)

plt.figure(figsize=(8, 6))
moran_scatterplot(moran)
plt.title(f"Global Moran’s I: {moran.I:.3f}, p-value: {moran.p_sim:.3f}")
plt.show()

# Heatmap passenger flow
m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
heat_data = [[row['latitude'], row['longitude'], row['ridership']] for _, row in df.iterrows()]
HeatMap(heat_data).add_to(m)

m.save("nyc_subway_heatmap.html")
m

# Local Moran's I
moran_local = Moran_Local(df_gdf['ridership'], w)

# Identify hot&cold spots
df_gdf['moran_cluster'] = moran_local.q
df_gdf['moran_cluster'] = df_gdf['moran_cluster'].map({
    1: "HH",
    2: "LH",
    3: "LL",
    4: "HL"
})

df_gdf.plot(column='moran_cluster', cmap='coolwarm', legend=True, figsize=(12, 6))
plt.title("NYC Local Moran’s I Result")
plt.show()
