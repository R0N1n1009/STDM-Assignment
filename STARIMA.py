import numpy as np
import pandas as pd
import geopandas as gpd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from libpysal.weights import KNN
from DataPreprocessing import DataPreprocessor
import matplotlib.pyplot as plt

# Load and preprocess data
file_path = "D:/ucl/Spatial-Temporal Data Analysis and Data Mining/STDMAssignment/STDM-Assignment/Data5.csv"
processor = DataPreprocessor(file_path)
processor.preprocess_data()
processor.calculate_distance_matrix()
stations_df = processor.get_stations()

# Construct KNN spatial weight matrix (kept sparse for efficiency)
gdf = gpd.GeoDataFrame(stations_df, geometry=gpd.points_from_xy(stations_df['longitude'], stations_df['latitude']))
w = KNN.from_dataframe(gdf, k=10)  # Use 10 nearest neighbors
w.transform = 'r'  # Row-standardized weights
W_knn = w.sparse  # Convert to sparse matrix format

# Load time series data
df = pd.read_csv(file_path, parse_dates=['transit_timestamp'], date_format='%m/%d/%Y %I:%M:%S %p')
df.set_index('transit_timestamp', inplace=True)

# Handle missing values using forward fill and linear interpolation
df['ridership'] = df['ridership'].fillna(method='ffill').interpolate(method='linear')

# Aggregate ridership data by station and resample to hourly intervals
df_grouped = df.groupby(['transit_timestamp', 'station_complex_id'])['ridership'].sum().unstack()
df_grouped = df_grouped.resample('H').sum()

# Split data into training (28 days) and testing (3 days) sets
train_days, test_days = 28, 3
train_start = df_grouped.index.min()
train_end = train_start + pd.Timedelta(days=train_days)
df_train = df_grouped.loc[train_start:train_end]
df_test = df_grouped.loc[train_end:train_end + pd.Timedelta(days=test_days)]

# Compute spatial lag matrix (SARIMAX handles differencing internally)
Y_train = df_train.values  # Shape: (672, 428)

from scipy import sparse

Y_train_spatial_lag = sparse.csr_matrix(Y_train).dot(W_knn.T).toarray()  # Shape: (672, 428)

# Set ST-ARIMA parameters
p, d, q = 1, 0, 2  # Non-seasonal parameters: AR=1, Diff=0, MA=2
P, D, Q, s = 1, 0, 2, 7  # Seasonal parameters: SAR=1, SDiff=0, SMA=2, Period=7
spatial_P = 1  # Spatial lag order
forecast_steps = test_days * 24  # Predict 72 hours

# Ensure sufficient training data length
min_length = d + p + q + (D + P + Q) * s
if Y_train.shape[0] <= min_length:
    raise ValueError(f"Training data length ({Y_train.shape[0]}) is too short for parameters: "
                     f"p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q}, s={s}")

# Train ST-ARIMA model and generate forecasts
models = {}
Y_preds_test = np.zeros((forecast_steps, df_grouped.shape[1]))

for i, station_id in enumerate(df_grouped.columns):
    # Prepare exogenous variables (spatial lag)
    exog_train = Y_train_spatial_lag[:, i].reshape(-1, 1)  # Shape: (672, 1)
    exog_forecast = exog_train[-forecast_steps:]  # Shape: (72, 1)

    # Train SARIMAX model with seasonal components
    try:
        model = SARIMAX(
            Y_train[:, i],  # Time series for a single station, Shape: (672,)
            exog=exog_train,
            order=(p, d, q),  # Non-seasonal parameters
            seasonal_order=(P, D, Q, s),  # Seasonal parameters
            enforce_stationarity=False,
            enforce_invertibility=False,
            n_jobs=-1
        )
        models[station_id] = model.fit(disp=False)

        # Forecast next 72 hours
        Y_preds_test[:, i] = models[station_id].forecast(steps=forecast_steps, exog=exog_forecast)
        print(f"Fitted model for station {station_id}")
    except Exception as e:
        print(f"Error fitting model for station {station_id}: {str(e)}")
        Y_preds_test[:, i] = np.nan  # Fill NaN to continue execution

# Compute total ridership for real and forecasted data
Y_total_real_test = df_test.values.sum(axis=1)  # Shape: (72,)
Y_total_forecast_test = Y_preds_test.sum(axis=1)  # Shape: (72,)

# Calculate evaluation metrics
mae = np.mean(np.abs(Y_total_real_test - Y_total_forecast_test))
mse = np.mean((Y_total_real_test - Y_total_forecast_test) ** 2)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((Y_total_real_test - Y_total_forecast_test) / (Y_total_real_test + 1e-10))) * 100

# Print evaluation metrics
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')

# Visualization of actual vs forecasted total ridership
plt.figure(figsize=(12, 6))
ax = plt.gca()

# Generate x-axis indices for test data
x_test = np.arange(len(Y_total_real_test))  # 0 ~ 71
x_forecast = np.arange(len(Y_total_forecast_test))  # 0 ~ 71

# Plot actual vs forecasted data
plt.plot(x_test, Y_total_real_test, label='Actual Data', color='green', linewidth=1.5)
plt.plot(x_forecast, Y_total_forecast_test, label='ST-ARIMA Forecast', color='red', linestyle='--', linewidth=1.5)

# Set x-axis ticks (every 12 hours)
plt.xticks(np.arange(0, len(Y_total_real_test) + 1, step=12))
plt.xlim(0, len(Y_total_real_test))  # Limit display to test period

# Set labels and title
plt.xlabel('Hours (0-72)', fontsize=12)
plt.ylabel('Hourly Ridership', fontsize=12)
plt.title('NYC Total Ridership: Actual vs ST-ARIMA Forecast (Seasonal, s=24)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig("ST-ARIMA_seasonal.png", dpi=666, bbox_inches='tight')
plt.show()
