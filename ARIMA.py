import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# 🔹 **Data Preprocessing**
df = pd.read_csv("D:/ucl/Spatial-Temporal Data Analysis and Data Mining/STDMAssignment/STDM-Assignment/Data5.csv",
                 parse_dates=['transit_timestamp'],
                 date_format='%m/%d/%Y %I:%M:%S %p')

# Set the timestamp as the index
df.set_index('transit_timestamp', inplace=True)

# Aggregate ridership data by timestamp
df_grouped = df.groupby('transit_timestamp')['ridership'].sum()
df_grouped.index = pd.to_datetime(df_grouped.index)  # Ensure correct datetime format

# Resample data to hourly intervals
hourly_ridership = df.resample("H")["ridership"].sum()

# 🔹 **Plot ACF and PACF**
plt.figure(figsize=(10, 5))
plot_acf(df_grouped, lags=72, title=None)
plt.xlabel("Lag (hours)", fontsize=12)
plt.ylabel("ACF", fontsize=12)
plt.savefig("acf.png", dpi=666, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5))
plot_pacf(df_grouped, lags=72, title=None)
plt.xlabel("Lag (hours)", fontsize=12)
plt.ylabel("ST-ACF", fontsize=12)
plt.savefig("pacf.png", dpi=666, bbox_inches='tight')
plt.show()

# 🔹 **Automatically select SARIMA parameters**
auto_sarima_model = auto_arima(df_grouped,
                               seasonal=True,  # Enable seasonality
                               m=24,  # Set seasonal period (e.g., 24 hours for daily cycles)
                               trace=True,  # Show search process
                               suppress_warnings=True,
                               stepwise=True,  # Use stepwise parameter selection
                               n_jobs=1,  # Run sequentially
                               )

print(auto_sarima_model.summary())  # Output the optimal SARIMA parameters

# Retrieve optimal (p,d,q) and (P,D,Q,m) values
p, q, d = auto_sarima_model.order
P, Q, D, s = auto_sarima_model.seasonal_order

# 🔹 **Split data into training and testing sets**
train_hours = 28 * 24  # 28 days for training
test_hours = 3 * 24  # 3 days for testing

train = hourly_ridership.iloc[:train_hours]
test = hourly_ridership.iloc[train_hours:train_hours + test_hours]

# # 🔹 **SARIMA model with manually selected parameters**
# sarima_model = SARIMAX(train,
#                        order=(1, 0, 2),
#                        seasonal_order=(1, 0, 2, 24),
#                        enforce_stationarity=False,
#                        enforce_invertibility=False)
#
# sarima_fit = sarima_model.fit()
# print(sarima_fit.summary())

# 🔹 **Train SARIMA model with auto-selected parameters**
sarima_model = SARIMAX(train,
                       order=(p, q, d),
                       seasonal_order=(P, Q, D, s),
                       enforce_stationarity=False,
                       enforce_invertibility=False)

sarima_fit = sarima_model.fit()
print(sarima_fit.summary())

# 🔹 **Generate forecast**
forecast = sarima_fit.get_forecast(steps=test_hours)
forecast_values = forecast.predicted_mean

# 🔹 **Plot forecast results**
plt.figure(figsize=(12, 6))
ax = plt.gca()

# Generate new x-axis index
x_test = np.arange(len(test))  # Test data x-axis (0 ~ 72)
x_forecast = np.arange(len(forecast_values))  # Forecasted data x-axis

# Plot actual test data
plt.plot(x_test, test, label='Actual Data', color='green')

# Plot forecasted data
plt.plot(x_forecast, forecast_values, label='SARIMA Forecast', color='red', linestyle="--")

# ✅ **Manually adjust x-axis ticks**
plt.xticks(np.arange(0, len(test) + len(forecast_values) + 1, step=12))  # Tick every 12 hours
plt.xlim(0, len(test))  # Display only test data range

plt.xlabel('Hours (0-72)')
plt.ylabel('Hourly Ridership')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("SARIMA.png", dpi=666, bbox_inches='tight')
plt.show()

# 🔹 **Calculate evaluation metrics**
mae = np.mean(np.abs(forecast_values - test))
mse = np.mean((forecast_values - test) ** 2)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((forecast_values - test) / test)) * 100  # Mean Absolute Percentage Error

print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')
