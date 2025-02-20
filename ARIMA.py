import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv("D:/ucl/Spatial-Temporal Data Analysis and Data Mining/STDMAssignment/STDM-Assignment/Data.csv",
                 parse_dates=['transit_timestamp'],
                 date_format='%m/%d/%Y %I:%M:%S %p')

df['date'] = df['transit_timestamp'].dt.date

daily_ridership = df.groupby('date')['ridership'].sum().reset_index()

ts = daily_ridership.set_index('date')['ridership']

train_days = 25
test_days = 5
train = ts.iloc[:train_days]
test = ts.iloc[train_days:train_days+test_days]

arima_model = auto_arima(train,
                        seasonal=False,
                        trace=True,
                        suppress_warnings=True,
                        error_action='ignore',
                        stepwise=True)

print(f'Best ARIMA Model Order: {arima_model.order}')

model = ARIMA(train, order=arima_model.order)
model_fit = model.fit()

forecast = model_fit.forecast(steps=test_days)
forecast_index = pd.date_range(start=train.index[-1] + pd.Timedelta(days=1), periods=test_days)

# 可视化结果
plt.figure(figsize=(12, 6))
ax = plt.gca()
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Actual Data', color='green')
plt.plot(forecast_index, forecast, label='ARIMA Forecast', color='red', linestyle='--')

# 设置日期格式
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.xticks(rotation=45)

plt.title('Subway Ridership Forecast (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Daily Ridership')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# # 计算评估指标
# mae = np.mean(np.abs(forecast - test))
# mse = np.mean((forecast - test)**2)
# rmse = np.sqrt(mse)
#
# print(f'MAE: {mae:.2f}')
# print(f'MSE: {mse:.2f}')
# print(f'RMSE: {rmse:.2f}')