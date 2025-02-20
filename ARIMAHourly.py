import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima


# 🔹 **数据预处理**
df = pd.read_csv("D:/ucl/Spatial-Temporal Data Analysis and Data Mining/STDMAssignment/STDM-Assignment/Data.csv",
                     parse_dates=['transit_timestamp'],
                     date_format='%m/%d/%Y %I:%M:%S %p')

# 设置时间索引
df.set_index('transit_timestamp', inplace=True)


hourly_ridership = df.resample("H")["ridership"].sum()


# 🔹 **转换为时间序列**
ts = hourly_ridership

# 🔹 **划分训练集和测试集（按小时）**
train_hours = 25 * 24  # 25 天的小时数据
test_hours = 5 * 24  # 预测未来 6 天

train = ts.iloc[:train_hours]
test = ts.iloc[train_hours:train_hours + test_hours]

# # 🔹 **自动选择最佳 ARIMA 参数**
# arima_model = auto_arima(train,
#                          seasonal=False,
#                          trace=True,
#                          suppress_warnings=True,
#                          error_action='ignore',
#                          stepwise=True)

# print(f'Best ARIMA Model Order: {arima_model.order}')

# 🔹 **训练最终 ARIMA 模型**
model_manual = ARIMA(train, order=(24, 1, 24))
model_manual_fit = model_manual.fit()
#
# model = ARIMA(train, order=arima_model.order)
# model_fit = model.fit()

# 🔹 **生成预测**
forecast = model_manual_fit.forecast(steps=test_hours)
forecast_index = pd.date_range(start=train.index[-1] + pd.Timedelta(hours=1), periods=test_hours, freq="H")

# 🔹 **可视化结果**
# plt.figure(figsize=(12, 6))
# ax = plt.gca()
# plt.plot(train.index, train, label='Training Data')
# plt.plot(test.index, test, label='Actual Data', color='green')
# plt.plot(forecast_index, forecast, label='ARIMA Forecast', color='red', linestyle="--")
#
# # 🔹 **设置 X 轴格式**
# ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 每天显示一个刻度
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
# plt.xticks(rotation=45)
#
# plt.title('Hourly NYC Subway Ridership Forecast (ARIMA)')
# plt.xlabel('Time')
# plt.ylabel('Hourly Ridership')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# 🔹 **计算评估指标**
mae = np.mean(np.abs(forecast - test))
mse = np.mean((forecast - test) ** 2)
rmse = np.sqrt(mse)

print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')


# plt.figure(figsize=(12, 6))
# ax = plt.gca()
#
# # 🔹 **绘制训练数据（如果你想完全隐藏训练数据，这行可以注释掉）**
# # plt.plot(train.index, train, label='Training Data', alpha=0.3)  # 透明度降低
#
# # 🔹 **绘制测试数据**
# plt.plot(test.index, test, label='Actual Data', color='green')
#
# # 🔹 **绘制预测数据**
# plt.plot(forecast_index, forecast, label='ARIMA Forecast', color='red', linestyle="--")
#
# # **设置 x 轴格式，使其以小时为单位**
# ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # 每 6 小时显示一个刻度
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 显示格式为 "小时:分钟"
#
# # ✅ **只显示测试数据和预测数据的部分**
# plt.xlim(test.index.min(), forecast_index.max())  # 只显示测试和预测部分
#
# plt.xticks(rotation=45)
# plt.title('Hourly NYC Subway Ridership Forecast (ARIMA)')
# plt.xlabel('Hour')
# plt.ylabel('Hourly Ridership')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(12, 6))
ax = plt.gca()

# 🔹 **生成 0 ~ 120 作为新的 x 轴**
x_test = np.arange(len(test))  # 测试数据横坐标 (0 ~ 120)
x_forecast = np.arange(len(forecast))  # 预测数据横坐标 (120 ~ 240)

# 🔹 **绘制测试数据**
plt.plot(x_test, test, label='Actual Data', color='green')

# 🔹 **绘制预测数据**
plt.plot(x_forecast, forecast, label='ARIMA Forecast', color='red', linestyle="--")

# ✅ **手动设置横坐标刻度**
plt.xticks(np.arange(0, len(test) + len(forecast) + 1, step=12))  # 每 12 小时一个刻度
plt.xlim(0, len(test))  # 只显示 0~120 小时

plt.xlabel('Hours (0-120)')
plt.ylabel('Hourly Ridership')
plt.title('Hourly NYC Subway Ridership Forecast (ARIMA)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 🔹 **计算评估指标**
mae = np.mean(np.abs(forecast - test))
mse = np.mean((forecast - test) ** 2)
rmse = np.sqrt(mse)

print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')