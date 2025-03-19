import numpy as np
import pandas as pd
import geopandas as gpd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from libpysal.weights import KNN
from DataPreprocessing import DataPreprocessor
import matplotlib.pyplot as plt

# 数据加载和预处理
file_path = "D:/ucl/Spatial-Temporal Data Analysis and Data Mining/STDMAssignment/STDM-Assignment/Data5.csv"
processor = DataPreprocessor(file_path)
processor.preprocess_data()
processor.calculate_distance_matrix()
stations_df = processor.get_stations()

# KNN 空间权重（保持稀疏）
gdf = gpd.GeoDataFrame(stations_df, geometry=gpd.points_from_xy(stations_df['longitude'], stations_df['latitude']))
w = KNN.from_dataframe(gdf, k=10)
w.transform = 'r'
W_knn = w.sparse

# 时间序列数据准备
df = pd.read_csv(file_path, parse_dates=['transit_timestamp'], date_format='%m/%d/%Y %I:%M:%S %p')
df.set_index('transit_timestamp', inplace=True)
df['ridership'] = df['ridership'].fillna(method='ffill').interpolate(method='linear')
df_grouped = df.groupby(['transit_timestamp', 'station_complex_id'])['ridership'].sum().unstack()
df_grouped = df_grouped.resample('H').sum()

# 训练/测试拆分
train_days, test_days = 28, 3  # 28 天训练，3 天测试
train_start = df_grouped.index.min()
train_end = train_start + pd.Timedelta(days=train_days)
df_train = df_grouped.loc[train_start:train_end]
df_test = df_grouped.loc[train_end:train_end + pd.Timedelta(days=test_days)]

# 计算空间滞后（无需手动差分，SARIMAX 会处理 d 和 D）
Y_train = df_train.values  # Shape: (672, 428)
from scipy import sparse

Y_train_spatial_lag = sparse.csr_matrix(Y_train).dot(W_knn.T).toarray()  # Shape: (672, 428)

# 参数设置
p, d, q = 1, 0, 2  # 非季节性参数：AR=1, Diff=1, MA=3
P, D, Q, s = 1, 0, 2, 7  # 季节性参数：SAR=1, SDiff=1, SMA=1, Period=24
spatial_P = 1  # 空间滞后阶数
forecast_steps = test_days * 24  # 72

# 检查数据长度是否足够
min_length = d + p + q + (D + P + Q) * s
if Y_train.shape[0] <= min_length:
    raise ValueError(f"Training data length ({Y_train.shape[0]}) is too short for parameters: "
                     f"p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q}, s={s}")

# 训练和预测
models = {}
Y_preds_test = np.zeros((forecast_steps, df_grouped.shape[1]))

for i, station_id in enumerate(df_grouped.columns):
    # 准备外生变量（空间滞后）
    exog_train = Y_train_spatial_lag[:, i].reshape(-1, 1)  # Shape: (672, 1)
    exog_forecast = exog_train[-forecast_steps:]  # Shape: (72, 1)

    # 训练 SARIMAX 模型（包含季节性）
    try:
        model = SARIMAX(
            Y_train[:, i],  # 单站时间序列，Shape: (672,)
            exog=exog_train,
            order=(p, d, q),  # 非季节性参数
            seasonal_order=(P, D, Q, s),  # 季节性参数
            enforce_stationarity=False,
            enforce_invertibility=False,
            n_jobs = -1
        )
        models[station_id] = model.fit(disp=False)

        # 预测
        Y_preds_test[:, i] = models[station_id].forecast(steps=forecast_steps, exog=exog_forecast)
        print(f"Fitted model for station {station_id}")
    except Exception as e:
        print(f"Error fitting model for station {station_id}: {str(e)}")
        Y_preds_test[:, i] = np.nan  # 填充 NaN 以继续运行

# 计算总客流量
Y_total_real_test = df_test.values.sum(axis=1)  # Shape: (72,)
Y_total_forecast_test = Y_preds_test.sum(axis=1)  # Shape: (72,)

# 计算评估指标
mae = np.mean(np.abs(Y_total_real_test - Y_total_forecast_test))
mse = np.mean((Y_total_real_test - Y_total_forecast_test) ** 2)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((Y_total_real_test - Y_total_forecast_test) / (Y_total_real_test + 1e-10))) * 100

# 打印评估指标
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')

# 可视化
plt.figure(figsize=(12, 6))
ax = plt.gca()

# 生成横轴小时数
x_test = np.arange(len(Y_total_real_test))  # 0 ~ 71
x_forecast = np.arange(len(Y_total_forecast_test))  # 0 ~ 71

# 绘制实际数据和预测数据
plt.plot(x_test, Y_total_real_test, label='Actual Data', color='green', linewidth=1.5)
plt.plot(x_forecast, Y_total_forecast_test, label='ST-ARIMA Forecast', color='red', linestyle='--', linewidth=1.5)

# 设置横轴刻度（每 12 小时）
plt.xticks(np.arange(0, len(Y_total_real_test) + 1, step=12))
plt.xlim(0, len(Y_total_real_test))  # 限制显示范围为测试期

# 设置标签和标题
plt.xlabel('Hours (0-72)', fontsize=12)
plt.ylabel('Hourly Ridership', fontsize=12)
plt.title('NYC Total Ridership: Actual vs ST-ARIMA Forecast (Seasonal, s=24)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig("ST-ARIMA_seasonal.png", dpi=666, bbox_inches='tight')
plt.show()