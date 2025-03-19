import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras_tuner import RandomSearch

# 🔹 Load and preprocess the dataset
file_path = "D:/ucl/Spatial-Temporal Data Analysis and Data Mining/STDMAssignment/STDM-Assignment/Data5.csv"
df = pd.read_csv(file_path, parse_dates=['transit_timestamp'], date_format='%m/%d/%Y %I:%M:%S %p')

# 🔹 Set timestamp as index and handle missing values
df.set_index('transit_timestamp', inplace=True)
if df['ridership'].isna().any():
    print("Warning: Missing values detected before filling.")
    df['ridership'] = df['ridership'].fillna(method='ffill')
    df['ridership'] = df['ridership'].interpolate(method='linear', limit_direction='both')

# 🔹 Aggregate ridership data hourly
df_hourly = df.resample('H').sum()
print(f"Total hours in data: {len(df_hourly)}")

# 🔹 Normalize the data using Min-Max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_hourly[['ridership']].values.reshape(-1, 1))

# 🔹 Split the dataset into training (28 days) and testing (3 days)
train_days = 28
test_days = 3

train_size = train_days * 24  # 24 hours per day
test_size = test_days * 24

# Ensure the dataset is sufficient for training and testing
if len(scaled_data) < train_size + test_size:
    raise ValueError(f"Data length {len(scaled_data)} is insufficient for {train_size + test_size} hours.")

train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:train_size + test_size]
print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")

# 🔹 Create sequences for LSTM input
time_steps = 24  # Use past 24 hours as input features
def create_sequences(data, time_steps=24):
    X, Y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        Y.append(data[i + time_steps])
    return np.array(X), np.array(Y)

X_train, Y_train = create_sequences(train_data, time_steps)
X_test, Y_test = create_sequences(test_data, time_steps)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# 🔹 Hyperparameter tuning using Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units1', min_value=64, max_value=256, step=64),
                   return_sequences=True, input_shape=(time_steps, 1)))
    model.add(Dropout(hp.Float('dropout1', 0.2, 0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units2', min_value=64, max_value=256, step=64),
                   return_sequences=True))
    model.add(Dropout(hp.Float('dropout2', 0.2, 0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units3', min_value=32, max_value=128, step=32)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=hp.Choice('lr', [1e-2, 1e-3, 1e-4])), loss='mse')
    return model

# 🔹 Perform hyperparameter tuning
tuner = RandomSearch(build_model, objective='val_loss', max_trials=10)
tuner.search(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test), verbose=1)

# 🔹 Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# 🔹 Perform rolling predictions for the test set
def predict_sequence(model, data, time_steps, scaler, n_future):
    current_sequence = data[:time_steps]  # Start from the beginning of test data
    predicted = []
    for _ in range(n_future):
        next_pred = model.predict(current_sequence.reshape(1, time_steps, 1), verbose=0)
        predicted.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred[0, 0]
    predicted = np.array(predicted).reshape(-1, 1)
    return scaler.inverse_transform(predicted)

# 🔹 Predict the next 72 hours
predictions = predict_sequence(best_model, test_data, time_steps, scaler, n_future=test_size)
print(f"Predictions shape: {predictions.shape}")

# 🔹 Reverse normalization for the actual test values
Y_test_actual = scaler.inverse_transform(test_data)
print(f"Y_test_actual shape: {Y_test_actual.shape}")

# 🔹 Calculate evaluation metrics (MSE, RMSE, MAE) for the first 48 hours
eval_start = time_steps  # Evaluate from the 24th hour
eval_end = test_size  # Evaluate for the entire test set
mse = mean_squared_error(Y_test_actual[eval_start:eval_end], predictions[eval_start:eval_end])
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test_actual[eval_start:eval_end], predictions[eval_start:eval_end])

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# 🔹 Visualize the LSTM forecast vs. actual values
plt.figure(figsize=(12, 6))
plt.plot(range(test_size), Y_test_actual, label="Actual Data", color='green', linewidth=2)
plt.plot(range(test_size), predictions, label="LSTM Forecast", color='red', linestyle='dashed', linewidth=2)
plt.legend()
plt.xlabel("Hour (0-72)")
plt.ylabel("Ridership")
plt.grid(True)
plt.savefig("LSTM.png", dpi=666, bbox_inches='tight')
plt.show()

# 🔹 Retrieve the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hyperparameters.values)
