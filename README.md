# STDM-Assignment
# NYC Subway Ridership Forecasting

This repository contains the implementation of various time series forecasting models to predict NYC subway ridership, including ARIMA, SARIMA, ST-ARIMA, and LSTM. The dataset is preprocessed and analyzed to extract both temporal and spatial dependencies for improved forecasting accuracy.

## Project Structure

```
📂 NYC_Subway_Forecasting
│── 📜 ARIMA.py               # ARIMA & SARIMA implementation
│── 📜 ST_ARIMA.py            # Spatio-Temporal ARIMA model
│── 📜 LSTM.py                # LSTM deep learning model
│── 📜 ST_ACF&PACF.py         # ST-ACF & ST-PACF analysis
│── 📜 Heatmap.py             # Visualization of ridership distribution
│── 📜 DataPreprocessing.py    # Data cleaning and transformation
│── 📜 EDA.py                 # Exploratory Data Analysis
│── 📜 README.md              # Project documentation
```

## Installation

To set up the environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset
The dataset consists of NYC subway ridership records, including timestamps, station IDs, and ridership counts. Data preprocessing steps include:
- Timestamp parsing and resampling
- Handling missing values
- Aggregating ridership per station and per hour
- Constructing a spatial weight matrix for ST-ARIMA

## Methodologies

### 🔹 ARIMA & SARIMA
Implemented for univariate time series forecasting using ACF/PACF analysis. SARIMA incorporates seasonal components with a period of 24 hours.

### 🔹 ST-ARIMA
Extends SARIMA by integrating a spatial weight matrix (constructed using K-Nearest Neighbors) to capture station interactions.

### 🔹 LSTM
A deep learning model trained on past ridership data, utilizing a stacked LSTM architecture optimized via Keras-Tuner.

### 🔹 ST-ACF & ST-PACF
Analyzes spatio-temporal autocorrelation to select optimal parameters for ST-ARIMA.

### 🔹 Heatmap Visualization
Generates geospatial heatmaps to visualize ridership distribution across stations.

## Results

| Model                | RMSE      | MAE       |
|----------------------|----------|----------|
| SARIMA (Manual)      | 26,387.03 | 16,508.57 |
| SARIMA (Auto)        | 33,818.07 | 25,648.28 |
| ST-ARIMA            | 118,777.15 | 82,872.42 |
| LSTM                | 15,171.61 | 10,384.08 |

## Limitations & Future Work
- **Computational Constraints**: ST-ARIMA was run with a limited seasonal period (s=24) due to hardware limitations; a weekly cycle (s=168) might yield better results.
- **Data Availability**: The dataset covers a limited period, affecting generalizability.
- **External Factors**: Weather, holidays, and service disruptions were not included but could improve model performance.

## Usage
Run the respective scripts for different models:

```bash
python ARIMA.py
python ST_ARIMA.py
python LSTM.py
```

To visualize the ridership distribution:

```bash
python Heatmap.py
```

## Acknowledgments
This project is based on NYC subway ridership data and utilizes statistical and deep learning techniques to enhance urban transit analysis.

## Contact
For any inquiries, feel free to reach out!
