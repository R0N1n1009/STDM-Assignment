import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.stations = None
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df['transit_timestamp'] = pd.to_datetime(self.df['transit_timestamp'], format='%m/%d/%Y %I:%M:%S %p')
        self.df.set_index('transit_timestamp', inplace=True)

    def preprocess_data(self):
        # Check if there is null
        if self.df.isnull().sum().sum() > 0:
            self.df['ridership'].fillna(method='ffill', inplace=True)  # First, replace the null with formal
            self.df['ridership'].interpolate(method='linear', inplace=True)  # Second, interpolation smoothing
        else:
            print('Not Null')

        # Generate temporal feature
        self.df['hour'] = self.df.index.hour
        self.df['day_of_week'] = self.df.index.dayofweek  # 0=Monday, 6=Sunday
        self.df['is_weekend'] = self.df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # if it is weekend
        self.df['day_of_month'] = self.df.index.day

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0088
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def calculate_distance_matrix(self):
        # Generate spatial feature
        self.stations = self.df[['station_complex_id', 'latitude', 'longitude']].drop_duplicates()
        self.stations = self.stations.drop_duplicates(subset=['station_complex_id'], keep='first')
        self.stations = self.stations.set_index('station_complex_id')  # reset index
        self.stations = self.stations[~self.stations.index.duplicated(keep='first')].reset_index()

        # Calculate the distances between stations
        self.stations['neighboring_stations'] = self.stations.apply(
            lambda row: {r['station_complex_id']: self.haversine(row['latitude'], row['longitude'], r['latitude'], r['longitude'])
                for _, r in self.stations[self.stations['station_complex_id'] != row['station_complex_id']].iterrows()}, axis=1)

    def save_processed_data(self, file_name="processed_subway_data.pkl"):
        self.df.to_pickle(file_name)
        print(f"Processed data saved to {file_name}")

    def load_processed_data(self, file_name="processed_subway_data.pkl"):
        self.df = pd.read_pickle(file_name)
        print(f"Processed data loaded from {file_name}")

    def get_processed_data(self):
        return self.df

    def get_stations(self):
        return self.stations


