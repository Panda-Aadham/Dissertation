from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from dotenv import load_dotenv
import numpy as np
import warnings
import requests
import json
import time
import os

warnings.filterwarnings('ignore')
load_dotenv()

class OpenAQDataPipeline:
    """
    A comprehensive pipeline for fetching and processing air quality data from OpenAQ API
    for neural network training on AQI classification tasks.
    """
    
    """
    Initialise the OpenAQ data pipeline.
    
    Args:
        api_key: OpenAQ API key
    """
    def __init__(self, api_key: str):
        self.base_url = "https://api.openaq.org/v3"
        self.headers = {}
        if api_key:
            self.headers['X-API-Key'] = api_key
        
        # AQI breakpoints for classification (US EPA standard)
        self.aqi_breakpoints = {
            'pm25': [(0, 12.0), (12.1, 35.4), (35.5, 55.4), (55.5, 150.4), (150.5, 250.4), (250.5, 500.4)],
            'pm10': [(0, 54), (55, 154), (155, 254), (255, 354), (355, 424), (425, 604)],
            'o3': [(0, 54), (55, 70), (71, 85), (86, 105), (106, 200), (201, 504)],
            'no2': [(0, 53), (54, 100), (101, 360), (361, 649), (650, 1249), (1250, 2049)],
            'so2': [(0, 35), (36, 75), (76, 185), (186, 304), (305, 604), (605, 1004)],
            'co': [(0, 4.4), (4.5, 9.4), (9.5, 12.4), (12.5, 15.4), (15.5, 30.4), (30.5, 50.4)]
        }
        
        self.aqi_categories = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 
                              'Unhealthy', 'Very Unhealthy', 'Hazardous']
        
        # Parameter mapping
        self.parameter_mapping = {
            'pm25': ['pm25', 'pm2.5'],
            'pm10': ['pm10'],
            'o3': ['o3'],
            'no2': ['no2'],
            'so2': ['so2'],
            'co': ['co']
        }
        
    """
    Fetch air quality monitoring locations.
    
    Args:
        country_iso: ISO country code (default: US)
        limit: Maximum number of locations to fetch
        
    Returns:
        DataFrame with location information
    """
    def get_locations(self, country_iso: str = 'US', limit: int = 1000) -> pd.DataFrame:
        url = f"{self.base_url}/locations"
        params = {
            'iso': country_iso,
            'limit': limit,
            'order_by': 'id'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'results' in data:
                locations_df = pd.json_normalize(data['results'])
                print(f"Fetched {len(locations_df)} locations from {country_iso}")
                return locations_df
            else:
                print("No locations found")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching locations: {e}")
            return pd.DataFrame()
    
    """
    Fetch available parameters (pollutants).
    
    Returns:
        DataFrame with parameter information
    """
    def get_parameters(self) -> pd.DataFrame:
        url = f"{self.base_url}/parameters"
        params = {'limit': 1000}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'results' in data:
                parameters_df = pd.json_normalize(data['results'])
                print(f"Fetched {len(parameters_df)} parameters")
                return parameters_df
            else:
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching parameters: {e}")
            return pd.DataFrame()
    
    """
    Fetch measurements for a specific location using the sensors endpoint.
    
    Args:
        location_id: Location ID
        date_from: Start date (YYYY-MM-DD format)
        date_to: End date (YYYY-MM-DD format)
        limit: Maximum number of measurements
        
    Returns:
        DataFrame with measurements
    """
    def get_measurements_by_location(self, location_id: int, 
                                   date_from: str, date_to: str,
                                   limit: int = 1000) -> pd.DataFrame:
        # First get sensors for this location
        sensors_url = f"{self.base_url}/locations/{location_id}/sensors"
        
        try:
            sensors_response = requests.get(sensors_url, headers=self.headers)
            sensors_response.raise_for_status()
            sensors_data = sensors_response.json()
            
            if 'results' not in sensors_data:
                return pd.DataFrame()
            
            all_measurements = []
            
            for sensor in sensors_data['results']:
                sensor_id = sensor.get('id')
                parameter = sensor.get('parameter', {}).get('name', '').lower()
                
                if not sensor_id:
                    continue
                # HERE
                # Get measurements for this sensor
                measurements_url = f"{self.base_url}/sensors/{sensor_id}/measurements"
                params = {
                    'datetime_to': date_to,
                    'datetime_from': date_from,
                    'limit': limit
                }
                
                try:
                    time.sleep(0.1)
                    measurements_response = requests.get(measurements_url, 
                                                       headers=self.headers, 
                                                       params=params)
                    measurements_response.raise_for_status()
                    measurements_data = measurements_response.json()
                    
                    if 'results' in measurements_data:
                        for measurement in measurements_data['results']:
                            measurement['location_id'] = location_id
                            measurement['sensor_id'] = sensor_id
                            measurement['parameter_name'] = parameter
                            all_measurements.append(measurement)

                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code if e.response else None

                    if e.response.status_code == 429:
                        headers = e.response.headers
                        retry_after = int(headers.get("X-Ratelimit-Reset", 5))
                        print(f"Rate limited. Retrying after {retry_after} seconds.")
                        time.sleep(retry_after)
                    else:
                        print(f"HTTP error fetching measurements for sensor {e.response.status_code}")
                        print(f"HTTP error fetching measurements for sensor {sensor_id}: {e}")
                    
                    continue

                except requests.exceptions.RequestException as e:
                    print(f"Non-HTTP error fetching measurements for sensor {sensor_id}: {e}")
                    continue
            
            if all_measurements:
                measurements_df = pd.json_normalize(all_measurements)
                print(f"Fetched {len(measurements_df)} measurements for location {location_id}")
                return measurements_df
            else:
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching sensors for location {location_id}: {e}")
            return pd.DataFrame()
    
    """
    Calculate AQI value and category for a given concentration and parameter.
    
    Args:
        concentration: Pollutant concentration
        parameter: Parameter name (pm25, pm10, o3, no2, so2, co)
        
    Returns:
        Tuple of (AQI value, AQI category)
    """
    def calculate_aqi(self, concentration: float, parameter: str) -> Tuple[int, str]:
        if parameter not in self.aqi_breakpoints:
            return 0, 'Unknown'
        
        breakpoints = self.aqi_breakpoints[parameter]
        aqi_breakpoints = [(0, 50), (51, 100), (101, 150), (151, 200), (201, 300), (301, 500)]
        
        for i, (bp_low, bp_high) in enumerate(breakpoints):
            if bp_low <= concentration <= bp_high:
                aqi_low, aqi_high = aqi_breakpoints[i]
                # LErp
                aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (concentration - bp_low) + aqi_low
                return int(aqi), self.aqi_categories[i]
        
        # If concentration exceeds all breakpoints, return hazardous
        return 500, 'Hazardous'
    
    """
    Process raw measurements into a clean dataset suitable for ML.
    
    Args:
        measurements_df: Raw measurements DataFrame
        
    Returns:
        Processed DataFrame
    """
    def process_measurements(self, measurements_df: pd.DataFrame) -> pd.DataFrame:
        if measurements_df.empty:
            return pd.DataFrame()
        
        # Convert datetime
        if 'coverage.datetimeTo.utc' in measurements_df.columns:
            measurements_df['datetime'] = pd.to_datetime(measurements_df['coverage.datetimeTo.utc'])
        elif 'datetime.instant' in measurements_df.columns:
            measurements_df['datetime'] = pd.to_datetime(measurements_df['datetime.instant'])
        elif 'datetime' in measurements_df.columns:
            measurements_df['datetime'] = pd.to_datetime(measurements_df['datetime'])
        else:
            print("Warning: No datetime column found")
            return pd.DataFrame()

        
        # Clean parameter names
        measurements_df['parameter_clean'] = measurements_df['parameter_name'].str.lower().str.strip()
        
        # Filter for key pollutants
        key_pollutants = ['pm25', 'pm2.5', 'pm10', 'o3', 'no2', 'so2', 'co']
        measurements_df = measurements_df[
            measurements_df['parameter_clean'].isin(key_pollutants)
        ].copy()
        
        # Standardise parameter names
        measurements_df['parameter_standard'] = measurements_df['parameter_clean'].replace({
            'pm2.5': 'pm25'
        })
        
        # Extract concentration values
        if 'value' in measurements_df.columns:
            measurements_df['concentration'] = pd.to_numeric(measurements_df['value'], errors='coerce')
        else:
            print("Warning: No 'value' column found in measurements")
            return pd.DataFrame()
        
        # Remove invalid measurements
        measurements_df = measurements_df.dropna(subset=['concentration'])
        measurements_df = measurements_df[measurements_df['concentration'] >= 0]
        
        # Calculate AQI for each measurement
        aqi_data = []
        for _, row in measurements_df.iterrows():
            aqi_value, aqi_category = self.calculate_aqi(
                row['concentration'], 
                row['parameter_standard']
            )
            aqi_data.append({
                'aqi_value': aqi_value,
                'aqi_category': aqi_category
            })
        
        aqi_df = pd.DataFrame(aqi_data)
        measurements_df = pd.concat([measurements_df.reset_index(drop=True), aqi_df], axis=1)
        print(measurements_df)
        
        # Add temporal features
        measurements_df['hour'] = measurements_df['datetime'].dt.hour
        measurements_df['day_of_week'] = measurements_df['datetime'].dt.dayofweek
        measurements_df['month'] = measurements_df['datetime'].dt.month
        measurements_df['season'] = measurements_df['month'].apply(self._get_season)
        
        return measurements_df
    
    """Get season from month."""
    def _get_season(self, month: int) -> str:
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    """
    Create time series features for neural network training.
    
    Args:
        df: Processed measurements DataFrame
        location_col: Column name for location identifier
        parameter_col: Column name for parameter identifier
        window_sizes: List of window sizes for rolling features
        
    Returns:
        DataFrame with time series features
    """
    def create_time_series_features(self, df: pd.DataFrame, 
                                  location_col: str = 'location_id',
                                  parameter_col: str = 'parameter_standard',
                                  window_sizes: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        df_sorted = df.sort_values(['location_id', 'parameter_standard', 'datetime']).copy()
        
        # Create rolling features for each location-parameter combination
        for window in window_sizes:
            df_sorted[f'concentration_mean_{window}h'] = df_sorted.groupby(
                [location_col, parameter_col]
            )['concentration'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            
            df_sorted[f'concentration_std_{window}h'] = df_sorted.groupby(
                [location_col, parameter_col]
            )['concentration'].transform(lambda x: x.rolling(window=window, min_periods=1).std())
            
            df_sorted[f'aqi_mean_{window}h'] = df_sorted.groupby(
                [location_col, parameter_col]
            )['aqi_value'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        
        # Create lag features
        for lag in [1, 2, 3, 6, 12]:
            df_sorted[f'concentration_lag_{lag}h'] = df_sorted.groupby(
                [location_col, parameter_col]
            )['concentration'].shift(lag)
            
            df_sorted[f'aqi_lag_{lag}h'] = df_sorted.groupby(
                [location_col, parameter_col]
            )['aqi_value'].shift(lag)
        
        return df_sorted
    
    """
    Prepare the final dataset for machine learning with proper encoding and scaling.
    
    Args:
        df: Processed DataFrame with time series features
        
    Returns:
        Tuple of (final_df, preprocessing_info)
    """
    def prepare_ml_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        # Create pivot table for multi-parameter features
        pivot_df = df.pivot_table(
            index=['location_id', 'datetime', 'hour', 'day_of_week', 'month', 'season'],
            columns='parameter_standard',
            values=['concentration', 'aqi_value'],
            aggfunc='mean'
        ).reset_index()
        
        # Flatten column names
        pivot_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in pivot_df.columns.values]
        
        # Calculate overall AQI (maximum of all pollutant AQIs)
        aqi_columns = [col for col in pivot_df.columns if col.startswith('aqi_value_')]
        if aqi_columns:
            pivot_df['overall_aqi'] = pivot_df[aqi_columns].max(axis=1)
            pivot_df['overall_aqi_category'] = pivot_df['overall_aqi'].apply(self._aqi_to_category)
        
        # Encode categorical variables
        label_encoders = {}
        categorical_columns = ['season']
        
        for col in categorical_columns:
            if col in pivot_df.columns:
                le = LabelEncoder()
                pivot_df[f'{col}_encoded'] = le.fit_transform(pivot_df[col].fillna('Unknown'))
                label_encoders[col] = le
        
        # Prepare features for scaling
        feature_columns = [col for col in pivot_df.columns 
                          if col not in ['location_id', 'datetime', 'season', 'overall_aqi_category']]
        
        # Handle missing values
        pivot_df[feature_columns] = pivot_df[feature_columns].fillna(0)
        
        # Scale numerical features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(pivot_df[feature_columns])
        scaled_df = pd.DataFrame(scaled_features, columns=feature_columns, index=pivot_df.index)
        
        # Combine scaled features with identifiers and targets
        final_df = pd.concat([
            pivot_df[['location_id', 'datetime']].reset_index(drop=True),
            scaled_df.reset_index(drop=True),
            pivot_df[['overall_aqi', 'overall_aqi_category']].reset_index(drop=True)
        ], axis=1)
        
        preprocessing_info = {
            'scaler': scaler,
            'label_encoders': label_encoders,
            'feature_columns': feature_columns,
            'target_column': 'overall_aqi_category'
        }
        
        return final_df, preprocessing_info
    
    """Convert AQI value to category."""
    def _aqi_to_category(self, aqi_value: float) -> str:
        if pd.isna(aqi_value):
            return 'Unknown'
        elif aqi_value <= 50:
            return 'Good'
        elif aqi_value <= 100:
            return 'Moderate'
        elif aqi_value <= 150:
            return 'Unhealthy for Sensitive Groups'
        elif aqi_value <= 200:
            return 'Unhealthy'
        elif aqi_value <= 300:
            return 'Very Unhealthy'
        else:
            return 'Hazardous'
    
    """
    Fetch a comprehensive dataset for neural network training.
    
    Args:
        country_iso: ISO country code
        max_locations: Maximum number of locations to process
        days_back: Number of days to look back for data
        
    Returns:
        Tuple of (final_dataset, preprocessing_info)
    """
    def fetch_dataset(self, country_iso: str = 'US',
                    max_locations: int = 50,
                    days_back: int = 30) -> Tuple[pd.DataFrame, Dict]:

        print("Starting data fetch...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        date_from = start_date.strftime('%Y-%m-%d')
        date_to = end_date.strftime('%Y-%m-%d')
        
        print(f"Fetching data from {date_from} to {date_to}")
        
        # Get locations
        locations_df = self.get_locations(country_iso, limit=max_locations)
        if locations_df.empty:
            print("No locations found")
            return pd.DataFrame(), {}
        
        # Process locations in batches
        all_measurements = []
        processed_locations = 0
        
        for _, location in locations_df.iterrows():
            location_id = location.get('id')
            if not location_id:
                continue
            
            print(f"Processing location {location_id} ({processed_locations + 1}/{len(locations_df)})")
            
            measurements = self.get_measurements_by_location(
                location_id, date_from, date_to
            )
            
            if not measurements.empty:
                all_measurements.append(measurements)
            
            processed_locations += 1
            if processed_locations >= max_locations:
                break
            
            time.sleep(0.5)
        
        if not all_measurements:
            print("No measurements found")
            return pd.DataFrame(), {}
        
        # Combine all measurements
        combined_measurements = pd.concat(all_measurements, ignore_index=True)
        print(f"Total measurements collected: {len(combined_measurements)}")
        
        # Process measurements
        print("Processing measurements...")
        processed_df = self.process_measurements(combined_measurements)
        
        if processed_df.empty:
            print("No valid measurements after processing")
            return pd.DataFrame(), {}
        
        # Create time series features
        print("Creating time series features")
        ts_df = self.create_time_series_features(processed_df)
        
        # Prepare final ML dataset
        print("Preparing final ML dataset...")
        final_df, preprocessing_info = self.prepare_ml_dataset(ts_df)
        
        print(f"Final dataset shape: {final_df.shape}")
        print(f"AQI category distribution:")
        if 'overall_aqi_category' in final_df.columns:
            print(final_df['overall_aqi_category'].value_counts())
        
        return final_df, preprocessing_info



if __name__ == "__main__":
    # Initialise pipeline
    api_key = os.getenv('OpenAQ_API_KEY')
    pipeline = OpenAQDataPipeline(api_key)
    
    dataset, preprocessing_info = pipeline.fetch_dataset(
        country_iso='US',
        max_locations=50,
        days_back=7
    )
    
    if not dataset.empty:
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        print(f"Dataset shape: {dataset.shape}")
        print(f"\nColumns: {list(dataset.columns)}")
        print(f"\nFirst few rows:")
        print(dataset.head())
        
        # Prepare train/test split if enough data
        if len(dataset) > 100:
            # Separate features and target
            feature_cols = [col for col in dataset.columns 
                           if col not in ['location_id', 'datetime', 'overall_aqi', 'overall_aqi_category']]
            
            X = dataset[feature_cols]
            y = dataset['overall_aqi_category'] if 'overall_aqi_category' in dataset.columns else dataset['overall_aqi']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if 'overall_aqi_category' in dataset.columns else None
            )
            
            print(f"\nTrain/Test Split:")
            print(f"Training set: {X_train.shape}")
            print(f"Test set: {X_test.shape}")
            
            # Save the dataset
            dataset.to_csv('openaq_aqi_dataset.csv', index=False)
            print(f"\nDataset saved to 'openaq_aqi_dataset.csv'")
            
            # Save preprocessing info
            import pickle
            with open('preprocessing_info.pkl', 'wb') as f:
                pickle.dump(preprocessing_info, f)
            print("Preprocessing info saved to 'preprocessing_info.pkl'")
    else:
        print("No data was successfully fetched. Please check your API access and parameters.")