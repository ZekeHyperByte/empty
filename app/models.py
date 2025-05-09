import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

class DataModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.linear_model = None
        self.xgb_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the data from the CSV file"""
        try:
            self.data = pd.read_csv(self.data_path)
            # Convert date column to datetime if it exists
            if 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'])
                self.data.sort_values('Date', inplace=True)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self, target_col, feature_cols, test_size=0.2):
        """Preprocess the data for model training"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Ensure all needed columns exist
        for col in [target_col] + feature_cols:
            if col not in self.data.columns:
                raise ValueError(f"Column {col} not found in the dataset")
        
        # Handle missing values
        self.data = self.data.dropna(subset=[target_col] + feature_cols)
        
        # Add time-based features if Date column exists
        if 'Date' in self.data.columns:
            self.data['day_of_week'] = self.data['Date'].dt.dayofweek
            self.data['month'] = self.data['Date'].dt.month
            self.data['year'] = self.data['Date'].dt.year
            # Add these new features to the feature list if they're not already included
            for feat in ['day_of_week', 'month', 'year']:
                if feat not in feature_cols:
                    feature_cols.append(feat)
        
        # Add lag features for time series forecasting
        if target_col in ['Gold', 'Bitcoin']:
            for lag in [1, 3, 7, 14]:
                lag_col = f"{target_col}_lag_{lag}"
                self.data[lag_col] = self.data[target_col].shift(lag)
                if lag_col not in feature_cols:
                    feature_cols.append(lag_col)
            
            # Add rolling statistics
            for window in [7, 14, 30]:
                # Rolling mean
                mean_col = f"{target_col}_rolling_mean_{window}"
                self.data[mean_col] = self.data[target_col].rolling(window=window).mean()
                if mean_col not in feature_cols:
                    feature_cols.append(mean_col)
                
                # Rolling std
                std_col = f"{target_col}_rolling_std_{window}"
                self.data[std_col] = self.data[target_col].rolling(window=window).std()
                if std_col not in feature_cols:
                    feature_cols.append(std_col)
        
        # Drop rows with NaN values after creating lag features
        self.data = self.data.dropna()
        
        # Split data into features and target
        X = self.data[feature_cols]
        y = self.data[target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Store feature names for later use
        self.feature_cols = feature_cols
        self.target_col = target_col
        
    def train_linear_model(self):
        """Train a simple linear regression model"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        # Create and train the model
        self.linear_model = LinearRegression()
        self.linear_model.fit(self.X_train, self.y_train)
        
    def train_xgboost_model(self, params=None):
        """Train an XGBoost model"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        # Default XGBoost parameters
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'learning_rate': 0.1,
                'max_depth': 6,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42
            }
        
        # Create and train the XGBoost model
        self.xgb_model = xgb.XGBRegressor(**params)
        self.xgb_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
    def evaluate_models(self):
        """Evaluate both models' performance"""
        results = {}
        
        # Evaluate Linear Regression model
        if self.linear_model is not None:
            y_pred_linear = self.linear_model.predict(self.X_test)
            results['linear'] = {
                'mse': mean_squared_error(self.y_test, y_pred_linear),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_linear)),
                'mae': mean_absolute_error(self.y_test, y_pred_linear),
                'r2': r2_score(self.y_test, y_pred_linear)
            }
        
        # Evaluate XGBoost model
        if self.xgb_model is not None:
            y_pred_xgb = self.xgb_model.predict(self.X_test)
            results['xgboost'] = {
                'mse': mean_squared_error(self.y_test, y_pred_xgb),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_xgb)),
                'mae': mean_absolute_error(self.y_test, y_pred_xgb),
                'r2': r2_score(self.y_test, y_pred_xgb)
            }
        
        return results
    
    def predict_future(self, steps=30, model_type='xgboost'):
        """Predict future values using the trained model"""
        if model_type == 'xgboost' and self.xgb_model is None:
            raise ValueError("XGBoost model not trained. Call train_xgboost_model() first.")
        elif model_type == 'linear' and self.linear_model is None:
            raise ValueError("Linear model not trained. Call train_linear_model() first.")
        
        # Get the last data point
        last_data = self.data.iloc[-1:].copy()
        predictions = []
        
        # Create a copy of the last data point for predictions
        current_data = last_data.copy()
        
        for _ in range(steps):
            # Extract features
            features = current_data[self.feature_cols].values
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            if model_type == 'xgboost':
                pred = self.xgb_model.predict(features_scaled)[0]
            else:
                pred = self.linear_model.predict(features_scaled)[0]
            
            # Store prediction
            predictions.append(pred)
            
            # Update current data for next prediction
            current_data[self.target_col] = pred
            
            # Update lag features
            if self.target_col in ['Gold', 'Bitcoin']:
                for lag in [1, 3, 7, 14]:
                    lag_col = f"{self.target_col}_lag_{lag}"
                    if lag == 1:
                        current_data[lag_col] = pred
                    else:
                        # For higher lags, use previous data points if available
                        if len(predictions) >= lag:
                            current_data[lag_col] = predictions[-lag]
            
            # Update rolling statistics (simplified approach)
            for window in [7, 14, 30]:
                mean_col = f"{self.target_col}_rolling_mean_{window}"
                std_col = f"{self.target_col}_rolling_std_{window}"
                
                if len(predictions) < window:
                    # Not enough predictions yet, use the last known values
                    current_data[mean_col] = current_data[mean_col]
                    current_data[std_col] = current_data[std_col]
                else:
                    # Calculate new rolling statistics
                    recent_vals = predictions[-window:]
                    current_data[mean_col] = np.mean(recent_vals)
                    current_data[std_col] = np.std(recent_vals)
        
        return predictions
    
    def get_feature_importance(self, model_type='xgboost'):
        """Get feature importance from the trained model"""
        if model_type == 'xgboost' and self.xgb_model is None:
            raise ValueError("XGBoost model not trained. Call train_xgboost_model() first.")
        elif model_type == 'linear' and self.linear_model is None:
            raise ValueError("Linear model not trained. Call train_linear_model() first.")
        
        if model_type == 'xgboost':
            importance = self.xgb_model.feature_importances_
        else:
            importance = np.abs(self.linear_model.coef_)
        
        # Create a dictionary of feature importance
        feature_importance = dict(zip(self.feature_cols, importance))
        
        # Sort by importance
        feature_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                                      key=lambda item: item[1], 
                                                      reverse=True)}
        
        return feature_importance
    
    def get_summary_stats(self):
        """Get summary statistics of the data"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'summary': self.data.describe().to_dict(),
            'missing_values': self.data.isnull().sum().to_dict()
        } 