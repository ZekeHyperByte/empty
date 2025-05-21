import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pickle

class DataModel:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        self.linear_model = None
        self.xgb_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.target_col = None
        # Column normalization mapping
        self.col_map = {
            'sp500': 'S&P500',
            'bitcoin': 'BITCOIN',
            'gold': 'Gold',
            'oil': 'Brent Oil',
            'crude oil': 'Crude Oil WTI',
        }
        
    def normalize_columns(self):
        """Normalize column names to expected format."""
        if self.data is not None:
            col_map = {col: col for col in self.data.columns}
            for key, val in self.col_map.items():
                for col in self.data.columns:
                    if col.lower().replace(' ', '').replace('&', '').replace('_', '') == key.replace(' ', '').replace('&', '').replace('_', ''):
                        col_map[col] = val
            self.data.rename(columns=col_map, inplace=True)

    def load_data(self):
        """Load the data from the CSV file and normalize columns."""
        try:
            self.data = pd.read_csv(self.data_path)
            self.normalize_columns()
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
        # Create evaluation set
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        # Train the model
        self.xgb_model.fit(
            self.X_train, 
            self.y_train,
            eval_set=eval_set,
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
        if self.data is None or self.data.empty:
            raise ValueError("No valid data available for prediction after feature engineering.")
        if model_type == 'xgboost' and self.xgb_model is None:
            raise ValueError("XGBoost model not trained. Call train_xgboost_model() first.")
        elif model_type == 'linear' and self.linear_model is None:
            raise ValueError("Linear model not trained. Call train_linear_model() first.")
        # Get the last data point
        last_data = self.data.iloc[-1:].copy()
        if last_data.empty:
            raise ValueError("No valid data available for prediction after feature engineering.")
        predictions = []
        # Create a copy of the last data point for predictions
        current_data = last_data.copy()
        for _ in range(steps):
            # Extract features as DataFrame
            features = current_data[self.feature_cols]
            # Scale features (keep as DataFrame)
            features_scaled = pd.DataFrame(self.scaler.transform(features), columns=self.feature_cols)
            # Check for NaNs or empty input before prediction
            if features_scaled.isnull().any().any():
                raise ValueError("Prediction input contains NaN values. Cannot make prediction.")
            if features_scaled.empty:
                raise ValueError("Prediction input is empty. Cannot make prediction.")
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
                    if lag_col in current_data.columns:
                        if lag == 1:
                            current_data[lag_col] = pred
                        else:
                            if len(predictions) >= lag:
                                current_data[lag_col] = predictions[-lag]
            # Update moving averages and volatility
            ma7_col = f"{self.target_col}_MA7"
            ma14_col = f"{self.target_col}_MA14"
            ma30_col = f"{self.target_col}_MA30"
            momentum_col = f"{self.target_col}_momentum"
            volatility_col = f"{self.target_col}_volatility"
            if ma7_col in current_data.columns:
                current_data[ma7_col] = np.mean([pred] + predictions[-6:]) if len(predictions) >= 6 else current_data[ma7_col]
            if ma14_col in current_data.columns:
                current_data[ma14_col] = np.mean([pred] + predictions[-13:]) if len(predictions) >= 13 else current_data[ma14_col]
            if ma30_col in current_data.columns:
                current_data[ma30_col] = np.mean([pred] + predictions[-29:]) if len(predictions) >= 29 else current_data[ma30_col]
            if momentum_col in current_data.columns:
                current_data[momentum_col] = (pred - current_data[self.target_col].iloc[0]) / current_data[self.target_col].iloc[0] if current_data[self.target_col].iloc[0] != 0 else 0
            if volatility_col in current_data.columns:
                current_data[volatility_col] = np.std([pred] + predictions[-13:]) if len(predictions) >= 13 else current_data[volatility_col]
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
    
    def get_trading_recommendation(self, asset, timeframe='1d'):
        """
        Generate trading recommendations (Buy/Sell/Hold) based on XGBoost predictions
        
        Args:
            asset (str): 'gold' or 'bitcoin'
            timeframe (str): '1d', '1w', or '1m'
        
        Returns:
            dict: Trading recommendation with action, confidence, and price movement
        """
        if self.data is None or self.data.empty:
            raise ValueError("No valid data available for trading recommendation after feature engineering.")
        # Use correct target column for bitcoin
        if asset.lower() == 'bitcoin':
            target_col = 'BITCOIN'
        else:
            target_col = asset.capitalize()
        self.target_col = target_col
        if self.xgb_model is None:
            raise ValueError("XGBoost model not trained. Call train_xgboost_model() first.")
        # Map timeframe to number of days
        timeframe_days = {
            '1d': 1,
            '1w': 7,
            '1m': 30
        }
        days = timeframe_days.get(timeframe, 1)
        # Get the last data point
        last_data = self.data.iloc[-1:].copy()
        if last_data.empty:
            raise ValueError("No valid data available for trading recommendation after feature engineering.")
        # Make prediction for the specified timeframe
        predictions = self.predict_future(steps=days, model_type='xgboost')
        # Calculate price movement
        current_price = last_data[target_col].iloc[0]
        predicted_price = predictions[-1]
        price_change = ((predicted_price - current_price) / current_price) * 100
        # Determine trading action based on price movement and confidence
        if price_change > 2:  # Strong buy signal
            action = "BUY"
            confidence = min(95, 70 + abs(price_change))
        elif price_change < -2:  # Strong sell signal
            action = "SELL"
            confidence = min(95, 70 + abs(price_change))
        else:  # Hold signal
            action = "HOLD"
            confidence = 60 + abs(price_change)
        # Get model's confidence based on R² score
        metrics = self.evaluate_models()
        model_confidence = metrics['xgboost']['r2'] * 100 if 'xgboost' in metrics else 0
        # Adjust confidence based on model performance
        confidence = (confidence + model_confidence) / 2
        return {
            'action': action,
            'confidence': round(confidence, 1),
            'price_movement': f"{price_change:+.1f}%",
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2)
        }
    
    def train_trading_model(self, asset):
        """
        Train XGBoost model specifically for trading predictions
        
        Args:
            asset (str): 'gold' or 'bitcoin'
        """
        if self.data is None:
            self.load_data()
        
        available_columns = self.data.columns.tolist()
        print(f"Available columns: {available_columns}")  # Debug print
        
        # Use correct target column for bitcoin
        if asset.lower() == 'bitcoin':
            target_col = 'BITCOIN'
        else:
            target_col = asset.capitalize()
        
        # Define base features that might be available
        possible_features = ['S&P500', 'Brent Oil', 'Crude Oil WTI', 'Gold', 'BITCOIN']
        feature_cols = [col for col in possible_features if col in available_columns and col != target_col]
        
        # Add technical indicators for the target asset
        if target_col not in available_columns:
            raise ValueError(f"Target column '{target_col}' not found in data. Available columns: {available_columns}")
        
        # Add technical indicators
        for lag in [1, 3, 7, 14]:
            self.data[f'{target_col}_lag_{lag}'] = self.data[target_col].shift(lag)
        for window in [7, 14, 30]:
            self.data[f'{target_col}_rolling_mean_{window}'] = self.data[target_col].rolling(window=window).mean()
            self.data[f'{target_col}_rolling_std_{window}'] = self.data[target_col].rolling(window=window).std()
        
        feature_cols.extend([
            f'{target_col}_lag_1', f'{target_col}_lag_3', f'{target_col}_lag_7', f'{target_col}_lag_14',
            f'{target_col}_rolling_mean_7', f'{target_col}_rolling_std_7',
            f'{target_col}_rolling_mean_14', f'{target_col}_rolling_std_14',
            f'{target_col}_rolling_mean_30', f'{target_col}_rolling_std_30',
            'day_of_week', 'month', 'year'
        ])
        # Drop NaN only for required features and target
        self.data = self.data.dropna(subset=feature_cols + [target_col])
        if len(self.data) == 0:
            raise ValueError("No valid data points after preprocessing")
        X = self.data[feature_cols]
        y = self.data[target_col]
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }
        
        self.xgb_model = xgb.XGBRegressor(**params)
        
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        
        self.xgb_model.fit(
            self.X_train, 
            self.y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        feature_importance = self.xgb_model.feature_importances_
        self.feature_importance_dict = dict(zip(feature_cols, feature_importance))
        
        y_pred = self.xgb_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        self.model_performance = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'feature_importance': self.feature_importance_dict
        }
        
        return self.model_performance
    
    def get_model_performance(self, asset):
        """Get model performance metrics for a specific asset."""
        if not hasattr(self, 'models') or asset not in self.models:
            self.train_trading_model(asset)
        
        model = self.models[asset]
        scaler = self.scalers[asset]
        
        # Get the test data
        X_test = self.X_test[asset]
        y_test = self.y_test[asset]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Get feature importance
        feature_importance = {}
        for feature, importance in zip(self.features[asset], model.feature_importances_):
            feature_importance[feature] = float(importance)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'feature_importance': feature_importance
        }

    def save_model(self, filename):
        """Save the trained model, scaler, features, and target_col to a .pkl file."""
        with open(filename, 'wb') as f:
            pickle.dump({
                'xgb_model': self.xgb_model,
                'scaler': self.scaler,
                'feature_cols': self.feature_cols,
                'target_col': self.target_col
            }, f)

    def load_model(self, filename):
        """Load the model, scaler, features, and target_col from a .pkl file."""
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            self.xgb_model = obj['xgb_model']
            self.scaler = obj['scaler']
            # Normalize feature names to match current data
            self.feature_cols = [
                'BITCOIN' if f.lower() == 'bitcoin' and 'BITCOIN' in self.data.columns else
                'S&P500' if f.lower() in ['sp500', 's&p500'] and 'S&P500' in self.data.columns else
                f for f in obj['feature_cols']
            ]
            # Check for missing features
            missing = [f for f in self.feature_cols if f not in self.data.columns]
            if missing:
                raise ValueError(f"Missing features in data: {missing}. Available columns: {self.data.columns.tolist()}")
            self.target_col = obj['target_col']

    def add_all_features(self):
        # Add date features
        if 'Date' in self.data.columns:
            self.data['day_of_week'] = self.data['Date'].dt.dayofweek
            self.data['month'] = self.data['Date'].dt.month
            self.data['year'] = self.data['Date'].dt.year
        # Add lag and rolling features for Gold
        if 'Gold' in self.data.columns:
            for lag in [1, 3, 7, 14]:
                self.data[f'Gold_lag_{lag}'] = self.data['Gold'].shift(lag)
            for window in [7, 14, 30]:
                self.data[f'Gold_rolling_mean_{window}'] = self.data['Gold'].rolling(window=window).mean()
                self.data[f'Gold_rolling_std_{window}'] = self.data['Gold'].rolling(window=window).std()
        # Add lag and rolling features for BITCOIN
        if 'BITCOIN' in self.data.columns:
            for lag in [1, 3, 7, 14]:
                self.data[f'BITCOIN_lag_{lag}'] = self.data['BITCOIN'].shift(lag)
            for window in [7, 14, 30]:
                self.data[f'BITCOIN_rolling_mean_{window}'] = self.data['BITCOIN'].rolling(window=window).mean()
                self.data[f'BITCOIN_rolling_std_{window}'] = self.data['BITCOIN'].rolling(window=window).std()
        self.data = self.data.dropna()
        print("Shape after feature engineering:", self.data.shape)
        print("Columns after feature engineering:", self.data.columns.tolist())
        if self.data is None or self.data.empty:
            raise ValueError("No data available for prediction after feature engineering. Your dataset may be too short for the required lags/rolling windows.") 