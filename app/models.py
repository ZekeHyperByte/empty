import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class ImprovedDataModel:
    """
    Enhanced DataModel class for financial market prediction with proper time series methodology.
    
    Features:
    - Time-aware data splitting
    - Advanced feature engineering with technical indicators
    - Multiple model support with hyperparameter optimization
    - Proper cross-validation using TimeSeriesSplit
    - Enhanced prediction capabilities with uncertainty estimation
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize the DataModel with enhanced capabilities."""
        # Core attributes
        self.data_path = data_path
        self.data = None
        self.original_data = None  # Keep original data for reference
        
        # Models
        self.models = {}
        self.scalers = {}
        self.best_models = {}
        
        # Data splits
        self.train_data = {}
        self.test_data = {}
        
        # Feature information
        self.feature_cols = {}
        self.target_cols = []
        
        # Cross-validation
        self.cv_results = {}
        
        # Column normalization mapping (enhanced)
        self.col_map = {
            'sp500': 'S&P500',
            's&p500': 'S&P500',
            's&p 500': 'S&P500',
            'bitcoin': 'BITCOIN',
            'btc': 'BITCOIN',
            'gold': 'Gold',
            'oil': 'Brent Oil',
            'brent': 'Brent Oil',
            'crude oil': 'Crude Oil WTI',
            'wti': 'Crude Oil WTI',
            'crude': 'Crude Oil WTI'
        }
        
        # Model parameters
        self.model_params = {
            'xgboost': {
                'objective': 'reg:squarederror',
                'learning_rate': 0.1,
                'max_depth': 6,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'verbosity': 0
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'linear': {}
        }
        
        # Hyperparameter grids for optimization
        self.param_grids = {
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'linear': {}
        }

    def normalize_columns(self) -> None:
        """Enhanced column normalization with better pattern matching."""
        if self.data is None:
            return
            
        # Create mapping for current columns
        col_mapping = {}
        current_cols = self.data.columns.tolist()
        
        for current_col in current_cols:
            # Clean current column name for comparison
            cleaned_current = current_col.lower().replace(' ', '').replace('&', '').replace('_', '')
            
            # Check against our mapping
            for pattern, standard_name in self.col_map.items():
                cleaned_pattern = pattern.lower().replace(' ', '').replace('&', '').replace('_', '')
                if cleaned_current == cleaned_pattern:
                    col_mapping[current_col] = standard_name
                    break
            else:
                # Keep original name if no mapping found
                col_mapping[current_col] = current_col
        
        # Apply the mapping
        self.data.rename(columns=col_mapping, inplace=True)
        logger.info(f"Column normalization complete. Final columns: {self.data.columns.tolist()}")

    def load_data(self) -> bool:
        """Load and preprocess the data with enhanced error handling."""
        try:
            if self.data_path is None:
                raise ValueError("Data path not provided")
                
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            self.original_data = self.data.copy()  # Keep original for reference
            
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            
            # Normalize column names
            self.normalize_columns()
            
            # Handle date column
            date_columns = ['Date', 'date', 'DATE', 'Datetime', 'datetime']
            date_col = None
            for col in date_columns:
                if col in self.data.columns:
                    date_col = col
                    break
            
            if date_col:
                self.data[date_col] = pd.to_datetime(self.data[date_col])
                self.data.sort_values(date_col, inplace=True)
                self.data.reset_index(drop=True, inplace=True)
                logger.info(f"Date column '{date_col}' processed and data sorted chronologically")
            else:
                logger.warning("No date column found. Assuming data is already sorted chronologically.")
            
            # Basic data validation
            logger.info(f"Data overview:\n{self.data.info()}")
            logger.info(f"Missing values:\n{self.data.isnull().sum()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False

    def create_technical_features(self, target_col: str = None) -> pd.DataFrame:
        """
        Create comprehensive technical indicators and features.
        
        Args:
            target_col: If specified, creates target-specific features
            
        Returns:
            Enhanced dataframe with technical features
        """
        logger.info("Creating technical features...")
        df_enhanced = self.data.copy()
        
        # Get numeric columns
        numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
        
        # Add time-based features if date column exists
        date_col = None
        for col in ['Date', 'date', 'Datetime', 'datetime']:
            if col in df_enhanced.columns:
                date_col = col
                break
        
        if date_col:
            df_enhanced['day_of_week'] = df_enhanced[date_col].dt.dayofweek
            df_enhanced['month'] = df_enhanced[date_col].dt.month
            df_enhanced['year'] = df_enhanced[date_col].dt.year
            df_enhanced['quarter'] = df_enhanced[date_col].dt.quarter
            df_enhanced['day_of_year'] = df_enhanced[date_col].dt.dayofyear
            
            # Cyclical encoding for better ML performance
            df_enhanced['day_of_week_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_week'] / 7)
            df_enhanced['day_of_week_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_week'] / 7)
            df_enhanced['month_sin'] = np.sin(2 * np.pi * df_enhanced['month'] / 12)
            df_enhanced['month_cos'] = np.cos(2 * np.pi * df_enhanced['month'] / 12)
        
        # Create features for each numeric column
        for col in numeric_cols:
            logger.debug(f"Creating features for {col}")
            
            # Lag features (1, 2, 3, 5, 10 days)
            for lag in [1, 2, 3, 5, 10]:
                df_enhanced[f'{col}_lag_{lag}'] = df_enhanced[col].shift(lag)
            
            # Moving averages (5, 10, 20, 50 days)
            for window in [5, 10, 20, 50]:
                df_enhanced[f'{col}_ma_{window}'] = df_enhanced[col].rolling(window=window).mean()
            
            # Exponential moving averages
            for window in [5, 10, 20]:
                df_enhanced[f'{col}_ema_{window}'] = df_enhanced[col].ewm(span=window).mean()
            
            # Volatility (rolling standard deviation)
            for window in [5, 10, 20]:
                df_enhanced[f'{col}_volatility_{window}'] = df_enhanced[col].rolling(window=window).std()
            
            # Price change features
            df_enhanced[f'{col}_change_1d'] = df_enhanced[col].pct_change(1)
            df_enhanced[f'{col}_change_5d'] = df_enhanced[col].pct_change(5)
            df_enhanced[f'{col}_change_10d'] = df_enhanced[col].pct_change(10)
            
            # Momentum indicators
            df_enhanced[f'{col}_momentum_5'] = df_enhanced[col] / df_enhanced[col].shift(5) - 1
            df_enhanced[f'{col}_momentum_10'] = df_enhanced[col] / df_enhanced[col].shift(10) - 1
            
            # RSI (Relative Strength Index)
            delta = df_enhanced[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_enhanced[f'{col}_rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            for window in [20]:
                rolling_mean = df_enhanced[col].rolling(window=window).mean()
                rolling_std = df_enhanced[col].rolling(window=window).std()
                df_enhanced[f'{col}_bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
                df_enhanced[f'{col}_bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
                df_enhanced[f'{col}_bb_position_{window}'] = (df_enhanced[col] - rolling_mean) / (rolling_std * 2)
            
            # MACD
            exp1 = df_enhanced[col].ewm(span=12).mean()
            exp2 = df_enhanced[col].ewm(span=26).mean()
            df_enhanced[f'{col}_macd'] = exp1 - exp2
            df_enhanced[f'{col}_macd_signal'] = df_enhanced[f'{col}_macd'].ewm(span=9).mean()
            df_enhanced[f'{col}_macd_histogram'] = df_enhanced[f'{col}_macd'] - df_enhanced[f'{col}_macd_signal']
        
        # Cross-asset features
        if 'S&P500' in numeric_cols and 'Gold' in numeric_cols:
            df_enhanced['sp500_gold_ratio'] = df_enhanced['S&P500'] / df_enhanced['Gold']
            df_enhanced['sp500_gold_correlation_10'] = df_enhanced['S&P500'].rolling(10).corr(df_enhanced['Gold'])
            df_enhanced['sp500_gold_correlation_30'] = df_enhanced['S&P500'].rolling(30).corr(df_enhanced['Gold'])
        
        if 'Brent Oil' in numeric_cols and 'Crude Oil WTI' in numeric_cols:
            df_enhanced['oil_spread'] = df_enhanced['Brent Oil'] - df_enhanced['Crude Oil WTI']
            df_enhanced['oil_ratio'] = df_enhanced['Brent Oil'] / df_enhanced['Crude Oil WTI']
        
        if 'Gold' in numeric_cols and 'BITCOIN' in numeric_cols:
            df_enhanced['gold_bitcoin_ratio'] = df_enhanced['Gold'] / df_enhanced['BITCOIN']
            df_enhanced['gold_bitcoin_correlation_10'] = df_enhanced['Gold'].rolling(10).corr(df_enhanced['BITCOIN'])
        
        # Market volatility indicator (if S&P500 available)
        if 'S&P500' in numeric_cols:
            df_enhanced['market_volatility'] = df_enhanced['S&P500'].rolling(window=20).std()
            df_enhanced['market_trend'] = df_enhanced['S&P500'].rolling(window=20).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1
            )
        
        # Remove rows with NaN values (from rolling calculations)
        initial_rows = len(df_enhanced)
        df_enhanced = df_enhanced.dropna()
        final_rows = len(df_enhanced)
        
        logger.info(f"Feature engineering complete. Rows: {initial_rows} → {final_rows}")
        logger.info(f"Total features created: {len(df_enhanced.columns) - len(self.data.columns)}")
        
        return df_enhanced

    def time_series_split_data(self, X: pd.DataFrame, y: pd.Series, 
                              test_size: float = 0.2, n_splits: int = 5) -> Tuple:
        """
        Split data using proper time series methodology.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion for test set
            n_splits: Number of cross-validation splits
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, tscv)
        """
        logger.info(f"Performing time series split with test_size={test_size}")
        
        # Calculate split point for final train/test
        split_point = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_point].copy()
        X_test = X.iloc[split_point:].copy()
        y_train = y.iloc[:split_point].copy()
        y_test = y.iloc[split_point:].copy()
        
        # Create TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test, tscv

    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                 tscv: TimeSeriesSplit, model_type: str = 'xgboost') -> object:
        """
        Optimize hyperparameters using GridSearchCV with TimeSeriesSplit.
        
        Args:
            X_train: Training features
            y_train: Training target
            tscv: TimeSeriesSplit object
            model_type: Type of model to optimize
            
        Returns:
            Best model after hyperparameter optimization
        """
        logger.info(f"Optimizing hyperparameters for {model_type}...")
        
        if model_type == 'xgboost':
            model = xgb.XGBRegressor(**self.model_params['xgboost'])
            param_grid = self.param_grids['xgboost']
        elif model_type == 'random_forest':
            model = RandomForestRegressor(**self.model_params['random_forest'])
            param_grid = self.param_grids['random_forest']
        elif model_type == 'linear':
            model = LinearRegression()
            param_grid = self.param_grids['linear']
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if param_grid:
            grid_search = GridSearchCV(
                model, param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
            logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
            return grid_search.best_estimator_
        else:
            model.fit(X_train, y_train)
            return model

    def cross_validate_model(self, model: object, X_train: pd.DataFrame, 
                           y_train: pd.Series, tscv: TimeSeriesSplit) -> List[float]:
        """
        Perform cross-validation using TimeSeriesSplit.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
            tscv: TimeSeriesSplit object
            
        Returns:
            List of cross-validation scores
        """
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_train_cv = X_train.iloc[train_idx]
            X_val_cv = X_train.iloc[val_idx]
            y_train_cv = y_train.iloc[train_idx]
            y_val_cv = y_train.iloc[val_idx]
            
            # Clone model to avoid fitting on previous folds
            if hasattr(model, 'fit'):
                model_cv = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                model_cv.fit(X_train_cv, y_train_cv)
                y_pred_cv = model_cv.predict(X_val_cv)
                score = r2_score(y_val_cv, y_pred_cv)
                cv_scores.append(score)
                logger.debug(f"Fold {fold + 1}: R² = {score:.4f}")
        
        return cv_scores

    def train_enhanced_models(self, target_col: str, feature_cols: List[str] = None, 
                            test_size: float = 0.2, optimize_params: bool = True) -> Dict:
        """
        Train multiple models with enhanced methodology.
        
        Args:
            target_col: Target column name
            feature_cols: List of feature columns (if None, uses all available)
            test_size: Test set proportion
            optimize_params: Whether to optimize hyperparameters
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training enhanced models for target: {target_col}")
        
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create technical features
        enhanced_data = self.create_technical_features(target_col)
        
        # Prepare features and target
        if feature_cols is None:
            # Use all features except target
            all_features = [col for col in enhanced_data.columns if col != target_col]
            # Remove date column if present
            all_features = [col for col in all_features if not any(
                date_name in col.lower() for date_name in ['date', 'datetime']
            )]
        else:
            all_features = feature_cols
        
        # Ensure target exists
        if target_col not in enhanced_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        X = enhanced_data[all_features].copy()
        y = enhanced_data[target_col].copy()
        
        logger.info(f"Features: {len(all_features)}, Samples: {len(X)}")
        
        # Time series split
        X_train, X_test, y_train, y_test, tscv = self.time_series_split_data(X, y, test_size)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Store data splits and scaler
        self.train_data[target_col] = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled
        }
        self.scalers[target_col] = scaler
        self.feature_cols[target_col] = all_features
        
        # Train models
        models = {}
        results = {}
        
        model_types = ['linear', 'random_forest', 'xgboost']
        
        for model_type in model_types:
            logger.info(f"Training {model_type} model...")
            
            try:
                if optimize_params:
                    if model_type == 'linear':
                        model = self.optimize_hyperparameters(X_train_scaled, y_train, tscv, model_type)
                    else:
                        model = self.optimize_hyperparameters(X_train, y_train, tscv, model_type)
                else:
                    # Use default parameters
                    if model_type == 'xgboost':
                        model = xgb.XGBRegressor(**self.model_params['xgboost'])
                        model.fit(X_train, y_train)
                    elif model_type == 'random_forest':
                        model = RandomForestRegressor(**self.model_params['random_forest'])
                        model.fit(X_train, y_train)
                    else:  # linear
                        model = LinearRegression()
                        model.fit(X_train_scaled, y_train)
                
                models[model_type] = model
                
                # Cross-validation
                if model_type == 'linear':
                    cv_scores = self.cross_validate_model(model, X_train_scaled, y_train, tscv)
                    y_pred = model.predict(X_test_scaled)
                else:
                    cv_scores = self.cross_validate_model(model, X_train, y_train, tscv)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[model_type] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'cv_scores': cv_scores,
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'predictions': y_pred
                }
                
                logger.info(f"{model_type.upper()} - R²: {r2:.4f}, CV: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {str(e)}")
                continue
        
        # Store models and results
        self.models[target_col] = models
        self.cv_results[target_col] = results
        
        # Identify best model
        best_model_type = max(results.keys(), key=lambda k: results[k]['r2'])
        self.best_models[target_col] = {
            'type': best_model_type,
            'model': models[best_model_type],
            'scaler': scaler if best_model_type == 'linear' else None
        }
        
        logger.info(f"Best model for {target_col}: {best_model_type} (R² = {results[best_model_type]['r2']:.4f})")
        
        return results

    def predict_future(self, target_col: str, steps: int = 30, 
                      model_type: str = None, return_confidence: bool = False) -> Union[List[float], Tuple]:
        """
        Enhanced future prediction with uncertainty estimation.
        
        Args:
            target_col: Target column to predict
            steps: Number of steps to predict
            model_type: Model type to use (if None, uses best model)
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Predictions or (predictions, confidence_intervals) if return_confidence=True
        """
        logger.info(f"Predicting {steps} steps ahead for {target_col}")
        
        if target_col not in self.models:
            raise ValueError(f"No trained model found for {target_col}")
        
        # Use best model if not specified
        if model_type is None:
            model_info = self.best_models[target_col]
            model_type = model_info['type']
            model = model_info['model']
            scaler = model_info.get('scaler')
        else:
            model = self.models[target_col][model_type]
            scaler = self.scalers[target_col] if model_type == 'linear' else None
        
        # Get enhanced data with features
        enhanced_data = self.create_technical_features()
        
        if enhanced_data.empty:
            raise ValueError("No data available for prediction")
        
        # Get last data point
        last_data = enhanced_data.iloc[-1:].copy()
        features = self.feature_cols[target_col]
        
        predictions = []
        confidence_intervals = [] if return_confidence else None
        
        current_data = last_data.copy()
        
        for step in range(steps):
            try:
                # Extract features
                feature_values = current_data[features].values.reshape(1, -1)
                
                # Scale if needed
                if scaler is not None:
                    feature_values = scaler.transform(feature_values)
                
                # Make prediction
                pred = model.predict(feature_values)[0]
                predictions.append(pred)
                
                # Estimate confidence interval (simplified approach)
                if return_confidence:
                    # Use model's uncertainty if available, otherwise use historical std
                    if hasattr(model, 'predict_proba'):
                        # For models that support uncertainty
                        ci = [pred - 1.96 * np.std(predictions[-min(10, len(predictions)):]), 
                              pred + 1.96 * np.std(predictions[-min(10, len(predictions)):])]
                    else:
                        # Use historical volatility
                        hist_std = enhanced_data[target_col].rolling(20).std().iloc[-1]
                        ci = [pred - 1.96 * hist_std, pred + 1.96 * hist_std]
                    confidence_intervals.append(ci)
                
                # Update current data for next prediction
                self._update_features_for_prediction(current_data, target_col, pred, predictions, enhanced_data)
                
            except Exception as e:
                logger.error(f"Error in prediction step {step}: {str(e)}")
                break
        
        if return_confidence and confidence_intervals:
            return predictions, confidence_intervals
        return predictions

    def _update_features_for_prediction(self, current_data: pd.DataFrame, target_col: str, 
                                      prediction: float, all_predictions: List[float], 
                                      enhanced_data: pd.DataFrame) -> None:
        """
        Update features for next prediction step.
        
        Args:
            current_data: Current data row
            target_col: Target column name
            prediction: Current prediction
            all_predictions: All predictions so far
            enhanced_data: Enhanced dataset for reference
        """
        # Update target value
        current_data[target_col] = prediction
        
        # Update lag features
        for lag in [1, 2, 3, 5, 10]:
            lag_col = f'{target_col}_lag_{lag}'
            if lag_col in current_data.columns:
                if lag == 1:
                    current_data[lag_col] = prediction
                elif len(all_predictions) >= lag:
                    current_data[lag_col] = all_predictions[-lag]
        
        # Update rolling features (simplified)
        for window in [5, 10, 20]:
            ma_col = f'{target_col}_ma_{window}'
            if ma_col in current_data.columns and len(all_predictions) >= window:
                current_data[ma_col] = np.mean(all_predictions[-window:])
            
            vol_col = f'{target_col}_volatility_{window}'
            if vol_col in current_data.columns and len(all_predictions) >= window:
                current_data[vol_col] = np.std(all_predictions[-window:])

    def get_trading_recommendation(self, asset: str, timeframe: str = '1d', 
                                 confidence_threshold: float = 0.6) -> Dict:
        """
        Enhanced trading recommendations with improved logic.
        
        Args:
            asset: Asset name ('gold' or 'bitcoin')
            timeframe: Time horizon ('1d', '1w', '1m')
            confidence_threshold: Minimum confidence for recommendations
            
        Returns:
            Trading recommendation dictionary
        """
        # Map asset to target column
        target_col = 'BITCOIN' if asset.lower() == 'bitcoin' else 'Gold'
        
        if target_col not in self.models:
            raise ValueError(f"No trained model found for {asset}")
        
        # Map timeframe to days
        timeframe_days = {'1d': 1, '1w': 7, '1m': 30}
        days = timeframe_days.get(timeframe, 1)
        
        # Get predictions with confidence
        predictions, confidence_intervals = self.predict_future(
            target_col, steps=days, return_confidence=True
        )
        
        # Calculate metrics
        enhanced_data = self.create_technical_features()
        current_price = enhanced_data[target_col].iloc[-1]
        predicted_price = predictions[-1]
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        # Calculate prediction confidence based on model performance and volatility
        model_performance = self.cv_results[target_col][self.best_models[target_col]['type']]
        model_r2 = model_performance['r2']
        
        # Volatility-adjusted confidence
        recent_volatility = enhanced_data[f'{target_col}_volatility_20'].iloc[-1]
        avg_volatility = enhanced_data[f'{target_col}_volatility_20'].mean()
        volatility_factor = min(1.0, avg_volatility / recent_volatility) if recent_volatility > 0 else 1.0
        
        # Base confidence from model performance
        base_confidence = model_r2 * 100
        
        # Adjust for prediction magnitude and volatility
        magnitude_penalty = min(20, abs(price_change) * 2)  # Penalty for extreme predictions
        adjusted_confidence = (base_confidence * volatility_factor) - magnitude_penalty
        adjusted_confidence = max(30, min(95, adjusted_confidence))  # Bound between 30-95%
        
        # Determine action based on price change and confidence
        if adjusted_confidence < confidence_threshold * 100:
            action = "HOLD"
            reason = "Low model confidence"
        elif price_change > 3:  # Strong buy signal
            action = "BUY"
            reason = f"Strong upward trend predicted ({price_change:+.1f}%)"
        elif price_change < -3:  # Strong sell signal
            action = "SELL" 
            reason = f"Strong downward trend predicted ({price_change:+.1f}%)"
        elif price_change > 1:  # Moderate buy
            action = "BUY"
            reason = f"Moderate upward trend predicted ({price_change:+.1f}%)"
        elif price_change < -1:  # Moderate sell
            action = "SELL"
            reason = f"Moderate downward trend predicted ({price_change:+.1f}%)"
        else:  # Hold
            action = "HOLD"
            reason = f"Minimal price movement expected ({price_change:+.1f}%)"
        
        # Risk assessment
        if confidence_intervals:
            lower_bound, upper_bound = confidence_intervals[-1]
            risk_range = ((upper_bound - lower_bound) / current_price) * 100
        else:
            risk_range = abs(price_change) * 2  # Simplified risk estimate
        
        risk_level = "LOW" if risk_range < 5 else "MEDIUM" if risk_range < 15 else "HIGH"
        
        return {
            'action': action,
            'confidence': round(adjusted_confidence, 1),
            'price_movement': f"{price_change:+.1f}%",
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'timeframe': timeframe,
            'risk_level': risk_level,
            'risk_range': f"±{risk_range:.1f}%",
            'reason': reason,
            'model_performance': {
                'r2_score': round(model_r2, 3),
                'cv_mean': round(model_performance['cv_mean'], 3),
                'cv_std': round(model_performance['cv_std'], 3)
            }
        }

    def get_feature_importance(self, target_col: str, model_type: str = None, top_n: int = 20) -> Dict:
        """
        Get feature importance with enhanced analysis.
        
        Args:
            target_col: Target column name
            model_type: Model type (if None, uses best model)
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature importance
        """
        if target_col not in self.models:
            raise ValueError(f"No trained model found for {target_col}")
        
        if model_type is None:
            model_type = self.best_models[target_col]['type']
        
        model = self.models[target_col][model_type]
        features = self.feature_cols[target_col]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            raise ValueError(f"Model {model_type} doesn't support feature importance")
        
        # Create feature importance dictionary
        feature_importance = dict(zip(features, importance))
        
        # Sort by importance
        sorted_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n])
        
        return sorted_importance

    def evaluate_model_performance(self, target_col: str) -> Dict:
        """
        Comprehensive model performance evaluation.
        
        Args:
            target_col: Target column name
            
        Returns:
            Performance metrics for all models
        """
        if target_col not in self.cv_results:
            raise ValueError(f"No evaluation results found for {target_col}")
        
        results = self.cv_results[target_col].copy()
        
        # Add additional metrics
        for model_type, metrics in results.items():
            # Prediction accuracy categories
            r2 = metrics['r2']
            if r2 >= 0.8:
                accuracy = "Excellent"
            elif r2 >= 0.6:
                accuracy = "Good" 
            elif r2 >= 0.4:
                accuracy = "Fair"
            else:
                accuracy = "Poor"
            
            metrics['accuracy_category'] = accuracy
            
            # Stability (based on CV std)
            cv_std = metrics['cv_std']
            if cv_std <= 0.05:
                stability = "Very Stable"
            elif cv_std <= 0.1:
                stability = "Stable"
            elif cv_std <= 0.2:
                stability = "Moderate"
            else:
                stability = "Unstable"
            
            metrics['stability'] = stability
        
        return results

    def save_complete_model(self, target_col: str, filepath: str) -> None:
        """
        Save complete model package including data preprocessing components.
        
        Args:
            target_col: Target column name
            filepath: File path to save model
        """
        if target_col not in self.models:
            raise ValueError(f"No trained model found for {target_col}")
        
        model_package = {
            'models': self.models[target_col],
            'best_model': self.best_models[target_col],
            'scaler': self.scalers[target_col],
            'features': self.feature_cols[target_col],
            'target_col': target_col,
            'cv_results': self.cv_results[target_col],
            'model_params': self.model_params,
            'col_map': self.col_map
        }
        
        joblib.dump(model_package, filepath)
        logger.info(f"Complete model package saved to {filepath}")

    def load_complete_model(self, filepath: str, target_col: str) -> None:
        """
        Load complete model package.
        
        Args:
            filepath: File path to load model from
            target_col: Target column name
        """
        try:
            model_package = joblib.load(filepath)
            
            self.models[target_col] = model_package['models']
            self.best_models[target_col] = model_package['best_model']
            self.scalers[target_col] = model_package['scaler']
            self.feature_cols[target_col] = model_package['features']
            self.cv_results[target_col] = model_package['cv_results']
            
            # Update class parameters if available
            if 'model_params' in model_package:
                self.model_params.update(model_package['model_params'])
            if 'col_map' in model_package:
                self.col_map.update(model_package['col_map'])
            
            logger.info(f"Complete model package loaded for {target_col}")
            
        except Exception as e:
            logger.error(f"Error loading model package: {str(e)}")
            raise

    def get_model_summary(self, target_col: str = None) -> Dict:
        """
        Get comprehensive summary of trained models.
        
        Args:
            target_col: Specific target (if None, returns all)
            
        Returns:
            Model summary dictionary
        """
        if target_col:
            targets = [target_col]
        else:
            targets = list(self.models.keys())
        
        summary = {}
        
        for target in targets:
            if target not in self.models:
                continue
                
            target_summary = {
                'available_models': list(self.models[target].keys()),
                'best_model': self.best_models[target]['type'],
                'feature_count': len(self.feature_cols[target]),
                'performance': self.cv_results[target],
                'data_shape': {
                    'train_samples': len(self.train_data[target]['X_train']),
                    'test_samples': len(self.train_data[target]['X_test'])
                } if target in self.train_data else None
            }
            
            summary[target] = target_summary
        
        return summary

    def backtest_strategy(self, target_col: str, start_date: str = None, 
                         end_date: str = None, initial_capital: float = 10000) -> Dict:
        """
        Simple backtesting for trading strategy.
        
        Args:
            target_col: Target column name
            start_date: Start date for backtesting
            end_date: End date for backtesting  
            initial_capital: Initial capital amount
            
        Returns:
            Backtesting results
        """
        # This is a simplified backtesting implementation
        # In production, you'd want more sophisticated backtesting
        
        if target_col not in self.models:
            raise ValueError(f"No trained model found for {target_col}")
        
        enhanced_data = self.create_technical_features()
        
        # Filter data by date range if provided
        if start_date or end_date:
            date_col = None
            for col in ['Date', 'date', 'Datetime', 'datetime']:
                if col in enhanced_data.columns:
                    date_col = col
                    break
            
            if date_col:
                if start_date:
                    enhanced_data = enhanced_data[enhanced_data[date_col] >= start_date]
                if end_date:
                    enhanced_data = enhanced_data[enhanced_data[date_col] <= end_date]
        
        # Simple strategy simulation
        capital = initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        trades = []
        
        # Use test period for backtesting
        test_start = len(enhanced_data) - len(self.train_data[target_col]['X_test'])
        test_data = enhanced_data.iloc[test_start:].copy()
        
        for i in range(1, len(test_data)):
            current_price = test_data[target_col].iloc[i]
            
            # Simple momentum strategy based on recent predictions
            # This is a placeholder - you'd implement your actual strategy here
            price_change = (current_price - test_data[target_col].iloc[i-1]) / test_data[target_col].iloc[i-1]
            
            if price_change > 0.02 and position <= 0:  # Buy signal
                if position == -1:  # Close short
                    capital *= (1 - price_change)
                position = 1
                trades.append({'action': 'BUY', 'price': current_price, 'capital': capital})
            elif price_change < -0.02 and position >= 0:  # Sell signal
                if position == 1:  # Close long
                    capital *= (1 + price_change)
                position = -1
                trades.append({'action': 'SELL', 'price': current_price, 'capital': capital})
        
        # Calculate performance metrics
        total_return = (capital - initial_capital) / initial_capital * 100
        num_trades = len(trades)
        
        return {
            'initial_capital': initial_capital,
            'final_capital': round(capital, 2),
            'total_return': round(total_return, 2),
            'num_trades': num_trades,
            'trades': trades[-10:] if len(trades) > 10 else trades,  # Last 10 trades
            'avg_trade_return': round(total_return / num_trades, 2) if num_trades > 0 else 0
        }

    def get_data_summary(self) -> Dict:
        """Get comprehensive data summary and statistics."""
        if self.data is None:
            return {"error": "No data loaded"}
        
        summary = {
            'basic_info': {
                'shape': self.data.shape,
                'columns': self.data.columns.tolist(),
                'data_types': self.data.dtypes.to_dict()
            },
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_summary': self.data.describe().to_dict(),
            'date_range': None
        }
        
        # Add date range if date column exists
        for col in ['Date', 'date', 'Datetime', 'datetime']:
            if col in self.data.columns:
                summary['date_range'] = {
                    'start': str(self.data[col].min()),
                    'end': str(self.data[col].max()),
                    'total_days': (self.data[col].max() - self.data[col].min()).days
                }
                break
        
        return summary

# Legacy compatibility methods for existing code
class DataModel(ImprovedDataModel):
    """
    Legacy DataModel class that inherits from ImprovedDataModel.
    Provides backward compatibility while using enhanced functionality.
    """
    
    def __init__(self, data_path=None):
        super().__init__(data_path)
        # Add legacy attributes for compatibility
        self.linear_model = None
        self.xgb_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.target_col = None
    
    def preprocess_data(self, target_col, feature_cols, test_size=0.2):
        """Legacy preprocessing method - now uses enhanced methodology."""
        logger.info("Using legacy preprocessing method with enhanced backend")
        
        # Train enhanced models
        results = self.train_enhanced_models(target_col, feature_cols, test_size, optimize_params=False)
        
        # Set legacy attributes for compatibility
        self.target_col = target_col
        self.feature_cols = feature_cols
        
        if target_col in self.train_data:
            train_data = self.train_data[target_col]
            self.X_train = train_data['X_train'].values
            self.X_test = train_data['X_test'].values
            self.y_train = train_data['y_train'].values
            self.y_test = train_data['y_test'].values
            self.scaler = self.scalers[target_col]
    
    def train_linear_model(self):
        """Legacy linear model training."""
        if self.target_col and self.target_col in self.models:
            self.linear_model = self.models[self.target_col].get('linear')
    
    def train_xgboost_model(self, params=None):
        """Legacy XGBoost training.""" 
        if self.target_col and self.target_col in self.models:
            self.xgb_model = self.models[self.target_col].get('xgboost')
    
    def evaluate_models(self):
        """Legacy model evaluation."""
        if self.target_col and self.target_col in self.cv_results:
            return self.cv_results[self.target_col]
        return {}
    
    def save_model(self, filename):
        """Legacy model saving."""
        if self.target_col:
            self.save_complete_model(self.target_col, filename)
    
    def load_model(self, filename):
        """Legacy model loading - enhanced to detect target from filename."""
        # Try to detect target from filename
        target_col = 'Gold' if 'gold' in filename.lower() else 'BITCOIN' if 'bitcoin' in filename.lower() else None
        
        if target_col:
            try:
                self.load_complete_model(filename, target_col)
                self.target_col = target_col
            except:
                # Fallback to old pickle format
                with open(filename, 'rb') as f:
                    obj = pickle.load(f)
                    if 'xgb_model' in obj:
                        self.xgb_model = obj['xgb_model']
                        self.scaler = obj.get('scaler', StandardScaler())
                        self.feature_cols = obj.get('feature_cols', [])
                        self.target_col = obj.get('target_col', target_col)
