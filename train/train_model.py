import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import ImprovedDataModel as DataModel

def create_technical_features(df):
    """Create technical indicators and lag features"""
    df_enhanced = df.copy()
    
    # Create features for each numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Lag features (1, 2, 3, 5, 10 days)
        for lag in [1, 2, 3, 5, 10]:
            df_enhanced[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Moving averages (5, 10, 20, 50 days)
        for window in [5, 10, 20, 50]:
            df_enhanced[f'{col}_ma_{window}'] = df[col].rolling(window=window).mean()
        
        # Exponential moving averages
        for window in [5, 10, 20]:
            df_enhanced[f'{col}_ema_{window}'] = df[col].ewm(span=window).mean()
        
        # Volatility (rolling standard deviation)
        for window in [5, 10, 20]:
            df_enhanced[f'{col}_volatility_{window}'] = df[col].rolling(window=window).std()
        
        # Price change features
        df_enhanced[f'{col}_change_1d'] = df[col].pct_change(1)
        df_enhanced[f'{col}_change_5d'] = df[col].pct_change(5)
        df_enhanced[f'{col}_change_10d'] = df[col].pct_change(10)
        
        # Momentum indicators
        df_enhanced[f'{col}_momentum_5'] = df[col] / df[col].shift(5) - 1
        df_enhanced[f'{col}_momentum_10'] = df[col] / df[col].shift(10) - 1
        
        # RSI (Relative Strength Index) - simplified version
        delta = df[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_enhanced[f'{col}_rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for window in [20]:
            rolling_mean = df[col].rolling(window=window).mean()
            rolling_std = df[col].rolling(window=window).std()
            df_enhanced[f'{col}_bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
            df_enhanced[f'{col}_bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
            df_enhanced[f'{col}_bb_position_{window}'] = (df[col] - rolling_mean) / (rolling_std * 2)
    
    # Cross-asset features
    if 'S&P500' in df.columns and 'Gold' in df.columns:
        df_enhanced['sp500_gold_ratio'] = df['S&P500'] / df['Gold']
        df_enhanced['sp500_gold_correlation_10'] = df['S&P500'].rolling(10).corr(df['Gold'])
    
    if 'Brent Oil' in df.columns and 'Crude Oil WTI' in df.columns:
        df_enhanced['oil_spread'] = df['Brent Oil'] - df['Crude Oil WTI']
        df_enhanced['oil_ratio'] = df['Brent Oil'] / df['Crude Oil WTI']
    
    # VIX-like indicator (volatility of S&P500)
    if 'S&P500' in df.columns:
        df_enhanced['market_volatility'] = df['S&P500'].rolling(window=20).std()
    
    # Remove rows with NaN values (from rolling calculations)
    df_enhanced = df_enhanced.dropna()
    
    return df_enhanced

def time_series_split_data(X, y, test_size=0.2, n_splits=5):
    """Split data using time series methodology"""
    # Calculate split point for final train/test
    split_point = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    
    # Create TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    return X_train, X_test, y_train, y_test, tscv

def plot_feature_importance(model, feature_names, output_path='feature_importance.png', top_n=20):
    """Plot feature importance for a model"""
    plt.figure(figsize=(12, 8))
    
    if hasattr(model, 'feature_importances_'):
        # For models with feature_importances_ attribute
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.title('Feature Importance (Tree-based Model)')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    else:
        # For models with coef_ attribute
        if hasattr(model, 'coef_'):
            coefs = model.coef_
            if len(coefs.shape) > 1:
                coefs = coefs[0]
                
            indices = np.argsort(np.abs(coefs))[::-1][:top_n]
            sorted_coefs = coefs[indices]
            
            plt.title('Feature Coefficients (Linear Model)')
            plt.barh(range(len(indices)), sorted_coefs, align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    
    plt.xlabel('Importance/Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(y_true, y_pred, title, output_path):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series plot
    axes[0, 0].plot(y_true.values, label='Actual', color='blue', linewidth=2)
    axes[0, 0].plot(y_pred, label='Predicted', color='red', linewidth=2, linestyle='--')
    axes[0, 0].set_title(f'{title} - Time Series')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[0, 1].scatter(y_true, y_pred, alpha=0.5)
    axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 1].set_title(f'{title} - Actual vs Predicted')
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true.values - y_pred
    axes[1, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title(f'{title} - Residuals')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].set_title(f'{title} - Residuals Distribution')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(data, output_path):
    """Plot correlation matrix for numeric features"""
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Limit to most relevant features to avoid overcrowding
    if len(numeric_data.columns) > 50:
        # Select original features plus some key technical indicators
        original_cols = ['S&P500', 'Brent Oil', 'Crude Oil WTI', 'Gold', 'BITCOIN']
        tech_cols = [col for col in numeric_data.columns if any(x in col for x in ['_ma_', '_rsi', '_change_', '_momentum_'])]
        selected_cols = [col for col in original_cols if col in numeric_data.columns] + tech_cols[:20]
        numeric_data = numeric_data[selected_cols]
    
    corr_matrix = numeric_data.corr()
    
    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, fmt=".2f", cmap='coolwarm', 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix (Selected Features)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def optimize_hyperparameters(X_train, y_train, tscv, model_type='xgboost'):
    """Optimize hyperparameters using GridSearchCV with TimeSeriesSplit"""
    print(f"Optimizing hyperparameters for {model_type}...")
    
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(random_state=42, verbosity=0)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    elif model_type == 'linear':
        model = LinearRegression()
        param_grid = {}  # Linear regression doesn't have hyperparameters to tune
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    if param_grid:
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for {model_type}: {grid_search.best_params_}")
        print(f"Best cross-validation score: {-grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)
        return model

def cross_validate_model(model, X_train, y_train, tscv):
    """Perform cross-validation using TimeSeriesSplit"""
    cv_scores = []
    
    for train_idx, val_idx in tscv.split(X_train):
        X_train_cv = X_train.iloc[train_idx]
        X_val_cv = X_train.iloc[val_idx]
        y_train_cv = y_train.iloc[train_idx]
        y_val_cv = y_train.iloc[val_idx]
        
        # Fit model on training fold
        model.fit(X_train_cv, y_train_cv)
        
        # Predict on validation fold
        y_pred_cv = model.predict(X_val_cv)
        
        # Calculate score
        score = r2_score(y_val_cv, y_pred_cv)
        cv_scores.append(score)
    
    return cv_scores

def enhanced_train_and_evaluate(data_model, target_col, feature_cols, output_dir='train/output'):
    """Enhanced training and evaluation with time series methodology"""
    print(f"\n{'='*60}")
    print(f"Enhanced Training for target: {target_col}")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and enhance data with technical features
    print("\nCreating technical features...")
    enhanced_data = create_technical_features(data_model.data)
    
    print(f"Original features: {len(feature_cols)}")
    print(f"Enhanced features: {enhanced_data.shape[1] - 1}")  # -1 for target
    
    # Prepare features and target
    # Use all available features (except target) for enhanced model
    date_columns = ['Date', 'date', 'Datetime', 'datetime']
    all_features = [col for col in enhanced_data.columns if col != target_col and col not in date_columns]
    X = enhanced_data[all_features]
    y = enhanced_data[target_col]
    
    print(f"Final feature count: {len(all_features)}")
    print(f"Data points after feature engineering: {len(X)}")
    
    # Time series split
    print("\nPerforming time series split...")
    X_train, X_test, y_train, y_test, tscv = time_series_split_data(X, y)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Scale features
    print("\nScaling features...")
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
    
    # Train models with hyperparameter optimization
    models = {}
    
    # Linear Regression
    print("\nTraining Linear Regression...")
    models['linear'] = optimize_hyperparameters(X_train_scaled, y_train, tscv, 'linear')
    
    # Random Forest
    print("\nTraining Random Forest...")
    models['random_forest'] = optimize_hyperparameters(X_train, y_train, tscv, 'random_forest')
    
    # XGBoost
    print("\nTraining XGBoost...")
    models['xgboost'] = optimize_hyperparameters(X_train, y_train, tscv, 'xgboost')
    
    # Evaluate models
    print("\nEvaluating models...")
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name.upper()} Results:")
        
        # Cross-validation scores
        if model_name == 'linear':
            cv_scores = cross_validate_model(model, X_train_scaled, y_train, tscv)
            y_pred = model.predict(X_test_scaled)
        else:
            cv_scores = cross_validate_model(model, X_train, y_train, tscv)
            y_pred = model.predict(X_test)
        
        # Test set metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[model_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores)
        }
        
        print(f"Cross-validation R² - Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")
        print(f"Test set - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Generate visualizations
        plot_feature_importance(
            model, 
            all_features, 
            os.path.join(output_dir, f'{target_col}_{model_name}_importance.png')
        )
        
        plot_predictions(
            y_test, 
            y_pred, 
            f'{model_name.replace("_", " ").title()}: {target_col}',
            os.path.join(output_dir, f'{target_col}_{model_name}_predictions.png')
        )
    
    # Save best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = models[best_model_name]
    
    print(f"\nBest model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})")
    
    # Save the best model and scaler
    import joblib
    joblib.dump(best_model, os.path.join(output_dir, f'{target_col}_best_model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, f'{target_col}_scaler.pkl'))
    
    return results, enhanced_data

def main():
    """Enhanced main function with improved methodology"""
    data_file = os.path.join('data', 'SP500 oil gold bitcoin.csv')
    
    # Initialize data model
    model = DataModel(data_file)
    
    # Load data
    print("Loading data...")
    if not model.load_data():
        print("Failed to load data. Exiting.")
        return
    
    print(f"Data loaded successfully. Shape: {model.data.shape}")
    print(f"Date range: {model.data.index.min()} to {model.data.index.max()}")
    
    # Examine data
    print("\nData Overview:")
    print(model.data.head())
    print("\nData Info:")
    print(model.data.info())
    
    # Enhanced training for Gold
    if 'Gold' in model.data.columns:
        print("\n" + "="*80)
        print("GOLD PREDICTION MODEL")
        print("="*80)
        
        gold_features = ['S&P500', 'Brent Oil', 'Crude Oil WTI', 'BITCOIN']
        gold_features = [f for f in gold_features if f in model.data.columns]
        
        gold_results, gold_enhanced_data = enhanced_train_and_evaluate(
            model, 'Gold', gold_features, 'train/output/gold'
        )
        
        # Plot correlation matrix for gold analysis
        plot_correlation_matrix(
            gold_enhanced_data, 
            'train/output/gold/gold_correlation_matrix.png'
        )
    
    # Enhanced training for Bitcoin
    if 'BITCOIN' in model.data.columns:
        print("\n" + "="*80)
        print("BITCOIN PREDICTION MODEL")
        print("="*80)
        
        bitcoin_features = ['S&P500', 'Brent Oil', 'Crude Oil WTI', 'Gold']
        bitcoin_features = [f for f in bitcoin_features if f in model.data.columns]
        
        bitcoin_results, bitcoin_enhanced_data = enhanced_train_and_evaluate(
            model, 'BITCOIN', bitcoin_features, 'train/output/bitcoin'
        )
        
        # Plot correlation matrix for bitcoin analysis
        plot_correlation_matrix(
            bitcoin_enhanced_data, 
            'train/output/bitcoin/bitcoin_correlation_matrix.png'
        )
    
    print("\n" + "="*80)
    print("ENHANCED TRAINING COMPLETED!")
    print("="*80)
    print("\nKey Improvements Implemented:")
    print("✓ Time-based train/test splitting (instead of random)")
    print("✓ Technical indicators (MA, EMA, RSI, Bollinger Bands)")
    print("✓ Lag features and momentum indicators")
    print("✓ Cross-validation with TimeSeriesSplit")
    print("✓ Hyperparameter optimization with GridSearchCV")
    print("✓ Multiple model comparison (Linear, Random Forest, XGBoost)")
    print("✓ Enhanced visualizations with residual analysis")
    print("✓ Feature scaling for linear models")
    print("✓ Model persistence with joblib")

if __name__ == "__main__":
    main()
