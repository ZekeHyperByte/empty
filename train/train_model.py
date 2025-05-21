import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import DataModel

def plot_feature_importance(model, feature_names, output_path='feature_importance.png'):
    """Plot feature importance for a model"""
    plt.figure(figsize=(12, 8))
    
    if hasattr(model, 'feature_importances_'):
        # For models with feature_importances_ attribute (like RandomForest or XGBoost)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]  # Sort in descending order
        
        plt.title('Feature Importance (XGBoost/RandomForest)')
        plt.barh(range(min(20, len(indices))), importances[indices[:20]], align='center')
        plt.yticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]])
    else:
        # For models with coef_ attribute (like LinearRegression)
        if hasattr(model, 'coef_'):
            coefs = model.coef_
            if len(coefs.shape) > 1:
                coefs = coefs[0]  # For multi-output models
                
            # Sort coefficients by absolute value
            indices = np.argsort(np.abs(coefs))[::-1]  # Sort in descending order
            sorted_coefs = coefs[indices[:20]]
            
            plt.title('Feature Coefficients (Linear Model)')
            plt.barh(range(min(20, len(indices))), sorted_coefs, align='center')
            plt.yticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]])
    
    plt.xlabel('Importance/Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_predictions(y_true, y_pred, title, output_path):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    
    # Plot actual values
    plt.plot(y_true, label='Actual', color='blue', linewidth=2)
    
    # Plot predicted values
    plt.plot(y_pred, label='Predicted', color='red', linewidth=2, linestyle='--')
    
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_correlation_matrix(data, output_path):
    """Plot correlation matrix for numeric features"""
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def train_and_evaluate(data_model, target_col, feature_cols, output_dir='train/output'):
    """Train and evaluate models for a specific target"""
    print(f"\n{'='*50}")
    print(f"Training models for target: {target_col}")
    print(f"{'='*50}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess data
    print("\nPreprocessing data...")
    data_model.preprocess_data(target_col, feature_cols)
    
    # Train linear regression model
    print("\nTraining Linear Regression model...")
    data_model.train_linear_model()
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    data_model.train_xgboost_model()
    
    # Evaluate models
    metrics = data_model.evaluate_models()
    
    print("\nModel Evaluation:")
    
    # Linear Regression results
    if 'linear' in metrics:
        print("\nLinear Regression:")
        print(f"MSE: {metrics['linear']['mse']:.4f}")
        print(f"RMSE: {metrics['linear']['rmse']:.4f}")
        print(f"MAE: {metrics['linear']['mae']:.4f}")
        print(f"R²: {metrics['linear']['r2']:.4f}")
        
        # Plot feature importance for Linear Regression
        print("\nPlotting Linear Regression feature importance...")
        plot_feature_importance(
            data_model.linear_model, 
            feature_cols, 
            os.path.join(output_dir, f'{target_col}_linear_importance.png')
        )
        
        # Plot predictions for Linear Regression
        y_pred_linear = data_model.linear_model.predict(data_model.X_test)
        plot_predictions(
            data_model.y_test, 
            y_pred_linear, 
            f'Linear Regression: Actual vs Predicted {target_col}',
            os.path.join(output_dir, f'{target_col}_linear_predictions.png')
        )
    
    # XGBoost results
    if 'xgboost' in metrics:
        print("\nXGBoost:")
        print(f"MSE: {metrics['xgboost']['mse']:.4f}")
        print(f"RMSE: {metrics['xgboost']['rmse']:.4f}")
        print(f"MAE: {metrics['xgboost']['mae']:.4f}")
        print(f"R²: {metrics['xgboost']['r2']:.4f}")
        
        # Plot feature importance for XGBoost
        print("\nPlotting XGBoost feature importance...")
        plot_feature_importance(
            data_model.xgb_model, 
            feature_cols, 
            os.path.join(output_dir, f'{target_col}_xgboost_importance.png')
        )
        
        # Plot predictions for XGBoost
        y_pred_xgb = data_model.xgb_model.predict(data_model.X_test)
        plot_predictions(
            data_model.y_test, 
            y_pred_xgb, 
            f'XGBoost: Actual vs Predicted {target_col}',
            os.path.join(output_dir, f'{target_col}_xgboost_predictions.png')
        )
    
    # Make future predictions with XGBoost
    print("\nGenerating future predictions...")
    future_steps = 30  # Predict 30 days into the future
    predictions = data_model.predict_future(future_steps, 'xgboost')
    
    print(f"\nFuture {target_col} predictions (next {future_steps} days):")
    for i, pred in enumerate(predictions[:5], 1):  # Show first 5 predictions
        print(f"Day {i}: {pred:.2f}")
    print("...")
    
    # Save the trained XGBoost model
    data_model.save_model(os.path.join(output_dir, f'{target_col}_xgb_model.pkl'))
    
    return metrics

def main():
    # Data file path
    data_file = os.path.join('data', 'SP500 oil gold bitcoin.csv')
    
    # Initialize data model
    model = DataModel(data_file)
    
    # Load data
    print("Loading data...")
    if not model.load_data():
        print("Failed to load data. Exiting.")
        return
    
    print(f"Data loaded successfully. Shape: {model.data.shape}")
    
    # Examine data
    print("\nData Overview:")
    print(model.data.head())
    
    # Plot correlation matrix
    print("\nGenerating correlation matrix...")
    plot_correlation_matrix(model.data, 'train/output/correlation_matrix.png')
    
    # Define targets and features for Gold prediction
    if 'Gold' in model.data.columns:
        gold_features = ['S&P500', 'Brent Oil', 'Crude Oil WTI', 'BITCOIN']
        gold_features = [f for f in gold_features if f in model.data.columns]
        gold_metrics = train_and_evaluate(model, 'Gold', gold_features)
        model.save_model(os.path.join('train/output', 'gold_xgb_model.pkl'))

    # Define targets and features for Bitcoin prediction
    if 'BITCOIN' in model.data.columns:
        bitcoin_features = ['S&P500', 'Brent Oil', 'Crude Oil WTI', 'Gold']
        bitcoin_features = [f for f in bitcoin_features if f in model.data.columns]
        bitcoin_metrics = train_and_evaluate(model, 'BITCOIN', bitcoin_features)
        model.save_model(os.path.join('train/output', 'bitcoin_xgb_model.pkl'))
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 