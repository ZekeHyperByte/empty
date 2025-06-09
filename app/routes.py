from flask import render_template, jsonify, request
import os
import pandas as pd
import json
import numpy as np
import logging
from typing import Dict, Any, Optional
from app.models import DataModel  # Now uses the improved DataModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the data model with the path to the data file
data_model = DataModel(os.path.join('data', 'SP500 oil gold bitcoin.csv'))

def to_python_type(obj: Any) -> Any:
    """Convert numpy types and complex structures to Python native types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return obj

def ensure_data_loaded() -> bool:
    """Ensure data is loaded into the model."""
    if data_model.data is None:
        success = data_model.load_data()
        if not success:
            logger.error("Failed to load data")
            return False
    return True

def get_standardized_column_names() -> Dict[str, str]:
    """Get mapping of potential column names to standardized names."""
    return {
        'Gold': 'Gold',
        'BITCOIN': 'Bitcoin',  # For display purposes
        'S&P500': 'S&P 500',
        'Brent Oil': 'Brent Oil',
        'Crude Oil WTI': 'WTI Oil'
    }

def init_routes(app):
    """Initialize all application routes with enhanced functionality"""
    
    @app.route('/')
    def index():
        """Route for the home page"""
        return render_template('index.html')
    
    @app.route('/data')
    def data():
        """Route for the data page with enhanced statistics"""
        if not ensure_data_loaded():
            return render_template('error.html', 
                                 error_message="Failed to load data. Please check the data file."), 500
        
        try:
            # Get comprehensive data summary
            stats = data_model.get_data_summary()
            stats = to_python_type(stats)
            
            return render_template('data.html', stats=stats)
        except Exception as e:
            logger.error(f"Error in data route: {str(e)}")
            return render_template('error.html', 
                                 error_message=f"Error processing data: {str(e)}"), 500
    
    @app.route('/analytics')
    def analytics():
        """Route for the analytics page with enhanced charting data"""
        if not ensure_data_loaded():
            return render_template('error.html', 
                                 error_message="Failed to load data. Please check the data file."), 500
        
        try:
            # Get the last 100 data points for charting with proper column names
            chart_data = {}
            
            # Handle date column
            date_col = None
            for col in ['Date', 'date', 'Datetime', 'datetime']:
                if col in data_model.data.columns:
                    date_col = col
                    break
            
            if date_col:
                chart_data['dates'] = data_model.data[date_col].tail(100).dt.strftime('%Y-%m-%d').tolist()
            else:
                chart_data['dates'] = list(range(len(data_model.data.tail(100))))
            
            # Map actual column names to chart data
            column_mapping = {
                'Gold': 'gold',
                'BITCOIN': 'bitcoin', 
                'S&P500': 'sp500',
                'Brent Oil': 'oil_brent',
                'Crude Oil WTI': 'oil_wti'
            }
            
            for actual_col, chart_key in column_mapping.items():
                if actual_col in data_model.data.columns:
                    chart_data[chart_key] = data_model.data[actual_col].tail(100).tolist()
                else:
                    chart_data[chart_key] = []
            
            # Add technical indicators for visualization
            try:
                enhanced_data = data_model.create_technical_features()
                
                # Add moving averages for Gold and Bitcoin
                for asset in ['Gold', 'BITCOIN']:
                    if asset in enhanced_data.columns:
                        asset_key = 'gold' if asset == 'Gold' else 'bitcoin'
                        chart_data[f'{asset_key}_ma20'] = enhanced_data[f'{asset}_ma_20'].tail(100).tolist()
                        chart_data[f'{asset_key}_volatility'] = enhanced_data[f'{asset}_volatility_20'].tail(100).tolist()
            except Exception as e:
                logger.warning(f"Could not add technical indicators: {str(e)}")
            
            chart_data = to_python_type(chart_data)
            return render_template('analytics.html', chart_data=chart_data)
            
        except Exception as e:
            logger.error(f"Error in analytics route: {str(e)}")
            return render_template('error.html', 
                                 error_message=f"Error generating analytics: {str(e)}"), 500
    
    @app.route('/models')
    def models():
        """Route for the models page with enhanced model information"""
        if not ensure_data_loaded():
            return render_template('error.html', 
                                 error_message="Failed to load data. Please check the data file."), 500
        
        try:
            # Get model summaries
            model_summary = data_model.get_model_summary()
            
            # Try to load saved models and get their performance
            gold_metrics = None
            bitcoin_metrics = None
            
            # Check for enhanced model files first
            for asset, target_col in [('gold', 'Gold'), ('bitcoin', 'BITCOIN')]:
                try:
                    enhanced_model_path = f'train/output/{asset}/{target_col.lower()}_best_model.pkl'
                    legacy_model_path = f'train/output/{asset}_xgb_model.pkl'
                    
                    if os.path.exists(enhanced_model_path):
                        # Load enhanced model
                        data_model.load_complete_model(enhanced_model_path, target_col)
                        performance = data_model.evaluate_model_performance(target_col)
                        if asset == 'gold':
                            gold_metrics = to_python_type(performance)
                        else:
                            bitcoin_metrics = to_python_type(performance)
                    elif os.path.exists(legacy_model_path):
                        # Load legacy model
                        data_model.load_model(legacy_model_path)
                        metrics = data_model.evaluate_models()
                        if 'xgboost' in metrics:
                            if asset == 'gold':
                                gold_metrics = to_python_type(metrics['xgboost'])
                            else:
                                bitcoin_metrics = to_python_type(metrics['xgboost'])
                except Exception as e:
                    logger.warning(f"Could not load {asset} model: {str(e)}")
            
            return render_template('models.html', 
                                 gold_metrics=gold_metrics, 
                                 bitcoin_metrics=bitcoin_metrics,
                                 model_summary=to_python_type(model_summary))
            
        except Exception as e:
            logger.error(f"Error in models route: {str(e)}")
            return render_template('error.html', 
                                 error_message=f"Error loading models: {str(e)}"), 500
    
    # Enhanced API Endpoints
    
    @app.route('/api/data/summary')
    def api_data_summary():
        """API endpoint to get comprehensive data summary"""
        if not ensure_data_loaded():
            return jsonify({'error': 'Failed to load data'}), 500
        
        try:
            stats = data_model.get_data_summary()
            return jsonify({
                'success': True,
                'data': to_python_type(stats)
            })
        except Exception as e:
            logger.error(f"Error in data summary API: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/model/train', methods=['POST'])
    def api_model_train():
        """API endpoint to train models with enhanced methodology"""
        try:
            # Get parameters from request
            params = request.json or {}
            target_col = params.get('target_col')
            feature_cols = params.get('feature_cols')
            optimize_params = params.get('optimize_params', True)
            test_size = params.get('test_size', 0.2)
            
            # Validate parameters
            if not target_col:
                return jsonify({'error': 'Missing required parameter: target_col'}), 400
            
            if target_col not in ['Gold', 'BITCOIN']:
                return jsonify({'error': 'target_col must be "Gold" or "BITCOIN"'}), 400
            
            if not ensure_data_loaded():
                return jsonify({'error': 'Failed to load data'}), 500
            
            # Train enhanced models
            results = data_model.train_enhanced_models(
                target_col=target_col,
                feature_cols=feature_cols,
                test_size=test_size,
                optimize_params=optimize_params
            )
            
            # Get feature importance for best model
            best_model_type = data_model.best_models[target_col]['type']
            feature_importance = data_model.get_feature_importance(target_col, best_model_type)
            
            # Save the trained model
            os.makedirs(f'train/output/{target_col.lower()}', exist_ok=True)
            data_model.save_complete_model(
                target_col, 
                f'train/output/{target_col.lower()}/{target_col.lower()}_best_model.pkl'
            )
            
            return jsonify({
                'success': True,
                'results': to_python_type(results),
                'best_model': best_model_type,
                'feature_importance': to_python_type(feature_importance),
                'model_saved': True
            })
            
        except Exception as e:
            logger.error(f"Error in model training API: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/model/predict', methods=['POST'])
    def api_model_predict():
        """API endpoint to make enhanced predictions with confidence intervals"""
        try:
            params = request.json or {}
            target_col = params.get('target_col', 'Gold')
            steps = params.get('steps', 30)
            model_type = params.get('model_type')  # None = use best model
            return_confidence = params.get('return_confidence', True)
            
            if target_col not in ['Gold', 'BITCOIN']:
                return jsonify({'error': 'target_col must be "Gold" or "BITCOIN"'}), 400
            
            if not ensure_data_loaded():
                return jsonify({'error': 'Failed to load data'}), 500
            
            # Check if model exists
            if target_col not in data_model.models:
                # Try to load saved model
                try:
                    enhanced_path = f'train/output/{target_col.lower()}/{target_col.lower()}_best_model.pkl'
                    legacy_path = f'train/output/{target_col.lower()}_xgb_model.pkl'
                    
                    if os.path.exists(enhanced_path):
                        data_model.load_complete_model(enhanced_path, target_col)
                    elif os.path.exists(legacy_path):
                        data_model.load_model(legacy_path)
                    else:
                        return jsonify({'error': f'No trained model found for {target_col}'}), 400
                except Exception as e:
                    return jsonify({'error': f'Failed to load model: {str(e)}'}), 400
            
            # Make predictions
            if return_confidence:
                predictions, confidence_intervals = data_model.predict_future(
                    target_col, steps, model_type, return_confidence=True
                )
            else:
                predictions = data_model.predict_future(
                    target_col, steps, model_type, return_confidence=False
                )
                confidence_intervals = None
            
            # Generate future dates
            future_dates = []
            date_col = None
            for col in ['Date', 'date', 'Datetime', 'datetime']:
                if col in data_model.data.columns:
                    date_col = col
                    break
            
            if date_col:
                last_date = data_model.data[date_col].iloc[-1]
                for i in range(1, steps + 1):
                    future_date = pd.Timestamp(last_date) + pd.Timedelta(days=i)
                    future_dates.append(future_date.strftime('%Y-%m-%d'))
            else:
                last_idx = len(data_model.data)
                future_dates = list(range(last_idx, last_idx + steps))
            
            response = {
                'success': True,
                'target': target_col,
                'dates': future_dates,
                'predictions': [float(p) for p in predictions],
                'model_type': model_type or data_model.best_models[target_col]['type']
            }
            
            if confidence_intervals:
                response['confidence_intervals'] = [
                    [float(ci[0]), float(ci[1])] for ci in confidence_intervals
                ]
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in prediction API: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/data/correlation')
    def api_data_correlation():
        """API endpoint to get enhanced correlation matrix"""
        if not ensure_data_loaded():
            return jsonify({'error': 'Failed to load data'}), 500
        
        try:
            # Get correlation matrix for numeric columns
            numeric_cols = data_model.data.select_dtypes(include=[np.number]).columns
            corr_matrix = data_model.data[numeric_cols].corr().round(3)
            
            # Convert to dictionary format
            corr_dict = {}
            for col in corr_matrix.columns:
                corr_dict[col] = corr_matrix[col].to_dict()
            
            # Also get cross-asset correlations specifically
            cross_correlations = {}
            if 'Gold' in numeric_cols and 'BITCOIN' in numeric_cols:
                cross_correlations['Gold_Bitcoin'] = float(
                    data_model.data['Gold'].corr(data_model.data['BITCOIN'])
                )
            if 'S&P500' in numeric_cols and 'Gold' in numeric_cols:
                cross_correlations['SP500_Gold'] = float(
                    data_model.data['S&P500'].corr(data_model.data['Gold'])
                )
            
            return jsonify({
                'success': True,
                'correlation_matrix': to_python_type(corr_dict),
                'cross_correlations': cross_correlations,
                'columns': numeric_cols.tolist()
            })
            
        except Exception as e:
            logger.error(f"Error in correlation API: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/model/trading-recommendation', methods=['POST'])
    def api_trading_recommendation():
        """API endpoint to get enhanced trading recommendations"""
        try:
            params = request.json or {}
            asset = params.get('asset', '').lower()
            timeframe = params.get('timeframe', '1d')
            confidence_threshold = params.get('confidence_threshold', 0.6)
            
            # Validate parameters
            if asset not in ['gold', 'bitcoin']:
                return jsonify({'error': 'Invalid asset. Must be "gold" or "bitcoin"'}), 400
            
            if timeframe not in ['1d', '1w', '1m']:
                return jsonify({'error': 'Invalid timeframe. Must be "1d", "1w", or "1m"'}), 400
            
            if not ensure_data_loaded():
                return jsonify({'error': 'Failed to load data'}), 500
            
            # Map asset to target column
            target_col = 'BITCOIN' if asset == 'bitcoin' else 'Gold'
            
            # Check if model exists, try to load if not
            if target_col not in data_model.models:
                try:
                    enhanced_path = f'train/output/{asset}/{asset}_best_model.pkl'
                    legacy_path = f'train/output/{asset}_xgb_model.pkl'
                    
                    if os.path.exists(enhanced_path):
                        data_model.load_complete_model(enhanced_path, target_col)
                    elif os.path.exists(legacy_path):
                        # For legacy compatibility, train a quick model
                        data_model.train_enhanced_models(target_col, optimize_params=False)
                    else:
                        return jsonify({'error': f'No trained model found for {asset}'}), 400
                except Exception as e:
                    return jsonify({'error': f'Failed to load/train model: {str(e)}'}), 400
            
            # Get trading recommendation
            recommendation = data_model.get_trading_recommendation(
                asset, timeframe, confidence_threshold
            )
            
            return jsonify({
                'success': True,
                'asset': asset,
                'timeframe': timeframe,
                'recommendation': to_python_type(recommendation)
            })
            
        except Exception as e:
            logger.error(f"Error in trading recommendation API: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/model/performance')
    def api_model_performance():
        """API endpoint to get comprehensive model performance metrics"""
        try:
            if not ensure_data_loaded():
                return jsonify({'error': 'Failed to load data'}), 500
            
            performance_data = {}
            
            # Check both Gold and Bitcoin models
            for asset, target_col in [('gold', 'Gold'), ('bitcoin', 'BITCOIN')]:
                try:
                    # Try to load enhanced model first
                    enhanced_path = f'train/output/{asset}/{asset}_best_model.pkl'
                    legacy_path = f'train/output/{asset}_xgb_model.pkl'
                    
                    if os.path.exists(enhanced_path):
                        data_model.load_complete_model(enhanced_path, target_col)
                        performance = data_model.evaluate_model_performance(target_col)
                        performance_data[asset] = {
                            'model_type': 'enhanced',
                            'performance': to_python_type(performance),
                            'feature_importance': to_python_type(
                                data_model.get_feature_importance(target_col, top_n=10)
                            )
                        }
                    elif os.path.exists(legacy_path):
                        data_model.load_model(legacy_path)
                        metrics = data_model.evaluate_models()
                        if 'xgboost' in metrics:
                            performance_data[asset] = {
                                'model_type': 'legacy',
                                'performance': {'xgboost': to_python_type(metrics['xgboost'])},
                                'feature_importance': {}
                            }
                    else:
                        performance_data[asset] = {
                            'model_type': 'none',
                            'performance': None,
                            'error': 'No trained model found'
                        }
                        
                except Exception as e:
                    performance_data[asset] = {
                        'model_type': 'error',
                        'performance': None,
                        'error': str(e)
                    }
            
            return jsonify({
                'success': True,
                'models': performance_data
            })
            
        except Exception as e:
            logger.error(f"Error in model performance API: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/model/backtest', methods=['POST'])
    def api_model_backtest():
        """API endpoint for backtesting trading strategies"""
        try:
            params = request.json or {}
            target_col = params.get('target_col', 'Gold')
            start_date = params.get('start_date')
            end_date = params.get('end_date')
            initial_capital = params.get('initial_capital', 10000)
            
            if target_col not in ['Gold', 'BITCOIN']:
                return jsonify({'error': 'target_col must be "Gold" or "BITCOIN"'}), 400
            
            if not ensure_data_loaded():
                return jsonify({'error': 'Failed to load data'}), 500
            
            # Ensure model is trained
            if target_col not in data_model.models:
                # Try to load or train model
                asset = 'gold' if target_col == 'Gold' else 'bitcoin'
                try:
                    enhanced_path = f'train/output/{asset}/{asset}_best_model.pkl'
                    if os.path.exists(enhanced_path):
                        data_model.load_complete_model(enhanced_path, target_col)
                    else:
                        # Train a quick model for backtesting
                        data_model.train_enhanced_models(target_col, optimize_params=False)
                except Exception as e:
                    return jsonify({'error': f'Failed to prepare model for backtesting: {str(e)}'}), 400
            
            # Run backtest
            backtest_results = data_model.backtest_strategy(
                target_col, start_date, end_date, initial_capital
            )
            
            return jsonify({
                'success': True,
                'target': target_col,
                'backtest_results': to_python_type(backtest_results)
            })
            
        except Exception as e:
            logger.error(f"Error in backtest API: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/data/technical-indicators', methods=['POST'])
    def api_technical_indicators():
        """API endpoint to get technical indicators for visualization"""
        try:
            params = request.json or {}
            target_col = params.get('target_col', 'Gold')
            indicators = params.get('indicators', ['ma_20', 'rsi', 'volatility_20'])
            lookback_days = params.get('lookback_days', 100)
            
            if not ensure_data_loaded():
                return jsonify({'error': 'Failed to load data'}), 500
            
            # Create technical features
            enhanced_data = data_model.create_technical_features()
            
            # Get the requested indicators
            indicator_data = {}
            
            # Get date column for x-axis
            date_col = None
            for col in ['Date', 'date', 'Datetime', 'datetime']:
                if col in enhanced_data.columns:
                    date_col = col
                    break
            
            if date_col:
                indicator_data['dates'] = enhanced_data[date_col].tail(lookback_days).dt.strftime('%Y-%m-%d').tolist()
            else:
                indicator_data['dates'] = list(range(len(enhanced_data.tail(lookback_days))))
            
            # Add base price data
            if target_col in enhanced_data.columns:
                indicator_data['price'] = enhanced_data[target_col].tail(lookback_days).tolist()
            
            # Add requested technical indicators
            for indicator in indicators:
                indicator_col = f'{target_col}_{indicator}'
                if indicator_col in enhanced_data.columns:
                    indicator_data[indicator] = enhanced_data[indicator_col].tail(lookback_days).tolist()
            
            return jsonify({
                'success': True,
                'target': target_col,
                'indicators': to_python_type(indicator_data)
            })
            
        except Exception as e:
            logger.error(f"Error in technical indicators API: {str(e)}")
            return jsonify({'error': str(e)}), 500
