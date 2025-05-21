from flask import render_template, jsonify, request
import os
import pandas as pd
import json
import numpy as np
from app.models import DataModel

# Initialize the data model with the path to the data file
data_model = DataModel(os.path.join('data', 'SP500 oil gold bitcoin.csv'))

def to_python_type(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    return obj

def init_routes(app):
    """Initialize all application routes"""
    
    @app.route('/')
    def index():
        """Route for the home page"""
        return render_template('index.html')
    
    @app.route('/data')
    def data():
        """Route for the data page"""
        # Load data if not already loaded
        if data_model.data is None:
            data_model.load_data()
        
        # Get basic info about the data
        stats = data_model.get_summary_stats()
        
        return render_template('data.html', stats=stats)
    
    @app.route('/analytics')
    def analytics():
        """Route for the analytics page"""
        # Load data if not already loaded
        if data_model.data is None:
            data_model.load_data()
            
        # Get the last 100 data points for charting
        if 'Date' in data_model.data.columns:
            chart_data = {
                'dates': data_model.data['Date'].tail(100).dt.strftime('%Y-%m-%d').tolist(),
                'gold': data_model.data['Gold'].tail(100).tolist() if 'Gold' in data_model.data.columns else [],
                'bitcoin': data_model.data['Bitcoin'].tail(100).tolist() if 'Bitcoin' in data_model.data.columns else [],
                'sp500': data_model.data['SP500'].tail(100).tolist() if 'SP500' in data_model.data.columns else [],
                'oil': data_model.data['Oil'].tail(100).tolist() if 'Oil' in data_model.data.columns else []
            }
        else:
            chart_data = {
                'dates': list(range(len(data_model.data.tail(100)))),
                'gold': data_model.data['Gold'].tail(100).tolist() if 'Gold' in data_model.data.columns else [],
                'bitcoin': data_model.data['Bitcoin'].tail(100).tolist() if 'Bitcoin' in data_model.data.columns else [],
                'sp500': data_model.data['SP500'].tail(100).tolist() if 'SP500' in data_model.data.columns else [],
                'oil': data_model.data['Oil'].tail(100).tolist() if 'Oil' in data_model.data.columns else []
            }
            
        return render_template('analytics.html', chart_data=chart_data)
    
    @app.route('/models')
    def models():
        """Route for the models page"""
        if data_model.data is None:
            data_model.load_data()
        data_model.add_all_features()
        gold_metrics = None
        bitcoin_metrics = None
        try:
            data_model.load_model('train/output/gold_xgb_model.pkl')
            gold_metrics = data_model.evaluate_models().get('xgboost')
            gold_metrics = to_python_type(gold_metrics)
        except Exception as e:
            gold_metrics = None
        try:
            data_model.load_model('train/output/bitcoin_xgb_model.pkl')
            bitcoin_metrics = data_model.evaluate_models().get('xgboost')
            bitcoin_metrics = to_python_type(bitcoin_metrics)
        except Exception as e:
            bitcoin_metrics = None
        return render_template('models.html', gold_metrics=gold_metrics, bitcoin_metrics=bitcoin_metrics)
    
    @app.route('/api/data/summary')
    def api_data_summary():
        """API endpoint to get data summary"""
        # Load data if not already loaded
        if data_model.data is None:
            data_model.load_data()
        
        # Get basic info about the data
        stats = data_model.get_summary_stats()
        
        return jsonify(stats)
    
    @app.route('/api/model/train', methods=['POST'])
    def api_model_train():
        """API endpoint to train a model"""
        # Get parameters from request
        params = request.json
        target_col = params.get('target_col')
        feature_cols = params.get('feature_cols', [])
        model_type = params.get('model_type', 'xgboost')
        
        # Validate parameters
        if not target_col or not feature_cols:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        try:
            # Load data if not already loaded
            if data_model.data is None:
                data_model.load_data()
            
            # Preprocess data
            data_model.preprocess_data(target_col, feature_cols)
            
            # Train models
            if model_type in ['linear', 'both']:
                data_model.train_linear_model()
                
            if model_type in ['xgboost', 'both']:
                data_model.train_xgboost_model()
            
            # Evaluate models
            metrics = data_model.evaluate_models()
            
            # Get feature importance
            feature_importance = {}
            if model_type in ['linear', 'both'] and data_model.linear_model is not None:
                feature_importance['linear'] = data_model.get_feature_importance('linear')
                
            if model_type in ['xgboost', 'both'] and data_model.xgb_model is not None:
                feature_importance['xgboost'] = data_model.get_feature_importance('xgboost')
            
            return jsonify({
                'success': True,
                'metrics': metrics,
                'feature_importance': feature_importance
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/model/predict', methods=['POST'])
    def api_model_predict():
        """API endpoint to make predictions"""
        # Get parameters from request
        params = request.json
        steps = params.get('steps', 30)
        model_type = params.get('model_type', 'xgboost')
        
        try:
            # Make predictions
            predictions = data_model.predict_future(steps, model_type)
            
            # Generate future dates if Date column exists
            future_dates = []
            if 'Date' in data_model.data.columns:
                last_date = data_model.data['Date'].iloc[-1]
                for i in range(1, steps + 1):
                    future_date = pd.Timestamp(last_date) + pd.Timedelta(days=i)
                    future_dates.append(future_date.strftime('%Y-%m-%d'))
            else:
                # Use indices if no date column
                last_idx = len(data_model.data)
                future_dates = list(range(last_idx, last_idx + steps))
            
            return jsonify({
                'success': True,
                'dates': future_dates,
                'predictions': [float(p) for p in predictions]
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
            
    @app.route('/api/data/correlation')
    def api_data_correlation():
        """API endpoint to get correlation matrix"""
        # Load data if not already loaded
        if data_model.data is None:
            data_model.load_data()
        
        # Calculate correlation matrix for numeric columns
        numeric_cols = data_model.data.select_dtypes(include=[np.number]).columns
        corr_matrix = data_model.data[numeric_cols].corr().round(2).to_dict()
        
        return jsonify({
            'success': True,
            'correlation': corr_matrix
        })

    @app.route('/api/model/trading-recommendation', methods=['POST'])
    def api_trading_recommendation():
        """API endpoint to get trading recommendations"""
        try:
            params = request.json
            asset = params.get('asset')
            timeframe = params.get('timeframe', '1d')
            if not asset or asset.lower() not in ['gold', 'bitcoin']:
                return jsonify({'error': 'Invalid asset. Must be "gold" or "bitcoin"'}), 400
            if timeframe not in ['1d', '1w', '1m']:
                return jsonify({'error': 'Invalid timeframe. Must be "1d", "1w", or "1m"'}), 400
            if data_model.data is None:
                if not data_model.load_data():
                    return jsonify({'error': 'Failed to load data file. Please check if the data file exists and is properly formatted.'}), 500
            data_model.add_all_features()
            try:
                data_model.load_model(f'train/output/{asset.lower()}_xgb_model.pkl')
            except Exception as e:
                return jsonify({'error': f'Model not available: {str(e)}'}), 400
            try:
                recommendation = data_model.get_trading_recommendation(asset, timeframe)
                recommendation = to_python_type(recommendation)
            except Exception as e:
                return jsonify({'error': f'Error generating recommendation: {str(e)}'}), 500
            return jsonify({
                'success': True,
                'recommendation': recommendation
            })
        except Exception as e:
            return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

    @app.route('/api/model/performance')
    def api_model_performance():
        try:
            if data_model.data is None:
                data_model.load_data()
            data_model.add_all_features()
            gold_performance = None
            bitcoin_performance = None
            try:
                data_model.load_model('train/output/gold_xgb_model.pkl')
                gold_performance = data_model.evaluate_models().get('xgboost')
                gold_performance = to_python_type(gold_performance)
            except Exception as e:
                gold_performance = None
            try:
                data_model.load_model('train/output/bitcoin_xgb_model.pkl')
                bitcoin_performance = data_model.evaluate_models().get('xgboost')
                bitcoin_performance = to_python_type(bitcoin_performance)
            except Exception as e:
                bitcoin_performance = None
            return jsonify({
                'success': True,
                'gold': gold_performance,
                'bitcoin': bitcoin_performance
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500 