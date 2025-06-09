# Gold and Bitcoin Predictor

This project focuses on predicting Gold and Bitcoin prices using XGBoost algorithm based on Linear Regression techniques. The application analyzes historical data to identify patterns and make future price predictions.

## Important

Before you use this project, you need to create a virtual environment and install the needed packages:

### For Linux:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install Python packages
pip install Flask pandas matplotlib seaborn scikit-learn numpy xgboost
```

### For Windows:
```bash
# Create virtual environment
py -3 -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate

# Install Python packages
pip install Flask pandas matplotlib seaborn scikit-learn numpy xgboost
```

## Frontend Setup

Install these Node.js packages using npm:

```bash
# Install all required packages
npm install

# Or install them individually
npm install tailwindcss flowbite concurrently nodemon
```

## Running the Project

To run the project, use the following command:

```bash
npm run dev
```

## Project Features

- Historical data analysis of Gold and Bitcoin prices
- Price prediction using XGBoost algorithm
- Comparative analysis with Linear Regression models
- Interactive visualization dashboard

## Model Information

The prediction model utilizes XGBoost (eXtreme Gradient Boosting), an advanced implementation of gradient boosted decision trees designed for speed and performance. Our approach:

1. Starts with Linear Regression as a baseline model
2. Enhances predictions using XGBoost's tree-based ensemble learning
3. Incorporates feature engineering to capture market patterns
4. Optimizes hyperparameters for maximum prediction accuracy

## Data Sources

The project uses historical price data for:
- Gold (XAUUSD)
- Bitcoin (BTC/USD)

## Project Structure

- `/app` - Core application logic and models
- `/data` - Historical price datasets
- `/train` - Model training scripts
- `/templates` - Web interface templates
- `/static` - CSS, JavaScript, and assets
