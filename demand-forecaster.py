import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error

def train_sales_forecaster(dates, sales):
    df = pd.DataFrame({'date': pd.to_datetime(dates), 'sales': sales})
    df = df.sort_values('date').reset_index(drop=True)
    
    # Feature Engineering for Time Series
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['rolling_mean_7'] = df['sales'].shift(1).rolling(7).mean()
    df = df.dropna().reset_index(drop=True)
    
    features = ['day_of_week', 'month', 'lag_1', 'lag_7', 'rolling_mean_7']
    train_size = int(len(df) * 0.85)
    
    X_train, y_train = df.iloc[:train_size][features], df.iloc[:train_size]['sales']
    X_test, y_test = df.iloc[train_size:][features], df.iloc[train_size:]['sales']
    
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)
    print(f"Validation Mean Absolute Percentage Error: {mape:.2%}")
    return model

# Synthetic 1-year sales trajectory
rng = np.random.default_rng(42)
dates = pd.date_range(start="2025-01-01", periods=365, freq='D')
trend = np.linspace(100, 250, 365)
seasonality = 30 * np.sin(2 * np.pi * dates.dayofweek / 7)
noise = rng.normal(0, 10, 365)
sales_series = np.maximum(trend + seasonality + noise, 0)

model = train_sales_forecaster(dates, sales_series)