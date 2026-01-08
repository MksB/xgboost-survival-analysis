```python
# XGBoost Regression Program for Stock Market Forecasting
# This program uses historical stock data to predict future closing prices.
# It demonstrates a basic regression model using XGBoost for continuous value prediction.

# Installation instructions (run these in your terminal if needed):
# pip install yfinance xgboost pandas numpy scikit-learn matplotlib

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Download historical stock data
# Example: Apple Inc. (AAPL) from 2010 to 2023
stock_symbol = 'AAPL'
data = yf.download(stock_symbol, start='2010-01-01', end='2023-01-01',auto_adjust=True)

# Use 'Close' price as the target variable
data = data[['Close']]

# Step 2: Feature Engineering
# Create lagged features (previous days' prices) and moving averages
def add_features(df, lags=5, ma_windows=[5, 10, 20]):
    # Lagged closes
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    
    # Moving averages
    for window in ma_windows:
        df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
    
    # Target: Next day's close price
    df['target'] = df['Close'].shift(-1)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df

data = add_features(data)

# Step 3: Prepare features (X) and target (y)
X = data.drop(['Close', 'target'], axis=1)  # Features: lags and MAs
y = data['target']  # Target: next close

# Step 4: Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 5: Initialize and train XGBoost Regressor
model = XGBRegressor(
    objective='reg:squarederror',  # For regression
    n_estimators=100,              # Number of boosting rounds
    learning_rate=0.1,             # Step size
    max_depth=5,                   # Max tree depth
    random_state=42                # For reproducibility
)

model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.4f}")

# Step 8: Plot actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted', color='red')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Optional: Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)
```