import pandas as pd
import numpy as np

# ----------------------------- Load and Prepare Data -----------------------------
file_path = "tesla.15.24.csv"
df = pd.read_csv(file_path)

print(df.head())
print(df.info())

df['Date'] = pd.to_datetime(df['Date'])

df.dropna(inplace=True)
df.sort_values(by='Date', inplace=True)

# ----------------------------- Feature Engineering -----------------------------
df['Price_Movement'] = df['Close'] - df['Open']
df['Price_Movement_Percent'] = (df['Price_Movement'] / df['Open']) * 100
df['Daily_Range'] = df['High'] - df['Low']
df['Close_t-1'] = df['Close'].shift(1)
df['Volume_t-1'] = df['Volume'].shift(1)
df['MA_Close_5'] = df['Close'].rolling(window=5).mean()
df['MA_Volume_5'] = df['Volume'].rolling(window=5).mean()

df.dropna(inplace=True)
print(df.describe())

# ----------------------------- Data Visualization -----------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Closing Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Closing Price Over Time')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Volume'], label='Volume', color='green')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Trading Volume Over Time')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(df['Volume'], df['Price_Movement_Percent'], alpha=0.5, color='purple')
plt.xlabel('Volume')
plt.ylabel('Price Movement (%)')
plt.title('Volume vs. Price Movement (%)')
plt.show()

# ----------------------------- Linear Regression Model -----------------------------
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

X = df[['Volume', 'Close_t-1', 'Volume_t-1', 'MA_Close_5', 'MA_Volume_5']]
y = df['Target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Linear Regression Model Evaluation:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R^2 Score:", r2)

plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test.values, label='Actual Prices', color='blue', marker='o')
plt.plot(range(len(y_pred)), y_pred, label='Predicted Prices', color='red', linestyle='--', marker='x')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices (Linear Regression)')
plt.legend()
plt.show()

# ----------------------------- XGBoost Model -----------------------------
from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

y_xgb_pred = xgb_model.predict(X_test)

mae = mean_absolute_error(y_test, y_xgb_pred)
mse = mean_squared_error(y_test, y_xgb_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_xgb_pred)

print("XGBoost Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

threshold = 0.05
percent_error = abs((y_test - y_xgb_pred) / y_test)
accuracy = (percent_error <= threshold).mean() * 100
print(f"Accuracy within ±5% error: {accuracy:.2f}%")

plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test.values, label='Actual Prices', color='blue', marker='o')
plt.plot(range(len(y_xgb_pred)), y_xgb_pred, label='Predicted Prices (XGBoost)', color='green', linestyle='--', marker='x')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices (XGBoost)')
plt.legend()
plt.show()

# ----------------------------- Hyperparameter Tuning with GridSearch -----------------------------
from sklearn.model_selection import GridSearchCV

tesla_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}

grid_search = GridSearchCV(estimator=XGBRegressor(random_state=42), param_grid=tesla_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_xgb_model = grid_search.best_estimator_
y_best_xgb_pred = best_xgb_model.predict(X_test)

best_mae = mean_absolute_error(y_test, y_best_xgb_pred)
best_accuracy = (abs((y_test - y_best_xgb_pred) / y_test) <= threshold).mean() * 100

print(f"Best XGBoost Model MAE: {best_mae:.2f}")
print(f"Accuracy within ±5% error (tuned): {best_accuracy:.2f}%")