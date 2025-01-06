import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ----------------------------- Load Data -----------------------------
file_path = "tesla.15.24.csv"
df = pd.read_csv(file_path)

# ----------------------------- Initial Data Inspection -----------------------------
print(df.head())
print(df.info())

# ----------------------------- Data Preprocessing -----------------------------
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

# ----------------------------- Prepare Data for Predictive Models -----------------------------
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)
X = df[['Volume', 'Close_t-1', 'Volume_t-1', 'MA_Close_5', 'MA_Volume_5']]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------- Linear Regression Model -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ----------------------------- Linear Regression Metrics -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print("Linear Regression Model Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# ----------------------------- Plot Linear Regression Results -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test.values, label='Actual Prices', color='blue', marker='o')
plt.plot(range(len(y_pred)), y_pred, label='Predicted Prices', color='red', linestyle='--', marker='x')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Actual vs Predicted (Linear Regression)')
plt.legend()
plt.show()

# ----------------------------- Random Forest Model -----------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)

# ----------------------------- Random Forest Metrics -----------------------------
mae = mean_absolute_error(y_test, y_rf_pred)
mse = mean_squared_error(y_test, y_rf_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_rf_pred)
print("Random Forest Model Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# ----------------------------- Accuracy -----------------------------
threshold = 0.05
percent_error = abs((y_test - y_rf_pred) / y_test)
accuracy = (percent_error <= threshold).mean() * 100
print(f"Accuracy within ±5% error: {accuracy:.2f}%")

# ----------------------------- Plotting Random Forest -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test.values, label='Actual Prices', color='blue', marker='o')
plt.plot(range(len(y_rf_pred)), y_rf_pred, label='Predicted Prices (Random Forest)', color='red', linestyle='--', marker='x')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Actual vs Predicted (Random Forest)')
plt.legend()
plt.show()