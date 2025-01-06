import pandas as pd
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ----------------------------- Load and Preprocess Data -----------------------------
file_path = "tesla.15.24.csv"
df = pd.read_csv(file_path)

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

# ----------------------------- Data Summary and Visualization -----------------------------
print(df.head())
print(df.info())
print(df.describe())

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Linear Regression Evaluation:")
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

# ----------------------------- LSTM Model for Time Series -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close', 'Volume']])

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 0])  # Predict 'Close' price
    return np.array(X), np.array(y)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

predicted_prices = model.predict(X_test)

predicted_prices = scaler.inverse_transform(
    np.concatenate([predicted_prices, np.zeros((predicted_prices.shape[0], 1))], axis=1))[:, 0]
actual_prices = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 1))], axis=1))[:, 0]

mae = mean_absolute_error(actual_prices, predicted_prices)
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
r2 = r2_score(actual_prices, predicted_prices)

print("LSTM Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label="Actual Prices", color="blue")
plt.plot(predicted_prices, label="Predicted Prices", color="red")
plt.xlabel("Days")
plt.ylabel("Price")
plt.title("Actual vs Predicted Prices (LSTM)")
plt.legend()
plt.show()

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(actual_prices, predicted_prices)
accuracy = 100 - mape

print(f"LSTM Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Accuracy within ±5% error: {accuracy:.2f}%")