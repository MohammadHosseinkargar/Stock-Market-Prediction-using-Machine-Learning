import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Fetch historical stock data
def fetch_data(ticker):
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    return data

# Train model and predict prices
def predict_prices(data):
    data['Date'] = data.index
    data['Date'] = pd.to_numeric(data['Date'])
    X = data[['Date']]
    y = data['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, y, label='Actual Prices')
    plt.plot(X_test.index, predictions, label='Predicted Prices')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    ticker = "AAPL"
    data = fetch_data(ticker)
    predict_prices(data)
