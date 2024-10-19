import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Activation, Input
from keras.optimizers import Adam

def download_data(ticker, start_date, end_date):
    """Downloads historical data for a given ticker and time range."""
    data = yf.download(tickers=ticker, start=start_date, end=end_date)
    return data

def add_indicators(data):
    """Adds technical indicators to the data."""
    data['RSI'] = ta.rsi(data['Close'], length=15)
    data['EMAF'] = ta.ema(data['Close'], length=20)
    data['EMAM'] = ta.ema(data['Close'], length=100)
    data['EMAS'] = ta.ema(data['Close'], length=150)
    return data

def create_target(data):
    """Creates target columns for regression."""
    data['Target'] = data['Adj Close'] - data['Open']
    data['Target'] = data['Target'].shift(-1)
    data['TargetNextClose'] = data['Adj Close'].shift(-1)
    return data

def preprocess_data(data):
    """Preprocesses data by dropping unnecessary columns and scaling features."""
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_sequences(data, backcandles=30):
    """Creates sequences for LSTM input."""
    X = np.array([data[i - backcandles:i, :-2] for i in range(backcandles, len(data))])
    y = data[backcandles:, -1]  # Regression target: next day's adjusted close price
    y = np.reshape(y, (len(y), 1))
    return X, y

def split_data(X, y, split_ratio=0.8):
    """Splits data into training and testing sets."""
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

def build_lstm_model(input_shape):
    """Builds and compiles the LSTM model."""
    lstm_input = Input(shape=input_shape, name='lstm_input')
    x = LSTM(150, name='lstm_layer')(lstm_input)
    x = Dense(1, name='dense_layer')(x)
    output = Activation('linear', name='output')(x)
    model = Model(inputs=lstm_input, outputs=output)
    
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

def plot_predictions(y_test, y_pred):
    """Plots the true vs. predicted values."""
    plt.figure(figsize=(16, 8))
    plt.plot(y_test, color='black', label='Test', alpha=0.8)
    plt.plot(y_pred, color='green', label='Pred', alpha=0.8)
    plt.title('Price Movement Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    # Parameters
    ticker = '^RUI'
    start_date = '2012-03-11'
    end_date = '2022-07-10'
    backcandles = 30
    
    # Step 1: Data Preparation
    data = download_data(ticker, start_date, end_date)
    data = add_indicators(data)
    data = create_target(data)
    scaled_data, scaler = preprocess_data(data)
    
    # Step 2: Create Sequences for LSTM
    X, y = create_sequences(scaled_data, backcandles)
    
    # Step 3: Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4: Build and Train LSTM Model
    model = build_lstm_model((backcandles, X_train.shape[2]))
    history = model.fit(X_train, y_train, batch_size=15, epochs=30, validation_split=0.1, shuffle=True)
    
    # Step 5: Evaluate Model on Test Set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Step 6: Make Predictions and Plot Results
    y_pred = model.predict(X_test)
    plot_predictions(y_test, y_pred)

    # Step 7: Plot Training History
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot MAE
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE During Training')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

# Requirements:
# pip install numpy pandas matplotlib yfinance pandas-ta scikit-learn tensorflow

# python lstm_trading.py
# MSE: 0.0005
# MAE: 0.0161