import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class ElasticLNNStep(layers.Layer):
    def __init__(self, reservoir_weights, input_weights, leak_rate, **kwargs):
        super().__init__(**kwargs)
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)
        self.leak_rate = leak_rate
        self._state_size = reservoir_weights.shape[0]

    @property
    def state_size(self):
        return self._state_size

    def call(self, inputs, states):
        prev_state = states[0][:, :tf.shape(self.reservoir_weights)[0]]
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_part + reservoir_part)
        return state, [state]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros((batch_size, self._state_size), dtype=tf.float32)]

def initialize_lnn_reservoir(input_dim, reservoir_dim, spectral_radius):
    """Initializes the reservoir and input weights for the Elastic LNN."""
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights

def download_data(ticker, start_date, end_date):
    """Downloads historical data for a given ticker and time range."""
    try:
        data = yf.download(tickers=ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data found for the specified ticker.")
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

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
    """Creates sequences for LNN input."""
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

def build_hybrid_model(input_shape, reservoir_dim, spectral_radius, leak_rate, lstm_units):
    """Builds and compiles the Hybrid ELNN-LSTM model."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)

    input_dim = x.shape[-1]
    reservoir_weights, input_weights = initialize_lnn_reservoir(input_dim, reservoir_dim, spectral_radius)

    lnn_layer = ElasticLNNStep(
        reservoir_weights=reservoir_weights,
        input_weights=input_weights,
        leak_rate=leak_rate
    )

    rnn_layer = layers.RNN(lnn_layer, return_sequences=True)
    lnn_output = rnn_layer(layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x))

    # Add LSTM layer
    lstm_output = layers.LSTM(lstm_units, return_sequences=False)(lnn_output)

    outputs = layers.Dense(1, activation='linear')(lstm_output)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

def plot_predictions(y_test, y_pred):
    """Plots the true vs. predicted values."""
    plt.figure(figsize=(16, 8))
    plt.plot(y_test, color='black', label='Test', alpha=0.8)
    plt.plot(y_pred, color='green', label='Pred', alpha=0.8)
    plt.title('Price Movement Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_training_history(history):
    """Plots the training history."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE During Training')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

def main():
    # Parameters
    config = {
        'ticker': '^RUI',
        'start_date': '2012-03-11',
        'end_date': '2022-07-10',
        'backcandles': 30,
        'reservoir_dim': 1000,
        'spectral_radius': 1.5,
        'leak_rate': 0.5,
        'lstm_units': 150,
        'batch_size': 15,
        'epochs': 50,
        'split_ratio': 0.8
    }

    # Step 1: Data Preparation
    data = download_data(config['ticker'], config['start_date'], config['end_date'])
    if data is None:
        return  # Exit if data downloading failed

    data = add_indicators(data)
    data = create_target(data)
    scaled_data, scaler = preprocess_data(data)

    # Step 2: Create Sequences for LNN
    X, y = create_sequences(scaled_data, config['backcandles'])

    # Step 3: Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = split_data(X, y, config['split_ratio'])

    # Step 4: Build and Train Hybrid ELNN-LSTM Model
    model = build_hybrid_model((config['backcandles'], X_train.shape[2]), 
                                config['reservoir_dim'], 
                                config['spectral_radius'], 
                                config['leak_rate'], 
                                config['lstm_units'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

    history = model.fit(
        X_train, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr]
    )

    # Step 5: Evaluate Model on Test Set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # Step 6: Make Predictions and Plot Results
    y_pred = model.predict(X_test)
    plot_predictions(y_test, y_pred)

    # Step 7: Plot Training History
    plot_training_history(history)

if __name__ == '__main__':
    main()


# Requirements:
# pip install numpy pandas matplotlib yfinance pandas-ta scikit-learn tensorflow

# python elnn_lstm_trading.py
# Test Loss (MSE): 0.0149
# Test MAE: 0.1088