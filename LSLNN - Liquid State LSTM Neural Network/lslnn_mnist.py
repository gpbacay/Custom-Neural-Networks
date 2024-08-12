import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Custom LNN Layer
class LNNStep(tf.keras.layers.Layer):
    def __init__(self, reservoir_weights, input_weights, leak_rate, **kwargs):
        super(LNNStep, self).__init__(**kwargs)
        self.reservoir_weights = tf.constant(reservoir_weights, dtype=tf.float32)
        self.input_weights = tf.constant(input_weights, dtype=tf.float32)
        self.leak_rate = leak_rate
        self.reservoir_dim = reservoir_weights.shape[0]  # Define the size of the state

    @property
    def state_size(self):
        return (self.reservoir_dim,)  # Return state size as a tuple

    def call(self, inputs, states):
        prev_state = states[0]
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights, transpose_b=True)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_part + reservoir_part)
        return state, [state]

# Initialize LNN Reservoir
def initialize_lnn_reservoir(input_dim, reservoir_dim, spectral_radius):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights

# LNN-LSTM Model
def create_lnn_lstm_model(input_shape, reservoir_weights, input_weights, leak_rate, lstm_units, output_dim):
    inputs = Input(shape=input_shape)
    lnn_layer = tf.keras.layers.RNN(LNNStep(reservoir_weights, input_weights, leak_rate), return_sequences=True)
    lnn_output = lnn_layer(inputs)
    lstm_output = LSTM(lstm_units)(lnn_output)
    outputs = Dense(output_dim, activation='softmax')(lstm_output)
    model = keras.Model(inputs, outputs)
    return model

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Normalize the data
def normalize_data(x):
    num_samples, height, width = x.shape
    x = x.reshape(-1, width)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x.reshape(num_samples, height, width)

x_train = normalize_data(x_train)
x_val = normalize_data(x_val)
x_test = normalize_data(x_test)

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train)
y_val = keras.utils.to_categorical(y_val)
y_test = keras.utils.to_categorical(y_test)

# Set LNN and LSTM hyperparameters
input_dim = 28
reservoir_dim = 1000
spectral_radius = 1.5
leak_rate = 0.3
lstm_units = 50
output_dim = 10
num_epochs = 10

# Initialize LNN weights
reservoir_weights, input_weights = initialize_lnn_reservoir(input_dim, reservoir_dim, spectral_radius)

# Create LNN-LSTM model
input_shape = (28, 28)
model = create_lnn_lstm_model(input_shape, reservoir_weights, input_weights, leak_rate, lstm_units, output_dim)

# Compile and train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=num_epochs, batch_size=64, validation_data=(x_val, y_val), callbacks=[early_stopping, reduce_lr])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Liquid State LSTM Neural Network (LSLNN)
# python lslnn_mnist.py
# Test Accuracy: 0.9732 (slow)
