import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, RNN
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Custom Spiking Gated LNN Layer
class SpikingLNNStep(tf.keras.layers.Layer):
    def __init__(self, reservoir_weights, input_weights, gate_weights, leak_rate, spike_threshold, **kwargs):
        super().__init__(**kwargs)
        self.reservoir_weights = tf.constant(reservoir_weights, dtype=tf.float32)
        self.input_weights = tf.constant(input_weights, dtype=tf.float32)
        self.gate_weights = tf.constant(gate_weights, dtype=tf.float32)
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.reservoir_dim = reservoir_weights.shape[0]

    @property
    def state_size(self):
        return (self.reservoir_dim,)

    def call(self, inputs, states):
        prev_state = states[0]
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights, transpose_b=True)
        gate_part = tf.matmul(inputs, self.gate_weights, transpose_b=True)

        # Gate activations
        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)

        # State calculation with gating and spiking dynamics
        state = (1 - self.leak_rate) * (f_gate * prev_state) + self.leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        return state, [state]

# Initialize LNN Reservoir with gate weights
def initialize_lnn_reservoir(input_dim, reservoir_dim, spectral_radius):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    gate_weights = np.random.randn(3 * reservoir_dim, input_dim) * 0.1  # Initialize gate weights
    return reservoir_weights, input_weights, gate_weights

# Create Gated LNN-LSTM Model
def create_spiking_gated_lnn_lstm_model(input_shape, reservoir_weights, input_weights, gate_weights, leak_rate, spike_threshold, lstm_units, output_dim):
    inputs = Input(shape=input_shape)
    lnn_layer = RNN(SpikingLNNStep(reservoir_weights, input_weights, gate_weights, leak_rate, spike_threshold), return_sequences=True)
    lnn_output = lnn_layer(inputs)
    lstm_output = LSTM(lstm_units)(lnn_output)
    outputs = Dense(output_dim, activation='softmax')(lstm_output)
    return tf.keras.Model(inputs, outputs)

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Normalize the data
def normalize_data(x):
    return StandardScaler().fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)

x_train = normalize_data(x_train)
x_val = normalize_data(x_val)
x_test = normalize_data(x_test)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)

# Set LNN and LSTM hyperparameters
input_dim = 28
reservoir_dim = 1000
spectral_radius = 1.5
leak_rate = 0.3
spike_threshold = 0.5
lstm_units = 50
output_dim = 10
num_epochs = 10

# Initialize LNN weights
reservoir_weights, input_weights, gate_weights = initialize_lnn_reservoir(input_dim, reservoir_dim, spectral_radius)

# Create Spiking Gated LNN-LSTM model
model = create_spiking_gated_lnn_lstm_model((28, 28), reservoir_weights, input_weights, gate_weights, leak_rate, spike_threshold, lstm_units, output_dim)

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=num_epochs, batch_size=64, validation_data=(x_val, y_val),
          callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")



# Spiking Gated Liquid Neural Network (SGLNN)
# python sglnn_mnist.py
# Test Accuracy: 0.9818 (but slow)
