import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Input

# Custom LNN Layer
class LNNStep(tf.keras.layers.Layer):
    def __init__(self, reservoir_weights, input_weights, leak_rate, **kwargs):
        super(LNNStep, self).__init__(**kwargs)
        self.reservoir_weights = tf.constant(reservoir_weights, dtype=tf.float32)
        self.input_weights = tf.constant(input_weights, dtype=tf.float32)
        self.leak_rate = leak_rate
        self.reservoir_dim = reservoir_weights.shape[0]

    @property
    def state_size(self):
        return (self.reservoir_dim,)

    def call(self, inputs, states):
        prev_state = states[0]
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights, transpose_b=True)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_part + reservoir_part)
        return state, [state]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "reservoir_weights": self.reservoir_weights.numpy().tolist(),
            "input_weights": self.input_weights.numpy().tolist(),
            "leak_rate": self.leak_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        reservoir_weights = np.array(config['reservoir_weights'])
        input_weights = np.array(config['input_weights'])
        leak_rate = config['leak_rate']
        return cls(reservoir_weights, input_weights, leak_rate)

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
