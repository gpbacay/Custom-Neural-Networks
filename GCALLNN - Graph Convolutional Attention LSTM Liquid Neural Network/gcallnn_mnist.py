import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Input, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.dense = Dense(output_dim)
        self.attention_dense = Dense(1, use_bias=True)
        
    def call(self, inputs):
        x = self.dense(inputs)
        attention_scores = self.attention_dense(x)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        context = tf.reduce_sum(attention_weights * x, axis=1)
        return context

def initialize_lnn_reservoir(input_dim, reservoir_dim, spectral_radius):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights

def create_hybrid_model(input_shape, reservoir_dim, spectral_radius, leak_rate, lstm_units, output_dim):
    inputs = Input(shape=input_shape)

    # CNN Layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Flatten and reshape for graph conversion
    x = Flatten()(x)
    cnn_output_shape = x.shape[1]
    x = Reshape((cnn_output_shape, 1))(x)

    # Apply Graph Attention Layer
    x = GraphAttentionLayer(64)(x)
    x = Reshape((1, 64))(x)

    # LNN Layer
    reservoir_weights, input_weights = initialize_lnn_reservoir(64, reservoir_dim, spectral_radius)
    lnn_layer = tf.keras.layers.RNN(LNNStep(reservoir_weights, input_weights, leak_rate), return_sequences=True)
    x = lnn_layer(x)

    # LSTM Layer
    x = LSTM(lstm_units)(x)

    # Output Layer
    outputs = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

def normalize_data(x):
    num_samples, height, width = x.shape
    x = x.reshape(-1, width)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x.reshape(num_samples, height, width)

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

x_train = normalize_data(x_train)
x_val = normalize_data(x_val)
x_test = normalize_data(x_test)

x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Set hyperparameters
input_shape = (28, 28, 1)
reservoir_dim = 512
spectral_radius = 0.9
leak_rate = 0.2
lstm_units = 128
output_dim = 10

# Create and compile the model
hybrid_model = create_hybrid_model(input_shape, reservoir_dim, spectral_radius, leak_rate, lstm_units, output_dim)
hybrid_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = hybrid_model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))

# Evaluate the model
test_loss, test_accuracy = hybrid_model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Graph Convolutional Attention LSTM Liquid Neural Network (GCALLNN)
# python gcallnn_mnist.py
# Test Accuracy: 0.9918