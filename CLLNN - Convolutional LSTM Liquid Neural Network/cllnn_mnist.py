import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Input, Flatten, Reshape
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

# Initialize LNN Reservoir
def initialize_lnn_reservoir(input_dim, reservoir_dim, spectral_radius):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights

# Convolutional LSTM Liquid Neural Network (CLLNN) Model
def create_cllnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, lstm_units, output_dim):
    inputs = Input(shape=input_shape)

    # CNN Layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Flatten and Reshape for LSTM
    x = Flatten()(x)
    cnn_output_shape = x.shape[1]
    
    # Ensure input_dim for LNN matches the flattened output shape
    input_dim = cnn_output_shape
    x = Reshape((1, cnn_output_shape))(x)  # Reshape for LSTM input

    # Initialize LNN weights
    reservoir_weights, input_weights = initialize_lnn_reservoir(input_dim, reservoir_dim, spectral_radius)

    # LNN Layer
    lnn_layer = tf.keras.layers.RNN(LNNStep(reservoir_weights, input_weights, leak_rate), return_sequences=True)
    lnn_output = lnn_layer(x)

    # LSTM Layer
    lstm_output = LSTM(lstm_units)(lnn_output)

    # Output Layer
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

# Reshape for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train)
y_val = keras.utils.to_categorical(y_val)
y_test = keras.utils.to_categorical(y_test)

# Set hyperparameters
input_shape = (28, 28, 1)
reservoir_dim = 1000
spectral_radius = 1.5
leak_rate = 0.3
lstm_units = 50
output_dim = 10
num_epochs = 10

# Create Convolutional LSTM Liquid Neural Network (CLLNN) model
model = create_cllnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, lstm_units, output_dim)

# Compile and train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=num_epochs, batch_size=64, validation_data=(x_val, y_val), callbacks=[early_stopping, reduce_lr])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Convolutional LSTM Liquid Nueral Network (CLLNN)
#_____________python cllnn_mnist.py
# Test Accuracy: 0.9928