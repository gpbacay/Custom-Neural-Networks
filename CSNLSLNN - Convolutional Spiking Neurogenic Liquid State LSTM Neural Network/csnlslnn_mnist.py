import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

class SpikingLNNStep(tf.keras.layers.Layer):
    """
    A custom Keras layer implementing a Spiking Liquid Neural Network (LNN) step.
    This layer simulates the behavior of spiking neurons in a reservoir computing framework.
    """
    def __init__(self, reservoir_weights, input_weights, leak_rate, max_reservoir_dim, spike_threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        # Reservoir weights (fixed and not trainable)
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        # Input weights (fixed and not trainable)
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)
        # Leak rate for updating the reservoir state
        self.leak_rate = leak_rate
        # Maximum dimension of the reservoir
        self.max_reservoir_dim = max_reservoir_dim
        # Threshold for spiking
        self.spike_threshold = spike_threshold

    @property
    def state_size(self):
        # Size of the state to be returned
        return (self.max_reservoir_dim,)

    def call(self, inputs, states):
        # Extract previous state
        prev_state = states[0][:, :self.reservoir_weights.shape[0]]
        # Compute contributions from inputs and previous state
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights, transpose_b=True)
        # Update reservoir state with leakage
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_part + reservoir_part)
        # Determine spikes and adjust state accordingly
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)
        # Pad the state to match max_reservoir_dim
        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_reservoir_dim - tf.shape(state)[-1]])], axis=1)
        return padded_state, [padded_state]

def initialize_reservoir(input_dim, reservoir_dim, spectral_radius):
    """
    Initialize reservoir weights and input weights for the Spiking LNN layer.
    The reservoir weights are scaled according to the spectral radius.
    """
    # Initialize reservoir weights with random values
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    # Scale weights to achieve desired spectral radius
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    # Initialize input weights with random values
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights

def create_csnlslnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim, output_dim):
    """
    Create a Convolutional Spiking Neurogenic Liquid State LSTM Neural Network (CSNLSLNN) model.
    This model combines convolutional layers, a spiking LNN layer, and LSTM layers for classification.
    """
    inputs = Input(shape=input_shape)
    
    # Convolutional layers for feature extraction
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    
    # Initialize reservoir and input weights
    reservoir_weights, input_weights = initialize_reservoir(128 * 3 * 3, reservoir_dim, spectral_radius)
    
    # Create and apply the spiking LNN layer
    lnn_layer = tf.keras.layers.RNN(SpikingLNNStep(reservoir_weights, input_weights, leak_rate, max_reservoir_dim), return_sequences=True)
    def apply_spiking_lnn(x):
        # Apply the spiking LNN layer and flatten the output
        lnn_output = lnn_layer(tf.expand_dims(x, axis=1))
        return Flatten()(lnn_output)
    lnn_output = Lambda(apply_spiking_lnn)(x)
    lnn_output_reshaped = tf.keras.layers.Reshape((1, -1))(lnn_output)

    # Add LSTM layers for sequential processing
    x = LSTM(128, return_sequences=True, dropout=0.3)(lnn_output_reshaped)
    x = LSTM(64, dropout=0.3)(x)
    
    # Final classification layer
    outputs = Dense(output_dim, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Normalize and reshape data
x_train = x_train.astype(np.float32) / 255.0
x_val = x_val.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical format
y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)

# Model parameters
input_shape = (28, 28, 1)
reservoir_dim = 100
max_reservoir_dim = 200
spectral_radius = 1.5
leak_rate = 0.3
output_dim = 10
num_epochs = 10
batch_size = 64

# Create and train model
model = create_csnlslnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim, output_dim)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
]

# Compile and fit the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Convolutional Spiking Neurogenic Liquid State LSTM Neural Network (CSNLSLNN)
# python csnlslnn_mnist.py
# Test Accuracy: 0.9932
