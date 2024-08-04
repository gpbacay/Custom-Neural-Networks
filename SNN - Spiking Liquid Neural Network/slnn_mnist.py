import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Custom Spiking LNN Layer with Spiking Dynamics
class SpikingLNNStep(tf.keras.layers.Layer):
    def __init__(self, reservoir_weights, input_weights, leak_rate, max_reservoir_dim, spike_threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)
        self.leak_rate = leak_rate
        self.max_reservoir_dim = max_reservoir_dim
        self.spike_threshold = spike_threshold

    @property
    def state_size(self):
        return (self.max_reservoir_dim,)

    def call(self, inputs, states):
        prev_state = states[0][:, :self.reservoir_weights.shape[0]]
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights, transpose_b=True)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_part + reservoir_part)

        # Spiking dynamics: Apply a threshold to produce discrete spikes
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        # Ensure the state size matches the max reservoir size
        active_size = tf.shape(state)[-1]
        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_reservoir_dim - active_size])], axis=1)
        return padded_state, [padded_state]

def initialize_spiking_lnn_reservoir(input_dim, reservoir_dim, spectral_radius, max_reservoir_dim):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights

def create_spiking_nlnn_model(input_dim, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim, output_dim):
    inputs = Input(shape=(input_dim,))

    # Initialize Spiking LNN weights
    reservoir_weights, input_weights = initialize_spiking_lnn_reservoir(input_dim, reservoir_dim, spectral_radius, max_reservoir_dim)

    # Spiking LNN Layer
    lnn_layer = tf.keras.layers.RNN(
        SpikingLNNStep(reservoir_weights, input_weights, leak_rate, max_reservoir_dim),
        return_sequences=True
    )

    def apply_spiking_lnn(x):
        lnn_output = lnn_layer(tf.expand_dims(x, axis=1))
        return Flatten()(lnn_output)

    lnn_output = Lambda(apply_spiking_lnn)(inputs)

    # Additional layers for improved performance
    x = Dense(128, activation='relu')(lnn_output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(output_dim, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

def preprocess_data(x):
    x = x.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return x.reshape(-1, 28 * 28)

x_train = preprocess_data(x_train)
x_val = preprocess_data(x_val)
x_test = preprocess_data(x_test)

y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)

# Set hyperparameters
input_dim = 28 * 28
reservoir_dim = 100
max_reservoir_dim = 1000
spectral_radius = 1.5
leak_rate = 0.3
output_dim = 10
num_epochs = 10
batch_size = 64

# Create Spiking Liquid Neural Network (SNLNN) model
model = create_spiking_nlnn_model(input_dim, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim, output_dim)

# Compile and train the model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")


# Spiking Liquid Nueral Network (SLNN)
# python slnn_mnist.py
# Test Accuracy: 0.9463