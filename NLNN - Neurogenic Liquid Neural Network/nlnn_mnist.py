import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Custom LNN Layer with Neurogenic behavior
class LNNStep(tf.keras.layers.Layer):
    def __init__(self, reservoir_weights, input_weights, leak_rate, max_reservoir_dim, **kwargs):
        super().__init__(**kwargs)
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32)
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32)
        self.leak_rate = leak_rate
        self.max_reservoir_dim = max_reservoir_dim

    @property
    def state_size(self):
        return (self.max_reservoir_dim,)

    def call(self, inputs, states):
        prev_state = states[0][:, :self.reservoir_weights.shape[0]]
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights, transpose_b=True)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_part + reservoir_part)
        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_reservoir_dim - tf.shape(state)[-1]])], axis=1)
        return padded_state, [padded_state]

# Initialize LNN Reservoir
def initialize_lnn_reservoir(input_dim, reservoir_dim, spectral_radius, max_reservoir_dim):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    spectral_norm = np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    reservoir_weights *= spectral_radius / spectral_norm
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights

# Neurogenic Liquid Neural Network (NLNN) Model
class NLNNModel(tf.keras.Model):
    def __init__(self, input_dim, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim, output_dim):
        super().__init__()
        self.reservoir_weights, self.input_weights = initialize_lnn_reservoir(
            input_dim, reservoir_dim, spectral_radius, max_reservoir_dim
        )
        self.lnn_layer = tf.keras.layers.RNN(
            LNNStep(self.reservoir_weights, self.input_weights, leak_rate, max_reservoir_dim),
            return_sequences=True
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=1)
        x = self.lnn_layer(x)
        x = self.flatten(x)
        return self.dense(x)

# Load and preprocess MNIST dataset
def preprocess_data(x):
    x = x.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return x.reshape(-1, 28 * 28)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

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

# Create Neurogenic Liquid Neural Network (NLNN) model
model = NLNNModel(input_dim, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim, output_dim)

# Compile and train the model
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=num_epochs, batch_size=64, validation_data=(x_val, y_val), callbacks=callbacks)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")


# Neurogenic Liquid Nueral Network (NLNN)
# python nlnn_mnist.py
# Test Accuracy: 0.8767