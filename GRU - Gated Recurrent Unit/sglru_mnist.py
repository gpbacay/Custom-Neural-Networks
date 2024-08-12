import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, RNN, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Custom GRU Cell with Neurogenic and Spiking Dynamics
class NeurogenicSpikingGRUCell(tf.keras.layers.Layer):
    def __init__(self, units, spike_threshold=0.5, leak_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.spike_threshold = spike_threshold
        self.leak_rate = leak_rate

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units * 3), initializer='glorot_uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 3), initializer='orthogonal', name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self.units * 3,), initializer='zeros', name='bias')
        super().build(input_shape)

    def call(self, inputs, state):
        h_prev = state[0]

        # Compute GRU gate activations
        z, r, h = tf.split(tf.matmul(inputs, self.kernel) + tf.matmul(h_prev, self.recurrent_kernel) + self.bias, 3, axis=-1)
        z = tf.sigmoid(z)
        r = tf.sigmoid(r)
        h = tf.tanh(h)

        # Compute new state
        new_state = (1 - z) * h_prev + z * h

        # Spiking dynamics: generate spikes based on threshold
        spikes = tf.cast(tf.greater(new_state, self.spike_threshold), dtype=tf.float32)
        new_state = tf.where(spikes > 0, new_state - self.spike_threshold, new_state)

        # Apply leak rate
        new_state = (1 - self.leak_rate) * h_prev + self.leak_rate * new_state

        return new_state, [new_state]

    @property
    def state_size(self):
        return self.units

# Function to create the Neurogenic Spiking GRU model
def create_neurogenic_spiking_gru_model(input_dim, output_dim, num_units=128):
    # Define the input layer
    inputs = Input(shape=(input_dim,))

    # Add a Reshape layer to match the expected GRU input shape
    reshaped_inputs = Reshape((1, input_dim))(inputs)  # Adding sequence dimension

    # Define the custom GRU layer with neurogenic and spiking dynamics
    gru_layer = RNN(NeurogenicSpikingGRUCell(num_units), return_sequences=False)(reshaped_inputs)

    # Flatten the output (GRU layer outputs a 2D tensor)
    x = Flatten()(gru_layer)

    # Add dense layers for classification
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout for regularization
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout for regularization

    # Output layer with softmax activation for classification
    outputs = Dense(output_dim, activation='softmax')(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model

# Function to preprocess MNIST data
def preprocess_data(x):
    # Normalize pixel values to [0, 1]
    x = x.astype(np.float32) / 255.0
    # Flatten the 28x28 images to vectors of size 784
    return x.reshape(-1, 28 * 28)

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

x_train = preprocess_data(x_train)
x_val = preprocess_data(x_val)
x_test = preprocess_data(x_test)

# Convert class labels to one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)

# Set hyperparameters
input_dim = 28 * 28
output_dim = 10  # Number of output classes
num_epochs = 10  # Number of training epochs
batch_size = 64  # Batch size for training

# Create the Neurogenic Spiking GRU model
model = create_neurogenic_spiking_gru_model(input_dim, output_dim)

# Define callbacks for early stopping and learning rate reduction
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
]

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")


# Neurogenic Spiking Gated Liquid Recurrent Unit (SGLRU)
# python sglru_mnist.py
# Test Accuracy: 0.9793