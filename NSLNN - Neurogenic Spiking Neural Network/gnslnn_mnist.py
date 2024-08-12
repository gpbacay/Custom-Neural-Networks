import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Custom Layer: Gated Neurogenic Spiking Liquid Neural Network Step
class GatedNSLNNStep(tf.keras.layers.Layer):
    def __init__(self, reservoir_weights, input_weights, gate_weights, leak_rate, spike_threshold, max_reservoir_dim, **kwargs):
        super().__init__(**kwargs)
        # Initialize weights as non-trainable variables
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)
        self.gate_weights = tf.Variable(gate_weights, dtype=tf.float32, trainable=False)
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_reservoir_dim = max_reservoir_dim

    @property
    def state_size(self):
        return (self.max_reservoir_dim,)

    def call(self, inputs, states):
        prev_state = states[0][:, :self.reservoir_weights.shape[0]]
        
        # Compute input, reservoir, and gate parts of the state update
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights, transpose_b=True)
        gate_part = tf.matmul(inputs, self.gate_weights, transpose_b=True)

        # Split gate activations into input, forget, and output gates
        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)

        # Update state with gating and reservoir dynamics
        state = (1 - self.leak_rate) * (f_gate * prev_state) + self.leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state

        # Apply threshold to produce discrete spikes
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        # Ensure the state size matches the maximum reservoir dimension
        active_size = tf.shape(state)[-1]
        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_reservoir_dim - active_size])], axis=1)
        
        return padded_state, [padded_state]

# Function to initialize reservoir, input, and gate weights
def initialize_reservoir(input_dim, reservoir_dim, spectral_radius):
    # Initialize reservoir weights with random values
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    
    # Initialize input weights with small random values
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    
    # Initialize gate weights with small random values
    gate_weights = np.random.randn(3 * reservoir_dim, input_dim) * 0.1
    
    return reservoir_weights, input_weights, gate_weights

# Function to create the Gated Neurogenic Spiking Liquid Neural Network (GNSLNN) model
def create_gnslnn_model(input_dim, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim):
    # Define the input layer
    inputs = Input(shape=(input_dim,))
    
    # Initialize Spiking LNN weights
    reservoir_weights, input_weights, gate_weights = initialize_reservoir(input_dim, reservoir_dim, spectral_radius)

    # Define the Spiking LNN layer with custom dynamics and gating
    lnn_layer = tf.keras.layers.RNN(
        GatedNSLNNStep(reservoir_weights, input_weights, gate_weights, leak_rate, spike_threshold, max_reservoir_dim),
        return_sequences=True
    )

    # Define a function to apply the Spiking LNN layer and flatten the output
    def apply_gated_nslnn(x):
        lnn_output = lnn_layer(tf.expand_dims(x, axis=1))
        return Flatten()(lnn_output)

    # Apply the Spiking LNN layer and flatten the output
    lnn_output = Lambda(apply_gated_nslnn)(inputs)

    # Add dense layers for classification
    x = Dense(128, activation='relu')(lnn_output)
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
reservoir_dim = 500  # Dimension of the reservoir
max_reservoir_dim = 500  # Maximum dimension of the reservoir
spectral_radius = 1.5  # Spectral radius for reservoir scaling
leak_rate = 0.3  # Leak rate for state update
spike_threshold = 0.5  # Threshold for spike generation
output_dim = 10  # Number of output classes
num_epochs = 10  # Number of training epochs
batch_size = 64  # Batch size for training

# Create the Gated Neurogenic Spiking Liquid Neural Network (GNSLNN) model
model = create_gnslnn_model(input_dim, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim)

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



# Gated Neurogenic Spiking Liquid Neural Network (GNSLNN)
# A counterpart of Spiking Gated Liquid Recurrent Unit (SGLRU)
# python gnslnn_mnist.py
# Test Accuracy: 0.9688 (fast)