import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, MaxPooling2D, Layer, Reshape, MultiHeadAttention, BatchNormalization
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Custom Keras Layer for Spiking Elastic Liquid Neural Network (SELNN) Step
class SpikingElasticLNNStep(tf.keras.layers.Layer):
    def __init__(self, initial_reservoir_size, input_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, **kwargs):
        super().__init__(**kwargs)
        self.initial_reservoir_size = initial_reservoir_size
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_reservoir_dim = max_reservoir_dim

        # Set the state size to be the size of the reservoir
        self.state_size = self.initial_reservoir_size

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize reservoir and input weights
        reservoir_weights = np.random.randn(self.initial_reservoir_size, self.initial_reservoir_size)
        reservoir_weights *= self.spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        
        input_weights = np.random.randn(self.initial_reservoir_size, self.input_dim) * 0.1
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)

    def call(self, inputs, states):
        # Compute state and spikes
        prev_state = states[0]
        input_contribution = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_contribution = tf.matmul(prev_state, self.reservoir_weights)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_contribution + reservoir_contribution)
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)
        return state, [state]

    def add_neurons(self, new_neurons):
        # Add new neurons to the reservoir
        current_size = self.reservoir_weights.shape[0]
        new_size = current_size + new_neurons
        if new_size > self.max_reservoir_dim:
            raise ValueError(f"Cannot add {new_neurons} neurons. Max reservoir size is {self.max_reservoir_dim}")

        # Create new weights and adjust spectral radius
        new_reservoir_weights = np.random.randn(new_neurons, new_size)
        full_new_weights = np.block([
            [self.reservoir_weights.numpy(), np.zeros((current_size, new_neurons))],
            [new_reservoir_weights[:, :current_size], new_reservoir_weights[:, current_size:]]
        ])
        spectral_radius = np.max(np.abs(np.linalg.eigvals(full_new_weights)))
        scaling_factor = self.spectral_radius / spectral_radius
        full_new_weights *= scaling_factor
        
        # Update reservoir weights
        self.reservoir_weights = tf.Variable(full_new_weights, dtype=tf.float32, trainable=False)
        new_input_weights = np.random.randn(new_neurons, self.input_dim) * 0.1
        self.input_weights = tf.Variable(np.vstack([self.input_weights.numpy(), new_input_weights]), dtype=tf.float32, trainable=False)

    def prune_connections(self, threshold):
        # Prune connections based on the threshold
        mask = tf.abs(self.reservoir_weights) > threshold
        pruned_weights = tf.where(mask, self.reservoir_weights, tf.zeros_like(self.reservoir_weights))
        self.reservoir_weights.assign(pruned_weights)

# Callback for Gradual Addition of Neurons and Pruning Connections
class GradualAddNeuronsAndPruneCallback(Callback):
    def __init__(self, selnn_step_layer, total_epochs, initial_neurons, final_neurons, pruning_threshold):
        super().__init__()
        self.selnn_step_layer = selnn_step_layer
        self.total_epochs = total_epochs
        self.initial_neurons = initial_neurons
        self.final_neurons = final_neurons
        self.neurons_per_addition = (final_neurons - initial_neurons) // (total_epochs - 1)
        self.pruning_threshold = pruning_threshold

    def on_epoch_end(self, epoch, logs=None):
        # Add neurons and prune connections at the end of each epoch
        if epoch < self.total_epochs - 1:
            print(f" - Adding {self.neurons_per_addition} neurons at epoch {epoch + 1}")
            self.selnn_step_layer.add_neurons(self.neurons_per_addition)
        
        if (epoch + 1) % 1 == 0:  # Prune every epoch (or adjust as needed)
            print(f" - Pruning connections at epoch {epoch + 1}")
            self.selnn_step_layer.prune_connections(self.pruning_threshold)

# Function to Create the SELNN Model with Attention
def create_selnn_with_attention_model(input_shape, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim):
    inputs = Input(shape=input_shape)
    
    # Convolutional layers
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    
    # Flatten the output
    x = Flatten()(x)
    
    # Reshape the data for RNN layer
    x = Reshape((1, x.shape[1]))(x)  # Reshape to (batch_size, sequence_length, features)
    
    # Apply self-attention
    attention = MultiHeadAttention(num_heads=4, key_dim=x.shape[-1])(x, x)
    
    # Initialize SELNN layer
    selnn_step_layer = SpikingElasticLNNStep(
        initial_reservoir_size=initial_reservoir_size,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim
    )
    
    rnn_layer = tf.keras.layers.RNN(selnn_step_layer, return_sequences=False)
    selnn_output = rnn_layer(attention)
    
    # Build the readout layers
    x = readout_layer(selnn_output, output_dim)
    
    model = tf.keras.Model(inputs, x)
    return model, selnn_step_layer

# Function to Build Readout Layers
def readout_layer(x, output_dim):
    """Build the readout layers for classification."""
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    
    outputs = Dense(output_dim, activation='softmax')(x)
    return outputs

# Function to Preprocess Data
def preprocess_data(x):
    """Preprocess the data."""
    return x.astype(np.float32) / 255.0

# Main Function to Run the Model
def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    x_train = preprocess_data(x_train).reshape(-1, 28, 28, 1)
    x_val = preprocess_data(x_val).reshape(-1, 28, 28, 1)
    x_test = preprocess_data(x_test).reshape(-1, 28, 28, 1)
    
    # Model parameters
    input_shape = (28, 28, 1)
    initial_reservoir_size = 100
    spectral_radius = 0.95
    leak_rate = 0.1
    spike_threshold = 0.5
    max_reservoir_dim = 1000
    output_dim = 10
    total_epochs = 10
    initial_neurons = 100
    final_neurons = 200
    pruning_threshold = 0.1
    
    # Create and compile the model
    model, selnn_step_layer = create_selnn_with_attention_model(
        input_shape, 
        initial_reservoir_size, 
        spectral_radius, 
        leak_rate, 
        spike_threshold, 
        max_reservoir_dim, 
        output_dim
    )
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Callback for early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Callback for reducing learning rate on plateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    
    # Custom callback for adding neurons and pruning connections
    gradual_add_prune = GradualAddNeuronsAndPruneCallback(selnn_step_layer, total_epochs, initial_neurons, final_neurons, pruning_threshold)
    
    # Train the model
    history = model.fit(
        x_train, y_train, 
        epochs=total_epochs, 
        batch_size=32, 
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr, gradual_add_prune],
        verbose=1
    )
    
    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()



# Convolutional Attention Spiking Elastic Liquid Nueral Network (CASE-LNN)
# python case_lnn_mnist.py
# Test Accuracy: (too slow)