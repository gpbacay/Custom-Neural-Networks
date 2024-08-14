import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Layer
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SpikingElasticLNNStep(tf.keras.layers.Layer):
    def __init__(self, initial_reservoir_size, input_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, **kwargs):
        super().__init__(**kwargs)
        self.initial_reservoir_size = initial_reservoir_size
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_reservoir_dim = max_reservoir_dim
        
        self.reservoir_weights = None
        self.input_weights = None
        self.initialize_weights()

    def initialize_weights(self):
        reservoir_weights = np.random.randn(self.initial_reservoir_size, self.initial_reservoir_size)
        reservoir_weights *= self.spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        
        input_weights = np.random.randn(self.initial_reservoir_size, self.input_dim) * 0.1
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)

    @property
    def state_size(self):
        return (self.max_reservoir_dim,)

    def call(self, inputs, states):
        prev_state = states[0][:, :tf.shape(self.reservoir_weights)[0]]
        
        input_contribution = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_contribution = tf.matmul(prev_state, self.reservoir_weights)

        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_contribution + reservoir_contribution)

        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        active_size = tf.shape(state)[-1]
        padded_state = tf.pad(state, [[0, 0], [0, self.max_reservoir_dim - active_size]])
        
        return padded_state, [padded_state]

    def add_neurons(self, new_neurons):
        current_size = tf.shape(self.reservoir_weights)[0]
        new_size = current_size + new_neurons
        if new_size > self.max_reservoir_dim:
            raise ValueError(f"Cannot add {new_neurons} neurons. Max reservoir size is {self.max_reservoir_dim}")

        # Create new weights for added neurons
        new_reservoir_weights = tf.random.normal((new_neurons, new_size))
        
        # Compute spectral radius for the entire new weight matrix
        full_new_weights = tf.concat([
            tf.concat([self.reservoir_weights, tf.zeros((current_size, new_neurons))], axis=1),
            new_reservoir_weights
        ], axis=0)
        
        # Ensure the matrix is square before computing eigenvalues
        spectral_radius = tf.math.real(tf.reduce_max(tf.abs(tf.linalg.eigvals(full_new_weights))))
        scaling_factor = self.spectral_radius / spectral_radius
        
        new_reservoir_weights *= scaling_factor
        new_input_weights = tf.random.normal((new_neurons, self.input_dim)) * 0.1

        # Combine old and new weights
        updated_reservoir_weights = tf.concat([self.reservoir_weights, new_reservoir_weights[:, :current_size]], axis=0)
        updated_reservoir_weights = tf.concat([updated_reservoir_weights, 
                                               tf.concat([tf.transpose(new_reservoir_weights[:, :current_size]), 
                                                          new_reservoir_weights[:, current_size:]], axis=0)], axis=1)
        
        updated_input_weights = tf.concat([self.input_weights, new_input_weights], axis=0)

        # Update weights
        self.reservoir_weights = tf.Variable(updated_reservoir_weights, dtype=tf.float32, trainable=False)
        self.input_weights = tf.Variable(updated_input_weights, dtype=tf.float32, trainable=False)

    def prune_connections(self, threshold):
        mask = tf.abs(self.reservoir_weights) > threshold
        self.reservoir_weights.assign(tf.where(mask, self.reservoir_weights, tf.zeros_like(self.reservoir_weights)))

class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

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
        # Add neurons
        if epoch < self.total_epochs - 1:
            print(f" - Adding {self.neurons_per_addition} neurons at epoch {epoch + 1}")
            self.selnn_step_layer.add_neurons(self.neurons_per_addition)
        
        # Prune connections
        if (epoch + 1) % 5 == 0:  # Prune every 5 epochs (or adjust as needed)
            print(f" - Pruning connections at epoch {epoch + 1}")
            self.selnn_step_layer.prune_connections(self.pruning_threshold)

def create_selnn_model(input_dim, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    
    expanded_inputs = ExpandDimsLayer(axis=1)(inputs)

    selnn_step_layer = SpikingElasticLNNStep(initial_reservoir_size, input_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim)
    rnn_layer = tf.keras.layers.RNN(selnn_step_layer, return_sequences=False)

    selnn_output = rnn_layer(expanded_inputs)
    selnn_output = Flatten()(selnn_output)

    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(selnn_output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)

    outputs = Dense(output_dim, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model, selnn_step_layer

def preprocess_data(x):
    return x.astype(np.float32) / 255.0

def main():
    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    x_train = preprocess_data(x_train).reshape(-1, 28 * 28)
    x_val = preprocess_data(x_val).reshape(-1, 28 * 28)
    x_test = preprocess_data(x_test).reshape(-1, 28 * 28)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)
    y_test = tf.keras.utils.to_categorical(y_test)

    # Set hyperparameters
    input_dim = 28 * 28
    initial_reservoir_size = 500
    final_reservoir_size = 1000
    max_reservoir_dim = 1000
    spectral_radius = 0.9
    leak_rate = 0.1
    spike_threshold = 0.5
    output_dim = 10
    num_epochs = 10
    batch_size = 64
    pruning_threshold = 0.01  # Set appropriate pruning threshold

    # Create the Spiking Elastic Liquid Neural Network (SELNN) model
    model, selnn_step_layer = create_selnn_model(input_dim, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim)

    # Define callbacks
    gradual_add_neurons_and_prune_callback = GradualAddNeuronsAndPruneCallback(
        selnn_step_layer=selnn_step_layer,
        total_epochs=num_epochs,
        initial_neurons=initial_reservoir_size,
        final_neurons=final_reservoir_size,
        pruning_threshold=pruning_threshold
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
        gradual_add_neurons_and_prune_callback
    ]

    # Compile and train the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, 
                        validation_data=(x_val, y_val), callbacks=callbacks, verbose=1)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Display the model summary
    model.summary()

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()




# Spiking Elastic Liquid Nueral Network (SELNN)
# python selnn_mnist.py
# Test Accuracy: 0.9015
# Needs improvement, usage of adding and prunning connections while training
# Key for self-aware neural networks (elastic neural network)