import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Layer, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SynaptogenesisLayer(tf.keras.layers.Layer):
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
        # Initialize reservoir weights and scale them by the spectral radius
        reservoir_weights = np.random.randn(self.initial_reservoir_size, self.initial_reservoir_size)
        # Apply spectral normalization to ensure stability in RNN dynamics
        reservoir_weights *= self.spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        
        # Initialize input weights with small random values
        input_weights = np.random.randn(self.initial_reservoir_size, self.input_dim) * 0.1
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)

    @property
    def state_size(self):
        return (self.max_reservoir_dim,)

    def call(self, inputs, states):
        prev_state = states[0][:, :tf.shape(self.reservoir_weights)[0]]
        input_contribution = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_contribution = tf.matmul(prev_state, self.reservoir_weights)
        
        # Leaky integration and spiking mechanism
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_contribution + reservoir_contribution)
        
        # Spike-based threshold mechanism: Introduces non-linearity to mimic spiking behavior
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)  # Spike-reset mechanism
        
        # Pad the state to match the maximum reservoir dimension
        active_size = tf.shape(state)[-1]
        padded_state = tf.pad(state, [[0, 0], [0, self.max_reservoir_dim - active_size]])
        return padded_state, [padded_state]

    def add_synapses(self):
        current_size = tf.shape(self.reservoir_weights)[0]
        growth_rate = max(1, int(current_size * 0.1))  # Add 10% or at least 1 neuron
        new_neurons = min(growth_rate, self.max_reservoir_dim - current_size)
        if new_neurons <= 0:
            return

        new_size = current_size + new_neurons
        new_reservoir_weights = tf.random.normal((new_size, new_size))
        new_input_weights = tf.random.normal((new_neurons, self.input_dim)) * 0.1

        # Update reservoir weights and input weights with spectral radius scaling
        updated_reservoir_weights = tf.concat([
            tf.concat([self.reservoir_weights, tf.zeros((current_size, new_neurons))], axis=1),
            tf.concat([tf.zeros((new_neurons, current_size)), new_reservoir_weights[current_size:, current_size:]], axis=1)
        ], axis=0)
        
        # Apply spectral normalization to maintain stability in the expanded reservoir
        spectral_radius = tf.math.real(tf.reduce_max(tf.abs(tf.linalg.eigvals(updated_reservoir_weights))))
        scaling_factor = self.spectral_radius / spectral_radius
        updated_reservoir_weights *= scaling_factor
        
        updated_input_weights = tf.concat([self.input_weights, new_input_weights], axis=0)
        
        self.reservoir_weights.assign(updated_reservoir_weights)
        self.input_weights.assign(updated_input_weights)

    def prune_synapses(self):
        # Prune synapses with low weight magnitude (below the 10th percentile)
        threshold = np.percentile(np.abs(self.reservoir_weights.numpy()), 10)
        mask = tf.abs(self.reservoir_weights) > threshold
        self.reservoir_weights.assign(tf.where(mask, self.reservoir_weights, tf.zeros_like(self.reservoir_weights)))

class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

class SynaptogenesisCallback(Callback):
    def __init__(self, synaptogenesis_layer, performance_metric='val_classification_output_accuracy', target_metric=0.95,
                 add_synapses_threshold=0.01, prune_synapses_threshold=0.1, growth_phase_length=10, pruning_phase_length=5):
        super().__init__()
        self.synaptogenesis_layer = synaptogenesis_layer
        self.performance_metric = performance_metric
        self.target_metric = target_metric
        self.initial_add_synapses_threshold = add_synapses_threshold
        self.initial_prune_synapses_threshold = prune_synapses_threshold
        self.growth_phase_length = growth_phase_length
        self.pruning_phase_length = pruning_phase_length
        self.current_phase = 'growth'
        self.phase_counter = 0
        self.performance_history = []

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.performance_metric, 0)
        self.performance_history.append(current_metric)
        
        # Adjust thresholds dynamically based on current performance
        self.add_synapses_threshold = self.initial_add_synapses_threshold * (1 - current_metric)
        self.prune_synapses_threshold = self.initial_prune_synapses_threshold * current_metric

        # Phase management (growth vs pruning)
        self.phase_counter += 1
        if self.current_phase == 'growth' and self.phase_counter >= self.growth_phase_length:
            self.current_phase = 'pruning'
            self.phase_counter = 0
        elif self.current_phase == 'pruning' and self.phase_counter >= self.pruning_phase_length:
            self.current_phase = 'growth'
            self.phase_counter = 0

        # Synapse management based on improvement rate
        if len(self.performance_history) > 5:
            improvement_rate = (current_metric - self.performance_history[-5]) / 5
            if improvement_rate > 0.01:
                self.synaptogenesis_layer.add_synapses()
            elif improvement_rate < 0.001:
                self.synaptogenesis_layer.prune_synapses()

        if current_metric >= self.target_metric:
            if self.current_phase == 'growth' and current_metric < self.add_synapses_threshold:
                self.synaptogenesis_layer.add_synapses()
            elif self.current_phase == 'pruning' and current_metric > self.prune_synapses_threshold:
                self.synaptogenesis_layer.prune_synapses()

def create_csmselnn_model(input_shape, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    x = Flatten()(x)
    
    synaptogenesis_layer = SynaptogenesisLayer(
        initial_reservoir_size=initial_reservoir_size,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim
    )
    
    expanded_inputs = ExpandDimsLayer(axis=1)(x)
    rnn_layer = tf.keras.layers.RNN(synaptogenesis_layer, return_sequences=False)
    selnn_output = rnn_layer(expanded_inputs)
    selnn_output = Flatten()(selnn_output)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(selnn_output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    
    predicted_hidden = Dense(np.prod(input_shape), name="self_modeling_output")(x)
    
    outputs = Dense(output_dim, activation='softmax', name="classification_output")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=[outputs, predicted_hidden])
    
    model.compile(optimizer='adam', loss={'classification_output': 'sparse_categorical_crossentropy', 'self_modeling_output': 'mse'},
                  metrics={'classification_output': 'accuracy'})
    
    return model, synaptogenesis_layer

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize and expand dimensions for the image data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Flatten the training and testing data to match the output shape of the self-modeling head
    x_train_flat = np.reshape(x_train, (x_train.shape[0], -1))  # Flatten to [num_samples, 784]
    x_test_flat = np.reshape(x_test, (x_test.shape[0], -1))      # Flatten to [num_samples, 784]

    initial_reservoir_size = 128
    spectral_radius = 1.2
    leak_rate = 0.2
    spike_threshold = 0.5
    max_reservoir_dim = 256
    output_dim = 10

    # Build the model
    model, synaptogenesis_layer = create_csmselnn_model(
        input_shape=x_train.shape[1:], 
        initial_reservoir_size=initial_reservoir_size,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim,
        output_dim=output_dim
    )
    
    synaptogenesis_callback = SynaptogenesisCallback(synaptogenesis_layer)

    # Update the EarlyStopping callback with the mode parameter
    early_stopping = EarlyStopping(monitor='val_classification_output_accuracy', patience=10, restore_best_weights=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_classification_output_accuracy', factor=0.5, patience=5, min_lr=1e-6)

    # Update the inputs to the flattened version for the self-modeling task
    history = model.fit(
        x_train, 
        {'classification_output': y_train, 'self_modeling_output': x_train_flat},  # Use flattened version for self-modeling
        validation_data=(x_test, {'classification_output': y_test, 'self_modeling_output': x_test_flat}),  # Flattened test data
        epochs=10,
        callbacks=[early_stopping, reduce_lr, synaptogenesis_callback],
        batch_size=64
    )
    
    test_loss, test_acc = model.evaluate(x_test, {'classification_output': y_test, 'self_modeling_output': x_test.reshape(x_test.shape[0], -1)}, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

    plt.plot(history.history['classification_output_accuracy'])
    plt.plot(history.history['val_classification_output_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()





# Convolutional Self-Modeling Spiking Elastic Liquid Neural Network (CSMSELNN) version 2
# python csmselnn_mnist_v2.py
# Test Accuracy: 0.9920