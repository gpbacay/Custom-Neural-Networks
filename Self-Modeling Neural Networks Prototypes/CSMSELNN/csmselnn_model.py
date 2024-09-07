import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Layer, Conv2D, MaxPooling2D

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
        
        # Initialize variables
        self.reservoir_weights = self.add_weight(shape=(self.initial_reservoir_size, self.initial_reservoir_size),
                                                initializer=self._init_reservoir_weights,
                                                trainable=False,
                                                name='reservoir_weights')
        self.input_weights = self.add_weight(shape=(self.initial_reservoir_size, self.input_dim),
                                             initializer='random_normal',
                                             trainable=False,
                                             name='input_weights')

    def _init_reservoir_weights(self, shape, dtype=None, partition_info=None):
        reservoir_weights = np.random.randn(self.initial_reservoir_size, self.initial_reservoir_size)
        reservoir_weights *= self.spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
        return tf.convert_to_tensor(reservoir_weights, dtype=tf.float32)

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

        new_reservoir_weights = tf.random.normal((new_neurons, new_size))
        full_new_weights = tf.concat([
            tf.concat([self.reservoir_weights, tf.zeros((current_size, new_neurons))], axis=1),
            new_reservoir_weights
        ], axis=0)
        spectral_radius = tf.math.real(tf.reduce_max(tf.abs(tf.linalg.eigvals(full_new_weights))))
        scaling_factor = self.spectral_radius / spectral_radius
        new_reservoir_weights *= scaling_factor
        new_input_weights = tf.random.normal((new_neurons, self.input_dim)) * 0.1

        updated_reservoir_weights = tf.concat([self.reservoir_weights, new_reservoir_weights[:, :current_size]], axis=0)
        updated_reservoir_weights = tf.concat([updated_reservoir_weights, 
                                               tf.concat([tf.transpose(new_reservoir_weights[:, :current_size]), 
                                                          new_reservoir_weights[:, current_size:]], axis=0)], axis=1)
        updated_input_weights = tf.concat([self.input_weights, new_input_weights], axis=0)
        self.reservoir_weights.assign(updated_reservoir_weights)
        self.input_weights.assign(updated_input_weights)

    def prune_connections(self, threshold):
        mask = tf.abs(self.reservoir_weights) > threshold
        self.reservoir_weights.assign(tf.where(mask, self.reservoir_weights, tf.zeros_like(self.reservoir_weights)))

# Custom Layer for Expanding Dimensions
class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

# Function to Create the CSMSELNN Model
def create_csmselnn_model(input_shape, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim):
    inputs = Input(shape=input_shape)
    
    # Add convolutional layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    x = Flatten()(x)
    
    # Initialize SELNN layer
    selnn_step_layer = SpikingElasticLNNStep(
        initial_reservoir_size=initial_reservoir_size,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim
    )
    
    expanded_inputs = ExpandDimsLayer(axis=1)(x)
    rnn_layer = tf.keras.layers.RNN(selnn_step_layer, return_sequences=False)
    selnn_output = rnn_layer(expanded_inputs)
    selnn_output = Flatten()(selnn_output)
    
    # Build the rest of the model
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(selnn_output)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Add self-modeling output
    predicted_hidden = Dense(np.prod(input_shape), name="self_modeling_output")(x)
    
    # Classification output
    outputs = Dense(output_dim, activation='softmax', name="classification_output")(x)
    
    model = tf.keras.Model(inputs, [outputs, predicted_hidden])
    return model, selnn_step_layer
