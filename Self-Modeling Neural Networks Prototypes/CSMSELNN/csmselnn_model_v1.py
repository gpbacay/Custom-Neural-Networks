import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Layer, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback

class SpikingElasticLNNStep(tf.keras.layers.Layer):
    def __init__(self, initial_reservoir_size, input_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, pruning_frequency=5, pruning_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.initial_reservoir_size = initial_reservoir_size
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_reservoir_dim = max_reservoir_dim
        self.pruning_frequency = pruning_frequency
        self.pruning_rate = pruning_rate
        self.epoch_counter = 0
        
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

    def add_neurons(self):
        current_size = tf.shape(self.reservoir_weights)[0]
        growth_rate = max(1, int(current_size * 0.1))  # Add 10% or at least 1 neuron
        new_neurons = min(growth_rate, self.max_reservoir_dim - current_size)
        new_size = current_size + new_neurons
        if new_size > self.max_reservoir_dim:
            return

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
        self.reservoir_weights = tf.Variable(updated_reservoir_weights, dtype=tf.float32, trainable=False)
        self.input_weights = tf.Variable(updated_input_weights, dtype=tf.float32, trainable=False)

    def prune_connections(self):
        self.epoch_counter += 1
        if self.epoch_counter % self.pruning_frequency == 0:
            threshold = np.percentile(np.abs(self.reservoir_weights.numpy()), self.pruning_rate * 100)
            mask = tf.abs(self.reservoir_weights) > threshold
            self.reservoir_weights.assign(tf.where(mask, self.reservoir_weights, tf.zeros_like(self.reservoir_weights)))

class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

class SelfModelingCallback(Callback):
    def __init__(self, selnn_step_layer, performance_metric='classification_output_accuracy', target_metric=0.95, 
                 add_neurons_threshold=0.01, prune_connections_threshold=0.1, growth_phase_length=10, pruning_phase_length=5):
        super().__init__()
        self.selnn_step_layer = selnn_step_layer
        self.performance_metric = performance_metric
        self.target_metric = target_metric
        self.initial_add_neurons_threshold = add_neurons_threshold
        self.initial_prune_connections_threshold = prune_connections_threshold
        self.growth_phase_length = growth_phase_length
        self.pruning_phase_length = pruning_phase_length
        self.current_phase = 'growth'
        self.phase_counter = 0
        self.performance_history = []

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.performance_metric, 0)
        self_modeling_output = logs.get('self_modeling_output_loss', float('inf'))
        
        self.performance_history.append(current_metric)
        
        # Adjust thresholds based on current performance
        self.add_neurons_threshold = self.initial_add_neurons_threshold * (1 - current_metric)
        self.prune_connections_threshold = self.initial_prune_connections_threshold * current_metric

        self.phase_counter += 1
        if self.current_phase == 'growth' and self.phase_counter >= self.growth_phase_length:
            self.current_phase = 'pruning'
            self.phase_counter = 0
        elif self.current_phase == 'pruning' and self.phase_counter >= self.pruning_phase_length:
            self.current_phase = 'growth'
            self.phase_counter = 0

        if len(self.performance_history) > 5:
            improvement_rate = (current_metric - self.performance_history[-5]) / 5
            
            if improvement_rate > 0.01:  # Fast improvement
                self.selnn_step_layer.add_neurons()  # Add more neurons
            elif improvement_rate < 0.001:  # Slow improvement
                self.selnn_step_layer.prune_connections()  # More aggressive pruning

        if current_metric >= self.target_metric:
            print(f" - Performance metric {self.performance_metric} reached target {self.target_metric}. Current phase: {self.current_phase}")
            if self.current_phase == 'growth':
                if self_modeling_output < self.add_neurons_threshold:
                    self.selnn_step_layer.add_neurons()
            elif self.current_phase == 'pruning':
                self.selnn_step_layer.prune_connections()

def create_csmselnn_model(input_shape, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    x = Flatten()(x)
    
    selnn_step_layer = SpikingElasticLNNStep(
        initial_reservoir_size=initial_reservoir_size,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim
    )
    
    expanded_inputs = ExpandDimsLayer(axis=1)(x)
    rnn_layer = tf.keras.layers.RNN(selnn_step_layer, return_sequences=True)
    selnn_output = rnn_layer(expanded_inputs)
    selnn_output = Flatten()(selnn_output)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(selnn_output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    
    predicted_hidden = Dense(np.prod(input_shape), name="self_modeling_output")(x)
    
    outputs = Dense(output_dim, activation='softmax', name="classification_output")(x)
    
    model = tf.keras.Model(inputs, [outputs, predicted_hidden])
    return model, selnn_step_layer