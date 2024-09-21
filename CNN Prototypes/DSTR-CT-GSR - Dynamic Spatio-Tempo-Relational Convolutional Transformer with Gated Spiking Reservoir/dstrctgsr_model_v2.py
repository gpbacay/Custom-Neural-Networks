import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Reshape

# Gated Spiking Reservoir Layer
class GatedSpikingReservoirStep(tf.keras.layers.Layer):
    def __init__(self, spatiotemporal_reservoir_weights, spatiotemporal_input_weights, spiking_gate_weights, leak_rate, spike_threshold, max_dynamic_reservoir_dim, **kwargs):
        super().__init__(**kwargs)
        self.spatiotemporal_reservoir_weights = tf.convert_to_tensor(spatiotemporal_reservoir_weights, dtype=tf.float32)
        self.spatiotemporal_input_weights = tf.convert_to_tensor(spatiotemporal_input_weights, dtype=tf.float32)
        self.spiking_gate_weights = tf.convert_to_tensor(spiking_gate_weights, dtype=tf.float32)
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'spatiotemporal_reservoir_weights': self.spatiotemporal_reservoir_weights.numpy().tolist(),
            'spatiotemporal_input_weights': self.spatiotemporal_input_weights.numpy().tolist(),
            'spiking_gate_weights': self.spiking_gate_weights.numpy().tolist(),
            'leak_rate': self.leak_rate,
            'spike_threshold': self.spike_threshold,
            'max_dynamic_reservoir_dim': self.max_dynamic_reservoir_dim,
        })
        return config

    @property
    def state_size(self):
        return (self.max_dynamic_reservoir_dim,)

    def call(self, inputs, states):
        prev_state = states[0]
        
        # Ensure prev_state is a tensor
        prev_state = tf.convert_to_tensor(prev_state, dtype=tf.float32)

        # Slice the previous state to match the reservoir dimensions
        prev_state = prev_state[:, :self.spatiotemporal_reservoir_weights.shape[0]]

        input_part = tf.matmul(inputs, self.spatiotemporal_input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.spatiotemporal_reservoir_weights, transpose_b=True)
        gate_part = tf.matmul(inputs, self.spiking_gate_weights, transpose_b=True)

        # Split gates into input, forget, and output gates
        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)
        
        # Update state based on gates and inputs
        state = (1 - self.leak_rate) * (f_gate * prev_state) + self.leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state

        # Generate spikes based on spike threshold
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        # Pad the state to the maximum reservoir dimension
        active_size = tf.shape(state)[-1]
        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_dynamic_reservoir_dim - active_size])], axis=1)

        return padded_state, [padded_state]

# Initialize the spatio-temporal reservoir weights
def initialize_spatiotemporal_reservoir(input_dim, reservoir_dim, spectral_radius):
    spatiotemporal_reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    spatiotemporal_reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(spatiotemporal_reservoir_weights)))
    spatiotemporal_input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    spiking_gate_weights = np.random.randn(3 * reservoir_dim, input_dim) * 0.1
    return spatiotemporal_reservoir_weights, spatiotemporal_input_weights, spiking_gate_weights

# Spatio-Temporal Summary Mixing Layer (Replacement for Attention Mechanism)
class SpatioTemporalSummaryMixing(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff=None, dropout_rate=0.1, **kwargs):
        super(SpatioTemporalSummaryMixing, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        
        # Local transformation (spatial aspect)
        self.local_dense1 = Dense(self.d_ff, activation='gelu')
        self.local_dense2 = Dense(self.d_model)
        self.local_dropout = Dropout(dropout_rate)
        
        # Summary function (temporal aspect)
        self.summary_dense1 = Dense(self.d_ff, activation='gelu')
        self.summary_dense2 = Dense(self.d_model)
        self.summary_dropout = Dropout(dropout_rate)
        
        # Combiner function (spatio-temporal combination)
        self.combiner_dense1 = Dense(self.d_ff, activation='gelu')
        self.combiner_dense2 = Dense(self.d_model)
        self.combiner_dropout = Dropout(dropout_rate)
        
        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dynamic layer for reshaping inputs if needed
        self.dynamic_dense = Dense(self.d_model)

    def call(self, inputs, training=False):
        # Local (spatial) transformation
        local_output = self.local_dense1(inputs)
        local_output = self.local_dense2(local_output)
        local_output = self.local_dropout(local_output, training=training)
        
        # Summary (temporal) function
        summary = self.summary_dense1(inputs)
        summary = self.summary_dense2(summary)
        summary = self.summary_dropout(summary, training=training)
        
        # Calculate mean summary (temporal)
        mean_summary = tf.reduce_mean(summary, axis=1, keepdims=True)
        
        # Repeat mean summary for each time step (temporal extension)
        mean_summary = tf.tile(mean_summary, [1, tf.shape(inputs)[1], 1])
        
        # Combine local (spatial) and summary (temporal) information
        combined = tf.concat([local_output, mean_summary], axis=-1)
        output = self.combiner_dense1(combined)
        output = self.combiner_dense2(output)
        output = self.combiner_dropout(output, training=training)
        
        # Ensure the dimensionality of 'inputs' matches 'output' for residual connection
        if inputs.shape[-1] != output.shape[-1]:
            inputs = self.dynamic_dense(inputs)

        # Residual connection and layer normalization
        return self.layer_norm(inputs + output)

    def get_config(self):
        config = super(SpatioTemporalSummaryMixing, self).get_config()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'dropout_rate': self.local_dropout.rate,
        })
        return config

# Model creation
def create_dstr_ct_gsr_model(input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, output_dim, d_model=64, l2_reg=1e-4):
    inputs = Input(shape=input_shape)

    # Convolutional Layers for Spatio-Temporal Feature Extraction
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Spatio-Temporal Summary Mixing for Relational Reasoning (Replacing Attention)
    summary_mixing_layer = SpatioTemporalSummaryMixing(d_model=128)
    x = Reshape((1, x.shape[-1]))(x)
    x = summary_mixing_layer(x)

    # Gated Spiking Reservoir Processing
    spatiotemporal_reservoir_weights, spatiotemporal_input_weights, spiking_gate_weights = initialize_spatiotemporal_reservoir(x.shape[-1], reservoir_dim, spectral_radius)
    lnn_layer = tf.keras.layers.RNN(
        GatedSpikingReservoirStep(spatiotemporal_reservoir_weights, spatiotemporal_input_weights, spiking_gate_weights, leak_rate, spike_threshold, max_dynamic_reservoir_dim),
        return_sequences=True
    )

    lnn_output = lnn_layer(x)
    lnn_output = Flatten()(lnn_output)

    # Fully Connected Layers
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(lnn_output)
    x = Dropout(0.5)(x)
    outputs = Dense(output_dim, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)

    model = tf.keras.Model(inputs, outputs)
    return model