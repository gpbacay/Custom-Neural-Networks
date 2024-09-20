import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Reshape, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2

# EfficientNet Block
def efficientnet_block(inputs, filters, expansion_factor, stride, l2_reg=1e-4):
    expanded_filters = filters * expansion_factor
    x = Conv2D(expanded_filters, kernel_size=1, padding='same', use_bias=False, kernel_regularizer=l2(l2_reg))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(expanded_filters, kernel_size=3, padding='same', strides=stride, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if stride == 1 and inputs.shape[-1] == x.shape[-1]:
        x = tf.keras.layers.Add()([inputs, x])
    return x

# Spatio-Temporal Summary Mixing Layer
class SpatioTemporalSummaryMixing(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff=None, dropout_rate=0.1, **kwargs):
        super(SpatioTemporalSummaryMixing, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.local_dense1 = Dense(self.d_ff, activation='gelu')
        self.local_dense2 = Dense(d_model)
        self.local_dropout = Dropout(dropout_rate)
        self.summary_dense1 = Dense(self.d_ff, activation='gelu')
        self.summary_dense2 = Dense(d_model)
        self.summary_dropout = Dropout(dropout_rate)
        self.combiner_dense1 = Dense(self.d_ff, activation='gelu')
        self.combiner_dense2 = Dense(d_model)
        self.combiner_dropout = Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        local_output = self.local_dense1(inputs)
        local_output = self.local_dense2(local_output)
        local_output = self.local_dropout(local_output, training=training)
        summary = self.summary_dense1(inputs)
        summary = self.summary_dense2(summary)
        summary = self.summary_dropout(summary, training=training)
        mean_summary = tf.reduce_mean(summary, axis=1, keepdims=True)
        mean_summary = tf.tile(mean_summary, [1, tf.shape(inputs)[1], 1])
        combined = tf.concat([local_output, mean_summary], axis=-1)
        output = self.combiner_dense1(combined)
        output = self.combiner_dense2(output)
        output = self.combiner_dropout(output, training=training)
        return self.layer_norm(inputs + output)

# Spiking Elastic Liquid Neural Network (SELNN) Layer
class SpikingElasticLNNStep(tf.keras.layers.Layer):
    def __init__(self, initial_reservoir_size, input_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, hebbian_learning_rate=0.01, **kwargs):
        super().__init__(**kwargs)
        self.initial_reservoir_size = initial_reservoir_size
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_reservoir_dim = max_reservoir_dim
        self.hebbian_learning_rate = hebbian_learning_rate
        self.state_size = [self.max_reservoir_dim]
        self.reservoir_weights, self.input_weights, self.gate_weights = self.initialize_reservoir_weights()

    def initialize_reservoir_weights(self):
        reservoir_weights = np.random.randn(self.initial_reservoir_size, self.initial_reservoir_size)
        reservoir_weights *= self.spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
        reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        input_weights = np.random.randn(self.input_dim, self.initial_reservoir_size) * 0.1
        input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)
        gate_weights = np.random.randn(self.input_dim, 3 * self.initial_reservoir_size) * 0.1
        gate_weights = tf.Variable(gate_weights, dtype=tf.float32, trainable=False)
        return reservoir_weights, input_weights, gate_weights

    def call(self, inputs, states):
        prev_state = states[0][:, :self.reservoir_weights.shape[0]]
        input_contribution = tf.matmul(inputs, self.input_weights)
        reservoir_contribution = tf.matmul(prev_state, self.reservoir_weights)
        gate_contribution = tf.matmul(inputs, self.gate_weights)
        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_contribution), 3, axis=-1)
        state = (1 - self.leak_rate) * (f_gate * prev_state) + self.leak_rate * tf.tanh(i_gate * (input_contribution + reservoir_contribution))
        state = o_gate * state
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)
        self.update_weights(inputs, spikes)
        active_size = tf.shape(state)[-1]
        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_reservoir_dim - active_size])], axis=1)
        return padded_state, [padded_state]

    def update_weights(self, inputs, spikes):
        hebbian_input_update = self.hebbian_learning_rate * tf.matmul(tf.transpose(inputs), spikes)
        hebbian_reservoir_update = self.hebbian_learning_rate * tf.matmul(tf.transpose(spikes), spikes)
        self.input_weights.assign_add(hebbian_input_update)
        self.reservoir_weights.assign_add(hebbian_reservoir_update)

# Combining Spatio-Temporal Mixing with SELNN
def create_sm_stc_snn_model(input_shape, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1, l2_reg=1e-4)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2, l2_reg=1e-4)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2, l2_reg=1e-4)
    x = GlobalAveragePooling2D()(x)
    x = Reshape((-1, x.shape[-1]))(x)
    x = SpatioTemporalSummaryMixing(d_model=40)(x)
    selnn_step_layer = SpikingElasticLNNStep(
        initial_reservoir_size=initial_reservoir_size,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim
    )
    rnn_layer = tf.keras.layers.RNN(selnn_step_layer, return_sequences=True)
    selnn_output = rnn_layer(x)
    selnn_output = Flatten()(selnn_output)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(selnn_output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    predicted_hidden = Dense(np.prod(input_shape), name="self_modeling_output")(x)
    outputs = Dense(output_dim, activation='softmax', name="classification_output")(x)
    model = tf.keras.Model(inputs, [outputs, predicted_hidden])
    return model
