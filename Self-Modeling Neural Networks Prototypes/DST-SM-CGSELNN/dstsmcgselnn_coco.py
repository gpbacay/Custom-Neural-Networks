import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pycocotools.coco import COCO  # Required for working with COCO dataset
import matplotlib.pyplot as plt


class HebbianHomeostaticLayer(tf.keras.layers.Layer):
    def __init__(self, units, learning_rate=0.001, target_avg=0.5, homeostatic_rate=0.001, **kwargs):
        super(HebbianHomeostaticLayer, self).__init__(**kwargs)
        self.units = units
        self.learning_rate = learning_rate
        self.target_avg = target_avg
        self.homeostatic_rate = homeostatic_rate

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs):
        inputs = tf.squeeze(inputs, axis=1)
        outputs = tf.matmul(inputs, self.kernel)

        # Hebbian update
        delta_weights = self.learning_rate * tf.matmul(tf.transpose(inputs), outputs)
        self.kernel.assign_add(delta_weights)

        # Homeostatic update
        avg_activation = tf.reduce_mean(self.kernel)
        self.kernel.assign_sub(self.homeostatic_rate * (avg_activation - self.target_avg))

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'learning_rate': self.learning_rate,
            'target_avg': self.target_avg,
            'homeostatic_rate': self.homeostatic_rate,
        })
        return config


class SpatioTemporalSummaryMixing(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout_rate=0.1, **kwargs):
        super(SpatioTemporalSummaryMixing, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.local_dense1 = Dense(4 * self.d_model, activation='gelu')
        self.local_dense2 = Dense(self.d_model)
        self.local_dropout = Dropout(dropout_rate)
        self.summary_dense1 = Dense(4 * self.d_model, activation='gelu')
        self.summary_dense2 = Dense(self.d_model)
        self.summary_dropout = Dropout(dropout_rate)
        self.combiner_dense1 = Dense(4 * self.d_model, activation='gelu')
        self.combiner_dense2 = Dense(self.d_model)
        self.combiner_dropout = Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dynamic_dense = Dense(self.d_model)

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

        if inputs.shape[-1] != output.shape[-1]:
            inputs = self.dynamic_dense(inputs)
        return self.layer_norm(inputs + output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
        })
        return config


class GatedSpikingElasticLNNStep(tf.keras.layers.Layer):
    def __init__(self, initial_reservoir_size, input_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, **kwargs):
        super().__init__(**kwargs)
        self.initial_reservoir_size = initial_reservoir_size
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim

        self.state_size = [self.max_dynamic_reservoir_dim]
        self.output_size = self.max_dynamic_reservoir_dim

        self.initialize_weights()

    def initialize_weights(self):
        self.spatiotemporal_reservoir_weights = self.add_weight(
            shape=(self.initial_reservoir_size, self.initial_reservoir_size),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=False,
            name='spatiotemporal_reservoir_weights'
        )
        self.spatiotemporal_input_weights = self.add_weight(
            shape=(self.initial_reservoir_size, self.input_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=False,
            name='spatiotemporal_input_weights'
        )
        self.spiking_gate_weights = self.add_weight(
            shape=(3 * self.initial_reservoir_size, self.input_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=False,
            name='spiking_gate_weights'
        )

    def call(self, inputs, states):
        prev_state = states[0][:, :tf.shape(self.spatiotemporal_reservoir_weights)[0]]

        input_part = tf.matmul(inputs, self.spatiotemporal_input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.spatiotemporal_reservoir_weights)
        gate_part = tf.matmul(inputs, self.spiking_gate_weights, transpose_b=True)

        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)

        state = (1 - self.leak_rate) * (f_gate * prev_state) + self.leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state

        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_dynamic_reservoir_dim - tf.shape(state)[-1]])], axis=1)

        return padded_state, [padded_state]

    def add_neurons(self, growth_rate):
        current_size = tf.shape(self.spatiotemporal_reservoir_weights)[0]
        new_neurons = min(growth_rate, self.max_dynamic_reservoir_dim - current_size)
        if new_neurons <= 0:
            return

        new_reservoir_weights = tf.random.normal((new_neurons, current_size + new_neurons))
        full_new_weights = tf.concat([
            tf.concat([self.spatiotemporal_reservoir_weights, tf.zeros((current_size, new_neurons))], axis=1),
            new_reservoir_weights
        ], axis=0)

        spectral_radius = tf.math.real(tf.reduce_max(tf.abs(tf.linalg.eigvals(full_new_weights))))
        scaling_factor = self.spectral_radius / spectral_radius
        new_reservoir_weights *= scaling_factor

        self.spatiotemporal_reservoir_weights = tf.Variable(full_new_weights, trainable=False)

        new_input_weights = tf.random.normal((new_neurons, self.input_dim)) * 0.1
        self.spatiotemporal_input_weights = tf.concat([self.spatiotemporal_input_weights, new_input_weights], axis=0)

        new_gate_weights = tf.random.normal((3 * new_neurons, self.input_dim)) * 0.1
        self.spiking_gate_weights = tf.concat([self.spiking_gate_weights, new_gate_weights], axis=0)

    def prune_connections(self, prune_rate):
        weights = self.spatiotemporal_reservoir_weights.numpy()
        threshold = np.percentile(np.abs(weights), prune_rate * 100)
        mask = np.abs(weights) > threshold
        self.spatiotemporal_reservoir_weights.assign(weights * mask)

    def get_config(self):
        config = super().get_config()
        config.update({
            'initial_reservoir_size': self.initial_reservoir_size,
            'input_dim': self.input_dim,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.leak_rate,
            'spike_threshold': self.spike_threshold,
            'max_dynamic_reservoir_dim': self.max_dynamic_reservoir_dim
        })
        return config


def create_dstsmcgselnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, output_dim):
    """Create a dynamic spiking temporal model for object detection."""
    inputs = Input(shape=input_shape)

    # Convolutional layers for feature extraction
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Spiking elastic liquid neural network
    x = Reshape((1, -1))(x)
    x = tf.keras.layers.RNN(GatedSpikingElasticLNNStep(
        initial_reservoir_size=reservoir_dim,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_dynamic_reservoir_dim=max_dynamic_reservoir_dim))(x)

    # Classification head (output for object detection)
    outputs = Dense(output_dim, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def main():
    """Main function to create and compile the model for COCO dataset."""
    input_shape = (640, 640, 3)  # For COCO dataset RGB images
    reservoir_dim = 128
    spectral_radius = 0.9
    leak_rate = 0.1
    spike_threshold = 0.5
    max_dynamic_reservoir_dim = 256
    output_dim = 80  # For COCO's 80 classes

    # Create the model
    model = create_dstsmcgselnn_model(
        input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, output_dim
    )

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Display the model summary
    model.summary()


if __name__ == "__main__":
    main()

# python dstsmcgselnn_coco.py
# Test Accuracy: Under active development