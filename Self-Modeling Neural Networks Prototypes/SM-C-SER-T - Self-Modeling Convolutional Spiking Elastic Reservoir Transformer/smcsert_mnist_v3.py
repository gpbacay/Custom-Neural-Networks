import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, GlobalAveragePooling2D, RNN, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


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


class ReservoirComputingLayer(tf.keras.layers.Layer):
    def __init__(self, initial_reservoir_size, input_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, synaptogenesis_rate=0.01, prune_threshold=0.05, **kwargs):
        super().__init__(**kwargs)
        self.initial_reservoir_size = initial_reservoir_size
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_reservoir_dim = max_reservoir_dim
        self.synaptogenesis_rate = synaptogenesis_rate
        self.prune_threshold = prune_threshold
        self.reservoir_weights = None
        self.input_weights = None
        self.state_size = max_reservoir_dim
        self.output_size = max_reservoir_dim
        self.current_size = initial_reservoir_size

    def build(self, input_shape):
        self._initialize_weights()
        super().build(input_shape)

    def _initialize_weights(self):
        # Initializing weights as in the previous version
        reservoir_weights = np.random.randn(self.initial_reservoir_size, self.initial_reservoir_size) * 0.1
        reservoir_weights = np.dot(reservoir_weights, reservoir_weights.T)
        spectral_radius = np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
        reservoir_weights *= self.spectral_radius / spectral_radius
        self.reservoir_weights = self.add_weight(
            name='reservoir_weights',
            shape=(self.max_reservoir_dim, self.max_reservoir_dim),
            initializer=tf.constant_initializer(np.pad(reservoir_weights, ((0, self.max_reservoir_dim - self.initial_reservoir_size), (0, self.max_reservoir_dim - self.initial_reservoir_size)))),
            trainable=False
        )

        input_weights = np.random.randn(self.initial_reservoir_size, self.input_dim) * 0.1
        self.input_weights = self.add_weight(
            name='input_weights',
            shape=(self.max_reservoir_dim, self.input_dim),
            initializer=tf.constant_initializer(np.pad(input_weights, ((0, self.max_reservoir_dim - self.initial_reservoir_size), (0, 0)))),
            trainable=False
        )

    def call(self, inputs, states):
        prev_state = states[0][:, :self.current_size]
        input_contribution = tf.matmul(inputs, tf.transpose(self.input_weights[:self.current_size]))
        reservoir_contribution = tf.matmul(prev_state, self.reservoir_weights[:self.current_size, :self.current_size])

        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_contribution + reservoir_contribution)

        # Spiking dynamics
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)
        padded_state = tf.pad(state, [[0, 0], [0, self.max_reservoir_dim - tf.shape(state)[1]]])

        # Synaptogenesis: grow new synapses based on activity
        self._apply_synaptogenesis(state)

        return padded_state, [padded_state]

    def _apply_synaptogenesis(self, state):
        active_neurons = tf.reduce_sum(tf.cast(state > self.spike_threshold, dtype=tf.float32), axis=0)[:self.current_size]  # Active neurons across the batch, only for current active size
        grow_synapses = tf.random.uniform(active_neurons.shape) < self.synaptogenesis_rate  # Probabilistic growth
        weak_synapses = tf.abs(self.reservoir_weights[:self.current_size, :self.current_size]) < self.prune_threshold  # Identify weak synapses in the active region

        # Grow new synapses for active neurons in the active region
        new_synapses = tf.cast(grow_synapses, dtype=tf.float32) * tf.random.normal(self.reservoir_weights[:self.current_size, :self.current_size].shape)
        self.reservoir_weights.assign(tf.pad(
            self.reservoir_weights[:self.current_size, :self.current_size] + new_synapses,
            [[0, self.max_reservoir_dim - self.current_size], [0, self.max_reservoir_dim - self.current_size]]
        ))

        # Prune weak synapses in the active region
        pruned_weights = tf.where(weak_synapses, tf.zeros_like(self.reservoir_weights[:self.current_size, :self.current_size]), self.reservoir_weights[:self.current_size, :self.current_size])
        self.reservoir_weights.assign(tf.pad(
            pruned_weights,
            [[0, self.max_reservoir_dim - self.current_size], [0, self.max_reservoir_dim - self.current_size]]
        ))

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros((batch_size, self.max_reservoir_dim), dtype=tf.float32)]


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_position, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, max_position, d_model):
        angle_rads = self.get_angles(np.arange(max_position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]


class MultiDimAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MultiDimAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.temporal_dense = Dense(self.channels, activation='sigmoid')
        self.channel_dense = Dense(self.channels, activation='sigmoid')
        self.spatial_dense = Dense(1, activation='sigmoid')
        super(MultiDimAttention, self).build(input_shape)

    def temporal_attention(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.temporal_dense(concat)
        return inputs * attention

    def channel_attention(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.channel_dense(concat)
        return inputs * attention

    def spatial_attention(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.spatial_dense(concat)
        return inputs * attention

    def call(self, inputs):
        x = self.temporal_attention(inputs)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class FeedbackModulationLayer(tf.keras.layers.Layer):
    def __init__(self, internal_units=128, initial_feedback_strength=0.1, output_dense=4096, **kwargs):
        super().__init__(**kwargs)
        self.internal_units = internal_units
        self.initial_feedback_strength = initial_feedback_strength
        
        self.state_dense = Dense(internal_units, activation='relu')
        self.gate_dense = Dense(internal_units, activation='sigmoid')
        self.output_dense = Dense(output_dense)

        # Recurrent feedback layer
        self.recurrent_feedback = tf.keras.layers.SimpleRNN(
            internal_units, 
            return_sequences=True, 
            return_state=True
        )

        self.feedback_strength = self.add_weight(
            shape=(),
            initializer=tf.constant_initializer(initial_feedback_strength),
            trainable=True,
            name='feedback_strength'
        )

    def build(self, input_shape):
        self.feedback_weights = self.add_weight(
            shape=(self.internal_units, self.internal_units),
            initializer='random_normal',
            trainable=True,
            name='feedback_weights'
        )

        self.bias = self.add_weight(
            shape=(self.internal_units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

    def call(self, inputs, training=None):
        internal_state = self.state_dense(inputs)
        gate = self.gate_dense(inputs)

        # Reshape to add time dimension for RNN
        internal_state_reshaped = tf.expand_dims(internal_state, axis=1)

        # Recurrent feedback mechanism
        recurrent_output, recurrent_state = self.recurrent_feedback(internal_state_reshaped)

        # Reshape back to 2D (batch_size, internal_units)
        feedback = tf.matmul(recurrent_state, self.feedback_weights) + self.bias

        # Modulate internal state with adaptive feedback
        modulated_internal = internal_state + self.feedback_strength * gate * feedback
        modulated_output = self.output_dense(modulated_internal)

        if training:
            error_factor = tf.reduce_mean(tf.abs(internal_state))
            self.feedback_strength.assign(tf.clip_by_value(self.feedback_strength * (1 + 0.01 * error_factor), 0.0, 1.0))

        return modulated_output


def create_reservoir_cnn_rnn_model(input_shape, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim, l2_reg=1e-4):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_regularizer=l2(l2_reg))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1, l2_reg=l2_reg)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2, l2_reg=l2_reg)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2, l2_reg=l2_reg)

    x = GlobalAveragePooling2D()(x)
    x = Reshape((-1, x.shape[-1]))(x)

    x = RNN(ReservoirComputingLayer(
        initial_reservoir_size=initial_reservoir_size,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim), return_sequences=True)(x)

    x = PositionalEncoding(max_position=x.shape[1], d_model=x.shape[-1])(x)
    x = MultiDimAttention()(x)
    x = Flatten()(x)

    x = FeedbackModulationLayer()(x)

    outputs = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    input_shape = (28, 28, 1)  # Example shape for MNIST
    num_classes = 10
    initial_reservoir_size = 100
    spectral_radius = 1.25
    leak_rate = 0.1
    spike_threshold = 0.5
    max_reservoir_dim = 500
    l2_reg = 1e-4

    model = create_reservoir_cnn_rnn_model(
        input_shape=input_shape,
        initial_reservoir_size=initial_reservoir_size,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim,
        output_dim=num_classes,
        l2_reg=l2_reg
    )

    model.summary()

    # Load data (Example: MNIST)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Train model
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

    # Test accuracy
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()


# Self-Modeling Convolutional Spiking Elastic Reservoir Transformer (SM-C-SER-T) version 3
# with Positional Encoding and Multi-Dimensional Attention
# python smcsert_mnist_v3.py
# Test Accuracy: 0.9840 (not yet deployed)
