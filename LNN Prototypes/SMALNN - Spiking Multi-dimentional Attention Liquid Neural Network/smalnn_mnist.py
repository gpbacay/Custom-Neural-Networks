import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout, Flatten, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Attention class to define layers outside the functions
class MultiDimAttention:
    def __init__(self, temporal_units, channel_units, spatial_units):
        self.temporal_dense = Dense(temporal_units, activation='sigmoid')
        self.channel_dense = Dense(channel_units, activation='sigmoid')
        self.spatial_dense = Dense(spatial_units, activation='sigmoid')

    def temporal_attention(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-2, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-2, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.temporal_dense(concat)
        return inputs * attention

    def channel_attention(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.channel_dense(concat)
        return inputs * attention

    def spatial_attention(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.spatial_dense(concat)
        return inputs * attention

# Custom Spiking LNN Layer with Spiking Dynamics
class SpikingLNNStep(tf.keras.layers.Layer):
    def __init__(self, reservoir_weights, input_weights, leak_rate, max_reservoir_dim, spike_threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)
        self.leak_rate = leak_rate
        self.max_reservoir_dim = max_reservoir_dim
        self.spike_threshold = spike_threshold

    @property
    def state_size(self):
        return (self.max_reservoir_dim,)

    def call(self, inputs, states):
        prev_state = states[0][:, :self.reservoir_weights.shape[0]]
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights, transpose_b=True)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_part + reservoir_part)

        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        active_size = tf.shape(state)[-1]
        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_reservoir_dim - active_size])], axis=1)
        return padded_state, [padded_state]

def initialize_spiking_lnn_reservoir(input_dim, reservoir_dim, spectral_radius, max_reservoir_dim):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights

def create_spiking_nlnn_model(input_dim, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    reservoir_weights, input_weights = initialize_spiking_lnn_reservoir(input_dim, reservoir_dim, spectral_radius, max_reservoir_dim)

    lnn_layer = tf.keras.layers.RNN(
        SpikingLNNStep(reservoir_weights, input_weights, leak_rate, max_reservoir_dim),
        return_sequences=True
    )

    attention_module = MultiDimAttention(temporal_units=1, channel_units=max_reservoir_dim, spatial_units=1)

    def apply_spiking_lnn(x):
        x = tf.expand_dims(x, axis=1)  # Add time dimension
        lnn_output = lnn_layer(x)
        temporal_att = attention_module.temporal_attention(lnn_output)
        channel_att = attention_module.channel_attention(temporal_att)
        spatial_att = attention_module.spatial_attention(channel_att)
        return Flatten()(spatial_att)

    lnn_output = Lambda(apply_spiking_lnn, output_shape=(max_reservoir_dim,))(inputs)

    x = Dense(128, activation='relu')(lnn_output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(output_dim, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def preprocess_data(x):
    x = x.astype(np.float32) / 255.0
    return x.reshape(-1, 28 * 28)

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    x_train = preprocess_data(x_train)
    x_val = preprocess_data(x_val)
    x_test = preprocess_data(x_test)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)
    y_test = tf.keras.utils.to_categorical(y_test)

    input_dim = 28 * 28
    reservoir_dim = 512
    max_reservoir_dim = 4096
    spectral_radius = 1.5
    leak_rate = 0.3
    output_dim = 10
    num_epochs = 10
    batch_size = 64

    model = create_spiking_nlnn_model(input_dim, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim, output_dim)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
    ]

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()




# Spiking Multi-dimentional Attention Liquid Nueral Network (SMALNN)
# python smalnn_mnist.py
# Test Accuracy: 0.9616