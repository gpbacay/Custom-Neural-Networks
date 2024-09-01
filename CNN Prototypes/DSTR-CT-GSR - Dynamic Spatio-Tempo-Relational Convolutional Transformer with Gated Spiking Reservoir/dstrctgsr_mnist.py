import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, LayerNormalization, Reshape, MultiHeadAttention, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class GatedSpikingReservoirStep(tf.keras.layers.Layer):
    def __init__(self, spatiotemporal_reservoir_weights, spatiotemporal_input_weights, spiking_gate_weights, leak_rate, spike_threshold, max_dynamic_reservoir_dim, **kwargs):
        super().__init__(**kwargs)
        self.spatiotemporal_reservoir_weights = spatiotemporal_reservoir_weights
        self.spatiotemporal_input_weights = spatiotemporal_input_weights
        self.spiking_gate_weights = spiking_gate_weights
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim

    @property
    def state_size(self):
        return (self.max_dynamic_reservoir_dim,)

    def call(self, inputs, states):
        prev_state = states[0][:, :self.spatiotemporal_reservoir_weights.shape[0]]
        input_part = tf.matmul(inputs, self.spatiotemporal_input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.spatiotemporal_reservoir_weights, transpose_b=True)
        gate_part = tf.matmul(inputs, self.spiking_gate_weights, transpose_b=True)

        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)
        state = (1 - self.leak_rate) * (f_gate * prev_state) + self.leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state

        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        active_size = tf.shape(state)[-1]
        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_dynamic_reservoir_dim - active_size])], axis=1)
        
        return padded_state, [padded_state]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "leak_rate": self.leak_rate,
            "spike_threshold": self.spike_threshold,
            "max_dynamic_reservoir_dim": self.max_dynamic_reservoir_dim,
            "spatiotemporal_reservoir_weights": self.spatiotemporal_reservoir_weights.tolist(),
            "spatiotemporal_input_weights": self.spatiotemporal_input_weights.tolist(),
            "spiking_gate_weights": self.spiking_gate_weights.tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        spatiotemporal_reservoir_weights = np.array(config.pop('spatiotemporal_reservoir_weights'))
        spatiotemporal_input_weights = np.array(config.pop('spatiotemporal_input_weights'))
        spiking_gate_weights = np.array(config.pop('spiking_gate_weights'))
        return cls(spatiotemporal_reservoir_weights, spatiotemporal_input_weights, spiking_gate_weights, **config)

def initialize_spatiotemporal_reservoir(input_dim, reservoir_dim, spectral_radius):
    spatiotemporal_reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    spatiotemporal_reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(spatiotemporal_reservoir_weights)))
    spatiotemporal_input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    spiking_gate_weights = np.random.randn(3 * reservoir_dim, input_dim) * 0.1
    return spatiotemporal_reservoir_weights, spatiotemporal_input_weights, spiking_gate_weights

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position, d_model):
        super(PositionalEncoding, self).__init__()
        self.max_position = max_position
        self.d_model = d_model
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

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'max_position': self.max_position,
            'd_model': self.d_model
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(config['max_position'], config['d_model'])

def create_dstr_ct_gsr_model(input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, output_dim, d_model=64, num_heads=4, l2_reg=1e-4):
    inputs = Input(shape=input_shape)

    # Convolutional Layers for Spatio-Temporal Feature Extraction
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Reshape and Apply Spatio-Temporal Positional Encoding
    x = Reshape((1, x.shape[-1]))(x)
    pos_encoding_layer = PositionalEncoding(max_position=1, d_model=x.shape[-1])
    x = pos_encoding_layer(x)

    # Multi-Head Attention for Relational Reasoning
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attention_output = Add()([x, attention_output])
    attention_output = LayerNormalization()(attention_output)

    # Gated Spiking Reservoir Processing
    spatiotemporal_reservoir_weights, spatiotemporal_input_weights, spiking_gate_weights = initialize_spatiotemporal_reservoir(x.shape[-1], reservoir_dim, spectral_radius)
    lnn_layer = tf.keras.layers.RNN(
        GatedSpikingReservoirStep(spatiotemporal_reservoir_weights, spatiotemporal_input_weights, spiking_gate_weights, leak_rate, spike_threshold, max_dynamic_reservoir_dim),
        return_sequences=True
    )

    lnn_output = lnn_layer(attention_output)
    lnn_output = Flatten()(lnn_output)

    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(lnn_output)
    x = Dropout(0.5)(x)
    outputs = Dense(output_dim, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def main():
    # Load and Prepare MNIST Data for Spatio-Temporal Processing
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    input_shape = x_train.shape[1:]  # (28, 28, 1)
    reservoir_dim = 256
    max_dynamic_reservoir_dim = 512
    spectral_radius = 1.5
    leak_rate = 0.3
    spike_threshold = 0.5
    output_dim = 10
    l2_reg = 1e-4

    model = create_dstr_ct_gsr_model(input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, output_dim, l2_reg=l2_reg)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)

    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr]
    )

    # Evaluate the Dynamic Spatio-Temporal Model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()





# DSTR-CT-GSR - Dynamic Spatio-Tempo-Relational Convolutional Transformer with Gated Spiking Reservoir
# python dstrctgsr_mnist.py
# Test Accuracy: 0.9922