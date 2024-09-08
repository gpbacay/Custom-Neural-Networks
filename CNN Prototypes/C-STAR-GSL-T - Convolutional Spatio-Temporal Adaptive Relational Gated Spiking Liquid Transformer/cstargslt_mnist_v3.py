import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, GlobalAveragePooling2D, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

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

# Custom Gated Spiking Liquid Neural Network Step
class GatedSLNNStep(tf.keras.layers.Layer):
    def __init__(self, reservoir_weights, input_weights, gate_weights, leak_rate, spike_threshold, max_reservoir_dim, **kwargs):
        super().__init__(**kwargs)
        self.reservoir_weights = reservoir_weights
        self.input_weights = input_weights
        self.gate_weights = gate_weights
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_reservoir_dim = max_reservoir_dim

    @property
    def state_size(self):
        return (self.max_reservoir_dim,)

    def call(self, inputs, states):
        prev_state = states[0][:, :self.reservoir_weights.shape[0]]
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights, transpose_b=True)
        gate_part = tf.matmul(inputs, self.gate_weights, transpose_b=True)

        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)
        state = (1 - self.leak_rate) * (f_gate * prev_state) + self.leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state

        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        active_size = tf.shape(state)[-1]
        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_reservoir_dim - active_size])], axis=1)
        
        return padded_state, [padded_state]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "leak_rate": self.leak_rate,
            "spike_threshold": self.spike_threshold,
            "max_reservoir_dim": self.max_reservoir_dim,
            "reservoir_weights": self.reservoir_weights.tolist(),
            "input_weights": self.input_weights.tolist(),
            "gate_weights": self.gate_weights.tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        reservoir_weights = np.array(config.pop('reservoir_weights'))
        input_weights = np.array(config.pop('input_weights'))
        gate_weights = np.array(config.pop('gate_weights'))
        return cls(reservoir_weights, input_weights, gate_weights, **config)

# Initialize Reservoir
def initialize_reservoir(input_dim, reservoir_dim, spectral_radius):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    gate_weights = np.random.randn(3 * reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights, gate_weights

# Positional Encoding Layer
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

# Multi-Dimensional Attention Layer
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

# Create C-STAR-GSL-T Model
def create_cstar_gsl_t_model(input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim, l2_reg=1e-4):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_regularizer=l2(l2_reg))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1, l2_reg=l2_reg)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2, l2_reg=l2_reg)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2, l2_reg=l2_reg)

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, x.shape[-1]))(x)

    pos_encoding_layer = PositionalEncoding(max_position=1, d_model=x.shape[-1])
    x = pos_encoding_layer(x)

    # Multi-Dimensional Attention for Relational Reasoning
    multi_dim_attention = MultiDimAttention()
    x = multi_dim_attention(x)

    # Gated Spiking Liquid Neural Network Step
    reservoir_weights, input_weights, gate_weights = initialize_reservoir(x.shape[-1], reservoir_dim, spectral_radius)
    gsl_nn_step = GatedSLNNStep(reservoir_weights, input_weights, gate_weights, leak_rate, spike_threshold, max_reservoir_dim)
    x = tf.keras.layers.RNN(gsl_nn_step)(x)

    x = Dense(output_dim, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def preprocess_data(x):
    x = x.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=-1)

# Main Function
def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    x_train = preprocess_data(x_train)
    x_val = preprocess_data(x_val)
    x_test = preprocess_data(x_test)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    input_shape = (28, 28, 1)
    reservoir_dim = 256
    max_reservoir_dim = 512
    spectral_radius = 1.5
    leak_rate = 0.3
    spike_threshold = 0.5
    output_dim = 10
    l2_reg = 1e-4

    model = create_cstar_gsl_t_model(input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim, l2_reg=l2_reg)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)

    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    main()



# Convolutional Spatio-Temporal Adaptive Relational Gated Spiking Liquid Transformer (C-STAR-GSL-T) 
# with Positional Encoding and Multi-Dimensional Attention
# python cstargslt_mnist_v3.py
# Test Accuracy: 0.9920
