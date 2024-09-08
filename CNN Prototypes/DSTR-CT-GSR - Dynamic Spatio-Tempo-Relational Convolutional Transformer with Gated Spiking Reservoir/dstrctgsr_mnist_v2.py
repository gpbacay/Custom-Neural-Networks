import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Reshape, Add, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position, d_model):
        super(PositionalEncoding, self).__init__()
        self.max_position = max_position
        self.d_model = d_model

    def get_angles(self, positions, i):
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(self.d_model, tf.float32))
        return positions * angles

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.max_position, delta=1, dtype=tf.float32)
        i = tf.range(start=0, limit=self.d_model, delta=1, dtype=tf.float32)
        angles = self.get_angles(
            positions=tf.expand_dims(positions, -1),
            i=tf.expand_dims(i, 0)
        )
        
        sines = tf.sin(angles[:, 0::2])
        cosines = tf.cos(angles[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return inputs + tf.cast(pos_encoding, dtype=inputs.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

class MultiHeadSpikingAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model):
        super(MultiHeadSpikingAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        batch_size = tf.shape(x)[0]

        q = self.wq(x)  # (batch_size, seq_len, d_model)
        k = self.wk(x)  # (batch_size, seq_len, d_model)
        v = self.wv(x)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply spiking activation (using ReLU as an approximation)
        attention_weights = tf.nn.relu(scaled_attention_logits)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

class GatedSpikingReservoirStep(tf.keras.layers.Layer):
    def __init__(self, reservoir_weights, input_weights, gate_weights, leak_rate, spike_threshold, max_dynamic_reservoir_dim):
        super(GatedSpikingReservoirStep, self).__init__()
        self.reservoir_weights = reservoir_weights
        self.input_weights = input_weights
        self.gate_weights = gate_weights
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim

        # Define state size
        self.state_size = [reservoir_weights.shape[0], reservoir_weights.shape[0]]

    def call(self, inputs, states):
        prev_output = states[0]
        reservoir_state = states[1]

        input_contribution = tf.matmul(inputs, self.input_weights)
        reservoir_contribution = tf.matmul(reservoir_state, self.reservoir_weights)
        gate = tf.sigmoid(tf.matmul(inputs, self.gate_weights))

        new_reservoir_state = (1 - self.leak_rate) * reservoir_state + self.leak_rate * tf.tanh(input_contribution + reservoir_contribution)
        new_reservoir_state = new_reservoir_state * gate

        output = tf.where(new_reservoir_state > self.spike_threshold, tf.ones_like(new_reservoir_state), tf.zeros_like(new_reservoir_state))

        # Dynamic resizing of reservoir
        dynamic_dim = tf.minimum(tf.shape(output)[1], self.max_dynamic_reservoir_dim)
        output = output[:, :dynamic_dim]
        new_reservoir_state = new_reservoir_state[:, :dynamic_dim]

        return output, [output, new_reservoir_state]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        return [tf.zeros((batch_size, self.reservoir_weights.shape[0]), dtype=dtype),
                tf.zeros((batch_size, self.reservoir_weights.shape[0]), dtype=dtype)]

def initialize_spatiotemporal_reservoir(input_dim, reservoir_dim, spectral_radius):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights = (spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))) * reservoir_weights
    input_weights = np.random.randn(input_dim, reservoir_dim)
    gate_weights = np.random.randn(input_dim, reservoir_dim)
    return tf.Variable(reservoir_weights, dtype=tf.float32), tf.Variable(input_weights, dtype=tf.float32), tf.Variable(gate_weights, dtype=tf.float32)

def create_dstr_ct_gsr_model(input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, output_dim, num_heads=4, l2_reg=1e-4):
    inputs = Input(shape=input_shape)

    # Convolutional Layers for Spatio-Temporal Feature Extraction
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Get the flattened dimension
    d_model = x.shape[-1]

    # Reshape and Apply Spatio-Temporal Positional Encoding
    x = Reshape((1, d_model))(x)
    pos_encoding_layer = PositionalEncoding(max_position=1, d_model=d_model)
    x = pos_encoding_layer(x)

    # Multi-Head Spiking Attention for Relational Reasoning
    multi_head_spiking_attention = MultiHeadSpikingAttention(num_heads=num_heads, d_model=d_model)
    attention_output, _ = multi_head_spiking_attention(x)
    
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)

    # Gated Spiking Reservoir Processing
    spatiotemporal_reservoir_weights, spatiotemporal_input_weights, spiking_gate_weights = initialize_spatiotemporal_reservoir(d_model, reservoir_dim, spectral_radius)
    reservoir_layer = GatedSpikingReservoirStep(reservoir_weights=spatiotemporal_reservoir_weights,
                                                input_weights=spatiotemporal_input_weights,
                                                gate_weights=spiking_gate_weights,
                                                leak_rate=leak_rate,
                                                spike_threshold=spike_threshold,
                                                max_dynamic_reservoir_dim=max_dynamic_reservoir_dim)
    
    x, _ = tf.keras.layers.RNN(reservoir_layer, return_sequences=False)(x)

    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = Dropout(0.5)(x)  # Added dropout for regularization
    x = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs, x)
    return model

def main():
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

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()





# DSTR-CT-GSR - Dynamic Spatio-Tempo-Relational Convolutional Transformer with Gated Spiking Reservoir
# with Spiking Attention Mechanism (SAM)
# python dstrctgsr_mnist_v2.py
# Test Accuracy: 
