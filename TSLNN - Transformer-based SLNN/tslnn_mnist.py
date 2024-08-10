import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout, Flatten, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.datasets import mnist
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Custom Keras layer implementing a spiking Liquid Neural Network (LNN)
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

# Initialize the reservoir and input weights for the LNN
def initialize_spiking_lnn_reservoir(input_dim, reservoir_dim, spectral_radius, max_reservoir_dim):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights

# Transformer Decoder Block
def transformer_decoder_block(x, enc_output, num_heads, ff_dim):
    dec_self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    dec_self_attention = Dropout(0.1)(dec_self_attention)
    dec_self_attention = LayerNormalization(epsilon=1e-6)(x + dec_self_attention)

    dec_enc_attention = MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(dec_self_attention, enc_output)
    dec_enc_attention = Dropout(0.1)(dec_enc_attention)
    dec_enc_attention = LayerNormalization(epsilon=1e-6)(dec_self_attention + dec_enc_attention)
    
    x_ff = Dense(ff_dim, activation='relu')(dec_enc_attention)
    x_ff = Dense(x.shape[-1])(x_ff)
    x_ff = Dropout(0.1)(x_ff)
    x_ff = LayerNormalization(epsilon=1e-6)(dec_enc_attention + x_ff)
    
    return x_ff

# Create the hybrid Transformer-Spiking LNN model with Decoder
def create_transformer_spiking_model_with_decoder(input_dim, num_heads, ff_dim, num_blocks, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim):
    inputs = Input(shape=(1, input_dim))
    x = inputs
    
    # Transformer Encoder
    for _ in range(num_blocks):
        x = MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)(x, x)
        x = Dropout(0.1)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x_ff = Dense(ff_dim, activation='relu')(x)
        x_ff = Dense(input_dim)(x_ff)
        x = x + x_ff
        x = Dropout(0.1)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
    enc_output = GlobalAveragePooling1D()(x)
    
    # Initialize and apply the Spiking LNN layer
    reservoir_weights, input_weights = initialize_spiking_lnn_reservoir(input_dim, reservoir_dim, spectral_radius, max_reservoir_dim)
    lnn_layer = tf.keras.layers.RNN(
        SpikingLNNStep(reservoir_weights, input_weights, leak_rate, max_reservoir_dim),
        return_sequences=True
    )
    
    def apply_spiking_lnn(x):
        lnn_output = lnn_layer(tf.expand_dims(x, axis=1))
        return Flatten()(lnn_output)
    
    def get_output_shape(input_shape):
        return (input_shape[0], max_reservoir_dim)

    lnn_output = Lambda(apply_spiking_lnn, output_shape=get_output_shape)(enc_output)
    
    # Transformer Decoder
    x_dec = Input(shape=(1, input_dim))
    dec_output = transformer_decoder_block(x_dec, enc_output, num_heads, ff_dim)
    
    # Final dense layers for feature extraction
    x = Dense(64, activation='relu')(lnn_output)  # Reduced size
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)  # Reduced size
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=[inputs, x_dec], outputs=outputs)
    return model

# Load and preprocess MNIST data
def load_and_preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train.reshape(-1, 784), axis=1)
    x_test = np.expand_dims(x_test.reshape(-1, 784), axis=1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# Main execution
(input_dim, num_heads, ff_dim, num_blocks, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim) = (
    784, 2, 64, 1, 50, 1.2, 0.3, 500  # Reduced parameters
)

(x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()

# Create and compile the model
model = create_transformer_spiking_model_with_decoder(input_dim, num_heads, ff_dim, num_blocks, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([x_train, x_train], y_train, epochs=10, batch_size=64, validation_split=0.2)  # Reduced epochs and batch size

# Evaluate the model
test_loss, test_acc = model.evaluate([x_test, x_test], y_test)
print(f"Test accuracy: {test_acc}")

# Make predictions
predictions = model.predict([x_test, x_test])
print(f"Sample prediction: {np.argmax(predictions[0])}")



# pip install tensorflow-addons
# Transformer-based SLNN (TSLNN)
# python tslnn_mnist.py
# Test Accuracy: Error/Slow/Neep Optimization
