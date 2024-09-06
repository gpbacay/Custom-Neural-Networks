import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, GlobalAveragePooling2D, Dropout, BatchNormalization, Reshape, MultiHeadAttention, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

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
        pos_encoding = self.pos_encoding[:, :seq_len, :]
        return inputs + pos_encoding

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

# Meta-Learning Mechanism
class MetaLearningMechanism(tf.keras.layers.Layer):
    def __init__(self, learning_rate=0.001):
        super(MetaLearningMechanism, self).__init__()
        self.learning_rate = learning_rate
        self.meta_weight = self.add_weight(shape=(), initializer="ones", trainable=True)
    
    def call(self, inputs, gradients):
        # Adjust the input weights based on the meta-learning weights
        adjusted_weights = inputs * self.meta_weight
        # Adjust learning rate dynamically
        adjusted_gradients = gradients * self.meta_weight
        return adjusted_weights, adjusted_gradients

    def get_config(self):
        config = super().get_config().copy()
        config.update({'learning_rate': self.learning_rate})
        return config

# Adaptive Gated Spiking Liquid Neural Network (GSLNN) Step
class AdaptiveGatedSLNNStep(Layer):
    def __init__(self, reservoir_dim, input_dim, leak_rate, spike_threshold, max_reservoir_dim, **kwargs):
        super().__init__(**kwargs)
        self.reservoir_dim = reservoir_dim
        self.input_dim = input_dim
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_reservoir_dim = max_reservoir_dim

    def build(self, input_shape):
        self.reservoir_weights = self.add_weight(
            shape=(self.max_reservoir_dim, self.max_reservoir_dim),
            initializer='glorot_uniform',
            name='reservoir_weights',
            trainable=True
        )
        self.input_weights = self.add_weight(
            shape=(self.max_reservoir_dim, self.input_dim),
            initializer='glorot_uniform',
            name='input_weights',
            trainable=True
        )
        self.gate_weights = self.add_weight(
            shape=(3 * self.max_reservoir_dim, self.input_dim),
            initializer='glorot_uniform',
            name='gate_weights',
            trainable=True
        )

    def call(self, inputs, states):
        inputs = tf.squeeze(inputs, axis=1)  # Remove the extra dimension
        prev_state = tf.squeeze(states[0], axis=1)[:, :self.reservoir_dim]

        input_part = tf.matmul(inputs, self.input_weights[:self.reservoir_dim, :], transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights[:self.reservoir_dim, :self.reservoir_dim])
        gate_part = tf.matmul(inputs, self.gate_weights[:3 * self.reservoir_dim, :], transpose_b=True)

        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)

        state = (1 - self.leak_rate) * (f_gate * prev_state) + self.leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state

        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        padded_state = tf.pad(state, [[0, 0], [0, self.max_reservoir_dim - self.reservoir_dim]])
        padded_state = tf.expand_dims(padded_state, axis=1)  # Add back the time dimension

        return padded_state, [padded_state]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "reservoir_dim": self.reservoir_dim,
            "input_dim": self.input_dim,
            "leak_rate": self.leak_rate,
            "spike_threshold": self.spike_threshold,
            "max_reservoir_dim": self.max_reservoir_dim
        })
        return config

# Improved Convolutional Self-Modeling Transformer (CSMT) with Meta-Learning and GSLNN
def create_csmt_model(input_shape, output_dim, d_model=64, self_modeling_weight=0.1):
    inputs = Input(shape=input_shape)
    
    # Enhanced Convolutional Layers
    def adaptive_conv(x, filters, kernel_size, strides=1):
        conv = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(x)
        norm = BatchNormalization()(conv)
        return tf.keras.layers.ReLU()(norm)
    
    x = adaptive_conv(inputs, 32, kernel_size=3, strides=2)
    x = adaptive_conv(x, 64, kernel_size=3, strides=2)
    x = adaptive_conv(x, 128, kernel_size=3, strides=2)
    x = adaptive_conv(x, 256, kernel_size=3, strides=2)
    x = adaptive_conv(x, 512, kernel_size=3, strides=2)
    
    # Apply Global Average Pooling
    model_features = GlobalAveragePooling2D()(x)
    x = Reshape((1, model_features.shape[-1]))(model_features)  # Add seq_len dimension for Dense layer

    # Add Positional Encoding
    pos_encoding_layer = PositionalEncoding(max_position=1, d_model=model_features.shape[-1])
    x = pos_encoding_layer(x)

    # Self-Modeling Component: Meta-Learning Mechanism for Adaptive Weights
    meta_learner = MetaLearningMechanism()
    adjusted_weights, _ = meta_learner(x, gradients=x)

    # Dynamic Self-Modeling Mechanism with Multi-Head Attention
    self_modeling_dense = Dense(d_model, activation='relu')(adjusted_weights)
    attention_output = MultiHeadAttention(num_heads=8, key_dim=d_model)(self_modeling_dense, self_modeling_dense)
    self_modeling_output = Dense(output_dim, name='self_modeling_output')(attention_output)  # Named output for self-modeling

    # Adding the Adaptive Gated SLNN Step Layer
    reservoir_layer = AdaptiveGatedSLNNStep(
        reservoir_dim=512,
        input_dim=adjusted_weights.shape[-1],
        leak_rate=0.5,
        spike_threshold=0.2,
        max_reservoir_dim=512
    )

    reservoir_output, _ = reservoir_layer(adjusted_weights, states=[adjusted_weights])

    # Final Flattening for classification
    x = Flatten()(reservoir_output)
    
    # Final Classification Layers
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    classification_output = Dense(output_dim, activation='softmax', name='classification_output')(x)  # Named output for classification

    # Create the Model with Two Outputs
    model = Model(inputs, [classification_output, self_modeling_output])

    # Compile the Model with a Combined Loss Function and Multiple Metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
        loss={'classification_output': 'categorical_crossentropy', 'self_modeling_output': 'mse'},  # Corrected names
        loss_weights={'classification_output': 1.0, 'self_modeling_output': self_modeling_weight},  # Adjust as needed
        metrics={'classification_output': 'accuracy'}
    )
    
    return model

# Data Loading and Preprocessing
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# Training the Model
def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    model = create_csmt_model(input_shape=(28, 28, 1), output_dim=10)
    
    history = model.fit(
        x_train, 
        [y_train, y_train],  # Replace with the actual targets for both outputs
        validation_data=(x_val, [y_val, y_val]),
        epochs=10,
        batch_size=64
    )
    
    # Evaluate the model
    test_loss, test_accuracy, _ = model.evaluate(x_test, [y_test, y_test])
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')

if __name__ == "__main__":
    main()



# Convolutional Self-Modeling Transformer (CSMT)
# python csmt_mnist.py
# Test Accuracy: 0.9855
