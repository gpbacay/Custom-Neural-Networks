import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Define Vision Transformer Model
class VisionTransformer(layers.Layer):
    def __init__(self, patch_size, num_patches, embedding_dim, num_heads, num_layers, num_classes):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.projection = layers.Dense(embedding_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=embedding_dim)
        self.encoder_layers = [
            layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embedding_dim // num_heads
            ) for _ in range(num_layers)
        ]
        self.mlp_head = models.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.GlobalAveragePooling1D(),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        x = self.projection(inputs)
        positions = tf.range(start=0, limit=self.num_patches, dtype=tf.int32)
        x += self.position_embedding(positions)
        for layer in self.encoder_layers:
            x = layer(x, x)
        x = self.mlp_head(x)
        return x

# Custom Spiking and Neurogenic LNN Layer
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

def initialize_reservoir(input_dim, reservoir_dim, spectral_radius, max_reservoir_dim):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights

def create_csnllt_model(input_shape, patch_size, embedding_dim, num_heads, num_layers, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim, output_dim):
    inputs = Input(shape=input_shape)
    
    # Vision Transformer part
    num_patches = (input_shape[0] // patch_size[0]) * (input_shape[1] // patch_size[1])
    x = Reshape(target_shape=(num_patches, np.prod(patch_size)))(inputs)
    x_vit = VisionTransformer(
        patch_size=patch_size,
        num_patches=num_patches,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=output_dim
    )(x)
    
    # Convolutional layers
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x_conv = MaxPooling2D((2, 2))(x_conv)
    x_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(x_conv)
    x_conv = MaxPooling2D((2, 2))(x_conv)
    x_conv = Conv2D(128, (3, 3), activation='relu', padding='same')(x_conv)
    x_conv = MaxPooling2D((2, 2))(x_conv)
    x_conv = Flatten()(x_conv)
    
    reservoir_weights, input_weights = initialize_reservoir(128 * 3 * 3, reservoir_dim, spectral_radius, max_reservoir_dim)
    
    lnn_layer = tf.keras.layers.RNN(
        SpikingLNNStep(reservoir_weights, input_weights, leak_rate, max_reservoir_dim),
        return_sequences=True
    )
    
    def apply_spiking_lnn(x):
        lnn_output = lnn_layer(tf.expand_dims(x, axis=1))
        return Flatten()(lnn_output)
    
    lnn_output = Lambda(apply_spiking_lnn)(x_conv)
    lnn_output_reshaped = tf.keras.layers.Reshape((1, -1))(lnn_output)

    # LSTM layers
    x_lstm = LSTM(128, return_sequences=True)(lnn_output_reshaped)
    x_lstm = LSTM(64)(x_lstm)
    
    # Dense layers
    x_dense = Dense(128, activation='relu')(x_lstm)
    x_dense = Dropout(0.5)(x_dense)
    x_dense = Dense(64, activation='relu')(x_dense)
    x_dense = Dropout(0.5)(x_dense)
    outputs = Dense(output_dim, activation='softmax')(x_dense)

    model = models.Model(inputs, outputs)
    return model

# Set hyperparameters
input_shape = (28, 28, 1)
patch_size = (7, 7)
embedding_dim = 64
num_heads = 4
num_layers = 2
reservoir_dim = 100
max_reservoir_dim = 1000
spectral_radius = 1.5
leak_rate = 0.3
output_dim = 10
num_epochs = 10
batch_size = 64

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

x_train = x_train.astype(np.float32) / 255.0
x_val = x_val.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)

# Create and compile CSNLLT Model
csnllt_model = create_csnllt_model(input_shape, patch_size, embedding_dim, num_heads, num_layers, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim, output_dim)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
]

csnllt_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = csnllt_model.fit(
    x_train, y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)

# Evaluate the model on the test set
test_loss, test_acc = csnllt_model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc:.4f}')


# Convolutional Spiking Neurogenic Liquid State LSTM Transformer (CSNLLT) Version 2
# python csnllt_mnist.py
# Test accuracy: 0.9917
