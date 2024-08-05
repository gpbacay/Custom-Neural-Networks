import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

class SpikingLNNLayer(tf.keras.layers.Layer):
    def __init__(self, reservoir_weights, input_weights, leak_rate, spike_threshold=1.0, **kwargs):
        super(SpikingLNNLayer, self).__init__(**kwargs)
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold

    def call(self, x):
        batch_size = tf.shape(x)[0]
        reservoir_dim = self.reservoir_weights.shape[0]
        state = tf.zeros((batch_size, reservoir_dim), dtype=tf.float32)
        
        input_part = tf.matmul(x, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(state, self.reservoir_weights, transpose_b=True)
        state = (1 - self.leak_rate) * state + self.leak_rate * tf.tanh(input_part + reservoir_part)
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)
        return state

    def get_config(self):
        config = super(SpikingLNNLayer, self).get_config()
        config.update({
            "reservoir_weights": self.reservoir_weights.numpy().tolist(),
            "input_weights": self.input_weights.numpy().tolist(),
            "leak_rate": self.leak_rate,
            "spike_threshold": self.spike_threshold,
        })
        return config

    @classmethod
    def from_config(cls, config):
        reservoir_weights = np.array(config.pop("reservoir_weights"))
        input_weights = np.array(config.pop("input_weights"))
        return cls(reservoir_weights, input_weights, **config)

def create_csnlslnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, output_dim):
    inputs = Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    reservoir_weights, input_weights = initialize_reservoir(x.shape[-1], reservoir_dim, spectral_radius)
    
    spiking_lnn_layer = SpikingLNNLayer(reservoir_weights, input_weights, leak_rate)
    lnn_output = spiking_lnn_layer(x)
    lnn_output_reshaped = tf.keras.layers.Reshape((1, -1))(lnn_output)

    x = LSTM(128, return_sequences=True, dropout=0.3)(lnn_output_reshaped)
    x = LSTM(64, dropout=0.3)(x)

    outputs = Dense(output_dim, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def initialize_reservoir(input_dim, reservoir_dim, spectral_radius):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights

def efficientnet_block(inputs, filters, expansion_factor, stride):
    expanded_filters = filters * expansion_factor
    x = tf.keras.layers.Conv2D(expanded_filters, kernel_size=1, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', strides=stride, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if stride == 1 and inputs.shape[-1] == x.shape[-1]:
        x = tf.keras.layers.Add()([inputs, x])
    return x

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train[..., tf.newaxis]
    x_val = x_val[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main():
    input_shape = (28, 28, 1)
    reservoir_dim = 1000
    spectral_radius = 1.5
    leak_rate = 0.3
    output_dim = 10
    num_epochs = 10
    batch_size = 64

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    model = create_csnlslnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, output_dim)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    ]

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        x_train, y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=callbacks
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_accuracy:.4f}')

    if not os.path.exists('TrainedModels'):
        os.makedirs('TrainedModels')
    model.save('TrainedModels/csnlslnn_mnist.keras')

if __name__ == "__main__":
    main()


# python train_csnlslnn.py
# Test Accuracy: 0.9917 (Max)