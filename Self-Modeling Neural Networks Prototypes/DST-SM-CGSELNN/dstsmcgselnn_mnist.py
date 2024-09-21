import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class HebbianHomeostaticLayer(tf.keras.layers.Layer):
    def __init__(self, units, learning_rate=0.001, target_avg=0.5, homeostatic_rate=0.001, **kwargs):
        super(HebbianHomeostaticLayer, self).__init__(**kwargs)
        self.units = units
        self.learning_rate = learning_rate
        self.target_avg = target_avg
        self.homeostatic_rate = homeostatic_rate
    
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='random_normal',
                                      trainable=True)
    
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
        config = super(HebbianHomeostaticLayer, self).get_config()
        config.update({
            'units': self.units,
            'learning_rate': self.learning_rate,
            'target_avg': self.target_avg,
            'homeostatic_rate': self.homeostatic_rate,
        })
        return config

class GatedSpikingReservoirStep(tf.keras.layers.Layer):
    def __init__(self, spatiotemporal_reservoir_weights, spatiotemporal_input_weights, spiking_gate_weights, leak_rate, spike_threshold, max_dynamic_reservoir_dim, **kwargs):
        super().__init__(**kwargs)
        self.spatiotemporal_reservoir_weights = tf.convert_to_tensor(spatiotemporal_reservoir_weights, dtype=tf.float32)
        self.spatiotemporal_input_weights = tf.convert_to_tensor(spatiotemporal_input_weights, dtype=tf.float32)
        self.spiking_gate_weights = tf.convert_to_tensor(spiking_gate_weights, dtype=tf.float32)
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_dynamic_reservoir_dim = max_dynamic_reservoir_dim
        
        self.state_size = [self.max_dynamic_reservoir_dim]
        self.output_size = self.max_dynamic_reservoir_dim

    def call(self, inputs, states):
        prev_state = states[0]
        prev_state = tf.convert_to_tensor(prev_state, dtype=tf.float32)
        prev_state = prev_state[:, :self.spatiotemporal_reservoir_weights.shape[0]]

        input_part = tf.matmul(inputs, self.spatiotemporal_input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.spatiotemporal_reservoir_weights, transpose_b=True)
        gate_part = tf.matmul(inputs, self.spiking_gate_weights, transpose_b=True)

        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)

        state = (1 - self.leak_rate) * (f_gate * prev_state) + self.leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state

        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_dynamic_reservoir_dim - tf.shape(state)[-1]])], axis=1)

        return padded_state, [padded_state]

    def get_config(self):
        config = super().get_config()
        config.update({
            'spatiotemporal_reservoir_weights': self.spatiotemporal_reservoir_weights.numpy().tolist(),
            'spatiotemporal_input_weights': self.spatiotemporal_input_weights.numpy().tolist(),
            'spiking_gate_weights': self.spiking_gate_weights.numpy().tolist(),
            'leak_rate': self.leak_rate,
            'spike_threshold': self.spike_threshold,
            'max_dynamic_reservoir_dim': self.max_dynamic_reservoir_dim,
        })
        return config

def initialize_spatiotemporal_reservoir(input_dim, reservoir_dim, spectral_radius):
    spatiotemporal_reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    spatiotemporal_reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(spatiotemporal_reservoir_weights)))
    spatiotemporal_input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    spiking_gate_weights = np.random.randn(3 * reservoir_dim, input_dim) * 0.1
    return spatiotemporal_reservoir_weights, spatiotemporal_input_weights, spiking_gate_weights

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
        config = super(SpatioTemporalSummaryMixing, self).get_config()
        config.update({
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
        })
        return config

def create_dst_sm_cgselnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, output_dim):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    summary_mixing_layer = SpatioTemporalSummaryMixing(d_model=128)
    x = Reshape((1, x.shape[-1]))(x)
    x = summary_mixing_layer(x)

    spatiotemporal_reservoir_weights, spatiotemporal_input_weights, spiking_gate_weights = initialize_spatiotemporal_reservoir(x.shape[-1], reservoir_dim, spectral_radius)
    lnn_layer = tf.keras.layers.RNN(
        GatedSpikingReservoirStep(spatiotemporal_reservoir_weights, spatiotemporal_input_weights, spiking_gate_weights, leak_rate, spike_threshold, max_dynamic_reservoir_dim),
        return_sequences=True
    )
    lnn_output = lnn_layer(x)

    # Integrate HebbianHomeostaticLayer
    hebbian_homeostatic_layer = HebbianHomeostaticLayer(units=reservoir_dim, name='hebbian_homeostatic_layer')
    hebbian_output = hebbian_homeostatic_layer(lnn_output)

    # Output Layer
    x = Flatten()(hebbian_output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(output_dim, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def main():
    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Hyperparameters
    input_shape = (28, 28, 1)
    reservoir_dim = 100
    spectral_radius = 0.9
    leak_rate = 0.2
    spike_threshold = 0.5
    max_dynamic_reservoir_dim = reservoir_dim
    output_dim = 10

    # Create model
    model = create_dst_sm_cgselnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, output_dim)

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Train model
    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

    # Evaluate model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()


# Dynamic Spatio-Temporal Self-Modeling Convolutional Gated Spiking Elastic Liquid Neural Network (DST-SM-CGSELNN)
# python dstsmcgselnn_mnist.py
# Test Accuracy: 0.9679