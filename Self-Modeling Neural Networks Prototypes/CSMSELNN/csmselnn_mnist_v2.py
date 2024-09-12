import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Layer, Conv2D, MaxPooling2D, RNN
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SynaptogenesisLayer(tf.keras.layers.Layer):
    def __init__(self, initial_reservoir_size, input_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, **kwargs):
        super().__init__(**kwargs)
        self.initial_reservoir_size = initial_reservoir_size
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_reservoir_dim = max_reservoir_dim
        self.reservoir_weights = None
        self.input_weights = None
        self.refractory_period = 5  # Adding a refractory period
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize reservoir weights with Hebbian-based principles
        reservoir_weights = np.random.randn(self.initial_reservoir_size, self.initial_reservoir_size) * 0.1
        reservoir_weights = np.dot(reservoir_weights, reservoir_weights.T)  # Hebbian-like
        spectral_radius = np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
        reservoir_weights *= self.spectral_radius / spectral_radius
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        
        input_weights = np.random.randn(self.initial_reservoir_size, self.input_dim) * 0.1
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)

    @property
    def state_size(self):
        return (self.max_reservoir_dim,)

    def call(self, inputs, states):
        prev_state = states[0][:, :tf.shape(self.reservoir_weights)[0]]
        input_contribution = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_contribution = tf.matmul(prev_state, self.reservoir_weights)
        
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_contribution + reservoir_contribution)
        
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)
        
        # Apply refractory period
        refractory_mask = tf.reduce_sum(spikes, axis=1) > self.refractory_period
        state = tf.where(tf.expand_dims(refractory_mask, 1), tf.zeros_like(state), state)
        
        active_size = tf.shape(state)[-1]
        padded_state = tf.pad(state, [[0, 0], [0, self.max_reservoir_dim - active_size]])
        return padded_state, [padded_state]

    def add_synapses(self):
        current_size = tf.shape(self.reservoir_weights)[0]
        growth_rate = max(1, int(current_size * 0.1))
        new_neurons = min(growth_rate, self.max_reservoir_dim - current_size)
        if new_neurons <= 0:
            return

        new_size = current_size + new_neurons
        new_reservoir_weights = tf.random.normal((new_size, new_size)) * 0.1
        new_input_weights = tf.random.normal((new_neurons, self.input_dim)) * 0.1

        updated_reservoir_weights = tf.concat([
            tf.concat([self.reservoir_weights, tf.zeros((current_size, new_neurons))], axis=1),
            tf.concat([tf.zeros((new_neurons, current_size)), new_reservoir_weights[current_size:, current_size:]], axis=1)
        ], axis=0)
        
        spectral_radius = tf.math.real(tf.reduce_max(tf.abs(tf.linalg.eigvals(updated_reservoir_weights))))
        scaling_factor = self.spectral_radius / spectral_radius
        updated_reservoir_weights *= scaling_factor
        
        updated_input_weights = tf.concat([self.input_weights, new_input_weights], axis=0)
        
        self.reservoir_weights.assign(updated_reservoir_weights)
        self.input_weights.assign(updated_input_weights)

    def prune_synapses(self):
        # Activity-based pruning
        activity = tf.reduce_mean(tf.abs(self.reservoir_weights), axis=0)
        threshold = np.percentile(activity.numpy(), 10)
        mask = activity > threshold
        self.reservoir_weights.assign(tf.where(tf.tile(mask[None, :], [self.reservoir_weights.shape[0], 1]), self.reservoir_weights, tf.zeros_like(self.reservoir_weights)))

class TemporalAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Simple temporal attention mechanism: weighted sum based on time
        weights = tf.reduce_mean(inputs, axis=1, keepdims=True)
        return tf.reduce_sum(inputs * weights, axis=1)

class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

class SynaptogenesisCallback(Callback):
    def __init__(self, synaptogenesis_layer, performance_metric='val_accuracy', target_metric=0.95,
                 add_synapses_threshold=0.01, prune_synapses_threshold=0.1, growth_phase_length=10, pruning_phase_length=5):
        super().__init__()
        self.synaptogenesis_layer = synaptogenesis_layer
        self.performance_metric = performance_metric
        self.target_metric = target_metric
        self.initial_add_synapses_threshold = add_synapses_threshold
        self.initial_prune_synapses_threshold = prune_synapses_threshold
        self.growth_phase_length = growth_phase_length
        self.pruning_phase_length = pruning_phase_length
        self.current_phase = 'growth'
        self.phase_counter = 0
        self.performance_history = []

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.performance_metric, 0)
        self.performance_history.append(current_metric)
        
        self.add_synapses_threshold = self.initial_add_synapses_threshold * (1 - current_metric)
        self.prune_synapses_threshold = self.initial_prune_synapses_threshold * current_metric

        self.phase_counter += 1
        if self.current_phase == 'growth' and self.phase_counter >= self.growth_phase_length:
            self.current_phase = 'pruning'
            self.phase_counter = 0
        elif self.current_phase == 'pruning' and self.phase_counter >= self.pruning_phase_length:
            self.current_phase = 'growth'
            self.phase_counter = 0

        if len(self.performance_history) > 5:
            improvement_rate = (current_metric - self.performance_history[-5]) / 5
            if improvement_rate > 0.01:
                self.synaptogenesis_layer.add_synapses()
            elif improvement_rate < 0.001:
                self.synaptogenesis_layer.prune_synapses()

        if current_metric >= self.target_metric:
            if self.current_phase == 'growth' and current_metric < self.add_synapses_threshold:
                self.synaptogenesis_layer.add_synapses()
            elif self.current_phase == 'pruning' and current_metric > self.prune_synapses_threshold:
                self.synaptogenesis_layer.prune_synapses()

def create_csmselnn_model(input_shape, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = Flatten()(x)

    synaptogenesis_layer = SynaptogenesisLayer(
        initial_reservoir_size=initial_reservoir_size,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim
    )

    expanded_inputs = ExpandDimsLayer(axis=1)(x)
    rnn_layer = RNN(synaptogenesis_layer, return_sequences=True)
    temporal_attention = TemporalAttentionLayer()(rnn_layer(expanded_inputs))

    x = Dense(output_dim, activation='softmax')(temporal_attention)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model, synaptogenesis_layer

def train_model(model, synaptogenesis_layer, x_train, y_train, x_val, y_val, epochs=10):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callback = SynaptogenesisCallback(
        synaptogenesis_layer=synaptogenesis_layer,
        performance_metric='val_accuracy',
        target_metric=0.95,
        add_synapses_threshold=0.01,
        prune_synapses_threshold=0.1,
        growth_phase_length=10,
        pruning_phase_length=5
    )

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[callback, EarlyStopping(monitor='val_accuracy', patience=3),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)]
    )
    return history

def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Create and train model
    input_shape = x_train.shape[1:]
    model, synaptogenesis_layer = create_csmselnn_model(input_shape, initial_reservoir_size=100, spectral_radius=1.25, 
                                                        leak_rate=0.3, spike_threshold=0.5, max_reservoir_dim=500, output_dim=10)
    history = train_model(model, synaptogenesis_layer, x_train, y_train, x_val, y_val, epochs=10)

    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()





# Convolutional Self-Modeling Spiking Elastic Liquid Neural Network (CSMSELNN) version 2
# python csmselnn_mnist_v2.py
# Test Accuracy: 0.9934