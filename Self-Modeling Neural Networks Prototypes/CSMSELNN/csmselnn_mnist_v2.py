import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, MaxPooling2D, RNN, Reshape
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ReservoirComputingLayer(tf.keras.layers.Layer):
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
        self.refractory_period = 5
        self.state_size = max_reservoir_dim
        self.output_size = max_reservoir_dim
        self.current_size = initial_reservoir_size

    def build(self, input_shape):
        self._initialize_weights()
        super().build(input_shape)

    def _initialize_weights(self):
        reservoir_weights = np.random.randn(self.initial_reservoir_size, self.initial_reservoir_size) * 0.1
        reservoir_weights = np.dot(reservoir_weights, reservoir_weights.T)
        spectral_radius = np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
        reservoir_weights *= self.spectral_radius / spectral_radius
        self.reservoir_weights = self.add_weight(
            name='reservoir_weights',
            shape=(self.max_reservoir_dim, self.max_reservoir_dim),
            initializer=tf.constant_initializer(np.pad(reservoir_weights, ((0, self.max_reservoir_dim - self.initial_reservoir_size), (0, self.max_reservoir_dim - self.initial_reservoir_size)))),
            trainable=False
        )

        input_weights = np.random.randn(self.initial_reservoir_size, self.input_dim) * 0.1
        self.input_weights = self.add_weight(
            name='input_weights',
            shape=(self.max_reservoir_dim, self.input_dim),
            initializer=tf.constant_initializer(np.pad(input_weights, ((0, self.max_reservoir_dim - self.initial_reservoir_size), (0, 0)))),
            trainable=False
        )

    def call(self, inputs, states):
        prev_state = states[0][:, :self.current_size]
        input_contribution = tf.matmul(inputs, tf.transpose(self.input_weights[:self.current_size]))
        reservoir_contribution = tf.matmul(prev_state, self.reservoir_weights[:self.current_size, :self.current_size])

        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_contribution + reservoir_contribution)
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)
        refractory_mask = tf.reduce_sum(spikes, axis=1) > self.refractory_period
        state = tf.where(tf.expand_dims(refractory_mask, 1), tf.zeros_like(state), state)
        padded_state = tf.pad(state, [[0, 0], [0, self.max_reservoir_dim - tf.shape(state)[1]]])

        return padded_state, [padded_state]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if dtype is None:
            dtype = tf.float32
        return [tf.zeros((batch_size, self.max_reservoir_dim), dtype=dtype)]

    def _expand_reservoir(self):
        growth_rate = tf.maximum(1, tf.cast(tf.math.floor(tf.cast(self.current_size, tf.float32) * 0.1), tf.int32))
        new_neurons = tf.minimum(growth_rate, self.max_reservoir_dim - self.current_size)
        
        if new_neurons <= 0:
            return  # No room to grow
        
        new_size = self.current_size + new_neurons
        
        # Create new weights for the expanded part
        new_weights = tf.random.normal((new_neurons, new_size)) * 0.1
        new_weights = tf.concat([tf.zeros((new_neurons, self.current_size)), new_weights[:, self.current_size:]], axis=1)
        
        # Update reservoir weights
        updated_weights = tf.concat([
            self.reservoir_weights[:self.current_size, :self.current_size],
            tf.random.normal((self.current_size, new_neurons)) * 0.1
        ], axis=1)
        updated_weights = tf.concat([updated_weights, new_weights], axis=0)
        
        # Ensure symmetry
        updated_weights = (updated_weights + tf.transpose(updated_weights)) / 2
        
        # Adjust spectral radius
        spectral_radius = tf.math.real(tf.reduce_max(tf.abs(tf.linalg.eigvals(updated_weights))))
        scaling_factor = self.spectral_radius / spectral_radius
        updated_weights *= scaling_factor

        # Update input weights
        new_input_weights = tf.concat([
            self.input_weights[:self.current_size],
            tf.random.normal((new_neurons, self.input_dim)) * 0.1
        ], axis=0)

        # Assign new weights
        self.reservoir_weights.assign(tf.pad(updated_weights, [[0, self.max_reservoir_dim - new_size], [0, self.max_reservoir_dim - new_size]]))
        self.input_weights.assign(tf.pad(new_input_weights, [[0, self.max_reservoir_dim - new_size], [0, 0]]))
        self.current_size = new_size

    def _prune_reservoir(self):
        if self.current_size <= self.initial_reservoir_size:
            return  # Don't prune below the initial size

        activity = tf.reduce_mean(tf.abs(self.reservoir_weights[:self.current_size, :self.current_size]), axis=0)
        k = tf.cast(tf.cast(self.current_size, tf.float32) * 0.9, tf.int32)  # Ensure k is an integer
        _, indices = tf.nn.top_k(activity, k=k)
        indices = tf.sort(indices)

        pruned_reservoir_weights = tf.gather(tf.gather(self.reservoir_weights[:self.current_size, :self.current_size], indices), indices, axis=1)
        pruned_input_weights = tf.gather(self.input_weights[:self.current_size], indices)

        new_size = tf.shape(pruned_reservoir_weights)[0]
        self.reservoir_weights.assign(tf.pad(pruned_reservoir_weights, [[0, self.max_reservoir_dim - new_size], [0, self.max_reservoir_dim - new_size]]))
        self.input_weights.assign(tf.pad(pruned_input_weights, [[0, self.max_reservoir_dim - new_size], [0, 0]]))
        self.current_size = new_size

class TemporalAttentionAggregator(tf.keras.layers.Layer):
    def call(self, inputs):
        weights = tf.reduce_mean(inputs, axis=1, keepdims=True)
        return tf.reduce_sum(inputs * weights, axis=1)

class FeedbackModulationLayer(tf.keras.layers.Layer):
    def __init__(self, internal_units=128, feedback_strength=0.1, output_dense=4096, **kwargs):
        super().__init__(**kwargs)
        self.internal_units = internal_units
        self.feedback_strength = feedback_strength
        self.state_dense = Dense(internal_units, activation='relu')
        self.gate_dense = Dense(internal_units, activation='sigmoid')
        self.output_dense = Dense(output_dense)

    def build(self, input_shape):
        self.feedback_weights = self.add_weight(
            shape=(self.internal_units, self.internal_units),
            initializer='random_normal',
            trainable=True,
            name='feedback_weights'
        )
        self.bias = self.add_weight(
            shape=(self.internal_units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

    def call(self, inputs):
        internal_state = self.state_dense(inputs)
        gate = self.gate_dense(inputs)
        feedback = tf.matmul(internal_state, self.feedback_weights) + self.bias
        modulated_internal = internal_state + self.feedback_strength * gate * feedback
        modulated_output = self.output_dense(modulated_internal)
        return modulated_output

class DynamicReservoirGrowthCallback(Callback):
    def __init__(self, reservoir_layer, performance_metric='val_accuracy', target_metric=0.95,
                 add_synapses_threshold=0.01, prune_synapses_threshold=0.1, growth_phase_length=10, pruning_phase_length=5):
        super().__init__()
        self.reservoir_layer = reservoir_layer
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
                self.reservoir_layer._expand_reservoir()
            elif improvement_rate < 0.001:
                self.reservoir_layer._prune_reservoir()

        if current_metric >= self.target_metric:
            if self.current_phase == 'growth' and current_metric < self.add_synapses_threshold:
                self.reservoir_layer._expand_reservoir()
            elif self.current_phase == 'pruning' and current_metric > self.prune_synapses_threshold:
                self.reservoir_layer._prune_reservoir()

def create_reservoir_cnn_rnn_model(input_shape, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    x = Reshape((1, -1))(x)

    reservoir_layer = ReservoirComputingLayer(
        initial_reservoir_size=initial_reservoir_size,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim
    )
    rnn_layer = RNN(reservoir_layer, return_sequences=True)(x)
    x = TemporalAttentionAggregator()(rnn_layer)
    x = FeedbackModulationLayer(internal_units=128, output_dense=max_reservoir_dim)(x)
    outputs = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, reservoir_layer

def preprocess_data(data):
    return data.astype('float32') / 255.0

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    x_train = preprocess_data(x_train).reshape(-1, 28, 28, 1)
    x_val = preprocess_data(x_val).reshape(-1, 28, 28, 1)
    x_test = preprocess_data(x_test).reshape(-1, 28, 28, 1)

    input_shape = (28, 28, 1)
    initial_reservoir_size = 512
    max_reservoir_dim = 4096
    spectral_radius = 0.5
    leak_rate = 0.1
    spike_threshold = 0.5
    output_dim = 10
    epochs = 10
    batch_size = 64

    model, reservoir_layer = create_reservoir_cnn_rnn_model(
        input_shape=input_shape,
        initial_reservoir_size=initial_reservoir_size,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim,
        output_dim=output_dim
    )

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3)
    dynamic_reservoir_growth_callback = DynamicReservoirGrowthCallback(reservoir_layer)

    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, dynamic_reservoir_growth_callback]
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()



# Convolutional Self-Modeling Spiking Elastic Liquid Neural Network (CSMSELNN) version 2
# python csmselnn_mnist_v2.py
# Test Accuracy: 98.74%