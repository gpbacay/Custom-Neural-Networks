import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, MaxPooling2D, RNN, Reshape
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
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
        self.refractory_period = 5
        self.state_size = max_reservoir_dim
        self.output_size = max_reservoir_dim

    def build(self, input_shape):
        self.initialize_weights()
        super().build(input_shape)

    def initialize_weights(self):
        reservoir_weights = np.random.randn(self.initial_reservoir_size, self.initial_reservoir_size) * 0.1
        reservoir_weights = np.dot(reservoir_weights, reservoir_weights.T)
        spectral_radius = np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
        reservoir_weights *= self.spectral_radius / spectral_radius
        self.reservoir_weights = self.add_weight(
            name='reservoir_weights',
            shape=(self.initial_reservoir_size, self.initial_reservoir_size),
            initializer=tf.constant_initializer(reservoir_weights),
            trainable=False
        )

        input_weights = np.random.randn(self.initial_reservoir_size, self.input_dim) * 0.1
        self.input_weights = self.add_weight(
            name='input_weights',
            shape=(self.initial_reservoir_size, self.input_dim),
            initializer=tf.constant_initializer(input_weights),
            trainable=False
        )

    def call(self, inputs, states):
        prev_state = states[0][:, :tf.shape(self.reservoir_weights)[0]]
        input_contribution = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_contribution = tf.matmul(prev_state, self.reservoir_weights)

        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_contribution + reservoir_contribution)
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)
        refractory_mask = tf.reduce_sum(spikes, axis=1) > self.refractory_period
        state = tf.where(tf.expand_dims(refractory_mask, 1), tf.zeros_like(state), state)
        padded_state = tf.pad(state, [[0, 0], [0, self.max_reservoir_dim - tf.shape(state)[-1]]])

        return padded_state, [padded_state]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if dtype is None:
            dtype = tf.float32
        return [tf.zeros((batch_size, self.max_reservoir_dim), dtype=dtype)]

    def add_synapses(self):
        current_size = tf.shape(self.reservoir_weights)[0]
        growth_rate = tf.maximum(1, tf.cast(tf.math.floor(tf.cast(current_size, tf.float32) * 0.1), tf.int32))
        new_neurons = tf.minimum(growth_rate, self.max_reservoir_dim - current_size)
        
        if new_neurons <= 0:
            return  # No room to grow
        
        new_size = current_size + new_neurons
        
        # Create the new part of the weight matrix (for new synapses)
        new_reservoir_weights = tf.random.normal((new_neurons, new_neurons)) * 0.1
        new_input_weights = tf.random.normal((new_neurons, self.input_dim)) * 0.1

        # Expand the existing reservoir and input weight matrices
        updated_reservoir_weights = tf.concat([
            tf.concat([self.reservoir_weights, tf.zeros((current_size, new_neurons))], axis=1),
            tf.concat([tf.zeros((new_neurons, current_size)), new_reservoir_weights], axis=1)
        ], axis=0)

        # Scale the updated reservoir weights based on spectral radius
        spectral_radius = tf.math.real(tf.reduce_max(tf.abs(tf.linalg.eigvals(updated_reservoir_weights))))
        scaling_factor = self.spectral_radius / spectral_radius
        updated_reservoir_weights *= scaling_factor

        # Concatenate the new input weights to the existing ones
        updated_input_weights = tf.concat([self.input_weights, new_input_weights], axis=0)

        # Now update the weight variables in place
        self.reservoir_weights.assign(updated_reservoir_weights[:self.max_reservoir_dim, :self.max_reservoir_dim])
        self.input_weights.assign(updated_input_weights[:self.max_reservoir_dim, :self.input_dim])


    def prune_synapses(self):
        activity = tf.reduce_mean(tf.abs(self.reservoir_weights), axis=0)
        threshold = np.percentile(activity.numpy(), 10)
        mask = activity > threshold
        pruned_weights = tf.where(tf.tile(mask[None, :], [tf.shape(self.reservoir_weights)[0], 1]), self.reservoir_weights, tf.zeros_like(self.reservoir_weights))
        
        # Update the existing weight variable instead of creating a new one
        self.reservoir_weights.assign(pruned_weights)

class TemporalAttentionLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        weights = tf.reduce_mean(inputs, axis=1, keepdims=True)
        return tf.reduce_sum(inputs * weights, axis=1)

class SelfModelingLayer(tf.keras.layers.Layer):
    def __init__(self, internal_units=128, feedback_strength=0.1, output_dense=4096, **kwargs):
        super().__init__(**kwargs)
        self.internal_units = internal_units
        self.feedback_strength = feedback_strength
        self.state_dense = Dense(internal_units, activation='relu')
        self.gate_dense = Dense(internal_units, activation='sigmoid')
        self.output_dense = Dense(output_dense)  # Match the input dimension

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
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    # Reshape the input for RNN layer
    x = Reshape((1, -1))(x)

    synaptogenesis_layer = SynaptogenesisLayer(
        initial_reservoir_size=initial_reservoir_size,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim
    )
    rnn_layer = RNN(synaptogenesis_layer, return_sequences=True)(x)
    x = TemporalAttentionLayer()(rnn_layer)
    x = SelfModelingLayer(internal_units=128, output_dense=max_reservoir_dim)(x)
    outputs = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, synaptogenesis_layer

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

    model, synaptogenesis_layer = create_csmselnn_model(
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
    synaptogenesis_callback = SynaptogenesisCallback(synaptogenesis_layer)

    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, synaptogenesis_callback]
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
# Test Accuracy: 98.26%