import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, GlobalAveragePooling2D, Reshape, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

class ReservoirComputingLayer(tf.keras.layers.Layer):
    def __init__(self, initial_reservoir_size, input_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, pruning_frequency=5, pruning_rate=0.1, refractory_period=5, **kwargs):
        super().__init__(**kwargs)
        self.initial_reservoir_size = initial_reservoir_size
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_reservoir_dim = max_reservoir_dim
        self.pruning_frequency = pruning_frequency
        self.pruning_rate = pruning_rate
        self.refractory_period = refractory_period
        self.epoch_counter = 0
        self.current_size = initial_reservoir_size
        self.state_size = max_reservoir_dim
        self.reservoir_weights = None
        self.input_weights = None
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize reservoir weights
        reservoir_weights = np.random.randn(self.initial_reservoir_size, self.initial_reservoir_size) * 0.1
        reservoir_weights = np.dot(reservoir_weights, reservoir_weights.T)  # Ensure symmetric matrix
        spectral_radius = np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
        reservoir_weights *= self.spectral_radius / spectral_radius
        self.reservoir_weights = tf.Variable(
            np.pad(reservoir_weights, ((0, self.max_reservoir_dim - self.initial_reservoir_size), 
                                       (0, self.max_reservoir_dim - self.initial_reservoir_size))),
            dtype=tf.float32, trainable=False)

        # Initialize input weights
        input_weights = np.random.randn(self.initial_reservoir_size, self.input_dim) * 0.1
        self.input_weights = tf.Variable(
            np.pad(input_weights, ((0, self.max_reservoir_dim - self.initial_reservoir_size), (0, 0))),
            dtype=tf.float32, trainable=False)

    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, inputs, states):
        prev_state = states[0][:, :self.current_size]
        
        # Input and reservoir contributions
        input_contribution = tf.matmul(inputs, tf.transpose(self.input_weights[:self.current_size]))
        reservoir_contribution = tf.matmul(prev_state, self.reservoir_weights[:self.current_size, :self.current_size])
        
        # Update state using leak rate
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_contribution + reservoir_contribution)
        
        # Spiking mechanism
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)
        
        # Pad state to max reservoir size
        padded_state = tf.pad(state, [[0, 0], [0, self.max_reservoir_dim - tf.shape(state)[1]]])
        
        return padded_state, [padded_state]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros((batch_size, self.max_reservoir_dim), dtype=tf.float32)]

    def add_neurons(self):
        current_size = self.current_size
        growth_rate = max(1, int(current_size * 0.1))  # Add 10% or at least 1 neuron
        new_neurons = min(growth_rate, self.max_reservoir_dim - current_size)
        if current_size + new_neurons > self.max_reservoir_dim:
            return  # No more neurons can be added
        
        # Create new neurons
        new_reservoir_weights = tf.random.normal((new_neurons, self.current_size + new_neurons))
        new_input_weights = tf.random.normal((new_neurons, self.input_dim)) * 0.1
        
        # Update reservoir weights
        self.reservoir_weights = tf.concat([
            tf.concat([self.reservoir_weights[:current_size, :current_size], 
                       tf.zeros((current_size, new_neurons))], axis=1),
            tf.concat([new_reservoir_weights[:, :current_size], 
                       new_reservoir_weights[:, current_size:]], axis=1)], axis=0)

        # Update input weights
        self.input_weights = tf.concat([self.input_weights[:current_size, :], new_input_weights], axis=0)
        
        # Scale new reservoir weights to maintain spectral radius
        spectral_radius = tf.reduce_max(tf.abs(tf.linalg.eigvals(self.reservoir_weights[:self.current_size + new_neurons, 
                                                                                    :self.current_size + new_neurons])))
        scaling_factor = self.spectral_radius / spectral_radius
        self.reservoir_weights = self.reservoir_weights * scaling_factor
        
        # Update current size
        self.current_size += new_neurons

    def prune_connections(self):
        self.epoch_counter += 1
        if self.epoch_counter % self.pruning_frequency == 0:
            # Prune small weights based on the pruning rate
            threshold = np.percentile(np.abs(self.reservoir_weights.numpy()), self.pruning_rate * 100)
            mask = tf.abs(self.reservoir_weights) > threshold
            self.reservoir_weights.assign(tf.where(mask, self.reservoir_weights, tf.zeros_like(self.reservoir_weights)))
            
    def get_config(self):
        config = super().get_config()
        config.update({
            "initial_reservoir_size": self.initial_reservoir_size,
            "input_dim": self.input_dim,
            "spectral_radius": self.spectral_radius,
            "leak_rate": self.leak_rate,
            "spike_threshold": self.spike_threshold,
            "max_reservoir_dim": self.max_reservoir_dim,
            "pruning_frequency": self.pruning_frequency,
            "pruning_rate": self.pruning_rate,
            "refractory_period": self.refractory_period
        })
        return config

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
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
        config = super().get_config()
        config.update({
            "max_position": self.max_position,
            "d_model": self.d_model
        })
        return config

class MultiDimAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MultiDimAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.temporal_dense = tf.keras.layers.Dense(self.channels, activation='sigmoid')
        self.channel_dense = tf.keras.layers.Dense(self.channels, activation='sigmoid')
        self.spatial_dense = tf.keras.layers.Dense(1, activation='sigmoid')
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
    
    def get_config(self):
        config = super().get_config()
        return config

class FeedbackModulationLayer(tf.keras.layers.Layer):
    def __init__(self, internal_units=128, feedback_strength=0.1, output_dense=4096, **kwargs):
        super().__init__(**kwargs)
        self.internal_units = internal_units
        self.feedback_strength = feedback_strength
        self.output_dense_units = output_dense
        self.state_dense = tf.keras.layers.Dense(internal_units, activation='relu')
        self.gate_dense = tf.keras.layers.Dense(internal_units, activation='sigmoid')
        self.output_dense = tf.keras.layers.Dense(output_dense)

    def build(self, input_shape):
        super().build(input_shape)
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
        # Build internal layers
        self.state_dense.build(input_shape)
        self.gate_dense.build(input_shape)
        self.output_dense.build((input_shape[0], self.internal_units))

    def call(self, inputs):
        internal_state = self.state_dense(inputs)
        gate = self.gate_dense(inputs)
        feedback = tf.matmul(internal_state, self.feedback_weights) + self.bias
        modulated_internal = internal_state + self.feedback_strength * gate * feedback
        modulated_output = self.output_dense(modulated_internal)
        return modulated_output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "internal_units": self.internal_units,
            "feedback_strength": self.feedback_strength,
            "output_dense_units": self.output_dense_units
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class SelfModelingCallback(Callback):
    def __init__(self, reservoir_layer, performance_metric='classification_output_accuracy', target_metric=0.95, 
                 add_neurons_threshold=0.01, prune_connections_threshold=0.1, growth_phase_length=10, pruning_phase_length=5, **kwargs):
        super().__init__(**kwargs)
        self.reservoir_layer = reservoir_layer
        self.performance_metric = performance_metric
        self.target_metric = target_metric
        self.initial_add_neurons_threshold = add_neurons_threshold
        self.initial_prune_connections_threshold = prune_connections_threshold
        self.growth_phase_length = growth_phase_length
        self.pruning_phase_length = pruning_phase_length
        self.current_phase = 'growth'
        self.phase_counter = 0
        self.performance_history = []

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.performance_metric, 0)
        self_modeling_output = logs.get('self_modeling_output_loss', float('inf'))
        
        self.performance_history.append(current_metric)
        
        # Adjust thresholds based on current performance
        self.add_neurons_threshold = self.initial_add_neurons_threshold * (1 - current_metric)
        self.prune_connections_threshold = self.initial_prune_connections_threshold * current_metric

        self.phase_counter += 1
        if self.current_phase == 'growth' and self.phase_counter >= self.growth_phase_length:
            self.current_phase = 'pruning'
            self.phase_counter = 0
        elif self.current_phase == 'pruning' and self.phase_counter >= self.pruning_phase_length:
            self.current_phase = 'growth'
            self.phase_counter = 0

        if len(self.performance_history) > 5:
            improvement_rate = (current_metric - self.performance_history[-5]) / 5
            
            if improvement_rate > 0.01:  # Fast improvement
                self.reservoir_layer.add_neurons()  # Add more neurons
            elif improvement_rate < 0.001:  # Slow improvement
                self.reservoir_layer.prune_connections()  # More aggressive pruning

        if current_metric >= self.target_metric:
            print(f" - Performance metric {self.performance_metric} reached target {self.target_metric}. Current phase: {self.current_phase}")
            if self.current_phase == 'growth':
                if self_modeling_output < self.add_neurons_threshold:
                    self.reservoir_layer.add_neurons()
            elif self.current_phase == 'pruning':
                self.reservoir_layer.prune_connections()

class ExpandDimsLayer(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

def create_smcsert_model(input_shape, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim, l2_reg=1e-4):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_regularizer=l2(l2_reg))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1, l2_reg=l2_reg)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2, l2_reg=l2_reg)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2, l2_reg=l2_reg)

    x = GlobalAveragePooling2D()(x)
    x = Reshape((-1, x.shape[-1]))(x)

    x = PositionalEncoding(max_position=x.shape[1], d_model=x.shape[-1])(x)
    x = MultiDimAttention()(x)
    x = Flatten()(x)
    
    # Spiking Elastic LNN Step Layer
    reservoir_layer = ReservoirComputingLayer(
        initial_reservoir_size=initial_reservoir_size,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim
    )
    
    expanded_inputs = ExpandDimsLayer(axis=1)(x)
    rnn_layer = tf.keras.layers.RNN(reservoir_layer, return_sequences=True)
    reservoir_output = rnn_layer(expanded_inputs)
    reservoir_output = Flatten()(reservoir_output)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(reservoir_output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.5)(x)

    x = FeedbackModulationLayer(internal_units=64, output_dense=np.prod(input_shape))(x)

    predicted_hidden = Dense(np.prod(input_shape), name="self_modeling_output")(x)
    outputs = Dense(output_dim, activation='softmax', name="classification_output")(x)

    model = Model(inputs, [outputs, predicted_hidden])
    return model, reservoir_layer

def preprocess_data(x):
    return x.astype(np.float32) / 255.0

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
    add_neurons_threshold = 0.1
    prune_connections_threshold = 0.1
    
    model, reservoir_layer = create_smcsert_model(
        input_shape=input_shape,
        initial_reservoir_size=initial_reservoir_size,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim,
        output_dim=output_dim
    )
    
    model.compile(optimizer='adam',
                  loss={'classification_output': 'sparse_categorical_crossentropy', 'self_modeling_output': 'mean_squared_error'},
                  metrics={'classification_output': 'accuracy'})
    
    early_stopping = EarlyStopping(monitor='val_classification_output_accuracy', patience=10, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_classification_output_accuracy', factor=0.1, patience=5, mode='max')
    self_modeling_callback = SelfModelingCallback(
        reservoir_layer=reservoir_layer,
        performance_metric='val_classification_output_accuracy',
        target_metric=0.90,
        add_neurons_threshold=add_neurons_threshold,
        prune_connections_threshold=prune_connections_threshold
    )
    
    history = model.fit(
        x_train, 
        {'classification_output': y_train, 'self_modeling_output': x_train.reshape(x_train.shape[0], -1)},
        validation_data=(x_val, {'classification_output': y_val, 'self_modeling_output': x_val.reshape(x_val.shape[0], -1)}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, self_modeling_callback]
    )
    
    test_loss, test_acc = model.evaluate(x_test, {'classification_output': y_test, 'self_modeling_output': x_test.reshape(x_test.shape[0], -1)}, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")
    
    plt.plot(history.history['classification_output_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_classification_output_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

# Self-Modeling Convolutional Spiking Elastic Reservoir Transformer (SM-C-SER-T) version 3
# with Positional Encoding and Multi-Dimensional Attention
# python smcsert_mnist_v3.py
# Test Accuracy: 0.9758
