import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, GlobalAveragePooling2D, RNN, Reshape, LayerNormalization, BatchNormalization, Add, ReLU
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Simplified Reservoir Computing Layer for CPU Usage
class ReservoirComputingLayer(tf.keras.layers.Layer):
    def __init__(self, initial_reservoir_size, input_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, **kwargs):
        super().__init__(**kwargs)
        self.initial_reservoir_size = initial_reservoir_size
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_reservoir_dim = max_reservoir_dim
        self.refractory_period = 5
        self.state_size = max_reservoir_dim
        self.output_size = max_reservoir_dim
        self.current_size = initial_reservoir_size
        self.learning_rate = 0.01
        self.plasticity_rate = 0.001

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

        # Apply synaptic plasticity (without modifying the weights directly)
        pre_synaptic = tf.expand_dims(prev_state, 2)
        post_synaptic = tf.expand_dims(state, 1)
        weight_change = self.plasticity_rate * (tf.matmul(pre_synaptic, post_synaptic) - self.reservoir_weights[:self.current_size, :self.current_size])
        
        # Instead of updating weights here, we'll return the weight change
        padded_state = tf.pad(state, [[0, 0], [0, self.max_reservoir_dim - tf.shape(state)[1]]])

        return padded_state, [padded_state, weight_change]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if dtype is None:
            dtype = tf.float32
        return [tf.zeros((batch_size, self.max_reservoir_dim), dtype=dtype),
                tf.zeros((batch_size, self.current_size, self.current_size), dtype=dtype)]

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

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_position, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, max_position, d_model):
        angle_rads = self.get_angles(np.arange(max_position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(TemporalAttention, self).__init__()
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

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

# Simplified Self-Modeling Callback
class SelfModelingCallback(Callback):
    def __init__(self, reservoir_layer, performance_metric='classification_output_accuracy', target_metric=0.95, 
                 growth_phase_length=10, pruning_phase_length=5):
        super().__init__()
        self.reservoir_layer = reservoir_layer
        self.performance_metric = performance_metric
        self.target_metric = target_metric
        self.growth_phase_length = growth_phase_length
        self.pruning_phase_length = pruning_phase_length
        self.current_phase = 'growth'
        self.phase_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.performance_metric, 0)
        self.phase_counter += 1

        if self.current_phase == 'growth' and self.phase_counter >= self.growth_phase_length:
            self.current_phase = 'pruning'
            self.phase_counter = 0
        elif self.current_phase == 'pruning' and self.phase_counter >= self.pruning_phase_length:
            self.current_phase = 'growth'
            self.phase_counter = 0

        if current_metric >= self.target_metric:
            if self.current_phase == 'growth':
                self.reservoir_layer._expand_reservoir()
            elif self.current_phase == 'pruning':
                self.reservoir_layer._prune_reservoir()

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

# EfficientNet Block
def efficientnet_block(inputs, filters, expansion_factor, stride, l2_reg=1e-4):
    expanded_filters = filters * expansion_factor
    x = Conv2D(expanded_filters, kernel_size=1, padding='same', use_bias=False, kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(expanded_filters, kernel_size=3, padding='same', strides=stride, use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    if stride == 1 and inputs.shape[-1] == x.shape[-1]:
        x = Add()([inputs, x])
    return x

def create_reservoir_cnn_rnn_model(input_shape, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim, l2_reg=1e-4):
    inputs = Input(shape=input_shape)

    # Convolutional layers process the 2D image
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_regularizer=l2(l2_reg))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1, l2_reg=l2_reg)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2, l2_reg=l2_reg)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2, l2_reg=l2_reg)

    x = GlobalAveragePooling2D()(x)

    # Positional encoding after flattening
    x = Reshape((1, x.shape[-1]))(x)
    x = PositionalEncoding(max_position=1, d_model=x.shape[-1])(x)

    # Temporal Attention Mechanism
    attention_layer = TemporalAttention(d_model=x.shape[-1], num_heads=8)
    attention_output, _ = attention_layer(x, x, x, mask=None)
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)

    # Reservoir Computing Layer inside RNN
    reservoir_layer = ReservoirComputingLayer(
        initial_reservoir_size=initial_reservoir_size,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim
    )
    x = RNN(reservoir_layer, return_sequences=False)(x)
    x = Flatten()(x)

    # Feedback Modulation Layer
    x = FeedbackModulationLayer()(x)
    x = Dropout(0.5)(x)

    predicted_hidden = Dense(np.prod(input_shape), name="self_modeling_output")(x)
    classification_output = Dense(output_dim, activation='softmax', name="classification_output")(x)

    model = Model(inputs, [classification_output, predicted_hidden])
    model.compile(
        optimizer='adam',
        loss={
            'classification_output': 'categorical_crossentropy',
            'self_modeling_output': 'mse'
        },
        loss_weights={
            'classification_output': 1.0,
            'self_modeling_output': 0.1
        },
        metrics={
            'classification_output': 'accuracy'
        }
    )

    return model, reservoir_layer

def preprocess_data(x):
    return x.astype('float32') / 255.0

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    x_train = preprocess_data(x_train).reshape(-1, 28, 28, 1)
    x_val = preprocess_data(x_val).reshape(-1, 28, 28, 1)
    x_test = preprocess_data(x_test).reshape(-1, 28, 28, 1)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

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

    early_stopping = EarlyStopping(monitor='val_classification_output_accuracy', patience=5, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_classification_output_accuracy', factor=0.1, patience=3, mode='max')
    self_modeling_callback = SelfModelingCallback(
        reservoir_layer=reservoir_layer,
        performance_metric='val_classification_output_accuracy',
        target_metric=0.90
    )

    # Create dummy targets for self_modeling_output
    dummy_targets = np.zeros((x_train.shape[0], np.prod(input_shape)))
    dummy_val_targets = np.zeros((x_val.shape[0], np.prod(input_shape)))
    dummy_test_targets = np.zeros((x_test.shape[0], np.prod(input_shape)))

    history = model.fit(
        x_train, 
        {'classification_output': y_train, 'self_modeling_output': dummy_targets},
        validation_data=(x_val, {'classification_output': y_val, 'self_modeling_output': dummy_val_targets}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, self_modeling_callback]
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, {'classification_output': y_test, 'self_modeling_output': dummy_test_targets}, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Plot training history
    plt.plot(history.history['classification_output_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_classification_output_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()



# Self-Modeling Convolutional Spiking Elastic Reservoir Transformer (SM-C-SER-T) version 3
# with Enhanced Self-Modeling Mechanism
# python smcsert_mnist_v3.py
# Test Accuracy: 0.9921
