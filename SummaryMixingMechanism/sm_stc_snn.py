import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Reshape, Conv2D, GlobalAveragePooling2D, LSTM
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
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

# Spatio-Temporal Summary Mixing Layer
class SpatioTemporalSummaryMixing(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff=None, dropout_rate=0.1):
        super(SpatioTemporalSummaryMixing, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model

        self.local_dense1 = Dense(self.d_ff, activation='gelu')
        self.local_dense2 = Dense(d_model)
        self.local_dropout = Dropout(dropout_rate)

        self.summary_dense1 = Dense(self.d_ff, activation='gelu')
        self.summary_dense2 = Dense(d_model)
        self.summary_dropout = Dropout(dropout_rate)

        self.combiner_dense1 = Dense(self.d_ff, activation='gelu')
        self.combiner_dense2 = Dense(d_model)
        self.combiner_dropout = Dropout(dropout_rate)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

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

        return self.layer_norm(inputs + output)

# Spiking Elastic Liquid Neural Network (SELNN) Layer
class SpikingElasticLNNStep(tf.keras.layers.Layer):
    def __init__(self, initial_reservoir_size, input_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, **kwargs):
        super().__init__(**kwargs)
        self.initial_reservoir_size = initial_reservoir_size
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_reservoir_dim = max_reservoir_dim

        self.state_size = [self.max_reservoir_dim]

        self.reservoir_weights = None
        self.input_weights = None
        self.initialize_weights()

    def initialize_weights(self):
        reservoir_weights = np.random.randn(self.initial_reservoir_size, self.initial_reservoir_size)
        reservoir_weights *= self.spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)

        input_weights = np.random.randn(self.initial_reservoir_size, self.input_dim) * 0.1
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)

    def call(self, inputs, states):
        prev_state = states[0][:, :tf.shape(self.reservoir_weights)[0]]
        input_contribution = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_contribution = tf.matmul(prev_state, self.reservoir_weights)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_contribution + reservoir_contribution)
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)
        active_size = tf.shape(state)[-1]
        padded_state = tf.pad(state, [[0, 0], [0, self.max_reservoir_dim - active_size]])
        return padded_state, [padded_state]

# Combining Spatio-Temporal Mixing with SELNN
def create_combined_model(input_shape, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim):
    inputs = Input(shape=input_shape)

    # Convolutional layers for feature extraction
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1, l2_reg=1e-4)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2, l2_reg=1e-4)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2, l2_reg=1e-4)

    x = GlobalAveragePooling2D()(x)

    # Reshape to add time dimension
    x = Reshape((1, -1))(x)

    # Spatio-Temporal Summary Mixing Layer
    x = SpatioTemporalSummaryMixing(d_model=40)(x)

    # SELNN Layer
    selnn_step_layer = SpikingElasticLNNStep(
        initial_reservoir_size=initial_reservoir_size,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim
    )

    rnn_layer = tf.keras.layers.RNN(selnn_step_layer, return_sequences=True)
    selnn_output = rnn_layer(x)
    
    # Flatten the output
    x = Flatten()(selnn_output)

    # Dense layers for classification
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)

    # Multi-scale self-modeling mechanism (iterative refinement)
    x_reshaped = Reshape((1, 64))(x)
    predicted_hidden = LSTM(64, return_sequences=False)(x_reshaped)
    predicted_hidden = Dense(np.prod(input_shape), activation='linear')(predicted_hidden)
    predicted_hidden = Reshape(input_shape, name="predicted_hidden")(predicted_hidden)

    # Final classification output
    classification_output = Dense(output_dim, activation='softmax', name="classification_output")(x)

    # Self-modeling output
    self_modeling_output = Dense(np.prod(input_shape), activation='sigmoid', name="self_modeling_dense")(x)
    self_modeling_output = Reshape(input_shape, name="self_modeling_output")(self_modeling_output)

    model = tf.keras.Model(inputs, [classification_output, self_modeling_output])
    return model

# Preprocess Data
def preprocess_data(x):
    return x.astype('float32') / 255.0

def main():
    # Load your dataset here
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(preprocess_data(x_train), -1)
    x_test = np.expand_dims(preprocess_data(x_test), -1)
    
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Split for validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    input_shape = x_train.shape[1:]
    output_dim = 10  # Number of classes

    model = create_combined_model(
        input_shape=input_shape,
        initial_reservoir_size=512,
        spectral_radius=0.9,
        leak_rate=0.1,
        spike_threshold=0.5,
        max_reservoir_dim=4096,
        output_dim=output_dim
    )

    model.compile(optimizer='adam', 
            loss={'classification_output': 'categorical_crossentropy', 'self_modeling_output': 'mse'},
            metrics={
                'classification_output': 'accuracy',
                'self_modeling_output': 'mse'
            })

    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(patience=3)

    history = model.fit(x_train, 
                    {'classification_output': y_train, 'self_modeling_output': x_train},
                    validation_data=(x_val, {'classification_output': y_val, 'self_modeling_output': x_val}),
                    epochs=10,
                    batch_size=64,
                    callbacks=[early_stopping, reduce_lr])

    # Evaluate the model
    test_results = model.evaluate(x_test, {'classification_output': y_test, 'self_modeling_dense': x_test}, verbose=0)
    print(f'Test loss: {test_results[0]:.4f}, Test accuracy: {test_results[1]:.4f}')

    # Plot training history
    plt.plot(history.history['classification_output_accuracy'], label='train accuracy')
    plt.plot(history.history['val_classification_output_accuracy'], label='val accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()




# Self-Modeling Spatio-Temporal Convolutional Spiking Neural Network (SM-STC-SNN)
# with Summary Mixing Mechanism (alternative for Attention Mechanism)
# python sm_stc_snn.py
# Test Accuracy: 0.9926