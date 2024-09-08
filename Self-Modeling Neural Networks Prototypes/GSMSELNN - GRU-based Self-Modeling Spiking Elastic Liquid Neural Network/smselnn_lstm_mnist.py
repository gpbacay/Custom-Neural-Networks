import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Custom Keras Layer for Spiking Elastic Liquid Neural Network (SELNN) Step with LSTM-like gating
class SpikingElasticLNNStep(tf.keras.layers.Layer):
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
        self.initialize_weights()

    def initialize_weights(self):
        reservoir_weights = np.random.randn(self.initial_reservoir_size, self.initial_reservoir_size)
        reservoir_weights *= self.spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        
        input_weights = np.random.randn(self.initial_reservoir_size, self.input_dim) * 0.1
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)
        
        # Initialize weights for LSTM-like gates
        self.Wf = self.add_weight(shape=(self.input_dim + self.initial_reservoir_size, self.initial_reservoir_size),
                                  initializer='glorot_uniform', name='Wf')
        self.Wi = self.add_weight(shape=(self.input_dim + self.initial_reservoir_size, self.initial_reservoir_size),
                                  initializer='glorot_uniform', name='Wi')
        self.Wc = self.add_weight(shape=(self.input_dim + self.initial_reservoir_size, self.initial_reservoir_size),
                                  initializer='glorot_uniform', name='Wc')
        self.Wo = self.add_weight(shape=(self.input_dim + self.initial_reservoir_size, self.initial_reservoir_size),
                                  initializer='glorot_uniform', name='Wo')
        
        self.bf = self.add_weight(shape=(self.initial_reservoir_size,), initializer='zeros', name='bf')
        self.bi = self.add_weight(shape=(self.initial_reservoir_size,), initializer='zeros', name='bi')
        self.bc = self.add_weight(shape=(self.initial_reservoir_size,), initializer='zeros', name='bc')
        self.bo = self.add_weight(shape=(self.initial_reservoir_size,), initializer='zeros', name='bo')

    @property
    def state_size(self):
        return (self.max_reservoir_dim, self.max_reservoir_dim)

    def call(self, inputs, states):
        prev_h, prev_c = states
        prev_h = prev_h[:, :tf.shape(self.reservoir_weights)[0]]
        prev_c = prev_c[:, :tf.shape(self.reservoir_weights)[0]]
        
        # Concatenate input and previous hidden state
        concat = tf.concat([inputs, prev_h], axis=-1)
        
        # Forget gate
        f = tf.sigmoid(tf.matmul(concat, self.Wf) + self.bf)
        
        # Input gate
        i = tf.sigmoid(tf.matmul(concat, self.Wi) + self.bi)
        
        # Candidate memory cell
        c_tilde = tf.tanh(tf.matmul(concat, self.Wc) + self.bc)
        
        # Update memory cell
        c = f * prev_c + i * c_tilde
        
        # Output gate
        o = tf.sigmoid(tf.matmul(concat, self.Wo) + self.bo)
        
        # Update hidden state
        h = o * tf.tanh(c)
        
        # Apply spiking mechanism
        spikes = tf.cast(tf.greater(h, self.spike_threshold), dtype=tf.float32)
        h = tf.where(spikes > 0, h - self.spike_threshold, h)
        
        # Apply reservoir dynamics
        reservoir_contribution = tf.matmul(h, self.reservoir_weights)
        h = (1 - self.leak_rate) * h + self.leak_rate * tf.tanh(reservoir_contribution)
        
        active_size = tf.shape(h)[-1]
        padded_h = tf.pad(h, [[0, 0], [0, self.max_reservoir_dim - active_size]])
        padded_c = tf.pad(c, [[0, 0], [0, self.max_reservoir_dim - active_size]])
        
        return padded_h, [padded_h, padded_c]

    def add_neurons(self, new_neurons):
        current_size = tf.shape(self.reservoir_weights)[0]
        new_size = current_size + new_neurons
        if new_size > self.max_reservoir_dim:
            raise ValueError(f"Cannot add {new_neurons} neurons. Max reservoir size is {self.max_reservoir_dim}")

        new_reservoir_weights = tf.random.normal((new_neurons, new_size))
        full_new_weights = tf.concat([
            tf.concat([self.reservoir_weights, tf.zeros((current_size, new_neurons))], axis=1),
            new_reservoir_weights
        ], axis=0)
        spectral_radius = tf.math.real(tf.reduce_max(tf.abs(tf.linalg.eigvals(full_new_weights))))
        scaling_factor = self.spectral_radius / spectral_radius
        new_reservoir_weights *= scaling_factor
        new_input_weights = tf.random.normal((new_neurons, self.input_dim)) * 0.1

        updated_reservoir_weights = tf.concat([self.reservoir_weights, new_reservoir_weights[:, :current_size]], axis=0)
        updated_reservoir_weights = tf.concat([updated_reservoir_weights, 
                                               tf.concat([tf.transpose(new_reservoir_weights[:, :current_size]), 
                                                          new_reservoir_weights[:, current_size:]], axis=0)], axis=1)
        updated_input_weights = tf.concat([self.input_weights, new_input_weights], axis=0)
        self.reservoir_weights = tf.Variable(updated_reservoir_weights, dtype=tf.float32, trainable=False)
        self.input_weights = tf.Variable(updated_input_weights, dtype=tf.float32, trainable=False)
        
        # Update LSTM-like gate weights
        self.Wf = self.add_weight(shape=(self.input_dim + new_size, new_size),
                                  initializer='glorot_uniform', name='Wf')
        self.Wi = self.add_weight(shape=(self.input_dim + new_size, new_size),
                                  initializer='glorot_uniform', name='Wi')
        self.Wc = self.add_weight(shape=(self.input_dim + new_size, new_size),
                                  initializer='glorot_uniform', name='Wc')
        self.Wo = self.add_weight(shape=(self.input_dim + new_size, new_size),
                                  initializer='glorot_uniform', name='Wo')
        
        self.bf = self.add_weight(shape=(new_size,), initializer='zeros', name='bf')
        self.bi = self.add_weight(shape=(new_size,), initializer='zeros', name='bi')
        self.bc = self.add_weight(shape=(new_size,), initializer='zeros', name='bc')
        self.bo = self.add_weight(shape=(new_size,), initializer='zeros', name='bo')

    def prune_connections(self, threshold):
        mask = tf.abs(self.reservoir_weights) > threshold
        self.reservoir_weights.assign(tf.where(mask, self.reservoir_weights, tf.zeros_like(self.reservoir_weights)))

# Custom Layer for Expanding Dimensions
class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

# Callback for Self-Modeling Based on Performance
class SelfModelingCallback(Callback):
    def __init__(self, selnn_step_layer, performance_metric='classification_output_accuracy', target_metric=0.95, add_neurons_threshold=0.01, prune_connections_threshold=0.1):
        super().__init__()
        self.selnn_step_layer = selnn_step_layer
        self.performance_metric = performance_metric
        self.target_metric = target_metric
        self.add_neurons_threshold = add_neurons_threshold
        self.prune_connections_threshold = prune_connections_threshold

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.performance_metric, 0)
        self_modeling_output = logs.get('self_modeling_output_loss', float('inf'))
        if current_metric >= self.target_metric:
            print(f" - Performance metric {self.performance_metric} reached target {self.target_metric}. Checking for neuron addition or pruning.")
            if self_modeling_output < self.add_neurons_threshold:
                self.selnn_step_layer.add_neurons(1)  # Add 1 neuron
                self.selnn_step_layer.prune_connections(self.prune_connections_threshold)

# Function to Create the GSMSELNN Model
def create_gsmselnn_model(input_shape, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim):
    inputs = Input(shape=input_shape)
    
    x = Flatten()(inputs)
    
    # Initialize SELNN layer
    selnn_step_layer = SpikingElasticLNNStep(
        initial_reservoir_size=initial_reservoir_size,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim
    )
    
    expanded_inputs = ExpandDimsLayer(axis=1)(x)
    rnn_layer = tf.keras.layers.RNN(selnn_step_layer, return_sequences=False)
    selnn_output = rnn_layer(expanded_inputs)
    selnn_output = Flatten()(selnn_output)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(selnn_output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    
    # Add self-modeling output
    predicted_hidden = Dense(np.prod(input_shape), name="self_modeling_output")(x)
    
    # Classification output
    outputs = Dense(output_dim, activation='softmax', name="classification_output")(x)
    
    model = tf.keras.Model(inputs, [outputs, predicted_hidden])
    return model, selnn_step_layer

# Function to Preprocess Data
def preprocess_data(x):
    return x.astype(np.float32) / 255.0

# Main Function to Run the Model
def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    x_train = preprocess_data(x_train).reshape(-1, 28, 28, 1)
    x_val = preprocess_data(x_val).reshape(-1, 28, 28, 1)
    x_test = preprocess_data(x_test).reshape(-1, 28, 28, 1)
    
    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Define model parameters
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
    
    # Create and compile the model
    model, selnn_step_layer = create_gsmselnn_model(
        input_shape=input_shape,
        initial_reservoir_size=initial_reservoir_size,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim,
        output_dim=output_dim
    )
    
    model.compile(optimizer='adam',
                  loss={'classification_output': 'categorical_crossentropy', 'self_modeling_output': 'mean_squared_error'},
                  metrics={'classification_output': 'accuracy'})
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_classification_output_accuracy', patience=5, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_classification_output_accuracy', factor=0.1, patience=3, mode='max')
    self_modeling_callback = SelfModelingCallback(
        selnn_step_layer=selnn_step_layer,
        performance_metric='val_classification_output_accuracy',
        target_metric=0.90,
        add_neurons_threshold=add_neurons_threshold,
        prune_connections_threshold=prune_connections_threshold
    )
    
    # Train the model
    history = model.fit(
        x_train, 
        {'classification_output': y_train, 'self_modeling_output': x_train.reshape(x_train.shape[0], -1)},
        validation_data=(x_val, {'classification_output': y_val, 'self_modeling_output': x_val.reshape(x_val.shape[0], -1)}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, self_modeling_callback]
    )
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, {'classification_output': y_test, 'self_modeling_output': x_test.reshape(x_test.shape[0], -1)}, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Plot training history
    plt.plot(history.history['classification_output_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_classification_output_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

# Self-Modeling Spiking Elastic Liquid Neural Network with LSTM-like gating (SMSELNN-LSTM)
# python smselnn_lstm_mnist.py
# Test Accuracy: 0.9851