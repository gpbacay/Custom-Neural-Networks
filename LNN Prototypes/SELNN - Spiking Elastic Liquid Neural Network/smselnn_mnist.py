import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Layer
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Custom Keras Layer for Spiking Elastic Liquid Neural Network (SELNN) Step
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
        active_size = tf.shape(state)[-1]
        padded_state = tf.pad(state, [[0, 0], [0, self.max_reservoir_dim - active_size]])
        return padded_state, [padded_state]

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
    def __init__(self, selnn_step_layer, performance_metric='accuracy', target_metric=0.95, add_neurons_threshold=0.01, prune_connections_threshold=0.1):
        super().__init__()
        self.selnn_step_layer = selnn_step_layer
        self.performance_metric = performance_metric
        self.target_metric = target_metric
        self.add_neurons_threshold = add_neurons_threshold
        self.prune_connections_threshold = prune_connections_threshold

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.performance_metric, 0)
        if current_metric >= self.target_metric:
            print(f" - Performance metric {self.performance_metric} reached target {self.target_metric}. Checking for neuron addition or pruning.")
            # Example: Add neurons if the model is performing well
            self.selnn_step_layer.add_neurons(1)  # Example: Add 1 neuron
            
            # Example: Prune connections if the criterion is met
            self.selnn_step_layer.prune_connections(self.prune_connections_threshold)

# Function to Create the SELNN Model
def create_selnn_model(input_dim, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    expanded_inputs = ExpandDimsLayer(axis=1)(inputs)
    
    # Initialize SELNN layer
    selnn_step_layer = SpikingElasticLNNStep(
        initial_reservoir_size=initial_reservoir_size,
        input_dim=input_dim,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim
    )
    
    rnn_layer = tf.keras.layers.RNN(selnn_step_layer, return_sequences=False)
    selnn_output = rnn_layer(expanded_inputs)
    selnn_output = Flatten()(selnn_output)
    
    # Build the rest of the model
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(selnn_output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(output_dim, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model, selnn_step_layer

# Function to Preprocess Data
def preprocess_data(x):
    return x.astype(np.float32) / 255.0

# Main Function to Run the Model
def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    x_train = preprocess_data(x_train).reshape(-1, 28 * 28)
    x_val = preprocess_data(x_val).reshape(-1, 28 * 28)
    x_test = preprocess_data(x_test).reshape(-1, 28 * 28)
    
    # Define model parameters
    input_dim = x_train.shape[1]
    initial_reservoir_size = 1024
    max_reservoir_dim = 8192
    spectral_radius = 0.5
    leak_rate = 0.1
    spike_threshold = 0.5
    output_dim = 10
    epochs = 10
    batch_size = 64
    add_neurons_threshold = 0.1
    prune_connections_threshold = 0.1
    
    # Create and compile the model
    model, selnn_step_layer = create_selnn_model(
        input_dim=input_dim,
        initial_reservoir_size=initial_reservoir_size,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim,
        output_dim=output_dim
    )
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    self_modeling_callback = SelfModelingCallback(
        selnn_step_layer=selnn_step_layer,
        performance_metric='accuracy',
        target_metric=0.95,
        add_neurons_threshold=add_neurons_threshold,
        prune_connections_threshold=prune_connections_threshold
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=3,  # Number of epochs to wait for improvement
        restore_best_weights=True
    )
    
    reduce_lr_on_plateau_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Factor by which to reduce the learning rate
        patience=2,  # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=1e-6  # Minimum learning rate
    )
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[self_modeling_callback, early_stopping_callback, reduce_lr_on_plateau_callback]
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Plot training & validation accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    main()




# Self-Modeling Spiking Elastic Liquid Nueral Network (SMSELNN)
# python smselnn_mnist.py
# Test Accuracy: 0.9147 (fast)