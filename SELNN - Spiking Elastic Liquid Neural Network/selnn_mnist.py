import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Custom Layer: Spiking Elastic Liquid Neural Network Step
class SpikingElasticLNNStep(tf.keras.layers.Layer):
    def __init__(self, reservoir_weights, input_weights, leak_rate, spike_threshold, max_reservoir_dim, **kwargs):
        super().__init__(**kwargs)
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_reservoir_dim = max_reservoir_dim

    @property
    def state_size(self):
        return (self.max_reservoir_dim,)

    def call(self, inputs, states):
        prev_state = states[0][:, :self.reservoir_weights.shape[0]]
        
        # Compute contributions from input and reservoir
        input_contribution = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_contribution = tf.matmul(prev_state, self.reservoir_weights, transpose_b=True)

        # Update state with spiking dynamics
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_contribution + reservoir_contribution)

        # Apply spiking threshold
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        # Pad state to match maximum reservoir dimension
        active_size = tf.shape(state)[-1]
        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_reservoir_dim - active_size])], axis=1)
        
        return padded_state, [padded_state]

    def add_neurons(self, new_neurons):
        new_weights = np.random.randn(new_neurons, self.reservoir_weights.shape[1])
        self.reservoir_weights.assign(tf.concat([self.reservoir_weights, tf.constant(new_weights, dtype=tf.float32)], axis=0))

    def prune_connections(self, threshold):
        mask = tf.abs(self.reservoir_weights) > threshold
        self.reservoir_weights.assign(tf.where(mask, self.reservoir_weights, tf.zeros_like(self.reservoir_weights)))

# Function to initialize reservoir and input weights
def initialize_weights(input_dim, reservoir_dim, spectral_radius):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1

    return reservoir_weights, input_weights

# Function to create the Spiking Elastic Liquid Neural Network (SELNN) model
def create_selnn_model(input_dim, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    
    reservoir_weights, input_weights = initialize_weights(input_dim, reservoir_dim, spectral_radius)

    selnn_layer = tf.keras.layers.RNN(
        SpikingElasticLNNStep(reservoir_weights, input_weights, leak_rate, spike_threshold, max_reservoir_dim),
        return_sequences=True
    )

    def apply_selnn(x):
        selnn_output = selnn_layer(tf.expand_dims(x, axis=1))
        return Flatten()(selnn_output)

    selnn_output = Lambda(apply_selnn)(inputs)

    x = Dense(128, activation='relu')(selnn_output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(output_dim, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

# Function to preprocess MNIST data
def preprocess_data(x):
    return x.astype(np.float32) / 255.0

def main():
    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    x_train = preprocess_data(x_train).reshape(-1, 28 * 28)
    x_val = preprocess_data(x_val).reshape(-1, 28 * 28)
    x_test = preprocess_data(x_test).reshape(-1, 28 * 28)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)
    y_test = tf.keras.utils.to_categorical(y_test)

    # Set hyperparameters
    input_dim = 28 * 28
    reservoir_dim = 500
    max_reservoir_dim = 1000
    spectral_radius = 1.5
    leak_rate = 0.3
    spike_threshold = 0.5
    output_dim = 10
    num_epochs = 10
    batch_size = 64

    # Create the Spiking Elastic Liquid Neural Network (SELNN) model
    model = create_selnn_model(input_dim, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim)

    # Define callbacks for early stopping and learning rate reduction
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
    ]

    # Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Display the model summary
    model.summary()

if __name__ == "__main__":
    main()


# Spiking Elastic Liquid Nueral Network (SELNN)
# python selnn_mnist.py
# Test Accuracy: 0.9694