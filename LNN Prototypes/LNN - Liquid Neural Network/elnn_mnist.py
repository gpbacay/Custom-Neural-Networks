import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ElasticLNNStep(layers.Layer):
    def __init__(self, reservoir_weights, input_weights, leak_rate, **kwargs):
        super().__init__(**kwargs)
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)
        self.leak_rate = leak_rate
        self._state_size = reservoir_weights.shape[0]

    @property
    def state_size(self):
        return self._state_size

    def call(self, inputs, states):
        prev_state = states[0][:, :tf.shape(self.reservoir_weights)[0]]
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_part + reservoir_part)
        return state, [state]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros((batch_size, self._state_size), dtype=tf.float32)]

def initialize_lnn_reservoir(input_dim, reservoir_dim, spectral_radius):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights

def create_lnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, output_dim):
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    
    input_dim = x.shape[-1]
    reservoir_weights, input_weights = initialize_lnn_reservoir(input_dim, reservoir_dim, spectral_radius)

    lnn_layer = ElasticLNNStep(
        reservoir_weights=reservoir_weights,
        input_weights=input_weights,
        leak_rate=leak_rate
    )
    
    rnn_layer = layers.RNN(lnn_layer, return_sequences=False)
    lnn_output = rnn_layer(layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x))
    outputs = layers.Dense(output_dim, activation='softmax')(lnn_output)

    model = models.Model(inputs, outputs)
    return model

def preprocess_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, 28*28)).reshape(-1, 28, 28, 1)
    x_val = scaler.transform(x_val.reshape(-1, 28*28)).reshape(-1, 28, 28, 1)
    x_test = scaler.transform(x_test.reshape(-1, 28*28)).reshape(-1, 28, 28, 1)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)
    y_test = tf.keras.utils.to_categorical(y_test)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main():
    # Set hyperparameters
    input_shape = (28, 28, 1)
    reservoir_dim = 1000
    spectral_radius = 1.5
    leak_rate = 0.3
    output_dim = 10
    num_epochs = 10
    batch_size = 64
    pruning_threshold = 0.01

    # Prepare data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_mnist_data()

    # Create Liquid Neural Network (LNN) model
    model = create_lnn_model(
        input_shape=input_shape,
        reservoir_dim=reservoir_dim,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        output_dim=output_dim
    )

    # Compile and train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

    callbacks = [
        early_stopping,
        reduce_lr
    ]

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        x_train, y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=callbacks
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()



# Elastic Liquid Nueral Network (ELNN)
# python elnn_mnist.py
# Test Accuracy: 0.9301 (needs improvements, add neurogenesis and prunning of connections)