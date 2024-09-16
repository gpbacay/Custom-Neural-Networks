import numpy as np
import tensorflow as tf
from smcsert_model import create_reservoir_cnn_rnn_model

def load_mnist_data():
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, axis=-1)  # Shape (28, 28, 1)
    x_test = np.expand_dims(x_test, axis=-1)
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def train_model():
    input_shape = (28, 28, 1)  # MNIST shape
    num_classes = 10
    initial_reservoir_size = 100
    spectral_radius = 1.25
    leak_rate = 0.1
    spike_threshold = 0.5
    max_reservoir_dim = 500
    l2_reg = 1e-4

    model = create_reservoir_cnn_rnn_model(
        input_shape=input_shape,
        initial_reservoir_size=initial_reservoir_size,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim,
        output_dim=num_classes,
        l2_reg=l2_reg
    )

    model.summary()

    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Training the model
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

    # Save the trained model
    model.save("Trained Models/smcsert_mnist_v2.keras")

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train_model()



# Self-Modeling Convolutional Spiking Elastic Reservoir Transformer (SM-C-SER-T) version 2
# with Positional Encoding and Multi-Dimensional Attention
# python smcsert_train.py
# Test Accuracy: 0.9783