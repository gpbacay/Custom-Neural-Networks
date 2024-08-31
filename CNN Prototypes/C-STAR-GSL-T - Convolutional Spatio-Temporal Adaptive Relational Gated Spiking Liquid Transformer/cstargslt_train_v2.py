import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from cstargslt_model_v2 import create_cstar_gsl_t_model

def preprocess_data(x):
    x = x.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=-1)

def train_and_evaluate():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    x_train = preprocess_data(x_train)
    x_val = preprocess_data(x_val)
    x_test = preprocess_data(x_test)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    input_shape = x_train.shape[1:]
    model = create_cstar_gsl_t_model(
        input_shape=input_shape,
        reservoir_dim=256,
        spectral_radius=1.2,
        leak_rate=0.5,
        spike_threshold=0.8,
        max_reservoir_dim=256,
        output_dim=10
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=128,
        epochs=10
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    model.save('Trained Models/cstargslt_mnist_v2.keras')

if __name__ == "__main__":
    train_and_evaluate()


# Convolutional Spatio-Temporal Adaptive Relational Gated Spiking Liquid Transformer (C-STAR-GSL-T) version 2
# python cstargslt_train_v2.py
# Test Accuracy: 0.9864

