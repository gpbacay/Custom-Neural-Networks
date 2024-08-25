import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from cstargslt_model import create_cstar_gsl_t_model

def preprocess_data(x):
    # Normalize pixel values to [0, 1]
    x = x.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=-1)  # Add channel dimension

def main():
    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    x_train = preprocess_data(x_train)
    x_val = preprocess_data(x_val)
    x_test = preprocess_data(x_test)

    # Convert class labels to one-hot encoded vectors
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Set hyperparameters
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 channel
    reservoir_dim = 512  # Dimension of the reservoir
    max_reservoir_dim = 1024  # Maximum dimension of the reservoir
    spectral_radius = 1.5  # Spectral radius for reservoir scaling
    leak_rate = 0.3  # Leak rate for state update
    spike_threshold = 0.5  # Threshold for spike generation
    output_dim = 10  # Number of output classes
    num_epochs = 10  # Number of training epochs
    batch_size = 64  # Batch size for training

    # Create the C-STAR-GSL-T model
    model = create_cstar_gsl_t_model(input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim)

    # Define callbacks for early stopping and learning rate reduction
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
    ]

    # Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save the trained model
    model.save('Trained Models/cstargslt_mnist.keras')

if __name__ == "__main__":
    main()



# Convolutional Spatio-Temporal Adaptive Relational Gated Spiking Liquid Transformer (C-STAR-GSL-T)
# python cstargslt_train.py
# Test Accuracy: 0.9926 (Very Impressive)

