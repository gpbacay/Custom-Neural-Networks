import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from dstrctgsr_model_v2 import create_dstr_ct_gsr_model

def main():
    # Load and Prepare MNIST Data for Spatio-Temporal Processing
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    input_shape = x_train.shape[1:]  # (28, 28, 1)
    reservoir_dim = 256
    max_dynamic_reservoir_dim = 512
    spectral_radius = 1.5
    leak_rate = 0.3
    spike_threshold = 0.5
    output_dim = 10
    l2_reg = 1e-4

    model = create_dstr_ct_gsr_model(input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, output_dim, l2_reg=l2_reg)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
    
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64, callbacks=[early_stopping, reduce_lr])

    # Evaluate the model and print the final test accuracy
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    
    # Save Model
    model.save('Trained Models/dstrctgsr_mnist_v2.keras')
    
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()

# Dynamic Spatio-Tempo-Relational Convolutional Transformer with Gated Spiking Reservoir (DSTR-CT-GSR) version 2
# python dstrctgsr_train_v2.py
# Test Accuracy: 0.9948