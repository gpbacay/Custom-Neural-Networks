import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from dstsmcgselnn_model import create_dst_sm_cgselnn_model

def main():
    # Load and Prepare MNIST Data for Spatio-Temporal Processing
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    # Model Parameters
    input_shape = x_train.shape[1:]
    output_dim = 10  # Number of classes for MNIST
    reservoir_dim = 256
    spectral_radius = 1.25
    leak_rate = 0.1
    spike_threshold = 0.5
    max_dynamic_reservoir_dim = 512
    d_model = 64

    # Create Model
    model = create_dst_sm_cgselnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, spike_threshold, max_dynamic_reservoir_dim, output_dim, d_model)

    # Compile Model
    model.compile(optimizer='adam', 
                  loss={'main_output': 'categorical_crossentropy', 'self_modeling_output': 'categorical_crossentropy'},
                  metrics={'main_output': 'accuracy', 'self_modeling_output': 'accuracy'})

    # Define Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Train Model
    history = model.fit(x_train, 
                        {'main_output': y_train, 'self_modeling_output': y_train}, 
                        validation_data=(x_test, {'main_output': y_test, 'self_modeling_output': y_test}), 
                        epochs=10, batch_size=64, 
                        callbacks=[early_stopping, reduce_lr])

    # Evaluate Model
    test_loss, test_acc_main, test_acc_self = model.evaluate(x_test, {'main_output': y_test, 'self_modeling_output': y_test}, verbose=2)
    print(f'Test accuracy (main output): {test_acc_main:.4f}')
    print(f'Test accuracy (self-modeling output): {test_acc_self:.4f}')
    
    # Save the model
    model.save('Trained Models/dstsmcgselnn_mnist.keras')

    # Plot Training History
    plt.plot(history.history['main_output_accuracy'], label='Main Output Accuracy')
    plt.plot(history.history['self_modeling_output_accuracy'], label='Self-Modeling Output Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()


# Dynamic Spatio-Temporal Self-Modeling Convolutional Gated Spiking Elastic Liquid Neural Network (DST-SM-CGSELNN)
# python dstsmcgselnn_mnist.py
# Test Accuracy: 0.9942