import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from dstsmcgselnn_model import create_dstsmcgselnn_model, SelfModelingCallback

def main():
    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Flatten images for self-modeling task
    x_train_flat = x_train.reshape((x_train.shape[0], -1))
    x_test_flat = x_test.reshape((x_test.shape[0], -1))

    # Hyperparameters
    input_shape = (28, 28, 1)
    reservoir_dim = 100
    spectral_radius = 0.9
    leak_rate = 0.2
    spike_threshold = 0.5
    max_dynamic_reservoir_dim = 1000
    output_dim = 10

    # Create model
    model, reservoir_layer = create_dstsmcgselnn_model(
        input_shape=input_shape,
        reservoir_dim=reservoir_dim,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_dynamic_reservoir_dim=max_dynamic_reservoir_dim,
        output_dim=output_dim
    )

    # Compile the model
    model.compile(
        optimizer='adam', 
        loss={'classification_output': 'categorical_crossentropy', 'self_modeling_output': 'mse'},
        metrics={'classification_output': 'accuracy', 'self_modeling_output': 'mse'}
    )

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_classification_output_accuracy', patience=10, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_classification_output_accuracy', factor=0.1, patience=5, mode='max')
    self_modeling_callback = SelfModelingCallback(
        reservoir_layer=reservoir_layer,
        performance_metric='val_classification_output_accuracy',
        target_metric=0.95,
        add_neurons_threshold=0.01,
        prune_connections_threshold=0.1,
        growth_phase_length=10,
        pruning_phase_length=5
    )

    # Train the model
    history = model.fit(
        x_train, 
        {'classification_output': y_train, 'self_modeling_output': x_train_flat}, 
        validation_data=(x_test, {'classification_output': y_test, 'self_modeling_output': x_test_flat}),
        epochs=10,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr, self_modeling_callback]
    )
    
    # Evaluate the model
    evaluation_results = model.evaluate(x_test, {'classification_output': y_test, 'self_modeling_output': x_test_flat}, verbose=2)
    classification_acc = evaluation_results[1]
    print(f"Test accuracy: {classification_acc:.4f}")
    
    # Save the model
    model.save('Trained Models/dstsmcgselnn_mnist.keras')

    # Plot Training History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['classification_output_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_classification_output_accuracy'], label='Validation Accuracy')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['self_modeling_output_mse'], label='Train MSE')
    plt.plot(history.history['val_self_modeling_output_mse'], label='Validation MSE')
    plt.title('Self-Modeling MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# Dynamic Spatio-Temporal Self-Modeling Convolutional Gated Spiking Elastic Liquid Neural Network (DST-SM-CGSELNN)
# python dstsmcgselnn_train.py
# Test Accuracy: 0.9920