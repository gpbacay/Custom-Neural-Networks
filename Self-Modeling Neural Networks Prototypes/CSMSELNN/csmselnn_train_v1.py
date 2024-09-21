import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from csmselnn_model_v1 import create_csmselnn_model, SelfModelingCallback

def preprocess_data(x):
    return x.astype(np.float32) / 255.0

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    x_train = preprocess_data(x_train).reshape(-1, 28, 28, 1)
    x_val = preprocess_data(x_val).reshape(-1, 28, 28, 1)
    x_test = preprocess_data(x_test).reshape(-1, 28, 28, 1)
    
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
    
    model, selnn_step_layer = create_csmselnn_model(
        input_shape=input_shape,
        initial_reservoir_size=initial_reservoir_size,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim,
        output_dim=output_dim
    )
    
    model.compile(optimizer='adam',
                  loss={'classification_output': 'sparse_categorical_crossentropy', 'self_modeling_output': 'mean_squared_error'},
                  metrics={'classification_output': 'accuracy'})
    
    early_stopping = EarlyStopping(monitor='val_classification_output_accuracy', patience=10, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_classification_output_accuracy', factor=0.1, patience=5, mode='max')
    self_modeling_callback = SelfModelingCallback(
        selnn_step_layer=selnn_step_layer,
        performance_metric='val_classification_output_accuracy',
        target_metric=0.90,
        add_neurons_threshold=add_neurons_threshold,
        prune_connections_threshold=prune_connections_threshold
    )
    
    history = model.fit(
        x_train, 
        {'classification_output': y_train, 'self_modeling_output': x_train.reshape(x_train.shape[0], -1)},
        validation_data=(x_val, {'classification_output': y_val, 'self_modeling_output': x_val.reshape(x_val.shape[0], -1)}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, self_modeling_callback]
    )
    
    test_loss, test_acc = model.evaluate(x_test, {'classification_output': y_test, 'self_modeling_output': x_test.reshape(x_test.shape[0], -1)}, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save the model
    model.save('Trained Models/csmselnn_mnist_v1.keras')
    
    plt.plot(history.history['classification_output_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_classification_output_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

# Convolutional Self-Modeling Spiking Elastic Liquid Neural Network (CSMSELNN)
# python csmselnn_train_v1.py
# Test Accuracy: 0.9904