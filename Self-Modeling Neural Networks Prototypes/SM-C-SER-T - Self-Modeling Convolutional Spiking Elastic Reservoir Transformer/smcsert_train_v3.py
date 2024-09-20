import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from smcsert_model_v3 import create_sm_stc_snn_model

# Preprocess Data
def preprocess_data(x):
    return x.astype(np.float32) / 255.0

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)
    
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
    batch_size = 32
    
    model = create_sm_stc_snn_model(
        input_shape=input_shape,
        initial_reservoir_size=initial_reservoir_size,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim,
        output_dim=output_dim
    )
    
    model.compile(optimizer='adam',
                  loss={'classification_output': 'categorical_crossentropy', 'self_modeling_output': 'mean_squared_error'},
                  metrics={'classification_output': 'accuracy'})
    
    early_stopping = EarlyStopping(monitor='val_classification_output_accuracy', patience=10, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_classification_output_accuracy', factor=0.1, patience=5, mode='max')
    
    history = model.fit(
        x_train, 
        {'classification_output': y_train, 'self_modeling_output': x_train.reshape(x_train.shape[0], -1)},
        validation_data=(x_val, {'classification_output': y_val, 'self_modeling_output': x_val.reshape(x_val.shape[0], -1)}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr]
    )
    
    # Evaluate Model
    test_loss, test_accuracy = model.evaluate(x_test, {'classification_output': y_test, 'self_modeling_output': x_test.reshape(x_test.shape[0], -1)}, verbose=1)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    # Save the model
    model.save('Trained Models/smcsert_mnist_v3.keras')
    
    # Plot Training History
    plt.plot(history.history['classification_output_accuracy'])
    plt.plot(history.history['val_classification_output_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()



# Self-Modeling Convolutional Spiking Elastic Reservoir Transformer (SM-C-SER-T) version 3
# with Summary Mixing Mechanism (alternative for Attention Mechanism)
# python smcsert_train_v3.py
# Test Accuracy: 0.9853