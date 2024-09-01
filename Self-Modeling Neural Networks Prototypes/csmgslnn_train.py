# train_eval.py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from csmgslnn_model import create_self_modeling_gslnn_model

def preprocess_data(x):
    x = x.astype(np.float32) / 255.0
    return x.reshape(-1, 28, 28, 1)

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = preprocess_data(x_train)
    x_test = preprocess_data(x_test)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    model = create_self_modeling_gslnn_model(
        input_shape=(28, 28, 1),
        initial_reservoir_dim=100,
        max_reservoir_dim=200,
        min_reservoir_dim=50,
        leak_rate=0.1,
        spike_threshold=0.5,
        output_dim=10
    )

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test Accuracy: {test_acc:.4f}')

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    
    # Save Model
    model.save('Trained Models/csmgslnn_mnist.keras')

if __name__ == "__main__":
    main()



# CSM-GSLNN - Convolutional Self-Modeling Gated Spiking Liquid Neural Network
# python csmgslnn_train.py
# Test Accuracy: 0.9905
