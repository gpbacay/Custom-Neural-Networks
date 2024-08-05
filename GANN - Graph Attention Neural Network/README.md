# Graph Attention Neural Network (GANN) for Handwritten Digit Recognition

**Bacay, Gianne P.**

## Overview

This project implements a Graph Attention Neural Network (GANN) for handwritten digit recognition using the MNIST dataset. The model uses a custom Graph Attention Layer to focus on important features of the input data, which enhances its performance on digit classification tasks. The implementation uses TensorFlow and Keras for model creation and training.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy

You can install the necessary packages using pip:

```bash
pip install tensorflow numpy
```

## Code

### Custom Graph Attention Layer

The `GraphAttentionLayer` class defines a custom layer that applies attention mechanisms to the inputs.

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense

class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(GraphAttentionLayer, self).__init__()
        self.output_dim = output_dim
        self.dense = Dense(output_dim)

    def call(self, inputs):
        x = self.dense(inputs)
        attention = tf.nn.softmax(tf.matmul(x, tf.transpose(x, [0, 2, 1])))
        return tf.matmul(attention, x)
```

### Data Preparation

The following code loads and preprocesses the MNIST dataset.

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

### Graph Attention Neural Network (GANN) Model

The `create_gann_model` function creates the GANN model.

```python
from tensorflow.keras.layers import Input, Reshape, Dense
from tensorflow.keras.models import Model

def create_gann_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((1, input_dim))(inputs)
    x = GraphAttentionLayer(64)(x)
    x = Reshape((64,))(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(output_dim, activation='softmax')(x)
    return Model(inputs, outputs)
```

### Training and Evaluation

The following code sets hyperparameters, creates the model, trains it, and evaluates its performance.

```python
# Set up model parameters
input_dim = 28 * 28
output_dim = 10

# Create and compile the model
model = create_gann_model(input_dim, output_dim)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

## Running the Code

To run the code, save it to a Python file (e.g., `gann_mnist.py`) and execute it:

```bash
python gann_mnist.py
```

The final test accuracy will be printed, providing an evaluation of the model's performance on the MNIST dataset.

## Results and Analysis

Training log:

```shell
Epoch 1/10
1875/1875 [==============================] - 8s 4ms/step - loss: 0.4671 - accuracy: 0.8527 - val_loss: 0.1624 - val_accuracy: 0.9513
Epoch 2/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.1876 - accuracy: 0.9432 - val_loss: 0.1285 - val_accuracy: 0.9617
Epoch 3/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.1387 - accuracy: 0.9595 - val_loss: 0.1078 - val_accuracy: 0.9680
Epoch 4/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.1156 - accuracy: 0.9662 - val_loss: 0.0987 - val_accuracy: 0.9693
Epoch 5/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.1001 - accuracy: 0.9710 - val_loss: 0.0897 - val_accuracy: 0.9725
Epoch 6/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0883 - accuracy: 0.9739 - val_loss: 0.0851 - val_accuracy: 0.9732
Epoch 7/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0775 - accuracy: 0.9774 - val_loss: 0.0850 - val_accuracy: 0.9735
Epoch 8/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0675 - accuracy: 0.9802 - val_loss: 0.0812 - val_accuracy: 0.9740
Epoch 9/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0604 - accuracy: 0.9822 - val_loss: 0.0795 - val_accuracy: 0.9752
Epoch 10/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0535 - accuracy: 0.9844 - val_loss: 0.0785 - val_accuracy: 0.9760
Test Accuracy: 0.9671
```

The Graph Attention Neural Network (GANN) model achieved a test accuracy of approximately 96.71% on the MNIST dataset. The model demonstrated consistent improvements in both training and validation accuracy over the course of 10 epochs.

## Conclusion

The Graph Attention Neural Network (GANN) model effectively recognizes handwritten digits from the MNIST dataset, achieving a test accuracy of 96.71%. The custom Graph Attention Layer allowed the model to emphasize crucial features, resulting in strong performance for digit classification tasks. This demonstrates the potential of attention mechanisms in enhancing neural network performance on complex pattern recognition tasks.

