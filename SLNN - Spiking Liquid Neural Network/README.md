# Spiking Liquid Neural Network (SNLNN) for Handwritten Digit Recognition

###### Bacay, Gianne P.

## Overview

This project implements a Spiking Liquid Neural Network (SNLNN) for handwritten digit recognition using the MNIST dataset. The model incorporates a custom Spiking LNN layer with spiking dynamics to simulate neuronal activity. The implementation uses TensorFlow and Keras.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- scikit-learn

You can install the necessary packages using pip:

```bash
pip install numpy tensorflow scikit-learn
```

## Code

### Custom Spiking LNN Layer

The `SpikingLNNStep` class defines a custom LNN layer with spiking dynamics. It simulates the behavior of neurons that generate discrete spikes when their state exceeds a certain threshold.


```python

class SpikingLNNStep(tf.keras.layers.Layer):
    def __init__(self, reservoir_weights, input_weights, leak_rate, max_reservoir_dim, spike_threshold=1.0, **kwargs):
        super(SpikingLNNStep, self).__init__(**kwargs)
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32)
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32)
        self.leak_rate = leak_rate
        self.max_reservoir_dim = max_reservoir_dim
        self.spike_threshold = spike_threshold

    @property
    def state_size(self):
        return (self.max_reservoir_dim,)

    def call(self, inputs, states):
        prev_state = states[0][:, :self.reservoir_weights.shape[0]]
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights, transpose_b=True)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_part + reservoir_part)

        # Spiking dynamics: Apply a threshold to produce discrete spikes
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        # Ensure the state size matches the max reservoir size
        active_size = self.reservoir_weights.shape[0]
        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_reservoir_dim - active_size])], axis=1)
        return padded_state, [padded_state]

```

### Initialization Function

The `initialize_spiking_lnn_reservoir` function initializes the reservoir and input weights for the Spiking LNN layer.

```python

def initialize_spiking_lnn_reservoir(input_dim, reservoir_dim, spectral_radius, max_reservoir_dim):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights, max_reservoir_dim

```

### Model Creation Function

The `create_spiking_nlnn_model` function creates the SNLNN model with the custom Spiking LNN layer.

```python

def create_spiking_nlnn_model(input_dim, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim, output_dim):
    inputs = Input(shape=(input_dim,))

    # Initialize Spiking LNN weights
    reservoir_weights, input_weights, max_reservoir_dim = initialize_spiking_lnn_reservoir(input_dim, reservoir_dim, spectral_radius, max_reservoir_dim)

    # Spiking LNN Layer
    lnn_layer = tf.keras.layers.RNN(SpikingLNNStep(reservoir_weights, input_weights, leak_rate, max_reservoir_dim), return_sequences=True)
    lnn_output = lnn_layer(tf.expand_dims(inputs, axis=1))

    # Flatten the LNN output
    lnn_output = tf.keras.layers.Flatten()(lnn_output)

    # Output Layer
    outputs = Dense(output_dim, activation='softmax')(lnn_output)

    model = keras.Model(inputs, outputs)
    return model
```

### Data Preparation

The following code loads and preprocesses the MNIST dataset.

```python

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Normalize the data
def normalize_data(x):
    num_samples, height, width = x.shape
    x = x.reshape(-1, width)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x.reshape(num_samples, height, width)

x_train = normalize_data(x_train)
x_val = normalize_data(x_val)
x_test = normalize_data(x_test)

# Flatten the images for the dense layer
x_train = x_train.reshape(-1, 28 * 28)
x_val = x_val.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train)
y_val = keras.utils.to_categorical(y_val)
y_test = keras.utils.to_categorical(y_test)
```

### Training and Evaluation

The following code sets hyperparameters, creates the model, and trains it.

```python

# Set hyperparameters
input_dim = 28 * 28
reservoir_dim = 100
max_reservoir_dim = 1000
spectral_radius = 1.5
leak_rate = 0.3
output_dim = 10
num_epochs = 10

# Create Spiking Liquid Neural Network (SNLNN) model
model = create_spiking_nlnn_model(input_dim, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim, output_dim)

# Compile and train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=num_epochs, batch_size=64, validation_data=(x_val, y_val), callbacks=[early_stopping, reduce_lr])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")

```

## Running the Code

To run the code, save it to a Python file (e.g., `slnn_mnist.py`) and execute it:

```bash
python slnn_mnist.py
```

The final test accuracy will be printed, providing an evaluation of the model's performance on the MNIST dataset.

### Results and Analysis

The following is the training log for the Spiking Liquid Neural Network (SNLNN) on the MNIST dataset:

```bash

Epoch 1/10
844/844 [==============================] - 4s 3ms/step - loss: 0.5455 - accuracy: 0.8801 - val_loss: 0.2560 - val_accuracy: 0.9290 - lr: 0.0010
Epoch 2/10
844/844 [==============================] - 2s 2ms/step - loss: 0.2165 - accuracy: 0.9396 - val_loss: 0.1984 - val_accuracy: 0.9412 - lr: 0.0010
Epoch 3/10
844/844 [==============================] - 2s 3ms/step - loss: 0.1654 - accuracy: 0.9529 - val_loss: 0.1693 - val_accuracy: 0.9483 - lr: 0.0010
Epoch 4/10
844/844 [==============================] - 2s 2ms/step - loss: 0.1342 - accuracy: 0.9628 - val_loss: 0.1517 - val_accuracy: 0.9542 - lr: 0.0010
Epoch 5/10
844/844 [==============================] - 2s 2ms/step - loss: 0.1116 - accuracy: 0.9695 - val_loss: 0.1428 - val_accuracy: 0.9573 - lr: 0.0010
Epoch 6/10
844/844 [==============================] - 2s 2ms/step - loss: 0.0946 - accuracy: 0.9746 - val_loss: 0.1336 - val_accuracy: 0.9605 - lr: 0.0010
Epoch 7/10
844/844 [==============================] - 2s 3ms/step - loss: 0.0810 - accuracy: 0.9789 - val_loss: 0.1281 - val_accuracy: 0.9635 - lr: 0.0010
Epoch 8/10
844/844 [==============================] - 2s 2ms/step - loss: 0.0691 - accuracy: 0.9822 - val_loss: 0.1229 - val_accuracy: 0.9625 - lr: 0.0010
Epoch 9/10
844/844 [==============================] - 2s 3ms/step - loss: 0.0594 - accuracy: 0.9857 - val_loss: 0.1176 - val_accuracy: 0.9648 - lr: 0.0010
Epoch 10/10
844/844 [==============================] - 2s 2ms/step - loss: 0.0514 - accuracy: 0.9879 - val_loss: 0.1212 - val_accuracy: 0.9635 - lr: 0.0010

313/313 - 0s - loss: 0.1254 - accuracy: 0.9624 - 321ms/epoch - 1ms/step
Test Accuracy: 0.9624

```

The Spiking Liquid Neural Network (SNLNN) achieved a test accuracy of approximately 96.24% in recognizing handwritten digits from the MNIST dataset. This performance highlights the SNLNN's effectiveness in adapting to the input data and learning from it over the course of training. The model's use of spiking dynamics and liquid state networks contributes to its capability to process and classify the data efficiently.

The training log reveals a consistent improvement in both training and validation metrics across the 10 epochs. The model begins with a training accuracy of 88.01% and a validation accuracy of 92.90% in the first epoch. By the final epoch, the training accuracy rises to 98.79%, while the validation accuracy reaches 96.35%. The loss function also demonstrates a significant reduction, decreasing from 0.5455 in the first epoch to 0.0514 by the last epoch for the training set, and from 0.2560 to 0.1212 for the validation set.

### Conclusion

The Spiking Liquid Neural Network (SNLNN) demonstrates strong performance in recognizing handwritten digits from the MNIST dataset, with a test accuracy of *96.24%*. The model shows substantial improvements throughout the training process, reflecting its ability to effectively learn and generalize from the data. The steady decrease in loss and increase in accuracy across epochs suggest that the SNLNN is successfully capturing the underlying patterns in the data. The combination of spiking dynamics and liquid state networks has proven advantageous, offering a robust approach to classification tasks. This model's performance underscores its potential for applications in complex pattern recognition and adaptive learning scenarios.