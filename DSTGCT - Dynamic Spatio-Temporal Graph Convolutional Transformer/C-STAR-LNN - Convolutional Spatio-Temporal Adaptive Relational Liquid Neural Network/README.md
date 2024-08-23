# **Convolutional Spatio-Temporal Adaptive Relational Liquid Neural Network (C-STAR-LNN)**

Bacay, Gianne P. (2024)

![C-STAR-LNN Model Architecture Diagram][]

## Table of Contents
1. [Introduction](#introduction)
2. [Statement of the Problem](#statement-of-the-problem)
3. [What is Convolutional Spatio-Temporal Adaptive Relational Liquid Neural Network (C-STAR-LNN)?](#what-is-convolutional-spatio-temporal-adaptive-relational-liquid-neural-network-c-star-lnn)
4. [Architecture of C-STAR-LNN](#architecture-of-c-star-lnn)
5. [How Does C-STAR-LNN Work?](#how-does-c-star-lnn-work)
6. [Implementation of C-STAR-LNN](#implementation-of-c-star-lnn)
   - [Import Libraries](#import-libraries)
   - [Define Custom Layers](#define-custom-layers)
   - [Build C-STAR-LNN Model](#build-c-star-lnn-model)
   - [Preprocess MNIST Data](#preprocess-mnist-data)
   - [Set Hyperparameters and Train](#set-hyperparameters-and-train)
7. [Results](#results)
8. [Conclusion](#conclusion)
9. [Advantages and Disadvantages of C-STAR-LNN](#advantages-and-disadvantages-of-c-star-lnn)
10. [Applications of C-STAR-LNN](#applications-of-c-star-lnn)

## Introduction

The Convolutional Spatio-Temporal Adaptive Relational Liquid Neural Network (C-STAR-LNN) is an advanced model that integrates convolutional neural networks (CNNs) with dynamic and adaptive mechanisms. This hybrid model combines the spatial feature extraction power of CNNs with the spatio-temporal processing capabilities of liquid neural networks and adaptive message passing. The goal is to achieve superior performance in image classification tasks by leveraging both spatial and relational features.

## Statement of the Problem

Despite significant advancements in deep learning models for image classification, there remain challenges in effectively capturing and integrating spatial, temporal, and relational information within a single architecture. Traditional convolutional neural networks excel at spatial feature extraction but may struggle with temporal dependencies and complex relational structures in data. Additionally, many existing models lack adaptability to changing input distributions and dynamic environments.

The C-STAR-LNN model aims to address these limitations by:

1. Enhancing spatial feature extraction with dynamic temporal processing.
2. Incorporating adaptive relational learning mechanisms.
3. Improving model flexibility and generalization across various image classification tasks.
4. Balancing computational efficiency with high accuracy in complex visual recognition scenarios.

By tackling these challenges, the C-STAR-LNN model seeks to push the boundaries of image classification performance and provide a more versatile architecture for a wide range of computer vision applications.

## What is Convolutional Spatio-Temporal Adaptive Relational Liquid Neural Network (C-STAR-LNN)?

The C-STAR-LNN model incorporates several innovative components:

- **Convolutional Neural Networks (CNNs):** Used for extracting hierarchical spatial features from images.
- **Dynamic Spatial Reservoir Layer:** A custom layer designed to dynamically process spatial information with a reservoir-like mechanism.
- **Adaptive Message Passing Layer:** A custom layer that performs adaptive message passing for relational learning.

By integrating these components, C-STAR-LNN aims to enhance both spatial feature extraction and relational learning capabilities.

## Architecture of C-STAR-LNN

The C-STAR-LNN architecture features:

1. **Input Layer:**
   - Accepts input images for processing.
2. **Convolutional Layers:**
   - A series of convolutional and pooling layers for hierarchical feature extraction.
3. **Dynamic Spatial Reservoir Layer:**
   - Processes spatial information with dynamic reservoir-based computations.
4. **Adaptive Message Passing Layer:**
   - Performs message passing based on learned relational weights.
5. **Output Layer:**
   - Final dense layers for classification.

![C-STAR-LNN Model Architecture Diagram][]

## How Does C-STAR-LNN Work?

1. **Feature Extraction:**
   - Convolutional layers extract hierarchical spatial features from input images.
2. **Dynamic Reservoir Processing:**
   - The Dynamic Spatial Reservoir Layer processes these features dynamically, integrating information over time.
3. **Relational Processing:**
   - The Adaptive Message Passing Layer refines the processed features by capturing relational dependencies.
4. **Classification:**
   - The output layer generates predictions based on the processed and refined features.

## Implementation of C-STAR-LNN

Below is the code to implement the C-STAR-LNN model using TensorFlow:

### Import Libraries

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
```

### Define Custom Layers

```python
# Custom Layer for Dynamic Spatial Reservoir Processing
class DynamicSpatialReservoirLayer(layers.Layer):
    def __init__(self, reservoir_dim, input_dim, spectral_radius, leak_rate, **kwargs):
        super().__init__(**kwargs)
        self.reservoir_dim = reservoir_dim
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.reservoir_weights = None
        self.input_weights = None

    def build(self, input_shape):
        self.reservoir_weights = self.add_weight(
            shape=(self.reservoir_dim, self.reservoir_dim),
            initializer='glorot_uniform',
            name='reservoir_weights'
        )
        self.input_weights = self.add_weight(
            shape=(self.reservoir_dim, self.input_dim),
            initializer='glorot_uniform',
            name='input_weights'
        )

    def call(self, inputs):
        prev_state = tf.zeros((tf.shape(inputs)[0], self.reservoir_dim))
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_part + reservoir_part)
        return state

# Custom Layer for Adaptive Message Passing
class AdaptiveMessagePassingLayer(layers.Layer):
    def __init__(self, num_relations, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_relations = num_relations
        self.output_dim = output_dim

    def build(self, input_shape):
        self.relation_weights = self.add_weight(
            shape=(self.num_relations, input_shape[-1], self.output_dim),
            initializer='glorot_uniform',
            name='relation_weights'
        )
        self.relation_scales = self.add_weight(
            shape=(self.num_relations, 1),
            initializer='ones',
            name='relation_scales'
        )

    def call(self, inputs):
        messages = []
        for i in range(self.num_relations):
            scaled_weights = self.relation_weights[i] * self.relation_scales[i]
            message = tf.matmul(inputs, scaled_weights)
            messages.append(message)
        return tf.reduce_sum(messages, axis=0)
```

### Build C-STAR-LNN Model

```python
# Create Spatio-Temporal Adaptive Relational Liquid Neural Network (STAR-LNN)
def create_star_lnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, output_dim, num_relations):
    inputs = layers.Input(shape=input_shape)

    # Add Convolutional Layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    
    input_dim = x.shape[-1]

    # Dynamic Spatial Reservoir Layer
    dynamic_spatial_reservoir = DynamicSpatialReservoirLayer(reservoir_dim, input_dim, spectral_radius, leak_rate)
    reservoir_output = dynamic_spatial_reservoir(x)

    # Adaptive Message Passing Layer
    adaptive_message_passing = AdaptiveMessagePassingLayer(num_relations, reservoir_dim)
    multi_relational_output = adaptive_message_passing(reservoir_output)

    # Combine and output
    combined_features = layers.Concatenate()([reservoir_output, multi_relational_output])
    outputs = layers.Dense(output_dim, activation='softmax')(combined_features)

    model = models.Model(inputs, outputs)
    return model
```

### Preprocess MNIST Data

```python
def preprocess_mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, 28*28)).reshape(-1, 28, 28, 1)
    x_val = scaler.transform(x_val.reshape(-1, 28*28)).reshape(-1, 28, 28, 1)
    x_test = scaler.transform(x_test.reshape(-1, 28*28)).reshape(-1, 28, 28, 1)

    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)
    y_test = keras.utils.to_categorical(y_test)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
```

### Set Hyperparameters and Train

```python
def main():
    # Set hyperparameters
    input_shape = (28, 28, 1)
    reservoir_dim = 1000
    spectral_radius = 1.5
    leak_rate = 0.3
    output_dim = 10
    num_relations = 4
    num_epochs = 10

    # Prepare data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_mnist_data()

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    datagen.fit(x_train)

    # Create Spatio-Temporal Adaptive Relational Liquid Neural Network (STAR-LNN)
    model = create_star_lnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, output_dim, num_relations)

    # Compile and train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=num_epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
```

## Results

The model was evaluated on the MNIST test set, achieving the following results:

- Test Accuracy: 99.22%
- Test Loss: 0.0247

## Conclusion

The C-STAR-LNN model demonstrates strong performance on the MNIST dataset, achieving high accuracy and low loss. The integration of dynamic reservoir processing and adaptive message passing has proven effective in enhancing the model's capabilities.

## Advantages and Disadvantages of C-STAR-LNN

Advantages:

- High performance on image classification tasks.
- Effective integration of spatial, temporal, and relational processing.

Disadvantages:

- Long training times, which may be improved with hardware accelerators or optimizations.

## Applications of C-STAR-LNN

The C-STAR-LNN model can be applied to various tasks requiring advanced feature extraction and relational learning, such as:

- Image classification and object recognition
- Temporal sequence modeling
- Graph-based learning and relational data analysis
```

Would you like me to explain or break down any part of this markdown content?