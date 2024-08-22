# **Efficient Convolutional Transformer Network (ECNNT)**

Bacay, Gianne P. (2024)

![ECNNT Model Architecture Diagram][]

## Table of Contents
1. [Introduction](#introduction)
2. [What is Efficient Convolutional Transformer Network (ECNNT)?](#what-is-efficient-convolutional-transformer-network-ecnnt)
3. [Architecture of ECNNT](#architecture-of-ecnnt)
4. [How Does ECNNT Work?](#how-does-ecnnt-work)
5. [Implementation of ECNNT](#implementation-of-ecnnt)
   - [Import Libraries](#import-libraries)
   - [Define EfficientNet-like Block](#define-efficientnet-like-block)
   - [Build ECNNT Model](#build-ecnnt-model)
   - [Load and Preprocess MNIST Data](#load-and-preprocess-mnist-data)
   - [Set Hyperparameters and Train](#set-hyperparameters-and-train)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Advantages and Disadvantages of ECNNT](#advantages-and-disadvantages-of-ecnnt)
9. [Applications of ECNNT](#applications-of-ecnnt)

## Introduction

The Efficient Convolutional Transformer Network (ECNNT) is a sophisticated hybrid model combining the strengths of Convolutional Neural Networks (CNNs) with Transformer architectures. This model harnesses the feature extraction capabilities of CNNs and the sequence modeling power of Transformers, aiming to achieve high performance in image classification tasks. This document provides a comprehensive overview of ECNNT, including its architecture, implementation, results, and applications, with a focus on MNIST digit classification.

## What is Efficient Convolutional Transformer Network (ECNNT)?

The Efficient Convolutional Transformer Network (ECNNT) merges CNNs and Transformers to leverage their respective strengths:

- Convolutional Neural Networks (CNNs): Known for their ability to extract spatial features from images through convolutional and pooling operations.
- Transformer Architecture: Utilizes self-attention mechanisms to capture long-range dependencies and complex patterns in data sequences.

The integration of these architectures in ECNNT aims to enhance performance in tasks that require both spatial feature extraction and sequence modeling.

## Architecture of ECNNT

The ECNNT architecture features:

1. Input Layer:
   - Accepts input images for processing.
2. EfficientNet-like Convolutional Layers:
   - Includes a series of convolutional blocks for hierarchical feature extraction.
3. Global Average Pooling and Reshaping:
   - Converts 2D features into a format suitable for Transformer processing.
4. Transformer-like Multi-Head Attention Layer:
   - Applies self-attention to capture dependencies and refine feature representations.
5. Output Layer:
   - Final dense layers for classification.

![ECNNT Model Architecture Diagram][]

## How Does ECNNT Work?

1. Feature Extraction:
   - Convolutional layers extract hierarchical spatial features from input images.
2. Transformer Processing:
   - The Transformer layer refines feature representations by capturing long-range dependencies.
3. Classification:
   - The output layer generates predictions based on the processed features.

## Implementation of ECNNT

Below is the code to implement the ECNNT model using TensorFlow:

### Import Libraries

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, GlobalAveragePooling2D, Dropout, Reshape
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
```

### Define EfficientNet-like Block

```python
def efficientnet_block(inputs, filters, expansion_factor, stride):
    expanded_filters = filters * expansion_factor
    x = Conv2D(expanded_filters, kernel_size=1, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(expanded_filters, kernel_size=3, padding='same', strides=stride, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if stride == 1 and inputs.shape[-1] == x.shape[-1]:
        x = tf.keras.layers.Add()([inputs, x])
    return x
```

### Build ECNNT Model

```python
def create_ecnn_transformer_model(input_shape, output_dim, d_model=64, num_heads=4):
    inputs = Input(shape=input_shape)
    
    # EfficientNet-like Convolutional layers for feature extraction
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2)
    
    # Prepare for Transformer layer
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, x.shape[-1]))(x)  # Add seq_len dimension for MultiHeadAttention
    
    # Transformer-like Multi-Head Attention layer
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    x = Flatten()(x)
    
    # Final classification layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model
```

### Load and Preprocess MNIST Data

```python
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train[..., tf.newaxis]
    x_val = x_val[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
```

### Set Hyperparameters and Train

```python
def main():
    input_shape = (28, 28, 1)
    output_dim = 10
    num_epochs = 10
    batch_size = 64

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    model = create_ecnn_transformer_model(input_shape, output_dim)

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=num_epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_accuracy:.4f}')

if __name__ == "__main__":
    main()
```

## Results

```shell
2024-08-02 17:23:09.978943: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/10
844/844 [==============================] - 74s 78ms/step - loss: 0.1791 - accuracy: 0.9451 - val_loss: 0.0487 - val_accuracy: 0.9857 - lr: 0.0010
Epoch 2/10
844/844 [==============================] - 65s 77ms/step - loss: 0.0711 - accuracy: 0.9776 - val_loss: 0.0436 - val_accuracy: 0.9877 - lr: 0.0010
...
Epoch 10/10
844/844 [==============================] - 64s 75ms/step - loss: 0.0102 - accuracy: 0.9970 - val_loss: 0.0394 - val_accuracy: 0.9883 - lr: 0.0010
157/157 [==============================] - 7s 43ms/step - loss: 0.0204 - accuracy: 0.9947
Test accuracy: 0.9947
```

## Conclusion

The Efficient Convolutional Transformer Network (ECNNT) effectively combines the strengths of CNNs and Transformer architectures, demonstrating significant performance improvements in image classification tasks. The model achieved an impressive test accuracy of 99.47% on the MNIST dataset, validating its efficiency and capability in handling complex image recognition challenges.

## Advantages and Disadvantages of ECNNT

### Advantages:
- Enhanced Feature Extraction: Utilizes CNNs for detailed spatial feature extraction.
- Long-Range Dependencies: Leverages Transformers for capturing long-range dependencies and complex patterns.
- High Accuracy: Demonstrates high accuracy on benchmark datasets like MNIST.

### Disadvantages:
- Resource Intensive: May require significant computational resources, especially for larger datasets and models.
- Complexity: The integration of CNN and Transformer layers increases model complexity and training time.

## Applications of ECNNT

- Image Classification: Effective for tasks requiring detailed image feature extraction and sequence learning.
- Object Detection: Can be adapted for identifying and classifying objects within images.
- Medical Image Analysis: Useful for analyzing complex medical imaging data, such as MRI and CT scans.