import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model

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

def create_ecnn_model(input_shape, output_dim):
    inputs = Input(shape=input_shape)
    
    # EfficientNet Convolutional layers for feature extraction
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    # Final classification layer
    outputs = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model
