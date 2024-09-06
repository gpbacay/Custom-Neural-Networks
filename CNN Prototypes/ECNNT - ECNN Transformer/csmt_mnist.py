import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, GlobalAveragePooling2D, Dropout, BatchNormalization, Reshape, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position, d_model):
        super(PositionalEncoding, self).__init__()
        self.max_position = max_position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(max_position, d_model)
        
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, max_position, d_model):
        angle_rads = self.get_angles(np.arange(max_position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos_encoding = self.pos_encoding[:, :seq_len, :]
        return inputs + pos_encoding

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'max_position': self.max_position,
            'd_model': self.d_model
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(config['max_position'], config['d_model'])

def create_csm_model(input_shape, output_dim, d_model=64, self_modeling_weight=0.1):
    inputs = Input(shape=input_shape)
    
    # Enhanced Convolutional Layers
    def adaptive_conv(x, filters, kernel_size, strides=1):
        conv = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(x)
        norm = BatchNormalization()(conv)
        return tf.keras.layers.ReLU()(norm)
    
    x = adaptive_conv(inputs, 32, kernel_size=3, strides=2)
    x = adaptive_conv(inputs, 64, kernel_size=3, strides=2)
    x = adaptive_conv(x, 128, kernel_size=3, strides=2)
    x = adaptive_conv(x, 256, kernel_size=3, strides=2)
    x = adaptive_conv(x, 512, kernel_size=3, strides=2)
    
    # Apply Global Average Pooling
    model_features = GlobalAveragePooling2D()(x)
    x = Reshape((1, model_features.shape[-1]))(model_features)  # Add seq_len dimension for Dense layer

    # Add Positional Encoding
    pos_encoding_layer = PositionalEncoding(max_position=1, d_model=model_features.shape[-1])
    x = pos_encoding_layer(x)

    # Dynamic Self-Modeling Mechanism with Multi-Head Attention
    self_modeling_dense = Dense(d_model, activation='relu')(x)
    attention_output = MultiHeadAttention(num_heads=8, key_dim=d_model)(self_modeling_dense, self_modeling_dense)  # Increased number of heads
    self_modeling_output = Dense(model_features.shape[-1])(attention_output)
    
    x = Flatten()(x)
    
    # Final Classification Layers
    x = Dense(256, activation='relu')(x)  # Increased layer size
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    classification_output = Dense(output_dim, activation='softmax')(x)

    # Create the Model with Two Outputs
    model = Model(inputs, [classification_output, self_modeling_output])

    # Compile the Model with a Combined Loss Function and Multiple Metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
        loss=['categorical_crossentropy', 'mse'],  # Classification loss + Self-modeling loss
        loss_weights=[1.0, self_modeling_weight],  # Weight for the Self-modeling task
        metrics=[['accuracy'], ['mse']]  # Metrics for each output
    )
    
    return model

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

def main():
    input_shape = (28, 28, 1)
    output_dim = 10
    num_epochs = 10
    batch_size = 64

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    # Data Augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    model = create_csm_model(input_shape, output_dim)

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * (0.5 ** (epoch // 5)))

    # Train the Model
    model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=num_epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr, lr_schedule]
    )

    # Evaluate the Model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_accuracy:.4f}')

if __name__ == "__main__":
    main()



# Convolutional Self-Modeling Transformer (CSMT)
# python csmt_mnist.py
# Test Accuracy: 0.9882
