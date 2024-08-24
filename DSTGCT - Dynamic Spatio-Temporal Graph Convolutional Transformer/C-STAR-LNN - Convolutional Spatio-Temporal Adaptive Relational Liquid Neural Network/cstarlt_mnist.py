import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
import logging

# Set up logging and suppress warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

# Improved Custom Layer for Dynamic Spatial Reservoir Processing
class DynamicSpatialReservoirLayer(layers.Layer):
    def __init__(self, reservoir_dim, input_dim, spectral_radius, leak_rate, **kwargs):
        super().__init__(**kwargs)
        self.reservoir_dim = reservoir_dim
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate

    def build(self, input_shape):
        self.reservoir_weights = self.add_weight(
            shape=(self.reservoir_dim, self.reservoir_dim),
            initializer='glorot_uniform',
            name='reservoir_weights'
        )
        eigenvalues, _ = tf.linalg.eig(self.reservoir_weights)
        max_eigenvalue = tf.reduce_max(tf.abs(eigenvalues))
        self.reservoir_weights.assign(self.reservoir_weights * (self.spectral_radius / max_eigenvalue))
        
        self.input_weights = self.add_weight(
            shape=(self.reservoir_dim, self.input_dim),
            initializer='glorot_uniform',
            name='input_weights'
        )

    def call(self, inputs):
        prev_state = tf.zeros((tf.shape(inputs)[0], self.reservoir_dim))
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.nn.tanh(input_part + reservoir_part)
        return state

# Improved Custom Layer for Adaptive Message Passing
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
        messages = [tf.matmul(inputs, self.relation_weights[i] * self.relation_scales[i]) 
                    for i in range(self.num_relations)]
        return tf.reduce_sum(messages, axis=0)

# Efficient Convolutional Block (inspired by EfficientNet)
def efficient_conv_block(inputs, filters, expansion_factor, stride):
    expanded_filters = filters * expansion_factor
    x = layers.Conv2D(expanded_filters, kernel_size=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', strides=stride, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if stride == 1 and inputs.shape[-1] == filters:
        x = layers.Add()([inputs, x])
    return x

# Improved C-STAR-LT Model
def create_improved_cstarlt_model(input_shape, reservoir_dim, spectral_radius, leak_rate, output_dim, num_relations, d_model=64, num_heads=4):
    inputs = layers.Input(shape=input_shape)
    
    # Efficient Convolutional layers for feature extraction
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = efficient_conv_block(x, 16, expansion_factor=1, stride=1)
    x = efficient_conv_block(x, 24, expansion_factor=6, stride=2)
    x = efficient_conv_block(x, 40, expansion_factor=6, stride=2)
    
    # Prepare for Transformer layer
    x = layers.GlobalAveragePooling2D()(x)
    
    # Transformer-based Multi-Head Attention layer
    x = layers.Dense(d_model, activation='relu')(x)
    x = layers.Reshape((1, d_model))(x)
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
    x = layers.Flatten()(x)
    
    # Dynamic Spatial Reservoir Layer
    dynamic_spatial_reservoir = DynamicSpatialReservoirLayer(reservoir_dim, x.shape[-1], spectral_radius, leak_rate)
    reservoir_output = dynamic_spatial_reservoir(x)
    
    # Adaptive Message Passing Layer
    adaptive_message_passing = AdaptiveMessagePassingLayer(num_relations, reservoir_dim)
    multi_relational_output = adaptive_message_passing(reservoir_output)
    
    # Output layer
    x = layers.Dense(128, activation='relu')(multi_relational_output)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(output_dim, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Improved data preprocessing
def preprocess_mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train[..., tf.newaxis]
    x_val = x_val[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)
    y_test = keras.utils.to_categorical(y_test)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main():
    # Set hyperparameters
    input_shape = (28, 28, 1)
    reservoir_dim = 256
    spectral_radius = 0.95
    leak_rate = 0.1
    output_dim = 10
    num_relations = 3
    num_epochs = 10
    batch_size = 64

    # Prepare data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_mnist_data()

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )

    datagen.fit(x_train)

    # Create Improved C-STAR-LT Model
    model = create_improved_cstarlt_model(input_shape, reservoir_dim, spectral_radius, leak_rate, output_dim, num_relations)

    # Compile and train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    try:
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=num_epochs,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, reduce_lr]
        )

        # Evaluate the model
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
        logging.info(f'Test accuracy: {test_acc:.4f}')

        # Plot training history
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig('improved_cstarlt_training_history.png')
        plt.close()

    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    main()





# Convolutonal Spatio-Temporal Adaptive Relational Liquid Transformer (C-STAR-LT) in DSTGCT
# python cstarlt_mnist.py
# Test Accuracy: 0.9801
