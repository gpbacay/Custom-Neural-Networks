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

# EfficientNet Block
def efficientnet_block(inputs, filters, expansion_factor, stride):
    expanded_filters = filters * expansion_factor
    x = layers.Conv2D(expanded_filters, kernel_size=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(expanded_filters, kernel_size=3, padding='same', strides=stride, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if stride == 1 and inputs.shape[-1] == filters:
        x = layers.Add()([inputs, x])
    return x

# Custom Layer for Dynamic Spatial Reservoir Processing
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

# Create Spatio-Temporal Adaptive Relational Liquid Neural Network (C-STAR-LNN)
def create_cstarl_transformer_model(input_shape, reservoir_dim, spectral_radius, leak_rate, output_dim, num_relations, d_model=64, num_heads=4):
    inputs = layers.Input(shape=input_shape)
    
    # EfficientNet-based Convolutional layers for feature extraction
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2)
    
    # Prepare for Transformer layer
    x = layers.GlobalAveragePooling2D()(x)
    
    # Transformer-based Multi-Head Attention layer for Convolutional Outputs
    x = layers.Dense(d_model, activation='relu')(x)
    x = layers.Reshape((1, d_model))(x)  # Add sequence dimension for attention
    attention_output_conv = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = layers.Flatten()(attention_output_conv)
    x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Flatten()(x))
    
    # Dynamic Spatial Reservoir Layer with Transformer-based Attention
    dynamic_spatial_reservoir = DynamicSpatialReservoirLayer(reservoir_dim, x.shape[-1], spectral_radius, leak_rate)
    reservoir_output = dynamic_spatial_reservoir(x)
    reservoir_output = layers.Reshape((1, reservoir_dim))(reservoir_output)
    attention_output_reservoir = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(reservoir_output, reservoir_output)
    reservoir_output = layers.Flatten()(attention_output_reservoir)
    reservoir_output = layers.LayerNormalization(epsilon=1e-6)(reservoir_output + layers.Flatten()(reservoir_output))
    
    # Adaptive Message Passing Layer with Transformer-based Attention
    adaptive_message_passing = AdaptiveMessagePassingLayer(num_relations, reservoir_dim)
    multi_relational_output = adaptive_message_passing(reservoir_output)
    multi_relational_output = layers.Reshape((1, reservoir_dim))(multi_relational_output)
    attention_output_message = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(multi_relational_output, multi_relational_output)
    multi_relational_output = layers.Flatten()(attention_output_message)
    multi_relational_output = layers.LayerNormalization(epsilon=1e-6)(multi_relational_output + layers.Flatten()(multi_relational_output))
    
    # Combine and output
    x = layers.Flatten()(multi_relational_output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(output_dim, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Preprocess MNIST Data with Data Augmentation
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

def main():
    # Set hyperparameters
    input_shape = (28, 28, 1)
    reservoir_dim = 256
    spectral_radius = 0.95
    leak_rate = 0.1
    output_dim = 10
    num_relations = 3
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

    # Create Spatio-Temporal Adaptive Relational Liquid Neural Network (C-STAR-LNN)
    model = create_cstarl_transformer_model(input_shape, reservoir_dim, spectral_radius, leak_rate, output_dim, num_relations)

    # Compile and train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=num_epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    main()





# Convolutonal Spatio-Temporal Adaptive Relational Liquid Transformer (C-STAR-LT) in DSTGCT
# python cstarlt_mnist.py
# Test Accuracy: 
