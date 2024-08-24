import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Input, Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

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

def efficientnet_block(inputs, filters, expansion_factor, stride):
    expanded_filters = filters * expansion_factor
    x = Conv2D(expanded_filters, kernel_size=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = Conv2D(expanded_filters, kernel_size=3, padding='same', strides=stride, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if stride == 1 and inputs.shape[-1] == filters:
        x = layers.Add()([inputs, x])
    return x

def create_cstarlt_model(input_shape, reservoir_dim, spectral_radius, leak_rate, output_dim, num_relations, d_model=64, num_heads=4):
    inputs = Input(shape=input_shape)
    
    # Convolutional feature extraction using EfficientNet-based blocks
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2)
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Dynamic Spatial Reservoir Layer
    dynamic_spatial_reservoir = DynamicSpatialReservoirLayer(reservoir_dim, 40, spectral_radius, leak_rate)
    reservoir_output = dynamic_spatial_reservoir(x)
    
    # Adaptive Message Passing Layer
    adaptive_message_passing = AdaptiveMessagePassingLayer(num_relations, reservoir_dim)
    multi_relational_output = adaptive_message_passing(reservoir_output)
    
    # Transformer-based Multi-Head Attention layer
    attention_input = layers.Reshape((1, -1))(x)  # Add sequence length dimension
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(attention_input, attention_input)
    
    # Adjust the dimension of attention_output to match x's dimension
    attention_output = layers.Reshape((-1,))(attention_output)  # Remove sequence length dimension
    attention_output = Dense(40, activation='relu')(attention_output)  # Adjust to match x's dimension
    attention_output = LayerNormalization(epsilon=1e-6)(layers.Add()([x, attention_output]))
    
    # Combine features with a gated mechanism
    combined_features = layers.Concatenate()([reservoir_output, multi_relational_output, attention_output])
    combined_features = layers.Dense(128, activation='relu')(combined_features)
    
    # Fully connected layers
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(combined_features)
    x = Dropout(0.5)(x)
    outputs = Dense(output_dim, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

def preprocess_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    # Reshape and normalize
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_val = x_val.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

    # Convert labels to categorical
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main():
    input_shape = (28, 28, 1)
    reservoir_dim = 200
    spectral_radius = 1.5
    leak_rate = 0.3
    output_dim = 10
    num_relations = 3
    num_epochs = 10
    batch_size = 64

    # Prepare data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_mnist_data()
    
    model = create_cstarlt_model(input_shape, reservoir_dim, spectral_radius, leak_rate, output_dim, num_relations)

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

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





# Convolutonal Spatio-Temporal Adaptive Relational Liquid Transformer (C-STAR-LT) in DSTGCT
# python cstarlt_mnist.py
# Test Accuracy: 0.9941
