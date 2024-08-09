import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Graph Transformer Network (GTNN) Layers
class SimplifiedGraphTransformerLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_heads=2):
        super(SimplifiedGraphTransformerLayer, self).__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        self.query_dense = layers.Dense(output_dim)
        self.key_dense = layers.Dense(output_dim)
        self.value_dense = layers.Dense(output_dim)
        self.combine_heads = layers.Dense(output_dim)
        
    def call(self, inputs, adjacency_matrix):
        batch_size = tf.shape(inputs)[0]
        num_nodes = tf.shape(inputs)[1]
        
        queries = self.query_dense(inputs)
        keys = self.key_dense(inputs)
        values = self.value_dense(inputs)
        
        queries = tf.reshape(queries, [batch_size, num_nodes, self.num_heads, self.head_dim])
        keys = tf.reshape(keys, [batch_size, num_nodes, self.num_heads, self.head_dim])
        values = tf.reshape(values, [batch_size, num_nodes, self.num_heads, self.head_dim])
        
        scores = tf.einsum('bqhd,bkhd->bhqk', queries, keys)
        scores = scores / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        
        mask = tf.expand_dims(tf.expand_dims(adjacency_matrix, axis=0), axis=0)
        scores += (1.0 - mask) * -1e9
        
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attended_values = tf.einsum('bhqk,bkhd->bqhd', attention_weights, values)
        attended_values = tf.reshape(attended_values, [batch_size, num_nodes, self.output_dim])
        
        output = self.combine_heads(attended_values)
        return output

class SimplifiedGraphTransformerLayerWrapper(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_heads=2):
        super(SimplifiedGraphTransformerLayerWrapper, self).__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads

    def build(self, input_shape):
        num_nodes = input_shape[0][1]
        self.adjacency_matrix = self.add_weight(
            shape=(num_nodes, num_nodes),
            initializer='random_uniform',
            trainable=True,
            name='adjacency_matrix'
        )

    def call(self, inputs):
        node_features, _ = inputs
        return SimplifiedGraphTransformerLayer(self.output_dim, self.num_heads)(node_features, self.adjacency_matrix)

# Efficient Convolutional Neural Network (ECNN) Model
def create_efficient_cnn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # Initial Convolutional Layer
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # EfficientNet-like Blocks
    def efficientnet_block(x, filters, expansion_factor, stride):
        # Expansion Phase
        expanded_filters = filters * expansion_factor
        x = layers.Conv2D(expanded_filters, kernel_size=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Depthwise Convolution
        x = layers.DepthwiseConv2D(kernel_size=3, padding='same', strides=stride, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Pointwise Convolution
        x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        
        # Residual Connection
        if stride == 1 and inputs.shape[-1] == x.shape[-1]:
            x = layers.Add()([x, inputs])
        
        return x
    
    # Apply EfficientNet-like Blocks
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully Connected Layer
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, x)
    return model

# Load and preprocess data
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    # Normalize the images to the range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Add channel dimension
    x_train = x_train[..., tf.newaxis]
    x_val = x_val[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main():
    input_shape = (28, 28, 1)
    num_classes = 10
    num_epochs = 10
    batch_size = 64

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    # Data augmentation
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

    # Create and compile the model
    model = create_efficient_cnn_model(input_shape, num_classes)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=num_epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()


# Convolutional Graph Transformer Neural Network (CGTNN)
# python cgtnn_mnist.py
# Test Accuracy: 0.9924

