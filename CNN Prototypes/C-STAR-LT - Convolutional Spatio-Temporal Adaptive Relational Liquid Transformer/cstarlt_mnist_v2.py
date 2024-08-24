import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, GlobalAveragePooling2D, Dropout, Reshape
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

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

class MultiRelationalLayer(tf.keras.layers.Layer):
    def __init__(self, num_relations, units):
        super().__init__()
        self.num_relations = num_relations
        self.units = units
        self.relation_networks = [Dense(units, activation='relu') for _ in range(num_relations)]

    def call(self, inputs):
        relations = [network(inputs) for network in self.relation_networks]
        return tf.stack(relations, axis=1)  # Shape: (batch_size, num_relations, units)

class MessagePassingLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.message_network = Dense(units, activation='relu')
        self.update_network = Dense(units, activation='relu')

    def call(self, node_features, relation_features):
        messages = tf.reduce_sum(relation_features, axis=1)
        messages = self.message_network(messages)
        updated_features = tf.concat([node_features, messages], axis=-1)
        return self.update_network(updated_features)

class LiquidReservoirLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation='tanh', use_bias=True):
        super(LiquidReservoirLayer, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=[int(input_shape[-1]), self.units],
            initializer="glorot_uniform",
            trainable=False
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.units],
                initializer="zeros",
                trainable=False
            )

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

def create_combined_model(input_shape, output_dim, num_relations=3, d_model=64, num_heads=4, reservoir_dim=200):
    inputs = Input(shape=input_shape)

    # EfficientNet-based Convolutional layers
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2)
    
    # Prepare for Transformer and Relational layers
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, x.shape[-1]))(x)  # Add seq_len dimension
    
    # Transformer-based Multi-Head Attention layer
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # Multi-Relational Learning
    x = Reshape((-1, x.shape[-1]))(x)  # Ensuring it has the shape (batch_size, time_steps, features)
    multi_relational = MultiRelationalLayer(num_relations, 128)(x)
    
    # Message Passing
    x = MessagePassingLayer(128)(x, multi_relational)
    
    # Flatten before Liquid Reservoir layer
    x = Flatten()(x)
    
    # Liquid Reservoir layer
    x = LiquidReservoirLayer(reservoir_dim)(x)
    
    # Final classification layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs, outputs)
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
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    input_shape = x_train.shape[1:]
    output_dim = 10

    model = create_combined_model(input_shape, output_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    # Train the model
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=10,
                        batch_size=64,
                        callbacks=[early_stopping, reduce_lr])

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    main()



# Convolutonal Spatio-Temporal Adaptive Relational Liquid Transformer (C-STAR-LT)
# python cstarlt_mnist_v2.py
# Test Accuracy: 0.9942
