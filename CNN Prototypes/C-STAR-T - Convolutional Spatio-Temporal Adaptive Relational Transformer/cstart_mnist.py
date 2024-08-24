import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, GlobalAveragePooling2D, Dropout, Reshape
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
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

class SpatioTemporalRelationalLayer(tf.keras.layers.Layer):
    def __init__(self, num_relations, units, **kwargs):
        super().__init__(**kwargs)
        self.num_relations = num_relations  # Number of different relation types
        self.units = units  # Number of output units

        # Creating a list of dense layers for each relation type (parameter sharing within relations)
        self.relation_networks = [Dense(units, activation='relu') for _ in range(num_relations)]
        self.temporal_network = Dense(units, activation='relu')  # Separate network for temporal dependencies

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.temporal_aggregation_weights = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='temporal_aggregation_weights'
        )
        super().build(input_shape)

    def call(self, inputs):
        # Extract the batch size and feature dimension
        batch_size, feature_dim = tf.shape(inputs)[0], inputs.shape[-1]

        # Apply relation-specific transformations
        relation_outputs = []
        for network in self.relation_networks:
            relation_output = network(inputs)
            relation_outputs.append(relation_output)
        
        # Stack relation outputs to combine them
        combined_relation_outputs = tf.stack(relation_outputs, axis=1)  # Shape: (batch_size, num_relations, units)
        combined_relation_outputs = tf.reduce_sum(combined_relation_outputs, axis=1)  # Summing across relations

        # Apply temporal aggregation
        temporal_aggregated = tf.tensordot(inputs, self.temporal_aggregation_weights, axes=[[-1], [0]])

        # Integrate temporal aggregation with relation outputs
        output = tf.add(combined_relation_outputs, temporal_aggregated)
        
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

def create_cstart_model(input_shape, output_dim, num_relations=3, d_model=64, num_heads=4):
    inputs = Input(shape=input_shape)
    
    # EfficientNet-based Convolutional layers for feature extraction
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
    
    # Multi-Relational Learning with Spatio-Temporal Awareness
    x = Flatten()(x)
    spatio_temporal_relational = SpatioTemporalRelationalLayer(num_relations, 128)(x)
    
    # Final classification layers
    x = Dense(128, activation='relu')(spatio_temporal_relational)
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

    model = create_cstart_model(input_shape, output_dim)

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




# Convolutonal Spatio-Temporal Adaptive Relational Transformer (C-STAR-T)
# python cstart_mnist.py
# Test Accuracy: 0.9941
