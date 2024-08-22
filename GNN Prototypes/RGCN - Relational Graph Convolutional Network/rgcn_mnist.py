import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Reshape, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping

class RGCNLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_relations, activation=None):
        super(RGCNLayer, self).__init__()
        self.units = units
        self.num_relations = num_relations
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.weight_list = [
            self.add_weight(
                shape=(input_shape[0][-1], self.units),
                initializer='glorot_uniform',
                name=f'weight_{i}',
                dtype=tf.float32
            ) for i in range(self.num_relations)
        ]
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bias',
            dtype=tf.float32
        )

    def call(self, inputs):
        x, adj_matrices = inputs
        aggregated_messages = []
        for i in range(self.num_relations):
            messages = tf.linalg.matmul(adj_matrices[i], x)
            aggregated_messages.append(tf.linalg.matmul(messages, self.weight_list[i]))
        
        aggregated_messages = tf.reduce_sum(tf.stack(aggregated_messages, axis=0), axis=0)
        output = tf.nn.bias_add(aggregated_messages, self.bias)
        
        return self.activation(output) if self.activation else output

def create_adjacency_matrices(image_shape, num_relations=4):
    height, width = image_shape
    num_nodes = height * width
    offsets = tf.constant([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=tf.int32)
    
    indices = tf.reshape(tf.stack(tf.meshgrid(tf.range(height), tf.range(width), indexing='ij'), axis=-1), [-1, 2])
    indices = tf.cast(indices, tf.int32)
    
    adj_matrices = []
    for i in range(num_relations):
        neighbor_indices = indices + offsets[i]
        valid_mask = tf.reduce_all((neighbor_indices >= 0) & (neighbor_indices < [height, width]), axis=1)
        valid_indices = tf.boolean_mask(indices, valid_mask)
        valid_neighbors = tf.boolean_mask(neighbor_indices, valid_mask)
        
        flat_indices = valid_indices[:, 0] * width + valid_indices[:, 1]
        flat_neighbors = valid_neighbors[:, 0] * width + valid_neighbors[:, 1]
        
        indices = tf.stack([flat_indices, flat_neighbors], axis=1)
        values = tf.ones(tf.shape(flat_indices), dtype=tf.float32)
        shape = [num_nodes, num_nodes]
        
        adj_matrix = tf.scatter_nd(indices, values, shape)
        adj_matrices.append(adj_matrix)
    
    return tf.stack(adj_matrices)

def create_rgcn_model(input_shape, num_classes, num_relations):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Reshape for RGCN layer
    x = Reshape((-1, 64))(x)
    
    # Create adjacency matrices
    adj_matrices = create_adjacency_matrices((14, 14), num_relations)
    
    # Apply RGCN layers
    x = RGCNLayer(128, num_relations, activation='relu')([x, adj_matrices])
    x = Dropout(0.5)(x)
    x = RGCNLayer(64, num_relations, activation='relu')([x, adj_matrices])
    
    x = Reshape((14, 14, 64))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomFlip('horizontal')
    ])
    
    x_train = data_augmentation(x_train, training=True)
    
    return x_train, y_train, x_test, y_test

def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    model = create_rgcn_model((28, 28, 1), 10, 4)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=10, 
        validation_split=0.1,
        callbacks=[early_stopping]
    )
    
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()






# Spatial Relational Graph Convolutional Network (SRGCN)
# python rgcn_mnist.py
# Test Accuracy: (slow)



