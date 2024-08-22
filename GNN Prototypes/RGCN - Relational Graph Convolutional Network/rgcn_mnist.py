import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, GlobalAveragePooling1D, Dropout
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
        input_dim = input_shape[0][-1]
        self.weight_list = [
            self.add_weight(
                shape=(input_dim, self.units),
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

    @tf.function
    def call(self, inputs):
        x, adj_matrices = inputs
        batch_size = tf.shape(x)[0]
        num_nodes = tf.shape(x)[1]
        
        aggregated_messages = tf.zeros((batch_size, num_nodes, self.units), dtype=tf.float32)
        for i in range(self.num_relations):
            # Perform sparse matrix multiplication for each sample in the batch
            messages = tf.map_fn(
                lambda sample: tf.sparse.sparse_dense_matmul(adj_matrices[i], sample),
                x
            )
            aggregated_messages += tf.matmul(messages, self.weight_list[i])
        
        output = tf.nn.bias_add(aggregated_messages, self.bias)
        return self.activation(output) if self.activation else output

def create_adjacency_matrices(height, width, num_relations=4):
    num_nodes = height * width
    offsets = tf.constant([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=tf.int32)
    
    indices = tf.reshape(tf.stack(tf.meshgrid(tf.range(height), tf.range(width), indexing='ij'), axis=-1), [-1, 2])
    
    adj_matrices = []
    for i in range(num_relations):
        neighbor_indices = indices + offsets[i]
        valid_mask = tf.reduce_all((neighbor_indices >= 0) & (neighbor_indices < [height, width]), axis=1)
        valid_indices = tf.boolean_mask(indices, valid_mask)
        valid_neighbors = tf.boolean_mask(neighbor_indices, valid_mask)
        
        flat_indices = valid_indices[:, 0] * width + valid_indices[:, 1]
        flat_neighbors = valid_neighbors[:, 0] * width + valid_neighbors[:, 1]
        
        sparse_indices = tf.stack([tf.cast(flat_indices, tf.int64), tf.cast(flat_neighbors, tf.int64)], axis=1)
        values = tf.ones(tf.shape(flat_indices)[0], dtype=tf.float32)
        shape = [num_nodes, num_nodes]
        
        adj_matrix = tf.sparse.SparseTensor(sparse_indices, values, shape)
        adj_matrices.append(adj_matrix)
    
    return adj_matrices

def create_rgcn_model(input_shape, num_classes, num_relations):
    inputs = Input(shape=input_shape)
    
    x = Reshape((input_shape[0] * input_shape[1], input_shape[2]))(inputs)
    
    adj_matrices = create_adjacency_matrices(input_shape[0], input_shape[1], num_relations)
    
    x = RGCNLayer(64, num_relations, activation='relu')([x, adj_matrices])
    x = Dropout(0.5)(x)
    x = RGCNLayer(32, num_relations, activation='relu')([x, adj_matrices])
    
    x = GlobalAveragePooling1D()(x)
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
    
    return x_train, y_train, x_test, y_test

@tf.function
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    model = create_rgcn_model((28, 28, 1), 10, 4)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
    train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test[:5000], y_test[:5000])).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test[5000:], y_test[5000:])).batch(64)
    
    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        callbacks=[early_stopping]
    )
    
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()



# Relational Graph Convolutional Network (RGCN)
# python rgcn_mnist.py
# Test Accuracy: (slow)



