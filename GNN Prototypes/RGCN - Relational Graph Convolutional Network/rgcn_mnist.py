import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

class RGCNLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_relations, activation=None):
        super(RGCNLayer, self).__init__()
        self.units = units
        self.num_relations = num_relations
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.weight_list = [
            self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                name=f'weight_{i}'
            ) for i in range(self.num_relations)
        ]
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bias'
        )

    def call(self, inputs, adj_matrices):
        outputs = []
        for i in range(self.num_relations):
            output = tf.matmul(adj_matrices[i], inputs)
            output = tf.matmul(output, self.weight_list[i])
            outputs.append(output)
        
        output = tf.add_n(outputs)
        output = tf.nn.bias_add(output, self.bias)
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output

def create_adjacency_matrices(image_shape, num_relations=4):
    height, width = image_shape
    num_nodes = height * width
    adj_matrices = []

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for offset in offsets:
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(height):
            for j in range(width):
                node = i * width + j
                neighbor_i, neighbor_j = i + offset[0], j + offset[1]
                if 0 <= neighbor_i < height and 0 <= neighbor_j < width:
                    neighbor_node = neighbor_i * width + neighbor_j
                    adj_matrix[node, neighbor_node] = 1
        adj_matrices.append(adj_matrix)

    return np.array(adj_matrices, dtype=np.float32)

def create_rgcn_model(input_shape, num_classes, num_relations):
    inputs = Input(shape=input_shape)
    
    x = tf.keras.layers.Reshape((input_shape[0] * input_shape[1], input_shape[2]))(inputs)
    
    adj_matrices = create_adjacency_matrices(input_shape[:2], num_relations)
    adj_matrices = tf.convert_to_tensor(adj_matrices)
    
    x = RGCNLayer(64, num_relations, activation='relu')(x, adj_matrices)
    x = RGCNLayer(32, num_relations, activation='relu')(x, adj_matrices)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Create and compile the model
model = create_rgcn_model((28, 28, 1), 10, 4)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")







# Spatial Relational Graph Convolutional Network (RGCN)
# python srgcn_mnist.py
# Test Accuracy: 0.9671



