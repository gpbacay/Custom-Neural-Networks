import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(GraphAttentionLayer, self).__init__()
        self.output_dim = output_dim
        self.dense = Dense(output_dim)
        self.attention_dense = Dense(1, use_bias=True)
    
    def call(self, inputs):
        x = self.dense(inputs)
        attention_scores = self.attention_dense(x)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        context = tf.reduce_sum(attention_weights * x, axis=1)
        return context

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

def mnist_to_graph(images):
    num_samples = images.shape[0]
    num_nodes = images.shape[1]
    # Create a fully connected adjacency matrix
    adjacency_matrix = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    # Node features are the pixel values
    node_features = images
    return node_features, adjacency_matrix

def create_gann_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((1, input_dim))(inputs)  # Reshape input to add singleton dimension
    x, _ = mnist_to_graph(x)  # Convert inputs to graph representation
    x = GraphAttentionLayer(64)(x)  # Apply Graph Attention Layer
    x = Dense(32, activation='relu')(x)  # Apply dense layer with ReLU activation
    outputs = Dense(output_dim, activation='softmax')(x)  # Output layer for classification
    return Model(inputs, outputs)

input_dim = 28 * 28  # Input dimension
output_dim = 10  # Number of classes

model = create_gann_model(input_dim, output_dim)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Increased batch size for better training speed, consider tuning for performance
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)

# Evaluate the model and print test accuracy
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")


# pip install tensorflow tensorflow-gnn
# Graph Attention Neural Network (GANN)
# python gann_mnist_v1.py
# Test Accuracy: 0.9694
