import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# Improved Graph Attention Layer
class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(GraphAttentionLayer, self).__init__()
        self.output_dim = output_dim
        self.dense = Dense(output_dim)
        self.attention_dense = Dense(1, use_bias=False)  # Added for improved attention calculation

    def call(self, inputs):
        x = self.dense(inputs)
        attention_scores = self.attention_dense(x)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        context = tf.reduce_sum(attention_weights * x, axis=1)  # Aggregate information
        return context

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Convert MNIST images to graph representation
def mnist_to_graph(images):
    num_samples = images.shape[0]
    num_nodes = images.shape[1]
    
    # Create a simple adjacency matrix (fully connected for simplicity)
    adjacency_matrix = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    
    # Node features are the pixel values
    node_features = images
    
    return node_features, adjacency_matrix

# Create model
def create_gann_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((1, input_dim))(inputs)
    x, adjacency_matrix = mnist_to_graph(x)
    x = GraphAttentionLayer(64)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(output_dim, activation='softmax')(x)
    return Model(inputs, outputs)

# Set up model parameters
input_dim = 28 * 28
output_dim = 10

# Create and compile the model
model = create_gann_model(input_dim, output_dim)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")


# Graph Attention Neural Network (GANN)
# python gann_mnist.py
# Test Accuracy: 0.9671
