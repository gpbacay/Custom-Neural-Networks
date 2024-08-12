import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

class SpatialGraphConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(SpatialGraphConvolutionLayer, self).__init__()
        self.output_dim = output_dim
        self.dense = Dense(output_dim, use_bias=False)
        self.spatial_dense = Dense(1, use_bias=True)

    def call(self, node_features, spatial_coords):
        num_nodes = tf.shape(spatial_coords)[1]
        
        # Reshape node_features to match spatial_coords shape
        node_features = tf.reshape(node_features, (-1, num_nodes, 1))
        
        # Compute spatial differences
        spatial_diff = spatial_coords[:, :, None, :] - spatial_coords[:, None, :, :]
        spatial_diff = tf.reshape(spatial_diff, (-1, spatial_diff.shape[-1]))
        attention_scores = self.spatial_dense(spatial_diff)
        attention_scores = tf.reshape(attention_scores, (-1, num_nodes, num_nodes))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply graph convolution
        x = self.dense(node_features)
        context = tf.matmul(attention_weights, x)
        return context  # This will be a 3D tensor

def mnist_to_graph(images):
    num_samples = images.shape[0]
    num_nodes = images.shape[1]
    
    # Create a fully connected adjacency matrix
    adjacency_matrix = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    
    # Node features are the pixel values
    node_features = images
    
    # Create coordinates
    coords = np.array([[i, j] for i in range(28) for j in range(28)])
    coords = np.expand_dims(coords, axis=0).repeat(num_samples, axis=0)
    
    return node_features, adjacency_matrix, coords

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    x_train, _, coords_train = mnist_to_graph(x_train)
    x_test, _, coords_test = mnist_to_graph(x_test)
    
    return (x_train, y_train, coords_train), (x_test, y_test, coords_test)

def create_sgcn_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    coords = Input(shape=(28*28, 2))  # Spatial coordinates input

    x = Flatten()(inputs)  # Use Flatten instead of tf.reshape
    x = SpatialGraphConvolutionLayer(64)(x, coords)
    x = Flatten()(x)  # Flatten the output of SpatialGraphConvolutionLayer
    x = Dense(32, activation='relu')(x)
    outputs = Dense(output_dim, activation='softmax')(x)
    
    model = Model(inputs=[inputs, coords], outputs=outputs)
    return model

def main():
    input_dim = 28 * 28
    output_dim = 10
    num_epochs = 10
    batch_size = 64

    (x_train, y_train, coords_train), (x_test, y_test, coords_test) = load_and_preprocess_data()
    
    model = create_sgcn_model(input_dim, output_dim)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        [x_train, coords_train],
        y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1
    )

    test_loss, test_accuracy = model.evaluate([x_test, coords_test], y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()




# Graph Attention Neural Network (GANN)
# python gann_mnist.py
# Test Accuracy: 
