import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
import spektral
from spektral.data import Dataset, Graph
from spektral.data.loaders import DisjointLoader
from scipy import sparse

# Define R-GCN Layer
class RelationalGCNLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_relations, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_relations = num_relations
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        self.dense_layers = [tf.keras.layers.Dense(self.units) for _ in range(self.num_relations)]
        self.built = True
    
    def call(self, inputs):
        x, a = inputs
        output = tf.zeros((tf.shape(x)[0], self.units), dtype=x.dtype)
        
        for i in range(self.num_relations):
            h = self.dense_layers[i](x)
            output += tf.sparse.sparse_dense_matmul(a, h)
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output

# Define R-GCN Model
class RGCN(tf.keras.Model):
    def __init__(self, num_relations, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = RelationalGCNLayer(32, num_relations, activation='relu')
        self.conv2 = RelationalGCNLayer(32, num_relations, activation='relu')
        self.dense = Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x, a, i = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        return self.dense(tf.math.segment_mean(x, i))

# Define Dataset Class
class MNISTGraphDataset(Dataset):
    def __init__(self, num_nodes=784, **kwargs):
        self.num_nodes = num_nodes
        super().__init__(**kwargs)
    
    def read(self):
        (x_train, y_train), _ = mnist.load_data()
        x_train = x_train.reshape(-1, self.num_nodes)
        enc = OneHotEncoder(sparse=False)
        y_train = enc.fit_transform(y_train.reshape(-1, 1))
        
        # Create a sparse identity matrix for the adjacency matrix
        a = sparse.eye(self.num_nodes, format='coo')
        
        graphs = []
        for i in range(len(x_train)):
            x = x_train[i].astype(float) / 255.0
            graphs.append(Graph(x=x[:, np.newaxis], a=a, y=y_train[i]))
        
        return graphs

# Load and preprocess data
def load_and_preprocess_data(batch_size):
    dataset = MNISTGraphDataset()
    loader = DisjointLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def main():
    input_shape = (784, 1)  # Flattened image with a single feature (pixel value)
    output_dim = 10
    num_epochs = 10
    batch_size = 32  # Reduced batch size to save memory
    
    loader = load_and_preprocess_data(batch_size)
    
    num_relations = 1  # Single relation type (identity adjacency)
    model = RGCN(num_relations=num_relations, num_classes=output_dim)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Training with data loader
    model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=num_epochs)
    
    # Create a new loader for testing
    test_loader = DisjointLoader(loader.dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_accuracy = model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch)
    print(f'Test accuracy: {test_accuracy:.4f}')

if __name__ == "__main__":
    main()



# Relational Graph Convolutional Network (RGCNN)
# python rgcn_mnist.py
# Test Accuracy: 



