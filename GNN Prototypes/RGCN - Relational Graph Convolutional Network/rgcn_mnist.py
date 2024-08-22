import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping

class RGCNLayer(tf.keras.layers.Layer):
    """
    Custom RGCN Layer with message passing for relational graph convolution.
    """
    def __init__(self, units, num_relations, activation=None):
        super(RGCNLayer, self).__init__()
        self.units = units
        self.num_relations = num_relations
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        """
        Initialize weights and bias for each relation.
        """
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
        """
        Forward pass with message passing in the RGCN layer.
        """
        # Aggregation
        aggregated_messages = []
        for i in range(self.num_relations):
            # Message passing: aggregate messages from neighbors
            messages = tf.matmul(adj_matrices[i], inputs)  # Aggregation step
            aggregated_messages.append(tf.matmul(messages, self.weight_list[i]))
        
        # Combine messages from different relations
        aggregated_messages = tf.reduce_sum(tf.stack(aggregated_messages, axis=0), axis=0)
        
        # Linear transformation and bias addition
        output = tf.nn.bias_add(aggregated_messages, self.bias)
        
        return self.activation(output) if self.activation else output

def create_adjacency_matrices(image_shape, num_relations=4):
    """
    Create adjacency matrices for the graph based on image spatial structure.
    """
    height, width = image_shape
    num_nodes = height * width
    offsets = tf.constant([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=tf.int32)
    
    # Generate indices for each pixel in the image
    indices = tf.reshape(tf.stack(tf.meshgrid(tf.range(height), tf.range(width), indexing='ij'), axis=-1), [-1, 2])
    indices = tf.cast(indices, tf.int32)
    
    adj_matrices = []
    for i in range(num_relations):
        # Calculate neighbor indices based on the offset
        neighbor_indices = indices + offsets[i]
        valid_mask = tf.reduce_all((neighbor_indices >= 0) & (neighbor_indices < [height, width]), axis=1)
        valid_indices = tf.boolean_mask(indices, valid_mask)
        valid_neighbors = tf.boolean_mask(neighbor_indices, valid_mask)
        
        # Convert 2D indices to 1D for adjacency matrix
        flat_indices = valid_indices[:, 0] * width + valid_indices[:, 1]
        flat_neighbors = valid_neighbors[:, 0] * width + valid_neighbors[:, 1]
        
        # Create adjacency matrix
        adj_matrix = tf.scatter_nd(
            tf.stack([flat_indices, flat_neighbors], axis=1),
            tf.ones(tf.shape(flat_indices), dtype=tf.float32),
            [num_nodes, num_nodes]
        )
        adj_matrices.append(adj_matrix)
    
    return tf.stack(adj_matrices)

def create_rgcn_model(input_shape, num_classes, num_relations):
    """
    Build and compile the RGCN model.
    """
    inputs = Input(shape=input_shape)
    
    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Reshape for RGCN layer
    x = Reshape((-1, 64))(x)
    
    # Create adjacency matrices
    adj_matrices = create_adjacency_matrices((14, 14), num_relations)  # Adjusted for downsampling
    
    # Apply RGCN layers
    x = RGCNLayer(128, num_relations, activation='relu')(x, adj_matrices)
    x = Dropout(0.5)(x)
    x = RGCNLayer(64, num_relations, activation='relu')(x, adj_matrices)
    
    # Global average pooling and output layer
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to the range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomFlip('horizontal')
    ])
    
    x_train = data_augmentation(x_train)
    
    return x_train, y_train, x_test, y_test

def main():
    """
    Main function to train and evaluate the RGCN model.
    """
    # Load and preprocess MNIST data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # Create and compile the model
    model = create_rgcn_model((14, 14, 64), 10, 4)  # Adjusted input shape for RGCN layer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Early stopping callback to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=20,  # Increased epochs for better training
        validation_split=0.1,
        callbacks=[early_stopping]
    )
    
    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()




# Spatial Relational Graph Convolutional Network (RGCN)
# python srgcn_mnist.py
# Test Accuracy: 0.9678



