import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, Layer
import numpy as np

# Custom Graph Transformer Layer
class GraphTransformerLayer(Layer):
    def __init__(self, d_model):
        super(GraphTransformerLayer, self).__init__()
        self.d_model = d_model
        self.attention = MultiHeadAttention(num_heads=4, key_dim=d_model)
        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, x):
        # Apply Multi-Head Attention
        attention_output = self.attention(x, x)
        x = self.layer_norm(x + attention_output)
        return x

def create_transformer_with_graph_model(input_shape, output_shape, d_model=64):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Reshape((28, 28))(inputs)  # Reshape to (28, 28) sequence
    
    # Apply Graph Transformer Layer
    x = GraphTransformerLayer(d_model)(x)
    
    # Pooling and Dense Layers
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(output_shape, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Example usage for MNIST
input_shape = (28, 28, 1)
output_shape = 10
model = create_transformer_with_graph_model(input_shape, output_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# Evaluate the model and print the test accuracy
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")



# python gtnn_mnist.py
# Test Accuracy: 0.8520