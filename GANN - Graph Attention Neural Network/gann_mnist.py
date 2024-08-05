import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Simple Graph Attention Layer
class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(GraphAttentionLayer, self).__init__()
        self.output_dim = output_dim
        self.dense = Dense(output_dim)

    def call(self, inputs):
        x = self.dense(inputs)
        attention = tf.nn.softmax(tf.matmul(x, tf.transpose(x, [0, 2, 1])))
        return tf.matmul(attention, x)

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create model
def create_gann_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Reshape((1, input_dim))(inputs)
    x = GraphAttentionLayer(64)(x)
    x = Reshape((64,))(x)
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
