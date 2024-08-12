import numpy as np
import tensorflow as tf
from spektral.data import Graph
from spektral.layers import GCNConv

# Sample data: Customers, Transactions, Products
customers = [
    {"Customer_ID": 1, "Name": "Alice"},
    {"Customer_ID": 2, "Name": "Bob"}
]

transactions = [
    {"Transaction_ID": 1, "Product_ID": 1, "Customer_ID": 1, "Price": 20},
    {"Transaction_ID": 2, "Product_ID": 2, "Customer_ID": 2, "Price": 30}
]

products = [
    {"Product_ID": 1, "Description": "Widget A"},
    {"Product_ID": 2, "Description": "Widget B"}
]

# Create a graph
num_nodes = len(customers) + len(products)
edges = []
for transaction in transactions:
    edges.append((transaction["Customer_ID"] - 1, transaction["Product_ID"] - 1 + len(customers)))

# Create adjacency matrix
adjacency = np.zeros((num_nodes, num_nodes))
for edge in edges:
    adjacency[edge] = 1
    adjacency[edge[::-1]] = 1  # Undirected graph

# Node features
node_features = np.zeros((num_nodes, 1))
for i in range(len(customers)):
    node_features[i] = 1  # Example feature for customers

# Create a Spektral graph
graph = Graph(x=node_features, a=adjacency)

# Define a simple GNN model using TensorFlow
class GNNModel(tf.keras.Model):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(1, activation='relu')  # Adjust input size
        self.conv2 = GCNConv(2, activation='relu')  # Adjust output size

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        return x

# Prepare data
x = tf.convert_to_tensor(graph.x, dtype=tf.float32)
a = tf.convert_to_tensor(graph.a, dtype=tf.float32)

# Train the model
model = GNNModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.MeanSquaredError()

# Dummy target for training
target = tf.convert_to_tensor([[1], [0], [0], [0]], dtype=tf.float32)

# Training loop
for epoch in range(50):
    with tf.GradientTape() as tape:
        output = model([x, a])
        loss = loss_fn(target, output)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

# Evaluation
predictions = model([x, a])
predicted_classes = tf.argmax(predictions, axis=1)
true_classes = tf.argmax(target, axis=1)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_classes, true_classes), dtype=tf.float32))
final_loss = loss_fn(target, predictions)

print(f'Final Loss: {final_loss.numpy()}')
print(f'Accuracy: {accuracy.numpy()}')



# Relational GNN (RGNN)
# python rgnn_v1.py
# Test Accuracy: Error