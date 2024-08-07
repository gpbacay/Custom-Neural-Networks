from tensorflow.keras import layers, models, Input
import tensorflow as tf
import numpy as np

class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(GraphAttentionLayer, self).__init__()
        self.output_dim = output_dim
        self.dense = layers.Dense(output_dim)
        self.attention_dense = layers.Dense(1, use_bias=True)
    
    def call(self, inputs):
        x = self.dense(inputs)
        attention_scores = self.attention_dense(x)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        context = tf.reduce_sum(attention_weights * x, axis=1)
        return context

def create_gcalnn_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = tf.reshape(inputs, (1, input_dim))  # Reshape properly
    node_features, _ = mnist_to_graph(x)
    x = GraphAttentionLayer(64)(node_features)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(output_dim, activation='softmax')(x)
    return models.Model(inputs, outputs)

# Example usage:
def main():
    input_dim = 28*28
    output_dim = 10
    model = create_gcalnn_model(input_dim, output_dim)
    model.summary()

if __name__ == "__main__":
    main()


# python gnn_prototype_tensorflow.py