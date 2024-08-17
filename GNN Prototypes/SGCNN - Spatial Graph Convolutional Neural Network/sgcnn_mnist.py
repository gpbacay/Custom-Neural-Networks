import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class SpatialGraphConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(SpatialGraphConvolutionLayer, self).__init__()
        self.output_dim = output_dim
        self.dense = Dense(output_dim, use_bias=False)
        self.spatial_dense = Dense(1, use_bias=True)

    def build(self, input_shape):
        self.coords = self.add_weight(shape=(28*28, 2), initializer='uniform', trainable=False, name='coords')
        super(SpatialGraphConvolutionLayer, self).build(input_shape)

    def call(self, inputs):
        node_features = tf.reshape(inputs, (-1, 28*28, 1))
        
        # Pre-compute spatial differences
        spatial_diff = self.coords[:, None, :] - self.coords[None, :, :]
        spatial_diff = tf.reshape(spatial_diff, (-1, 2))
        
        # Compute attention scores
        attention_scores = self.spatial_dense(spatial_diff)
        attention_scores = tf.reshape(attention_scores, (28*28, 28*28))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply graph convolution
        x = self.dense(node_features)
        context = tf.matmul(attention_weights, x)
        return context

@tf.function
def preprocess_images(images):
    return tf.cast(images, tf.float32) / 255.0

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def create_sgcn_model(input_dim, output_dim):
    inputs = Input(shape=(28, 28))
    x = Flatten()(inputs)
    x = SpatialGraphConvolutionLayer(64)(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(output_dim, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    input_dim = 28 * 28
    output_dim = 10
    num_epochs = 10
    batch_size = 128

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    model = create_sgcn_model(input_dim, output_dim)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()





# Spatial Graph Convolutional Neural Network (SGCNN)
# python sgcnn_mnist.py
# Test Accuracy: 0.1128 (slow)

