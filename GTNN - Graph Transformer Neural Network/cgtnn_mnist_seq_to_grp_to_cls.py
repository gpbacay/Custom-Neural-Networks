import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 1. Image Encoder: CNN-based to sequence embeddings
def create_cnn_encoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128)(x)  # Sequence-based embeddings
    model = models.Model(inputs, x)
    return model

# 2. Sequence-to-Graph Decoder
class SequenceToGraphDecoder(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(SequenceToGraphDecoder, self).__init__()
        self.output_dim = output_dim
        self.dense = layers.Dense(self.output_dim, activation='relu')

    def call(self, sequence_embeddings):
        return self.dense(sequence_embeddings)

# 3. Graph-to-Classification Decoder
class GraphToClassificationDecoder(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super(GraphToClassificationDecoder, self).__init__()
        self.num_classes = num_classes
        self.dense = layers.Dense(self.num_classes, activation='softmax')

    def call(self, graph_embeddings):
        x = layers.Flatten()(graph_embeddings)
        return self.dense(x)

# Create full model
def create_model(input_shape, num_classes):
    # Image Input
    image_input = layers.Input(shape=input_shape)
    
    # Encoder
    cnn_features = create_cnn_encoder(input_shape)(image_input)
    
    # Sequence-to-Graph Decoder
    graph_embeddings = SequenceToGraphDecoder(256)(cnn_features)
    
    # Graph-to-Classification Decoder
    predictions = GraphToClassificationDecoder(num_classes)(graph_embeddings)
    
    model = models.Model(inputs=image_input, outputs=predictions)
    return model

# Load and preprocess data
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    x_train = x_train[..., tf.newaxis]
    x_val = x_val[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main():
    input_shape = (28, 28, 1)
    num_classes = 10
    num_epochs = 10
    batch_size = 64
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    model = create_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_val, y_val))
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()




# Convolutional Graph Transformer Neural Network (CGTNN): encoder(IMG to SEQ) to decoder(GRP to CLS)
# python cgtnn_mnist_seq_to_grp_to_cls.py
# Test Accuracy: 0.8943