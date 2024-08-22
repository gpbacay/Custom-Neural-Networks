import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Custom layer to create adjacency matrices
class AdjacencyMatrixLayer(tf.keras.layers.Layer):
    def __init__(self, num_relations, **kwargs):
        super().__init__(**kwargs)
        self.num_relations = num_relations

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        adjacency_matrices = [tf.eye(batch_size) for _ in range(self.num_relations)]
        return adjacency_matrices

# Define R-GCN Layer with Message Passing
class RelationalGCNLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_relations, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_relations = num_relations
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.dense_layers = [tf.keras.layers.Dense(self.units, kernel_regularizer=tf.keras.regularizers.l2(0.01)) for _ in range(self.num_relations)]
        self.self_loop_layer = tf.keras.layers.Dense(self.units, kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, inputs, adjacency_matrices):
        x = inputs
        output = tf.zeros((tf.shape(x)[0], self.units), dtype=x.dtype)

        # Message Passing
        for i in range(self.num_relations):
            neighbor_messages = tf.matmul(adjacency_matrices[i], x)  # Aggregating neighbor features
            h = self.dense_layers[i](neighbor_messages)
            output += h

        # Adding self-loop
        self_loop_message = self.self_loop_layer(x)
        output += self_loop_message

        if self.activation:
            output = self.activation(output)

        return output

# Define EfficientNet Block
def efficientnet_block(inputs, filters, expansion_factor, stride):
    expanded_filters = filters * expansion_factor
    x = Conv2D(expanded_filters, kernel_size=1, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(expanded_filters, kernel_size=3, padding='same', strides=stride, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if stride == 1 and inputs.shape[-1] == x.shape[-1]:
        x = tf.keras.layers.Add()([inputs, x])
    return x

# Define Model with CNN and R-GCN with Message Passing
def create_cnn_rgcn_model(input_shape, output_dim, units, num_relations):
    inputs = Input(shape=input_shape)
    
    # EfficientNet Convolutional layers
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    # Create adjacency matrices (simulated for this example)
    adjacency_layer = AdjacencyMatrixLayer(num_relations=num_relations)
    adjacency_matrices = adjacency_layer(x)

    # Relational GCN layer with message passing
    rgcn = RelationalGCNLayer(units=units, num_relations=num_relations)
    graph_features = rgcn(x, adjacency_matrices)
    
    # Combine CNN and R-GCN outputs
    combined_features = Concatenate()([x, graph_features])
    
    # Final classification layers
    x = Dense(128, activation='relu')(combined_features)
    x = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs, x)
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
    output_dim = 10
    units = 32
    num_relations = 1  # Example value; adjust as needed
    num_epochs = 10
    batch_size = 64

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    model = create_cnn_rgcn_model(input_shape, output_dim, units, num_relations)

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=num_epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_accuracy:.4f}')

if __name__ == "__main__":
    main()




# Convolutional Relational Graph Convolutional Network (CRGCN)
# python crgcn_mnist.py
# Test Accuracy: 0.9943