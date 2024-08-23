import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Custom Layer for Spatial Processing
class SpatialReservoirLayer(layers.Layer):
    def __init__(self, reservoir_dim, input_dim, spectral_radius, leak_rate, **kwargs):
        super().__init__(**kwargs)
        self.reservoir_dim = reservoir_dim
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.reservoir_weights = None
        self.input_weights = None

    def build(self, input_shape):
        self.reservoir_weights = self.add_weight(
            shape=(self.reservoir_dim, self.reservoir_dim),
            initializer=self.reservoir_initializer,
            name='reservoir_weights'
        )
        self.input_weights = self.add_weight(
            shape=(self.reservoir_dim, self.input_dim),
            initializer='glorot_uniform',
            name='input_weights'
        )

    def reservoir_initializer(self, shape, dtype=None):
        weights = np.random.randn(*shape)
        weights *= self.spectral_radius / np.max(np.abs(np.linalg.eigvals(weights)))
        return tf.convert_to_tensor(weights, dtype=tf.float32)

    def call(self, inputs):
        prev_state = tf.zeros((tf.shape(inputs)[0], self.reservoir_dim))
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_part + reservoir_part)
        return state

# Custom Layer for Message Passing
class MessagePassingLayer(layers.Layer):
    def __init__(self, num_relations, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_relations = num_relations
        self.output_dim = output_dim

    def build(self, input_shape):
        self.relation_weights = self.add_weight(
            shape=(self.num_relations, input_shape[-1], self.output_dim),
            initializer='glorot_uniform',
            name='relation_weights'
        )

    def call(self, inputs):
        messages = []
        for i in range(self.num_relations):
            message = tf.matmul(inputs, self.relation_weights[i])
            messages.append(message)
        return tf.reduce_sum(messages, axis=0)

# Create Spatiotemporal Reservoir Model
def create_spatiotemporal_model(input_shape, reservoir_dim, spectral_radius, leak_rate, output_dim, num_relations):
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    
    input_dim = x.shape[-1]

    # Spatial Reservoir Layer
    spatial_reservoir = SpatialReservoirLayer(reservoir_dim, input_dim, spectral_radius, leak_rate)
    reservoir_output = spatial_reservoir(x)

    # Message Passing Layer
    message_passing = MessagePassingLayer(num_relations, reservoir_dim)
    multi_relational_output = message_passing(reservoir_output)

    # Combine and output
    combined_features = layers.Concatenate()([reservoir_output, multi_relational_output])
    outputs = layers.Dense(output_dim, activation='softmax')(combined_features)

    model = models.Model(inputs, outputs)
    return model

# Preprocess MNIST Data
def preprocess_mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, 28*28)).reshape(-1, 28, 28, 1)
    x_val = scaler.transform(x_val.reshape(-1, 28*28)).reshape(-1, 28, 28, 1)
    x_test = scaler.transform(x_test.reshape(-1, 28*28)).reshape(-1, 28, 28, 1)

    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)
    y_test = keras.utils.to_categorical(y_test)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main():
    # Set hyperparameters
    input_shape = (28, 28, 1)
    reservoir_dim = 1000
    spectral_radius = 1.5
    leak_rate = 0.3
    output_dim = 10
    num_relations = 2
    num_epochs = 10

    # Prepare data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_mnist_data()

    # Create Spatiotemporal Reservoir Model
    model = create_spatiotemporal_model(input_shape, reservoir_dim, spectral_radius, leak_rate, output_dim, num_relations)

    # Compile and train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs, batch_size=64, validation_data=(x_val, y_val), callbacks=[early_stopping, reduce_lr])

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()


# Relational Liquid Nueral Network (LNN)
# python rlnn_mnist_v1.py
# Test Accuracy: 0.9752
