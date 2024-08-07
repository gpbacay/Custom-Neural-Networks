import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Input, Flatten, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class GraphAttentionLayer(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.W = Dense(units, activation='tanh')
        self.V = Dense(1)

    def call(self, inputs):
        score = self.V(self.W(inputs))
        attention_weights = keras.activations.softmax(score, axis=1)
        context_vector = inputs * attention_weights
        return context_vector

class LNNLayer(keras.layers.Layer):
    def __init__(self, units, spectral_radius=0.9, leak_rate=0.2, return_sequences=True, **kwargs):
        super(LNNLayer, self).__init__(**kwargs)
        self.units = units
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.units, self.units),
                                 initializer='glorot_uniform',
                                 name='reservoir_weights')
        self.W_in = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer='glorot_uniform',
                                    name='input_weights')
        
        eigenvalues, _ = tf.linalg.eig(self.W)
        max_eigenvalue = tf.reduce_max(tf.abs(eigenvalues))
        self.W.assign(self.W * (self.spectral_radius / max_eigenvalue))

    def call(self, inputs):
        def step(prev_output, current_input):
            x = tf.matmul(current_input, self.W_in) + tf.matmul(prev_output, self.W)
            output = (1 - self.leak_rate) * prev_output + self.leak_rate * tf.tanh(x)
            return output

        initial_state = tf.zeros((tf.shape(inputs)[0], self.units))
        outputs = tf.scan(step, tf.transpose(inputs, [1, 0, 2]), initializer=initial_state)
        outputs = tf.transpose(outputs, [1, 0, 2])

        return outputs if self.return_sequences else outputs[:, -1, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units) if self.return_sequences else (input_shape[0], self.units)

def create_gcallnn_model(input_shape, lnn_units, lstm_units, output_dim):
    inputs = Input(shape=input_shape)
    
    # CNN layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Prepare for attention
    x = TimeDistributed(Flatten())(x)
    
    # Attention layer
    x = GraphAttentionLayer(64)(x)
    
    # LNN layer
    x = LNNLayer(lnn_units, return_sequences=True)(x)
    
    # LSTM layer
    x = LSTM(lstm_units)(x)
    
    # Output layer
    outputs = Dense(output_dim, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Set hyperparameters
input_shape = (28, 28, 1)
lnn_units = 512
lstm_units = 128
output_dim = 10

# Create and compile the model
model = create_gcallnn_model(input_shape, lnn_units, lstm_units, output_dim)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")


# Graph Convolutional Attention LSTM Liquid Neural Network (GCALLNN)
# python gcallnn_mnist.py
# Test Accuracy: 0.9845