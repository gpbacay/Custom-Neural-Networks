import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class LNNStep(layers.Layer):
    def __init__(self, reservoir_dim, input_dim, spectral_radius, leak_rate, **kwargs):
        super(LNNStep, self).__init__(**kwargs)
        self.reservoir_dim = reservoir_dim
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate

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
        weights = tf.random.normal(shape)
        eigenvalues, _ = tf.linalg.eig(weights)
        max_eigenvalue = tf.reduce_max(tf.abs(eigenvalues))
        return weights * (self.spectral_radius / max_eigenvalue)

    def call(self, inputs, states):
        prev_state = states[0]
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_part + reservoir_part)
        return state, [state]

    @property
    def state_size(self):
        return (self.reservoir_dim,)

def create_cllnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, lstm_units, output_dim):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    cnn_output_shape = x.shape[1]
    x = layers.Reshape((1, cnn_output_shape))(x)

    lnn_layer = layers.RNN(LNNStep(reservoir_dim, cnn_output_shape, spectral_radius, leak_rate), return_sequences=True)
    lnn_output = lnn_layer(x)

    lstm_output = layers.LSTM(lstm_units)(lnn_output)
    outputs = layers.Dense(output_dim, activation='softmax')(lstm_output)

    return models.Model(inputs, outputs)

def normalize_data(x):
    return StandardScaler().fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    x_train = normalize_data(x_train)
    x_val = normalize_data(x_val)
    x_test = normalize_data(x_test)

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_val = x_val.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)
    y_test = tf.keras.utils.to_categorical(y_test)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main():
    # Set hyperparameters
    input_shape = (28, 28, 1)
    reservoir_dim = 1000
    spectral_radius = 1.5
    leak_rate = 0.3
    lstm_units = 50
    output_dim = 10
    num_epochs = 10
    batch_size = 128

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    model = create_cllnn_model(input_shape, reservoir_dim, spectral_radius, leak_rate, lstm_units, output_dim)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),  # Use data augmentation
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Display the model summary
    model.summary()

if __name__ == "__main__":
    main()


# Convolutional LSTM Liquid Nueral Network (CLLNN)
# python cllnn_mnist.py
# Test Accuracy: 0.9921