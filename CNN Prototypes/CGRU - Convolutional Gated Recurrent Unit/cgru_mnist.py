import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GRU, Dense, Input, Flatten
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data to the range [0, 1]
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)

# Reshape the data to add a channel dimension for the convolutional layers
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Function to create the GRU model with convolutional layers
def create_gru_model_with_conv(input_shape, gru_units, output_dim):
    inputs = Input(shape=input_shape)
    
    # Add convolutional layers
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = Flatten()(x)  # Flatten the output for the GRU layer

    # Add GRU layer
    x = tf.keras.layers.Reshape((-1, x.shape[-1]))(x)  # Reshape for GRU input
    gru_output = GRU(gru_units, return_sequences=False)(x)

    # Add output layer
    outputs = Dense(output_dim, activation='softmax')(gru_output)

    # Define the model
    model = Model(inputs, outputs)
    return model

# Model parameters
input_shape = (28, 28, 1)  # Input shape: 28x28 images with 1 channel
gru_units = 128             # Number of GRU units
output_dim = 10             # Number of output classes (digits 0-9)

# Create the GRU model with convolutional layers
model = create_gru_model_with_conv(input_shape, gru_units, output_dim)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val),
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Display the model summary
model.summary()

# Convolutional Standard Gated Recurrent Unit (CGRU)
# python cgru_mnist.py
# Test Accuracy: 0.9905