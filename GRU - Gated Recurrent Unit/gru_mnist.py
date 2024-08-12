import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Input
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

# Reshape the data to fit the GRU input format (batch_size, timesteps, input_dim)
x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_val = x_val.reshape(x_val.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)

# Function to create the GRU model
def create_gru_model(input_shape, gru_units, output_dim):
    inputs = Input(shape=input_shape)
    gru_output = GRU(gru_units, return_sequences=False)(inputs)
    outputs = Dense(output_dim, activation='softmax')(gru_output)
    model = Model(inputs, outputs)
    return model

# Model parameters
input_shape = (28, 28)  # Input shape: 28 timesteps with 28 features
gru_units = 128         # Number of GRU units
output_dim = 10         # Number of output classes (digits 0-9)

# Create the GRU model
model = create_gru_model(input_shape, gru_units, output_dim)

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




# Standard Gated Recurrent Unit (GRU)
# python gru_mnist.py
# Test Accuracy: 0.9855