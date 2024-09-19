import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class SpatioTemporalSummaryMixing(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff=None, dropout_rate=0.1):
        super(SpatioTemporalSummaryMixing, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        
        # Layers for local transformation (spatial aspect)
        self.local_dense1 = tf.keras.layers.Dense(self.d_ff, activation='gelu')
        self.local_dense2 = tf.keras.layers.Dense(d_model)
        self.local_dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Layers for summary function (temporal aspect)
        self.summary_dense1 = tf.keras.layers.Dense(self.d_ff, activation='gelu')
        self.summary_dense2 = tf.keras.layers.Dense(d_model)
        self.summary_dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Layers for combiner function (spatio-temporal combination)
        self.combiner_dense1 = tf.keras.layers.Dense(self.d_ff, activation='gelu')
        self.combiner_dense2 = tf.keras.layers.Dense(d_model)
        self.combiner_dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        # Local (spatial) transformation
        local_output = self.local_dense1(inputs)
        local_output = self.local_dense2(local_output)
        local_output = self.local_dropout(local_output, training=training)
        
        # Summary (temporal) function
        summary = self.summary_dense1(inputs)
        summary = self.summary_dense2(summary)
        summary = self.summary_dropout(summary, training=training)
        
        # Calculate mean summary (temporal)
        mean_summary = tf.reduce_mean(summary, axis=1, keepdims=True)
        
        # Repeat mean summary for each time step (temporal extension)
        mean_summary = tf.tile(mean_summary, [1, tf.shape(inputs)[1], 1])
        
        # Combine local (spatial) and summary (temporal) information
        combined = tf.concat([local_output, mean_summary], axis=-1)
        output = self.combiner_dense1(combined)
        output = self.combiner_dense2(output)
        output = self.combiner_dropout(output, training=training)
        
        # Residual connection and layer normalization
        return self.layer_norm(inputs + output)

def create_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Convolutional layers to extract spatial features
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    # Reshape to prepare for spatio-temporal processing
    x = tf.keras.layers.Reshape((-1, x.shape[-1]))(x)
    
    # Spatio-Temporal SummaryMixing layer
    x = SpatioTemporalSummaryMixing(d_model=64)(x)
    
    # Global average pooling (across time)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers for classification
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create and compile the model
input_shape = (28, 28, 1)  # MNIST image shape
num_classes = 10  # 10 digits

model = create_model(input_shape, num_classes)
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Final Test Accuracy: {test_accuracy:.4f}")



# Spatio-Temporal Summary Mixing Mechanism
# python stsmm.py
# Test Accuracy: 0.9706