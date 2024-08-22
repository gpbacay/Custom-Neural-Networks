import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Define Relational Graph Convolutional Network (R-GCN) Layer
class RelationalGCNLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_relations, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_relations = num_relations
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        # Create dense layers for each relation
        self.dense_layers = [tf.keras.layers.Dense(self.units, kernel_regularizer=tf.keras.regularizers.l2(0.01)) for _ in range(self.num_relations)]
    
    def call(self, inputs):
        x = inputs
        output = tf.zeros((tf.shape(x)[0], self.units), dtype=x.dtype)
        
        # Apply dense layers for each relation and sum the results
        for i in range(self.num_relations):
            h = self.dense_layers[i](x)
            output += h
        
        # Apply activation function if specified
        if self.activation:
            output = self.activation(output)
        
        return output

# Define R-GCN Model
class RGCN(tf.keras.layers.Layer):
    def __init__(self, num_relations, units, **kwargs):
        super().__init__(**kwargs)
        # Define R-GCN layers with increasing number of units
        self.conv1 = RelationalGCNLayer(units, num_relations, activation='relu')
        self.conv2 = RelationalGCNLayer(units * 2, num_relations, activation='relu')
        self.conv3 = RelationalGCNLayer(units * 4, num_relations, activation='relu')
    
    def call(self, inputs):
        # Apply R-GCN layers sequentially
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

# Define the complete model architecture
def create_rgcn_model(input_shape, output_dim, units):
    inputs = Input(shape=input_shape)
    
    # CNN layers for feature extraction
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    
    # Flatten the spatial dimensions
    x = Flatten()(x)
    
    # Apply R-GCN for graph-based reasoning
    rgcn = RGCN(num_relations=4, units=units)
    x = rgcn(x)
    
    # Final classification layers
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Load and preprocess MNIST data
def load_and_preprocess_data():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Split training data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    # Normalize pixel values to range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Add channel dimension
    x_train = x_train[..., tf.newaxis]
    x_val = x_val[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Convert labels to one-hot encoded format
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main():
    # Set hyperparameters
    input_shape = (28, 28, 1)
    output_dim = 10
    units = 64
    num_epochs = 10
    batch_size = 64

    # Load and preprocess data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    # Define data augmentation function
    def augment_image(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_brightness(image, 0.2)
        return image, label
    
    # Create tf.data.Dataset for efficient data loading and augmentation
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.map(augment_image)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Create the model
    model = create_rgcn_model(input_shape, output_dim, units)

    # Define callbacks for early stopping and learning rate reduction
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_accuracy:.4f}')

if __name__ == "__main__":
    main()







# Spatial Relational Graph Convolutional Network (RGCN)
# python srgcn_mnist.py
# Test Accuracy: 0.9671



