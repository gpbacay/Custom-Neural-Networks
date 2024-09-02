import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, GlobalAveragePooling2D, Dropout, Reshape, Layer
from tensorflow.keras.layers import BatchNormalization, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class SelfModelingMechanism:
    def __init__(self, initial_reservoir_dim, max_reservoir_dim, min_reservoir_dim):
        self.reservoir_dim = initial_reservoir_dim
        self.max_reservoir_dim = max_reservoir_dim
        self.min_reservoir_dim = min_reservoir_dim
        self.performance_history = []
        self.structure_history = []
        self.num_meta_features = 4

    def adapt(self, performance):
        self.performance_history.append(performance)
        self.structure_history.append(self.reservoir_dim)

        if len(self.performance_history) >= 5:
            recent_trend = np.mean(self.performance_history[-5:]) - np.mean(self.performance_history[-10:-5])

            if recent_trend > 0.01:
                self.reservoir_dim = min(int(self.reservoir_dim * 1.1), self.max_reservoir_dim)
            elif recent_trend < -0.01:
                self.reservoir_dim = max(int(self.reservoir_dim * 0.9), self.min_reservoir_dim)

        return self.reservoir_dim

    def get_meta_features(self):
        if len(self.performance_history) < 10:
            return [0.0, 0.0, self.reservoir_dim / self.max_reservoir_dim, 0.0]
        return [
            np.mean(self.performance_history[-10:]),
            np.std(self.performance_history[-10:]),
            self.reservoir_dim / self.max_reservoir_dim,
            np.mean(self.structure_history[-10:]) / self.max_reservoir_dim
        ]

class MetaFeatureLayer(Layer):
    def __init__(self, self_modeling_mechanism, **kwargs):
        super().__init__(**kwargs)
        self.self_modeling_mechanism = self_modeling_mechanism

    def call(self, inputs):
        meta_features = self.self_modeling_mechanism.get_meta_features()
        meta_features = tf.convert_to_tensor(meta_features, dtype=tf.float32)
        meta_features = tf.reshape(meta_features, (1, -1))
        return tf.tile(meta_features, [tf.shape(inputs)[0], 1])

def efficientnet_block(inputs, filters, expansion_factor, stride):
    expanded_filters = filters * expansion_factor
    x = Conv2D(expanded_filters, kernel_size=1, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(expanded_filters, kernel_size=3, padding='same', strides=stride, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if stride == 1 and inputs.shape[-1] == x.shape[-1]:
        x = Add()([inputs, x])
    return x

def create_smect_model(input_shape, output_dim, d_model=64, self_modeling_weight=0.1):
    inputs = Input(shape=input_shape)
    
    # EfficientNet-based Convolutional layers for feature extraction
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2)
    
    # Apply self-modeling layer
    model_features = GlobalAveragePooling2D()(x)
    x = Reshape((1, model_features.shape[-1]))(model_features)  # Add seq_len dimension for further processing
    
    # Dynamic Self-Modeling Mechanism
    self_modeling_dense = Dense(d_model, activation='relu')(x)
    self_modeling_output = Dense(model_features.shape[-1])(self_modeling_dense)
    
    # Integrate MetaFeatureLayer
    self_modeling_mechanism = SelfModelingMechanism(initial_reservoir_dim=128, max_reservoir_dim=512, min_reservoir_dim=64)
    meta_feature_layer = MetaFeatureLayer(self_modeling_mechanism)(x)
    
    x = Flatten()(x)
    
    # Final classification layers
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    classification_output = Dense(output_dim, activation='softmax')(x)

    # Create the model with three outputs
    model = Model(inputs, [classification_output, self_modeling_output, meta_feature_layer])

    # Compile the model with a combined loss function and multiple metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
        loss=['categorical_crossentropy', 'mse', 'mse'],  # Classification loss + Self-modeling loss + Meta-feature loss
        loss_weights=[1.0, self_modeling_weight, 0.0],  # Weight for the self-modeling task, meta-feature loss can be adjusted
        metrics=[['accuracy'], ['mse'], ['mse']]  # Metrics for each output
    )
    
    return model

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
    num_epochs = 10
    batch_size = 64

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    model = create_smect_model(input_shape, output_dim)

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * (0.1 ** (epoch // 5)))

    # Train the model
    model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=num_epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr, lr_schedule]
    )

    # Evaluate the model
    eval_results = model.evaluate(x_test, y_test)
    test_loss = eval_results[0]
    self_modeling_loss = eval_results[1]
    meta_feature_loss = eval_results[2] if len(eval_results) > 2 else None
    test_accuracy = eval_results[3] if len(eval_results) > 3 else None

    print(f'Test loss: {test_loss:.4f}')
    print(f'Self-modeling loss: {self_modeling_loss:.4f}')
    if meta_feature_loss is not None:
        print(f'Meta-feature loss: {meta_feature_loss:.4f}')
    if test_accuracy is not None:
        print(f'Test accuracy: {test_accuracy:.4f}')

if __name__ == "__main__":
    main()




# Dynamic Self-Modeling Convolutional Neural Network (DSM-CNN)
# python dsmcnn_mnist.py
# Test Accuracy: 0.9853
