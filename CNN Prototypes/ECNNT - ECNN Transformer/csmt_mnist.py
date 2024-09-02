import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, GlobalAveragePooling2D, Dropout, BatchNormalization, Reshape, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

def create_smect_model(input_shape, output_dim, d_model=64, self_modeling_weight=0.1):
    inputs = Input(shape=input_shape)
    
    # Adaptive Convolutional Layers
    def adaptive_conv(x, filters, kernel_size, strides=1):
        conv = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(x)
        norm = BatchNormalization()(conv)
        return tf.keras.layers.ReLU()(norm)
    
    x = adaptive_conv(inputs, 32, kernel_size=3, strides=2)
    x = adaptive_conv(x, 64, kernel_size=3, strides=2)
    x = adaptive_conv(x, 128, kernel_size=3, strides=2)
    
    # Apply Global Average Pooling
    model_features = GlobalAveragePooling2D()(x)
    x = Reshape((1, model_features.shape[-1]))(model_features)  # Add seq_len dimension for Dense layer

    # Dynamic Self-Modeling Mechanism with Multi-Head Attention
    self_modeling_dense = Dense(d_model, activation='relu')(x)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=d_model)(self_modeling_dense, self_modeling_dense)
    self_modeling_output = Dense(model_features.shape[-1])(attention_output)
    
    x = Flatten()(x)
    
    # Final Classification Layers
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    classification_output = Dense(output_dim, activation='softmax')(x)

    # Create the Model with Two Outputs
    model = Model(inputs, [classification_output, self_modeling_output])

    # Compile the Model with a Combined Loss Function and Multiple Metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
        loss=['categorical_crossentropy', 'mse'],  # Classification loss + Self-modeling loss
        loss_weights=[1.0, self_modeling_weight],  # Weight for the Self-modeling task
        metrics=[['accuracy'], ['mse']]  # Metrics for each output
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

def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        labels = y_true[:, 0]
        y_pred = tf.nn.l2_normalize(y_pred, axis=-1)
        pos_pairs = tf.reduce_sum(labels * y_pred, axis=-1)
        neg_pairs = tf.reduce_sum((1 - labels) * y_pred, axis=-1)
        loss = tf.maximum(0.0, margin - pos_pairs + neg_pairs)
        return tf.reduce_mean(loss)
    return loss

def main():
    input_shape = (28, 28, 1)
    output_dim = 10
    num_epochs = 10
    batch_size = 64

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    # Data Augmentation
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

    # Train the Model
    model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=num_epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr, lr_schedule]
    )

    # Evaluate the Model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_accuracy:.4f}')

if __name__ == "__main__":
    main()




# Convolutional Self-Modeling Transformer (CSMT)
# python csmt_mnist.py
# Test Accuracy: 0.9541
