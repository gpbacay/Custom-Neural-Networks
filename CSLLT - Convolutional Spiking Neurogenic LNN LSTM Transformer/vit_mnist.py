import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Define Vision Transformer Model (Encoder)
class VisionTransformer(layers.Layer):
    def __init__(self, patch_size, num_patches, embedding_dim, num_heads, num_layers):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.projection = layers.Dense(embedding_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=embedding_dim)
        self.encoder_layers = [
            layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embedding_dim // num_heads
            ) for _ in range(num_layers)
        ]
        self.mlp_head = models.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.GlobalAveragePooling1D()
        ])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        x = self.projection(inputs)
        positions = tf.range(start=0, limit=self.num_patches, dtype=tf.int32)
        x += self.position_embedding(positions)
        for layer in self.encoder_layers:
            x = layer(x, x)
        x = self.mlp_head(x)
        return x

# Define Transformer Decoder Layer
class TransformerDecoder(layers.Layer):
    def __init__(self, embedding_dim, num_heads, num_layers, num_classes):
        super(TransformerDecoder, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.decoder_layers = [
            layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embedding_dim // num_heads
            ) for _ in range(num_layers)
        ]
        self.mlp_head = models.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.GlobalAveragePooling1D(),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs, encoder_output):
        x = inputs
        for layer in self.decoder_layers:
            x = layer(query=x, value=encoder_output, key=encoder_output)
        x = self.mlp_head(x)
        return x

# Model Builder
def build_encoder_decoder_model(input_shape, patch_size, embedding_dim, num_heads, num_layers, num_classes):
    inputs = Input(shape=input_shape)
    num_patches = (input_shape[0] // patch_size[0]) * (input_shape[1] // patch_size[1])
    x = Reshape(target_shape=(num_patches, np.prod(patch_size)))(inputs)
    
    # Encoder
    encoder_output = VisionTransformer(
        patch_size=patch_size,
        num_patches=num_patches,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )(x)
    
    # Decoder
    decoder_output = TransformerDecoder(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes
    )(encoder_output, encoder_output)
    
    model = models.Model(inputs, decoder_output)
    return model

# Set hyperparameters
input_shape = (28, 28, 1)
patch_size = (7, 7)
embedding_dim = 64
num_heads = 4
num_layers = 2
num_classes = 10

# Create and compile the encoder-decoder model
encoder_decoder_model = build_encoder_decoder_model(
    input_shape=input_shape,
    patch_size=patch_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    num_classes=num_classes
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
]

encoder_decoder_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

x_train = x_train.astype(np.float32) / 255.0
x_val = x_val.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)

# Train the model
history = encoder_decoder_model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)

# Evaluate the model
test_loss, test_acc = encoder_decoder_model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc:.4f}')



# Vision Transformer Encoder-Decoder Model (ViT-ED)
# python vit_mnist.py
# Test Accuracy: Error