# Importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout, Flatten, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model
import librosa
import soundfile as sf
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Custom Keras layer implementing a spiking Liquid Neural Network (LNN)
class SpikingLNNStep(tf.keras.layers.Layer):
    def __init__(self, reservoir_weights, input_weights, leak_rate, max_reservoir_dim, spike_threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)
        self.leak_rate = leak_rate
        self.max_reservoir_dim = max_reservoir_dim
        self.spike_threshold = spike_threshold

    @property
    def state_size(self):
        return (self.max_reservoir_dim,)

    def call(self, inputs, states):
        # Implement the spiking LNN dynamics
        prev_state = states[0][:, :self.reservoir_weights.shape[0]]
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights, transpose_b=True)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_part + reservoir_part)
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)
        active_size = tf.shape(state)[-1]
        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_reservoir_dim - active_size])], axis=1)
        return padded_state, [padded_state]

# Initialize the reservoir and input weights for the LNN
def initialize_spiking_lnn_reservoir(input_dim, reservoir_dim, spectral_radius, max_reservoir_dim):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights

# Create the hybrid Transformer-Spiking LNN model
def create_transformer_spiking_model(input_dim, num_heads, ff_dim, num_blocks, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim):
    inputs = Input(shape=(None, input_dim))
    x = inputs
    
    # Transformer blocks
    for _ in range(num_blocks):
        x = MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)(x, x)
        x = Dropout(0.1)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x_ff = Dense(ff_dim, activation='relu')(x)
        x_ff = Dense(input_dim)(x_ff)
        x = x + x_ff
        x = Dropout(0.1)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    
    # Initialize and apply the Spiking LNN layer
    reservoir_weights, input_weights = initialize_spiking_lnn_reservoir(input_dim, reservoir_dim, spectral_radius, max_reservoir_dim)
    lnn_layer = tf.keras.layers.RNN(
        SpikingLNNStep(reservoir_weights, input_weights, leak_rate, max_reservoir_dim),
        return_sequences=True
    )
    
    def apply_spiking_lnn(x):
        lnn_output = lnn_layer(tf.expand_dims(x, axis=1))
        return Flatten()(lnn_output)
    
    def get_output_shape(input_shape):
        return (input_shape[0], max_reservoir_dim)

    lnn_output = Lambda(apply_spiking_lnn, output_shape=get_output_shape)(x)

    # Final dense layers for feature extraction
    x = Dense(128, activation='relu')(lnn_output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    model = tf.keras.Model(inputs, x)
    return model

# Load audio file using librosa or soundfile
def load_audio(file_path):
    print(f"Attempting to load audio file: {file_path}")
    try:
        y, sr = librosa.load(file_path, sr=None)
        print(f"Loaded with librosa: Sample rate {sr}, shape {y.shape}")
    except Exception as e:
        print(f"Error with librosa: {e}. Trying soundfile...")
        try:
            y, sr = sf.read(file_path)
            print(f"Loaded with soundfile: Sample rate {sr}, shape {y.shape}")
        except Exception as e:
            print(f"Error with soundfile: {e}")
            return None, None
    return y, sr

# Preprocess audio file to mel spectrogram
def preprocess_audio(file_path):
    y, sr = load_audio(file_path)
    if y is None or sr is None:
        print(f"Failed to load audio file: {file_path}")
        return None

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram

# Prepare dataset from multiple audio files
def prepare_audio_dataset(file_paths):
    features = []
    for file_path in file_paths:
        feature = preprocess_audio(file_path)
        if feature is not None:
            features.append(feature)
    return np.array(features) if features else None

# Save extracted features to files
def save_features(features, output_dir, file_prefix='feature_'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, feature in enumerate(features):
        file_path = os.path.join(output_dir, f"{file_prefix}{i}.npy")
        np.save(file_path, feature)
        print(f"Saved feature to {file_path}")

# Main execution
audio_file_paths = ['applause.wav']
audio_features = prepare_audio_dataset(audio_file_paths)

if audio_features is not None and len(audio_features) > 0:
    input_dim = audio_features.shape[1]
    audio_features = audio_features.reshape(len(audio_features), -1)
    
    # Set model hyperparameters
    num_heads = 4
    ff_dim = 128
    num_blocks = 2
    reservoir_dim = 100
    max_reservoir_dim = 1000
    spectral_radius = 1.5
    leak_rate = 0.3

    # Create and compile the model
    model = create_transformer_spiking_model(input_dim, num_heads, ff_dim, num_blocks, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim)
    model.compile(optimizer='adam', loss='mse')

    # Function to extract features from a single audio file
    def extract_features(audio_file_path):
        audio_feature = preprocess_audio(audio_file_path)
        if audio_feature is not None:
            audio_feature = audio_feature.reshape(1, -1, input_dim)
            features = model.predict(audio_feature)
            return features
        return None

    # Extract features from a new audio file
    new_audio_file_path = 'applause.wav'
    extracted_features = extract_features(new_audio_file_path)
    if extracted_features is not None:
        print("Extracted Features:", extracted_features)
        save_features(extracted_features, 'Features')
    else:
        print("Failed to extract features from the audio file.")
else:
    print("No valid audio features extracted. Please check your audio files.")

# pip install --upgrade librosa soundfile audioread
# Transformer-based SLNN Encoder Model (TSLNN-E)
# python tslnn_encoder_wav_to_emb.py
"""
Attempting to load audio file: applause.wav
Loaded with librosa: Sample rate 48000, shape (1199232,)
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 761ms/step
Extracted Features: [[0.10294048 0.         0.08104667 0.0715439  0.03073235 0.00127863
  0.         0.         0.02023316 0.09165004 0.04390178 0.04869283
  0.16776186 0.         0.06369713 0.         0.         0.04182693
  0.         0.10296468 0.         0.0415316  0.         0.
  0.08364696 0.         0.02225025 0.05985401 0.         0.00986295
  0.08072463 0.         0.         0.04832723 0.05203104 0.04795967
  0.         0.         0.         0.         0.04754353 0.0487162
  0.00336315 0.05536728 0.0512665  0.         0.04158994 0.07643519
  0.         0.         0.03484162 0.         0.02666362 0.03639041
  0.039964   0.07919361 0.         0.         0.05009118 0.
  0.         0.         0.05819657 0.08545422]]
"""

