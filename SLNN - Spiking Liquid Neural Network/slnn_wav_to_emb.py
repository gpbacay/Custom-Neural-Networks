import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import librosa
import soundfile as sf
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Custom Spiking LNN Layer with Spiking Dynamics
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
        prev_state = states[0][:, :self.reservoir_weights.shape[0]]
        input_part = tf.matmul(inputs, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights, transpose_b=True)
        state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_part + reservoir_part)

        # Spiking dynamics: Apply a threshold to produce discrete spikes
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        # Ensure the state size matches the max reservoir size
        active_size = tf.shape(state)[-1]
        padded_state = tf.concat([state, tf.zeros([tf.shape(state)[0], self.max_reservoir_dim - active_size])], axis=1)
        return padded_state, [padded_state]

def initialize_spiking_lnn_reservoir(input_dim, reservoir_dim, spectral_radius, max_reservoir_dim):
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim) * 0.1
    return reservoir_weights, input_weights

def create_spiking_nlnn_model(input_dim, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim):
    inputs = Input(shape=(input_dim,))

    # Initialize Spiking LNN weights
    reservoir_weights, input_weights = initialize_spiking_lnn_reservoir(input_dim, reservoir_dim, spectral_radius, max_reservoir_dim)

    # Spiking LNN Layer
    lnn_layer = tf.keras.layers.RNN(
        SpikingLNNStep(reservoir_weights, input_weights, leak_rate, max_reservoir_dim),
        return_sequences=True
    )

    def apply_spiking_lnn(x):
        lnn_output = lnn_layer(tf.expand_dims(x, axis=1))
        return Flatten()(lnn_output)

    def get_output_shape(input_shape):
        # Adjust this size based on the actual dimensions of lnn_layer's output
        return (input_shape[0], 1000)  # Example fixed shape

    lnn_output = Lambda(apply_spiking_lnn, output_shape=get_output_shape)(inputs)

    # Feature Extraction Block
    x = Dense(128, activation='relu')(lnn_output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Output layer is removed for feature extraction
    model = tf.keras.Model(inputs, x)
    return model

def load_audio(file_path):
    """Load audio file using librosa or soundfile."""
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

def preprocess_audio(file_path):
    """Convert audio to Mel-spectrogram."""
    y, sr = load_audio(file_path)
    if y is None or sr is None:
        print(f"Failed to load audio file: {file_path}")
        return None

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram

def prepare_audio_dataset(file_paths):
    """Prepare dataset of audio features."""
    features = []
    for file_path in file_paths:
        feature = preprocess_audio(file_path)
        if feature is not None:
            features.append(feature)
    return np.array(features) if features else None

def save_features(features, output_dir, file_prefix='feature_'):
    """Save extracted features to specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, feature in enumerate(features):
        file_path = os.path.join(output_dir, f"{file_prefix}{i}.npy")
        np.save(file_path, feature)
        print(f"Saved feature to {file_path}")

# Example usage
audio_file_paths = ['applause.wav']  # Replace with your audio file paths
audio_features = prepare_audio_dataset(audio_file_paths)

if audio_features is not None and len(audio_features) > 0:
    input_dim = np.prod(audio_features.shape[1:])
    audio_features = audio_features.reshape(len(audio_features), -1)

    reservoir_dim = 100
    max_reservoir_dim = 1000
    spectral_radius = 1.5
    leak_rate = 0.3

    model = create_spiking_nlnn_model(input_dim, reservoir_dim, spectral_radius, leak_rate, max_reservoir_dim)
    model.compile(optimizer='adam', loss='mse')

    def extract_features(audio_file_path):
        audio_feature = preprocess_audio(audio_file_path)
        if audio_feature is not None:
            audio_feature = audio_feature.reshape(1, -1)  # Reshape for model input
            features = model.predict(audio_feature)
            return features
        return None

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
# python slnn_wav_to_emb.py
"""
Attempting to load audio file: applause.wav
Loaded with librosa: Sample rate 48000, shape (1199232,)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 358ms/step
Extracted Features: [[0.09611022 0.         0.         0.065415   0.19769707 0.
  0.0179268  0.         0.         0.04982481 0.         0.
  0.00574221 0.16034123 0.14872773 0.10052275 0.0742148  0.08185145
  0.1772695  0.         0.01892112 0.076842   0.         0.
  0.         0.         0.         0.03240908 0.         0.
  0.         0.15572977 0.         0.01907175 0.         0.04176007
  0.10696978 0.         0.04551827 0.         0.         0.
  0.         0.         0.         0.         0.04372714 0.18078138
  0.         0.22524853 0.11329865 0.04273732 0.14895827 0.10443781
  0.06155573 0.16128659 0.08623922 0.11235403 0.12218432 0.
  0.         0.16141224 0.01608228 0.        ]]
"""