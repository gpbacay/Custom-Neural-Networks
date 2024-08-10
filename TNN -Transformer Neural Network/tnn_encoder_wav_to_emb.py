import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LayerNormalization, MultiHeadAttention, Dropout, Flatten, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
import librosa
import soundfile as sf
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def create_transformer_encoder_model(input_dim, num_heads, ff_dim, num_blocks):
    inputs = Input(shape=(None, input_dim))
    
    x = inputs
    for _ in range(num_blocks):
        # Multi-Head Self-Attention
        x = MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)(x, x)
        x = Dropout(0.1)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feed Forward Network
        x_ff = Dense(ff_dim, activation='relu')(x)
        x_ff = Dense(input_dim)(x_ff)
        x = x + x_ff
        x = Dropout(0.1)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
    
    # Use GlobalAveragePooling1D to handle variable sequence length
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

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
    input_dim = audio_features.shape[1]
    audio_features = audio_features.reshape(len(audio_features), -1)
    
    num_heads = 4
    ff_dim = 128
    num_blocks = 2

    model = create_transformer_encoder_model(input_dim, num_heads, ff_dim, num_blocks)
    model.compile(optimizer='adam', loss='mse')

    def extract_features(audio_file_path):
        audio_feature = preprocess_audio(audio_file_path)
        if audio_feature is not None:
            audio_feature = audio_feature.reshape(1, -1, input_dim)  # Reshape for model input
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
# Transformer-based SLNN Encoder Model (TEM)
# python tnn_encoder_wav_to_emb.py
