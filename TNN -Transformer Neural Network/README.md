# Transformer-Spiking LNN Audio Feature Extractor

## Overview

This project implements a hybrid Transformer-Spiking Liquid Neural Network (LNN) model for extracting features from audio files. The system combines the power of Transformer architectures with the biological inspiration of spiking neural networks to create a unique and powerful audio processing tool.

## Features

- Loads and preprocesses audio files using librosa and soundfile
- Implements a custom Spiking LNN layer in TensorFlow
- Combines Transformer architecture with Spiking LNN for feature extraction
- Extracts mel spectrogram features from audio
- Saves extracted features to files

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- librosa
- soundfile

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/gpbacay/Custom-Neural-Networks.git
   cd TNN -Transformer Neural Network/tslnn_encoder_wav_to_emb.py
   ```

2. Install the required packages:
   ```
   pip install tensorflow numpy librosa soundfile
   ```

## Usage

1. Place your audio files (e.g., WAV format) in the project directory.

2. Modify the `audio_file_paths` list in the main execution section of the script to include your audio file names.

3. Run the script:
   ```
   python tslnn_encoder_wav_to_emb.py
   ```

4. The script will process the audio files, extract features, and save them in the `Features` directory.

## Model Architecture

The model consists of two main components:

1. **Transformer Blocks**: Multiple layers of multi-head attention and feed-forward networks for processing sequential data.

2. **Spiking LNN**: A custom layer implementing a spiking liquid neural network, adding biological-inspired dynamics to the model.

The extracted features are then processed through dense layers for final feature representation.

## Customization

You can customize the model by adjusting the following parameters in the main execution section:

- `num_heads`: Number of attention heads in the Transformer blocks
- `ff_dim`: Dimension of the feed-forward network in Transformer blocks
- `num_blocks`: Number of Transformer blocks
- `reservoir_dim`: Dimension of the Spiking LNN reservoir
- `max_reservoir_dim`: Maximum dimension of the reservoir (for padding)
- `spectral_radius`: Spectral radius for initializing reservoir weights
- `leak_rate`: Leak rate for the Spiking LNN dynamics

## Output

The script outputs:
- Console logs detailing the audio loading and processing steps
- Extracted features printed to the console
- Extracted features saved as NumPy files in the `Features` directory

## Limitations and Future Work

- The current implementation is designed for offline processing. Real-time processing could be a future enhancement.
- The model architecture and hyperparameters may need tuning for specific audio tasks or datasets.
- Integration with downstream tasks (e.g., classification, clustering) could be added to demonstrate the utility of the extracted features.

## Contributing

Contributions to improve the model or extend its functionality are welcome. Please submit pull requests or open issues to discuss proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.