# **Neurogenic Spiking Gated Liquid Recurrent Unit (NSGLRU) for MNIST Classification**

This project introduces the Neurogenic Spiking Gated Liquid Recurrent Unit (NSGLRU), a novel neural network architecture designed for classifying the MNIST dataset. The NSGLRU integrates ideas from reservoir computing, gated recurrent units, and spiking neural networks, resulting in a powerful and versatile model for sequence processing and classification tasks.

## Features

- Custom TensorFlow layer implementing the NSGLRU cell.
- Reservoir initialization with a controlled spectral radius.
- Gated dynamics for enhanced information flow.
- Spiking activation for sparse, event-driven computation.
- Application to the MNIST digit classification task.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- scikit-learn

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/nsglru-mnist.git
   cd nsglru-mnist
   ```

2. Install the required packages:

   ```bash
   pip install tensorflow numpy scikit-learn
   ```

## Usage

Run the main script to train and evaluate the NSGLRU model on the MNIST dataset:

```bash
python nsglru_mnist.py
```

## Model Architecture

The NSGLRU model comprises the following components:

1. Input Layer: Accepts the MNIST image data.
2. Convolutional Layers: Three Conv2D layers with increasing filter sizes for feature extraction.
3. NSGLRU Layer: A custom RNN cell incorporating reservoir dynamics, gating mechanisms, and spiking activations.
4. Flatten Layer: Converts the output of the NSGLRU layer into a 1D vector.
5. Dense Layers: Two fully connected layers with ReLU activation and Dropout for regularization.
6. Output Layer: A softmax layer for multi-class classification.

### NSGLRU Cell

The custom NSGLRU cell integrates the following features:

- Reservoir Computing: Utilizes fixed, random reservoir weights scaled by a spectral radius.
- Gated Update Mechanism: Employs input, forget, and output gates to control state updates.
- Spiking Activation: Implements a threshold mechanism to generate discrete spikes.
- Leaky Integration: Smooths state updates with a leak rate parameter.

## Hyperparameters

Key hyperparameters of the model include:

- `reservoir_dim`: Dimension of the reservoir (default: 500).
- `max_reservoir_dim`: Maximum dimension of the reservoir (default: 1000).
- `spectral_radius`: Spectral radius for reservoir scaling (default: 1.5).
- `leak_rate`: Leak rate for state update (default: 0.3).
- `spike_threshold`: Threshold for spike generation (default: 0.5).

## Results

The NSGLRU model achieves a test accuracy of approximately 99.12% on the MNIST dataset. Note that the exact performance may vary depending on the reservoir's dimensions and the initialization of random weights.

## Customization

You can modify the hyperparameters in the `main()` function to experiment with different model configurations. Additionally, the `create_NSGLRU_model()` function can be adjusted to change the architecture of the downstream layers after the NSGLRU cell.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation draws inspiration from concepts in reservoir computing, liquid state machines, and gated recurrent neural networks. The NSGLRU cell uniquely combines these ideas, creating a novel architecture for sequence processing and classification tasks.