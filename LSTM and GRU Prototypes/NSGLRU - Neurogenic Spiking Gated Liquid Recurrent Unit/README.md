# **Neurogenic Spiking Gated Liquid Recurrent Unit (NSGLRU) for MNIST Classification**

This project implements a novel neural network architecture called the Neurogenic Spiking Gated Liquid Recurrent Unit (NSGLRU) for classifying the MNIST dataset. The NSGLRU combines concepts from reservoir computing, gated recurrent units, and spiking neural networks to create a powerful and flexible model for sequence processing and classification tasks.

## Features

- Custom TensorFlow layer implementing the NSGLRU cell
- Reservoir initialization with controlled spectral radius
- Gated dynamics for improved information flow
- Spiking activation for sparse, event-driven computation
- Application to MNIST digit classification

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- scikit-learn

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/nsglru-mnist.git
   cd nsglru-mnist
   ```

2. Install the required packages:
   ```
   pip install tensorflow numpy scikit-learn
   ```

## Usage

Run the main script to train and evaluate the NSGLRU model on the MNIST dataset:

```
python nsglru_mnist.py
```

## Model Architecture

The NSGLRU model consists of the following components:

1. Input layer
2. NSGLRU layer (custom RNN cell)
3. Flatten layer
4. Two dense layers with ReLU activation and dropout
5. Output layer with softmax activation

The NSGLRU cell implements the following key features:
- Reservoir computing with fixed random weights
- Gated update mechanism (input, forget, and output gates)
- Spiking activation with adjustable threshold
- Leaky integration of state updates

## Hyperparameters

The main hyperparameters of the model include:

- `reservoir_dim`: Dimension of the reservoir (default: 500)
- `max_reservoir_dim`: Maximum dimension of the reservoir (default: 1000)
- `spectral_radius`: Spectral radius for reservoir scaling (default: 1.5)
- `leak_rate`: Leak rate for state update (default: 0.3)
- `spike_threshold`: Threshold for spike generation (default: 0.5)

## Results

The model achieves a test accuracy of approximately 96.75% on the MNIST dataset, although the exact performance may vary depending on the specific reservoir dimensions and random initialization.

## Customization

You can modify the hyperparameters in the `main()` function to experiment with different model configurations. Additionally, you can adapt the `create_NSGLRU_model()` function to change the architecture of the downstream layers after the NSGLRU cell.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation is inspired by concepts from reservoir computing, liquid state machines, and gated recurrent neural networks. The NSGLRU cell combines these ideas to create a novel architecture for sequence processing and classification tasks.