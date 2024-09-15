import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from smtstecgser_model import create_smtstecgser_model

class SelfModelingCallback(Callback):
    def __init__(self, reservoir_layer, performance_metric='classification_output_accuracy', target_metric=0.95, 
                 growth_phase_length=10, pruning_phase_length=5):
        super().__init__()
        self.reservoir_layer = reservoir_layer
        self.performance_metric = performance_metric
        self.target_metric = target_metric
        self.growth_phase_length = growth_phase_length
        self.pruning_phase_length = pruning_phase_length
        self.current_phase = 'growth'
        self.phase_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.performance_metric, 0)
        self.phase_counter += 1

        if self.current_phase == 'growth' and self.phase_counter >= self.growth_phase_length:
            self.current_phase = 'pruning'
            self.phase_counter = 0
        elif self.current_phase == 'pruning' and self.phase_counter >= self.pruning_phase_length:
            self.current_phase = 'growth'
            self.phase_counter = 0

        if current_metric >= self.target_metric:
            if self.current_phase == 'growth':
                self.reservoir_layer._expand_reservoir()
            elif self.current_phase == 'pruning':
                self.reservoir_layer._prune_reservoir()

def preprocess_data(x):
    return x.astype('float32') / 255.0

def load_and_prepare_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    x_train = preprocess_data(x_train).reshape(-1, 28, 28, 1)
    x_val = preprocess_data(x_val).reshape(-1, 28, 28, 1)
    x_test = preprocess_data(x_test).reshape(-1, 28, 28, 1)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def train_model(model, x_train, y_train, x_val, y_val, reservoir_layer, epochs=10, batch_size=64):
    early_stopping = EarlyStopping(monitor='val_classification_output_accuracy', patience=5, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_classification_output_accuracy', factor=0.1, patience=3, mode='max')
    self_modeling_callback = SelfModelingCallback(
        reservoir_layer=reservoir_layer,
        performance_metric='val_classification_output_accuracy',
        target_metric=0.90
    )

    # Create dummy targets for self_modeling_output
    dummy_targets = np.zeros((x_train.shape[0], np.prod(x_train.shape[1:])))
    dummy_val_targets = np.zeros((x_val.shape[0], np.prod(x_val.shape[1:])))

    history = model.fit(
        x_train, 
        {'classification_output': y_train, 'self_modeling_output': dummy_targets},
        validation_data=(x_val, {'classification_output': y_val, 'self_modeling_output': dummy_val_targets}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, self_modeling_callback]
    )

    return history

def evaluate_model(model, x_test, y_test):
    dummy_test_targets = np.zeros((x_test.shape[0], np.prod(x_test.shape[1:])))
    test_loss, test_acc = model.evaluate(x_test, {'classification_output': y_test, 'self_modeling_output': dummy_test_targets}, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

def save_model(model):
    # Save the model
    model.save('Trained Models/smtstecgser_mnist.keras')

def plot_training_history(history):
    plt.plot(history.history['classification_output_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_classification_output_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def main():
    input_shape = (28, 28, 1)
    initial_reservoir_size = 512 
    max_reservoir_dim = 4096
    spectral_radius = 0.5
    leak_rate = 0.1
    spike_threshold = 0.5
    output_dim = 10

    model, reservoir_layer = create_smtstecgser_model(
        input_shape=input_shape,
        initial_reservoir_size=initial_reservoir_size,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim,
        output_dim=output_dim
    )

    model.compile(
        optimizer='adam',
        loss={
            'classification_output': 'categorical_crossentropy',
            'self_modeling_output': 'mse'
        },
        loss_weights={
            'classification_output': 1.0,
            'self_modeling_output': 0.1
        },
        metrics={
            'classification_output': 'accuracy'
        }
    )

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_prepare_data()

    history = train_model(model, x_train, y_train, x_val, y_val, reservoir_layer)
    evaluate_model(model, x_test, y_test)
    save_model(model)
    plot_training_history(history)

if __name__ == '__main__':
    main()



# Self-Modeling Transformer-based Spatio-Temporal Efficient Convolutional Gated Spiking Elastic Reservoir (SMTSTECGSER)
# python smtstecgser_train.py
# Test Accuracy: 0.9920