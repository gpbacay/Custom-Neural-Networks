import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from csmselnn_model_v2 import create_csmselnn_model

class SynaptogenesisCallback(Callback):
    def __init__(self, synaptogenesis_layer, performance_metric='val_classification_output_accuracy', target_metric=0.95,
                 add_synapses_threshold=0.01, prune_synapses_threshold=0.1, growth_phase_length=10, pruning_phase_length=5):
        super().__init__()
        self.synaptogenesis_layer = synaptogenesis_layer
        self.performance_metric = performance_metric
        self.target_metric = target_metric
        self.initial_add_synapses_threshold = add_synapses_threshold
        self.initial_prune_synapses_threshold = prune_synapses_threshold
        self.growth_phase_length = growth_phase_length
        self.pruning_phase_length = pruning_phase_length
        self.current_phase = 'growth'
        self.phase_counter = 0
        self.performance_history = []

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.performance_metric, 0)
        
        self.performance_history.append(current_metric)
        
        # Adjust thresholds based on current performance
        self.add_synapses_threshold = self.initial_add_synapses_threshold * (1 - current_metric)
        self.prune_synapses_threshold = self.initial_prune_synapses_threshold * current_metric

        self.phase_counter += 1
        if self.current_phase == 'growth' and self.phase_counter >= self.growth_phase_length:
            self.current_phase = 'pruning'
            self.phase_counter = 0
        elif self.current_phase == 'pruning' and self.phase_counter >= self.pruning_phase_length:
            self.current_phase = 'growth'
            self.phase_counter = 0

        if len(self.performance_history) > 5:
            improvement_rate = (current_metric - self.performance_history[-5]) / 5
            
            if improvement_rate > 0.01:  # Fast improvement
                self.synaptogenesis_layer.add_synapses()  # Add more synapses
            elif improvement_rate < 0.001:  # Slow improvement
                self.synaptogenesis_layer.prune_synapses()  # More aggressive pruning

        if current_metric >= self.target_metric:
            print(f" - Performance metric {self.performance_metric} reached target {self.target_metric}. Current phase: {self.current_phase}")
            if self.current_phase == 'growth':
                if current_metric < self.add_synapses_threshold:
                    self.synaptogenesis_layer.add_synapses()
            elif self.current_phase == 'pruning':
                if current_metric > self.prune_synapses_threshold:
                    self.synaptogenesis_layer.prune_synapses()

def preprocess_data(x):
    return x.astype(np.float32) / 255.0

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    x_train = preprocess_data(x_train).reshape(-1, 28, 28, 1)
    x_val = preprocess_data(x_val).reshape(-1, 28, 28, 1)
    x_test = preprocess_data(x_test).reshape(-1, 28, 28, 1)
    
    input_shape = (28, 28, 1)
    initial_reservoir_size = 512
    max_reservoir_dim = 4096
    spectral_radius = 0.5
    leak_rate = 0.1
    spike_threshold = 0.5
    output_dim = 10
    epochs = 10
    batch_size = 64
    add_synapses_threshold = 0.1
    prune_synapses_threshold = 0.1
    
    model, synaptogenesis_layer = create_csmselnn_model(
        input_shape=input_shape,
        initial_reservoir_size=initial_reservoir_size,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim,
        output_dim=output_dim
    )
    
    model.compile(optimizer='adam',
                  loss={'classification_output': 'sparse_categorical_crossentropy', 'self_modeling_output': 'mean_squared_error'},
                  metrics={'classification_output': 'accuracy'})
    
    early_stopping = EarlyStopping(monitor='val_classification_output_accuracy', patience=10, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_classification_output_accuracy', factor=0.1, patience=5, mode='max')
    synaptogenesis_callback = SynaptogenesisCallback(
        synaptogenesis_layer=synaptogenesis_layer,
        performance_metric='val_classification_output_accuracy',
        target_metric=0.90,
        add_synapses_threshold=add_synapses_threshold,
        prune_synapses_threshold=prune_synapses_threshold
    )
    
    history = model.fit(
        x_train, 
        {'classification_output': y_train, 'self_modeling_output': x_train.reshape(x_train.shape[0], -1)},
        validation_data=(x_val, {'classification_output': y_val, 'self_modeling_output': x_val.reshape(x_val.shape[0], -1)}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, synaptogenesis_callback]
    )
    
    test_loss, test_acc = model.evaluate(x_test, {'classification_output': y_test, 'self_modeling_output': x_test.reshape(x_test.shape[0], -1)}, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")
    
    plt.plot(history.history['classification_output_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_classification_output_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    model.save('Trained Models/csmselnn_mnist_v2.keras')

if __name__ == "__main__":
    main()



# Convolutional Self-Modeling Spiking Elastic Liquid Neural Network (CSMSELNN) version 2
# python csmselnn_train_v2.py
# Test Accuracy: 0.9919

"""

"""